import json
import logging
import uuid
from datetime import datetime
from io import BytesIO
from itertools import groupby
from typing import Optional, List
from urllib import request
from urllib.parse import urlparse

import h5py
import numpy as np
import requests
from geojson import Feature, MultiPoint

from cyborg.app.settings import Settings
from cyborg.consts.her2 import Her2Consts
from cyborg.consts.pdl1 import Pdl1Consts
from cyborg.infra.fs import fs
from cyborg.infra.oss import oss
from cyborg.libs.heimdall.dispatch import open_slide
from cyborg.modules.partner.roche.domain.consts import HEAT_COLORLUTS, ROCHE_TIME_FORMAT, HER2_ALGORITHM_DISPLAY_ID, \
    PDL1_ALGORITHM_DISPLAY_ID
from cyborg.modules.partner.roche.domain.entities import RocheAITaskEntity
from cyborg.modules.partner.roche.domain.repositories import RocheRepository
from cyborg.modules.partner.roche.domain.value_objects import RocheAITaskStatus, RocheALGResult, \
    RocheDiplomat, RocheWsiInput, RocheIndexItem, RocheMarkerPreset, RocheMarkerGroup, \
    RocheMarkerShape, RochePanel, RocheHeatMap, RocheColor
from cyborg.modules.partner.roche.utils.color import hex_to_rgba
from cyborg.modules.partner.roche.utils.image import convert_image_to_png_data
from cyborg.modules.partner.roche.utils.paramid import gen_dimensions, gen_opacities
from cyborg.seedwork.domain.value_objects import AIType
from cyborg.utils.encoding import CyborgJsonEncoder
from cyborg.utils.id_worker import IdWorker

logger = logging.getLogger(__name__)


LABEL_TO_EN = {
    '0': 'Faint incomplete membranous staining tumor cells',  # 微弱的不完整膜阳性肿瘤细胞
    '1': 'Weak-moderate complete membranous staining tumor cells',  # 弱中等的完整膜阳性肿瘤细胞
    '2': 'Negative tumor cells',  # 阴性肿瘤细胞
    '3': 'Strong intensity complete membranous staining tumor cells',  # 强度的完整膜阳性肿瘤细胞
    '4': 'Moderate intensity incomplete membranous staining tumor cells',  # 中强度的不完整膜阳性肿瘤细胞
    '5': 'Tissue cells'  # 组织细胞
}


class RocheDomainService(object):

    def __init__(self, repository: RocheRepository):
        super(RocheDomainService, self).__init__()
        self.repository = repository

    def create_ai_task(
            self,
            algorithm_id: str,
            slide_url: str,
            input_info: dict
    ) -> Optional[RocheAITaskEntity]:

        algorithm = self.repository.get_algorithm(algorithm_id)
        if not algorithm:
            return None

        task = RocheAITaskEntity(raw_data={
            'analysis_id': str(uuid.uuid4()),
            'algorithm_id': algorithm_id,
            'ai_type': AIType.get_by_value(algorithm.algorithm_name),
            'slide_url': slide_url,
            'input_info': input_info,
            'status': RocheAITaskStatus.accepted,
        })

        if self.repository.save_ai_task(task):
            return task

        return None

    def update_ai_task(
            self, task: RocheAITaskEntity, status: Optional[RocheAITaskStatus] = None, result_id: Optional[str] = None
    ) -> bool:

        if task.status == RocheAITaskStatus.closed:
            return True

        if status is not None:
            task.update_data(status=status)
            if status == RocheAITaskStatus.in_progress:
                task.update_data(started_at=datetime.now())
                task.setup_expired_time()
                # cache.set(self.RANK0_TASK_ID_CACHE_KEY, task.id)

        if result_id is not None:
            task.update_data(result_id=result_id)

        if self.repository.save_ai_task(task):
            return self.callback_analysis_status(task)
        else:
            return False

    def download_slide(self, task: RocheAITaskEntity) -> str:
        if 'amazonaws' not in task.slide_url and 'dipath.cn' not in task.slide_url and Settings.ROCHE_IMAGE_SERVER:
            parsed_url = urlparse(task.slide_url)
            slide_url = task.slide_url.replace(
                f'{parsed_url.scheme}://{parsed_url.netloc}', Settings.ROCHE_IMAGE_SERVER)
        else:
            slide_url = task.slide_url

        logger.info(slide_url)
        opener = request.build_opener()
        opener.addheaders = [(
            'User-agent',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'
        )]
        request.install_opener(opener)
        request.urlretrieve(slide_url, task.slide_path)
        return task.slide_path

    def start_analysis(self, task: RocheAITaskEntity) -> RocheALGResult:
        if not fs.path_exists(task.slide_path):
            downloaded = self.download_slide(task)
            if not downloaded:
                return RocheALGResult(err_msg='切片url无法下载')

        input_info = task.input_info
        slide_width = input_info.get('slide_width')
        slide_height = input_info.get('slide_height')

        dimensions = gen_dimensions(slide_width, slide_height)
        wsi_input = RocheWsiInput(
            image_location='',
            image_id='0',
            sha256='',
            md5=input_info.get('md5', ''),
            scanner_name='',
            scanner_unit_number='',
            microns_per_pixel_x=input_info.get('microns_per_pixel_x', 0.242042),
            microns_per_pixel_y=input_info.get('microns_per_pixel_y', 0.242042),
            slide_magnification=40,
            slide_width=slide_width,
            slide_height=slide_height,
            slide_depth=1,
            number_levels=len(dimensions),
            dimensions=dimensions
        )
        # mpp = slide.mpp or 0.242042

        coordinates = task.coordinates
        if coordinates:
            x_coords, y_coords = zip(coordinates)
        else:
            x_coords, y_coords = [], []

        regions = input_info.get('regions', [])
        rois = [roi for region in regions for roi in region.get('artifacts', [])]
        roi_id = IdWorker.new_mark_id_worker().get_new_id()

        if regions:
            roi_list = [{
                'id': roi_id,
                'x': [coord['x'] for coord in roi.get('coordinates', [])],
                'y': [coord['y'] for coord in roi.get('coordinates', [])]
            } for roi in rois]
        else:
            roi_list = [{'id': roi_id, 'x': x_coords, 'y': y_coords}]

        if task.ai_type == AIType.her2:
            return self.run_her2(task=task, roi_list=roi_list, wsi_input=wsi_input)
        elif task.ai_type == AIType.pdl1:
            return self.run_pdl1(task=task, roi_list=roi_list, wsi_input=wsi_input)
        else:
            logger.error(f'{task.ai_type} does not support')
            return RocheALGResult(err_msg=f'{task.ai_type} does not support')

    def run_her2(self, task: RocheAITaskEntity, roi_list: List[dict], wsi_input: RocheWsiInput):
        mock_result_file_key = oss.path_join('test', 'her2_mock_result.json')
        try:
            mock_result = oss.get_object(mock_result_file_key)
            mock_result = tuple(json.loads(mock_result))
        except Exception:
            mock_result = None

        from cyborg.modules.ai.libs.algorithms.Her2New_.detect_all import run_her2_alg, roi_filter
        if mock_result and False:
            center_coords_np_with_id, cls_labels_np_with_id, summary_dict, lvl, flg = mock_result
            # roi_id = list(center_coords_np_with_id.keys())[0]
        else:
            center_coords_np_with_id, cls_labels_np_with_id, summary_dict, lvl, flg = run_her2_alg(
                slide_path=task.slide_path, roi_list=roi_list)

            result_data = json.dumps([center_coords_np_with_id, cls_labels_np_with_id, summary_dict, lvl, flg])
            buffer = BytesIO()
            buffer.write(result_data.encode('utf-8'))
            buffer.seek(0)
            oss.put_object_from_io(buffer, mock_result_file_key)

        ai_result = Her2Consts.rois_summary_dict.copy()
        label_to_roi_name = Her2Consts.cell_label_dict

        cells_in_tile = {}
        cell_index_items = []
        slide_width = wsi_input.slide_width
        slide_height = wsi_input.slide_height
        for roi in roi_list:
            center_coords, cls_labels = roi_filter(
                center_coords_np_with_id[roi['id']],
                cls_labels_np_with_id[roi['id']],
                roi['x'],
                roi['y']
            )

            for idx, coord in enumerate(center_coords):
                label = str(cls_labels[idx])
                roi_name = label_to_roi_name[label]
                ai_result[roi_name] += 1

                x = float(coord[0]) if slide_width > float(coord[0]) else float(slide_width - 1)
                y = float(coord[1]) if slide_height > float(coord[1]) else float(slide_height - 1)

                tile_x, tile_y = RocheIndexItem.locate(int(x), int(y))
                cells = cells_in_tile.setdefault((tile_x, tile_y), [])
                cells.append([label, [x, y]])

            for (tile_x, tile_y), cells in cells_in_tile.items():
                index_item = RocheIndexItem.new_item(tile_x, tile_y)
                for label, cell_group in groupby(cells, lambda c: c[0]):
                    feature = Feature(properties={'label': int(label)}, geometry=MultiPoint(coordinates=[]))
                    hot_value = round(Her2Consts.sorted_labels.index(label) * 255 / len(Her2Consts.sorted_labels))
                    for cell in cell_group:
                        feature['geometry']['coordinates'].append([int(val) for val in cell[1]])
                        index_item.heatmap_points.append([int(val) for val in cell[1]] + [hot_value])
                    index_item.geo_json['features'].append(feature)
                cell_index_items.append(index_item)

        ai_result.update({
            'her2_level': Her2Consts.level[int(lvl)]
        })

        marker_groups = []
        marker_shapes = {}
        for label in Her2Consts.sorted_labels:
            roi_name = Her2Consts.cell_label_dict[label]
            marker_type = label
            marker_group = RocheMarkerGroup(
                label=int(label),
                name=LABEL_TO_EN.get(label, roi_name),
                description=LABEL_TO_EN.get(label, roi_name),
                textgui='red',
                locked=False,
                visible=True,
                type=marker_type
            )
            marker_groups.append(marker_group)
            color = Her2Consts.type_color_dict[Her2Consts.label_to_diagnosis_type[int(label)]]
            color = hex_to_rgba(color)
            marker_shape = RocheMarkerShape(
                level=0,
                outline_width=0,
                size=7,
                type='circle',
                color=color,
                label_color=color,
                outline_color='rgba(0,0,0,0)'
            )
            marker_shapes[marker_type] = marker_shape

        wsi_marker_preset = RocheMarkerPreset(
            textgui='Nuclei',
            active=True,
            data=marker_groups
        )

        markers_panel = RochePanel(
            name='Markers',
            description='Markers'
        )

        heatmap_panel = RochePanel(
            name='Heatmaps',
            description='Heatmaps'
        )

        heatmap = RocheHeatMap(
            gui=HER2_ALGORITHM_DISPLAY_ID,
            labels=[{'0': 'None'}, {'255': 'High Density'}],
            level_opacity=gen_opacities(len(wsi_input.dimensions))
        )

        slide = open_slide(task.slide_path)
        thumbnail = slide.get_thumbnail(size=max(wsi_input.dimensions[-1]))

        return RocheALGResult(
            ai_results=ai_result,
            wsi_input=wsi_input,
            cells_index_items=cell_index_items,
            marker_presets=[wsi_marker_preset],
            marker_shapes=marker_shapes,
            heatmaps=[heatmap],
            panels=[markers_panel, heatmap_panel],
            thumbnail=convert_image_to_png_data(thumbnail)
        )

    def run_pdl1(self, task: RocheAITaskEntity, roi_list: List[dict], wsi_input: RocheWsiInput):
        from cyborg.modules.ai.utils.pdl1 import compute_pdl1_s

        all_center_coords_list = []
        all_cell_types_list = []
        all_ori_cell_types_list = []
        all_probs_list = []
        slide = open_slide(task.slide_path)

        total_pos_tumor_num, total_neg_tumor_num, total_pos_norm_num, total_neg_norm_num = 0, 0, 0, 0

        fitting_model = None
        smooth = None

        cell_index_items = []
        slide_width = wsi_input.slide_width
        slide_height = wsi_input.slide_height
        cells_in_tile = {}
        ai_result = {}

        for idx, roi in enumerate(roi_list):
            _, x_coords, y_coords = roi['id'], roi['x'], roi['y']

            count_summary_dict, center_coords, ori_labels, cls_labels, probs, annot_cls_labels = compute_pdl1_s(
                slide_path=task.slide_path,
                x_coords=x_coords,
                y_coords=y_coords, fitting_model=fitting_model, smooth=smooth)

            total_pos_tumor_num += count_summary_dict['pos_tumor']
            total_neg_tumor_num += count_summary_dict['neg_tumor']
            total_pos_norm_num += count_summary_dict['pos_norm']
            total_neg_norm_num += count_summary_dict['neg_norm']

            all_center_coords_list += center_coords
            all_cell_types_list += cls_labels
            all_ori_cell_types_list += ori_labels
            all_probs_list += probs

            for center_coords, cell_type, annot_type in zip(center_coords, cls_labels, annot_cls_labels):
                label = cls_labels[idx]

                x = float(center_coords[0]) if slide_width > float(center_coords[0]) else float(slide_width - 1)
                y = float(center_coords[1]) if slide_height > float(center_coords[1]) else float(slide_height - 1)

                tile_x, tile_y = RocheIndexItem.locate(int(x), int(y))
                cells = cells_in_tile.setdefault((tile_x, tile_y), [])
                cells.append([label, [x, y]])

            for (tile_x, tile_y), cells in cells_in_tile.items():
                index_item = RocheIndexItem.new_item(tile_x, tile_y)
                for label, cell_group in groupby(cells, lambda c: c[0]):
                    feature = Feature(properties={'label': label}, geometry=MultiPoint(coordinates=[]))
                    hot_value = {0: 0, 2: 0, 1: 128, 3: 255}.get(label)
                    for cell in cell_group:
                        feature['geometry']['coordinates'].append([int(val) for val in cell[1]])
                        index_item.heatmap_points.append([int(val) for val in cell[1]] + [hot_value])
                    index_item.geo_json['features'].append(feature)
                cell_index_items.append(index_item)

        ai_result['total_pos_tumor_num'] = total_pos_tumor_num
        ai_result['total_neg_tumor_num'] = total_neg_tumor_num
        ai_result['total_pos_norm_num'] = total_pos_norm_num
        ai_result['total_neg_norm_num'] = total_neg_norm_num
        ai_result['tps'] = str(
            round(100 * total_pos_tumor_num / (total_neg_tumor_num + total_pos_tumor_num + 1e-10), 2)) + '%'

        marker_groups = []
        marker_shapes = {}
        for label in Pdl1Consts.sorted_labels:
            marker_group_name = Pdl1Consts.label_to_en[label]
            marker_type = label
            marker_group = RocheMarkerGroup(
                label=int(label),
                name=marker_group_name,
                description=marker_group_name,
                textgui='red',
                locked=False,
                visible=True,
                type=marker_type
            )
            marker_groups.append(marker_group)
            color_name = Pdl1Consts.display_color_dict[label]
            color = hex_to_rgba(RocheColor.get_by_name(color_name).value)
            marker_shape = RocheMarkerShape(
                level=0,
                outline_width=0,
                size=7,
                type='circle',
                color=color,
                label_color=color,
                outline_color='rgba(0,0,0,0)'
            )
            marker_shapes[marker_type] = marker_shape

        wsi_marker_preset = RocheMarkerPreset(
            textgui='Nuclei',
            active=True,
            data=marker_groups
        )

        markers_panel = RochePanel(
            name='Markers',
            description='Markers'
        )

        heatmap_panel = RochePanel(
            name='Heatmaps',
            description='Heatmaps'
        )

        heatmap = RocheHeatMap(
            gui=PDL1_ALGORITHM_DISPLAY_ID,
            labels=[{'0': 'None'}, {'255': 'High Density'}],
            level_opacity=gen_opacities(len(wsi_input.dimensions))
        )

        thumbnail = slide.get_thumbnail(size=max(wsi_input.dimensions[-1]))

        return RocheALGResult(
            ai_results=ai_result,
            wsi_input=wsi_input,
            cells_index_items=cell_index_items,
            marker_presets=[wsi_marker_preset],
            marker_shapes=marker_shapes,
            heatmaps=[heatmap],
            panels=[markers_panel, heatmap_panel],
            thumbnail=convert_image_to_png_data(thumbnail))

    def save_ai_result(self, task: RocheAITaskEntity, result: RocheALGResult) -> bool:
        task.update_data(ai_results=result.ai_results)
        self.repository.save_ai_task(task)

        buffer = BytesIO()
        with h5py.File(buffer, 'w') as file:
            wsi_analysis_info = file.create_group('wsi_analysis_info')
            wsi_analysis_info['diplomat'] = json.dumps(RocheDiplomat(uuid=task.analysis_id).to_dict())

            algorithm_entity = self.repository.get_algorithm(task.algorithm_id)
            if algorithm_entity:
                wsi_analysis_info['algorithm'] = json.dumps(algorithm_entity.to_dict(), cls=CyborgJsonEncoder)

            wsi_analysis_info['input'] = json.dumps(result.wsi_input.to_dict())

            wsi_cells = file.create_group('wsi_cells')
            wsi_cells['index'] = json.dumps([item.to_dict() for item in result.cells_index_items])
            for index_item in result.cells_index_items:
                wsi_cells.create_dataset(name=index_item.filename, data=[json.dumps(index_item.geo_json)])

            wsi_presentation = file.create_group('wsi_presentation')
            wsi_presentation['marker_shapes'] = json.dumps(
                {k: v.to_dict() for k, v in result.marker_shapes.items() if v})
            wsi_presentation['markers'] = json.dumps([preset.to_dict() for preset in result.marker_presets if preset])
            wsi_presentation['panels'] = json.dumps([panel.to_dict() for panel in result.panels if panel])
            wsi_presentation['heatmaps'] = json.dumps([heatmap.to_dict() for heatmap in result.heatmaps])
            wsi_presentation['colorluts'] = json.dumps(HEAT_COLORLUTS)

            wsi_sparse_heatmaps = file.create_group('wsi_sparse_heatmaps')
            wsi_sparse_heatmaps['index'] = json.dumps([item.to_dict() for item in result.cells_index_items])
            for index_item in result.cells_index_items:
                wsi_sparse_heatmaps.create_dataset(name=index_item.filename, data=[json.dumps([{
                    HER2_ALGORITHM_DISPLAY_ID: index_item.heatmap_points
                }])])

            wsi_thumbnail = file.create_group('wsi_thumbnail')
            thumbnail_name = f'level{len(result.wsi_input.dimensions) - 1}.png'
            wsi_thumbnail.create_dataset(thumbnail_name, data=result.thumbnail, dtype=np.int8)

            # wsi_masks = file.create_group('wsi_masks')

        buffer.seek(0)
        return oss.put_object_from_io(buffer, task.result_file_key)

    def callback_analysis_status(self, ai_task: RocheAITaskEntity) -> bool:
        url = f'{Settings.ROCHE_API_SERVER}/analysis'
        data = {
            'analysis_id': ai_task.analysis_id,
            'status': ai_task.status_name,
            'status_detail_message': '',
            'percentage_completed': ai_task.percentage_completed,
            'started_timestamp': ai_task.started_at.strftime(ROCHE_TIME_FORMAT) if ai_task.started_at else None,
            'last_updated_timestamp': ai_task.last_modified.strftime(
                ROCHE_TIME_FORMAT) if ai_task.last_modified else None
        }
        res = requests.put(url, json=data, timeout=10)

        logger.info(url)
        logger.info(data)
        logger.info(res.status_code)
        if res.status_code == 200:
            return True
        else:
            return False
