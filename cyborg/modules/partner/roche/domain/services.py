import json
import logging
import uuid
from datetime import datetime
from io import BytesIO
from itertools import groupby
from typing import Optional
from urllib import request
from urllib.parse import urlparse

import h5py
from geojson import Feature, MultiPoint

from cyborg.app.settings import Settings
from cyborg.consts.her2 import Her2Consts
from cyborg.infra.oss import oss
from cyborg.libs.heimdall.dispatch import open_slide
from cyborg.modules.partner.roche.domain.entities import RocheAITaskEntity
from cyborg.modules.partner.roche.domain.repositories import RocheRepository
from cyborg.modules.partner.roche.domain.value_objects import RocheAITaskStatus, RocheALGResult, \
    RocheDiplomat, RocheWsiInput, RocheCellsIndexItem, RocheMarkerPreset, RocheMarkerGroup, \
    RocheMarkerShape, RochePanel
from cyborg.modules.partner.roche.utils.color import hex_to_rgba
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

        return self.repository.save_ai_task(task)

    def download_slide(self, task: RocheAITaskEntity) -> str:
        if 'amazonaws' not in task.slide_url and Settings.ROCHE_IMAGE_SERVER:
            parsed_url = urlparse(task.slide_url)
            slide_url = task.slide_url.replace(
                f'{parsed_url.scheme}://{parsed_url.netloc}', Settings.ROCHE_IMAGE_SERVER)
        else:
            slide_url = task.slide_url

        logger.info(slide_url)
        request.urlretrieve(slide_url, task.slide_path)
        return task.slide_path

    def run_her2(self, task: RocheAITaskEntity) -> RocheALGResult:
        if not self.download_slide(task):
            return RocheALGResult(err_msg='切片url无法下载')

        input_info = task.input_info
        slide_width = input_info.get('slide_width', 0)
        slide_height = input_info.get('slide_height', 0)
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
            slide_depth=0,
            number_levels=1,
            dimensions=[
                [slide_width, slide_height],
            ]
        )
        # mpp = slide.mpp or 0.242042

        coordinates = task.coordinates
        if coordinates:
            x_coords, y_coords = zip(coordinates)
        else:
            x_coords, y_coords = [], []

        roi_id = IdWorker.new_mark_id_worker().get_new_id()

        mock_result_file_key = oss.path_join('test', 'mock_result.json')

        try:
            mock_result = oss.get_object(mock_result_file_key)
            mock_result = tuple(json.loads(mock_result))
        except Exception:
            mock_result = None

        slide = open_slide(task.slide_path)
        slide_width, slide_height = slide.width, slide.height

        from cyborg.modules.ai.libs.algorithms.Her2New_.detect_all import run_her2_alg, roi_filter
        if mock_result:
            center_coords_np_with_id, cls_labels_np_with_id, summary_dict, lvl, flg = mock_result
            roi_id = list(center_coords_np_with_id.keys())[0]
        else:
            center_coords_np_with_id, cls_labels_np_with_id, summary_dict, lvl, flg = run_her2_alg(
                slide_path=task.slide_path, roi_list=[{'id': roi_id, 'x': x_coords, 'y': y_coords}])

            result_data = json.dumps([center_coords_np_with_id, cls_labels_np_with_id, summary_dict, lvl, flg])
            logger.info(result_data)
            buffer = BytesIO()
            buffer.write(result_data.encode('utf-8'))
            buffer.seek(0)
            oss.put_object_from_io(buffer, mock_result_file_key)

        center_coords, cls_labels = roi_filter(
            center_coords_np_with_id[roi_id],
            cls_labels_np_with_id[roi_id],
            x_coords,
            y_coords
        )

        ai_result = Her2Consts.rois_summary_dict.copy()
        label_to_roi_name = Her2Consts.cell_label_dict

        cells_in_tile = {}
        for idx, coord in enumerate(center_coords):
            label = str(cls_labels[idx])
            roi_name = label_to_roi_name[label]
            ai_result[roi_name] += 1

            x = float(coord[0]) if slide_width > float(coord[0]) else float(slide_width - 1)
            y = float(coord[1]) if slide_height > float(coord[1]) else float(slide_height - 1)

            tile_x, tile_y = RocheCellsIndexItem.locate(int(x), int(y))
            cells = cells_in_tile.setdefault((tile_x, tile_y), [])
            cells.append([label, [x, y]])

        index_items = []
        for (tile_x, tile_y), cells in cells_in_tile.items():
            index_item = RocheCellsIndexItem.new_item(tile_x, tile_y)
            for label, cell_group in groupby(cells, lambda c: c[0]):
                feature = Feature(properties={'label': int(label)}, geometry=MultiPoint(coordinates=[]))
                for cell in cell_group:
                    feature['geometry']['coordinates'].append([int(val) for val in cell[1]])
                index_item.geo_json['features'].append(feature)
            index_items.append(index_item)

        # group_id = group_name_to_id.get('ROI') if whole_slide != 1 else None

        whole_slide = 1 if len(x_coords) == 0 else 0
        ai_result.update({
            'whole_slide': whole_slide,
            '分级结果': Her2Consts.level[int(lvl)]
        })

        marker_groups = []
        marker_shapes = {}
        for label, roi_name in label_to_roi_name.items():
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

        panel = RochePanel(
            name='Markers',
            description='Markers'
        )

        return RocheALGResult(
            ai_suggest=ai_result['分级结果'],
            wsi_input=wsi_input,
            cells_index_items=index_items,
            marker_presets=[wsi_marker_preset],
            marker_shapes=marker_shapes,
            panels=[panel]
        )

    def save_ai_result(self, task: RocheAITaskEntity, result: RocheALGResult) -> bool:
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

            # wsi_masks = file.create_group('wsi_masks')
            # wsi_heatmaps = file.create_group('wsi_heatmaps')
            # wsi_thumbnail = file.create_group('wsi_thumbnail')

        buffer.seek(0)
        return oss.put_object_from_io(buffer, task.result_file_key)
