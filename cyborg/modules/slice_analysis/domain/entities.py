import json
import math
import sys
from json import JSONDecodeError
from typing import Tuple, List, Optional, Union
from urllib.parse import quote

from shapely.geometry import Polygon, Point

from cyborg.app.settings import Settings
from cyborg.modules.slice_analysis.domain.consts import HUMAN_TL_CELL_TYPES
from cyborg.modules.slice_analysis.domain.value_objects import SliceTile, MarkPosition, TiledSlice, AIType, \
    SliceMarkConfig
from cyborg.modules.slice_analysis.utils.polygon import cal_center
from cyborg.seedwork.domain.entities import BaseDomainEntity
from cyborg.utils.strings import camel_to_snake


class MarkGroupEntity(BaseDomainEntity):

    def to_dict(self):
        return {
            'id': str(self.id),
            'color': self.color,
            'is_show': self.is_show,
            'label': self.group_name
        }


class MarkEntity(BaseDomainEntity):
    group: Optional[MarkGroupEntity] = None
    area: float = 0
    is_in_manual: bool = False

    @property
    def json_fields(self) -> List[str]:
        return ['position', 'ai_result', 'diagnosis', 'doctor_diagnosis']

    def update_data(self, **kwargs):
        if 'type' in kwargs:
            kwargs['diagnosis'] = json.dumps({'type': kwargs.pop('type')})
        if 'image' in kwargs:
            kwargs['is_export'] = kwargs.pop('image')

        super().update_data(**kwargs)

    @property
    def diagnosis_type(self) -> Optional[int]:
        diagnosis = self.diagnosis
        return diagnosis['type'] if diagnosis else None

    def is_area_diagnosis_type(self, ai_type: AIType) -> bool:
        """
        某些算法不同的type之间可能不能互相转换，例如鼻息肉算法区域标注不能转换为细胞标注，需要根据实际的情况进行一次筛选
        """
        return ai_type == AIType.np and self.diagnosis_type in [5, 6, 7]

    def fix_diagnosis(self, ai_type: AIType):
        """此段代码是因为前端将该算法下的strokeColor作为type判定依据产生
        :param ai_type:
        :return:
        """
        if ai_type == AIType.fish_tissue:
            if self.stroke_color == '#fff' or self.stroke_color == 'white':
                self.update_data(diagnosis=json.dumps({'type': 0}))
            elif self.stroke_color == 'red':
                self.update_data(diagnosis=json.dumps({'type': 1}))
            elif self.stroke_color == '#00FF15':
                self.update_data(diagnosis=json.dumps({'type': 2}))

    @classmethod
    def fix_field_name(cls, column_name: str):
        """
        修复跟数据库不一致，且前端不愿意修改的字段名
        :param column_name:
        :return:
        """
        if column_name == 'image':
            column_name = 'is_export'
        return camel_to_snake(column_name)

    @property
    def mark_position(self) -> MarkPosition:
        position = self.position
        return MarkPosition(x_coords=position.get('x'), y_coords=position.get('y'))

    @property
    def center_point(self) -> Tuple[float, float]:
        position = self.mark_position
        return cal_center(position.x_coords, position.y_coords)

    def to_polygon(self) -> Optional[Polygon]:
        position = self.mark_position
        if position.is_polygon:
            return Polygon(list(zip(position.x_coords, position.y_coords)))
        return None

    def cal_polygon_area(self, mpp: float):
        polygon = self.to_polygon()
        return polygon.area * mpp ** 2

    def cal_tiles(
            self, tiled_slice: TiledSlice, level: Optional[int] = None, polygon_use_center: bool = False
    ) -> List[SliceTile]:
        position = self.mark_position
        x_coords, y_coords = position.x_coords, position.y_coords
        assert len(x_coords) == len(y_coords)
        if len(x_coords) >= 3 and polygon_use_center:
            center_point = self.center_point
            x_coords, y_coords = [center_point[0]], [center_point[1]]

        return tiled_slice.cal_tiles(x_coords=x_coords, y_coords=y_coords, level=level)

    def cal_pdl1s_count_tiles(self, tiled_slice: TiledSlice):
        position = self.mark_position
        x_coords, y_coords = position.x_coords[0], position.y_coords[0]
        tile_id_list = []
        maxlvl_store_count = tiled_slice.max_level - 1
        for z in range(max(maxlvl_store_count, 9), 9, -1):
            tile_x = math.floor(x_coords / (tiled_slice.tile_size * 2 ** (tiled_slice.max_level - z)))
            tile_y = math.floor(y_coords / (tiled_slice.tile_size * 2 ** (tiled_slice.max_level - z)))
            tile_id = tiled_slice.tile_to_id(SliceTile(x=tile_x, y=tile_y, z=z))
            tile_id_list.append(tile_id)
        return tile_id_list

    def cal_region(self, center_x: Union[int, float], center_y: Union[int, float]) -> str:
        """
        计算象限
        """
        position = self.mark_position
        x, y = position.x_coords[0], position.y_coords[0]
        if x < center_x and y < center_y:
            return '1'
        if x > center_x and y < center_y:
            return '2'
        if x < center_x and y > center_y:
            return '3'
        if x > center_x and y > center_y:
            return '4'

    @classmethod
    def parse_scope(cls, scope: Union[str, dict]) -> Tuple[dict, Polygon]:
        if isinstance(scope, str):
            scope = json.loads(scope)
        x_coords = scope.get('x')
        y_coords = scope.get('y')
        poly = Polygon(list(zip(x_coords, y_coords)))
        return scope, poly

    def is_in_polygon(self, polygon: Polygon):
        mark_position = self.mark_position
        x_coords = mark_position.x_coords
        y_coords = mark_position.y_coords
        return any(polygon.contains(Point(x, y_coords[i])) for i, x in enumerate(x_coords))

    def parse_ai_result(
            self, ai_type: AIType,
            is_deleted: Optional[int] = None, lesion_type: Optional[str] = None,
            page: int = 0, page_size: int = sys.maxsize, ai_suggest: Optional[dict] = None) -> dict:
        ai_result = self.ai_result
        if not ai_result:
            return {}
        new_ai_result = dict()
        if ai_type in [AIType.tct, AIType.lct, AIType.dna]:
            if ai_suggest:
                diagnosis, tbs_label = '', ''
                if '阴性' in ai_suggest:
                    diagnosis, tbs_label = '阴性', ''
                elif '阳性' in ai_suggest:
                    for label in ['HSIL', 'ASC-US', 'LSIL', 'AGC', 'ASC-H']:
                        if label in ai_suggest:
                            diagnosis, tbs_label = '阳性', label
                else:
                    pass
                if '样本不满意' in ai_suggest:
                    tbs_label = tbs_label + '-样本不满意'

                ai_result['diagnosis'] = [diagnosis, tbs_label]
            else:
                ai_result['diagnosis'] = ['', '']
        elif ai_type == AIType.her2:
            if ai_result:
                mark_1 = ai_result.get('微弱的不完整膜阳性肿瘤细胞') if ai_result.get(
                    '微弱的不完整膜阳性肿瘤细胞') else 0
                mark_2 = ai_result.get('弱中等的完整膜阳性肿瘤细胞') if ai_result.get(
                    '弱中等的完整膜阳性肿瘤细胞') else 0
                mark_3 = ai_result.get('中强度的不完整膜阳性肿瘤细胞') if ai_result.get(
                    '中强度的不完整膜阳性肿瘤细胞') else 0
                mark_4 = ai_result.get('强度的完整膜阳性肿瘤细胞') if ai_result.get('强度的完整膜阳性肿瘤细胞') else 0
                mark_7 = ai_result.get('阴性肿瘤细胞') if ai_result.get('阴性肿瘤细胞') else 0
                mark_9 = ai_result.get('组织细胞') if ai_result.get('组织细胞') else 0
                mark_10 = ai_result.get('淋巴细胞') if ai_result.get('淋巴细胞') else 0
                mark_11 = ai_result.get('纤维细胞') if ai_result.get('纤维细胞') else 0
                mark_12 = ai_result.get('其他非肿瘤细胞') if ai_result.get('其他非肿瘤细胞') else 0
                mark_6 = mark_1 + mark_2 + mark_3 + mark_4  # 阳性肿瘤细胞
                mark_8 = mark_6 + mark_7  # 肿瘤细胞总数
                mark_5 = mark_2 + mark_4
                new_ai_result['微弱的不完整膜阳性肿瘤细胞'] = [mark_1, mark_1 / mark_8] if mark_8 else [mark_1, 0]
                new_ai_result['弱中等的完整膜阳性肿瘤细胞'] = [mark_2, mark_2 / mark_8] if mark_8 else [mark_2, 0]
                new_ai_result['中强度的不完整膜阳性肿瘤细胞'] = [mark_3, mark_3 / mark_8] if mark_8 else [mark_3, 0]
                new_ai_result['强度的完整膜阳性肿瘤细胞'] = [mark_4, mark_4 / mark_8] if mark_8 else [mark_4, 0]
                new_ai_result['完整膜阳性肿瘤细胞'] = [mark_5, mark_5 / mark_8] if mark_8 else [mark_5, 0]
                new_ai_result['阳性肿瘤细胞'] = [mark_6, mark_6 / mark_8] if mark_8 else [mark_6, 0]
                new_ai_result['阴性肿瘤细胞'] = mark_7
                new_ai_result['肿瘤细胞总数'] = mark_8
                new_ai_result['其他'] = mark_9 + mark_10 + mark_11 + mark_12
                new_ai_result['阳性肿瘤细胞占比'] = mark_6 / mark_8 if mark_8 else 0
                new_ai_result['分级结果'] = ''

                def get_rank():
                    if mark_8:
                        if (mark_4 / mark_8) > 0.1:
                            new_ai_result['分级结果'] = 'HER-2 3+'
                            return
                        if ((mark_4 / mark_8) >= 0.005) and (mark_4 / mark_8) <= 0.1:
                            new_ai_result['分级结果'] = 'HER-2 2+'
                            return
                        if ((mark_5 / mark_8) > 0.1) and (mark_4 / mark_8) < 0.1:
                            new_ai_result['分级结果'] = 'HER-2 2+'
                            return
                        if ((mark_2 / mark_8) >= 0.02) and (mark_2 / mark_8) <= 0.1:
                            new_ai_result['分级结果'] = 'HER-2 1+'
                            return
                        if (((mark_1 + mark_3) / mark_8) > 0.1) and (mark_5 / mark_8) <= 0.1:
                            new_ai_result['分级结果'] = 'HER-2 1+'
                            return
                        if (((mark_1 + mark_3) / mark_8) <= 0.1) or (mark_6 / mark_8) < 0.01:
                            new_ai_result['分级结果'] = 'HER-2 0'
                            return

                get_rank()
                new_ai_result['whole_slide'] = ai_result.get('whole_slide') if ai_result.get(
                    'whole_slide') is not None else 1
                ai_result = new_ai_result
        elif ai_type == AIType.dna_ploidy:
            nuclei = ai_result.get('nuclei', [])
            if is_deleted == 1:
                new_nuclei = []
                for nucleus in nuclei:
                    if nucleus["is_deleted"] == 1:
                        new_nuclei.append(nucleus)
            if lesion_type == "all":
                new_nuclei = []
                for nucleus in nuclei:
                    if nucleus["is_deleted"] == 0:
                        new_nuclei.append(nucleus)
            if is_deleted == 0 and lesion_type != "all":
                new_nuclei = []
                for nucleus in nuclei:
                    if nucleus["lesion_type"] == lesion_type and nucleus["is_deleted"] == 0:
                        new_nuclei.append(nucleus)
            total = len(new_nuclei)
            start = (page - 1) * page_size if page > 1 else 0
            end = page * page_size
            paged_data = new_nuclei[start: end]
            ai_result["total"] = total
            ai_result["nuclei"] = paged_data
            if ai_suggest:
                ai_result["dna_diagnosis"] = ai_suggest

        return ai_result

    def to_dict_for_show(
            self, ai_type: AIType, radius_ratio: float = 1, is_max_level: bool = False,
            mark_config: Optional[SliceMarkConfig] = None, show_groups: Optional[List[int]] = None
    ):
        item = super().to_dict()
        item.update({
            'id': str(self.id),
            'diagnosis': self.diagnosis,
            'aiResult': self.ai_result,
            'path': self.position,
            'image': item.pop('is_export'),
            'radius': self.radius * radius_ratio if self.radius else 0,
            'area_id': str(self.area_id) if self.area_id else None,
            'fillColor': self.fill_color,
            'strokeColor': self.stroke_color
        })
        if ai_type in [AIType.ki67, AIType.er, AIType.pr, AIType.celldet, AIType.pdl1, AIType.her2, AIType.ki67hot]:
            if self.mark_type == 2:
                if self.ai_type == 'pdl1':  # pdl1只有最大层级显示其他类型标注, 其他层级只显示阳性肿瘤细胞（产品提的特殊定制需求）
                    if not is_max_level and item['diagnosis'].get('type') != 3:
                        return None
            elif self.mark_type == 3:
                ai_result = item['aiResult']
                if ai_result:
                    if ai_result.get('whole_slide') == 1:
                        return None
            item['show_layer'] = [3, 1, 2, 4].index(self.mark_type)

        elif ai_type == AIType.fish_tissue:
            if self.mark_type == 3:
                return None
            item['show_layer'] = 1 if self.method == 'spot' else 0

        elif ai_type == AIType.label:
            assert mark_config is not None
            if self.radius:
                if self.method == 'spot':
                    item['radius'] = mark_config.radius
                else:
                    item['radius'] = self.radius * radius_ratio

            if self.method == 'spot':
                mark_color = self.group.color if self.group else self.fill_color
                if mark_config.is_solid == 1:
                    item['fillColor'] = mark_color
                    item.pop('stroke_color')
                else:
                    item['strokeColor'] = mark_color
                    item.pop('fillColor')
            else:
                item['strokeColor'] = self.group.color if self.group else self.stroke_color

            if show_groups and self.group_id not in show_groups:
                return None

            item['show_layer'] = 0 if Settings.LAST_SHOW_GROUPS and self.group_id in Settings.LAST_SHOW_GROUPS else 1
        else:
            try:
                item['doctorDiagnosis'] = self.doctor_diagnosis
            except (TypeError, JSONDecodeError):
                item['doctorDiagnosis'] = None
        return item

    @classmethod
    def make_roi_image_url(cls, id: str, caseid: str, fileid: str, filename: str, path: dict, company: str, **_):
        roi = [
            [min(path['x']), min(path['y'])],
            [max(path['x']), max(path['y'])]
        ]
        return f'{Settings.IMAGE_SERVER}/files/ROI2?caseid={caseid}&fileid={fileid}&filename={quote(filename)}&roi={json.dumps(roi)}&roiid={id}&companyid={company}'

    @classmethod
    def make_image_url(cls, caseid: str, fileid: str, filename: str, company: str, **_):
        return f'{Settings.IMAGE_SERVER}/files/image?caseid={caseid}&fileid={fileid}&filename={quote(filename)}&companyid={quote(company)}'

    def to_roi(self, ai_type: AIType, is_deleted: int, lesion_type: str, page: int, page_size: int,
               ai_suggest: Optional[dict] = None):
        d = self.to_dict()
        ai_result = self.parse_ai_result(
            ai_type=ai_type,
            is_deleted=is_deleted,
            lesion_type=lesion_type,
            page=page,
            page_size=page_size,
            ai_suggest=ai_suggest
        )
        d['aiResult'] = ai_result
        d['path'] = self.position
        d['id'] = str(self.id)
        d['area_id'] = str(self.area_id) if self.area_id else None
        d['image'] = self.is_export
        d['fillColor'] = self.fill_color
        d['strokeColor'] = self.stroke_color
        d['doctorDiagnosis'] = self.doctor_diagnosis
        d['remark'] = self.doctor_diagnosis if ai_type == AIType.human_tl else self.remark
        return d

    @classmethod
    def mock_roi(cls) -> dict:
        cells = {cell_type: {'num': 0, 'data': []} for cell_type in HUMAN_TL_CELL_TYPES}
        ai_result = {
            'cell_num': 0,
            'clarity': 0.0,
            'slide_quality': '',
            'diagnosis': ["", ""],
            'microbe': [""],
            'cells': cells,
            'whole_slide': 1,
            'aiNotCompleted': True,  # 前端需求，如果没有选中算法没计算过，则在切片详情中不展示任何数据
            # DNA
            'nuclei': [],
            'num_abnormal_low': 0,
            'num_abnormal_high': 0,
            'num_normal': 0,
            'dna_diagnosis': '',
            'nuclei_num': 0,
            'control_iod': 0
        }
        return {'aiResult': ai_result, 'method': 'rectangle', 'mark_type': 3, 'area_id': None,
                'path': {"x": [0], "y": [0]}}

    def get_cell_type(self, diagnosis: List[str], roi_type: str) -> str:
        if not diagnosis:
            return roi_type
        if not isinstance(diagnosis, (list, tuple)):
            return roi_type
        if diagnosis[0] == "阴性":
            if roi_type == "HSIL":
                return "可疑1"
            elif roi_type == "ASC-H":
                return "可疑2"
            elif roi_type == "LSIL":
                return "可疑3"
            elif roi_type == "ASCUS":
                return "可疑4"
            elif roi_type == "AGC":
                return "可疑5"
        return roi_type


class MarkToTileEntity(BaseDomainEntity):
    ...


class NPCountEntity(BaseDomainEntity):

    def to_dict(self):
        return {
            'pos_tumor': self.pos_tumor,
            'neg_tumor': self.neg_tumor,
            'pos_norm': self.pos_norm,
            'neg_norm': self.neg_norm
        }


class Pdl1sCountEntity(BaseDomainEntity):

    def to_dict(self):
        return {
            '中性粒细胞': self.neutrophils,
            '嗜酸性粒细胞': self.eosinophils,
            '浆细胞': self.plasmocyte,
            '淋巴细胞': self.lymphocyte
        }


class ChangeRecordEntity(BaseDomainEntity):
    ...
