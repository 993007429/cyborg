import enum
import json
import math
from typing import List, Optional

from cyborg.consts.her2 import Her2Consts
from cyborg.consts.np import NPConsts
from cyborg.consts.pdl1 import Pdl1Consts
from cyborg.seedwork.domain.value_objects import BaseEnum, BaseValueObject, AIType


@enum.unique
class MarkType(BaseEnum):
    manual = 1
    ai_point = 2
    ai_area = 3
    reference = 4


class AIResult(BaseValueObject):

    data: Optional[dict]

    @classmethod
    def initialize(cls, ai_type: AIType) -> 'AIResult':
        return AIResult(data=cls.get_init_data(ai_type))

    def to_string(self):
        return json.dumps(self.data)

    def to_dict(self):
        return self.data

    @classmethod
    def get_init_data(cls, ai_type: AIType) -> Optional[dict]:
        """
        初始化某个区域的算法结果（用于新增算法区域的情况）
        """
        if ai_type == AIType.celldet:
            return {'heterogeneous_area': 0,
                    'total': 0,
                    'index': 0.0,
                    'whole_slide': 0,  # 默认不是全场
                    }
        elif ai_type in [AIType.ki67, AIType.er, AIType.pr, AIType.ki67hot]:
            return {
                'total': 0,  # 肿瘤细胞数
                'pos_tumor': 0,  # 阳性肿瘤
                'neg_tumor': 0,  # 阴性肿瘤
                'normal_cell': 0,  # 非肿瘤
                'index': 0.0,  # 指数
                'whole_slide': 0,  # 默认不是全场
            }
        elif ai_type == AIType.pdl1:
            return {
                'neg_norm': 0,
                'neg_tumor': 0,
                'pos_norm': 0,
                'pos_tumor': 0,
                'total': 0,
                'tps': 0,
                'whole_slide': 0,
            }
        elif ai_type in [AIType.tct, AIType.lct, AIType.dna]:
            return {
                'cell_num': 0,
                'clarity': 0.0,
                'slide_quality': "",
                'diagnosis': ["", ""],
                'microbe': [""],
                'cells': {
                    'ASCUS': {'num': 0, 'data': []},
                    'ASC-H': {'num': 0, 'data': []},
                    'LSIL': {'num': 0, 'data': []},
                    'HSIL': {'num': 0, 'data': []},
                    'AGC': {'num': 0, 'data': []},
                    '滴虫': {'num': 0, 'data': []},
                    '霉菌': {'num': 0, 'data': []},
                    '线索': {'num': 0, 'data': []},
                    '疱疹': {'num': 0, 'data': []},
                    '放线菌': {'num': 0, 'data': []}
                },
                'whole_slide': 1,
            }
        elif ai_type == AIType.her2:
            summary = Her2Consts.rois_summary_dict
            summary['whole_slide'] = 0
            return summary
        elif ai_type == AIType.np:
            return {
                '嗜酸性粒细胞': {'count': 0, 'index': 0, 'area': None},
                '淋巴细胞': {'count': 0, 'index': 0, 'area': None},
                '浆细胞': {'count': 0, 'index': 0, 'area': None},
                '中性粒细胞': {'count': 0, 'index': 0, 'area': None},
                '上皮区域': {'count': None, 'index': 0, 'area': 0, 'total_area': 0},
                '腺体区域': {'count': None, 'index': 0, 'area': 0, 'total_area': 0},
                '血管区域': {'count': None, 'index': 0, 'area': 0, 'total_area': 0},
                'whole_slide': 0
            }
        elif ai_type == AIType.bm:
            return {
                'cell_num': 0,
                'clarity': 0.0,
                'slide_quality': "",
                'diagnosis': "",
                'cells': [
                    {
                        'label': '骨髓细胞分类-红系',
                        'num': 0,
                        'data': [
                            {
                                'label': '晚幼红细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '原始红细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '中幼红细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '早幼红细胞',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': '骨髓细胞分类-粒系',
                        'num': 0,
                        'data': [
                            {
                                'label': '中性分叶核粒细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '嗜酸性粒细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '原始粒细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '早幼粒细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '中性中幼粒细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '中性杆状核粒细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '嗜碱性粒细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '异常早幼粒细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '异常中幼粒细胞t(8,21)',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '异常嗜酸性粒细胞 inv(16)',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '中性晚幼粒细胞',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': '骨髓细胞分类-淋巴系',
                        'num': 0,
                        'data': [
                            {
                                'label': '淋巴细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '原始淋巴细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '反应性淋巴细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '大颗粒淋巴细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '毛细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '套细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '滤泡细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': 'Burkkit细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '淋巴瘤细胞（其他）',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '幼稚淋巴细胞',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': 'MDS病态造血-红系',
                        'num': 0,
                        'data': [
                            {
                                'label': '核异常-核出芽',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '核异常-核间桥',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '核异常-核碎裂',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '核异常-多个核',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '胞浆异常-胞浆空泡',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '大小异常-巨幼样变-红',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': '骨髓细胞分类-单核系',
                        'num': 0,
                        'data': [
                            {
                                'label': '原始单核细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '单核细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '异常单核细胞（PB）',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '幼稚单核细胞',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': '骨髓细胞分类-其他细胞',
                        'num': 0,
                        'data': [
                            {
                                'label': '成骨细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '破骨细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '戈谢细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '海蓝细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '尼曼匹克细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '分裂象',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '转移瘤细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '吞噬细胞',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': '骨髓细胞分类-浆细胞',
                        'num': 0,
                        'data': [
                            {
                                'label': '浆细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '骨髓瘤细胞',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': '骨髓细胞分类-Auer小体',
                        'num': 0,
                        'data': [
                            {
                                'label': '柴捆细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '含Auer小体细胞',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': 'MDS病态造血-粒系',
                        'num': 0,
                        'data': [
                            {
                                'label': '核异常-分叶过多',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '核异常-分叶减少',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '胞浆异常-颗粒减少',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '胞浆异常-杜勒小体',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '胞浆异常-Auer小体',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '大小异常-巨幼样变',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': '骨髓细胞分类-巨核系',
                        'num': 0,
                        'data': [
                            {
                                'label': '原始巨核细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '幼稚巨核细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '颗粒型巨核细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '产版型巨核细胞',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '裸核型巨核细胞',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': 'MDS病态造血-巨核系',
                        'num': 0,
                        'data': [
                            {
                                'label': '大小异常-微小巨核',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '大小异常-单圆巨核',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '大小异常-多圆巨核',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': '核异常-核分叶减少',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': 'MPN巨核细胞',
                        'num': 0,
                        'data': [
                            {
                                'label': 'CML-侏儒状巨核',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': 'ET-鹿角状巨核',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': 'PMF-气球状巨核',
                                'num': 0,
                                'data': []
                            },
                        ]
                    },
                    {
                        'label': '骨髓细胞分类-Artefacts',
                        'num': 0,
                        'data': [
                            {
                                'label': 'Smudge cell',
                                'num': 0,
                                'data': []
                            },
                            {
                                'label': 'Artefact',
                                'num': 0,
                                'data': []
                            },
                        ]
                    }
                ],
                'whole_slide': 1,
            }
        return None


class SliceTile(BaseValueObject):

    x: int
    y: int
    z: int


class MarkPosition(BaseValueObject):

    x_coords: List[float]
    y_coords: List[float]

    @property
    def is_polygon(self):
        return len(self.x_coords) >= 3 and len(self.y_coords) >= 3

    def to_path(self):
        return {
            'x': self.x_coords,
            'y': self.y_coords
        }


class TiledSlice(BaseValueObject):

    width: float
    height: float
    max_level: int
    mpp: float
    tile_size: int = 128
    max_num_per_tile: int = 5000

    def cal_tiles(
            self, x_coords: List[float], y_coords: List[float], level: Optional[int] = None
    ) -> List[SliceTile]:
        assert len(x_coords) == len(y_coords)
        tiles = []
        num_coords = len(x_coords)
        z = self.max_level if level is None else level
        this_level_tile_size = self.tile_size * (2 ** (self.max_level - z))
        tile_xmax, tile_ymax = math.ceil(self.width / this_level_tile_size), math.ceil(
            self.height / this_level_tile_size)

        if num_coords == 1:
            x_coord, y_coord = x_coords[0], y_coords[0]

            x = math.floor(x_coord / this_level_tile_size)
            y = math.floor(y_coord / this_level_tile_size)
            if 0 <= x <= tile_xmax and 0 <= y <= tile_ymax:
                tiles.append(SliceTile(x=x, y=y, z=z))
        else:  # Polygon
            xmin, xmax = max(0, math.floor(min(x_coords) / this_level_tile_size)), min(math.floor(
                max(x_coords) / this_level_tile_size), tile_xmax)
            ymin, ymax = max(0, math.floor(min(y_coords) / this_level_tile_size)), min(math.floor(
                max(y_coords) / this_level_tile_size), tile_ymax)
            for x in range(xmin, xmax + 1):
                for y in range(ymin, ymax + 1):
                    tiles.append(SliceTile(x=x, y=y, z=z))
        return tiles

    def tile_to_id(self, tile: SliceTile):
        """
        根据tile的坐标生成对应的唯一id
        :x: tile坐标x
        :y: tile坐标y
        :z: tile坐标z
        :return: tile唯一id
        """
        tile_id = 0
        for level in range(self.max_level, tile.z, -1):
            num_col = math.ceil(self.width / (self.tile_size * 2 ** (self.max_level - level)))
            num_row = math.ceil(self.height / (self.tile_size * 2 ** (self.max_level - level)))
            tile_id += num_col * num_row
        num_col = math.ceil(self.width / (self.tile_size * 2 ** (self.max_level - tile.z)))
        tile_id += (tile.x + tile.y * num_col)
        return tile_id

    def id_to_tile(self, tile_id: int) -> SliceTile:
        """
        :x: tile唯一id
        :return: SliceTile对象
        """
        z = self.max_level
        for level in range(self.max_level, -1, -1):
            num_col = math.ceil(self.width / (self.tile_size * 2 ** (self.max_level - level)))
            num_row = math.ceil(self.height / (self.tile_size * 2 ** (self.max_level - level)))
            if tile_id < num_col * num_row:
                z = level
                break
            else:
                tile_id -= num_col * num_row
        num_col = math.ceil(self.width / (self.tile_size * 2 ** (self.max_level - z)))
        y = math.floor(tile_id / num_col)
        x = tile_id - y * num_col
        return SliceTile(x=x, y=y, z=z)

    def get_shadow_tiles(self, source_tile: SliceTile, dest_level=0) -> List[SliceTile]:
        x, y, z = source_tile.x, source_tile.y, source_tile.z

        if x < 0 or y < 0 or x * 2 ** (self.max_level - z) * self.tile_size > self.width or y * 2 ** (
                self.max_level - z) * self.tile_size > self.height:
            return []

        xmin, xmax = math.floor(x * 2 ** (dest_level - z)), math.ceil((x + 1) * 2 ** (dest_level - z))
        ymin, ymax = math.floor(y * 2 ** (dest_level - z)), math.ceil((y + 1) * 2 ** (dest_level - z))

        shadow_tiles = []
        for col in range(xmin, xmax):
            for row in range(ymin, ymax):
                shadow_tiles.append(SliceTile(x=col, y=row, z=dest_level))
        return shadow_tiles

    def is_tile_intersect(self, x_coords: List[float], y_coords: List[float], tile: SliceTile) -> bool:
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))

        x, y, z = tile.x, tile.y, tile.z

        x_tile_min, x_tile_max = x * self.tile_size * 2 ** (self.max_level - z), (x + 1) * self.tile_size * 2 ** (
                self.max_level - z) - 1
        y_tile_min, y_tile_max = y * self.tile_size * 2 ** (self.max_level - z), (y + 1) * self.tile_size * 2 ** (
                self.max_level - z) - 1

        if x_tile_min > xmax or x_tile_max < xmin or y_tile_max < ymin or y_tile_min > ymax:
            return False
        else:
            return True

    def is_tile_inside_region(self, x_coords: List[float], y_coords: List[float], tile: SliceTile) -> bool:
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))

        x, y, z = tile.x, tile.y, tile.z

        x_tile_min, x_tile_max = x * self.tile_size * 2 ** (self.max_level - z), (x + 1) * self.tile_size * 2 ** (
                self.max_level - z) - 1
        y_tile_min, y_tile_max = y * self.tile_size * 2 ** (self.max_level - z), (y + 1) * self.tile_size * 2 ** (
                self.max_level - z) - 1
        if xmin <= x_tile_min and xmax >= x_tile_max and ymin <= y_tile_min and ymax >= y_tile_max:
            return True
        else:
            return False

    def recur_1(
            self, x_coords: List[float], y_coords: List[float], search_tile_list: List[SliceTile],
            level: int, max_search_lvl: int
    ) -> List[SliceTile]:
        return_tile_list = []
        next_search_list = []
        if level == max_search_lvl:
            for tile in search_tile_list:
                if self.is_tile_intersect(x_coords, y_coords, tile):
                    return_tile_list.append(tile)
        elif level < max_search_lvl:
            for tile in search_tile_list:
                if self.is_tile_inside_region(x_coords, y_coords, tile):
                    return_tile_list.append(tile)
                else:
                    if level + 1 <= max_search_lvl:
                        next_search_list += self.get_shadow_tiles(tile, dest_level=level + 1)
        else:
            return []
        return return_tile_list + self.recur_1(x_coords, y_coords, next_search_list, level + 1, max_search_lvl)

    def cal_tiles_in_quadrant(self, x_coords: List[float], y_coords: List[float], level: Optional[int] = None):
        search_max_lvl = self.max_level - 2
        if level >= 13:
            tile_list = self.cal_tiles(x_coords, y_coords, level=search_max_lvl)
        else:
            lvl10_tile_list = self.cal_tiles(x_coords, y_coords, level=10)
            tile_list = self.recur_1(x_coords, y_coords, lvl10_tile_list, level=10, max_search_lvl=search_max_lvl)

        return [self.tile_to_id(tile) for tile in tile_list]


class CellCount(BaseValueObject):

    tile_id: int
    ai_type: AIType
    cell_count_dict: Optional[dict] = None

    def __add__(self, other):
        cell_count_dict = self.to_dict()
        for k, v in other.to_dict().items():
            if k in cell_count_dict:
                cell_count_dict[k] += v.get('count', 0) if isinstance(v, dict) else v
            else:
                cell_count_dict[k] = v
        return CellCount(tile_id=self.tile_id, ai_type=self.ai_type, cell_count_dict=cell_count_dict)

    @property
    def total(self):
        total = 0
        for v in self.cell_count_dict.values():
            total += v
        return total

    def set_tile_id(self, tile_id):
        self.tile_id = tile_id
        return True

    def to_dict(self):
        if self.cell_count_dict:
            return self.cell_count_dict
        else:
            if self.ai_type == AIType.np:
                return {cell_type: 0 for cell_type in NPConsts.cell_type_list}
            elif self.ai_type == AIType.pdl1:
                return {cell_type: 0 for cell_type in Pdl1Consts.cell_type_list}
        return {}

    @classmethod
    def get_empty_dict(cls, tile_id: int, ai_type: AIType):
        if tile_id == -1:
            return {str(tile_id): CellCount(tile_id=tile_id, ai_type=ai_type).to_dict() for tile_id in [1, 2, 3, 4]}
        else:
            return CellCount(tile_id=tile_id, ai_type=ai_type).to_dict()


class SliceMarkConfig(BaseValueObject):

    radius: Optional[float] = None    # mpp倍率转换过后的半径
    is_solid: bool = False
