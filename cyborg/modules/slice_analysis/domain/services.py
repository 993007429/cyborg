import logging
import random
import sys
import json
import time
import traceback
from collections import OrderedDict
from itertools import groupby
from typing import List, Tuple, Optional, Union

import numpy as np

from cyborg.app.settings import Settings
from cyborg.infra.session import transaction
from cyborg.modules.slice_analysis.domain.consts import HUMAN_TL_CELL_TYPES
from cyborg.modules.slice_analysis.domain.entities import MarkEntity, MarkToTileEntity, ChangeRecordEntity, \
    MarkGroupEntity
from cyborg.modules.slice_analysis.domain.repositories import SliceMarkRepository, AIConfigRepository
from cyborg.modules.slice_analysis.domain.value_objects import TiledSlice, AIType, AIResult, SliceMarkConfig, CellCount
from cyborg.modules.slice_analysis.utils.polygon import is_intersected
from cyborg.utils.id_worker import IdWorker
from cyborg.modules.slice_analysis.domain.consts import AI_TYPE_MANUAL_MARK_TABLE_MAPPING

logger = logging.getLogger(__name__)


class SliceAnalysisDomainService(object):

    def __init__(self, repository: SliceMarkRepository, config_repository: AIConfigRepository):
        super(SliceAnalysisDomainService, self).__init__()
        self.repository = repository
        self.config_repository = config_repository

    def get_mark_table_suffix(self, ai_type: AIType, template_id: int = 0) -> str:
        if ai_type == AIType.label and template_id:
            ai_name = self.config_repository.get_ai_name_by_template_id(template_id=template_id)
            return f'{ai_type.value}_{ai_name}'
        else:
            return ai_type.value

    @transaction
    def generate_mark_in_pyramid(
            self, mark: MarkEntity, tiled_slice: TiledSlice):
        """
        对于一个新增的标注，需要在分层金字塔中的每层找到对应的tiles并建立关系
        """
        mark_to_tiles = []

        # 每个标注点分层时，第零层全部显示
        tiles = mark.cal_tiles(tiled_slice=tiled_slice)  # 获取第零层的tile坐标
        for tile in tiles:
            tile_id = tiled_slice.tile_to_id(tile)
            mark_to_tiles.append(MarkToTileEntity(raw_data=dict(mark_id=mark.id, tile_id=tile_id)))

        current_tile = mark.cal_tiles(tiled_slice=tiled_slice, polygon_use_center=True)[0]

        # 从底层开始逐层映射tile，确定每个tile关联的标注数量，若关联的标注数量少于25，则在MarkToTile中插入记录反之则不插入
        for level in range(tiled_slice.max_level - 1, -1, -1):
            next_tile = tiled_slice.get_shadow_tiles(source_tile=current_tile, dest_level=level)[0]
            next_tile_id = tiled_slice.tile_to_id(next_tile)
            if mark.mark_type in [1, 2]:
                mark_count = self.repository.get_mark_count_by_tile_id(tile_id=next_tile_id)
                current_tile = next_tile
                decide = None
                if mark_count <= 5:
                    decide = 1
                elif mark_count <= 10:
                    decide = random.choice([0, 1, 1, 1, 1])
                elif mark_count <= 15:
                    decide = random.choice([0, 0, 1, 1, 1])
                elif mark_count <= 20:
                    decide = random.choice([0, 0, 0, 1, 1])
                elif mark_count <= 25:
                    decide = random.choice([0, 0, 0, 0, 1])

                if decide == 1:
                    mark_to_tiles.append(MarkToTileEntity(raw_data=dict(mark_id=mark.id, tile_id=next_tile_id)))
                else:
                    break

        self.repository.delete_mark_to_tiles_by_mark_id(mark.id)

        self.repository.save_mark_to_tiles(mark_to_tiles)

    @transaction
    def generate_ai_marks_in_pyramid(
            self, marks: List[MarkEntity], tiled_slice: TiledSlice, downsample_ratio: int = 4,
            downsample_threshold: int = 4000
    ):
        """
        对于ai生成的海量标注，需要在分层金字塔中的每层找到对应的tiles并建立关系, 数据一般已经经过采样
        """
        mark_to_tiles = []
        current_tiles = {}

        for mark in marks:
            tiles = mark.cal_tiles(tiled_slice=tiled_slice)  # 获取第零层的tile坐标
            for tile in tiles:
                tile_id = tiled_slice.tile_to_id(tile)
                mark_to_tiles.append(MarkToTileEntity(raw_data=dict(mark_id=mark.id, tile_id=tile_id)))
            tiles = mark.cal_tiles(tiled_slice=tiled_slice, polygon_use_center=True)
            if tiles:
                current_tiles[mark.id] = tiles[0]

        for level in range(tiled_slice.max_level - 1, 6, -1):
            if len(marks) > downsample_threshold and level < tiled_slice.max_level:
                marks = marks[::downsample_ratio]

            for mark in marks:
                next_tile = tiled_slice.get_shadow_tiles(source_tile=current_tiles[mark.id], dest_level=level)[0]
                next_tile_id = tiled_slice.tile_to_id(next_tile)
                mark_to_tiles.append(MarkToTileEntity(raw_data=dict(mark_id=mark.id, tile_id=next_tile_id)))
                current_tiles[mark.id] = next_tile

        self.repository.save_mark_to_tiles(mark_to_tiles)

    def show_marks(
            self, ai_type: AIType, marks: List[MarkEntity], radius_ratio: float = 1, is_max_level: bool = False,
            mark_config: Optional[SliceMarkConfig] = None, show_groups: Optional[List[int]] = None,
            is_manual: bool = False
    ) -> List[dict]:
        mark_list = []
        if not marks:
            return mark_list
        if not is_manual and ai_type in [AIType.tct, AIType.lct, AIType.dna]:
            """show_marks
            tct lct所有标注点不分层，不根据视野，在任何视野返回所有的点信息，标注点存储在aiResulit中
            """
            area_mark = marks[0]
            ai_result = area_mark.ai_result
            if ai_result:
                for k, v in ai_result.get('cells', {}).items():
                    for mark in v.get('data', []):
                        mark['mark_type'] = 1
                        mark['cell_type'] = area_mark.get_cell_type(ai_result["diagnosis"], k)
                        mark_list.append(mark)
                for nucleus in ai_result.get('nuclei', []):
                    nucleus['mark_type'] = 1
                    nucleus['iconType'] = 'dnaIcon'
                    mark_list.append(nucleus)
        elif ai_type == AIType.bm:
            area_mark = marks[0]
            ai_result = area_mark.ai_result
            if ai_result:
                for cell in ai_result.get('cells', []):
                    for item in cell.get('data', []):
                        for sub_item in item.get('data', []):
                            sub_item['mark_type'] = 1
                            cell_types = list(filter(None, [cell.get('label'), item.get('label'), sub_item.get('label')]))
                            sub_item['cell_type'] = '>'.join(cell_types)
                            mark_list.append(sub_item)
        else:
            mark_list = list(filter(None, [mark.to_dict_for_show(
                ai_type=ai_type, radius_ratio=radius_ratio, is_max_level=is_max_level,
                mark_config=mark_config, show_groups=show_groups
            ) for mark in marks if mark]))
            mark_list.sort(key=lambda x: x.get('show_layer', 0))
        return mark_list

    @transaction
    def create_mark(
            self,
            ai_type: AIType,
            tiled_slice: Optional[TiledSlice] = None,
            group_id: Optional[int] = None,
            position: Union[str, dict, None] = None,
            area_id: Optional[int] = None,
            stroke_color: Optional[str] = None,
            fill_color: Optional[str] = None,
            radius: Optional[float] = None,
            method: Optional[str] = None,
            editable: Optional[int] = None,
            diagnosis_type: Optional[int] = None,
            mark_type: Optional[int] = None,
            dashed: Optional[int] = None,
            doctor_diagnosis: Optional[dict] = None,
            ai_result_ready: Optional[bool] = None,
            is_export: Optional[int] = None,
            op_name: Optional[str] = None
    ) -> Tuple[str, Optional[MarkEntity]]:

        diagnosis = {'type': diagnosis_type} if diagnosis_type is not None else None

        ai_result = AIResult.initialize(ai_type=ai_type)
        if ai_result_ready is not None:
            ai_result.data['aiNotCompleted'] = 0 if ai_result_ready else 1

        if mark_type == 4:  # 若插入的标注为参考点，则需要删除原参考点
            self.repository.delete_mark_by_type(mark_type=mark_type)

        # TODO fishTissue算法需要获取group_id逻辑
        # group_id = self.get_group_id(ai_type=ai_type, mark=mark_params, group_id=group_id)

        new_mark = MarkEntity(raw_data=dict(
            position=position,
            method=method,
            ai_result=ai_result.to_string(),
            editable=editable,
            stroke_color=stroke_color,
            fill_color=fill_color,
            mark_type=mark_type,
            diagnosis=diagnosis,
            radius=radius,
            create_time=int(round(time.time() * 1000)),
            group_id=group_id,
            area_id=area_id,
            dashed=dashed,
            doctor_diagnosis=doctor_diagnosis,
            is_export=is_export
        ))

        saved = self.repository.save_mark(new_mark)
        if not saved:
            return 'create mark tailed', None

        if tiled_slice and tiled_slice.use_pyramid:
            self.generate_mark_in_pyramid(mark=new_mark, tiled_slice=tiled_slice)

        if group_id:
            self.repository.update_mark_group_status(group_id=group_id, is_empty=0)

        if op_name and ai_type in Settings.ai_log_list:
            table_name = 'Mark_{}'.format(ai_type)
            content = '新增标注-id:{}-position:{}-type:{}'.format(new_mark.id, new_mark.position, diagnosis_type)
            doctor_human_table_suffix = self.repository.manual.mark_table_suffix
            if doctor_human_table_suffix:
                content = '新增标注-id:{}-position:{}-type:{}'.format(
                    new_mark.id, new_mark.position, new_mark.doctor_diagnosis)
                table_name = 'Mark_{}'.format(doctor_human_table_suffix)

            change_record = ChangeRecordEntity(raw_data=dict(
                mark_id=new_mark.id, content=content, table_name=table_name, op_type='新增', op_name=op_name))
            self.repository.save_change_record(change_record)
        return '', new_mark

    @transaction
    def create_ai_marks(
            self,
            cell_marks: List[dict],
            roi_marks: List[dict],
            tiled_slice: Optional[TiledSlice] = None,
    ) -> Tuple[str, Optional[List[MarkEntity]]]:

        cell_mark_entities, roi_mark_entities = [], []
        group_ids = set()

        id_worker = IdWorker(1, 2, 0)
        for item in roi_marks + cell_marks:
            # TODO fishTissue算法需要获取group_id逻辑
            # group_id = self.get_group_id(ai_type=ai_type, mark=mark_params, group_id=group_id)
            item['create_time'] = int(round(time.time() * 1000))
            if 'id' not in item:
                item['id'] = id_worker.get_next_id() or id_worker.get_new_id()
            if not item['position']['x']:
                whole_slide_roi = self.repository.get_mark(item['id'])
                if whole_slide_roi:
                    whole_slide_roi.update_data(**item)
                else:
                    whole_slide_roi = MarkEntity(raw_data=item)
                self.repository.save_mark(whole_slide_roi)
                continue

            new_mark = MarkEntity(raw_data=item)
            if new_mark.group_id:
                group_ids.add(new_mark.group_id)
            if new_mark.mark_type == 3:
                roi_mark_entities.append(new_mark)
            else:
                cell_mark_entities.append(new_mark)

        saved = self.repository.batch_save_marks(
            roi_mark_entities) and self.repository.batch_save_marks(cell_mark_entities)
        if not saved:
            return 'create ai marks tailed', None

        if tiled_slice and tiled_slice.use_pyramid:
            self.generate_ai_marks_in_pyramid(marks=cell_mark_entities, tiled_slice=tiled_slice)
            self.generate_ai_marks_in_pyramid(
                marks=roi_mark_entities, tiled_slice=tiled_slice, downsample_ratio=2, downsample_threshold=500)

        for group_id in group_ids:
            self.repository.update_mark_group_status(group_id=group_id, is_empty=0)

        self.repository.backup_ai_mark_tables()

        return '', cell_mark_entities + roi_mark_entities

    @transaction
    def update_marks_by_scope(
            self, scope: Union[str, dict], target_group_id: int, tiled_slice: TiledSlice, ai_type: AIType,
            op_name: Optional[str] = None
    ) -> Tuple[int, str]:
        """
        画框多选模式下批量修改标注数据
        :param ai_type:
        :param scope:
        :param tiled_slice:
        :param target_group_id:
        :param op_name:
        :return:
        """
        scope, polygon = MarkEntity.parse_scope(scope)
        if tiled_slice.use_pyramid:
            tiles = tiled_slice.cal_tiles(x_coords=scope['x'], y_coords=scope['y'])
            if not tiles:  # 多选框范围超出切片范围
                return 2, 'no marks'
            tile_ids = [tiled_slice.tile_to_id(tile) for tile in tiles]
        else:
            tile_ids = None

        err_code, message = 0, 'modify marks succeed'
        _, marks = self.repository.get_marks(tile_ids=tile_ids)

        marks_for_ai_result = []
        involved_groups = set()
        for mark in marks:
            if not (mark and mark.is_in_polygon(polygon)):
                continue
            if ai_type == AIType.label and target_group_id and target_group_id != mark.group_id:
                involved_groups.add(mark.group_id)
                involved_groups.add(target_group_id)
                target_group = self.repository.get_mark_group_by_id(target_group_id)
                if target_group:
                    mark.update_data(
                        group_id=target_group.id,
                        stroke_color=target_group.color if mark.stroke_color else None,
                        fill_color=target_group.color if mark.fill_color else None,
                    )
            else:
                diagnosis_type = scope.get('type')
                if diagnosis_type is not None and mark.area_id:
                    if mark.is_area_diagnosis_type(ai_type=ai_type):
                        err_code, message = 1, '操作涉及区域标注，区域标注不支持修改'
                        continue

                    if ai_type in Settings.ai_log_list:
                        table_name = 'Mark_{}'.format(ai_type)
                        doctor_human_table_suffix = self.repository.manual.mark_table_suffix
                        if doctor_human_table_suffix:
                            table_name = 'Mark_{}'.format(doctor_human_table_suffix)
                        content = '修改标注类型-id:{}-before:{}-after:{}'.format(
                            mark.id, mark.diagnosis_type, diagnosis_type)
                        change_record = ChangeRecordEntity(raw_data=dict(
                            mark_id=mark.id, content=content, table_name=table_name, op_type='修改',
                            op_name=op_name))
                        self.repository.save_change_record(change_record)

                    target_color = scope.get('color')
                    mark.pre_diagnosis = mark.diagnosis
                    if mark.stroke_color and mark.area_id:
                        mark.update_data(stroke_color=target_color)
                    if mark.fill_color and mark.area_id:
                        mark.update_data(fill_color=target_color)
                    mark.update_data(diagnosis={'type': diagnosis_type})

                    marks_for_ai_result.append(mark)

                doctor_diagnosis = scope.get('doctorDiagnosis')
                if doctor_diagnosis:
                    if ai_type in Settings.ai_log_list:
                        content = '修改标注类型-id:{}-before:{}-after:{}'.format(
                            mark.id, mark.doctor_diagnosis, doctor_diagnosis)
                        table_name = 'Mark_{}'.format(ai_type)
                        doctor_human_table_suffix = self.repository.manual.mark_table_suffix
                        if doctor_human_table_suffix:
                            table_name = 'Mark_{}'.format(doctor_human_table_suffix)
                        change_record = ChangeRecordEntity(raw_data=dict(
                            mark_id=mark.id, content=content, table_name=table_name, op_type='新增',
                            op_name=op_name))
                        self.repository.save_change_record(change_record)

                    mark.update_data(doctor_diagnosis=doctor_diagnosis)

            self.repository.save_mark(mark)

        for group_id in involved_groups:
            has_mark = self.repository.has_mark(group_id=group_id)
            self.repository.update_mark_group_status(group_id=group_id, is_empty=has_mark is False)

        if marks_for_ai_result:
            self.update_ai_result(marks=marks_for_ai_result, option=3, ai_type=ai_type, tiled_slice=tiled_slice)

        return err_code, message

    @transaction
    def update_marks(self, marks_data: List[dict], target_group_id: int, tiled_slice: TiledSlice, ai_type: AIType):
        """
        非画框多选模式下批量修改标注数据
        """
        involved_groups = set()
        marks_for_ai_result = []
        err_code, message = 0, 'modify marks succeed'

        for mark_item in marks_data:
            mark_id = int(mark_item['id'])
            manual_mark = self.repository.manual.get_mark(mark_id)

            if ai_type in [AIType.tct, AIType.lct, AIType.dna] and not manual_mark:
                ############################################################################################
                # tct, lct, dna的mark数据存储在一个roi mark的ai_result里面，这里需要用一段单独的逻辑处理mark的修改
                ############################################################################################
                _, area_marks = self.repository.get_marks(mark_type=3, per_page=1)
                area_mark = area_marks[0] if area_marks else None
                wsi_ai_result = area_mark.ai_result if area_mark else None
                if wsi_ai_result:
                    for cell_type, value in wsi_ai_result['cells'].items():
                        data_list = value['data']
                        for idx, item in enumerate(data_list):
                            if str(item['id']) == str(mark_id):
                                for k, v in mark_item.items():
                                    wsi_ai_result['cells'][cell_type]['data'][idx][k] = v

                    for idx, item in enumerate(wsi_ai_result.get('nuclei', [])):
                        if str(item['id']) == str(mark_id):
                            for k, v in mark_item.items():
                                wsi_ai_result['nuclei'][idx][k] = v

                    area_mark.update_data(ai_result=wsi_ai_result)
                    self.repository.save_mark(area_mark)

            mark_item = {MarkEntity.fix_field_name(k): v for k, v in mark_item.items()}

            is_ai = False if manual_mark else True
            mark = self.repository.get_mark(mark_id) or manual_mark
            if not mark:
                continue

            if mark.is_area_diagnosis_type(ai_type=ai_type):
                err_code, message = 1, '操作涉及区域标注，区域标注不支持修改'
                continue

            mark.pre_diagnosis = mark.diagnosis

            mark.update_data(**mark_item)

            if ai_type == AIType.label and target_group_id:
                if target_group_id != mark.group_id:
                    involved_groups.add(mark.group_id)
                    involved_groups.add(target_group_id)
                    target_group = self.repository.get_mark_group_by_id(target_group_id)

                    if target_group:
                        mark.update_data(
                            group_id=target_group.id,
                            fill_color=target_group.color,
                            stroke_color=target_group.color
                        )

            if 'position' in mark_item:  # 修改标注位置（形状）
                if not mark_item.get('area_id'):
                    # 算法区域支持修改位置（形状），一旦修改需要删除算法区域内的点，并且清空算法区域的计算结果
                    self.repository.delete_marks(area_id=mark_id)
                    empty_ai_result = AIResult.initialize(ai_type=ai_type)
                    mark.update_data(ai_result=empty_ai_result.to_string() if empty_ai_result else None)

                if not mark.cal_tiles(tiled_slice=tiled_slice):  # 修改后的标注超出切片范围
                    if is_ai:
                        marks_for_ai_result.append(mark)
                    else:
                        self.repository.manual.delete_mark_by_id(mark.id)
                else:
                    if tiled_slice.use_pyramid:
                        self.generate_mark_in_pyramid(mark=mark, tiled_slice=tiled_slice)

            if 'type' in mark_item:
                if not mark.is_area_diagnosis_type(ai_type=ai_type):
                    mark.fix_diagnosis(ai_type=ai_type)
                    marks_for_ai_result.append(mark)

            if 'doctor_diagnosis' in mark_item:
                if self.repository.manual.mark_table_suffix:
                    mark.update_data(doctor_diagnosis=mark_item['doctor_diagnosis'])

            if is_ai:
                self.repository.save_mark(mark)
            else:
                self.repository.manual.save_mark(mark)

        for group_id in involved_groups:
            has_mark = self.repository.has_mark(group_id=group_id)
            self.repository.update_mark_group_status(group_id=group_id, is_empty=has_mark is False)

        if marks_for_ai_result:
            self.update_ai_result(marks=marks_for_ai_result, option=3, ai_type=ai_type, tiled_slice=tiled_slice)

        return err_code, message

    def get_marks(
            self, ai_type: AIType, view_path: dict, tiled_slice: TiledSlice,
            mark_config: Optional[SliceMarkConfig] = None,
            template_id: int = None
    ) -> List[dict]:
        x_coords = view_path.get('x')
        y_coords = view_path.get('y')
        z = view_path.get('z')
        if z is None:
            return []
        z = max(min(11, tiled_slice.max_level), z)
        is_max_level = (z == tiled_slice.max_level)
        radius_ratio = min(2.75 ** (tiled_slice.max_level - z), 32)  # 放大系数

        group_ids = self.repository.get_visible_mark_group_ids() if ai_type != 'human' else []
        # 所有手工标注不再分层，血细胞标注不分层
        if not tiled_slice.use_pyramid:
            _, marks = self.repository.get_marks()
        else:
            tiles = tiled_slice.cal_tiles(x_coords=x_coords, y_coords=y_coords, level=z)  # 获取视野tile坐标位置列表
            if not tiles and ai_type not in ['tct', 'lct']:  # 若tile_list为空，则说明当前视野超出切片范围，不返回任何标注
                return []

            tile_ids = [tiled_slice.tile_to_id(tile=tile) for tile in tiles]

            _, marks = self.repository.get_marks(tile_ids=tile_ids)
            _, area_marks = self.repository.get_marks(mark_type=3)
            marks += area_marks
        mark_list = self.show_marks(
            ai_type=ai_type, marks=marks, radius_ratio=radius_ratio, is_max_level=is_max_level,
            mark_config=mark_config, show_groups=group_ids)
        mark_cell_types = {}
        remove_mark = []
        for mark in mark_list:
            if ai_type == AIType.label and 'group_id' in mark:
                if mark['group_id'] not in mark_cell_types:
                    cell_type = self.get_mark_group(mark['group_id'], template_id=template_id)
                    if cell_type is None:
                        remove_mark.append(mark)
                        continue
                    mark_cell_types[mark['group_id']] = cell_type
                mark['cell_type'] = mark_cell_types[mark['group_id']]
        mark_list = [i for i in mark_list if i not in remove_mark]
        if ai_type not in [AIType.human, AIType.label]:
            """
            除了手工和标注模块外，还需显示手工模块标注。tct和lct模块显示专属的手工标注用于算法训练
            """
            value = AI_TYPE_MANUAL_MARK_TABLE_MAPPING.get(ai_type, 'human')
            ai_type = AIType.get_by_value(value)
            _, manual_marks = self.repository.manual.get_marks()
            manual_mark_list = self.show_marks(
                ai_type=ai_type, marks=manual_marks, radius_ratio=radius_ratio,
                mark_config=mark_config, show_groups=group_ids, is_manual=True)
            mark_list += manual_mark_list
        return mark_list

    def get_mark_group(self, group_id: int, template_id: int):
        if not group_id:
            return ''
        mark_groups = self.repository.get_mark_groups_by_template_id(template_id=template_id)
        group_map = dict()
        for group in mark_groups:
            group_map[group.id] = {"parent_id": group.parent_id, 'group_name': group.group_name}

        if group_id not in group_map:
            return ''

        group_name = [group_map[group_id].get('group_name')]
        parent_id = group_map[group_id]['parent_id']
        if parent_id:
            group_name.insert(0, group_map[parent_id].get('group_name'))
            first_parent_id = group_map[parent_id].get('parent_id')
            if first_parent_id:
                group_name.insert(0, group_map[first_parent_id].get('group_name'))
        return '>'.join(group_name)

    def can_use_pyramid(self, ai_type: AIType, is_imported_from_ai: bool) -> bool:
        if ai_type in [AIType.human_tl, AIType.human]:
            return False
        elif ai_type == AIType.label:
            return is_imported_from_ai
        else:
            return True

    def count_marks_in_scope(
            self, scope: Union[str, dict], tiled_slice: TiledSlice, ai_type: AIType
    ) -> Tuple[int, str, int]:
        scope, polygon = MarkEntity.parse_scope(scope)
        real_area = polygon.area * tiled_slice.mpp ** 2
        if real_area > Settings.MAX_AREA:
            return 1, '当前框选区域过大，请适当放大切片后框选。', 0

        tile_ids = None
        if tiled_slice.use_pyramid:
            tiles = tiled_slice.cal_tiles(x_coords=scope['x'], y_coords=scope['y'])
            if not tiles:
                return 0, 'query success', 0
            tile_ids = [tiled_slice.tile_to_id(tile) for tile in tiles]

        _, marks = self.repository.get_marks(tile_ids=tile_ids)

        if ai_type not in [AIType.human, AIType.label]:
            _, manual_marks = self.repository.manual.get_marks()
            _, area_marks = self.repository.get_marks(mark_type=3)
            marks += area_marks + manual_marks

        select_count = sum(1 for mark in set(marks)
                           if mark.is_in_polygon(polygon) and is_intersected(scope, mark.mark_position.to_path()))

        return 0, 'query success', select_count

    def _delete_marks(
            self, marks: List[MarkEntity], tiled_slice: TiledSlice, ai_type: AIType, op_name: str,
            from_manual: bool = False
    ) -> Tuple[int, str]:
        """
        根据标注类型删除标注
        :param marks:
        :param tiled_slice: 分层切片对象
        :param ai_type: 算法类型
        :param op_name: 操作人
        :param from_manual: 是否从算法对应的人工表删除
        :return:
        """
        err_code, message = 0, 'delete marks succeed'
        marks_for_ai_result = []
        mark_ids_for_delete = []
        repository = self.repository.manual if from_manual else self.repository
        for mark in marks:
            if mark.is_area_diagnosis_type(ai_type=ai_type):
                err_code, message = 1, '操作涉及区域标注，区域标注不支持删除'
                continue

            if mark.mark_type == 3:
                ai_result = mark.ai_result
                whole_slide = ai_result.get('whole_slide') if ai_result else None
                if whole_slide == 1:  # 全场roi
                    repository.clear_mark_table(ai_type=ai_type)
                    break

            if ai_type in Settings.ai_log_list:
                diagnosis_type = mark.diagnosis_type
                content = '删除标注-id:{}-position:{}-type:{}'.format(
                    mark.id, mark.position, diagnosis_type)
                table_name = 'Mark_{}'.format(ai_type)
                manual_mark_table_suffix = self.repository.manual.mark_table_suffix
                if manual_mark_table_suffix:
                    content = '删除标注-id:{}-position:{}-type:{}'.format(
                        mark.id, mark.position, mark.doctor_diagnosis)
                    table_name = 'Mark_{}'.format(manual_mark_table_suffix)

                change_record = ChangeRecordEntity(raw_data=dict(
                    mark_id=mark.id, content=content, table_name=table_name, op_type='删除', op_name=op_name))
                repository.save_change_record(change_record)

            mark_ids_for_delete.append(mark.id)
            if mark.mark_type == 3:
                marks_in_area = self.repository.get_marks_by_area_id(area_id=mark.id)
                for m in marks_in_area:
                    mark_ids_for_delete.append(m.id)
                if ai_type == AIType.pdl1:
                    tile_ids = []
                    for z in range(tiled_slice.max_level):
                        tiles = mark.cal_tiles(tiled_slice=tiled_slice, level=z)
                        tile_ids += [tiled_slice.tile_to_id(tile) for tile in tiles]

                    repository.delete_count(tile_ids=tile_ids, ai_type=ai_type)
            else:
                marks_for_ai_result.append(mark)

        logger.info(marks_for_ai_result)
        if marks_for_ai_result:
            self.update_ai_result(marks=marks_for_ai_result, option=2, ai_type=ai_type, tiled_slice=tiled_slice)

        repository.delete_marks(mark_ids=mark_ids_for_delete)

        return err_code, message

    @transaction
    def delete_marks_by_scope(
            self, scope: Union[str, dict], tiled_slice: TiledSlice, ai_type: AIType, op_name: str) -> Tuple[int, str]:
        """
        画框多选模式下批量删除标注
        """
        err_code, message = 0, ''
        scope, polygon = MarkEntity.parse_scope(scope)

        tile_ids = None
        if tiled_slice.use_pyramid:
            tiles = tiled_slice.cal_tiles(x_coords=scope['x'], y_coords=scope['y'])
            tile_ids = [tiled_slice.tile_to_id(tile) for tile in tiles]

        _, marks = self.repository.get_marks(tile_ids=tile_ids)
        _, area_marks = self.repository.get_marks(mark_type=3)
        if ai_type not in [AIType.human, AIType.label]:
            manual_marks_for_delete = []
            _, manual_marks = self.repository.manual.get_marks()
            for mark in manual_marks:
                if mark.is_in_polygon(polygon=polygon):
                    mark.is_in_manual = True
                    manual_marks_for_delete.append(mark)
            err, msg = self._delete_marks(
                marks=manual_marks_for_delete, tiled_slice=tiled_slice, ai_type=ai_type, op_name=op_name,
                from_manual=True)
            err_code, message = err_code or err, message or msg

        marks_for_delete = list(
            set([mark for mark in area_marks + marks if mark and mark.is_in_polygon(polygon=polygon)]))

        err, msg = self._delete_marks(
            marks=marks_for_delete, tiled_slice=tiled_slice, ai_type=ai_type, op_name=op_name)
        err_code, message = err_code or err, message or msg

        for group_id in set([mark.group_id for mark in marks_for_delete]):
            has_mark = self.repository.has_mark(group_id=group_id)
            self.repository.update_mark_group_status(group_id=group_id, is_empty=has_mark is False)

        return err_code, message

    @transaction
    def delete_marks(self, mark_ids: List[int], tiled_slice: TiledSlice, ai_type: AIType, op_name: str):
        """
        非画框多选模式下批量删除标注
        """
        err_code, message = 0, ''
        _, marks = self.repository.get_marks(
            mark_ids=mark_ids, mark_type=[1, 2, 3] if ai_type == AIType.label else [1, 2])
        if ai_type not in [AIType.human, AIType.label]:
            _, manual_marks = self.repository.manual.get_marks(mark_ids=mark_ids)
            if manual_marks:
                err, msg = self._delete_marks(
                    marks=manual_marks, tiled_slice=tiled_slice, ai_type=ai_type, op_name=op_name, from_manual=True)
                err_code, message = err_code or err, message or msg

            _, area_marks = self.repository.get_marks(mark_ids=mark_ids, mark_type=3)
            marks = list(set(marks + area_marks))

        if marks:
            err, msg = self._delete_marks(
                marks=marks, tiled_slice=tiled_slice, ai_type=ai_type, op_name=op_name)
            err_code, message = err_code or err, message or msg

        for group_id in set([mark.group_id for mark in marks]):
            has_mark = self.repository.has_mark(group_id=group_id)
            self.repository.update_mark_group_status(group_id=group_id, is_empty=has_mark is False)
        return err_code, message

    def import_ai_marks(self, template_id: int) -> str:

        err_msg = self.repository.create_mark_table_by_import()
        if err_msg:
            return err_msg

        self.repository.delete_mark_groups_by_template_id(template_id=template_id)
        groups = self.repository.get_default_mark_groups(template_id=template_id)
        for group in groups:
            self.repository.save_mark_group(group)

        return ''

    def get_or_create_default_area(self, ai_type: AIType) -> Tuple[str, Optional[MarkEntity]]:
        _, marks = self.repository.get_marks(mark_type=3)
        if marks:
            return '', marks[0]

        err_msg, mark = self.create_mark(
            ai_type, position={'x': [], 'y': []}, mark_type=3, method='rectangle', stroke_color='grey',
            radius=5, is_export=1)
        if err_msg:
            return err_msg, None
        return '', mark

    def _get_rois_by_cell_type(self, cell_type: str, ai_type: AIType) -> dict:
        """
        打包医生手工标注数据，打包成算法区域的形式（tct, lct, bm专用）
        :param cell_type: 细胞类型
        :param ai_type: 算法类型
        :return: roi数据
        """
        repository = self.repository
        if ai_type.is_human_type:
            repository = self.repository.manual
        marks = repository.get_marks_by_diagnosis_result(diagnosis_result=cell_type, ai_type=ai_type)
        return {
            'data': [mark.to_roi(ai_type=ai_type) for mark in marks],
            'num': len(marks),
            'label': cell_type
        }

    def get_human_rois(self, ai_type: AIType):
        """
        医生手工标注返回数据的方式不相同，在原有手工模块的基础上包装成算法区域的形式
        """
        is_empty = True
        if ai_type in [AIType.tct, AIType.lct, AIType.human_tl]:
            result = {}
            for cell_type in HUMAN_TL_CELL_TYPES:
                result[cell_type] = self._get_rois_by_cell_type(cell_type=cell_type, ai_type=ai_type)
                if result[cell_type]['num'] > 0:
                    is_empty = False
            return [] if is_empty else result
        elif ai_type in [AIType.bm, AIType.human_bm]:
            result = []
            label_cell_types = OrderedDict({
                '无': ['无'],
                '红系': ['晚幼红细胞', '原始红细胞', '中幼红细胞', '早幼红细胞'],
                '粒系': [
                    '中性分叶核粒细胞', '嗜酸性粒细胞', '原始粒细胞', '早幼粒细胞', '中性中幼粒细胞',
                    '中性杆状核粒细胞',
                    '嗜碱性粒细胞', '异常早幼粒细胞', '异常中幼粒细胞t(8,21)', '异常嗜酸性粒细胞 inv(16)',
                    '中性晚幼粒细胞'],
                '淋巴系': [
                    '淋巴细胞', '原始淋巴细胞', '反应性淋巴细胞', '大颗粒淋巴细胞', '毛细胞', '套细胞', '滤泡细胞',
                    'Burkkit细胞',
                    '淋巴瘤细胞（其他）', '幼稚淋巴细胞', '反应性淋巴细胞'],
                'MDS病态造血-红系': [
                    '核异常-核出芽', '核异常-核间桥', '核异常-核碎裂', '核异常-多个核', '胞浆异常-胞浆空泡',
                    '大小异常-巨幼样变-红'],
                '单核系': ['原始单核细胞', '单核细胞', '异常单核细胞（PB）', '幼稚单核细胞'],
                '其他细胞': [
                    '成骨细胞', '破骨细胞', '戈谢细胞', '海蓝细胞', '尼曼匹克细胞', '分裂象', '转移瘤细胞', '吞噬细胞'],
                '浆细胞': ['浆细胞', '骨髓瘤细胞'],
                'Auer小体': ['柴捆细胞', '含Auer小体细胞'],
                'MDS病态造血-粒系': [
                    '核异常-分叶过多', '核异常-分叶减少', '胞浆异常-颗粒减少', '胞浆异常-杜勒小体', '胞浆异常-Auer小体',
                    '大小异常-巨幼样变'],
                '巨核系': [
                    '原始巨核细胞', '幼稚巨核细胞', '颗粒型巨核细胞', '产版型巨核细胞', '裸核型巨核细胞'],
                'MDS病态造血-巨核系': [
                    '大小异常-微小巨核', '大小异常-单圆巨核', '大小异常-多圆巨核', '核异常-核分叶减少'],
                'MPN巨核细胞': ['CML-侏儒状巨核', 'ET-鹿角状巨核', 'PMF-气球状巨核'],
                'Artefacts': ['Smudge cell', 'Artefact'],
                '原虫及真菌': ['疟原虫', '利杜体', '马尔尼菲青霉菌', '荚膜组织胞浆菌'],
                '血小板': ['血小板']
            })
            for label, cell_types in label_cell_types.items():
                result.append({
                    'label': label,
                    'data': [self._get_rois_by_cell_type(
                        cell_type=cell_type, ai_type=ai_type) for cell_type in cell_types]
                })
            return result

    def get_rois(self, ai_type: AIType, ai_suggest: dict, ai_status: int,
                 is_deleted: int, lesion_type: str, page: int, page_size: int):
        if ai_type in [AIType.human_tl, AIType.human_bm]:
            rois = self.get_human_rois(ai_type)
        else:
            _, marks = self.repository.get_marks(mark_type=3 if ai_type != AIType.human else None)
            rois = list()
            for mark in marks:
                roi = mark.to_roi(ai_type=ai_type, ai_suggest=ai_suggest,
                                  is_deleted=is_deleted, lesion_type=lesion_type,
                                  page=page, page_size=page_size)
                # todo fish算法比较特殊，标注本身包含层级关系，细胞核和红绿信号点是包含关系，要做处理，只能返回全场roi
                if ai_type != AIType.fish_tissue or roi['aiResult']:
                    rois.append(roi)

            if not ai_status and ai_type not in [
                AIType.human, AIType.label, AIType.pdl1, AIType.ki67, AIType.er, AIType.pr, AIType.ki67hot,
                AIType.her2, AIType.fish_tissue, AIType.np
            ]:
                rois = [MarkEntity.mock_roi()] if ai_type in [AIType.tct, AIType.lct, AIType.dna] else []

        if len(rois) == 0:
            if ai_type in [AIType.tct, AIType.lct, AIType.bm]:
                msg, mark = self.create_mark(
                    ai_type, method='rectangle', mark_type=3,
                    position={'x': [0], 'y': [0]}, ai_result_ready=False)
                if msg:
                    logger.exception(msg)
                rois = [mark.to_roi(ai_type=ai_type)] if mark else []
            elif ai_type == AIType.human_tl:
                cells = {k: {'data': []} for k in HUMAN_TL_CELL_TYPES}
                rois = cells
        return rois

    def get_marks_in_group(self, group_id: int, page: int = 0, per_page: int = 20):
        return self.repository.get_marks(group_id=group_id, page=page, per_page=per_page, need_total=True)

    @transaction
    def update_mark_group(self, **kwargs):
        group = self.repository.get_mark_group_by_id(kwargs['id']) if 'id' in kwargs else None
        if not group:
            return
        group.update_data(group_name=kwargs.get('label'))
        group.update_data(color=kwargs.get('color'))
        group.update_data(default_color=kwargs.get('color'))
        group.update_data(op_time=time.time())
        group.update_data(parent_id=kwargs.get('parent_id'))
        self.repository.save_mark_group(group)

        _, marks = self.repository.get_marks(group_id=group.id, per_page=sys.maxsize)
        for mark in marks:
            if mark.stroke_color:
                mark.update_data(stroke_color=group.color)
            else:
                mark.update_data(fill_color=group.color)
            self.repository.save_mark(mark)

        for item in kwargs.get('children', []):
            self.update_mark_group(parent_id=group.id, **item)

    @transaction
    def sync_mark_groups(self, groups: List[MarkGroupEntity]) -> Tuple[int, str]:
        for group in groups:
            kwargs = {'group_name': group.group_name, 'template_id': group.template_id, 'id': group.id,
                      'label': group.label, 'color': group.color, 'parent_id': group.parent_id}
            group_ = self.repository.get_mark_group_by_kwargs(kwargs)
            try:
                if group_:
                    self.repository.save_mark_group(group)
                else:
                    self.repository.save_mark_group(group)
            except Exception:
                logger.info(traceback.format_exc())
                return 1, 'sync mark groups failed.'
        return 0, ''

    def show_mark_groups(self, groups: List[MarkGroupEntity]) -> List[dict]:
        data_list = list()
        for group in groups:
            mark_count = self.repository.get_mark_count(group_id=group.id)
            sub_groups = self.repository.get_mark_groups_by_parent_id(group.id)
            data_list.append({
                'id': group.id,
                'label': group.group_name,
                'color': group.default_color if sub_groups else group.color,
                'is_empty': group.is_empty,
                'is_show': group.is_show,
                'mark_count': mark_count,
                'children': self.show_mark_groups(sub_groups) if sub_groups else None
            })
            if sub_groups:
                group.update_data(is_show=any(group.is_show != -1 for group in sub_groups))
            self.repository.save_mark_group(group)
        return data_list

    @transaction
    def delete_mark_group(self, group_id: int) -> True:
        self.repository.delete_mark_group(group_id=group_id)
        self.repository.delete_marks(group_id=group_id)
        sub_groups = self.repository.get_mark_groups_by_parent_id(parent_id=group_id)
        for group in sub_groups:
            self.delete_mark_group(group.id)
        return True

    @transaction
    def switch_mark_group_show_status(self, group_id: int) -> Optional[str]:
        """
        切换标注分组的显示状态
        :param group_id:
        :return:
        """
        group = self.repository.get_mark_group_by_id(group_id)
        if not group:
            return 'group not existed'

        group.update_data(is_show=- group.is_show)
        self.repository.save_mark_group(group)

        sub_groups = self.repository.get_mark_groups_by_parent_id(group_id)
        for sub_group in sub_groups:
            sub_group.update_data(is_show=group.is_show)
            self.repository.save_mark_group(sub_group)

            lvl3_groups = self.repository.get_mark_groups_by_parent_id(sub_group.id)
            for lvl3_group in lvl3_groups:
                lvl3_group.update_data(is_show=group.is_show)
                self.repository.save_mark_group(lvl3_group)

        parent_group = self.repository.get_mark_group_by_id(group.parent_id) if group.parent_id else None
        if parent_group:
            sub_groups = self.repository.get_mark_groups_by_parent_id(parent_group.id)
            parent_group.update_data(is_show=1 if any(g.is_show == 1 for g in sub_groups) else -1)
            self.repository.save_mark_group(parent_group)

            grandpa_group = self.repository.get_mark_group_by_id(
                parent_group.parent_id) if parent_group.parent_id else None
            if grandpa_group:
                sub_groups = self.repository.get_mark_groups_by_parent_id(parent_group.parent_id)
                grandpa_group.update_data(is_show=1 if any(g.is_show == 1 for g in sub_groups) else -1)
                self.repository.save_mark_group(grandpa_group)

        return 'change group status success'

    def update_ai_result(
            self, marks: List[MarkEntity], option: int, ai_type: AIType, tiled_slice: TiledSlice
    ):
        """
        更新算法结果
        :param marks: 新增、删除或修改的标注列表
        :param option: 操作类型->1.新增;2.删除;3修改
        :param ai_type: 算法类型
        :param tiled_slice: 分层切片对象
        """
        slide_heterogeneous = 0  # 用于病例列表展示，当前切片所有阳性肿瘤细胞数量
        slide_total = 0  # 用于病例列表展示，当前切片所有肿瘤细胞数量

        for area_id, marks_in_area in groupby(marks, lambda m: int(m.area_id) if m.area_id else 0):
            area_mark = self.repository.get_mark(mark_id=area_id)
            if not area_mark:
                continue
            ai_result_before_modify = area_mark.ai_result  # 修改前的算法结果

            whole_slide = ai_result_before_modify.get('whole_slide')

            if ai_type in [AIType.celldet, ]:
                heterogeneous_area = ai_result_before_modify.get('heterogeneous_area')
                total = ai_result_before_modify.get('total')
                mark_1 = 0
                mark_3 = 0
                for mark in marks_in_area:
                    diagnosis_type = mark.diagnosis_type
                    if diagnosis_type == 1:
                        mark_1 += 1
                    if diagnosis_type == 3:
                        mark_3 += 1
                    if option == 1:  # 新增标注
                        mark_1 += total - heterogeneous_area
                        mark_3 += heterogeneous_area
                    elif option == 2:  # 删除标注
                        mark_1 = total - heterogeneous_area - mark_1
                        mark_3 = heterogeneous_area - mark_3
                    elif option == 3:  # 修改标注
                        previous_type = mark.pre_diagnosis['type']
                        mark_1_current = total - heterogeneous_area
                        mark_3_current = heterogeneous_area
                        if previous_type == 1:
                            mark_1_current -= 1
                        elif previous_type == 3:
                            mark_3_current -= 1
                        mark_1 += mark_1_current
                        mark_3 += mark_3_current

                heterogeneous_area = mark_3
                total = mark_1 + mark_3
                index = 0.0
                if total:
                    index = float(format(heterogeneous_area / total, '.4f'))
                region_ai_result = {
                    'heterogeneous_area': heterogeneous_area,
                    'total': total,
                    'index': index,
                    'whole_slide': whole_slide
                }
                slide_heterogeneous += heterogeneous_area
                slide_total += total  # 在项目与的实际使用中存在全场算法区域和局部算法区域共存的情况，所以全场计算区域和局部区域处理方式相同
            elif ai_type in [AIType.ki67, AIType.ki67hot, AIType.er, AIType.pr]:
                pos_tumor = ai_result_before_modify.get('pos_tumor')
                neg_tumor = ai_result_before_modify.get('neg_tumor')
                normal_cell = ai_result_before_modify.get('normal_cell')

                count_list = [0] * 3
                for mark in marks_in_area:
                    diagnosis_type = mark.diagnosis_type
                    if option == 1:
                        count_list[diagnosis_type] += 1
                    elif option == 2:
                        count_list[diagnosis_type] -= 1
                    elif option == 3:
                        previous_type = mark.pre_diagnosis['type']
                        count_list[previous_type] -= 1
                        count_list[diagnosis_type] += 1

                neg_tumor_new = neg_tumor + count_list[0]  # 阴性肿瘤
                pos_tumor_new = pos_tumor + count_list[1]  # 阳性肿瘤
                normal_cell_new = normal_cell + count_list[2]  # 非肿瘤

                total = neg_tumor_new + pos_tumor_new
                index = 0.0
                if total:
                    index = float(format(pos_tumor_new / total, '.4f'))
                region_ai_result = {
                    "total": total,  # 肿瘤细胞数
                    'pos_tumor': pos_tumor_new,  # 阳性肿瘤
                    'neg_tumor': neg_tumor_new,  # 阴性肿瘤
                    "normal_cell": normal_cell_new,  # 肺肿瘤
                    "index": index,  # 指数
                    "whole_slide": whole_slide,
                }
            elif ai_type == AIType.pdl1:
                pdl1_type_list = ["neg_norm", 'neg_tumor', 'pos_norm', 'pos_tumor']
                center_x, center_y = 0, 0
                if whole_slide:
                    center_coords = ai_result_before_modify.get("center_coords")
                    center_x, center_y = center_coords

                for mark in marks_in_area:
                    diagnosis_type = mark.diagnosis_type
                    cell_type, _ = pdl1_type_list[diagnosis_type], mark.position
                    tile_id_list = mark.cal_pdl1s_count_tiles(tiled_slice)
                    if option == 1:
                        # Update ROIList
                        if whole_slide:
                            region_id = mark.cal_region(center_x, center_y)
                            ai_result_before_modify[region_id][cell_type] += 1
                            ai_result_before_modify[region_id]['total'] += 1
                            ai_result_before_modify[region_id]['tps'] = round(
                                ai_result_before_modify[region_id]['pos_tumor'] / (
                                    ai_result_before_modify[region_id]['pos_tumor'] +
                                    ai_result_before_modify[region_id]['neg_tumor'] + 1e-10), 4)
                        ai_result_before_modify[cell_type] += 1
                        ai_result_before_modify['total'] += 1
                        try:
                            ai_result_before_modify['tps'] = round(
                                ai_result_before_modify['pos_tumor'] / (
                                    ai_result_before_modify['pos_tumor'] +
                                    ai_result_before_modify['neg_tumor'] + 1e-10), 4)
                        except (ValueError, KeyError) as e:
                            logger.warning(e)
                            ai_result_before_modify['tps'] = 0

                        # Update Pdl1sCount
                        for tile_id in tile_id_list:
                            self.repository.update_pdl1_count_in_tile(tile_id, cell_type, 1)

                    elif option == 2:
                        if whole_slide:
                            region_id = mark.cal_region(center_x, center_y)
                            ai_result_before_modify[region_id][cell_type] -= 1
                            ai_result_before_modify[region_id]['total'] -= 1
                            ai_result_before_modify[region_id]['tps'] = round(
                                ai_result_before_modify[region_id]['pos_tumor'] / (
                                    ai_result_before_modify[region_id]['pos_tumor'] +
                                    ai_result_before_modify[region_id]['neg_tumor'] + 1e-10), 4)
                        ai_result_before_modify[cell_type] -= 1
                        ai_result_before_modify['total'] -= 1
                        try:
                            ai_result_before_modify['tps'] = round(
                                ai_result_before_modify['pos_tumor'] / (
                                    ai_result_before_modify['pos_tumor'] +
                                    ai_result_before_modify['neg_tumor'] + 1e-10
                                ), 4)
                        except (ValueError, KeyError) as e:
                            logger.warning(e)
                            ai_result_before_modify['tps'] = 0

                        for tile_id in tile_id_list:
                            self.repository.update_pdl1_count_in_tile(tile_id, cell_type, - 1)

                    elif option == 3:
                        previous_type = pdl1_type_list[mark.pre_diagnosis['type']]

                        if whole_slide:
                            region_id = mark.cal_region(center_x, center_y)
                            ai_result_before_modify[region_id][previous_type] -= 1
                            ai_result_before_modify[region_id][cell_type] += 1
                            ai_result_before_modify[region_id]['tps'] = round(
                                ai_result_before_modify[region_id]['pos_tumor'] / (
                                    ai_result_before_modify[region_id]['pos_tumor'] +
                                    ai_result_before_modify[region_id]['neg_tumor'] + 1e-10), 4)
                        ai_result_before_modify[previous_type] -= 1
                        ai_result_before_modify[cell_type] += 1
                        try:
                            ai_result_before_modify['tps'] = round(
                                ai_result_before_modify['pos_tumor'] / (
                                    ai_result_before_modify['pos_tumor'] +
                                    ai_result_before_modify['neg_tumor'] + 1e-10
                                ), 4)
                        except (ValueError, KeyError) as e:
                            logger.warning(e)
                            ai_result_before_modify['tps'] = 0

                        for tile_id in tile_id_list:
                            self.repository.update_pdl1_count_in_tile(tile_id, previous_type, - 1)
                            self.repository.update_pdl1_count_in_tile(tile_id, cell_type, 1)

                slide_total = ai_result_before_modify['pos_tumor'] + ai_result_before_modify['neg_tumor']
                slide_heterogeneous = ai_result_before_modify['pos_tumor']
                region_ai_result = ai_result_before_modify
            elif ai_type == AIType.fish_tissue:
                count_fields = {
                    0: 'nuclues_num',
                    1: 'red_signal_num',
                    2: 'green_signal_num'
                }

                count_values = {mark_type: ai_result_before_modify.get(field_name, 0)
                                for mark_type, field_name in count_fields.items()}

                for mark in marks_in_area:

                    mark.fix_diagnosis(ai_type=ai_type)

                    diagnosis_type = mark.diagnosis_type
                    if option == 1:
                        count_values[diagnosis_type] += 1
                    elif option == 2:
                        count_values[diagnosis_type] -= 1
                    elif option == 3:
                        previous_type = mark.pre_diagnosis['type']
                        count_values[previous_type] -= 1
                        count_values[diagnosis_type] += 1

                region_ai_result = {count_fields[mark_type]: value for mark_type, value in count_values.items()}
                region_ai_result.update({
                    "whole_slide": 1
                })
            elif ai_type == AIType.her2:
                count_fields = {
                    1: '微弱的不完整膜阳性肿瘤细胞',
                    2: '弱中等的完整膜阳性肿瘤细胞',
                    3: '中强度的不完整膜阳性肿瘤细胞',
                    4: '强度的完整膜阳性肿瘤细胞',
                    7: '阴性肿瘤细胞',
                    9: '组织细胞',
                    10: '淋巴细胞',
                    11: '纤维细胞',
                    12: '其他非肿瘤细胞'
                }

                count_values = {mark_type: ai_result_before_modify.get(field_name, {})
                                for mark_type, field_name in count_fields.items()}

                for mark in marks_in_area:
                    diagnosis_type = mark.diagnosis_type
                    if option == 1:  # 新增标注
                        count_values[diagnosis_type] += 1
                    elif option == 2:  # 删除标注
                        count_values[diagnosis_type] -= 1
                    elif option == 3:  # 修改标注
                        previous_type = mark.pre_diagnosis['type']
                        count_values[previous_type] -= 1
                        count_values[diagnosis_type] += 1

                region_ai_result = {count_fields[mark_type]: value for mark_type, value in count_values.items()}
                region_ai_result.update({
                    "whole_slide": whole_slide
                })
            elif ai_type == AIType.np:

                count_fields = {
                    1: '嗜酸性粒细胞',
                    2: '淋巴细胞',
                    3: '浆细胞',
                    4: '中性粒细胞',
                    5: '上皮区域',
                    6: '腺体区域',
                    7: '血管区域',
                }
                rename_dict = {"嗜酸性粒细胞": "eosinophils", "淋巴细胞": "lymphocyte", "浆细胞": "plasmocyte",
                               "中性粒细胞": "neutrophils"}

                count_values = {mark_type: ai_result_before_modify.get(field_name, {'count': 0, 'index': 0})
                                for mark_type, field_name in count_fields.items()}

                center_x, center_y = 0, 0
                if whole_slide:
                    center_coords = ai_result_before_modify.get("center_coords")
                    center_x, center_y = center_coords

                for mark in marks_in_area:
                    diagnosis_type = mark.diagnosis_type
                    cell_type, _ = count_fields[diagnosis_type], mark.position
                    tile_id_list = mark.cal_pdl1s_count_tiles(tiled_slice)

                    if option == 1:
                        # Update ROIList
                        if whole_slide:
                            region_id = mark.cal_region(center_x, center_y)
                            ai_result_before_modify[region_id][cell_type] += 1

                        for tile_id in tile_id_list:
                            self.repository.update_np_count_in_tile(tile_id, rename_dict[cell_type], 1)

                    elif option == 2:
                        if whole_slide:
                            region_id = mark.cal_region(center_x, center_y)
                            ai_result_before_modify[region_id][cell_type] -= 1

                        for tile_id in tile_id_list:
                            self.repository.update_np_count_in_tile(tile_id, rename_dict[cell_type], - 1)

                    elif option == 3:
                        previous_type = count_fields[mark.pre_diagnosis['type']]

                        if whole_slide:
                            region_id = mark.cal_region(center_x, center_y)
                            ai_result_before_modify[region_id][previous_type] -= 1
                            ai_result_before_modify[region_id][cell_type] += 1

                        for tile_id in tile_id_list:
                            self.repository.update_np_count_in_tile(tile_id, rename_dict[previous_type], - 1)
                            self.repository.update_np_count_in_tile(tile_id, rename_dict[cell_type], 1)

                    if option == 1:  # 新增标注
                        count_values[diagnosis_type]['count'] += 1
                    elif option == 2:  # 删除标注
                        count_values[diagnosis_type]['count'] -= 1
                    elif option == 3:  # 修改标注
                        previous_type = mark.pre_diagnosis['type']
                        count_values[previous_type]['count'] -= 1
                        count_values[diagnosis_type]['count'] += 1

                    if mark.area:
                        if option == 1:  # 新增标注
                            count_values[diagnosis_type]['area'] += mark.area
                        elif option == 2:  # 删除标注
                            count_values[diagnosis_type]['area'] -= mark.area
                        elif option == 3:  # 修改标注
                            previous_type = mark.pre_diagnosis['type']
                            count_values[previous_type]['area'] -= mark.area
                            count_values[diagnosis_type]['area'] += mark.area

                total_cell_count = sum(
                    value.get('count') or 0 for value in count_values.values() if 'total_area' not in value)

                region_ai_result = ai_result_before_modify
                region_ai_result.update(
                    {count_fields[mark_type]: value for mark_type, value in count_values.items()})

                total_area = region_ai_result[count_fields[7]].get('total_area', 0)
                for mark_type, field_name in count_fields.items():
                    if mark_type in (1, 2, 3, 4):
                        region_ai_result[field_name]['index'] = round(
                            region_ai_result[field_name]['count'] / total_cell_count * 100, 2)
                    elif mark_type in (5, 6, 7):
                        region_ai_result[field_name]['index'] = round(
                            region_ai_result[field_name]['area'] / total_area * 100, 2) if total_area else 0

                region_ai_result.update({
                    "whole_slide": whole_slide
                })
            else:
                region_ai_result = {}

            if area_mark:
                area_mark.ai_result = region_ai_result
                self.repository.save_mark(area_mark)

    def get_cell_count_in_quadrant(self, view_path: dict, tiled_slice: TiledSlice, ai_type: AIType) -> dict:
        x_coords = view_path.get('x')
        y_coords = view_path.get('y')
        z = view_path.get('z')
        if z is None:
            z = tiled_slice.max_level
        is_maxlvl = (z == tiled_slice.max_level)

        if ai_type not in (AIType.pdl1, AIType.np) or not (x_coords and y_coords):
            return {'isMaxlvl': is_maxlvl, 'result': {}}

        xmin, xmax, xcent = int(min(x_coords)), int(max(x_coords)), int((min(x_coords) + max(x_coords)) / 2)
        ymin, ymax, ycent = int(min(y_coords)), int(max(y_coords)), int((min(y_coords) + max(y_coords)) / 2)

        # if fullscreen return wsi result
        margin = 100
        full_screen = False  # display regional image result
        if xmin <= margin and ymin <= margin and xmax >= tiled_slice.width - margin and (
                ymax >= tiled_slice.height - margin):
            slide_xcent, slide_ycent = tiled_slice.width // 2, tiled_slice.height // 2
            if abs(slide_xcent - xcent) < margin and abs(slide_ycent - ycent) < margin:
                full_screen = True  # display whole image result
        is_whole_slide = False

        _, roi_marks = self.repository.get_marks(mark_type=3)
        roi_res_list = []
        for roi in roi_marks:
            ai_result = roi.ai_result
            if ai_result.get('whole_slide') == 1:
                is_whole_slide = True
            roi_res_list.append(ai_result)

        if is_maxlvl or (not is_whole_slide):
            cell_count = CellCount(tile_id=-1, ai_type=ai_type)
            if full_screen:
                for result in roi_res_list:
                    c = CellCount(tile_id=-1, ai_type=ai_type, cell_count_dict=result)
                    cell_count += c
            else:
                tile_ids = tiled_slice.cal_tiles_in_quadrant(
                    x_coords=[xmin, xmax], y_coords=[ymin, ymax], level=z)
                count_entities = self.repository.get_cell_count(ai_type=ai_type, tile_ids=tile_ids)
                for entity in count_entities:
                    cell_count_dict = entity.to_dict()
                    cell_count += CellCount(tile_id=entity.tile_id, ai_type=ai_type, cell_count_dict=cell_count_dict)
            res_dict = cell_count.to_dict()
        else:
            # display four regions
            c1, c2, c3, c4 = tuple(CellCount(tile_id=-1, ai_type=ai_type) for _ in range(4))
            if full_screen:  # whole image count
                whole_slide_res = roi_res_list[0]
                c1.cell_count_dict = whole_slide_res.get('1')
                c2.cell_count_dict = whole_slide_res.get('2')
                c3.cell_count_dict = whole_slide_res.get('3')
                c4.cell_count_dict = whole_slide_res.get('4')
            else:  # regional image count
                r1 = tiled_slice.cal_tiles_in_quadrant(x_coords=[xmin, xcent], y_coords=[ymin, ycent], level=z)
                r2 = tiled_slice.cal_tiles_in_quadrant(x_coords=[xcent, xmax], y_coords=[ymin, ycent], level=z)
                r3 = tiled_slice.cal_tiles_in_quadrant(x_coords=[xmin, xcent], y_coords=[ycent, ymax], level=z)
                r4 = tiled_slice.cal_tiles_in_quadrant(x_coords=[xcent, xmax], y_coords=[ycent, ymax], level=z)

                tile_ids = list(set(r1 + r2 + r3 + r4))
                count_entities = self.repository.get_cell_count(ai_type=ai_type, tile_ids=tile_ids)
                for entity in count_entities:
                    cell_count_dict = entity.to_dict()
                    cell_count = CellCount(tile_id=entity.tile_id, ai_type=ai_type, cell_count_dict=cell_count_dict)
                    for c, r in zip([c1, c2, c3, c4], [r1, r2, r3, r4]):
                        if entity.tile_id in r:
                            c += cell_count
                            break
            res_dict = {
                '1': c1.to_dict(),
                '2': c2.to_dict(),
                '3': c3.to_dict(),
                '4': c4.to_dict()
            }
        return {'isMaxlvl': is_maxlvl, 'result': res_dict}

    @transaction
    def clear_ai_result(self, ai_type: AIType, exclude_area_marks: Optional[List[int]] = None) -> bool:
        self.repository.clear_mark_table(
            ai_type=ai_type, exclude_area_marks=exclude_area_marks)

        if exclude_area_marks:
            _, marks = self.repository.get_marks(mark_type=[3, 4])
            for mark in marks:
                mark.update_data(ai_result=AIResult.initialize(ai_type=ai_type).to_dict())
                self.repository.save_mark(mark)
        return True

    def delete_dna_roi(self, mark_id_list: list, deleted: int) -> Tuple[bool, int, int, int]:
        _, marks = self.repository.get_marks()
        ai_result = marks[0].ai_result if marks else None
        nuclei = ai_result.get('nuclei') if ai_result else []

        total = 0
        normal = 0
        abnormal_low = 0
        abnormal_high = 0
        deleted_count = 0

        for nucleus in nuclei:
            if nucleus["id"] in mark_id_list:
                nucleus["is_deleted"] = deleted

        for nucleus in nuclei:
            if nucleus["is_deleted"] == 1:
                deleted_count += 1
            else:
                total += 1
                if nucleus["lesion_type"] == "normal":
                    normal += 1
                if nucleus["lesion_type"] == "abnormal_low":
                    abnormal_low += 1
                if nucleus["lesion_type"] == "abnormal_high":
                    abnormal_high += 1

        ai_result["nuclei"] = nuclei
        for mark in marks:
            mark.update_data(ai_result=json.dumps(ai_result))
            self.repository.save_mark(mark)

        return True, abnormal_high, abnormal_low, total

    def get_statistics(self) -> dict:
        total = 0
        normal = 0
        abnormal_low = 0
        abnormal_high = 0
        deleted = 0

        _, marks = self.repository.get_marks()

        if not marks:
            data = {
                "total": total,
                "normal": normal,
                "abnormal_low": abnormal_low,
                "abnormal_high": abnormal_high,
                "deleted": deleted
            }
            return data

        ai_result = marks[0].ai_result if marks else None
        nuclei = ai_result.get('nuclei') if ai_result else []

        for nucleus in nuclei:
            if nucleus["is_deleted"] == 1:
                deleted += 1
            else:
                total += 1
                if nucleus["lesion_type"] == "normal":
                    normal += 1
                if nucleus["lesion_type"] == "abnormal_low":
                    abnormal_low += 1
                if nucleus["lesion_type"] == "abnormal_high":
                    abnormal_high += 1
        data = {
            "total": total,
            "normal": normal,
            "abnormal_low": abnormal_low,
            "abnormal_high": abnormal_high,
            "deleted": deleted
        }
        return data

    def get_feat(self) -> dict:
        return {}

    def get_histplot(self) -> dict:
        _, marks = self.repository.get_marks()

        if not marks:
            data = {
                "x": [],
                "y": []
            }
            return data

        ai_result = marks[0].ai_result if marks else None
        nuclei = ai_result.get('nuclei') if ai_result else []

        dna_index = [i["dna_index"] for i in nuclei]
        np_dna_index = np.array(dna_index)
        step = 0.05
        temp = (np_dna_index // step) * step
        x, count = np.unique(temp, return_counts=True)
        x = np.round(x, decimals=2)

        data = {
            "x": np.append(x, x[-1] + step).tolist(),
            "y": count.tolist()
        }
        return data

    def get_scatterplot(self) -> dict:
        _, marks = self.repository.get_marks()

        if not marks:
            data = {"x": [], "y": []}
            return data

        ai_result = marks[0].ai_result if marks else None
        nuclei = ai_result.get('nuclei') if ai_result else []

        dna_index_area = []
        for i in nuclei:
            if i["is_deleted"] != 0:
                continue
            if i["lesion_type"] != "normal" or i["id"] % 10 == 1:
                dna_index_area.append({"dna_index": i["dna_index"], "area": i["area"]})

        data = {
            "x": [i["dna_index"] for i in dna_index_area],
            "y": [i["area"] for i in dna_index_area],
        }
        return data

    def get_dna_statistics(self) -> dict:
        _, marks = self.repository.get_marks()
        if not marks:
            data = {'header': [], 'data': []}
            return data

        ai_result = marks[0].ai_result if marks else None

        total, total_di = 0, 0
        normal, normal_di = 0, 0
        abnormal_low, abnormal_low_di = 0, 0
        abnormal_high, abnormal_high_di = 0, 0
        deleted = 0

        nuclei = ai_result.get("nuclei", [])
        for nucleus in nuclei:
            if nucleus["is_deleted"] == 1:
                deleted += 1
            else:
                total += 1
                total_di += nucleus["dna_index"]
                if nucleus["lesion_type"] == "normal":
                    normal += 1
                    normal_di += nucleus["dna_index"]
                if nucleus["lesion_type"] == "abnormal_low":
                    abnormal_low += 1
                    abnormal_low_di += nucleus["dna_index"]
                if nucleus["lesion_type"] == "abnormal_high":
                    abnormal_high += 1
                    abnormal_high_di += nucleus["dna_index"]

        dna_statistics = {
            'header': ['', '数量', 'percent', 'DI均值'],
            'data': [
                [{'color': '#8fc53a', 'val': '正常二倍体细胞'}, {'color': '', 'val': normal},
                 {'color': '', 'val': '{:.2%}'.format(round((normal / total), 4)) if total != 0 else '0'},
                 {'color': '', 'val': round(total_di / total, 2) if total != 0 else 0}],
                [{'color': '#f5a337', 'val': '疑似病变细胞'}, {'color': '', 'val': abnormal_low},
                 {'color': '', 'val': '{:.2%}'.format(round((abnormal_low / total), 4)) if total != 0 else '0'},
                 {'color': '', 'val': round(abnormal_low_di / abnormal_low, 2) if abnormal_low != 0 else 0}],
                [{'color': '#e24055', 'val': '病变细胞'}, {'color': '', 'val': abnormal_high},
                 {'color': '', 'val': '{:.2%}'.format(round((abnormal_high / total), 4)) if total != 0 else '0'},
                 {'color': '', 'val': round(abnormal_high_di / abnormal_high, 2) if abnormal_high != 0 else 0}],
                [{'color': '#444444', 'val': '全部细胞'}, {'color': '', 'val': total},
                 {'color': '', 'val': '{:.0%}'.format(1) if total != 0 else '0'},
                 {'color': '', 'val': round(total_di / total, 2)} if total != 0 else 0]
            ]
        }
        return dna_statistics

    def get_np_info(self) -> Tuple[list, list]:
        _, rois = self.repository.get_marks()

        if not rois:
            return [], []

        _cell_statics = {
            '嗜酸性粒细胞': {'count': 0, 'area': 0, 'index': 0, 'color': '#f00'},
            '淋巴细胞': {'count': 0, 'area': 0, 'index': 0, 'color': '#0ef'},
            '浆细胞': {'count': 0, 'area': 0, 'index': 0, 'color': '#eafd00'},
            '中性粒细胞': {'count': 0, 'area': 0, 'index': 0, 'color': '#66fd00'},
            '总体炎症细胞': {'count': 0, 'area': 0, 'index': 0, 'color': ''},
        }
        _area_statics = {
            '上皮区域': {'count': 0, 'area': 0, 'index': 0, 'color': '#00febb'},
            '腺体区域': {'count': 0, 'area': 0, 'index': 0, 'color': '#ab86ff'},
            '血管区域': {'count': 0, 'area': 0, 'index': 0, 'color': '#d32cd6'},
            '总体区域': {'count': 0, 'area': 0, 'index': 0, 'color': ''},
        }

        export_count = 0
        total_area = 0
        for roi in rois:
            if roi.isExport and roi.aiResult:
                export_count += 1
                ai_result = json.loads(roi.aiResult)
                for cell_name in list(_cell_statics.keys())[:-1]:
                    _cell_statics[cell_name]['count'] += ai_result.get(cell_name, {}).get('count', 0)
                    _cell_statics['总体炎症细胞']['count'] += ai_result.get(cell_name, {}).get('count', 0)
                for area_name in list(_area_statics.keys())[:-1]:
                    _area_statics[area_name]['area'] += ai_result.get(area_name, {}).get('area', 0)
                    _area_statics['总体区域']['area'] += ai_result.get(area_name, {}).get('area', 0)
                    total_area = ai_result.get(area_name, {}).get('total_area', 0)

        for cell_name in _cell_statics.keys():
            cell_count = _cell_statics[cell_name]['count']
            _cell_statics[cell_name]['index'] = format(
                round(cell_count * 100 / _cell_statics['总体炎症细胞']['count'], 2) if cell_count else 0, '0.2f')
            _cell_statics[cell_name]['average'] = format(
                round(cell_count / export_count, 2) if export_count else 0, '0.2f')

        for area_name in _area_statics.keys():
            area = _area_statics[area_name]['area']
            _area_statics[area_name]['area'] = format(round(area / 1000000, 2), '0.2f')
            _area_statics[area_name]['index'] = format(round(area * 100 / total_area, 2) if area else 0, '0.2f')
            _area_statics[area_name]['average'] = format(
                round((area / 1000000) / export_count, 2) if export_count else 0, '0.2f')
        _area_statics['总体区域']['index'] = '-'

        cell_statics = [{'name': cell_name, 'stats': item} for cell_name, item in _cell_statics.items()]
        area_statics = [{'name': area_name, 'stats': item} for area_name, item in _area_statics.items()]

        return cell_statics, area_statics
