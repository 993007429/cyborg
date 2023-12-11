import logging
import json
import sys
import time
from copy import deepcopy
from contextvars import ContextVar
from typing import Optional, List, Tuple, Union

from sqlalchemy import distinct, or_
from sqlalchemy.orm import Session

from cyborg.infra.session import transaction
from cyborg.modules.slice_analysis.domain.entities import MarkEntity, MarkGroupEntity, MarkToTileEntity, \
    ChangeRecordEntity, NPCountEntity, Pdl1sCountEntity, TemplateEntity
from cyborg.modules.slice_analysis.domain.repositories import SliceMarkRepository, AIConfigRepository
from cyborg.modules.slice_analysis.domain.value_objects import AIType
from cyborg.modules.slice_analysis.infrastructure.models import MarkGroupModel, Pdl1sCountModel, \
    NPCountModel, AIModel, TemplateModel, ChangeRecordModel, get_ai_mark_model, \
    get_ai_mark_to_tile_model, ShareMarkGroupModel
from cyborg.seedwork.infrastructure.models import BaseModel
from cyborg.seedwork.infrastructure.repositories import SQLAlchemyRepository
from cyborg.utils.id_worker import IdWorker

logger = logging.getLogger(__name__)


class SQLAlchemySliceMarkRepository(SliceMarkRepository, SQLAlchemyRepository):

    def __init__(self, *, template_session: ContextVar, manual: Optional[SliceMarkRepository] = None, **kwargs):
        super().__init__(**kwargs)
        self._mark_table_suffix = None
        self._manual = manual
        self._template_session_cv = template_session
        self.mark_id_worker = IdWorker(1, 2, 0)

    @property
    def template_session(self) -> Session:
        s = self._template_session_cv.get()
        assert s is not None
        return s

    @property
    def mark_table_suffix(self):
        return self._mark_table_suffix

    @mark_table_suffix.setter
    def mark_table_suffix(self, value):
        self._mark_table_suffix = value

    @property
    def manual(self) -> 'SliceMarkRepository':
        return self._manual

    @property
    def mark_model_class(self):
        return get_ai_mark_model(self.mark_table_suffix)

    @property
    def mark_to_tile_model_class(self):
        return get_ai_mark_to_tile_model(self.mark_table_suffix)

    @transaction
    def create_mark_tables(self, ai_type: AIType):
        if not self.mark_table_suffix:
            return

        engine = self.session.get_bind()
        tables = [
            get_ai_mark_model(self.mark_table_suffix).__table__,
            get_ai_mark_to_tile_model(self.mark_table_suffix).__table__,
        ]

        if ai_type == AIType.np:
            tables.append(NPCountModel.__table__)
        elif ai_type == AIType.pdl1:
            tables.append(Pdl1sCountModel.__table__)

        tables.append(MarkGroupModel.__table__)
        tables.append(ChangeRecordModel.__table__)

        BaseModel.metadata.create_all(engine, tables=tables)

    @transaction
    def backup_ai_mark_tables(self) -> bool:
        for model_class in [self.mark_model_class, self.mark_to_tile_model_class]:
            table_name = model_class.__tablename__
            import_table_name = model_class.__import_table_name__
            self.session.execute(f'drop table if exists {import_table_name}')
            self.session.execute(f'create table `{import_table_name}` as select * from `{table_name}`')
        return True

    @transaction
    def create_mark_table_by_import(self) -> Optional[str]:
        rows = self.session.execute(f'select id from `{self.mark_model_class.__import_table_name__}` limit 1')
        self.session.commit()
        if not rows:
            return '未找到算法结果'
        for model_class in [self.mark_model_class, self.mark_to_tile_model_class]:
            table_name = model_class.__tablename__
            import_table_name = model_class.__import_table_name__
            self.session.execute(f'drop table if exists {table_name}')
            self.session.execute(f'create table `{table_name}` as select * from `{import_table_name}`')
        return None

    @transaction
    def clear_mark_table(
            self, ai_type: AIType, exclude_area_marks: Optional[List[int]] = None):
        mark_query = self.session.query(self.mark_model_class)
        if exclude_area_marks is not None:
            mark_query = mark_query.filter(self.mark_model_class.id.not_in(exclude_area_marks))
        mark_query.delete(synchronize_session=False)
        self.session.query(self.mark_to_tile_model_class).delete(synchronize_session=False)
        if ai_type == AIType.pdl1:
            self.session.query(Pdl1sCountModel).delete(synchronize_session=False)
        elif ai_type == AIType.np:
            self.session.query(NPCountModel).delete(synchronize_session=False)

    @transaction
    def save_mark(self, entity: MarkEntity) -> bool:
        model = self.convert_to_model(entity, self.mark_model_class, id_worker=self.mark_id_worker)
        if not model:
            return False
        self.session.add(model)
        self.session.flush([model])
        entity.update_data(**model.raw_data)
        return True

    @transaction
    def batch_save_marks(self, entities: List[MarkEntity]) -> bool:
        self.session.bulk_insert_mappings(self.mark_model_class, [entity.raw_data for entity in entities])
        return True

    def get_marks(
            self,
            group_id: Optional[int] = None,
            mark_ids: List[int] = None,
            mark_type: Union[List[int], int, None] = None,
            tile_ids: Optional[List[int]] = None,
            is_export: Optional[int] = None,
            page: int = 0,
            per_page: int = sys.maxsize,
            need_total: bool = False
    ) -> Tuple[int, List[MarkEntity]]:
        query = self.session.query(self.mark_model_class)
        if group_id is not None:
            query = query.filter(self.mark_model_class.group_id == group_id)
        if mark_ids is not None:
            query = query.filter(self.mark_model_class.id.in_(mark_ids))
        if is_export is not None:
            query = query.filter(self.mark_model_class.is_export == is_export)
        if mark_type is not None:
            if isinstance(mark_type, list):
                query = query.filter(self.mark_model_class.mark_type.in_(mark_type))
            else:
                query = query.filter(self.mark_model_class.mark_type == mark_type)
        if tile_ids is not None:
            query = query.join(
                self.mark_to_tile_model_class,
                self.mark_model_class.id == self.mark_to_tile_model_class.mark_id
            ).filter(self.mark_to_tile_model_class.tile_id.in_(tile_ids))

        total = query.count() if need_total else 0

        models = query.offset(per_page * page).limit(per_page).all()
        return total, [MarkEntity.from_dict(model.raw_data) for model in models]

    def get_marks_by_diagnosis_result(self, diagnosis_result: str, ai_type: AIType) -> List[MarkEntity]:
        models = []
        if ai_type in [AIType.tct, AIType.lct, AIType.human_tl]:
            if diagnosis_result == '阴性':
                models = self.session.query(self.mark_model_class).filter(
                    or_(self.mark_model_class.doctor_diagnosis.like('["单个细胞%'),
                        self.mark_model_class.doctor_diagnosis.like('["细胞团%'),
                        self.mark_model_class.doctor_diagnosis.contains(diagnosis_result))).all()
            elif diagnosis_result == '异物':
                models = self.session.query(self.mark_model_class).filter(
                    or_(self.mark_model_class.doctor_diagnosis.like('["絮状杂质%'),
                        self.mark_model_class.doctor_diagnosis.like('["黏液丝%'),
                        self.mark_model_class.doctor_diagnosis.like('["细胞折痕%'),
                        self.mark_model_class.doctor_diagnosis.like('["无%'),
                        self.mark_model_class.doctor_diagnosis.contains(diagnosis_result))).all()
            else:
                models = self.session.query(self.mark_model_class).filter(
                    self.mark_model_class.doctor_diagnosis.contains(diagnosis_result)).all()
        elif ai_type in [AIType.bm, AIType.human_bm]:
            models = self.session.query(self.mark_model_class).filter(
                self.mark_model_class.doctor_diagnosis == json.dumps(diagnosis_result, ensure_ascii=False)).all()
        return [MarkEntity.from_dict(model.raw_data) for model in models]

    def get_marks_by_area_id(self, area_id: int) -> List[MarkEntity]:
        models = self.session.query(self.mark_model_class).filter_by(area_id=area_id).all()
        return [MarkEntity.from_dict(model.raw_data) for model in models]

    @transaction
    def delete_marks(
            self, mark_ids: Optional[List[int]] = None, group_id: Optional[int] = None, area_id: Optional[int] = None
    ) -> bool:
        query = self.session.query(self.mark_model_class)
        if mark_ids is not None:
            query = query.filter(self.mark_model_class.id.in_(mark_ids))
        if group_id is not None:
            query = query.filter_by(group_id=group_id)
        if area_id is not None:
            query = query.filter_by(area_id=area_id)

        deleted_mark_ids = [mark.id for mark in query.all()]
        if deleted_mark_ids:
            self.session.query(self.mark_to_tile_model_class).filter(
                self.mark_to_tile_model_class.mark_id.in_(deleted_mark_ids)).delete(synchronize_session=False)
        query.delete(synchronize_session=False)
        return True

    @transaction
    def save_mark_to_tiles(self, entities: List[MarkToTileEntity]) -> bool:
        model_class = self.mark_to_tile_model_class
        models = [self.convert_to_model(entity, model_class) for entity in entities]
        filtered_models = list(filter(None, models))
        self.session.bulk_save_objects(filtered_models)
        self.session.flush(filtered_models)
        for idx, entity in enumerate(entities):
            entity.update_data(**models[idx].raw_data)
        return True

    @transaction
    def save_change_record(self, entity: ChangeRecordEntity) -> bool:
        model = self.convert_to_model(entity, ChangeRecordModel)
        if not model:
            return False
        self.session.add(model)
        self.session.flush([model])
        entity.update_data(**model.raw_data)
        return True

    def get_mark(self, mark_id: int) -> Optional[MarkEntity]:
        model = self.session.query(self.mark_model_class).get(mark_id)
        return MarkEntity.from_dict(model.raw_data) if model else None

    @transaction
    def delete_mark_by_type(self, mark_type: int) -> bool:
        self.session.query(self.mark_model_class).filter_by(mark_type=mark_type).delete(synchronize_session=False)
        return True

    @transaction
    def delete_mark_by_id(self, mark_id: int) -> bool:
        self.session.query(self.mark_model_class).filter_by(id=mark_id).delete(synchronize_session=False)
        self.session.query(self.mark_to_tile_model_class).filter_by(mark_id=mark_id).delete(synchronize_session=False)
        return True

    @transaction
    def delete_mark_to_tiles_by_mark_id(self, mark_id: int) -> bool:
        self.session.query(self.mark_to_tile_model_class).filter_by(mark_id=mark_id).delete(synchronize_session=False)
        return True

    def get_mark_count_by_tile_id(self, tile_id: int) -> int:
        return self.session.query(distinct(self.mark_to_tile_model_class.id)).filter_by(tile_id=tile_id).count()

    def get_mark_count(self, group_id: Optional[int] = None) -> int:
        query = self.session.query(self.mark_model_class)
        if group_id is not None:
            query = query.filter_by(group_id=group_id)
        return query.count()

    def has_mark(self, group_id: Optional[int] = None) -> bool:
        query = self.session.query(self.mark_model_class)
        if group_id is not None:
            query = query.filter_by(group_id=group_id)
        return query.exists()

    @transaction
    def save_mark_group(self, entity: MarkGroupEntity) -> bool:
        model = self.convert_to_model(entity, MarkGroupModel)
        if not model:
            return False
        self.session.add(model)
        self.session.flush([model])
        entity.update_data(**model.raw_data)
        return True

    @transaction
    def delete_mark_group(self, group_id: int) -> bool:
        self.session.query(MarkGroupModel).filter_by(id=group_id).delete()
        return True

    def get_mark_group_by_id(self, group_id: int) -> Optional[MarkGroupEntity]:
        model = self.session.get(MarkGroupModel, group_id)
        return MarkGroupEntity.from_dict(model.raw_data) if model else None

    @transaction
    def update_mark_group_selected(self, group_id: int) -> bool:
        self.session.query(MarkGroupModel).filter_by(is_selected=1).update({'is_selected': 0})
        self.session.query(MarkGroupModel).filter_by(id=group_id).update({'is_selected': 1})
        return True

    def get_selected_mark_group(self) -> Optional[MarkGroupEntity]:
        model = self.session.query(MarkGroupModel).filter_by(is_selected=1).first()
        return MarkGroupEntity.from_dict(model.raw_data) if model else None

    def get_mark_groups_by_template_id(
            self, template_id: int, primary_only: bool = False, is_import: Optional[int] = None,
            is_ai: Optional[int] = None
    ) -> List[MarkGroupEntity]:
        query = self.session.query(MarkGroupModel).filter(
            MarkGroupModel.template_id == template_id,
        )
        if primary_only:
            query = query.filter(MarkGroupModel.parent_id.is_(None))
        if is_import is not None:
            query = query.filter(MarkGroupModel.is_import == is_import)
        if is_ai:
            query = query.filter(MarkGroupModel.is_ai == is_ai)
        else:
            query = query.filter(MarkGroupModel.is_ai.isnot(1))
        models = query.all()
        return [MarkGroupEntity.from_dict(model.raw_data) for model in models]

    def get_default_mark_groups(self, template_id: Optional[int] = None) -> List[MarkGroupEntity]:
        if not self.template_session:
            raise RuntimeError('need connect template db file')
        query = self.template_session.query(MarkGroupModel)
        if template_id is not None:
            query = query.filter_by(template_id=template_id)
        models = query.all()
        return [MarkGroupEntity.from_dict(model.raw_data) for model in models]

    @transaction
    def delete_mark_groups_by_template_id(self, template_id: int) -> bool:
        self.session.query(MarkGroupModel).filter_by(template_id=template_id).delete()
        return True

    def get_mark_groups_by_parent_id(self, parent_id: int) -> List[MarkGroupEntity]:
        models = self.session.query(MarkGroupModel).filter(MarkGroupModel.parent_id == parent_id).all()
        return [MarkGroupEntity.from_dict(model.raw_data) for model in models]

    def get_visible_mark_group_ids(self) -> List[int]:
        rows = self.session.query(MarkGroupModel.id).filter_by(is_show=1).all()
        return [row[0] for row in rows]

    def get_mark_groups(self, ) -> List[MarkGroupEntity]:
        models = self.session.query(MarkGroupModel).filter_by(is_show=1).all()
        return [MarkGroupEntity.from_dict(model.raw_data) for model in models]

    @transaction
    def update_mark_group_status(self, group_id: int, is_empty: int) -> bool:
        self.session.query(MarkGroupModel).filter_by(id=group_id).update({'is_empty': is_empty})
        return True

    @transaction
    def update_pdl1_count_in_tile(self, tile_id: int, field_name: str, count_delta: int) -> bool:
        model = self.session.query(Pdl1sCountModel).filter_by(tile_id=tile_id).first()
        if not model:
            model = Pdl1sCountModel(tile_id=tile_id, pos_tumor=0, neg_tumor=0, pos_norm=0, neg_norm=0)
        setattr(model, field_name, max(getattr(model, field_name) + count_delta, 0))
        self.session.add(model)
        return True

    @transaction
    def update_np_count_in_tile(self, tile_id: int, field_name: str, count_delta: int) -> bool:
        model = self.session.query(NPCountModel).filter_by(tile_id=tile_id).first()
        if not model:
            model = NPCountModel(tile_id=tile_id, eosinophils=0, lymphocyte=0, plasmocyte=0, neutrophils=0)
        setattr(model, field_name, max(getattr(model, field_name) + count_delta, 0))
        self.session.add(model)
        return True

    @transaction
    def delete_count(self, tile_ids: List[int], ai_type: AIType) -> bool:
        if ai_type == AIType.pdl1:
            self.session.query(Pdl1sCountModel).filter(Pdl1sCountModel.tile_id.in_(tile_ids)).delete(
                synchronize_session=False)
        elif ai_type == AIType.np:
            self.session.query(Pdl1sCountModel).filter(NPCountModel.tile_id.in_(tile_ids)).delete(
                synchronize_session=False)
        return True

    def get_cell_count(self, ai_type: AIType, tile_ids: List[int]) -> List[Union[Pdl1sCountEntity, NPCountEntity]]:
        if ai_type == AIType.np:
            count_models = self.session.query(NPCountModel).filter(NPCountModel.tile_id.in_(tile_ids)).all()
        else:
            count_models = self.session.query(Pdl1sCountModel).filter(Pdl1sCountModel.tile_id.in_(tile_ids)).all()

        entities = []
        for model in count_models:
            entity_class = NPCountEntity if ai_type == AIType.np else Pdl1sCountEntity
            entity = entity_class.from_dict(model.raw_data)
            entities.append(entity)
        return entities

    def get_mark_group_by_kwargs(self, kwargs: Optional[dict]) -> List[MarkGroupEntity]:
        query = self.session.query(MarkGroupModel)
        if 'id' in kwargs:
            query = query.filter(MarkGroupModel.id == kwargs['id'])
        if 'group_name' in kwargs:
            query = query.filter(MarkGroupModel.group_name == kwargs['group_name'])
        if 'template_id' in kwargs:
            query = query.filter(MarkGroupModel.template_id == kwargs['template_id'])
        models = query.all()
        return [MarkGroupEntity.from_dict(model.raw_data) for model in models]

    @transaction
    def update_mark_group_by_kwargs(self, group_id: int, kwargs: dict) -> bool:
        logger.info('kwargs===%s' % kwargs)
        self.session.query(MarkGroupModel).filter_by(id=group_id).update(kwargs)
        return True


class SQLAlchemyAIConfigRepository(AIConfigRepository, SQLAlchemyRepository):

    def get_ai_id_by_type(self, ai_type: AIType) -> Optional[int]:
        row = self.session.query(AIModel.id).filter_by(ai_name=ai_type.value).first()
        return row[0] if row else None

    def get_ai_name_by_template_id(self, template_id: int) -> Optional[str]:
        row = self.session.query(AIModel.ai_name).join(TemplateModel, AIModel.id == TemplateModel.ai_id).filter(
            TemplateModel.id == template_id).first()
        return row[0] if row else None

    def get_all_templates(self) -> List[dict]:
        rows = self.session.query(TemplateModel.id, TemplateModel.template_name, AIModel.ai_name,
                                  TemplateModel.is_multi_mark).outerjoin(
            AIModel, TemplateModel.ai_id == AIModel.id).all()
        return [{'id': row[0], 'name': row[1], 'type': row[2], 'isMultiMark': row[3]} for row in rows]

    def get_templates(self, template_id: int) -> dict:
        # todo 待优化算法
        logger.info('template_id==%s' % template_id)
        row = self.session.query(TemplateModel.id, TemplateModel.template_name, AIModel.ai_name,
                                 TemplateModel.is_multi_mark).outerjoin(
            AIModel, TemplateModel.ai_id == AIModel.id).filter(TemplateModel.id == template_id).first()
        logger.info('row==%s' % row)
        mark_groups = self.session.query(ShareMarkGroupModel.id, ShareMarkGroupModel.parent_id,
                                         ShareMarkGroupModel.group_name,
                                         ShareMarkGroupModel.color).join(
            TemplateModel, TemplateModel.id == ShareMarkGroupModel.template_id).filter(
            TemplateModel.id == template_id).all()
        mark_group, group_dict = [], {}
        for group in mark_groups:
            mark_group.append(
                {"id": group[0], "parentId": group[1], "groupName": group[2], "color": group[3], 'children': []})
            if not group[1] and group[0] not in group_dict:
                # 一级标注组
                group_dict[group[0]] = {"id": group[0], "parentId": group[1], "groupName": group[2], "color": group[3],
                                        'children': []}
        second_level = []
        for group in mark_group:
            for k, v in group_dict.items():
                # 二级标注组
                if group['parentId'] == k:
                    second_level.append(group['id'])
                    group_dict[k]['children'].append(
                        {"id": group['id'], "parentId": group['parentId'], "groupName": group['groupName'],
                         "color": group['color'], 'children': []})
        logger.info('second_level==%s' % second_level)
        logger.info('mark_group==%s' % mark_group)
        logger.info('group_dict==%s' % group_dict)
        for group in mark_group:
            if not group['parentId']:
                continue
            for k in second_level:
                # 三级标注组
                first_level = None
                if group['parentId'] == k:
                    for group_ in mark_group:
                        if group_['id'] == k:
                            first_level = group_['parentId']
                            break
                    logger.info('first_level==%s' % first_level)
                    if first_level:
                        second = group_dict[first_level]['children']
                        logger.info('second==%s' % second)
                        for i in second:
                            index = second.index(i)
                            logger.info('index==%s' % index)
                            logger.info('group==%s' % group)
                            if i['id'] == k:
                                item = {"id": group['id'], "parentId": group['parentId'],
                                        "groupName": group['groupName'],
                                        "color": group['color'], 'children': []}
                                group_dict[first_level]['children'][index]['children'].append(item)
        logger.info('group_dict==%s' % group_dict)
        logger.info('row==%s' % row)
        return {'id': row[0], 'name': row[1], 'type': row[2], 'aiName': row[2], 'isMultiMark': row[3],
                'markGroups': list(group_dict.values())}

    @transaction
    def add_templates(self, mark_group: List[dict], template: TemplateEntity) -> Tuple[bool, int]:
        current_time = time.time()
        is_existed = self.session.query(TemplateModel).filter(
            TemplateModel.template_name == template.template_name).first()
        if is_existed:
            logger.info('the template has existed.')
            return False, -1
        template = self.convert_to_model(template, TemplateModel)
        if not template:
            return False, -1
        self.session.add(template)
        self.session.flush([template])
        mark = MarkGroupEntity(raw_data=dict(
            group_name='',
            shape='',
            color='',
            create_time=0,
            is_template=0,
            is_selected=0,
            selectable=0,
            editable=0,
            is_ai=0,
            parent_id=None,
            template_id=template.id,
            op_time=current_time,
            default_color='',
            is_empty=1,
            is_show=1,
            is_import=0
        ))
        for item in mark_group:
            if not item['parentId']:
                logger.info('item===%s' % item)
                first_level_mark = deepcopy(mark)
                first_level_mark.raw_data['group_name'] = item['groupName']
                first_level_mark.raw_data['color'] = item['color']
                first_level_mark.raw_data['default_color'] = item['color']
                model = self.convert_to_model(first_level_mark, ShareMarkGroupModel)
                if not model:
                    return False, -1
                logger.info('add first level mark begin. %s' % first_level_mark)
                self.session.add(model)
                self.session.flush([model])
                first_level_mark.update_data(**model.raw_data)
                first_parent_id = model.id
                logger.info('add first level mark success. %s' % first_level_mark)
                for second_mark in item['children'] if 'children' in item else []:
                    logger.info('second_mark===%s' % second_mark)
                    second_level_mark = deepcopy(mark)
                    second_level_mark.raw_data['group_name'] = second_mark['groupName']
                    second_level_mark.raw_data['color'] = second_mark['color']
                    second_level_mark.raw_data['default_color'] = second_mark['color']
                    second_level_mark.raw_data['parent_id'] = first_parent_id
                    model = self.convert_to_model(second_level_mark, ShareMarkGroupModel)
                    if not model:
                        return False, -1
                    logger.info('add second level mark begin. %s' % second_level_mark)
                    self.session.add(model)
                    self.session.flush([model])
                    second_level_mark.update_data(**model.raw_data)
                    second_parent_id = model.id
                    logger.info('add second level mark success. %s' % second_level_mark)
                    for third_mark in second_mark['children'] if 'children' in item else []:
                        logger.info('third_mark===%s' % third_mark)
                        third_level_mark = deepcopy(mark)
                        third_level_mark.raw_data['group_name'] = third_mark['groupName']
                        third_level_mark.raw_data['color'] = third_mark['color']
                        third_level_mark.raw_data['default_color'] = third_mark['color']
                        third_level_mark.raw_data['parent_id'] = second_parent_id
                        model = self.convert_to_model(third_level_mark, ShareMarkGroupModel)
                        if not model:
                            return False, -1
                        self.session.add(model)
                        self.session.flush([model])
                        third_level_mark.update_data(**model.raw_data)
                        logger.info('add third level mark success. %s' % third_level_mark)
            logger.info('update first level end.%s' % item)
        return True, template.id

    @transaction
    def edit_templates(self, mark_group: List[dict], template: TemplateEntity, template_id: int) -> bool:
        current_time = time.time()
        template = self.convert_to_model(template, TemplateModel)
        if not template:
            return False
        self.session.add(template)
        self.session.flush([template])
        self.session.query(ShareMarkGroupModel).filter(ShareMarkGroupModel.template_id == template_id).delete()
        mark = MarkGroupEntity(raw_data=dict(
            group_name='',
            shape='',
            color='',
            create_time=0,
            is_template=0,
            is_selected=0,
            selectable=0,
            editable=0,
            is_ai=0,
            parent_id=None,
            template_id=template_id,
            op_time=current_time,
            default_color='',
            is_empty=1,
            is_show=1,
            is_import=0
        ))
        for item in mark_group:
            logger.info(item)
            if not item['parentId']:
                first_level_mark = deepcopy(mark)
                id = item.get('id', '')
                first_parent_id = id
                is_add = True
                logger.info('id===%s' % id)
                logger.info('is_add===%s' % is_add)
                if is_add:
                    first_level_mark.raw_data['group_name'] = item['groupName']
                    first_level_mark.raw_data['color'] = item['color']
                    first_level_mark.raw_data['default_color'] = item['color']
                    model = self.convert_to_model(first_level_mark, ShareMarkGroupModel)
                    if not model:
                        return False
                    self.session.add(model)
                    self.session.flush([model])
                    first_level_mark.update_data(**model.raw_data)
                    first_parent_id = model.id
                logger.info('add first level mark success.first_parent_id= %s' % first_parent_id)
                for second_mark in item['children'] if 'children' in item else []:
                    logger.info('second_mark==%s' % second_mark)
                    second_level_mark = deepcopy(mark)
                    second_parent_id = ''
                    is_add = True
                    logger.info('is_add===%s' % is_add)
                    if is_add:
                        second_level_mark.raw_data['group_name'] = second_mark['groupName']
                        second_level_mark.raw_data['color'] = second_mark['color']
                        second_level_mark.raw_data['default_color'] = second_mark['color']
                        second_level_mark.raw_data['parent_id'] = first_parent_id
                        logger.info('second_level_mark===%s' % second_level_mark)
                        model = self.convert_to_model(second_level_mark, ShareMarkGroupModel)
                        if not model:
                            return False
                        self.session.add(model)
                        self.session.flush([model])
                        second_level_mark.update_data(**model.raw_data)
                        second_parent_id = model.id
                    logger.info('add second level mark success. first_parent_id=%s' % second_parent_id)
                    for third_mark in second_mark['children'] if 'children' in second_mark else []:
                        logger.info('third_mark==%s' % third_mark)
                        third_level_mark = deepcopy(mark)
                        logger.info('is_add===%s' % is_add)
                        if is_add:
                            third_level_mark.raw_data['group_name'] = third_mark['groupName']
                            third_level_mark.raw_data['color'] = third_mark['color']
                            third_level_mark.raw_data['default_color'] = third_mark['color']
                            third_level_mark.raw_data['parent_id'] = second_parent_id
                            model = self.convert_to_model(third_level_mark, ShareMarkGroupModel)
                            if not model:
                                return False
                            self.session.add(model)
                            self.session.flush([model])
                            third_level_mark.update_data(**model.raw_data)
                        logger.info('add third level mark success. %s' % third_level_mark)
        logger.info('add mark groups success.')
        return True

    @transaction
    def del_templates(self, id: int) -> bool:
        # 以往数据不变
        self.session.query(ShareMarkGroupModel).filter(ShareMarkGroupModel.template_id == id).delete(
            synchronize_session=False)
        self.session.query(TemplateModel).filter(TemplateModel.id == id).delete(synchronize_session=False)
        logger.info('del template end')
        return True

    def get_template_by_template_id(self, template_id: int) -> dict:
        row = self.session.query(TemplateModel.id, TemplateModel.template_name).filter(
            TemplateModel.id == template_id).first()
        return {'id': row[0], 'name': row[1]} if row else {}

    def get_template_by_template_name(self, template_name: str) -> dict:
        row = self.session.query(TemplateModel.id, TemplateModel.template_name).filter(
            TemplateModel.template_name == template_name).first()
        return {'id': row[0], 'name': row[1]} if row else {}

    def get_share_mark_groups(self) -> List[MarkGroupEntity]:
        models = self.session.query(ShareMarkGroupModel).filter().all()
        return [MarkGroupEntity.from_dict(model.raw_data) for model in models] if models else None
