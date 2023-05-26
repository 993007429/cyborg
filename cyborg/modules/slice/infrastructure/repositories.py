import json
import math
import sys
from typing import Optional, Type, Tuple, List, Any

from sqlalchemy import desc, or_, func, distinct

from cyborg.infra.session import transaction
from cyborg.modules.slice.domain.entities import CaseRecordEntity, SliceEntity
from cyborg.modules.slice.domain.repositories import CaseRecordRepository
from cyborg.modules.slice.domain.value_objects import SliceStartedStatus
from cyborg.modules.slice.infrastructure.models import SliceModel, CaseRecordModel
from cyborg.seedwork.domain.value_objects import AIType
from cyborg.seedwork.infrastructure.repositories import SQLAlchemySingleModelRepository


class SQLAlchemyCaseRecordRepository(CaseRecordRepository, SQLAlchemySingleModelRepository[CaseRecordEntity]):

    @property
    def model_class(self) -> Type[CaseRecordModel]:
        return CaseRecordModel

    @transaction
    def save_slice(self, entity: SliceEntity) -> bool:
        model = self.convert_to_model(entity, SliceModel)
        if not model:
            return False
        self.session.add(model)
        self.session.flush([model])
        entity.update_data(**model.raw_data)
        return True

    def get_slice_by_id(self, slice_id: int) -> Optional[SliceEntity]:
        model = self.session.query(SliceModel).get(slice_id)
        return SliceEntity.from_dict(model.raw_data) if model else None

    def get_slices_by_case_id(self, case_id: str,  company: str) -> List[SliceEntity]:
        models = self.session.query(
            SliceModel).filter_by(caseid=case_id, company=company).order_by(desc(SliceModel.id)).all()
        return [SliceEntity.from_dict(model.raw_data) for model in models]

    def get_slices(
            self, file_name: Optional[str] = None, ai_type: Optional[AIType] = None,
            started: Optional[SliceStartedStatus] = None, case_ids: List[int] = None, company: Optional[str] = None,
            page: int = 0, per_page: int = sys.maxsize
    ) -> List[SliceEntity]:
        query = self.session.query(SliceModel)
        if file_name is not None:
            query = query.filter_by(filename=file_name)
        if ai_type is not None:
            query = query.filter(SliceModel.alg.contains(ai_type.value))
        if started is not None:
            query = query.filter_by(started=started.value)
        if case_ids is not None:
            query = query.filter_by(SliceModel.caseid.in_(case_ids))
        if company is not None:
            query = query.filter_by(company=company)

        offset = page * per_page
        models = query.offset(offset).limit(per_page)
        return [SliceEntity.from_dict(model.raw_data) for model in models]

    def get_slice_count_by_case_id(self, case_id: str, company: str) -> int:
        return self.session.query(
            SliceModel).filter_by(caseid=case_id, company=company).order_by(desc(SliceModel.id)).count()

    def get_record_by_case_id(self, case_id: str, company: str) -> Optional[CaseRecordEntity]:
        model = self.session.query(CaseRecordModel).filter_by(caseid=case_id, company=company).first()
        return CaseRecordEntity.from_dict(model.raw_data) if model else None

    def get_slice(self, case_id: str, file_id: str, company: str) -> Optional[SliceEntity]:
        model = self.session.query(SliceModel).filter_by(caseid=case_id, fileid=file_id, company=company).first()
        return SliceEntity.from_dict(model.raw_data) if model else None

    def get_slice_by_local_filename(self, user_file_path: str, file_name: str, company: str) -> Optional[SliceEntity]:
        model = self.session.query(SliceModel).filter_by(
            company=company, user_file_path=user_file_path, file_name=file_name).first()
        return SliceEntity.from_dict(model.raw_data) if model else None

    def get_slices_by_ids(self, slice_ids: List[int]) -> List[SliceEntity]:
        slices = []
        for slice_id in slice_ids:
            model = self.session.query(SliceModel).get(slice_id)
            if model:
                slices.append(SliceEntity.from_dict(model.raw_data))
        return slices

    def get_all_sample_types(self, company_id: str) -> List[dict]:
        rows = self.session.query(distinct(CaseRecordModel.sample_part)).filter_by(company=company_id).all()
        return [{"text": row[0], "value": row[0]} for row in rows] if rows else [{"text": '无', "value": ""}]

    def get_all_sample_parts(self, company_id: str) -> List[dict]:
        rows = self.session.query(distinct(CaseRecordModel.sample_part)).filter_by(company=company_id).all()
        return [{"text": row[0], "value": row[0]} for row in rows] if rows else [{"text": '无', "value": ""}]

    def get_all_user_folders(self, company_id: str) -> List[dict]:
        rows = self.session.query(SliceModel.user_file_folder).filter_by(company=company_id).order_by(
                SliceModel.id.desc()).limit(20).all()
        user_folders = list(set(filter(None, [r[0] for r in rows])))
        return [{
            'text': user_folder,
            'value': user_folder
        } for user_folder in user_folders] if user_folders else [{"text": '无', "value": ""}]

    def get_all_operators(self, company_id: str) -> List[dict]:
        rows = self.session.query(distinct(SliceModel.operator)).filter_by(company=company_id).all()
        return [{"text": row[0], "value": row[0]} for row in rows] if rows else [{"text": '无', "value": ""}]

    def search_records(
            self,
            company_id: str,
            search_key: Optional[str] = None, search_value: Any = None,
            gender: Optional[str] = None,
            age_min: Optional[str] = None, age_max: Optional[int] = None,
            sample_part: Optional[str] = None, sample_type: Optional[str] = None,
            report: Optional[List[int]] = None,
            statuses: Optional[List[int]] = None,
            alg: Optional[List[str]] = None,
            slice_no: Optional[str] = None,
            is_has_label: Optional[int] = None,
            ai_suggest: Optional[str] = None,
            check_result: Optional[str] = None,
            user_file_folder: Optional[str] = None,
            operator: Optional[str] = None,
            seq_key: Optional[str] = None, seq: Optional[str] = None,
            update_min: Optional[str] = None, update_max: Optional[str] = None,
            create_time_min: Optional[str] = None, create_time_max: Optional[str] = None,
            page: int = 0, limit: int = 20
    ) -> Tuple[int, List[CaseRecordEntity]]:

        search_value = search_value.strip() if search_value else None
        is_record_search_key = search_key in ['sampleNum', 'name']
        is_slice_search_key = not is_record_search_key

        query = self.session.query(CaseRecordModel, func.group_concat(SliceModel.id)).join(
            SliceModel, CaseRecordModel.caseid == SliceModel.caseid).filter(
                CaseRecordModel.company == company_id,
                SliceModel.type == 'slice'
        )

        if search_key is not None and is_record_search_key and search_value:
            if search_key == 'sampleNum':
                query = query.filter(CaseRecordModel.sampleNum.contains(search_value))
            else:
                query = query.filter(getattr(CaseRecordModel, search_key) == search_value)

        if gender is not None:
            query = query.filter(CaseRecordModel.gender.in_(gender))

        if age_min is not None:
            query = query.filter(CaseRecordModel.age.between(age_min, age_max))

        if sample_type is not None:
            query = query.filter(CaseRecordModel.sampleType.in_(sample_type))

        if sample_part is not None:
            query = query.filter(CaseRecordModel.sample_part.in_(sample_part))

        if report == [1]:  # 有报告
            query = query.filter(CaseRecordModel.report != 2)
        elif report == [2]:  # 无报告
            query = query.filter(CaseRecordModel.report == 2)

        if update_min is not None and update_max is not None:
            query = query.filter(CaseRecordModel.update_time.between(update_min, update_max))

        if create_time_min is not None and create_time_max is not None:
            query = query.filter(CaseRecordModel.create_time.between(
                json.loads(create_time_min), json.loads(create_time_max)))

        if search_key is not None and is_slice_search_key and search_value:
            if search_key == 'filename':
                query = query.filter(SliceModel.filename.contains(search_value))
            elif search_key == 'userFilePath':
                query = query.filter(SliceModel.user_file_folder == search_value)
            else:
                query = query.filter(getattr(SliceModel, search_key) == search_value)

        if statuses is not None:
            query = query.filter(SliceModel.ai_status.in_(statuses))

        if alg is not None:
            query = query.filter(SliceModel.alg.in_(alg))

        if slice_no == [0]:  # 没有切片编号
            query = query.filter(SliceModel.slice_number == '')
        elif slice_no == [1]:  # 有切片编号
            query = query.filter(SliceModel.slice_number != '')

        if is_has_label == [0]:  # 无切片标签
            query = query.filter(SliceModel.isHasLabel == 0)
        elif is_has_label == [1]:  # 有切片标签
            query = query.filter(SliceModel.isHasLabel == 1)

        if ai_suggest is not None:
            temp = (1 == 0)
            for i in ai_suggest:
                if i == '无':
                    temp = or_(temp, SliceModel.ai_suggest == '')
                else:
                    temp = or_(temp, SliceModel.ai_suggest.contains(i))
            query = query.filter(temp)

        if check_result is not None:
            temp = (1 == 0)
            for i in check_result:
                if i == '无':
                    temp = or_(temp, SliceModel.check_result == '')
                else:
                    temp = or_(temp, SliceModel.check_result.contains(i))
            query = query.filter(temp)

        if user_file_folder is not None:
            query = query.filter(SliceModel.user_file_folder.in_(user_file_folder))

        if operator is not None:
            query = query.filter(SliceModel.operator.in_(operator))

        query = query.group_by(CaseRecordModel.id)

        if seq_key:
            # TODO 前段传参HACK
            if seq_key == 'slice_num':
                seq_key = 'slice_count'
            _order_by = getattr(CaseRecordModel, seq_key)
            if seq == '1':  # 倒序
                _order_by = desc(_order_by)
            query = query.order_by(_order_by)
        else:
            query = query.order_by(desc(CaseRecordModel.id))

        total = query.count()

        page = min(page, math.ceil(total / limit))
        offset = page * limit
        query = query.offset(offset).limit(limit)

        records = []
        for model, slice_ids in query.all():
            entity = CaseRecordEntity.from_dict(model.raw_data)
            slice_id_list = slice_ids.split(',')
            if slice_id_list:
                entity.slices = self.get_slices_by_ids(slice_id_list)
            records.append(entity)
        return total, records

    @transaction
    def delete_record(self, case_id: str, company_id: str) -> bool:
        self.session.query(CaseRecordModel).filter_by(caseid=case_id, company=company_id).delete()
        self.session.query(SliceModel).filter_by(caseid=case_id, company=company_id).delete()
        return True

    def delete_slice(self, file_id: str, company_id: str) -> bool:
        self.session.query(SliceModel).filter_by(fileid=file_id, company=company_id).delete()
        return True
