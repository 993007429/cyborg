import math
import sys
from datetime import datetime, timedelta
from typing import Optional, Type, Tuple, List, Any

from sqlalchemy import desc, or_, func, distinct, and_

from cyborg.infra.session import transaction
from cyborg.modules.ai.domain.entities import TCTProbEntity
from cyborg.modules.ai.infrastructure.models import TCTProbModel
from cyborg.modules.slice.domain.entities import CaseRecordEntity, SliceEntity, ReportConfigEntity
from cyborg.modules.slice.domain.repositories import CaseRecordRepository, ReportConfigRepository
from cyborg.modules.slice.domain.value_objects import SliceStartedStatus
from cyborg.modules.slice.infrastructure.models import SliceModel, CaseRecordModel, ReportConfigModel, SliceErrModel
from cyborg.seedwork.domain.value_objects import AIType
from cyborg.seedwork.infrastructure.repositories import SQLAlchemySingleModelRepository
from cyborg.utils.strings import camel_to_snake


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

    def get_slices_by_case_id(self, case_id: str, company: str) -> List[SliceEntity]:
        models = self.session.query(
            SliceModel).filter_by(caseid=case_id, company=company).order_by(desc(SliceModel.id)).all()
        return [SliceEntity.from_dict(model.raw_data) for model in models]

    def get_slices(
            self, file_name: Optional[str] = None, ai_type: Optional[AIType] = None,
            started: Optional[SliceStartedStatus] = None, case_ids: List[int] = None,
            slice_type: Optional[str] = None,
            company: Optional[str] = None,
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
            query = query.filter(SliceModel.caseid.in_(case_ids))
        if slice_type is not None:
            query = query.filter(SliceModel.type == 'slice')
        if company:
            query = query.filter_by(company=company)

        offset = page * per_page
        models = query.offset(offset).limit(per_page)
        return [SliceEntity.from_dict(model.raw_data) for model in models]

    def get_slice_count_by_case_id(self, case_id: str, company: str) -> int:
        return self.session.query(
            SliceModel).filter_by(caseid=case_id, company=company, type='slice').order_by(desc(SliceModel.id)).count()

    def get_record_by_id(self, record_id: int) -> Optional[CaseRecordEntity]:
        model = self.session.query(CaseRecordModel).get(record_id)
        return CaseRecordEntity.from_dict(model.raw_data) if model else None

    def get_record_by_case_id(self, case_id: str, company: str) -> Optional[CaseRecordEntity]:
        model = self.session.query(CaseRecordModel).filter_by(caseid=case_id, company=company).first()
        return CaseRecordEntity.from_dict(model.raw_data) if model else None

    def get_records(
            self, end_time: Optional[str] = None, sample_num: Optional[str] = None, company: Optional[str] = None
    ) -> List[CaseRecordEntity]:
        query = self.session.query(CaseRecordModel)
        if end_time is not None:
            query = query.filter(CaseRecordModel.create_time < end_time)
        if sample_num is not None:
            query = query.filter(CaseRecordModel.sample_num == sample_num)
        if company is not None:
            query = query.filter(CaseRecordModel.company == company)

        models = query.order_by(CaseRecordModel.id.desc()).all()
        return [CaseRecordEntity.from_dict(model.raw_data) for model in models]

    def get_new_slices(
            self, company: str, start_id: int, upload_batch_number: Optional[str] = None
    ) -> Tuple[int, int, List[dict]]:
        query = self.session.query(SliceModel.id, SliceModel.caseid, SliceModel.fileid, SliceModel.alg).filter(
            SliceModel.company == company,
            SliceModel.id > start_id
        ).order_by(SliceModel.id.desc())
        if upload_batch_number is not None:
            query = query.filter(
                SliceModel.upload_batch_number == upload_batch_number
            )
            rows = query.all()
            increased = len(rows)
            last_id = rows[0][0] if rows else start_id
            slices = [{'caseid': row[1], 'fileid': row[2], 'ai_type': AIType.get_by_value(row[3])} for row in rows]
            return last_id, increased, slices
        else:
            row = query.first()
            last_id = row[0] if row else start_id
            increased = query.count()
            return last_id, increased, []

    def get_new_updated_slices(
            self, company: str, updated_after: Optional[datetime] = None, upload_batch_number: Optional[str] = None
    ) -> Tuple[int, List[dict]]:

        if updated_after is None:
            # TODO hack, 返回300秒之后更新过的记录，实际的业务逻辑需要前端船体update_after参数，也就是上一次获取到的更新时间
            updated_after = datetime.now() - timedelta(seconds=300)

        query = self.session.query(
            SliceModel.id, SliceModel.caseid, SliceModel.fileid, SliceModel.alg, SliceModel.started,
            SliceModel.last_modified
        ).filter(
            SliceModel.company == company,
            SliceModel.last_modified > updated_after
        ).order_by(SliceModel.last_modified.asc())

        if upload_batch_number is not None:
            query = query.filter(
                SliceModel.upload_batch_number == upload_batch_number
            )

        rows = query.all()
        updated = len(rows)
        slices = [{'caseid': row[1],
                   'fileid': row[2],
                   'ai_type': AIType.get_by_value(row[3]),
                   'status': SliceStartedStatus.get_by_value(row[4]),
                   'last_modified': row[5]} for row in rows]
        return updated, slices

    def get_pending_slices_count(self, company: str, upload_batch_number: str) -> int:
        return self.session.query(SliceModel.id, SliceModel.caseid, SliceModel.fileid, SliceModel.alg).filter(
            SliceModel.company == company,
            SliceModel.started == 0,
            SliceModel.upload_batch_number == upload_batch_number
        ).count()

    def get_slice(
            self, case_id: Optional[str] = None, file_id: Optional[str] = None, company: Optional[str] = None
    ) -> Optional[SliceEntity]:
        query = self.session.query(SliceModel)
        if case_id is not None:
            query = query.filter_by(caseid=case_id)
        if file_id is not None:
            query = query.filter_by(fileid=file_id)
        if company is not None:
            query = query.filter_by(company=company)
        model = query.first()
        return SliceEntity.from_dict(model.raw_data) if model else None

    def get_slice_err(self, case_id: str, file_id: str) -> Tuple[int, str]:
        model = self.session.query(SliceErrModel.err_code, SliceErrModel.err_message).filter_by(caseid=case_id,
                                                                                                fileid=file_id).first()
        if model:
            return model[0], model[1]
        return 0, ''

    def get_slice_by_local_filename(self, user_file_path: str, file_name: str, company: str) -> Optional[SliceEntity]:
        model = self.session.query(SliceModel).filter_by(
            company=company, user_file_path=user_file_path, filename=file_name).first()
        return SliceEntity.from_dict(model.raw_data) if model else None

    def get_slices_by_ids(self, slice_ids: List[int]) -> List[SliceEntity]:
        slices = []
        for slice_id in slice_ids:
            model = self.session.query(SliceModel).get(slice_id)
            if model:
                slices.append(SliceEntity.from_dict(model.raw_data))
        return slices

    def get_all_sample_types(self, company_id: str) -> List[dict]:
        rows = self.session.query(distinct(CaseRecordModel.sample_type)).filter_by(company=company_id).all()
        return [{"text": row[0], "value": row[0]} for row in rows] if rows else [{"text": '无', "value": ""}]

    def get_all_sample_parts(self, company_id: str) -> List[dict]:
        rows = self.session.query(distinct(CaseRecordModel.sample_part)).filter_by(company=company_id).all()
        return [{"text": row[0], "value": row[0]} for row in rows] if rows else [{"text": '无', "value": ""}]

    def get_all_user_folders(self, company_id: str) -> List[dict]:
        rows = self.session.query(distinct(SliceModel.user_file_folder)).filter_by(company=company_id).order_by(
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
            page: int = 0, limit: int = 20,
            case_ids: Optional[List[str]] = None,
            is_marked: Optional[List[int]] = None,
            labels: Optional[List[str]] = None,
            clarity_level: Optional[List[str]] = None,
            slice_quality: Optional[List[str]] = None,
            clarity_standards_min: float = 0.2, clarity_standards_max: float = 0.6
    ) -> Tuple[int, List[CaseRecordEntity]]:

        search_value = search_value.strip() if search_value else None
        is_record_search_key = search_key in ['sampleNum', 'name']
        is_slice_search_key = not is_record_search_key
        query = self.session.query(
            CaseRecordModel, func.group_concat(SliceModel.id.op("ORDER BY")(SliceModel.id.desc()))).outerjoin(
            SliceModel, CaseRecordModel.caseid == SliceModel.caseid).filter(
            CaseRecordModel.company == company_id
        )
        if search_key is not None and is_record_search_key and search_value:
            if search_key == 'sampleNum':
                query = query.filter(CaseRecordModel.sample_num.contains(search_value))
            else:
                query = query.filter(getattr(CaseRecordModel, search_key) == search_value)

        if gender is not None:
            query = query.filter(CaseRecordModel.gender.in_(gender))

        if age_min is not None:
            query = query.filter(CaseRecordModel.age.between(age_min, age_max))

        if sample_type is not None:
            query = query.filter(CaseRecordModel.sample_type.in_(sample_type))

        if sample_part is not None:
            query = query.filter(CaseRecordModel.sample_part.in_(sample_part))
        if slice_quality:
            temp = (1 == 0)
            if 0 in slice_quality:
                temp = or_(temp, SliceModel.slide_quality.notin_(['0', '1']))
            if 1 in slice_quality:
                temp = or_(temp, SliceModel.slide_quality == '1')
            if 2 in slice_quality:
                temp = or_(temp, SliceModel.slide_quality == '0')
            query = query.filter(temp)
        if clarity_level:
            temp = (1 == 0)
            if 0 in clarity_level:
                temp = or_(temp, SliceModel.clarity == '', SliceModel.clarity.is_(None))
            if 1 in clarity_level:
                temp = or_(temp, SliceModel.clarity > clarity_standards_max)
            if 2 in clarity_level:
                temp = or_(temp,
                           SliceModel.clarity.between(clarity_standards_min, clarity_standards_max))
            if 3 in clarity_level:
                temp = or_(temp, SliceModel.clarity < clarity_standards_min)
            query = query.filter(temp)
        if report == [1]:  # 有报告
            query = query.filter(CaseRecordModel.report != 2)
        elif report == [2]:  # 无报告
            query = query.filter(CaseRecordModel.report == 2)

        if update_min is not None and update_max is not None:
            query = query.filter(CaseRecordModel.update_time.between(update_min, update_max))

        if create_time_min is not None and create_time_max is not None:
            query = query.filter(CaseRecordModel.create_time.between(create_time_min, create_time_max))

        if search_key is not None and is_slice_search_key and search_value:
            if search_key == 'filename':
                query = query.filter(SliceModel.filename.contains(search_value))
            elif search_key == 'userFilePath':
                query = query.filter(SliceModel.user_file_folder == search_value)
            else:
                query = query.filter(getattr(SliceModel, camel_to_snake(search_key)) == search_value)
        if is_marked:
            query = query.filter(SliceModel.is_marked.in_(is_marked))
        if statuses is not None:
            query = query.filter(SliceModel.started.in_(statuses))

        if alg is not None:
            query = query.filter(SliceModel.alg.in_(alg))

        if slice_no == [0]:  # 没有切片编号
            query = query.filter(SliceModel.slice_number == '')
        elif slice_no == [1]:  # 有切片编号
            query = query.filter(SliceModel.slice_number != '')

        if is_has_label == [0]:  # 无切片标签
            query = query.filter(SliceModel.is_has_label == 0)
        elif is_has_label == [1]:  # 有切片标签
            query = query.filter(SliceModel.is_has_label == 1)
        if case_ids:
            query = query.filter(CaseRecordModel.caseid.in_(case_ids))
        if labels:
            temp = (1 == 0)
            for label in labels:
                temp = or_(temp, SliceModel.labels.contains(label))
            query = query.filter(temp)
        if ai_suggest is not None:
            temp = (1 == 0)
            for i in ai_suggest:
                if i == '无':
                    temp = or_(temp, SliceModel.ai_suggest == '')
                elif i == '结果异常':
                    temp = or_(temp, SliceModel.ai_tips != '')
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
            _order_by = getattr(CaseRecordModel, camel_to_snake(seq_key), None)
            if _order_by:
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
            if slice_ids:
                slice_id_list = slice_ids.split(',')
                for s in self.get_slices_by_ids(slice_id_list):
                    if s.type == 'slice':
                        s.clarity_level = s.get_clarity_level(clarity_standards_max=clarity_standards_max,
                                                              clarity_standards_min=clarity_standards_min)
                        if s.started == SliceStartedStatus.failed:
                            slice_err = self.session.query(SliceErrModel).filter_by(caseid=s.caseid,
                                                                                    fileid=s.fileid).first()
                            s.err_code = slice_err.err_code if slice_err else ''
                            s.err_message = slice_err.err_message if slice_err else ''
                        entity.slices.append(s)
                    elif s.type == 'attachment':
                        entity.attachments.append(s)

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

    def get_prob_list(self, company: str, ai_type: AIType) -> List[Tuple[TCTProbEntity, SliceEntity]]:
        models = self.session.query(TCTProbModel, SliceModel).join(
            SliceModel, TCTProbModel.slice_id == SliceModel.id).filter(
            and_(SliceModel.company == company, SliceModel.alg.contains(ai_type.value), SliceModel.started == 2)
        ).all()
        return [(TCTProbEntity.from_dict(prob_model.raw_data), SliceEntity.from_dict(slice_model.raw_data))
                for prob_model, slice_model in models]

    @transaction
    def add_label(self, ids: List[str], name: str) -> Tuple[int, str]:
        slices = self.session.query(SliceModel).filter(SliceModel.fileid.in_(ids))
        for slice in slices:
            if not name or len(name) > 20:
                return 10, '标签格式错误，请重新添加'
            temp = slice.labels or []
            if name in temp:
                return 11, '添加自定义标签失败，该切片已有该标签'
            if len(temp) > 4:
                return 12, '添加自定义标签失败，单个切片支持添加的自定义标签数量上限为5'
            temp.append(name)
            self.session.query(SliceModel).filter_by(caseid=slice.caseid, fileid=slice.fileid).update({"labels": temp})
        return 0, ''

    @transaction
    def del_label(self, id: str, name: List[str]) -> Tuple[int, str]:
        slice = self.session.query(SliceModel).filter_by(fileid=id).first()
        if not slice:
            return 11, '该切片对象不存在'
        slice.labels = [i for i in slice.labels if slice.labels and i not in name]
        self.session.add(slice)
        self.session.flush([slice])
        return 0, ''

    def get_labels(self, company: str) -> List[str]:
        rows = self.session.query(distinct(SliceModel.labels)).filter_by(company=company).all()
        labels = []
        for row in rows:
            if row[0]:
                labels.extend(row[0])
        return list(set(labels))


class SQLAlchemyReportConfigRepository(ReportConfigRepository, SQLAlchemySingleModelRepository[ReportConfigEntity]):

    @property
    def model_class(self) -> Type[ReportConfigModel]:
        return ReportConfigModel

    def get_by_company(self, company: str) -> Optional[ReportConfigEntity]:
        model = self.session.query(ReportConfigModel).filter_by(company=company).one_or_none()
        return ReportConfigEntity.from_dict(model.raw_data) if model else None
