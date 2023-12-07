import sys
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import Optional, Type, Tuple, List, Any

from cyborg.modules.ai.domain.entities import TCTProbEntity
from cyborg.modules.slice.domain.entities import CaseRecordEntity, SliceEntity, ReportConfigEntity
from cyborg.modules.slice.domain.value_objects import SliceStartedStatus
from cyborg.seedwork.domain.value_objects import AIType
from cyborg.seedwork.infrastructure.repositories import SingleModelRepository


class CaseRecordRepository(SingleModelRepository[CaseRecordEntity], metaclass=ABCMeta):

    @property
    def entity_class(self) -> Type[CaseRecordEntity]:
        return CaseRecordEntity

    @abstractmethod
    def save_slice(self, entity: SliceEntity) -> bool:
        ...

    @abstractmethod
    def get_slice_by_id(self, slice_id: int) -> Optional[SliceEntity]:
        ...

    @abstractmethod
    def get_slices_by_case_id(self, case_id: str, company: str) -> List[SliceEntity]:
        ...

    @abstractmethod
    def get_slices(
            self, file_name: Optional[str] = None, ai_type: Optional[AIType] = None,
            started: Optional[SliceStartedStatus] = None, case_ids: List[int] = None,
            slice_type: Optional[str] = None,
            company: Optional[str] = None,
            page: int = 0, per_page: int = sys.maxsize
    ) -> List[SliceEntity]:
        ...

    @abstractmethod
    def get_slice_count_by_case_id(self, case_id: str, company: str) -> int:
        ...

    @abstractmethod
    def get_record_by_id(self, record_id: int) -> Optional[CaseRecordEntity]:
        ...

    @abstractmethod
    def get_record_by_case_id(self, case_id: str, company: str) -> Optional[CaseRecordEntity]:
        ...

    @abstractmethod
    def get_records(
            self, end_time: Optional[str] = None, sample_num: Optional[str] = None, company: Optional[str] = None
    ) -> List[CaseRecordEntity]:
        ...

    @abstractmethod
    def get_new_slices(
            self, company: str, start_id: int, upload_batch_number: Optional[str] = None
    ) -> Tuple[int, int, List[dict]]:
        ...

    @abstractmethod
    def get_pending_slices_count(self, company: str, upload_batch_number: str) -> int:
        ...

    @abstractmethod
    def get_new_updated_slices(
            self, company: str, updated_after: Optional[datetime] = None, upload_batch_number: Optional[str] = None
    ) -> Tuple[int, List[dict]]:
        ...

    @abstractmethod
    def get_slice(
            self, case_id: Optional[str] = None, file_id: Optional[str] = None, company: Optional[str] = None
    ) -> Optional[SliceEntity]:
        ...

    def get_slice_err(self, case_id: str, file_id: str) -> Tuple[int, str]:
        ...

    @abstractmethod
    def get_slice_by_local_filename(self, user_file_path: str, file_name: str, company: str) -> Optional[SliceEntity]:
        ...

    @abstractmethod
    def get_all_sample_types(self, company_id: str) -> List[dict]:
        ...

    @abstractmethod
    def get_all_sample_parts(self, company_id: str) -> List[dict]:
        ...

    @abstractmethod
    def get_all_user_folders(self, company_id: str) -> List[dict]:
        ...

    @abstractmethod
    def get_all_operators(self, company_id: str) -> List[dict]:
        ...

    @abstractmethod
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
            clarity_standards_min: float = 0.2, clarity_standards_max: float = 0.6,
            ai_threshold: Optional[dict] = None,
            pattern_name: Optional[List[str]] = None
    ) -> Tuple[int, List[CaseRecordEntity]]:
        ...

    @abstractmethod
    def delete_record(self, case_id: str, company_id: str) -> bool:
        ...

    @abstractmethod
    def delete_slice(self, file_id: str, company_id: str) -> bool:
        ...

    @abstractmethod
    def get_prob_list(self, company: str, ai_type: AIType, caseid_list: Optional[List[str]] = None) -> List[Tuple[TCTProbEntity, SliceEntity]]:
        ...

    @abstractmethod
    def add_label(self, ids: List[str], name: str) -> Tuple[int, str]:
        ...

    @abstractmethod
    def del_label(self, id: str, name: List[str]) -> Tuple[int, str]:
        ...

    @abstractmethod
    def get_labels(self, company: str) -> List[str]:
        ...


class ReportConfigRepository(SingleModelRepository[ReportConfigEntity], metaclass=ABCMeta):

    @property
    def entity_class(self) -> Type[ReportConfigEntity]:
        return ReportConfigEntity

    @abstractmethod
    def get_by_company(self, company: str) -> Optional[ReportConfigEntity]:
        ...
