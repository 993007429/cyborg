import sys
from abc import ABCMeta, abstractmethod
from typing import Optional, List, Tuple, Union

from cyborg.modules.slice_analysis.domain.entities import MarkEntity, MarkGroupEntity, MarkToTileEntity, \
    ChangeRecordEntity, NPCountEntity, Pdl1sCountEntity
from cyborg.modules.slice_analysis.domain.value_objects import AIType


class SliceMarkRepository(metaclass=ABCMeta):

    @property
    @abstractmethod
    def mark_table_suffix(self) -> str:
        ...

    @mark_table_suffix.setter
    @abstractmethod
    def mark_table_suffix(self, value):
        ...

    @property
    @abstractmethod
    def manual(self) -> 'SliceMarkRepository':
        ...

    @abstractmethod
    def create_mark_tables(self, ai_type: AIType):
        ...

    @abstractmethod
    def create_mark_table_by_import(self) -> Optional[str]:
        ...

    @abstractmethod
    def backup_ai_mark_tables(self) -> bool:
        ...

    @abstractmethod
    def clear_mark_table(
            self, ai_type: AIType, exclude_area_marks: Optional[List[int]] = None):
        ...

    @abstractmethod
    def save_mark(self, entity: MarkEntity) -> bool:
        ...

    @abstractmethod
    def batch_save_marks(self, entities: List[MarkEntity], sync_entity: bool = False) -> bool:
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def get_marks_by_diagnosis_result(self, diagnosis_result: str, ai_type: AIType) -> List[MarkEntity]:
        ...

    @abstractmethod
    def get_marks_by_area_id(self, area_id: int) -> List[MarkEntity]:
        ...

    @abstractmethod
    def delete_marks(
            self, mark_ids: Optional[List[int]] = None, group_id: Optional[int] = None, area_id: Optional[int] = None
    ) -> bool:
        ...

    @abstractmethod
    def delete_mark_by_id(self, mark_id: int) -> bool:
        ...

    @abstractmethod
    def delete_mark_to_tiles_by_mark_id(self, mark_id: int) -> bool:
        ...

    @abstractmethod
    def update_mark_group_selected(self, group_id: int) -> bool:
        ...

    @abstractmethod
    def save_mark_to_tiles(self, entities: List[MarkToTileEntity]) -> bool:
        ...

    @abstractmethod
    def save_change_record(self, entity: ChangeRecordEntity) -> bool:
        ...

    @abstractmethod
    def get_mark(self, mark_id: int) -> Optional[MarkEntity]:
        ...

    @abstractmethod
    def delete_mark_by_type(self, mark_type: int) -> bool:
        ...

    @abstractmethod
    def get_mark_count_by_tile_id(self, tile_id: int) -> int:
        ...

    @abstractmethod
    def get_mark_count(self, group_id: Optional[int] = None) -> int:
        ...

    @abstractmethod
    def save_mark_group(self, entity: MarkGroupEntity) -> bool:
        ...

    @abstractmethod
    def get_mark_group_by_id(self, group_id: int) -> Optional[MarkGroupEntity]:
        ...

    @abstractmethod
    def delete_mark_group(self, group_id: int) -> bool:
        ...

    @abstractmethod
    def get_selected_mark_group(self) -> Optional[MarkGroupEntity]:
        ...

    @abstractmethod
    def get_mark_groups_by_template_id(
            self, template_id: int, primary_only: bool = False, is_import: Optional[int] = None,
            is_ai: Optional[int] = None) -> List[MarkGroupEntity]:
        ...

    @abstractmethod
    def get_default_mark_groups(self, template_id: Optional[int] = None) -> List[MarkGroupEntity]:
        ...

    @abstractmethod
    def delete_mark_groups_by_template_id(self, template_id: int) -> bool:
        ...

    @abstractmethod
    def get_mark_groups_by_parent_id(self, parent_id: int) -> List[MarkGroupEntity]:
        ...

    @abstractmethod
    def get_visible_mark_group_ids(self) -> List[int]:
        ...

    @abstractmethod
    def update_mark_group_status(self, group_id: int, is_empty: int) -> bool:
        ...

    @abstractmethod
    def update_pdl1_count_in_tile(self, tile_id: int, field_name: str, count_delta: int) -> bool:
        ...

    @abstractmethod
    def update_np_count_in_tile(self, tile_id: int, field_name: str, count_delta: int) -> bool:
        ...

    @abstractmethod
    def delete_count(self, tile_ids: List[int], ai_type: AIType):
        ...

    @abstractmethod
    def get_cell_count(self, ai_type: AIType, tile_ids: List[int]) -> List[Union[Pdl1sCountEntity, NPCountEntity]]:
        ...

    @abstractmethod
    def get_mark_groups(self) -> List[MarkGroupEntity]:
        ...


class AIConfigRepository(metaclass=ABCMeta):

    @abstractmethod
    def get_ai_id_by_type(self, ai_type: AIType) -> Optional[int]:
        ...

    @abstractmethod
    def get_ai_name_by_template_id(self, template_id: int) -> Optional[str]:
        ...

    @abstractmethod
    def get_all_templates(self) -> List[dict]:
        ...
