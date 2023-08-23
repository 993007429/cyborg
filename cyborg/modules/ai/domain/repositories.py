from abc import ABCMeta, abstractmethod
from typing import Optional, List

from cyborg.modules.ai.domain.entities import AITaskEntity, AIStatisticsEntity, TCTProbEntity
from cyborg.modules.ai.domain.value_objects import AITaskStatus
from cyborg.seedwork.domain.value_objects import AIType


class AIRepository(metaclass=ABCMeta):

    @abstractmethod
    def save_ai_task(self, ai_task: AITaskEntity) -> bool:
        ...

    @abstractmethod
    def get_ai_task_by_id(self, task_id: int) -> Optional[AITaskEntity]:
        ...

    @abstractmethod
    def get_latest_ai_task(self, case_id: int, file_id, ai_type: AIType) -> Optional[AITaskEntity]:
        ...

    @abstractmethod
    def get_latest_calibrate_ai_task(self) -> Optional[AITaskEntity]:
        ...

    @abstractmethod
    def get_ai_task_ranking(self, task_id: int, start_id: Optional[int] = None) -> Optional[int]:
        ...

    @abstractmethod
    def get_ai_tasks(
            self, status: Optional[AITaskStatus], until_id: Optional[int], limit: int = 100) -> List[AITaskEntity]:
        ...

    @abstractmethod
    def get_ai_id_by_type(self, ai_type: AIType) -> Optional[int]:
        ...

    @abstractmethod
    def get_ai_name_by_template_id(self, template_id: int) -> Optional[str]:
        ...

    @abstractmethod
    def get_template_id_by_ai_name(self, ai_name: str) -> Optional[int]:
        ...

    @abstractmethod
    def save_ai_stats(self, stats: AIStatisticsEntity) -> bool:
        ...

    @abstractmethod
    def get_ai_stats(
            self, ai_type: AIType, company: str, date: Optional[str] = None,
            start_date: Optional[str] = None, end_date: Optional[str] = None, version: Optional[str] = None
    ) -> List[AIStatisticsEntity]:
        ...

    @abstractmethod
    def save_tct_prob(self, prob: TCTProbEntity) -> bool:
        ...

    @abstractmethod
    def get_tct_prob(self, slice_id: int) -> Optional[TCTProbEntity]:
        ...

    @abstractmethod
    def get_tct_probs_by_slices(self, slices: List[dict]) -> List[TCTProbEntity]:
        ...
