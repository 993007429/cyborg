from abc import ABCMeta, abstractmethod
from typing import Optional, List

from cyborg.modules.partner.roche.domain.entities import RocheAITaskEntity, RocheAlgorithmEntity


class RocheRepository(metaclass=ABCMeta):

    @abstractmethod
    def get_algorithms(self) -> List[RocheAlgorithmEntity]:
        ...

    @abstractmethod
    def get_algorithm(self, algorithm_id: str) -> Optional[RocheAlgorithmEntity]:
        ...

    @abstractmethod
    def save_ai_task(self, ai_task: RocheAITaskEntity) -> bool:
        ...

    @abstractmethod
    def get_ai_task_by_id(self, task_id: int) -> Optional[RocheAITaskEntity]:
        ...

    @abstractmethod
    def get_ai_task_by_analysis_id(self, analysis_id: str) -> Optional[RocheAITaskEntity]:
        ...
