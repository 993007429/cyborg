import logging
from typing import Optional, List

from cyborg.modules.partner.roche.domain.entities import RocheAITaskEntity
from cyborg.modules.partner.roche.domain.repositories import RocheRepository
from cyborg.modules.partner.roche.domain.value_objects import RocheAITaskStatus
from cyborg.seedwork.domain.value_objects import AIType

logger = logging.getLogger(__name__)


class RocheDomainService(object):

    def __init__(self, repository: RocheRepository):
        super(RocheDomainService, self).__init__()
        self.repository = repository

    def create_ai_task(
            self,
            algorithm_id: str,
            slide_url: str,
            rois: Optional[List[dict]] = None
    ) -> Optional[RocheAITaskEntity]:

        algorithm = self.repository.get_algorithm(algorithm_id)
        task = RocheAITaskEntity(raw_data={
            'ai_type': AIType.get_by_value(algorithm.algorithm_name),
            'slide_url': slide_url,
            'status': RocheAITaskStatus.default,
            'rois': rois
        })

        if self.repository.save_ai_task(task):
            return task

        return None

    def update_ai_task(
            self, task: RocheAITaskEntity, status: Optional[RocheAITaskStatus] = None, result_id: Optional[str] = None
    ) -> bool:

        if status is not None:
            task.update_data(status=status)
            if status == RocheAITaskStatus.analyzing:
                task.setup_expired_time()
                # cache.set(self.RANK0_TASK_ID_CACHE_KEY, task.id)

        if result_id is not None:
            task.update_data(result_id=result_id)

        return self.repository.save_ai_task(task)
