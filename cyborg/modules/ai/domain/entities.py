from datetime import datetime, timedelta
from typing import Optional, Dict, Type

from cyborg.consts.common import Consts
from cyborg.modules.ai.domain.value_objects import AITaskStatus
from cyborg.seedwork.domain.entities import BaseDomainEntity
from cyborg.seedwork.domain.value_objects import AIType, BaseEnum
from cyborg.utils.id_worker import IdWorker
from cyborg.utils.strings import snake_to_camel


class AITaskEntity(BaseDomainEntity):

    slice_info: Optional[dict] = None

    @property
    def enum_fields(self) -> Dict[str, Type[BaseEnum]]:
        return {'ai_type': AIType}

    def setup_expired_time(self):
        expired_at = datetime.now() + timedelta(seconds=Consts.ALGOR_OVERTIME.get(self.ai_type.value, 1800))
        self.update_data(expired_at=expired_at)

    def set_failed(self):
        self.update_data(status=AITaskStatus.failed)

    @property
    def is_timeout(self):
        return self.status == AITaskStatus.analyzing and datetime.now() > self.expired_at

    @property
    def is_finished(self):
        return self.status in (AITaskStatus.success, AITaskStatus.failed)

    @property
    def slide_path(self) -> str:
        return self.slice_info['slice_file_path'] if self.slice_info else ''

    @classmethod
    def new_default_roi(cls) -> dict:
        return {
            'id': IdWorker.new_mark_id_worker().get_new_id(),
            'x': [],
            'y': []
        }


class AIStatisticsEntity(BaseDomainEntity):

    def to_stats_data(self):
        return {
            'totalCount': self.total_count or 0,
            'negativeCount': self.negative_count or 0,
            'positiveCount': self.positive_count or 0,
            'abnormalCount': self.abnormal_count or 0,
            'totalCountDr': self.total_count_dr or 0,
            'negativeCountDr': self.negative_count_dr or 0,
            'positiveCountDr': self.positive_count_dr or 0,
            'abnormalCountDr': self.abnormal_count_dr or 0
        }

    def to_dict(self):
        d = {snake_to_camel(k): v for k, v in super().to_dict().items()}
        return d


class TCTProbEntity(BaseDomainEntity):

    check_result: str = ''

    def to_list(self):
        return [self.prob_nilm, self.prob_ascus, self.prob_lsil, self.prob_asch, self.prob_hsil, self.prob_agc]
