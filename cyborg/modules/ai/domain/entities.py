from datetime import datetime, timedelta
from typing import Optional

from cyborg.app.settings import Settings
from cyborg.consts.common import Consts
from cyborg.seedwork.domain.entities import BaseDomainEntity


class AITaskEntity(BaseDomainEntity):

    slice_info: Optional[dict] = None

    def setup_expired_time(self):
        expired_at = datetime.now() + timedelta(seconds=Consts.ALGOR_OVERTIME.get(self.ai_type.value, 1800))
        self.update_data(expired_at=expired_at)


class AIStatisticsEntity(BaseDomainEntity):

    ...


class TCTProbEntity(BaseDomainEntity):

    check_result: str
