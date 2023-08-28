from datetime import datetime, timedelta

from cyborg.consts.common import Consts
from cyborg.modules.partner.roche.domain.value_objects import RocheAITaskStatus
from cyborg.seedwork.domain.entities import BaseDomainEntity


class RocheAlgorithmEntity(BaseDomainEntity):
    ...


class RocheAITaskEntity(BaseDomainEntity):

    def setup_expired_time(self):
        expired_at = datetime.now() + timedelta(seconds=Consts.ALGOR_OVERTIME.get(self.ai_type.value, 1800))
        self.update_data(expired_at=expired_at)

    def set_failed(self):
        self.update_data(status=RocheAITaskStatus.failed)

    @property
    def is_timeout(self):
        return self.status == RocheAITaskStatus.analyzing and datetime.now() > self.expired_at

    @property
    def is_finished(self):
        return self.status in (RocheAITaskStatus.success, RocheAITaskStatus.failed)
