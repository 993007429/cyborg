from datetime import datetime, timedelta
from typing import Dict, Type
from urllib.parse import urlparse

from cyborg.consts.common import Consts
from cyborg.modules.partner.roche.domain.consts import ROCHE_TIME_FORMAT
from cyborg.modules.partner.roche.domain.value_objects import RocheAITaskStatus, RocheAlgorithmType
from cyborg.seedwork.domain.entities import BaseDomainEntity
from cyborg.seedwork.domain.value_objects import BaseEnum, AIType


class RocheAlgorithmEntity(BaseDomainEntity):

    @property
    def enum_fields(self) -> Dict[str, Type[BaseEnum]]:
        return {
            'algorithm_type': RocheAlgorithmType
        }


class RocheAITaskEntity(BaseDomainEntity):

    @property
    def enum_fields(self) -> Dict[str, Type[BaseEnum]]:
        return {
            'ai_type': AIType
        }

    @property
    def slide_path(self) -> str:
        parsed = urlparse(self.slide_url)
        suffix = parsed.path.split('.')[-1] if parsed.path and '.' in parsed.path else ''
        if suffix:
            return f'/data/download/{self.analysis_id}.{suffix}'
        else:
            return f'/data/download/{self.analysis_id}'

    def setup_expired_time(self):
        expired_at = datetime.now() + timedelta(seconds=Consts.ALGOR_OVERTIME.get(self.ai_type.value, 1800))
        self.update_data(expired_at=expired_at)

    def set_failed(self):
        self.update_data(status=RocheAITaskStatus.failed)

    @property
    def is_timeout(self):
        return self.status in (RocheAITaskStatus.accepted, RocheAITaskStatus.in_progress) and datetime.now() > self.expired_at

    @property
    def is_finished(self):
        return self.status in (
            RocheAITaskStatus.completed, RocheAITaskStatus.failed, RocheAITaskStatus.cancelled, RocheAITaskStatus.closed)

    @property
    def percentage_completed(self):
        if self.status == RocheAITaskStatus.completed:
            return 100
        else:
            return 0

    @property
    def status_name(self):
        status = RocheAITaskStatus.get_by_value(self.status)
        return status.display_name if status else ''

    @property
    def result_file_key(self):
        return f'partner/roche/analysis/{self.analysis_id}.h5'

    def to_dict(self):
        return {
            'analysis_id': self.analysis_id,
            'started_timestamp': self.started_at.strftime(ROCHE_TIME_FORMAT) if self.started_at else None,
            'last_updated_timestamp': self.last_modified.strftime(ROCHE_TIME_FORMAT) if self.last_modified else None,
            'status': self.status_name,
            'percentage_completed': self.percentage_completed,
            'status_detail_message': ''
        }
