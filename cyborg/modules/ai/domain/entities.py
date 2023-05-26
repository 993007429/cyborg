from datetime import datetime, timedelta
from typing import Optional, Dict, Type

from cyborg.consts.common import Consts
from cyborg.seedwork.domain.entities import BaseDomainEntity
from cyborg.seedwork.domain.value_objects import AIType, BaseEnum


class AITaskEntity(BaseDomainEntity):

    slice_info: Optional[dict] = None

    @property
    def enum_fields(self) -> Dict[str, Type[BaseEnum]]:
        return {'ai_type': AIType}

    def setup_expired_time(self):
        expired_at = datetime.now() + timedelta(seconds=Consts.ALGOR_OVERTIME.get(self.ai_type.value, 1800))
        self.update_data(expired_at=expired_at)

    @property
    def slide_path(self) -> str:
        return self.slice_info['slice_file_path'] if self.slice_info else ''


class AIStatisticsEntity(BaseDomainEntity):

    ...


class TCTProbEntity(BaseDomainEntity):

    check_result: str = ''
