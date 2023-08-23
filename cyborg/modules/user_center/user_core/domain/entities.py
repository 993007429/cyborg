import datetime
import json
import os
import shutil
from typing import Tuple, Optional, List

import aioshutil

from cyborg.app.settings import Settings
from cyborg.infra.fs import fs
from cyborg.seedwork.domain.entities import BaseDomainEntity
from cyborg.utils.strings import snake_to_camel

COMPANY_AI_THRESHOLD = {
    'tct': {'threshold_range': 0, 'threshold_value': 0.5, 'all_use': False},
    'lct': {'threshold_range': 0, 'threshold_value': 0.5, 'all_use': False}
}
COMPANY_DEFAULT_AI_THRESHOLD = {'tct': 0.5, 'lct': 0.5}


class CompanyEntity(BaseDomainEntity):

    @property
    def json_fields(self) -> List[str]:
        return ['ai_threshold', 'default_ai_threshold', 'trial_times', 'model_lis', 'table_checked']

    def is_available(self, client_ip: str) -> Tuple[int, str]:
        if self.allowed_ip:
            allowed_ip_list = self.allowed_ip.split(',')
            if client_ip not in allowed_ip_list:
                return 10, 'Client address is not in the allowed list'

        if self.is_test and str(datetime.datetime.now().date()) > self.end_time:
            return 400, 'this company already expire'

        return 0, ''

    @property
    def base_dir(self):
        return os.path.join(Settings.DATA_DIR, self.company)

    async def remove_data_dir(self):
        await aioshutil.rmtree(self.base_dir, ignore_errors=True)

    def to_dict(self):
        d = {snake_to_camel(k): v for k, v in super().to_dict().items()}
        # HACK for fe
        d['name'] = self.company
        d['model_lis'] = json.dumps(self.model_lis)
        d['defaultAiThreshold'] = json.dumps(self.default_ai_threshold)
        d['trialTimes'] = json.dumps(self.trial_times)
        return d


class UserEntity(BaseDomainEntity):

    companyEntity: Optional[CompanyEntity] = None

    @property
    def base_dir(self):
        return os.path.join(Settings.DATA_DIR, self.company)

    @property
    def user_dir(self):
        return os.path.join(self.base_dir, 'users', self.username)

    @property
    def sign_image_path(self):
        return os.path.join(self.user_dir, 'index.png')

    def create_user_dir(self):
        os.makedirs(self.user_dir, exist_ok=True)

    def remove_user_dir(self):
        if fs.path_exists(self.user_dir):
            shutil.rmtree(self.user_dir)

    def to_dict(self):
        d = super().to_dict()
        d.update({
            'name': self.username,
            'base_dir': self.base_dir,
            'user_dir': self.user_dir,
            'sign_image_path': self.sign_image_path,
        })
        if self.companyEntity:
            d.update({
                'importable': self.companyEntity.importable,
                'exportJson': self.companyEntity.export_json
            })
        return d
