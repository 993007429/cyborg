import json
import logging
import os
import shutil
import time
from typing import Optional, Tuple, List

from pubsub import pub

from cyborg.app.settings import Settings
from cyborg.infra.fs import fs
from cyborg.infra.session import transaction
from cyborg.modules.user_center.user_core.domain.entities import UserEntity, CompanyEntity
from cyborg.modules.user_center.user_core.domain.event import CompanyAIThresholdUpdatedEvent
from cyborg.modules.user_center.user_core.domain.repositories import UserRepository, CompanyRepository
from cyborg.seedwork.domain.value_objects import AIType

logger = logging.getLogger(__name__)


class UserCoreDomainService(object):

    def __init__(self, repository: UserRepository, company_repository: CompanyRepository):
        super(UserCoreDomainService, self).__init__()
        self.repository = repository
        self.company_repository = company_repository

    def get_user_by_name(self, username: str, company: str) -> Optional[UserEntity]:
        return self.repository.get_user_by_name(username, company)

    def validate_company(self, company_id: str, client_ip: str) -> Tuple[int, str, Optional[CompanyEntity]]:
        company = self.company_repository.get_company_by_id(company_id)
        if not company:
            return 1, f'no such group as {company_id}', None

        err_code, message = company.is_available(client_ip)
        return err_code, message, company if not err_code else None

    def login(
            self, username: str, password: str, company_id: str, client_ip: str
    ) -> Tuple[int, str, Optional[UserEntity]]:

        err_code, message, company = self.validate_company(company_id, client_ip=client_ip)
        if err_code:
            return err_code, message, None

        user = self.repository.get_user_by_name(username=username, company=company_id)
        if not user:
            return 2, '%s has no such user as %s' % (company_id, username), None

        if user.password != password:
            return 3, 'wrong password', None

        user.last_login = time.time()
        if not self.repository.save(user):
            return 500, 'login error', None

        user.companyEntity = company

        return 0, '', user

    def update_signed(self, user: UserEntity) -> bool:
        if not os.path.exists(user.user_dir):
            os.makedirs(user.user_dir, exist_ok=True)

        user.signed = 1
        return self.repository.save(user)

    def update_password(self, user: UserEntity, old_password: str, new_password: str) -> Tuple[int, str]:
        if user.password != old_password:
            return 3, 'wrong old password'

        user.update_data(password=new_password)
        self.repository.save(user)
        return 0, ''

    def get_customized_record_fields(self, company_id: str) -> List[str]:
        company = self.company_repository.get_company_by_id(company_id)
        table_checked = company.table_checked if company else None
        return table_checked or [
            '样本号', '姓名', '性别', '切片数量', '状态', '切片标签', '切片编号', '文件名', 'AI模块', 'AI建议', '复核结果',
            '最后更新', '报告', '创建时间'
        ]

    def set_customized_record_fields(self, company_id: str, record_fields: str) -> bool:
        company = self.company_repository.get_company_by_id(company_id)
        if not company:
            return False
        company.update_data(table_checked=record_fields)
        return self.company_repository.save(company)

    def update_report_settings(
            self, company: CompanyEntity, report_name: Optional[str] = None, report_info: Optional[str] = None
    ) -> bool:
        if report_name is not None:
            company.update_data(report_name=report_name)
        if report_info is not None:
            company.update_data(report_info=report_info)

        return self.company_repository.save(company)

    def get_company_storage_info(self, company_id: str) -> Optional[dict]:
        company = self.company_repository.get_company_by_id(company_id)
        if not company:
            return None

        company_data_path = company.base_dir
        if fs.path_exists(company_data_path):
            if len(fs.listdir(company_data_path)) == 0:
                company_usage = 0
                company.usage = company_usage
                company.save()

        try:
            total, used, _ = shutil.disk_usage(company_data_path)
        except:
            total, used, _ = shutil.disk_usage(Settings.DATA_DIR)

        total = total // (2 ** 30)  # 总空间
        if Settings.CLOUD:  # 公有云
            total_space = float(company.volume)  # 公有云用配置的值做分母
            company_usage = float(company.usage)  # 公有云用累计使用空间的值做分子
        else:  # 工作站
            # 工作站用物理空间做分母
            total_space = total - 10  # 预留10G的物理空间兜底
            company_usage = used // (2 ** 30)  # 工作站用实际的磁盘使用空间值做分子

        if total_space:
            space = company_usage / total_space if company_usage / total_space <= 1 else 1
        else:
            space = 1
        return {
            'remainingSpace': round(100 - space * 100, 0)
        }

    def update_company_storage_usage(self, company_id: str, increased_size: float) -> bool:
        company = self.company_repository.get_company_by_id(company_id)
        if not company:
            return False

        company.update_data(usage=max(float(company.usage) + increased_size, 0))
        success = self.company_repository.save(company)
        if success:
            logger.info(f'空间变化：{increased_size}GB')
        return success

    def update_company_trial(self, company_id: str, ai_name: str) -> Tuple[int, str, Optional[CompanyEntity]]:
        company = self.company_repository.get_company_by_id(company_id)
        if not company:
            return 2, '组织不存在', None

        if company.on_trial:  # 如果试用则需要维护算法剩余次数
            trial_times = company.trial_times
            ai_trial_count = trial_times.get(ai_name)
            if Settings.CLOUD and not ai_trial_count:
                return 1, '当前为试用账号，该模块的可用次数已用尽。', None

            trial_times[ai_name] = ai_trial_count - 1
            company.update_data(trial_times=json.dumps(trial_times))
            if not self.company_repository.save(company):
                return 3, '服务发生错误', None

        return 0, '', company

    def update_company_label(self, company: CompanyEntity, label: int) -> bool:
        company.update_data(label=label)
        return self.company_repository.save(company)

    def update_company_ai_threshold(
            self, company: CompanyEntity, model_name: str, threshold_value: Optional[float] = None,
            use_default_threshold: bool = False
    ) -> bool:
        ai_threshold = company.ai_threshold
        default_ai_threshold = company.default_ai_threshold
        for k in ai_threshold.keys():
            if use_default_threshold and k in default_ai_threshold:
                threshold_value = default_ai_threshold[k]
            ai_threshold[k]['threshold_value'] = threshold_value
            ai_threshold[k]['model_name'] = model_name
        company.update_data(ai_threshold=ai_threshold)
        return self.company_repository.save(company)

    ############################################
    # for admin use
    def create_user(self, username: str, password: str, company_id: str, role: str) -> Tuple[str, Optional[UserEntity]]:
        user = self.repository.get_user_by_name(username=username, company=company_id)
        if user:
            return 'username has been used', None

        company = self.company_repository.get_company_by_id(company=company_id)

        new_user = UserEntity(raw_data=dict(
            username=username,
            model_lis=company.model_lis,
            company=company.company,
            password=password,
            role=role,
            is_test=company.is_test,
            time_out=company.end_time
        ))
        if self.repository.save(new_user):
            logger.info('%s组织下增加了用户%s' % (company_id, username))
            return '', new_user
        return 'create user failed', None

    def update_user(self, user_id: int, username: str, password: str, role: str) -> str:
        user = self.repository.get(user_id)
        if not user:
            return 'user not exist'

        user.update_data(
            username=username,
            password=password,
            role=role
        )

        if not self.repository.save(user):
            return 'update user failed'

        return ''

    @transaction
    def delete_user(self, username: str, company: str) -> str:
        user = self.repository.get_user_by_name(username=username, company=company)
        if not user:
            return 'user not exist'

        if not self.repository.delete_by_id(user.id):
            return 'delete user failed'

        user.remove_user_dir()

        return ''

    def create_company(
            self, company_id: str, model_lis: str, volume: str, remark: str,
            ai_threshold: dict, default_ai_threshold: str, on_trial: int, importable: int, export_json: int,
            trial_times: str, is_test: int, end_time: str, **_
    ) -> str:
        company = self.company_repository.get_company_by_id(company_id)
        if company:
            return 'company id has been used'

        for k, v in ai_threshold.items():
            ai_threshold[k] = {
                'threshold_range': 0,  # 调参范围，默认阴性和ASC-US、ASC-H切片->0，所有类型切片->1
                'threshold_value': v,  # 参数数值
                'all_use': False  # 已处理切片也应用新参数，默认不选
            }

        new_company = CompanyEntity(raw_data=dict(
            company=company_id,
            model_lis=model_lis,
            volume=volume,
            remark=remark,
            ai_threshold=ai_threshold,
            default_ai_threshold=default_ai_threshold,
            create_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
            on_trial=on_trial,
            trial_times=trial_times,
            importable=importable,
            is_test=is_test,
            end_time=end_time,
            usage=0,
            export_json=export_json
        ))
        self.company_repository.save(new_company)

        logger.info('增加了%s组织' % new_company.company)
        return ''

    @transaction
    def update_company(
            self, uid: int, old_company_name: str, new_company_name: str, model_lis: str, volume: str, remark: str,
            default_ai_threshold: str, on_trial: int, importable: int, export_json: int, trial_times: str, is_test: int,
            end_time: str, **_
    ):
        existed_company = self.company_repository.get_company_by_id(new_company_name)
        if existed_company and existed_company.id != uid:
            return '组织名已存在'

        company = self.company_repository.get(uid)
        if not company:
            return 'company not found'

        old_company_data_dir = company.base_dir
        company.update_data(
            company=new_company_name,
            model_lis=model_lis,
            volume=volume,
            remark=remark,
            default_ai_threshold=default_ai_threshold,
            on_trial=on_trial,
            trial_times=trial_times,
            importable=importable,
            export_json=export_json,
            is_test=is_test,
            end_time=end_time,
        )
        self.company_repository.save(company)

        users = self.repository.get_users(company=old_company_name)
        for user in users:
            user.update_data(
                company=new_company_name,
                model_lis=model_lis,
                is_test=is_test,
                time_out=end_time
            )
            self.repository.save(user)

        new_company_data_dir = company.base_dir
        if new_company_data_dir != old_company_data_dir:
            if fs.path_exists(old_company_data_dir):
                os.rename(old_company_data_dir, new_company_data_dir)
            os.remove('clean_handle_1.bat')   # TODO what's this?

        return ''

    @transaction
    async def delete_company(self, company_id: str) -> str:
        if company_id == 'admin':
            return 'admin can not be deleted'

        company = self.company_repository.get_company_by_id(company=company_id)
        if not company:
            return 'company not found'

        users = self.repository.get_users(company=company_id)
        for user in users:
            self.repository.delete_by_id(user.id)

        self.company_repository.delete_by_id(company.id)

        await company.remove_data_dir()

        # TODO send event, 删除病例和切片记录
        return ''

    def save_ai_threshold(
            self, company_id: str, ai_type: AIType, threshold_range: int, threshold_value: float, all_use: bool):
        company = self.company_repository.get_company_by_id(company=company_id)

        ai_threshold = company.ai_threshold
        ai_threshold[ai_type.value].update({
            'threshold_range': threshold_range,
            'threshold_value': threshold_value,
            'all_use': all_use
        })
        company.update_data(ai_threshold=ai_threshold)
        if self.company_repository.save(company):
            if all_use:
                event = CompanyAIThresholdUpdatedEvent(threshold_range=threshold_range, threshold_value=threshold_value)
                pub.sendMessage(event.event_name, event=event)
            return True
        return False
