import json
from typing import List, Optional

from werkzeug.datastructures import FileStorage

from cyborg.app.request_context import request_context
from cyborg.modules.user_center.user_core.domain.services import UserCoreDomainService
from cyborg.seedwork.application.responses import AppResponse


class UserCoreService(object):

    def __init__(self, domain_service: UserCoreDomainService):
        super(UserCoreService, self).__init__()
        self.domain_service = domain_service

    def get_current_user(self) -> AppResponse[dict]:
        user = self.domain_service.get_user_by_name(
            request_context.current_user.username,
            request_context.current_company
        )
        return AppResponse(data=user.to_dict() if user else None)

    def check_user(self, username: str, company: str) -> AppResponse[bool]:
        user = self.domain_service.get_user_by_name(username, company)
        return AppResponse(data=bool(user))

    def login(self, username: str, password: str, company: str, client_ip: str) -> AppResponse:
        if username.endswith(' ') or company.endswith(' '):
            return AppResponse(err_code=3, message='wrong password')

        err_code, message, user = self.domain_service.login(
            username=username, password=password, company_id=company, client_ip=client_ip)
        if err_code:
            return AppResponse(err_code=err_code, message=message)

        return AppResponse(data=user.to_dict())

    def is_signed(self) -> AppResponse:
        user = self.domain_service.get_user_by_name(
            request_context.current_user.username,
            request_context.current_company
        )
        return AppResponse(data=user.is_signed)

    def update_signed(self, username: str, company: str, file: FileStorage):
        user = self.domain_service.get_user_by_name(username, company)
        updated = self.domain_service.update_signed(user) if user else False
        if updated:
            file.save(user.sign_image_path)
            return AppResponse()

        return AppResponse(message='更新失败')

    def update_password(self, old_password: str, new_password: str) -> AppResponse:
        user = self.domain_service.get_user_by_name(
            request_context.current_user.username, request_context.current_company)
        if not user:
            return AppResponse(err_code=3, message='user is not exist')

        err_code, message = self.domain_service.update_password(
            user, old_password=old_password, new_password=new_password)

        return AppResponse(err_code=err_code, message=message)

    def get_customized_record_fields(self) -> AppResponse[List[str]]:
        data = self.domain_service.get_customized_record_fields(company_id=request_context.current_company)
        return AppResponse(data=data)

    def set_customized_record_fields(self, record_fields: str) -> AppResponse[str]:
        if self.domain_service.set_customized_record_fields(
                company_id=request_context.current_company, record_fields=record_fields):
            return AppResponse(data='设置成功')
        else:
            return AppResponse(message='设置失败')

    def update_report_settings(
            self, report_name: Optional[str] = None, report_info: Optional[str] = None
    ) -> AppResponse[bool]:
        company = self.domain_service.company_repository.get_company_by_id(request_context.current_company)
        updated = self.domain_service.update_report_settings(company, report_name=report_name, report_info=report_info)
        return AppResponse(data=updated)

    def get_company_info(self, company_id: str) -> AppResponse[dict]:
        company = self.domain_service.company_repository.get_company_by_id(company_id)
        return AppResponse(data=company.to_dict() if company else None)

    def get_company_storage_info(self) -> AppResponse:
        data = self.domain_service.get_company_storage_info(company_id=request_context.current_company)
        return AppResponse(data=data)

    def update_company_storage_usage(self, company_id: str, increased_size: float) -> AppResponse[bool]:
        success = self.domain_service.update_company_storage_usage(company_id=company_id, increased_size=increased_size)
        return AppResponse(data=success)

    def update_company_trial(self, ai_name: str) -> AppResponse:
        err_code, message, company = self.domain_service.update_company_trial(
            company_id=request_context.current_company, ai_name=ai_name)
        return AppResponse(err_code=err_code, message=message, data=company.to_dict() if company else None)

    def get_company_trail_info(self) -> AppResponse:
        company = self.domain_service.company_repository.get_company_by_id(request_context.current_company)
        info = {
            'model_lis': json.dumps(company.model_lis),
            'onTrial': company.on_trial,
            'trialTimes': json.dumps(company.trial_times)
        } if company else {}
        return AppResponse(data=info)

    def get_company_label(self) -> AppResponse:
        company = self.domain_service.company_repository.get_company_by_id(request_context.current_company)
        label = company.label if company else ''
        return AppResponse(data=label)

    def update_company_label(self, label: int) -> AppResponse:
        company = self.domain_service.company_repository.get_company_by_id(request_context.current_company)
        if company:
            self.domain_service.update_company_label(company, label)
        return AppResponse()

    def update_company_ai_threshold(
            self, model_name: Optional[str], threshold_value: Optional[float] = None,
            use_default_threshold: bool = False
    ) -> AppResponse:
        company = self.domain_service.company_repository.get_company_by_id(request_context.current_company)
        if company:
            self.domain_service.update_company_ai_threshold(company, model_name, threshold_value, use_default_threshold)
        return AppResponse()

    ####################################
    # for admin below
    ####################################

    def get_users_by_company(self, company: str) -> AppResponse:
        users = self.domain_service.repository.get_users(company=company)
        return AppResponse(data=[user.to_dict() for user in users])

    def get_user(self, username: str, company: str) -> AppResponse:
        user = self.domain_service.repository.get_user_by_name(username=username, company=company)
        if not user:
            return AppResponse(err_code=1, message='no such user')
        return AppResponse(data=user.to_dict())

    def create_user(self, username: str, password: str, company_id: str, role: str) -> AppResponse:
        err_msg, new_user = self.domain_service.create_user(username, password, company_id, role)
        if err_msg:
            return AppResponse(err_code=1, message=err_msg, data={'status': -1})
        return AppResponse(data={'status': 1, 'user': new_user.to_dict()})

    def update_user(self, user_id: int, username: str, password: str, role: str):
        err_msg = self.domain_service.update_user(user_id=user_id, username=username, password=password, role=role)
        if err_msg:
            return AppResponse(err_code=1, message=err_msg)

        return AppResponse(message='update success', data={'message': '修改成功', 'status': 1})

    def delete_user(self, username: str, company: str):
        err_msg = self.domain_service.delete_user(username, company)
        if err_msg:
            return AppResponse(err_code=1, message=err_msg)

        return AppResponse(message='delete success', data={'status': 1})

    def get_all_comanies(self) -> AppResponse:
        companies = self.domain_service.company_repository.get_all_companies()
        return AppResponse(data=[company.to_dict() for company in companies])

    def get_company_detail(self, company_id: str) -> AppResponse:
        company = self.domain_service.company_repository.get_company_by_id(company=company_id)
        if not company:
            return AppResponse(err_code=1, message='no such company')
        return AppResponse(data=company.to_dict())

    def create_company(self, **kwargs) -> AppResponse:
        err_msg = self.domain_service.create_company(**kwargs)
        if err_msg:
            return AppResponse(err_code=1, message=err_msg, data={'status': -1})

        return AppResponse(data={'status': 1})

    def update_company(self, **kwargs) -> AppResponse:
        err_msg = self.domain_service.update_company(**kwargs)
        if err_msg:
            return AppResponse(err_code=1, message=err_msg, data={'status': -1})

        # TODO send event, 修改文件夹

        return AppResponse(data={'status': 1})

    async def delete_company(self, company_id: str):
        err_msg = await self.domain_service.delete_company(company_id)
        if err_msg:
            return AppResponse(err_code=1, message=err_msg, data={'status': -1})
        return AppResponse(data={'status': 1})

    def save_ai_threshold(self, threshold_range: int, threshold_value: float, all_use: bool):
        saved = self.domain_service.save_ai_threshold(
            company_id=request_context.current_company, ai_type=request_context.ai_type,
            threshold_range=threshold_range, threshold_value=threshold_value, all_use=all_use
        )
        if not saved:
            return AppResponse(err_code=1, message='modify ai threshold failed')
        return AppResponse(message='modify ai threshold succeed')

    def get_ai_threshold(self):
        company = self.domain_service.company_repository.get_company_by_id(company=request_context.current_company)
        ai_threshold = company.ai_threshold if company else {}
        return AppResponse(message='query succeed', data=ai_threshold.get(request_context.ai_type.value))

    def get_default_ai_threshold(self):
        company = self.domain_service.company_repository.get_company_by_id(company=request_context.current_company)
        default_ai_threshold = company.default_ai_threshold if company else {}
        return AppResponse(message='query succeed', data=default_ai_threshold.get(request_context.ai_type.value, 0.5))
