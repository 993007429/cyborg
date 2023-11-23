import logging
import os

from cyborg.app.auth import LoginUser
from cyborg.app.request_context import request_context
from cyborg.app.settings import Settings
from cyborg.modules.ai.application.services import AIService
from cyborg.modules.partner.motic.domain.services import MoticDomainService
from cyborg.modules.slice.application.services import SliceService
from cyborg.modules.slice_analysis.application.services import SliceAnalysisService
from cyborg.modules.user_center.user_core.application.services import UserCoreService
from cyborg.seedwork.application.responses import AppResponse
from cyborg.seedwork.domain.value_objects import AIType
from cyborg.utils.strings import dict_snake_to_camel, dict_camel_to_snake

logger = logging.getLogger(__name__)


class MoticService(object):

    def __init__(
            self, domain_service: MoticDomainService, ai_service: AIService, slice_service: SliceService,
            analysis_service: SliceAnalysisService, user_service: UserCoreService
    ):
        super(MoticService, self).__init__()
        self.domain_service = domain_service
        self.user_service = user_service
        self.ai_service = ai_service
        self.slice_service = slice_service
        self.analysis_service = analysis_service

    def start_analysis(self, motic_task_id: str) -> AppResponse:
        if not request_context.ai_type:
            return AppResponse(err_code=1, message='参数错误')
        rois = self.domain_service.get_task_rois(motic_task_id=motic_task_id)
        if not rois:
            return AppResponse(err_code=2, message='没有可分析的ROI')

        upload_id = motic_task_id

        tasks = []
        for roi in rois:
            file_id = roi.slideId
            slice_info = self.slice_service.get_slice_info(
                file_id=file_id, company_id=request_context.current_company).data
            if not slice_info:
                slide_path = os.path.join(
                    request_context.current_user.data_dir, 'upload_data', upload_id, 'slices', file_id)
                if not os.path.exists(slide_path):
                    os.makedirs(slide_path)

                file_name, file_path, file_size = self.domain_service.download_slide(
                    motic_task_id=motic_task_id, roi=roi, slide_path=slide_path)
                tool_type = {
                    AIType.tct: 'tct1',
                    AIType.lct: 'lct1',
                }.get(request_context.ai_type)

                res = self.slice_service.upload_slice(
                    upload_id=upload_id, case_id='', file_id=str(roi.slideId), company_id=request_context.current_company,
                    file_name=file_name, slide_type='slices', upload_path=slide_path,
                    total_upload_size=file_size, tool_type=tool_type,
                    user_file_path='', cover_slice_number=True, create_record=True
                )
                if res.err_code:
                    return res

                slice_info = res.data

            request_context.case_id = slice_info['caseid']
            request_context.file_id = file_id
            task = self.ai_service.start_ai(ai_name=request_context.ai_type.value, run_task_async=False).data

            tasks.append(dict_snake_to_camel(task))

        return AppResponse(data=tasks)

    def _find_record(self, motic_task_id: str) -> bool:
        records = self.slice_service.get_records_by_sample_num(sample_num=motic_task_id).data
        if not records or not records[0].get('slices'):
            return False

        record_info = records[0]
        slice_info = record_info['slices'][0]
        request_context.case_id = record_info['caseid']
        request_context.file_id = slice_info['fileid']
        request_context.ai_type = AIType.get_by_value(slice_info['alg'])
        return True

    def get_analysis_status(self, motic_task_id: str) -> AppResponse:
        if not self._find_record(motic_task_id=motic_task_id):
            return AppResponse(err_code=1, message='找不到切片')
        return self.ai_service.get_ai_task_result()

    def get_analysis_result(self, motic_task_id: str) -> AppResponse:
        if not self._find_record(motic_task_id=motic_task_id):
            return AppResponse(err_code=1, message='找不到切片')
        res = self.analysis_service.get_rois()
        if res.err_code:
            return res
        rois = res.data.get('ROIS')

        data = {
            'rois': [dict_snake_to_camel(roi) for roi in rois]
        }

        user_info = self.user_service.get_current_user(user_name=request_context.current_user.username).data
        user_info['cloud'] = Settings.CLOUD
        if user_info:
            login_user = LoginUser.from_dict(dict_camel_to_snake(user_info))
            data['url'] = f'/#/detail?caseid={request_context.case_id}&jwt={login_user.jwt_token}'
        return AppResponse(data=data)

    def cancel_analysis(self, motic_task_id: str):
        if not self._find_record(motic_task_id=motic_task_id):
            return AppResponse(err_code=1, message='找不到切片')
        return self.ai_service.cancel_task()
