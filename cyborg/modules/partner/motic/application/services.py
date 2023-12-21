import logging
import os

from cyborg.app.auth import LoginUser
from cyborg.app.request_context import request_context
from cyborg.app.settings import Settings
from cyborg.celery.app import app
from cyborg.infra.cache import cache
from cyborg.modules.ai.application.services import AIService
from cyborg.modules.partner.motic.domain.services import MoticDomainService
from cyborg.modules.partner.motic.application import tasks as motic_tasks
from cyborg.modules.slice.application.services import SliceService
from cyborg.modules.user_center.user_core.application.services import UserCoreService
from cyborg.seedwork.application.responses import AppResponse
from cyborg.seedwork.domain.value_objects import AIType
from cyborg.utils.strings import dict_snake_to_camel, dict_camel_to_snake

logger = logging.getLogger(__name__)


class MoticService(object):

    celery_task_id_cache_key = 'motic_task_id:{}:celery_task_id'

    case_id_file_id_cache_key = 'motic_task_id:{}:case_id_file_id'

    def __init__(
            self, domain_service: MoticDomainService, ai_service: AIService, slice_service: SliceService,
            user_service: UserCoreService
    ):
        super(MoticService, self).__init__()
        self.domain_service = domain_service
        self.user_service = user_service
        self.ai_service = ai_service
        self.slice_service = slice_service

    def start_analysis_async(self, motic_task_id: str, ai_type: AIType) -> AppResponse:
        celery_task_id = motic_tasks.start_analysis(motic_task_id, ai_type)
        if celery_task_id:
            cache.set(self.celery_task_id_cache_key.format(motic_task_id), celery_task_id, ex=3600)
        return AppResponse()

    def start_analysis(self, motic_task_id: str, ai_type: AIType) -> AppResponse:
        if not ai_type:
            return AppResponse(err_code=1, message='参数错误')

        ai_name = {
            AIType.tct: 'tct1',
            AIType.lct: 'lct1',
        }.get(ai_type)

        rois = self.domain_service.get_task_rois(motic_task_id=motic_task_id)
        logger.info(motic_task_id)
        if not rois:
            return AppResponse(err_code=2, message='没有可分析的ROI')

        upload_id = motic_task_id

        tasks = []
        for roi in rois:
            file_id = roi.slideId
            record_info = self.slice_service.get_record_by_upload_id(motic_task_id).data
            if record_info:
                case_id = record_info['caseid']
            else:
                slide_path = os.path.join(
                    request_context.current_user.data_dir, 'upload_data', upload_id, 'slices', file_id)
                if not os.path.exists(slide_path):
                    os.makedirs(slide_path)

                file_name, file_path, file_size = self.domain_service.download_slide(
                    motic_task_id=motic_task_id, roi=roi, slide_path=slide_path)

                logger.info(f'>>>>>>{file_name}>>>>>>>>{file_path}>>>>>>>>{file_size}')

                res = self.slice_service.upload_slice(
                    upload_id=upload_id, case_id='', file_id=str(roi.slideId),
                    company_id=request_context.current_company,
                    file_name=file_name, slide_type='slices', upload_path=slide_path,
                    total_upload_size=file_size, tool_type=ai_name,
                    user_file_path='', sample_num=motic_task_id, cover_slice_number=True, create_record=True
                )
                if res.err_code:
                    logger.info(res.err_code)
                    return res

                slice_info = res.data
                case_id = slice_info['caseid']

            request_context.case_id = case_id
            request_context.file_id = file_id
            cache.set(self.case_id_file_id_cache_key.format(motic_task_id), (case_id, file_id, ai_name), ex=3600)

            task = self.ai_service.start_ai(ai_name=ai_name, run_task_async=False).data

            self.domain_service.callback_analysis_status(motic_task_id, task['status'])

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
        res = self.slice_service.get_slice_info(
            case_id=request_context.case_id, file_id=request_context.file_id, company_id=request_context.current_company
        )
        if res.err_code:
            return res

        slice_info = res.data
        data = {
            'aiSuggest': slice_info['ai_suggest'],
            'checkResult': slice_info['check_result'],
            'slideQuality': slice_info['slide_quality'],
            'cellNum': slice_info['cell_num']
        }

        user_info = self.user_service.get_current_user(user_name=request_context.current_user.username).data
        user_info['cloud'] = Settings.CLOUD
        if user_info:
            login_user = LoginUser.from_dict(dict_camel_to_snake(user_info))
            data['url'] = f'/#/detail?caseid={request_context.case_id}&jwt={login_user.jwt_token}'
        return AppResponse(data=data)

    def cancel_analysis(self, motic_task_id: str) -> AppResponse:
        celery_task_id = cache.get(self.celery_task_id_cache_key.format(motic_task_id))
        if celery_task_id:
            app.control.revoke(celery_task_id, terminate=True)

        res = cache.get(self.case_id_file_id_cache_key.format(motic_task_id))
        if res:
            request_context.case_id = res[0]
            request_context.file_id = res[1]
            request_context.ai_type = AIType.get_by_value(res[2])
            return self.ai_service.cancel_task()
        return AppResponse()
