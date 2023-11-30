import datetime
import logging
import os
import sys
import time
from typing import List, Optional

from cyborg.app.request_context import request_context
from cyborg.celery.app import app
from cyborg.modules.ai.application import tasks
from cyborg.modules.ai.domain.services import AIDomainService
from cyborg.modules.ai.domain.value_objects import ALGResult, AITaskStatus
from cyborg.modules.slice.application.services import SliceService
from cyborg.modules.slice.domain.value_objects import SliceStartedStatus
from cyborg.modules.slice_analysis.application.services import SliceAnalysisService
from cyborg.modules.user_center.user_core.application.services import UserCoreService
from cyborg.seedwork.application.responses import AppResponse
from cyborg.seedwork.domain.value_objects import AIType


logger = logging.getLogger(__name__)


class AIService(object):

    def __init__(
            self, domain_service: AIDomainService, user_service: UserCoreService,
            slice_service: SliceService, analysis_service: SliceAnalysisService
    ):
        super(AIService, self).__init__()
        self.domain_service = domain_service
        self.user_service = user_service
        self.slice_service = slice_service
        self.analysis_service = analysis_service

    def check_available_gpu(self, ai_type: AIType, slide_path: str) -> AppResponse:
        gpu_list = self.domain_service.check_available_gpu(ai_type, slide_path)
        return AppResponse(data=gpu_list)

    def start_ai(
            self, ai_name: str, rois: Optional[list] = None, upload_batch_number: Optional[str] = None,
            ip_address: Optional[str] = None, is_calibrate: bool = False, run_task_async: bool = True) -> AppResponse:
        ai_type = AIType.get_by_value(ai_name)
        request_context.ai_type = ai_type

        task = self.domain_service.repository.get_latest_ai_task(
            case_id=request_context.case_id, file_id=request_context.file_id, ai_type=request_context.ai_type)
        if task and not task.is_finished and task.result_id:
            app.control.revoke(task.result_id, terminate=True)
            self.domain_service.update_ai_task(task, status=AITaskStatus.canceled)
        res = self.user_service.update_company_trial(ai_name=ai_name)
        if res.err_code:
            return res

        task_params = {
            'rois': rois,
            'is_calibrate': is_calibrate
        }
        company_info = res.data

        model_type = ai_name.strip(ai_type.value)
        is_tld = ai_type in [AIType.tct, AIType.lct, AIType.dna, AIType.dna_ploidy]

        if is_calibrate:
            task_params.update({'model_info': {
                'ai_threshold': None if is_tld else 0,
                'model_type': model_type if is_tld else None
            }})
        elif is_tld:
            logger.info(company_info)
            ai_threshold = company_info['aiThreshold'] or company_info['defaultAiThreshold']
            threshold_value = (ai_threshold[ai_type.value] or {}).get('threshold_value')
            model_name = (ai_threshold[ai_type.value] or {}).get('model_name')
            task_params.update({'model_info': {
                'ai_threshold': threshold_value,
                'model_type': model_type,
                'model_name': model_name
            }})

        template_id = self.domain_service.repository.get_template_id_by_ai_name(ai_name=ai_name)
        if template_id:
            task_params['template_id'] = template_id

        res = self.slice_service.update_ai_status(
            status=SliceStartedStatus.default, ai_name=ai_name, upload_batch_number=upload_batch_number,
            template_id=template_id, ip_address=ip_address)
        if res.err_code:
            res.err_code = 3
            return res

        slice_info = res.data

        task = self.domain_service.create_ai_task(
            ai_type,
            slice_info['caseid'],
            slice_info['fileid'],
            **task_params
        )

        if task:
            if run_task_async:
                result = tasks.run_ai_task(task.id)
                if result:
                    self.domain_service.update_ai_task(task, result_id=result.id)
            else:
                return self.run_ai_task(task.id)

        return AppResponse(data=task.to_dict() if task else None)

    def run_ai_task(self, task_id):
        task = self.domain_service.repository.get_ai_task_by_id(task_id)
        if not task:
            return AppResponse(err_code=1, message='任务不存在')
        if task.is_finished:
            return AppResponse()

        logger.info(f'收到任务{task.id} - caseid: {task.case_id} - fileid: {task.file_id}')

        request_context.case_id = task.case_id
        request_context.file_id = task.file_id
        task.slice_info = self.slice_service.get_slice_info(
            case_id=request_context.case_id, file_id=request_context.file_id).data
        if not task.slice_info:
            self.domain_service.update_ai_task(task, status=AITaskStatus.failed)
            return AppResponse(err_code=2, message='找不到切片信息')

        import torch.cuda
        if torch.cuda.is_available():
            gpu_list = []
            while not gpu_list:
                gpu_list = self.domain_service.check_available_gpu(task.ai_type, task.slide_path)
                if gpu_list and self.domain_service.mark_ai_task_running(ai_task=task):
                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
                else:
                    gpu_list = []
                    logger.info('显卡忙或者不可用, 等待5s...')
                    time.sleep(5)

        start_time = time.time()

        self.domain_service.update_ai_task(task, status=AITaskStatus.analyzing)
        self.slice_service.update_ai_status(status=SliceStartedStatus.analyzing)

        ai_type = task.ai_type
        request_context.ai_type = ai_type

        self.analysis_service.clear_ai_result()

        groups = self.analysis_service.get_mark_groups(template_id=task.template_id).data
        group_name_to_id = {group['label']: int(group['id']) for group in groups}

        try:
            if ai_type == AIType.tct:
                result = self.domain_service.run_tct(task)
            elif ai_type == AIType.lct:
                result = self.domain_service.run_lct(task)
            elif ai_type == AIType.dna:
                result = self.domain_service.run_tbs_dna(task)
            elif ai_type == AIType.dna_ploidy:
                result = self.domain_service.run_dna_ploidy(task)
            elif ai_type == AIType.her2:
                result = self.domain_service.run_her2(task, group_name_to_id)
            elif ai_type == AIType.ki67:
                result = self.domain_service.run_ki67_new(task, group_name_to_id)
            elif ai_type == AIType.ki67hot:
                result = self.domain_service.run_ki67(task, group_name_to_id, compute_wsi=False)
            elif ai_type == AIType.er:
                result = self.domain_service.run_ki67_new(task, group_name_to_id)
            elif ai_type == AIType.pr:
                result = self.domain_service.run_ki67_new(task, group_name_to_id)
            elif ai_type == AIType.fish_tissue:
                result = self.domain_service.run_fish_tissue(task, group_name_to_id)
            elif ai_type == AIType.pdl1:
                result = self.domain_service.run_pdl1(task, group_name_to_id, request_context.current_user.data_dir)
            elif ai_type == AIType.np:
                result = self.domain_service.run_np(task, group_name_to_id)
            elif ai_type == AIType.bm:
                result = self.domain_service.run_bm(task, group_name_to_id)
            else:
                logger.error(f'{ai_type} does not support')
                result = ALGResult(ai_suggest='', err_msg=f'{ai_type} does not support')
        except Exception as e:
            logger.exception(e)
            result = ALGResult(ai_suggest='', err_msg='run alg error')
        finally:
            self.domain_service.unmark_ai_task_running(ai_task=task)

        if result.err_msg:
            self.domain_service.update_ai_task(task, status=AITaskStatus.failed)
            self.slice_service.finish_ai(status=SliceStartedStatus.failed)
            return AppResponse(message=result.err_msg)

        alg_time = time.time() - start_time
        logger.info(f'任务{task.id} - caseid: {task.case_id} - fileid: {task.file_id} 算法部分完成,耗时{alg_time}')

        stats = self.domain_service.refresh_ai_statistics(
            is_error=bool(result.err_msg), ai_type=ai_type, ai_suggest=result.ai_suggest, slice_info=task.slice_info
        )

        if result.prob_dict:
            self.domain_service.save_prob(slice_id=task.slice_info['uid'], prob_info=result.prob_dict)

        self.analysis_service.clear_ai_result()
        self.analysis_service.create_ai_marks(
            cell_marks=[mark.to_dict() for mark in result.cell_marks],
            roi_marks=[mark.to_dict() for mark in result.roi_marks],
            skip_mark_to_tile=ai_type in [AIType.bm]
        )

        self.domain_service.update_ai_task(task, status=AITaskStatus.success)

        res = self.slice_service.finish_ai(
            status=SliceStartedStatus.failed if result.err_msg else SliceStartedStatus.success,
            ai_suggest=result.ai_suggest,
            slide_quality=result.slide_quality,
            cell_num=result.cell_num,
            as_id=stats.id if stats else None
        )
        if res.err_code:
            return res

        if ai_type in [AIType.model_calibrate_tct]:
            self.user_service.update_company_ai_threshold(
                model_name=result.get('model_name'),
                threshold_value=result.get('aiThreshold')
            )

        total_time = time.time() - start_time
        logger.info(f'任务{task.id} - caseid: {task.case_id} - fileid: {task.file_id} 全部完成,耗时{total_time}')

        return AppResponse(data=task.to_dict() if task else None)

    def batch_start_ai(self, ai_name: str, case_ids: List[int]) -> AppResponse:

        ai_type = AIType.get_by_value(ai_name)
        request_context.ai_type = ai_type

        slices = self.slice_service.get_slices(case_ids=case_ids, per_page=sys.maxsize).data

        failed = 0
        data = []
        for slice_info in slices:
            request_context.case_id = slice_info['caseid']
            request_context.file_id = slice_info['fileid']
            res = self.start_ai(ai_name=ai_name)
            if res.err_code == 1:
                return res
            elif res.err_code:
                failed += 1
            else:
                data.append(res.data)

        return AppResponse(
            message='操作成功', data=data)

    def do_model_calibration(self, ai_name: str, case_ids: List[int]) -> AppResponse:

        slices = self.slice_service.get_slices(case_ids=case_ids, per_page=5).data

        data = []
        for slice_info in slices:
            request_context.case_id = slice_info['caseid']
            request_context.file_id = slice_info['fileid']
            res = self.start_ai(ai_name=ai_name, is_calibrate=True)
            data.append(res.data)
        return AppResponse(message='操作成功', data=data)

    def get_ai_task_result(self, task_id: Optional[int] = None) -> AppResponse:
        err_msg, result = self.domain_service.get_ai_task_result(
            case_id=request_context.case_id, file_id=request_context.file_id, ai_type=request_context.ai_type,
            task_id=task_id
        )
        return AppResponse(err_code=1 if err_msg else 0, message=err_msg, data=result)

    def cancel_task(self) -> AppResponse:
        task = self.domain_service.repository.get_latest_ai_task(
            case_id=request_context.case_id, file_id=request_context.file_id, ai_type=request_context.ai_type)
        if not task:
            return AppResponse(err_code=1, message='ai task not found')

        if task.result_id:
            app.control.revoke(task.result_id, terminate=True)

        self.domain_service.unmark_ai_task_running(ai_task=task)

        if task.status not in (AITaskStatus.success, AITaskStatus.failed):
            self.domain_service.update_ai_task(task, status=AITaskStatus.failed)
            res = self.slice_service.finish_ai(
                status=SliceStartedStatus.default
            )
            if res.err_code:
                return res

        return AppResponse(data={'result': 'task is terminated successfully', 'status': 1})

    def kill_task_processes(self, pid: int) -> AppResponse:
        result = self.domain_service.kill_task_processes(pid=pid)
        return AppResponse(data=result)

    def cancel_calibration(self) -> AppResponse:
        task = self.domain_service.repository.get_latest_calibrate_ai_task()
        if not task:
            return AppResponse(err_code=1, message='calibrate ai task not found')

        if task.status in (AITaskStatus.success, AITaskStatus.failed):
            return AppResponse(err_code=2, message='任务已结束，无法取消')

        self.domain_service.update_ai_task(task, status=AITaskStatus.failed)
        self.slice_service.update_ai_status(status=SliceStartedStatus.default)

        return AppResponse()

    def get_analyze_threshold(self, threshold: float, analyse_mode: str) -> AppResponse:

        slices = self.slice_service.get_analyzed_slices().data

        data = self.domain_service.get_analyze_threshold(
            threshold=threshold, analyse_mode=analyse_mode, slices=slices)

        return AppResponse(data=data)

    def get_ai_statistics(self, start_date: Optional[str], end_date: Optional[str]) -> AppResponse:
        start_date = start_date or datetime.datetime.now().strftime('%Y-%m-%d')
        end_date = end_date or (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        data = self.domain_service.get_ai_statistics(
            ai_type=request_context.ai_type, company=request_context.current_company,
            start_date=start_date, end_date=end_date)
        return AppResponse(data=data)

    def maintain_ai_tasks(self) -> AppResponse:
        failed = self.domain_service.maintain_ai_tasks()
        for task in failed:
            request_context.case_id = task['case_id']
            request_context.file_id = task['file_id']
            request_context.company = None
            self.slice_service.update_ai_status(status=SliceStartedStatus.failed)
            app.control.revoke(task['result_id'], terminate=True)
        return AppResponse(data=failed)

    def purge_tasks(self) -> AppResponse:
        purged = self.domain_service.reset_running_tasks()
        app.control.purge()
        return AppResponse(data={'purged': purged})
