import logging
import os
import time
from typing import List

from cyborg.app.request_context import request_context
from cyborg.celery.app import app
from cyborg.infra.oss import oss
from cyborg.modules.ai.application.services import AIService
from cyborg.modules.partner.roche.application import tasks
from cyborg.modules.partner.roche.application.response import RocheAppResponse
from cyborg.modules.partner.roche.domain.services import RocheDomainService
from cyborg.modules.partner.roche.domain.value_objects import RocheAITaskStatus, RocheALGResult
from cyborg.modules.slice_analysis.application.services import SliceAnalysisService
from cyborg.seedwork.application.responses import AppResponse

logger = logging.getLogger(__name__)


class RocheService(object):

    def __init__(self, domain_service: RocheDomainService, ai_service: AIService, analysis_service: SliceAnalysisService):
        super(RocheService, self).__init__()
        self.domain_service = domain_service
        self.ai_service = ai_service
        self.analysis_service = analysis_service

    def get_algorithm_detail(self, algorithm_id: str) -> AppResponse:
        algorithm = self.domain_service.repository.get_algorithm(algorithm_id)
        return RocheAppResponse(data=algorithm.to_dict() if algorithm else None)

    def get_algorithm_list(self):
        algorithms = self.domain_service.repository.get_algorithms()
        return RocheAppResponse(data=[algo.to_dict() for algo in algorithms])

    def start_ai(self, algorithm_id: str, slide_url: str, input_info: dict) -> AppResponse:

        if not algorithm_id:
            return RocheAppResponse(rr_code=1, message='缺少参数')

        task = self.domain_service.create_ai_task(
            algorithm_id,
            slide_url,
            input_info
        )

        if task:
            result = tasks.run_ai_task(task.analysis_id)
            if result:
                self.domain_service.update_ai_task(task, result_id=result.id)

        return RocheAppResponse(data={'analysis_id': task.analysis_id} if task else None)

    def start_secondary_ai(self, analysis_id: str, regions: List[dict]) -> AppResponse:

        task = self.domain_service.repository.get_ai_task_by_analysis_id(analysis_id) if analysis_id else None
        if not task:
            return RocheAppResponse(err_code=1, message='参数错误')

        input_info = task.input_info or {}
        input_info['regions'] = regions

        secondary_task = self.domain_service.create_ai_task(
            task.algorithm_id,
            task.slide_url,
            task.input_info
        )

        result = tasks.run_ai_task(secondary_task.analysis_id)
        if result:
            self.domain_service.update_ai_task(secondary_task, result_id=result.id)

        return RocheAppResponse(data={'analysis_id': secondary_task.analysis_id} if task else None)

    def rescore(self, analysis_id: str, regions: List[dict]) -> RocheAppResponse:
        task = self.domain_service.repository.get_ai_task_by_analysis_id(analysis_id) if analysis_id else None
        if not task:
            return RocheAppResponse(err_code=1, message='参数错误')

        result = {}
        aggregate_result = self.domain_service.start_analysis(task, is_rescore=True)
        result['aggregate_score'] = aggregate_result.ai_results
        roi_scores = result.setdefault('roi_scores', [])
        for region in regions:
            region_result = self.domain_service.start_analysis(task, region=region, is_rescore=True)
            if region_result.ai_results:
                logger.info(region_result.ai_results)
                ai_results = region_result.ai_results
                ai_results['roi_id'] = region['roi_id']
                roi_scores.append(ai_results)

        return RocheAppResponse(data=result)

    def run_ai_task(self, analysis_id: str) -> RocheAppResponse:
        logger.info('run ai task')
        start_time = time.time()
        task = self.domain_service.repository.get_ai_task_by_analysis_id(analysis_id)
        if not task:
            return RocheAppResponse(err_code=1, message='任务不存在')
        if task.is_finished:
            return RocheAppResponse()

        if not task.slide_url:
            return RocheAppResponse(err_code=2, message='找不到切片信息')

        import torch.cuda
        if torch.cuda.is_available():
            gpu_list = []
            while not gpu_list:
                gpu_list = self.ai_service.check_available_gpu(task.ai_type, task.slide_path).data
                if gpu_list:
                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
                else:
                    logger.info('显卡忙或者不可用, 等待5s...')
                    time.sleep(5)

        self.domain_service.update_ai_task(task, status=RocheAITaskStatus.in_progress)

        ai_type = task.ai_type
        request_context.ai_type = ai_type

        try:
            result = self.domain_service.start_analysis(task)
        except Exception as e:
            logger.exception(e)
            result = RocheALGResult(err_msg='run alg error')

        if result.err_msg:
            self.domain_service.update_ai_task(task, status=RocheAITaskStatus.failed, err_msg=result.err_msg)
            return RocheAppResponse(message=result.err_msg)

        saved = self.domain_service.save_ai_result(task=task, result=result)
        if not saved:
            return RocheAppResponse(message='gen result file failed')

        self.domain_service.update_ai_task(task, status=RocheAITaskStatus.completed)

        alg_time = time.time() - start_time
        logger.info(f'任务{task.id} - caseid: {task.case_id} - fileid: {task.file_id} 计算完成,耗时{alg_time}')

        return RocheAppResponse(message='succeed')

    def get_task_result(self, analysis_id: str) -> RocheAppResponse:
        task = self.domain_service.repository.get_ai_task_by_analysis_id(analysis_id)
        if not task:
            return RocheAppResponse(err_code=1, message='ai task not found')

        if task.status in (RocheAITaskStatus.completed, ):
            result_file = oss.generate_sign_url(method='GET', key=task.result_file_key, expire_in=24 * 3600)
            return RocheAppResponse(data=[{
                'analysis_id': analysis_id,
                'result_file': result_file,
                'results': task.ai_results
            }])
        else:
            return RocheAppResponse(err_code=1, message='暂无结果')

    def cancel_task(self, analysis_id: str) -> RocheAppResponse:
        task = self.domain_service.repository.get_ai_task_by_analysis_id(analysis_id)
        if not task:
            return RocheAppResponse(err_code=1, message='ai task not found')

        app.control.revoke(task.result_id, terminate=True)

        if task.status not in (RocheAITaskStatus.completed, RocheAITaskStatus.failed, RocheAITaskStatus.cancelled):
            self.domain_service.update_ai_task(task, status=RocheAITaskStatus.cancelled)

        return RocheAppResponse(data=task.to_dict())

    def get_task_status(self, analysis_id) -> RocheAppResponse:
        task = self.domain_service.repository.get_ai_task_by_analysis_id(analysis_id)
        if not task:
            return RocheAppResponse(err_code=1, message='ai task not found')

        return RocheAppResponse(data=task.to_dict())

    def close_task(self, analysis_id: str) -> RocheAppResponse:
        task = self.domain_service.repository.get_ai_task_by_analysis_id(analysis_id)
        if not task:
            return RocheAppResponse(err_code=1, message='ai task not found')

        self.domain_service.update_ai_task(task, status=RocheAITaskStatus.closed)

        return RocheAppResponse(data=task.to_dict())
