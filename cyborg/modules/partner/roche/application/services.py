from cyborg.modules.ai.application.services import AIService
from cyborg.modules.partner.roche.application import tasks
from cyborg.modules.partner.roche.application.response import RocheAppResponse
from cyborg.modules.partner.roche.domain.services import RocheDomainService
from cyborg.seedwork.application.responses import AppResponse


class RocheService(object):

    def __init__(self, domain_service: RocheDomainService, ai_service: AIService):
        super(RocheService, self).__init__()
        self.domain_service = domain_service
        self.ai_service = ai_service

    def get_algorithm_detail(self, algorithm_id: str) -> AppResponse:
        algorithm = self.domain_service.repository.get_algorithm(algorithm_id)
        return RocheAppResponse(data=algorithm.to_dict() if algorithm else None)

    def get_algorithm_list(self):
        algorithms = self.domain_service.repository.get_algorithms()
        return RocheAppResponse(data=[algo.to_dict() for algo in algorithms])

    def start_ai(self, algorithm_id: str, slide_url: str) -> AppResponse:

        task_params = {
            'rois': None,
        }

        task = self.domain_service.create_ai_task(
            algorithm_id,
            slide_url,
            **task_params
        )

        if task:
            result = tasks.run_ai_task(task.id)
            if result:
                self.domain_service.update_ai_task(task, result_id=result.id)

        return RocheAppResponse(data=task.to_dict() if task else None)

    def run_ai_task(self, task_id) -> AppResponse:
        """
        start_time = time.time()
        task = self.domain_service.repository.get_ai_task_by_id(task_id)
        if not task:
            return AppResponse(err_code=1, message='任务不存在')
        if task.is_finished:
            return AppResponse()

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
                gpu_list = self.domain_service.check_available_gpu(task)
                if gpu_list:
                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
                else:
                    logger.info('显卡忙或者不可用, 等待5s...')
                    time.sleep(5)

        self.slice_service.update_ai_status(status=SliceStartedStatus.analyzing)
        self.domain_service.update_ai_task(task, status=AITaskStatus.analyzing)

        ai_type = task.ai_type
        request_context.ai_type = ai_type

        self.analysis_service.clear_ai_result()

        groups = self.analysis_service.get_mark_groups(template_id=task.template_id).data
        group_name_to_id = {group['label']: int(group['id']) for group in groups}

        has_manual_roi = task.rois and task.rois[0]['x']

        try:
            if ai_type == AIType.tct:
                result = self.domain_service.run_tct(task)
            elif ai_type == AIType.lct:
                result = self.domain_service.run_lct(task)
            elif ai_type == AIType.dna:
                result = self.domain_service.run_tbs_dna(task)
            elif ai_type == AIType.her2:
                result = self.domain_service.run_her2(task, group_name_to_id)
            elif ai_type == AIType.ki67:
                result = self.domain_service.run_ki67(task, group_name_to_id, compute_wsi=not has_manual_roi)
            elif ai_type == AIType.ki67hot:
                result = self.domain_service.run_ki67(task, group_name_to_id, compute_wsi=False)
            elif ai_type == AIType.er:
                result = self.domain_service.run_ki67(task, group_name_to_id, compute_wsi=not has_manual_roi)
            elif ai_type == AIType.pr:
                result = self.domain_service.run_ki67(task, group_name_to_id, compute_wsi=not has_manual_roi)
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

        if result.err_msg:
            self.domain_service.update_ai_task(task, status=AITaskStatus.failed)
            self.slice_service.finish_ai(status=SliceStartedStatus.failed)
            return AppResponse(message=result.err_msg)

        stats = self.domain_service.refresh_ai_statistics(
            is_error=bool(result.err_msg), ai_type=ai_type, ai_suggest=result.ai_suggest, slice_info=task.slice_info
        )

        if result.prob_dict:
            self.domain_service.save_prob(slice_id=task.slice_info['uid'], prob_info=result.prob_dict)

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

        alg_time = time.time() - start_time
        logger.info(f'任务{task.id} - caseid: {task.case_id} - fileid: {task.file_id} 计算完成,耗时{alg_time}')

        return AppResponse(message='succeed')
        """
        return RocheAppResponse()
