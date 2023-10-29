import logging

from cyborg.celery.app import celery_task

logger = logging.getLogger(__name__)


@celery_task
def run_ai_task(task_id: int):
    from cyborg.app.service_factory import AppServiceFactory
    res = AppServiceFactory.ai_service.run_ai_task(task_id=task_id)
    if res.err_code:
        logger.info(f'任务{task_id}失败, msg: {res.message}')
    return res
