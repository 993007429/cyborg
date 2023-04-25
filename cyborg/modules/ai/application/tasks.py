from cyborg.app.request_context import request_context
from cyborg.celery.app import celery_task


@celery_task
def run_ai_task(task_id: int):
    from cyborg.app.service_factory import AppServiceFactory
    with request_context:
        AppServiceFactory.ai_service.run_ai_task(task_id=task_id)
