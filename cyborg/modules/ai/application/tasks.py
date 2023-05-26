from cyborg.celery.app import celery_task


@celery_task
def run_ai_task(task_id: int):
    from cyborg.app.service_factory import AppServiceFactory
    res = AppServiceFactory.ai_service.run_ai_task(task_id=task_id)
    return res
