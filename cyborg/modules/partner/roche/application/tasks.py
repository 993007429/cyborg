from cyborg.celery.app import celery_task


@celery_task
def run_ai_task(analysis_id: str):
    from cyborg.app.service_factory import RocheAppServiceFactory
    res = RocheAppServiceFactory.roche_service.run_ai_task(analysis_id=analysis_id)
    return res
