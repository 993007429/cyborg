from cyborg.celery.app import celery_task
from cyborg.seedwork.domain.value_objects import AIType


@celery_task
def start_analysis(motic_task_id: str, ai_type: AIType):
    from cyborg.app.service_factory import PartnerAppServiceFactory
    res = PartnerAppServiceFactory.motic_service.start_analysis(motic_task_id=motic_task_id, ai_type=ai_type)
    return res
