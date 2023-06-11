from pubsub import pub

from cyborg.app.service_factory import AppServiceFactory
from cyborg.modules.user_center.user_core.domain.event import CompanyAIThresholdUpdatedEvent


def handle_company_ai_threshold_updated_event(event: CompanyAIThresholdUpdatedEvent):
    AppServiceFactory.slice_service.apply_ai_threshold(
        threshold_range=event.threshold_range, threshold_value=event.threshold_value)


def subscribe_events():
    pub.subscribe(handle_company_ai_threshold_updated_event, CompanyAIThresholdUpdatedEvent.event_name)
