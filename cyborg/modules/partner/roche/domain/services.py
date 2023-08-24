import logging

from cyborg.modules.ai.application.services import AIService

logger = logging.getLogger(__name__)


class RocheDomainService(object):

    def __init__(self, ai_service: AIService):
        super(RocheDomainService, self).__init__()
        self.ai_service = ai_service
