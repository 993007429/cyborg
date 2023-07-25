from cyborg.app.container import AppContainer
from cyborg.modules.ai.application.services import AIService
from cyborg.modules.openapi.authentication.application.services import OpenAPIAuthService
from cyborg.modules.slice.application.services import SliceService
from cyborg.modules.slice_analysis.application.services import SliceAnalysisService
from cyborg.modules.user_center.user_core.application.services import UserCoreService


class AppServiceFactory(object):

    user_service: UserCoreService = AppContainer.user_center.user_service()

    slice_service: SliceService = AppContainer.slice.slice_service()

    slice_analysis_service: SliceAnalysisService = AppContainer.slice_analysis.slice_analysis_service()

    ai_service: AIService = AppContainer.ai.ai_service()

    openapi_auth_service: OpenAPIAuthService = AppContainer.openapi.openapi_auth_service()
