from typing import Optional

from cyborg.app.request_context import request_context
from cyborg.modules.openapi.authentication.application.responses import UnregisteredClientResponse, \
    ParamsSignatureErrorResponse
from cyborg.modules.openapi.authentication.domain.services import OpenAPIAuthDomainService
from cyborg.seedwork.application.responses import AppResponse


class OpenAPIAuthService(object):

    def __init__(self, domain_service: OpenAPIAuthDomainService):
        self.domain_service = domain_service

    def get_client_by_app_name(self, app_name: str) -> Optional[dict]:
        client = self.domain_service.get_client_by_app_name(app_name)
        return client.to_dict() if client else None

    def check_params_signature(
            self, *,
            sign: str,
            params: dict,
    ) -> Optional[AppResponse]:
        if not request_context.openapi_client:
            return UnregisteredClientResponse()

        if not self.domain_service.check_signature(
                sign,
                access_key=request_context.openapi_client.access_key,
                secret_key=request_context.openapi_client.secret_key,
                params=params
        ):
            return ParamsSignatureErrorResponse(message='Invalid Signature')

        return None
