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
            token: str, method: str, content_type: str, gmt_date: str,
            query_string: str, request_body: str, user_id: str = ''
    ) -> Optional[AppResponse]:
        if not request_context.openapi_client:
            return UnregisteredClientResponse()

        if not self.domain_service.check_gmt_date(gmt_date):
            return ParamsSignatureErrorResponse(message='Request Expired.')

        if not self.domain_service.check_signature(
                token,
                access_key=request_context.openapi_client.access_key,
                secret_key=request_context.openapi_client.secret_key,
                app_name=request_context.openapi_client.app_name,
                method=method, content_type=content_type, gmt_date=gmt_date,
                query_string=query_string, request_body=request_body, user_id=user_id
        ):
            return ParamsSignatureErrorResponse(message='Invalid Signature')

        return None
