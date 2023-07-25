from typing import Optional

from cyborg.modules.openapi.authentication.domain.entities import OpenAPIClientEntity
from cyborg.modules.openapi.authentication.domain.repositories import OpenAPIClientRepository
from cyborg.modules.openapi.authentication.utils.signature import SignatureUtil
from cyborg.utils.datetime import DatetimeUtil
from cyborg.utils.url import UrlUtil


class OpenAPIAuthDomainService(object):

    def __init__(self, client_repository: OpenAPIClientRepository):
        self.client_repository = client_repository

    def get_client_by_app_name(self, app_name: str) -> Optional[OpenAPIClientEntity]:
        return self.client_repository.get_by_app_name(app_name)

    def check_gmt_date(self, gmt_date: str) -> Optional[str]:
        date = SignatureUtil.convert_gmt_to_local_datetime(gmt_date)
        if DatetimeUtil.is_available_date(date):
            return gmt_date
        return None

    def check_signature(
            self, token: str, access_key: str, secret_key: str, app_name: str, *,
            method: str, content_type: str, gmt_date: str,
            query_string: str, request_body: str, user_id: str
    ) -> bool:
        ak, signature = token[8:].split('.')
        if ak != access_key:
            return False

        query_string = UrlUtil.sort_query_string_by_key(query_string)
        items_to_sign = [method, content_type, gmt_date, app_name, query_string, request_body]
        if user_id:
            items_to_sign.append(user_id)
        str_to_sign = '\n'.join(items_to_sign)
        valid_request = SignatureUtil.validate_signature(
            signature, secret_key=secret_key, string_to_sign=str_to_sign,
        )
        return valid_request
