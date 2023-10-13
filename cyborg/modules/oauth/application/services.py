from typing import Optional, Tuple

from cyborg.app.settings import Settings
from cyborg.modules.oauth.application.responses import (
    UnregisteredOAuthClientResponse, OAuthGrantTypeUnSupportedClientResponse)
from cyborg.modules.oauth.domain.services import OAuthDomainService
from cyborg.modules.oauth.utils.oauth import OAuthUtil
from cyborg.modules.user_center.user_core.application.services import UserCoreService
from cyborg.seedwork.application.responses import AppResponse


class OAuthService(object):

    def __init__(self, domain_service: OAuthDomainService, user_service: UserCoreService):
        self.domain_service = domain_service
        self.user_service = user_service

    def create_oauth_app(self, name: str) -> Tuple[Optional[str], Optional[dict]]:
        """注册 OAuth 应用
        """
        client_id = OAuthUtil.generate_client_id()
        client_secret = OAuthUtil.generate_client_secret()
        if not name:
            return '名称不能为空', None

        err_msg, app = self.domain_service.create_oauth_app(
            name=name,
            client_id=client_id,
            client_secret=client_secret,
        )
        if err_msg is not None:
            return err_msg, None
        return None, app.to_dict() if app else None

    def get_client_by_access_token(self, token: str = '') -> AppResponse:
        oauth_client = self.domain_service.get_client_app_by_token(token)

        if not oauth_client:
            return UnregisteredOAuthClientResponse()

        return AppResponse(data=oauth_client.to_dict())

    def get_access_token(
            self, client_id: str, client_secret: str, grant_type: str = 'client_credentials', token_type: str = 'Bearer'
    ) -> AppResponse:
        """通过授权码 `code` 获取 access token
        """
        oauth_app = self.domain_service.get_oauth_app_by_client_id(client_id)
        if not oauth_app or client_secret != oauth_app.client_secret:
            return UnregisteredOAuthClientResponse(message='OAuth鉴权未通过')

        if grant_type == 'client_credentials':
            expire_in = 60 * 60 * 24 * 30
            access_token = self.domain_service.create_access_token(
                client_id, client_secret, Settings.JWT_SECRET, expiration=expire_in,
            )
            if token_type.upper() == 'BEARER':
                access_token = f'Bearer {access_token}'
            data = {
                'access_token': access_token,
                "token_type": token_type,
                'expire_in': expire_in,
            }
            return AppResponse(data=data)
        else:
            return OAuthGrantTypeUnSupportedClientResponse()
