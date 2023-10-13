import datetime
from typing import Optional, Tuple

from cyborg.app.settings import Settings
from cyborg.modules.oauth.domain.entities import OAuthApplicationEntity
from cyborg.modules.oauth.domain.repositories import OAuthApplicationRepository
from cyborg.modules.oauth.utils.oauth import OAuthUtil
from cyborg.utils.jwt import jwt_decode


class OAuthDomainService(object):

    def __init__(self, repository: OAuthApplicationRepository):
        self.repository = repository

    def create_oauth_app(
            self, name: str, client_id: str, client_secret: str
    ) -> Tuple[Optional[str], Optional[OAuthApplicationEntity]]:
        app = OAuthApplicationEntity.new_application(
            name=name, client_id=client_id, client_secret=client_secret,
        )
        if not self.repository.save(app):
            return '创建失败', None
        return None, app

    def get_oauth_app_by_client_id(self, client_id: str) -> Optional[OAuthApplicationEntity]:
        return self.repository.get_oauth_app_by_client_id(client_id)

    def create_access_token(self, client_id: str, client_secret: str, access_secret: str,
                            *, expiration: int) -> str:
        """创建应用 access_token

        :return: token
        """
        payload = dict(
            exp=datetime.datetime.now() + datetime.timedelta(seconds=expiration),
            client_id=client_id,
            client_secret=client_secret,
        )
        access_token = OAuthUtil.create_access_token(payload, access_secret)
        return access_token

    def get_client_app_by_token(self, token: str) -> Optional[OAuthApplicationEntity]:
        payload = jwt_decode(token, Settings.JWT_SECRET, algorithm='HS256') if token else None
        if not payload:
            return None
        client_id = payload.get('client_id')
        client_secret = payload.get('client_secret')
        if not (client_id and client_secret):
            return None

        client_app = self.get_oauth_app_by_client_id(client_id=client_id)
        if not client_app:
            return None

        if client_app.client_secret != client_secret:
            return None

        return client_app
