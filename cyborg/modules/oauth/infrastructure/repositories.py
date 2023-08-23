from typing import Type, Optional

from cyborg.modules.oauth.domain.entities import OAuthApplicationEntity
from cyborg.modules.oauth.domain.repositories import OAuthApplicationRepository
from cyborg.modules.oauth.infrastructure.models import OAuthApplication
from cyborg.seedwork.infrastructure.repositories import SQLAlchemySingleModelRepository


class SqlAlchemyOAuthApplicationRepository(
        OAuthApplicationRepository, SQLAlchemySingleModelRepository[OAuthApplicationEntity]):

    @property
    def model_class(self) -> Type[OAuthApplication]:
        return OAuthApplication

    def get_by_account_id_name(self, account_id: int, app_name: str) -> Optional[OAuthApplicationEntity]:
        model = self.session.query(OAuthApplication).filter(
            OAuthApplication.account_id == account_id, OAuthApplication.name == app_name).first()
        return OAuthApplicationEntity.from_dict(model.raw_data) if model else None

    def get_oauth_app_by_client_id(self, client_id: str) -> Optional[OAuthApplicationEntity]:
        model = self.session.query(OAuthApplication).filter(OAuthApplication.client_id == client_id).first()
        return OAuthApplicationEntity.from_dict(model.raw_data) if model else None
