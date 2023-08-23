import abc
from abc import ABCMeta
from typing import Type, Optional

from cyborg.modules.oauth.domain.entities import OAuthApplicationEntity
from cyborg.seedwork.infrastructure.repositories import SingleModelRepository


class OAuthApplicationRepository(SingleModelRepository[OAuthApplicationEntity], metaclass=ABCMeta):

    @property
    def entity_class(self) -> Type[OAuthApplicationEntity]:
        return OAuthApplicationEntity

    @abc.abstractmethod
    def get_by_account_id_name(self, account_id: int, app_name: str) -> Optional[OAuthApplicationEntity]:
        ...

    @abc.abstractmethod
    def get_oauth_app_by_client_id(self, client_id: str) -> Optional[OAuthApplicationEntity]:
        ...
