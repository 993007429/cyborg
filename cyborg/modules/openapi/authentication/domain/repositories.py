import abc
from abc import ABCMeta
from typing import Type, Optional

from cyborg.modules.openapi.authentication.domain.entities import OpenAPIClientEntity


class OpenAPIClientRepository(metaclass=ABCMeta):

    @property
    def entity_class(self) -> Type[OpenAPIClientEntity]:
        return OpenAPIClientEntity

    @abc.abstractmethod
    def get_by_app_name(self, app_name: str) -> Optional[OpenAPIClientEntity]:
        ...
