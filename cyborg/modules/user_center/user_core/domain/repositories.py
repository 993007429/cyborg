from abc import ABCMeta, abstractmethod
from typing import Optional, Type

from cyborg.modules.user_center.user_core.domain.entities import UserEntity, CompanyEntity
from cyborg.seedwork.infrastructure.repositories import SingleModelRepository


class UserRepository(SingleModelRepository[UserEntity], metaclass=ABCMeta):

    @property
    def entity_class(self) -> Type[UserEntity]:
        return UserEntity

    @abstractmethod
    def get_user_by_name(self, username: str, company: str) -> Optional[UserEntity]:
        ...


class CompanyRepository(SingleModelRepository[CompanyEntity], metaclass=ABCMeta):

    @property
    def entity_class(self) -> Type[CompanyEntity]:
        return CompanyEntity

    @abstractmethod
    def get_company_by_id(self, company: str) -> Optional[CompanyEntity]:
        ...
