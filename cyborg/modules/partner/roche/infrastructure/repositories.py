from typing import Optional, Type, List

from cyborg.modules.user_center.user_core.domain.entities import UserEntity, CompanyEntity
from cyborg.modules.user_center.user_core.domain.repositories import UserRepository, CompanyRepository
from cyborg.modules.user_center.user_core.infrastructure.models import UserModel, CompanyModel
from cyborg.seedwork.infrastructure.repositories import SQLAlchemySingleModelRepository


class SQLAlchemyUserRepository(UserRepository, SQLAlchemySingleModelRepository[UserEntity]):

    @property
    def model_class(self) -> Type[UserModel]:
        return UserModel

    def get_user_by_name(self, username: str, company: str) -> Optional[UserEntity]:
        model = self.session.query(UserModel).filter_by(username=username, company=company).first()
        return UserEntity.from_dict(model.raw_data) if model else None

    def get_user_by_id(self, user_id: int) -> Optional[UserEntity]:
        model = self.session.query(UserModel).get(user_id)
        return UserEntity.from_dict(model.raw_data) if model else None

    def get_users(self, company: Optional[str] = None) -> List[UserEntity]:
        query = self.session.query(UserModel)
        if company is not None:
            query = query.filter_by(company=company)
        models = query.all()
        return [UserEntity.from_dict(model.raw_data) for model in models]


class SQLAlchemyCompanyRepository(CompanyRepository, SQLAlchemySingleModelRepository[CompanyEntity]):

    @property
    def model_class(self) -> Type[CompanyModel]:
        return CompanyModel

    def get_company_by_id(self, company: str) -> Optional[CompanyEntity]:
        model = self.session.query(CompanyModel).filter_by(company=company).first()
        return CompanyEntity.from_dict(model.raw_data) if model else None

    def get_all_companies(self) -> List[CompanyEntity]:
        models = self.session.query(CompanyModel).all()
        return [CompanyEntity.from_dict(model.raw_data) for model in models]
