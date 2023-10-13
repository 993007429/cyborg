import logging
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from contextvars import ContextVar
from datetime import datetime
from typing import List, Tuple, Type, Union, Dict, Optional, Generic, Generator, Any

from sqlalchemy import desc, Column, asc
from sqlalchemy.orm import Session

from cyborg.infra.session import transaction
from cyborg.seedwork.domain.entities import E, BaseDomainEntity
from cyborg.seedwork.infrastructure.models import BaseModel
from cyborg.utils.id_worker import IdWorker

logger = logging.getLogger(__name__)

PageParams = namedtuple('PageParams', ['page', 'per_page'])

seq = 0


class RepoQuery(object):
    def __init__(
        self,
        params: Dict[str, Any] = None,
        order_by: List[Tuple[Column, Union[Type[desc], Type[asc], None]]] = None,
        fields: List[Column] = None,
        page_params: PageParams = None
    ):
        self.params = params or {}
        self.order_by = order_by or []
        self.fields = fields or []
        self.page_params = page_params or PageParams(page=0, per_page=99999999)


class SingleModelRepository(Generic[E], metaclass=ABCMeta):

    @property
    @abstractmethod
    def entity_class(self):
        ...

    @abstractmethod
    def get(self, uid: int, with_for_update: bool = False) -> Optional[E]:
        ...

    def gets(self, uids: List[int]) -> List[Optional[E]]:
        ...

    def scan_all(self, step=100) -> Generator[E, None, None]:
        ...

    @abstractmethod
    def save(self, entity: E) -> bool:
        ...

    @abstractmethod
    def batch_save(self, entities: List[E]) -> bool:
        ...

    @abstractmethod
    def delete_by_id(self, entity_id: int) -> bool:
        ...

    def touch(self, entity_id):
        ...

    def gets_by_query(self, repo_query: RepoQuery) -> Tuple[int, List[Union[E, Tuple]]]:
        ...


class SQLAlchemyRepository(object):

    def __init__(self, *, session: ContextVar):
        self._session_cv = session

    @property
    def session(self) -> Session:
        s = self._session_cv.get()
        assert s is not None
        return s

    def convert_to_model(
            self, entity: BaseDomainEntity, model_class: Type[BaseModel], id_worker: IdWorker = None
    ) -> Optional[BaseModel]:
        if entity.id:
            model = self.session.get(model_class, entity.id)
            if model:
                model.set_data(entity.raw_data)
                return model
        elif id_worker is not None:
            entity.update_data(id=id_worker.get_next_id() or id_worker.get_new_id())

        return model_class(**entity.raw_data)


class SQLAlchemySingleModelRepository(SingleModelRepository, SQLAlchemyRepository, Generic[E], metaclass=ABCMeta):

    @property
    @abstractmethod
    def model_class(self) -> Type[BaseModel]:
        ...

    def get(self, uid: int, with_for_update: bool = False) -> Optional[E]:
        query = self.session.query(self.model_class)
        if with_for_update:
            query = query.with_for_update()
        model = query.get(uid)
        return self.entity_class.from_dict(model.raw_data) if model else None

    def gets(self, uids: List[int]) -> List[Optional[E]]:
        return [self.get(uid) for uid in uids]

    def scan_all(self, step=100) -> Generator[E, None, None]:
        for model in self.session.query(self.model_class).yield_per(step):
            yield self.entity_class.from_dict(model.raw_data)

    @transaction
    def save(self, entity: E) -> bool:
        model = self.convert_to_model(entity, self.model_class)
        if not model:
            return False
        self.session.add(model)
        self.session.flush([model])
        entity.update_data(**model.raw_data)
        return True

    @transaction
    def batch_save(self, entities: List[E]) -> bool:
        for entity in entities:
            self.save(entity)
        return True

    @transaction
    def delete_by_id(self, entity_id: int) -> bool:
        self.session.query(self.model_class).filter_by(id=entity_id).delete()
        return True

    @transaction
    def touch(self, entity_id):
        self.session.query(self.model_class).filter_by(id=entity_id).update({'last_modified': datetime.now()})

    def gets_by_query(self, repo_query: RepoQuery) -> Tuple[int, List[Union[E, Tuple]]]:
        """
        查询对象列表
        :param repo_query: a RepoQuery object which define the query detail
        :return:
        """
        if repo_query.fields:
            query = self.session.query(*repo_query.fields)
        else:
            query = self.session.query(self.model_class)
        query = query.filter_by(**repo_query.params)
        order_bys = []
        for field, _desc in repo_query.order_by:
            order_bys.append(_desc(field) if _desc else field)

        total = query.count()

        if order_bys:
            query = query.order_by(*order_bys)
        else:
            query = query.order_by(desc(self.model_class.id))

        if repo_query.page_params:
            offset = repo_query.page_params.page * repo_query.page_params.per_page
            limit = repo_query.page_params.per_page
            query = query.offset(offset).limit(limit)

        results = query.all()
        return total, [(self.entity_class.from_dict(item.raw_data) if isinstance(
            item, self.model_class) else item) for item in results]
