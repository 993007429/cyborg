import json
import logging
from functools import cached_property
from typing import TypeVar

from sqlalchemy import inspect, JSON
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)


class _Base:
    __table_args__ = {'mysql_engine': 'InnoDB'}

    @property
    def raw_data(self):
        mapper = inspect(self.__class__)
        return {column.key: getattr(self, column.key) for column in mapper.attrs}

    @cached_property
    def json_columns(self):
        mapper = inspect(self.__class__)
        return [column.key for column in mapper.attrs if column.expression is not None and isinstance(
            column.expression.type, JSON)]

    def set_data(self, data: dict):
        mapper = inspect(self.__class__)
        fields = [column.key for column in mapper.attrs]
        for k, v in data.items():
            if k in fields and v != getattr(self, k):
                setattr(self, k, v)

    def __setattr__(self, key, value):
        if (isinstance(value, list) or isinstance(value, dict)) and key not in self.json_columns:
            value = json.dumps(value, ensure_ascii=False)
        super(_Base, self).__setattr__(key, value)

    def to_dict(self) -> dict:
        return {
            'id': self.id
        }

    def __str__(self):
        return f'{self.__class__.__name__}(id={self.id})'

    def __repr__(self):
        return self.__str__()


BaseModel = declarative_base(cls=_Base, name='BaseModel')  # Model 基类


M = TypeVar('M', bound=_Base)
