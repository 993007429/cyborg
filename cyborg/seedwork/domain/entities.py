from typing import TypeVar

from pydantic import BaseModel

from cyborg.utils.strings import camel_to_snake


class BaseDomainEntity(BaseModel):

    raw_data: dict = {}

    class Config:
        arbitrary_types_allowed = True

    def __getattr__(self, name):
        if name in self.raw_data:
            return self.raw_data[name]
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return None

    def __setattr__(self, key, value):
        if key in self.raw_data:
            self.raw_data[key] = value
        else:
            return object.__setattr__(self, key, value)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def update_data(self, **kwargs):
        self.raw_data.update(kwargs)

    def to_dict(self):
        return self.raw_data

    def to_index_doc(self):
        return self.raw_data

    @classmethod
    def from_dict(cls, data: dict, **kwargs):
        return cls(raw_data={camel_to_snake(k): v for k, v in data.items() if isinstance(k, str)}, **kwargs)


E = TypeVar('E', bound=BaseDomainEntity)
