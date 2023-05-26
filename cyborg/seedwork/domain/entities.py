import json
from typing import TypeVar, List, Dict, Type, Any

from pydantic import BaseModel

from cyborg.seedwork.domain.value_objects import BaseEnum
from cyborg.utils.strings import camel_to_snake


class BaseDomainEntity(BaseModel):

    raw_data: dict = {}

    class Config:
        arbitrary_types_allowed = True

    @property
    def json_fields(self) -> List[str]:
        return []

    @property
    def enum_fields(self) -> Dict[str, Type[BaseEnum]]:
        return {}

    def _convert_value(self, field_name: str, value: Any) -> Any:
        if field_name in self.json_fields:
            if value and isinstance(value, str):
                value = json.loads(value)
        if field_name in self.enum_fields:
            if value and isinstance(value, str):
                value = self.enum_fields[field_name].get_by_value(value)
        return value

    def __getattr__(self, name):
        if name in self.raw_data:
            value = self.raw_data[name]
            value = self._convert_value(name, value)
            return value
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return None

    def __setattr__(self, key, value):
        if key in self.raw_data:
            value = self._convert_value(key, value)
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
        d = self.raw_data
        for field_name in self.json_fields:
            d[field_name] = self.__getattr__(field_name)
        for field_name in self.enum_fields.keys():
            d[field_name] = self.__getattr__(field_name)
        return d

    def to_index_doc(self):
        return self.raw_data

    @classmethod
    def from_dict(cls, data: dict, **kwargs):
        return cls(raw_data={camel_to_snake(k): v for k, v in data.items() if isinstance(k, str)}, **kwargs)


E = TypeVar('E', bound=BaseDomainEntity)
