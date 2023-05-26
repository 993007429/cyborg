import datetime
import enum
import time
from typing import Optional, Callable, TypeVar, Any

from pydantic import BaseModel


class BaseEnum(enum.Enum):

    @classmethod
    def get_by_value(cls, value: Any):
        try:
            return cls(value.value if isinstance(value, BaseEnum) else value)
        except ValueError:
            return None

    @classmethod
    def get_by_name(cls, name):
        try:
            return cls[name.name if isinstance(name, BaseEnum) else name]
        except KeyError:
            return None

    def __eq__(self, other):
        return self.value == other or super(BaseEnum, self).__eq__(other)

    def __hash__(self):
        return hash(self.value)

    def translate(self, *args, **kwargs):
        return self.value


class BaseValueObject(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_dict(
            cls, d: Optional[dict], adapter: Optional[Callable[[dict], dict]] = None
    ):
        if d is None:
            return None
        if adapter:
            d = adapter(d)
        kwargs = {k: d.get(k) for k, _ in cls.schema()['properties'].items()}
        return cls(**kwargs)

    def to_dict(self):
        return self.dict()


@enum.unique
class AIType(BaseEnum):

    @classmethod
    def get_by_value(cls, value: Optional[str]):
        if isinstance(value, AIType):
            value = value.value
        if value and value.startswith('tct') or value.startswith('lct') or value.startswith('dna'):
            value = value[0:3]
        if value and value.startswith('fish'):
            value = 'fishTissue'
        if value == 'tagging':
            value = 'label'
        return super().get_by_value(value)

    human = 'human'
    human_tl = 'human_tl'
    human_bm = 'human_bm'
    label = 'label'
    np = 'np'
    er = 'er'
    pr = 'pr'
    bm = 'bm'
    tct = 'tct'
    lct = 'lct'
    dna = 'dna'
    her2 = 'her2'
    ki67 = 'ki67'
    pdl1 = 'pdl1'
    ki67hot = 'ki67hot'
    celldet = 'celldet'
    fish_tissue = 'fishTissue'
    model_calibrate_tct = 'model_calibrate_tct'

    @property
    def ai_name(self) -> str:
        if self == AIType.ki67:
            return AIType.ki67hot.value
        return self.value


A = TypeVar('A', bound=AIType)
