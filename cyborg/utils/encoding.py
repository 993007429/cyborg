import enum
import datetime
from decimal import Decimal
from json import JSONEncoder

from numpy import int64, int32
from pydantic import BaseModel


class CyborgJsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (int64, int32)):  # type: ignore
            return int(obj)
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, datetime.time):
            return obj.strftime('%H:%M:%S')
        elif isinstance(obj, Decimal):
            return round(float(obj), 3)
        elif isinstance(obj, float):
            return round(obj, 3)
        elif isinstance(obj, enum.Enum):
            return obj.value
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, BaseModel):
            return obj.dict()
        return JSONEncoder.default(self, obj)
