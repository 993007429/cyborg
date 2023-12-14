from typing import Union

from flask import Response
from orjson import orjson


def orjsonify(data: Union[dict, str]) -> Response:
    return Response(response=orjson.dumps(data), status=200, content_type='application/json')
