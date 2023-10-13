import asyncio
import functools
import logging
import os
import time
from typing import Optional, Union

from flask import current_app, request

from cyborg.app.settings import Settings
from cyborg.seedwork.application.responses import AppResponse
from cyborg.seedwork.domain.value_objects import BaseValueObject
from cyborg.utils.jwt import jwt_encode, jwt_decode

logger = logging.getLogger(__name__)


class LoginUser(BaseValueObject):

    username: str
    company: str
    role: Union[str, int] = 0
    importable: Optional[int] = None
    export_json: Optional[int] = None
    model_lis: str = ''
    signed: bool = False
    volume: Optional[int] = None
    time_out: Optional[str] = None
    cloud: bool = False
    is_test: bool = False

    @property
    def data_dir(self):
        return os.path.join(Settings.DATA_DIR, self.company)

    @property
    def expire_time(self) -> Optional[float]:
        return time.mktime(time.strptime(self.time_out, "%Y-%m-%d")) + 86399 if self.time_out else None

    @classmethod
    def get_payload_from_token(cls, token: str) -> Optional['LoginUser']:
        try:
            payload = jwt_decode(token, current_app.config.JWT_SECRET, algorithm='HS256')
            return LoginUser.from_dict(payload)
        except Exception:
            return None

    @property
    def is_expired(self):
        return self.expire_time and self.expire_time < time.time()

    @property
    def jwt_token(self) -> str:
        payload = self.to_dict()
        payload['expires'] = self.expire_time if self.time_out else None
        del payload['export_json']
        payload['exportJson'] = self.export_json
        return jwt_encode(payload, Settings.JWT_SECRET, algorithm='HS256')

    @classmethod
    def get_from_cookie(cls) -> Optional['LoginUser']:
        token = request.cookies.get('jwt')
        payload = jwt_decode(token, Settings.JWT_SECRET, algorithm='HS256') if token else None
        login_user = LoginUser.from_dict(payload) if payload else None
        if not login_user or (not request.path.endswith('upload2') and login_user.is_expired):
            return None

        from cyborg.app.service_factory import AppServiceFactory
        res = AppServiceFactory.user_service.check_user(login_user.username, login_user.company)
        if not res.data:
            return None

        return login_user


def login_required(f):
    from cyborg.app.request_context import request_context
    if asyncio.iscoroutinefunction(f):
        @functools.wraps(f)
        async def wrapped(*args, **kwargs):
            if not request_context.current_user:
                return AppResponse(message='请先登录', code=401)
            return await f(*args, **kwargs)
    else:
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            if not request_context.current_user:
                return AppResponse(message='请先登录', code=401)
            return f(*args, **kwargs)

    return wrapped
