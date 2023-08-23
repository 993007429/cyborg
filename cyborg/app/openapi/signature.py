import asyncio
import functools

from flask import request

from cyborg.app.service_factory import AppServiceFactory
from cyborg.seedwork.application.responses import AppResponse


class SignatureChecker:

    @classmethod
    def check_signature(cls, f):
        """校验参数的签名
        """

        def __check(self):
            result = cls.__do_checker()
            if isinstance(result, AppResponse):
                return result

        if asyncio.iscoroutinefunction(f):
            @functools.wraps(f)
            async def inner(self, *args, **kwargs):
                res = __check(self)
                if res is not None:
                    return res
                return await f(self, *args, **kwargs)
        else:
            @functools.wraps(f)
            def inner(self, *args, **kwargs):
                res = __check(self)
                if res is not None:
                    return res
                return f(self, *args, **kwargs)
        return inner

    @classmethod
    def __do_checker(cls):
        authorization: str = request.headers.get('Authorization', '')
        params = request.args
        params.update(request.form)

        return AppServiceFactory.openapi_auth_service.check_params_signature(
            sign=authorization, params=params
        )
