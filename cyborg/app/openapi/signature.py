import asyncio
import functools

from flask import request

from cyborg.app.service_factory import AppServiceFactory
from cyborg.seedwork.application.responses import AppResponse


class SignatureChecker:

    @classmethod
    def openapi_signature_check(cls, f):
        """开放平台内部调用鉴权

        目前比较简单，校验HEADER：
          - X-User-Agent
          - X-UID
        """
        if asyncio.iscoroutinefunction(f):
            @functools.wraps(f)
            async def inner(self, *args, **kwargs):
                x_ua = request.headers.get('X-User-Agent', '')
                if not x_ua or not x_ua.startswith('SPSSPRO OpenAPI'):
                    return AppResponse(err_code=403, status_code=403, message='Access Denied!')
                return await f(self, *args, **kwargs)
        else:
            @functools.wraps(f)
            def inner(self, *args, **kwargs):
                x_ua = request.headers.get('X-User-Agent', '')
                if not x_ua or not x_ua.startswith('SPSSPRO OpenAPI'):
                    return AppResponse(err_code=403, status_code=403, message='Access Denied!')

                return f(self, *args, **kwargs)
        return inner

    @classmethod
    def params_signature_check(cls, f):
        """校验参数的签名
        """

        def __check(self):
            result = cls.__do_checker(self)
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
    def __do_checker(cls, handler):
        authorization: str = handler.request.headers.get('Authorization', '')
        method = handler.request.method
        gmt_date = handler.request.headers.get('Request-Date', '')
        content_type = handler.request.headers.get('Content-Type', '')
        query_string = handler.request.query
        request_body = handler.request.body.decode('utf-8')
        return AppServiceFactory.openapi_auth_service.check_params_signature(
            token=authorization, method=method, content_type=content_type, gmt_date=gmt_date,
            query_string=query_string, request_body=request_body)
