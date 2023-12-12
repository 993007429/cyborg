from typing import Optional

from cyborg.seedwork.application.responses import AppResponse


class InvalidAuthorizeCodeResponse(AppResponse):
    err_code: int = 10030
    message: Optional[str] = '授权code无效或已过期'


class InvalidAccessTokenResponse(AppResponse):
    http_code: int = 401
    err_code: int = 10031
    message: Optional[str] = 'access_token不合法'


class UnauthorizedUserResponse(AppResponse):
    http_code: int = 403
    err_code: int = 10032
    message: Optional[str] = '用户未授权或授权用户不存在'


class UnregisteredOAuthClientResponse(AppResponse):
    http_code: int = 403
    err_code: int = 10033
    message: Optional[str] = 'OAuth应用不存在或未授权'


class OAuthGrantTypeUnSupportedClientResponse(AppResponse):
    http_code: int = 403
    err_code: int = 10034
    message: Optional[str] = '该OAuth鉴权类型不支持'
