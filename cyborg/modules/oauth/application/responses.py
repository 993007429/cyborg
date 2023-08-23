from cyborg.seedwork.application.responses import AppResponse

class InvalidAuthorizeCodeResponse(AppResponse):
    err_code = 10030
    message = '授权code无效或已过期'


class InvalidAccessTokenResponse(AppResponse):
    http_code = 401
    err_code = 10031
    message = 'access_token不合法'


class UnauthorizedUserResponse(AppResponse):
    http_code = 403
    err_code = 10032
    message = '用户未授权或授权用户不存在'


class UnregisteredOAuthClientResponse(AppResponse):
    http_code = 403
    err_code = 10033
    message = 'OAuth应用不存在或未授权'


class OAuthGrantTypeUnSupportedClientResponse(AppResponse):
    http_code = 403
    err_code = 10034
    message = '该OAuth鉴权类型不支持'
