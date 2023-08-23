from cyborg.seedwork.application.responses import AppResponse


class ParamsSignatureErrorResponse(AppResponse):
    """参数签名有问题通用
    """
    err_code = 10021
    message = '签名认证不通过'


class UnregisteredClientResponse(AppResponse):
    err_code = 10022
    message = '未授权的开发者应用'
