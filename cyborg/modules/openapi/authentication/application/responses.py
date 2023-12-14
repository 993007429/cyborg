from typing import Optional

from cyborg.seedwork.application.responses import AppResponse


class ParamsSignatureErrorResponse(AppResponse):
    """参数签名有问题通用
    """
    err_code: int = 10021
    message: Optional[str] = '签名认证不通过'


class UnregisteredClientResponse(AppResponse):
    err_code: int = 10022
    message: Optional[str] = '未授权的开发者应用'
