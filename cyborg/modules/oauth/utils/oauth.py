import os
from hashlib import sha1
from base64 import b64decode, b64encode, b85decode, b85encode, urlsafe_b64encode
from typing import Dict, Optional

from cyborg.utils.jwt import jwt_encode, jwt_decode


class OAuthUtil:

    @classmethod
    def generate_client_id(cls) -> str:
        """生成唯一的 client_id
        """
        random_key = urlsafe_b64encode(os.urandom(32))[:16].decode('utf-8')
        return sha1(random_key.encode('utf-8')).hexdigest()[10: -10]

    @classmethod
    def generate_client_secret(cls) -> str:
        random_key = urlsafe_b64encode(os.urandom(32))[:16].decode('utf-8')
        return sha1(random_key.encode('utf-8')).hexdigest()

    @classmethod
    def generate_code_by_client(cls, client_id: str) -> str:
        return sha1(client_id.encode('utf-8')).hexdigest()[10: -10]

    @classmethod
    def create_access_token(cls, payload: Dict, secret: str) -> str:
        """创建应用 access_token

        :return: token
        """
        access_token = jwt_encode(payload, secret)
        return access_token

    @classmethod
    def parse_access_token(cls, access_token: str, secret: str) -> Optional[Dict]:
        try:
            return jwt_decode(access_token, secret)
        except Exception:
            return None

    @classmethod
    def opaque_string(cls, string: str) -> str:
        """序列化混淆字符串
        """
        return b85encode(
            b64encode(string.encode('utf-8')), pad=True,
        ).hex()

    @classmethod
    def reverse_opaque_string(cls, opaque: str) -> Optional[str]:
        """The reverse of :meth: `opaque_string`
        """
        try:
            return b64decode(
                b85decode(bytes.fromhex(opaque))
            ).decode()
        except Exception:
            return None
