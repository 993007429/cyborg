import hashlib
import hmac
import time
import jwt
from base64 import urlsafe_b64encode
from datetime import datetime, timedelta
from typing import Dict, Optional

from cyborg.utils.jwt import jwt_encode, jwt_decode


class SignatureUtil:

    ONE_SECOND = 1
    ONE_MINUTE = 60 * ONE_SECOND
    ONE_HOUR = 60 * ONE_MINUTE
    ONE_DAY = 12 * ONE_HOUR

    @classmethod
    def hmac_sha1(cls, secret_key: str, string_to_sign: str) -> str:
        mac = hmac.new(
            secret_key.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha1,
        )
        return mac.hexdigest()

    @classmethod
    def urlsafe_base64_encode(cls, string: str) -> str:
        return urlsafe_b64encode(string.encode('utf-8')).decode('utf-8')

    @classmethod
    def validate_signature(cls, signature: str, secret_key: str, string_to_sign: str) -> bool:
        mac_str = cls.hmac_sha1(secret_key, string_to_sign)
        return cls.urlsafe_base64_encode(mac_str) == signature

    @classmethod
    def datetime_to_gmt_format(cls, dt: datetime) -> str:
        return dt.strftime('%a, %d %b %Y %H:%M:%S GMT')

    @classmethod
    def gmt_format_to_datetime(cls, gmt_str: str):
        return datetime.strptime(gmt_str, '%a, %d %b %Y %H:%M:%S GMT')

    @classmethod
    def convert_gmt_to_local_datetime(cls, dt_str: str) -> datetime:  # type: ignore
        try:
            dt = cls.gmt_format_to_datetime(dt_str)
            return dt + timedelta(hours=8)
        except ValueError:
            pass

    @classmethod
    def jwt_token_from_payload(cls, payload: Dict, secret: str, expire_in: int = ONE_HOUR) -> str:
        """根据payload生成token

        :param payload: data to encode
        :param secret: secret when encode
        :param expire_in: expiration to the token
        """
        issue_time = int(time.time())
        payload.update({
            'iat': issue_time,
            'exp': issue_time + expire_in,
        })
        return jwt_encode(payload, secret, algorithm='HS256')

    @classmethod
    def payload_from_jwt_token(cls, token: str, secret: str) -> Optional[Dict]:
        """从 jwt token 解析 payload

        :param token: jwt token str
        :param secret: secret when decode
        :return:
        """
        try:
            return jwt_decode(token, secret, 'HS256')
        except jwt.ExpiredSignatureError:
            return None


if __name__ == '__main__':
    test_secret_key = 'B_dS4UHuBoMQsnqow-eQNmQcCwMxxGT10Qo6_jh3097pSdhnGU'
    strToSign = "POST" + '\n' \
                + "application/json;charset=UTF-8" + '\n' \
                + "Wed, 02 Mar 2022 08:15:57 GMT" + '\n' \
                + "wenjuan" + '\n' \
                + "code=reliability&file_name=问卷标题.csv"

    unsafe = SignatureUtil.hmac_sha1(test_secret_key, strToSign)
    print('unsafe signature: ', unsafe)
    signature = SignatureUtil.urlsafe_base64_encode(unsafe)
    print('signature: ', signature)
    print('validation: ', SignatureUtil.validate_signature(signature, test_secret_key, strToSign))
    # print(SignatureComposer.gmt_format_to_datetime('Thu, 24 Feb 2022 03:35:31 GMT'))
    # a = SignatureComposer.jwt_token_from_payload({'a': 'wenjuan'}, test_secret_key, expire_in=1)
    # print(a)
    # print(SignatureComposer.payload_from_jtw_token(a, test_secret_key))
