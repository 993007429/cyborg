from typing import Optional

import jwt

from cyborg.app.settings import Settings


def jwt_encode(payload: dict, secret: str, algorithm: str = 'HS256', headers: Optional[dict] = None) -> str:
    return jwt.encode(payload, secret, algorithm=algorithm, headers=headers).decode('utf-8')


def jwt_decode(encoded: str, secret: str = Settings.JWT_SECRET, algorithm: str = 'HS256') -> Optional[dict]:
    try:
        return jwt.decode(encoded, secret, algorithms=[algorithm])
    except Exception as e:
        return None
