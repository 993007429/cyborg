import uuid
import hashlib


def encrypt_password(password: str, salt=None):
    salt = salt or uuid.uuid1().hex
    return salt, hashlib.sha1(f'{salt}{password}'.encode('utf-8')).hexdigest()
