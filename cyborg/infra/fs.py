import abc
import ctypes
import json
import logging
import os
import shutil
import sys
from io import BytesIO
from typing import Union

from cyborg.app.settings import Settings
from cyborg.infra.oss import oss

logger = logging.getLogger(__name__)


class FileSystem(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_free_space(self, file_path: str) -> float:
        ...


class LocalFileSystem(FileSystem):

    def path_join(self, *args) -> str:
        return os.path.join(*args)

    def path_exists(self, path) -> bool:
        return os.path.exists(path)

    def path_dirname(self, path) -> str:
        return os.path.dirname(path)

    def path_basename(self, path) -> str:
        return os.path.basename(path)

    def path_isfile(self, path) -> bool:
        return os.path.isfile(path)

    def path_splitext(self, path) -> tuple:
        return os.path.splitext(path)

    def get_free_space(self, file_path: str):
        if sys.platform == 'win32':
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(file_path), None, None, ctypes.pointer(free_bytes))
            return free_bytes.value / 1024 / 1024 / 1024
        else:
            st = os.statvfs(file_path)
            return st.f_bavail * st.f_frsize / 1024 / 1024

    def get_file_size(self, file_path: str) -> int:
        return os.path.getsize(file_path)

    def get_dir_size(self, path: str) -> int:
        size = 0
        for root, dirs, files in os.walk(path):
            size += sum([os.path.getsize(os.path.join(root, name)) for name in files if not os.path.islink(os.path.join(root, name))])
        return size

    def listdir(self, path: str):
        return os.listdir(path)

    def remove_dir(self, path: str):
        shutil.rmtree(path, ignore_errors=True)

    def save_object(self, file_key: str, obj: Union[list, dict]) -> str:
        raise NotImplementedError

    def get_object(self, file_key: str) -> Union[list, dict, None]:
        raise NotImplementedError


class HybridFileSystem(LocalFileSystem):

    def save_file(self, file_key: str, buffer: BytesIO) -> str:
        if oss:
            oss.put_object_from_io(buffer, file_key)
            return oss.generate_sign_url(method='GET', key=file_key, expire_in=3600 * 24)
        else:
            file_path = f'{Settings.DATA_DIR}/{file_key}'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(buffer.getvalue())
            return file_path

    def save_object(self, file_key: str, obj: Union[list, dict]) -> str:
        buffer = BytesIO()
        buffer.write(json.dumps(obj, ensure_ascii=False).encode('utf-8'))
        buffer.seek(0)

        if oss:
            oss.put_object_from_io(buffer, file_key)
            return oss.generate_sign_url(method='GET', key=file_key, expire_in=3600 * 24)
        else:
            file_path = f'{Settings.DATA_DIR}/{file_key}'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(buffer.getvalue())
            return file_path

    def get_object(self, file_key: str) -> Union[list, dict, None]:
        try:
            if oss:
                bytes_io = oss.get_object_to_io(file_key=file_key)
                return json.loads(bytes_io.getvalue().decode('utf-8'))
            else:
                file_path = f'{Settings.DATA_DIR}/{file_key}'
                with open(file_path, 'rb') as f:
                    return json.loads(f.read().decode('utf-8'))
        except Exception as e:
            logger.exception(e)

        return None


fs = HybridFileSystem()
