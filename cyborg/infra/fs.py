import abc
import ctypes
import os
import shutil
import sys


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
        return os.path.dirname(path)

    def path_splitext(self, path) -> str:
        return os.path.splitext(path)[1]

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
            size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
        return size

    def listdir(self, path: str):
        return os.listdir(path)

    def remove_dir(self, path: str):
        shutil.rmtree(path, ignore_errors=True)


fs = LocalFileSystem()
