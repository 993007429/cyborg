import calendar
import datetime
import os
from io import BytesIO
from typing import Optional
from uuid import uuid4

from minio.datatypes import Object

from cyborg.infra.oss import Oss, OSSHeadObject


class MockedOss(Oss):

    def __init__(self, *args, **kwargs):
        super(MockedOss, self).__init__(*args, **kwargs)
        self.client = None
        self.repository = {}

    def bucket_endpoint(self):
        url = self.public_endpoint
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'http://' + url
        return f'{url}/{self.bucket_name}'

    def object_exists(self, file_key: str) -> bool:
        return file_key in self.repository

    def put_object_from_file(self, file_key: str, filepath: str) -> bool:
        return True

    def get_object_to_file(self, file_key: str, filepath: str):
        return None

    def put_object_from_io(self, bytesio: BytesIO, file_key: str):
        self.repository[file_key] = bytesio
        return True

    def get_object_to_io(self, file_key: str) -> Optional[BytesIO]:
        return self.repository.get(file_key, None)

    def get_object(self, file_key: str):
        return None

    def head_object(self, file_key: str):
        obj = self.repository.get(file_key)
        if not obj:
            return None
        result = Object(
            self.bucket_name,
            file_key,
            last_modified=datetime.datetime.now(),
            etag='',
            size=obj.getbuffer().nbytes,
            content_type='text/csv',
            metadata={},
            version_id='',
        )
        tm = result.last_modified.timetuple()
        return OSSHeadObject(
            content_type=result.content_type,
            content_length=result.size,
            etag=result.etag,
            last_modified=calendar.timegm(tm)
        )

    def list_objects(self, prefix) -> list:
        return []

    def delete_object(self, file_key: str):
        del self.repository[file_key]
        return True

    def copy_object(self, source_key, target_key, source_bucket_name: str = ''):
        if source_key in self.repository:
            self.repository[target_key] = self.repository[source_key]
        return True

    def generate_sign_url(self, method: str, key: str, expire_in: int = 600, slash_safe=True) -> str:
        return ''

    def generate_sign_token(self, filetype: str, target_dir: str, expire_in: int = 300) -> dict:

        if target_dir.endswith(filetype):
            file_key = target_dir
        else:
            file_key = os.path.join(target_dir, f'{uuid4().hex}.{filetype}')

        return {
            'host': self.bucket_endpoint,
            'file_key': file_key,
            'headers': {}
        }
