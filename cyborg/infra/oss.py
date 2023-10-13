import abc
import calendar
import datetime
import logging
import os
import random
from functools import cached_property
from io import BytesIO
from urllib.parse import urlparse
from uuid import uuid4

from minio import Minio
from minio.datatypes import PostPolicy
from pydantic import BaseModel

from cyborg.app.settings import Settings

logger = logging.getLogger(__name__)


class OSSHeadObject(BaseModel):

    #: 文件最后修改时间，类型为int。参考 :ref:`unix_time` 。
    last_modified: int

    #: 文件的MIME类型
    content_type: str

    #: Content-Length，可能是None。
    content_length: int

    #: HTTP ETag
    etag: str


class Oss(object, metaclass=abc.ABCMeta):

    def __init__(
            self, access_key: str, secret: str, pub_endpoint: str, bucket_name: str, private_endpoint: str = '',
            use_https: bool = False
    ):
        if not all((access_key, secret, pub_endpoint, bucket_name)):
            logger.warning('Missing key OSS config!')
        self.access_key = access_key
        self.secret = secret
        self.bucket_name = bucket_name
        self.public_endpoint = pub_endpoint
        self.private_endpoint = private_endpoint or pub_endpoint
        self.use_https = use_https

    @cached_property
    def bucket_endpoint(self):
        url = self.public_endpoint
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'http://' + url
        parser = urlparse(url)
        parser = parser._replace(netloc=f'{self.bucket_name}.{parser.hostname}')
        return parser.geturl()

    @abc.abstractmethod
    def path_join(self, *args):
        pass

    @abc.abstractmethod
    def object_exists(self, file_key: str) -> bool:
        pass

    @abc.abstractmethod
    def put_object_from_file(self, file_key: str, filepath: str):
        pass

    @abc.abstractmethod
    def get_object_to_file(self, file_key: str, filepath: str):
        pass

    @abc.abstractmethod
    def put_object_from_io(self, bytesio: BytesIO, file_key: str):
        pass

    def get_object_to_io(self, file_key: str) -> BytesIO:
        pass

    @abc.abstractmethod
    def get_object(self, file_key: str):
        pass

    @abc.abstractmethod
    def head_object(self, file_key: str):
        pass

    @abc.abstractmethod
    def list_objects(self, prefix) -> list:
        pass

    @abc.abstractmethod
    def copy_object(self, source_key, target_key, source_bucket_name: str = ''):
        pass

    @abc.abstractmethod
    def delete_object(self, file_key: str):
        pass

    @abc.abstractmethod
    def generate_sign_token(self, filetype: str, target_dir: str, expire_in: int = 300) -> dict:
        pass

    @abc.abstractmethod
    def generate_sign_url(self, method: str, key: str, expire_in: int = 600, slash_safe=True) -> str:
        pass


class MinIO(Oss):

    def __init__(self, *args, **kwargs):
        super(MinIO, self).__init__(*args, **kwargs)
        self.client = Minio(
            self.private_endpoint,
            access_key=self.access_key,
            secret_key=self.secret,
            secure=False,
        )
        self.public_client = Minio(
            self.public_endpoint,
            access_key=self.access_key,
            secret_key=self.secret,
            secure=self.use_https,
        )

    @cached_property
    def bucket_endpoint(self):
        url = self.public_endpoint
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'http://' + url
        return f'{url}/{self.bucket_name}'

    def path_join(self, *args):
        return '/'.join(args)

    def object_exists(self, file_key: str) -> bool:
        result = self.client.stat_object(self.bucket_name, file_key)
        return bool(result)

    def put_object_from_file(self, file_key: str, filepath: str) -> bool:
        result = self.client.fput_object(self.bucket_name, file_key, filepath)
        return bool(result)

    def get_object_to_file(self, file_key: str, filepath: str):
        return self.client.fget_object(self.bucket_name, file_key, filepath)

    def put_object_from_io(self, bytesio: BytesIO, file_key: str):
        result = self.client.put_object(self.bucket_name, file_key, bytesio, length=bytesio.getbuffer().nbytes)
        return bool(result)

    def get_object_to_io(self, file_key: str) -> BytesIO:
        buffer = BytesIO()
        data = self.get_object(file_key)
        buffer.write(data)
        buffer.seek(0)
        return buffer

    def get_object(self, file_key: str):
        response = None
        try:
            response = self.client.get_object(self.bucket_name, file_key)
            return response.data
        finally:
            if response:
                response.close()
                response.release_conn()

    def head_object(self, file_key: str):
        result = self.client.stat_object(self.bucket_name, file_key)
        tm = result.last_modified.timetuple()
        return OSSHeadObject(
            content_type=result.content_type,
            content_length=result.size,
            etag=result.etag,
            last_modified=calendar.timegm(tm)
        )

    def list_objects(self, prefix) -> list:
        if not prefix:
            return []
        objects = self.client.list_objects(self.bucket_name, prefix)
        return [obj.object_name for obj in objects]

    def delete_object(self, file_key: str):
        return self.client.remove_object(self.bucket_name, file_key)

    def copy_object(self, source_key, target_key, source_bucket_name: str = ''):
        pass

    def generate_sign_url(self, method: str, key: str, expire_in: int = 600, slash_safe=True) -> str:
        rand = random.randint(100000, 500000)
        return self.public_client.get_presigned_url(
            method, bucket_name=self.bucket_name, object_name=key,
            expires=datetime.timedelta(seconds=expire_in),
            extra_query_params={'cyborg': f'{rand}'}
        )

    def generate_sign_token(self, filetype: str, target_dir: str, expire_in: int = 300) -> dict:
        """sign upload token with specify file suffix, :filetype:
        """
        expiration = datetime.datetime.utcnow() + datetime.timedelta(seconds=expire_in)

        if target_dir.endswith(filetype):
            file_key = target_dir
        else:
            file_key = os.path.join(target_dir, f'{uuid4().hex}.{filetype}')

        policy = PostPolicy(self.bucket_name, expiration)
        policy.add_starts_with_condition("key", file_key)
        policy.add_content_length_range_condition(1024, 20 * 1024 * 1024)
        form_data = self.client.presigned_post_policy(policy)
        return {
            'host': self.bucket_endpoint,
            'file_key': file_key,
            'headers': form_data
        }


oss: Oss = MinIO(
    access_key=Settings.MINIO_ACCESS_KEY,
    secret=Settings.MINIO_ACCESS_SECRET,
    pub_endpoint=Settings.PUBLIC_ENDPOINT,
    private_endpoint=Settings.PRIVATE_ENDPOINT,
    bucket_name=Settings.BUCKET_NAME,
    use_https=Settings.USE_HTTPS
)


if __name__ == '__main__':
    url = oss.generate_sign_url('GET', 'partner/roche/analysis/193169bc-2dc3-4b0e-8734-20c017c9fa0f.h5')
    print(url)