from unittest import mock

from sqlalchemy.orm import sessionmaker

from cyborg.app.request_context import request_context
from cyborg.infra.mock.oss import MockedOss
from cyborg.infra.mock.session import mocked_engine
from cyborg.infra.oss import oss


class BaseTest(object):

    @classmethod
    def setup_class(cls):
        mocked_session = sessionmaker(autocommit=False, autoflush=True, expire_on_commit=False, bind=mocked_engine)()
        request_context.get_session = mock.Mock(return_value=mocked_session)
        cls.mocked_oss = MockedOss(
            access_key=oss.access_key,
            secret=oss.secret,
            pub_endpoint=oss.public_endpoint,
            private_endpoint=oss.private_endpoint,
            bucket_name=oss.bucket_name
        )

    @classmethod
    def teardown_class(cls):
        ...
