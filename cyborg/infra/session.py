import functools
import json
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from cyborg.app.settings import Settings
from cyborg.utils.encoding import CyborgJsonEncoder

logger = logging.getLogger(__name__)


def json_serializer(val):
    return json.dumps(val, cls=CyborgJsonEncoder)


def json_deserializer(val):
    return json.loads(val) if val else None


_engine = create_engine(
    Settings.SQLALCHEMY_DATABASE_URI, json_serializer=json_serializer, json_deserializer=json_deserializer,
    pool_recycle=3600, echo=False)
_Session = sessionmaker(autocommit=False, autoflush=True, expire_on_commit=False, bind=_engine)


def get_session() -> Session:
    return _Session()


def get_session_by_db_uri(uri: str):
    engine = create_engine(
        uri, json_serializer=json_serializer, json_deserializer=json_deserializer, pool_recycle=300, echo=False)
    return Session(autocommit=False, autoflush=True, expire_on_commit=False, bind=engine)


def transaction(f):

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        from cyborg.app.request_context import request_context
        session: Session = request_context.db_session.get()
        assert session is not None

        if request_context.is_in_transaction:
            return f(*args, **kwargs)
        else:
            request_context.is_in_transaction = True

        ret = None
        try:
            ret = f(*args, **kwargs)
            session.commit()
        except Exception as e:
            logger.exception(e)
            session.rollback()
        request_context.is_in_transaction = False
        return ret

    return wrapper
