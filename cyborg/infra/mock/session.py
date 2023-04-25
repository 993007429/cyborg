from sqlalchemy import create_engine

from seal.infra import session

UT_DB_URL = 'sqlite:///:memory:'

mocked_engine = create_engine(UT_DB_URL, json_serializer=session.json_serializer, pool_recycle=3600, echo=False)
