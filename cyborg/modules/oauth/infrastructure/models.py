"""用户/账号"""

from sqlalchemy import (
    Column, Integer, String, DateTime, func,
)

from cyborg.seedwork.infrastructure.models import BaseModel


class OAuthApplicationModel(BaseModel):
    __tablename__ = 'oauth_application'

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_modified = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    name = Column(String(50), nullable=False, comment='应用名')
    client_id = Column(String(20), nullable=False, unique=True)
    client_secret = Column(String(40), nullable=False)
    user_id = Column(Integer, nullable=False, server_default='0')

    def to_dict(self):
        return {
            'name': self.name,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        }
