from sqlalchemy import Column, Integer, String, SmallInteger, JSON, DateTime

from cyborg.seedwork.infrastructure.models import BaseModel


class RocheAITaskModel(BaseModel):

    __tablename__ = 'roche_ai_task'

    __table_args__ = (
        # Index('idx_case_id_file_id_ai_type', 'case_id', 'file_id', 'ai_type'),
    )

    id = Column(Integer, primary_key=True, comment="主键ID")
    ai_type = Column(String(20), nullable=False)
    slide_url = Column(String(1000), nullable=False)
    status = Column(SmallInteger, nullable=False, default=0)
    rois = Column(JSON)
    model_info = Column(JSON)
    # template_id = Column(Integer, nullable=False, default=0)
    result_id = Column(String(50), nullable=False, default='')
    expired_at = Column(DateTime, nullable=True)
