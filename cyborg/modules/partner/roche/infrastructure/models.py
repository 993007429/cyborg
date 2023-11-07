from sqlalchemy import Column, Integer, String, SmallInteger, JSON, DateTime, func, Text

from cyborg.seedwork.infrastructure.models import BaseModel


class RocheAITaskModel(BaseModel):

    __tablename__ = 'roche_ai_task'

    __table_args__ = (
        # Index('idx_case_id_file_id_ai_type', 'case_id', 'file_id', 'ai_type'),
    )

    id = Column(Integer, primary_key=True, comment="主键ID")
    analysis_id = Column(String(50), nullable=False, unique=True)
    algorithm_id = Column(String(50), nullable=False)
    ai_type = Column(String(20), nullable=False)
    slide_url = Column(Text, nullable=False)
    input_info = Column(JSON, nullable=False)
    status = Column(SmallInteger, nullable=False, default=0)
    coordinates = Column(JSON)
    model_info = Column(JSON)
    ai_results = Column(JSON)
    # template_id = Column(Integer, nullable=False, default=0)
    result_id = Column(String(50), nullable=False, default='')
    started_at = Column(DateTime, nullable=True)
    expired_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_modified = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
