import json

from sqlalchemy import Column, String, Integer, Float, JSON, SmallInteger, DateTime, Index, Boolean, func, Text

from cyborg.seedwork.infrastructure.models import BaseModel
from cyborg.modules.user_center.user_core.domain.entities import COMPANY_AI_THRESHOLD


class AIModel(BaseModel):

    __tablename__ = 'ai'

    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, comment="主键ID")
    ai_name = Column(String(255), comment="AI模块名")


class AIPatternModel(BaseModel):

    __tablename__ = 'ai_pattern'

    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, comment="主键ID")
    ai_name = Column(String(255), comment="AI模块名")
    name = Column(String(255), comment="模块模式名称")
    model_name = Column(String(255), comment="模型名称")
    ai_threshold = Column(JSON, nullable=True, comment='算法阈值')
    company = Column(String(255), comment="组织名称")


class AIStatisticsModel(BaseModel):

    __table_args__ = {'extend_existing': True}

    __tablename__ = 'statistics'

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True, comment="主键ID")
    date = Column(String(255), nullable=True, comment="月份")
    total_count = Column('totalCount', Integer, nullable=True, comment="总用量")
    negative_count = Column('negativeCount', Integer, nullable=True, comment="阴性数量")
    positive_count = Column('positiveCount', Integer, nullable=True, comment="阳性数量")
    abnormal_count = Column('abnormalCount', Integer, nullable=True, comment="异常数量")
    total_count_dr = Column('totalCountDr', Integer, nullable=True, comment="总用量（去重）")
    negative_count_dr = Column('negativeCountDr', Integer, nullable=True, comment="阴性数量（去重）")
    positive_count_dr = Column('positiveCountDr', Integer, nullable=True, comment="阳性数量（去重）")
    abnormal_count_dr = Column('abnormalCountDr', Integer, nullable=True, comment="异常数量（去重）")
    company = Column(String(255), nullable=True, comment="组织名称")
    ai_type = Column('aiType', String(255), nullable=True, comment="算法类型")
    version = Column(String(255), nullable=True, comment="版本号")


class TemplateModel(BaseModel):

    __tablename__ = 'template'

    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    template_name = Column('templateName', String(255))
    create_time = Column('createTime', Float)
    is_selected = Column('isSelected', Integer)
    ai_id = Column('aiId', Integer)
    has_imported = Column('hasImported', Integer, default=0)
    is_multi_mark = Column('isMultiMark', Integer, default=0, comment='是否允许选择多标注组')


class TCTProbModel(BaseModel):

    __table_args__ = {'extend_existing': True}

    __tablename__ = 'tctprob'

    id = Column(Integer, primary_key=True, comment="主键ID")
    slice_id = Column(Integer, comment="切片ID")
    prob_nilm = Column(Float, comment='NILM 概率')
    prob_ascus = Column(Float, comment='ASCUS 概率')
    prob_lsil = Column(Float, comment='LSIL 概率')
    prob_asch = Column(Float, comment='ASCH 概率')
    prob_agc = Column(Float, comment='AGC 概率')
    prob_hsil = Column(Float, comment='HSIL 概率')


class AITaskModel(BaseModel):

    __tablename__ = 'ai_task'

    __table_args__ = (
        Index('idx_case_id_file_id_ai_type', 'case_id', 'file_id', 'ai_type'),
    )

    id = Column(Integer, primary_key=True, comment="主键ID")
    ai_type = Column(String(20), nullable=False)
    case_id = Column(String(255), nullable=False)
    file_id = Column(String(255), nullable=False)
    status = Column(SmallInteger, nullable=False, default=0)
    rois = Column(JSON)
    model_info = Column(JSON)
    is_calibrate = Column(Boolean, nullable=False, default=False)
    template_id = Column(Integer, nullable=False, default=0)
    result_id = Column(String(50), nullable=False, default='')
    expired_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_modified = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
