import json

from sqlalchemy import Column, String, Integer, Text, Float

from cyborg.modules.user_center.user_core.domain.entities import CompanyEntity, COMPANY_AI_THRESHOLD, \
    COMPANY_DEFAULT_AI_THRESHOLD
from cyborg.seedwork.infrastructure.models import BaseModel


class UserModel(BaseModel):

    __tablename__ = 'user'

    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True, comment='主键ID')
    username = Column(String(255), comment='姓名')
    password = Column(Text, nullable=True, comment='密码')
    role = Column(String(255), nullable=True, comment='角色')
    company = Column(String(255), nullable=True, comment='组织')
    signed = Column(Integer, default=0, comment='是否签名')
    model_lis = Column(Text, default='[]', comment='算法模块')
    last_login = Column(Float, nullable=True, comment='上次登录时间')
    is_test = Column(Integer, default=0, comment='是否是测试用户')
    time_out = Column(String(255), nullable=True, comment='超时时间')
    volume = Column(Integer, nullable=True, comment='磁盘容量')
    expires = Column(Float, nullable=True, comment='过期时间')


class CompanyModel(BaseModel):

    __tablename__ = 'company'

    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, comment='主键ID')
    company = Column(String(255), nullable=True, comment='名字')
    model_lis = Column(Text, default='[]', comment='算法模块')
    remark = Column(Text, nullable=True, comment='备注')
    volume = Column(String(255), nullable=True, comment='组织磁盘容量')
    usage = Column(String(255), nullable=True, comment='组织磁盘用量')
    allowed_ip = Column(String(255), nullable=True, comment="")
    status = Column(Integer, default=1, comment="")
    create_time = Column(String(255), nullable=True, comment='创建时间')
    report_name = Column(String(255), nullable=True, comment="")
    report_info = Column(String(255), nullable=True, comment="")
    table_checked = Column(String(255), nullable=True, comment='列表需要展示的列名')
    ai_threshold = Column('aiThreshold', Text, nullable=True, default=json.dumps(COMPANY_AI_THRESHOLD), comment='算法阈值')
    default_ai_threshold = Column(
        'defaultAiThreshold', Text, nullable=True, default=json.dumps(COMPANY_DEFAULT_AI_THRESHOLD),
        comment='默认算法阈值')
    on_trial = Column('onTrial', Integer, default=0, comment='是否试用，试用为1，默认为不试用即为0')
    trial_times = Column('trialTimes', Text, nullable=True, comment='试用次数')
    label = Column(Integer, default=4, comment='1为不识别，2为识别条形码，3为识别二维码，4为ocr识别，默认为ocr识别')
    importable = Column(Integer, default=0, comment='是否支持导出，支持导出为1，默认为不支持导出即为0')
    export_json = Column('exportJson', Integer, default=0, comment='是否支持导出json，支持导出为1，默认为不支持导出即为0')
    is_test = Column('isTest', Integer, default=0, comment='0永久 1试用')
    end_time = Column('endTime', String(20), nullable=True, comment='试用的截止时间')
    smooth_value = Column(Text, nullable=True, comment='平滑值（pdl1调参专用）')
