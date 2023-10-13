from sqlalchemy import Column, Integer, String, Text, Float, JSON, TIMESTAMP

from cyborg.seedwork.infrastructure.models import BaseModel


class CaseRecordModel(BaseModel):
    __tablename__ = 'record'

    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True, comment='主键ID')
    caseid = Column(String(255), nullable=True, comment='病例号')
    name = Column(String(255), nullable=True, comment='姓名')
    age = Column(Integer, nullable=True, comment='年龄')
    gender = Column(String(255), nullable=True, comment='性别')
    cancer_type = Column('cancerType', Text, nullable=True, comment='当前诊断')
    sample_num = Column('sampleNum', String(255), nullable=True, comment='样本号')
    family_history = Column('familyHistory', Text, nullable=True, comment='家族史')
    medical_history = Column('medicalHistory', Text, nullable=True, comment='病史')
    sample_part = Column('samplePart', Text, nullable=True, comment='取样部位')
    sample_time = Column('sampleTime', String(255), nullable=True, comment='采样日期')
    sample_collect_date = Column('sampleCollectDate', String(255), nullable=True, comment='收样日期')
    sample_type = Column('sampleType', String(255), nullable=True, comment='样本类型')
    generally_seen = Column('generallySeen', Text, nullable=True, comment='大体所见')
    inspection_hospital = Column('inspectionHospital', String(255), nullable=True, comment="")
    inspection_doctor = Column('inspectionDoctor', String(255), nullable=True, comment="")
    report_info = Column('reportInfo', Text, nullable=True, comment="")
    opinion = Column(Text, nullable=True, comment="")
    update_time = Column(String(255), nullable=True, comment="")
    stage = Column(Integer, nullable=True, comment="")
    slice_count = Column(Integer, nullable=True, comment='切片数量')
    started = Column(Integer, default=0, comment='是否AI处理')
    state = Column(Integer, default=0, comment="")
    analy = Column(Text, nullable=True, comment="")
    company = Column(String(255), nullable=True, comment='组织名')
    report = Column(String(255), default='2', comment='有无报告,默认2为无')
    create_time = Column(String(255), nullable=True, comment="")


class SliceModel(BaseModel):
    __tablename__ = 'slice'

    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True, comment='主键ID')
    fileid = Column(String(255), nullable=True, comment='切片id')
    caseid = Column(String(255), nullable=True, comment='病例号')
    filename = Column(String(255), nullable=True, comment='切片名')
    name = Column(String(255), nullable=True, comment='文件名')
    loaded = Column(String(255), nullable=True, comment='切片上传进度')
    total = Column(String(255), nullable=True, comment='切片大小，用于前端上传进度显示')
    stain = Column(String(255), nullable=True, comment="")
    state = Column(Integer, default=1, comment='切片状态')
    mppx = Column(Float, nullable=True, comment="")
    mppy = Column(Float, nullable=True, comment="")
    height = Column(Integer, nullable=True, comment="")
    width = Column(Integer, nullable=True, comment="")
    tool_type = Column('toolType', String(255), nullable=True, comment='正在使用模块名称')
    started = Column(Integer, default=0, comment='切片处理状态')
    objective_rate = Column(String(255), nullable=True, comment='倍率')
    radius = Column(Float, default=1, comment='标注模块点状标注直径')  # 默认系数是1
    is_solid = Column('isSolid', Integer, default=1, comment='标注模块点状标注实心空心状态，实心为1')  # 默认标注模块中显示实心
    company = Column(String(255), nullable=True, comment='组织')
    ajax_token = Column('ajaxToken', Text, default='{}', comment="")
    path = Column(Text, nullable=True, comment="")
    type = Column(String(255), default='slice', comment='类型')  # slice切片 attachment附件
    update_time = Column(String(255), nullable=True, comment='更新时间')
    operator = Column(String(255), nullable=True, comment="")
    alg = Column(String(255), nullable=True, comment='算法类型')
    ai_suggest = Column(String(255), default='', comment='AI建议结果')
    check_result = Column(String(255), default='', comment='人工复核结果')
    slice_number = Column(String(255), nullable=True, comment='样本号')
    clarity = Column(String(255), nullable=True, comment='清晰度')
    position = Column(String(255), default='{}', comment='切片详情视野坐标')
    roilist = Column(JSON, default=[], comment='ROI列表')
    ai_dict = Column(JSON, default={}, comment='各个模块计算状态')
    cell_num = Column('cellNum', Integer, nullable=True, comment='细胞数量')
    slide_quality = Column('slideQuality', String(255), nullable=True, comment='切片质量')
    csUpload_status = Column('csUploadStatus', Integer, nullable=True, default=0,
                             comment='上传众包状态（0未上传，1已上传）')
    csAll_finished = Column('csAllFinished', Integer, nullable=True, default=0,
                            comment='完成众包模式下所有流程（0未完成，1已完成）')
    is_cs = Column('isCs', Integer, nullable=True, default=0, comment='众包模式下上传切片为1')
    high_through = Column('highThrough', Integer, nullable=True, default=0, comment='高通量上传切片为1')
    ai_status = Column('aiStatus', Integer, nullable=True, comment='算法计算完成为1')
    uid = Column(String(255), default='{}', comment='切片唯一标识')
    user_file_path = Column('userFilePath', Text, default='', comment='上传端文件路径')
    user_file_folder = Column('userFileFolder', String(255), default='', comment='上传端文件夹')
    template_id = Column('templateId', Integer, nullable=True, default=1, comment='当前切片选中的标注模板id')
    is_has_label = Column('isHasLabel', Integer, nullable=True, default=0, comment='是否有切片标签')
    upload_batch_number = Column('uploadBatchNumber', String(255), default='', comment='高通量上传批次号')
    ipaddress = Column(String(255), default='', comment='切片上传客户端ip和端口')
    as_id = Column('asId', Integer, comment='用量统计记录id，用于算法用量统计去重')
    ai_angle = Column('aiAngle', Float, default=0, comment='算法给出的旋转角度')
    current_angle = Column('currentAngle', Float, default=0, comment='当前的旋转角度')
    last_modified = Column(TIMESTAMP, nullable=False)


class ReportConfigModel(BaseModel):
    __tablename__ = 'report_config'

    id = Column(Integer, primary_key=True, comment='主键ID')
    company = Column(String(255), nullable=True, unique=True, comment='名字')
    template_config = Column(JSON, nullable=True, default=[], comment='报告模版配置')
