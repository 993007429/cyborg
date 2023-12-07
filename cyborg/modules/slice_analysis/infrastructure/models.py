from typing import Type

from sqlalchemy import Column, Integer, Text, Float, String, JSON

from cyborg.seedwork.infrastructure.models import BaseModel


class AIModel(BaseModel):
    __tablename__ = 'ai'

    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, comment="主键ID")
    ai_name = Column(String(255), comment="AI模块名")


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


class MarkModel(BaseModel):
    __abstract__ = True

    __table_args__ = {'extend_existing': True}

    __import_table_name__ = ''

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键ID')
    position = Column(JSON, nullable=True, comment='标注点位置')
    method = Column(Text, nullable=True, comment='标注工具，例如自由笔')
    is_export = Column('isExport', Integer, nullable=True, comment='导出到报告1，否则为0')
    remark = Column(Text, nullable=True, comment='判读结果')
    ai_result = Column('aiResult', JSON, nullable=True, comment='算法结果')
    editable = Column(Integer, nullable=True, comment='可编辑为1，不可编辑为0')
    stroke_color = Column('strokeColor', Text, nullable=True, comment='边框颜色')
    fill_color = Column('fillColor', Text, nullable=True, comment='填充颜色')
    mark_type = Column('markType', Integer, nullable=True, comment='标注类型（手动标注1、算法标注2、算法标注区域3）')
    diagnosis = Column(JSON, nullable=True, comment="")
    radius = Column(Float, nullable=True, comment='标注直径')
    create_time = Column('createTime', Float, nullable=True, comment='标注创建时间')
    group_id = Column('groupId', Integer, nullable=True, comment='标注组id')
    area_id = Column('areaId', Integer, nullable=True, comment='算法区域标注id')
    dashed = Column(Integer, nullable=True, comment='虚线为1，实线为0')
    doctor_diagnosis = Column('doctorDiagnosis', JSON, comment='医生手工判读')


class MarkToTileModel(BaseModel):
    __abstract__ = True

    __table_args__ = {'extend_existing': True}

    __import_table_name__ = ''

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键ID')
    mark_id = Column('markId', Integer, comment='标注点id')
    tile_id = Column('tileId', Integer, comment='瓦片id')


def get_ai_mark_model(table_name_suffix: str) -> Type[MarkModel]:
    class AIMarkModel(MarkModel):
        __tablename__ = f'Mark_{table_name_suffix}'
        __import_table_name__ = __tablename__.replace('_label', '') + '_1'  # 对应的算法标注导入源表
    return AIMarkModel


def get_ai_mark_to_tile_model(table_name_suffix: str) -> Type[MarkToTileModel]:
    class AIMarkToTileModel(MarkToTileModel):
        __tablename__ = f'MarkToTile_{table_name_suffix}'
        __import_table_name__ = __tablename__.replace('_label', '') + '_1'  # 对应的算法标注导入源表

    return AIMarkToTileModel


class MarkGroupModel(BaseModel):
    __tablename__ = 'MarkGroup'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键ID')
    group_name = Column('groupName', Text, nullable=True, comment="")
    shape = Column(Text, nullable=True, comment="")
    color = Column(Text, nullable=True, comment="")
    create_time = Column('createTime', Float(64), nullable=True, comment='创建时间')
    is_template = Column('isTemplate', Integer, nullable=True, comment="")
    is_selected = Column('isSelected', Integer, nullable=True, comment="")
    selectable = Column(Integer, nullable=True, comment="")
    editable = Column(Integer, nullable=True, comment="")
    is_ai = Column('isAi', Integer, nullable=True, comment="")
    parent_id = Column('parentId', Integer, nullable=True, comment="")
    template_id = Column('templateId', Integer, nullable=True, comment="")
    op_time = Column('opTime', Float)
    default_color = Column('defaultColor', Text)
    is_empty = Column('isEmpty', Integer)
    is_show = Column('isShow', Integer)
    is_import = Column('isImport', Integer)


class Pdl1sCountModel(BaseModel):
    __tablename__ = "Pdl1sCount"

    tile_id = Column('tileId', Integer, primary_key=True)
    pos_tumor = Column('posTumor', Integer, nullable=True)
    neg_tumor = Column('negTumor', Integer, nullable=True)
    pos_norm = Column('posNorm', Integer, nullable=True)
    neg_norm = Column('negNorm', Integer, nullable=True)


class NPCountModel(BaseModel):
    __tablename__ = "NPCount"

    tile_id = Column('tileId', Integer, primary_key=True)
    eosinophils = Column(Integer, nullable=True)
    lymphocyte = Column(Integer, nullable=True)
    plasmocyte = Column(Integer, nullable=True)
    neutrophils = Column(Integer, nullable=True)


class ChangeRecordModel(BaseModel):
    __tablename__ = 'ChangeRecord'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键ID')
    mark_id = Column('markId', Integer, comment='标注id')
    content = Column('content', Text, nullable=True, comment='变更内容')
    table_name = Column('tableName', Text, nullable=True, comment='操作表')
    op_type = Column('opType', Text, nullable=True, comment='操作类型')
    op_name = Column('opName', Text, nullable=True, comment='操作人')


class ShareMarkGroupModel(BaseModel):
    __tablename__ = 'mark_group'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键ID')
    group_name = Column('groupName', Text, nullable=True, comment="")
    shape = Column(Text, nullable=True, comment="")
    color = Column(Text, nullable=True, comment="")
    create_time = Column('createTime', Float(64), nullable=True, comment='创建时间')
    is_template = Column('isTemplate', Integer, nullable=True, comment="")
    is_selected = Column('isSelected', Integer, nullable=True, comment="")
    selectable = Column(Integer, nullable=True, comment="")
    editable = Column(Integer, nullable=True, comment="")
    is_ai = Column('isAi', Integer, nullable=True, comment="")
    parent_id = Column('parentId', Integer, nullable=True, comment="")
    template_id = Column('templateId', Integer, nullable=True, comment="")
    op_time = Column('opTime', Float)
    default_color = Column('defaultColor', Text)
    is_empty = Column('isEmpty', Integer)
    is_show = Column('isShow', Integer)
    is_import = Column('isImport', Integer)
