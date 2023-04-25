import datetime
import json
import os
from typing import Optional, List

from cyborg.app.settings import Settings
from cyborg.infra.fs import fs
from cyborg.seedwork.domain.entities import BaseDomainEntity


class SliceEntity(BaseDomainEntity):

    @property
    def record_dir(self) -> str:
        return fs.path_join(Settings.DATA_DIR, self.company, 'data', self.caseid)

    @property
    def slice_dir(self) -> str:
        return fs.path_join(self.record_dir, 'slices', self.fileid)

    @property
    def attachment_dir(self) -> str:
        return fs.path_join(self.record_dir, 'attachments', self.fileid)

    @property
    def screenshot_dir(self) -> str:
        return fs.path_join(self.slice_dir, 'screenshot')

    @property
    def roi_dir(self) -> str:
        return fs.path_join(self.slice_dir, 'roi')

    @property
    def slice_file_path(self) -> str:
        return fs.path_join(self.slice_dir, self.filename)

    @property
    def thumb_file_path(self) -> str:
        return fs.path_join(self.slice_dir, 'thumbnail.jpeg')

    @property
    def attachment_file_path(self) -> str:
        return fs.path_join(self.attachment_dir, self.filename)

    @property
    def db_file_path(self) -> str:
        return fs.path_join(self.slice_dir, 'slice.db')

    @property
    def slice_label_path(self):
        return fs.path_join(self.slice_dir, 'slice_label.jpg')

    def get_tile_path(self, x: int, y: int, z: int) -> str:
        return os.path.join(
            self.slice_dir, os.path.splitext(self.filename)[0] + '_files', str(z), "{}_{}.jpeg".format(x, y))

    def get_screenshot_path(self, roi_id: str) -> str:
        return fs.path_join(self.screenshot_dir, f'{roi_id}.jpg')

    def get_roi_path(self, roi_id: str) -> str:
        return fs.path_join(self.roi_dir, f'{roi_id}.jpg')

    def update_ai_result(self, ai_suggest: dict) -> bool:
        # 只有tct和lct算法有复核结果覆盖ai建议的情况
        if Settings.COVER_RESULT and self.alg is not None and (
                self.alg.startswith('tct') or self.alg.startswith('lct')):
            # TODO 更新切片db
            """
            response, db_doc_path, slice_doc_path, _, caseid, fileid, db_connection = get_function_init_params(
                request=request, use_dbm=False)
            with db_connection:
                op = get_class(ai_type=slice_obj.alg[0:3],
                               session=db_connection.session)()  # 获取ORM类
                microbe = ai_suggest.split(' ')[-1].split(',')
                diagnosis_new = ai_suggest.strip(ai_suggest.split(' ')[-1]).strip().split(' ')
                if len(diagnosis_new) < 2:
                    diagnosis_new = diagnosis_new + [""] * (2 - len(diagnosis_new))
                if len(microbe) < 1:
                    microbe = microbe + [""]
                area_mark = db_connection.session.query(op.Mark).filter(
                    op.Mark.markType == 3).first()
                aiResult = json.loads(area_mark.aiResult)
                aiResult['diagnosis'] = diagnosis_new
                aiResult['microbe'] = microbe
                area_mark.aiResult = json.dumps(aiResult)
                cover = True
            """
            return True
        return False

    @property
    def data_paths(self):
        return {
            'slice_dir': self.slice_dir,
            'attachment_dir': self.attachment_dir,
            'slice_file_path': self.slice_file_path,
            'db_file_path': self.db_file_path
        }

    def to_dict(self, all_fields: bool = False):
        d = {
            'id': self.fileid,
            'uid': self.id,
            "ai_suggest": self.ai_suggest,
            "alg": self.alg,
            "check_result": self.check_result,
            'filename': self.filename,
            'fileid': self.fileid,
            "caseid": self.caseid,
            "operator": self.operator,
            "slice_number": self.slice_number,
            "started": self.started,
            'state': self.state,
            'name': self.name,
            'loaded': self.loaded,
            'total': self.total,
            'userFileFolder': self.user_file_folder,
            'userFilePath': self.user_file_path,
            'width': self.width,
            'height': self.height,
            'radius': self.radius,  # 默认系数是1
            'is_solid': self.is_solid,  # 默认标注模块中显示实心
            'ai_status': self.ai_status,
            'as_id': self.ai_id,
            "update_time": self.update_time,
        }

        d.update(self.data_paths)

        if all_fields:
            d.update({
                'stain': self.stain,
                'mppx': self.mppx,
                'mppy': self.mppy,
                'toolType': self.tool_type,
                'objective_rate': self.objective_rate,
                "company": self.company,
                "ajaxToken": json.loads(self.ajax_token),
                "path": self.path,
                "type": self.type,
                "clarity": self.clarity,
                "position": json.loads(self.position) if self.position else {},
                "roilist": json.loads(self.roilist),
                "ai_dict": json.loads(self.ai_dict),
                "slide_quality": int(self.slide_quality) if self.slide_quality else None,
                "cell_num": self.cell_num,
                'templateId': self.template_id,
            })
        return d


class CaseRecordEntity(BaseDomainEntity):

    slices: Optional[List[SliceEntity]] = None

    @property
    def data_dir(self) -> str:
        return os.path.join(Settings.DATA_DIR, self.company, 'data', self.caseid)

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.data_dir, 'reports')

    @property
    def reports(self) -> List[dict]:
        items = []
        if not os.path.exists(self.reports_dir):
            return items

        report_ids = os.listdir(self.reports_dir)
        for r_id in report_ids:
            path = os.path.join(self.reports_dir, r_id)
            if not os.path.isdir(path):
                continue
            timestamp = datetime.datetime.utcfromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%dT%H:%M:%SZ")
            item = {
                'id': r_id,
                'date': timestamp
            }
            items.append(item)
        return items

    @property
    def basic_info(self):
        # 病例的一些基础信息
        return {
            'caseid': self.sample_num,
            'name': self.name,
            'age': self.age,
            'cancerType': self.cancer_type,
            'familyHistory': self.family_history,
            'gender': self.gender,
            'generallySeen': self.generally_seen,
            'inspectionDoctor': self.inspection_doctor,
            'inspectionHospital': self.inspection_hospital,
            'medicalHistory': self.medical_history,
            'sampleTime': self.sample_time,
            'sampleCollectDate': self.sample_collect_date,
            'sampleType': self.sample_type,
            'samplePart': self.sample_part
        }

    def to_dict(self):
        d = {
            'id': self.caseid,
            'basic': self.basic_info,
            'attachments': [],
            'slices': [],
            'reports': [],
            'slices_count': self.slice_count,
            'create_time': self.create_time,
            'update': self.update_time,
            'reportInfo': json.loads(self.report_info) if self.report_info else None,

        }

        for entity in self.slices or []:
            if entity.type == 'slice':
                d['slices'].append(entity.to_dict())
            else:
                d['attachments'].append(entity.to_dict())
            d['reports'] = self.reports
        return d
