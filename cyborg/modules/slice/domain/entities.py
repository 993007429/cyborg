import datetime
import json
import logging
import os
import random
from typing import Optional, List

import aioshutil

from cyborg.app.settings import Settings
from cyborg.consts.common import Consts
from cyborg.infra.fs import fs
from cyborg.modules.slice.domain.value_objects import SliceImageType
from cyborg.seedwork.domain.entities import BaseDomainEntity
from cyborg.seedwork.domain.value_objects import AIType

logger = logging.getLogger(__name__)


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

    def get_image_url(self, image_type: SliceImageType):
        return f'{Settings.IMAGE_SERVER}/files/getImage?caseid={self.caseid}&fileid={self.fileid}&type={image_type.value}'

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

    @property
    def ai_type(self):
        return AIType.get_by_value(self.alg)

    @property
    def ai_diagnosis_state(self) -> Optional[int]:
        if self.ai_type in (AIType.tct, AIType.lct, AIType.dna) and self.ai_suggest:
            return 1 if '阳性' in self.ai_suggest else 2
        return None

    @property
    def checked_diagnosis_state(self) -> Optional[int]:
        if self.ai_type in (AIType.tct, AIType.lct, AIType.dna) and self.check_result:
            return 1 if '阳性' in self.check_result else 2
        return None

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
            'aiDiagnosisState': self.ai_diagnosis_state,
            'checkedDiagnosisState': self.checked_diagnosis_state
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
                "roilist": self.roilist,
                "ai_dict": self.ai_dict,
                "slide_quality": int(self.slide_quality) if self.slide_quality else None,
                "cell_num": self.cell_num,
                'templateId': self.template_id,
            })
        return d


class CaseRecordEntity(BaseDomainEntity):
    slices: List[SliceEntity] = []
    attachments: List[SliceEntity] = []
    opinion: str = ''
    dna_opinion: str = ''

    @property
    def json_fields(self) -> List[str]:
        return ['report_info']

    @classmethod
    def gen_case_id(cls):
        return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_' + str(random.randint(0, 1000000))

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

    def get_report_templates(self, template_config: List[dict]):
        if not self.slices:
            return []

        ai_types = [AIType.get_by_value(s.alg) for s in self.slices]
        template_codes = []
        for item in template_config:
            template_id = item['templateId']
            if template_id == 'reg':
                if len(self.slices) > 1:
                    template_codes.insert(0, item)
            elif template_id in ai_types:
                template_codes.append(item)
        return template_codes

    @property
    def report_opinion(self):
        slice_entity = self.slices[0] if self.slices else None
        if not slice_entity:
            return None
        opinion = self.opinion
        check_result = slice_entity.check_result

        ai_suggest = slice_entity.ai_suggest
        alg = slice_entity.alg if slice_entity.alg else ''

        temp_dict = {"阴性": "未见上皮内病变细胞或恶性细胞（NILM）",
                     "阳性HSIL": "高级别鳞状上皮内病变（HSIL）",
                     "阳性ASC-H": "不除外高级别鳞状上皮内病变细胞的非典型鳞状细胞（ASC-H）",
                     "阳性LSIL": "低级别鳞状上皮内病变（LSIL）",
                     "阳性ASC-US": "意义不明确的非典型鳞状细胞（ASC-US）",
                     "阳性ASCUS": "意义不明确的非典型鳞状细胞（ASC-US）",
                     "阳性AGC": "非典型腺细胞（不能明确意义）"}

        data = {'alg': alg}
        if opinion:
            data['opinion'] = opinion

            if alg == 'dna':
                mid_data = {}
                if self.report_info:
                    mid_data = self.report_info

                data['uniteOpinion'] = ''
                if 'DNA' in mid_data and mid_data.get('DNA').get('DNAOpinion'):
                    data['DNAOpinion'] = mid_data.get('DNA').get('DNAOpinion')
                    return data

                if check_result and ';' in check_result:
                    if check_result.split(';')[1]:
                        data['DNAOpinion'] = check_result.split(';')[1]
                    else:
                        if ai_suggest and ';' in ai_suggest:
                            data['DNAOpinion'] = ai_suggest.split(';')[1]

                if ai_suggest and ';' in ai_suggest:
                    data['DNAOpinion'] = ai_suggest.split(';')[1]

        elif check_result:
            if ';' in check_result:
                if check_result.split(';')[1]:
                    data['DNAOpinion'] = check_result.split(';')[1]
                else:
                    if ai_suggest and ';' in ai_suggest:
                        data['DNAOpinion'] = ai_suggest.split(';')[1]
                check_result = check_result.split(';')[0]

            check_result = check_result.replace(
                '霉菌', '').replace('滴虫', '').replace('线索', '').replace('放线菌','').replace('疱疹', '').replace(
                '巨细胞病毒', '').replace('HPV', '').replace('萎缩', '').replace('炎症', '').replace('修复', '').replace(
                '出血', '').replace('化生', '').replace(',', '').replace(' ', '')

            check_result_dict = check_result.split('+')
            on = check_result_dict[0].split('性')[0]
            res = ''
            if on == '阳':
                i = 0
                while i <= len(check_result_dict) - 1:
                    if i != 0:
                        res += temp_dict.get('阳性' + check_result_dict[i], '') + '\n'
                    else:
                        res += temp_dict.get(check_result_dict[i], '') + '\n'
                    i += 1
            else:
                res = temp_dict.get(check_result, '')
            data['opinion'] = res

        elif ai_suggest:
            if ';' in ai_suggest:
                data['DNAOpinion'] = ai_suggest.split(';')[1]
                ai_suggest = ai_suggest.split(';')[0]

            ai_suggest = ai_suggest.replace('霉菌', '').replace('滴虫', '').replace('线索', '').replace('放线菌', ''). \
                replace('疱疹', '').replace('巨细胞病毒', '').replace('HPV', '').replace('萎缩', '').replace('炎症',
                                                                                                             ''). \
                replace('修复', '').replace(',', '').replace(' ', '')
            res = temp_dict.get(ai_suggest, '')
            data['opinion'] = res

        return data

    def gen_report_info(self):
        slice_entity = self.slices[0] if self.slices else None
        report_info = self.report_info or {}
        opinion_data = self.report_opinion

        if opinion_data:
            alg = 'REG'
            if 'tct' in opinion_data['alg'] or 'lct' in opinion_data['alg']:
                alg = 'LCT'
            elif 'dna' in opinion_data['alg']:
                alg = opinion_data['alg'].upper()
                report_info['pageType'] = 'TBSPage'

            report_info['reportType'] = alg
            report_info['opinion'] = opinion_data.get('opinion')

            if slice_entity:
                report_info['isStatisfied'] = 1 if slice_entity.slideQuality else 0
                if slice_entity.cell_num:
                    report_info['cellNum'] = 'greater' if slice_entity.cellNum or 0 > 5000 else 'less'
                items = slice_entity.check_result.split(' ')
                report_info['pathogen'] = items[2].split(',') if len(items) >= 3 else []

            report_info.update({
                'opinion': opinion_data.get('opinion', ''),
                'DNAOpinion': opinion_data.get('DNAOpinion', '')
            })

        return report_info

    async def remove_data_dir(self):
        await aioshutil.rmtree(self.data_dir)

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

    def to_dict_for_report(self):
        d = self.basic_info
        d.update({
            'attachments': [entity.to_dict() for entity in self.attachments],
            # 'slices': [entity.to_dict() for entity in self.slices],
            'reports': self.reports if self.slices else [],
            'slices_count': self.slice_count,
            'create_time': self.create_time,
            'update': self.update_time,
        })
        slice_entity = self.slices[0] if self.slices else None
        if slice_entity:
            d.update({
                'sliceNumber': slice_entity.slice_number,
                'histplot': slice_entity.get_image_url(image_type=SliceImageType.histplot),
                'scatterplot': slice_entity.get_image_url(image_type=SliceImageType.scatterplot)
            })

        report_info = self.gen_report_info()
        if report_info:
            d.update(report_info)

        return d

    def to_dict(self):
        d = {
            'id': self.caseid,
            'basic': self.basic_info,
            'attachments': [entity.to_dict() for entity in self.attachments],
            'slices': [entity.to_dict() for entity in self.slices],
            'reports': self.reports if self.slices else [],
            'slices_count': self.slice_count,
            'create_time': self.create_time,
            'update': self.update_time,
            'reportInfo': self.report_info,
            'reportUid': self.id
        }
        return d


class ReportConfigEntity(BaseDomainEntity):

    @classmethod
    def new_entity(cls, company: str):
        return ReportConfigEntity(raw_data={
            'company': company,
            'template_config': ReportConfigEntity.new_template_config()
        })

    @classmethod
    def new_template_config(cls):
        template_config = [{'templateId': 'reg', 'templateCode': ''}]
        template_config.extend(
            [{'templateId': ai_type.value, 'templateCode': ''} for ai_type in [
                AIType.tct, AIType.dna, AIType.pdl1, AIType.ki67, AIType.ki67hot, AIType.er, AIType.pr, AIType.her2,
                AIType.fish_tissue, AIType.np, AIType.cd30
        ]])
        return template_config

    @classmethod
    def format_template_config_item(cls, template_config_item: dict):
        ai_type = AIType.get_by_value(template_config_item['templateId'])
        template_config_item['templateName'] = ai_type.display_name if ai_type else '通用模板'

    def to_dict(self):
        for item in self.template_config:
            self.format_template_config_item(item)
        return super().to_dict()
