import datetime
import json
import logging
import os
import random
from typing import Optional, List

import aioshutil

from cyborg.app.settings import Settings
from cyborg.infra.cache import cache
from cyborg.infra.fs import fs
from cyborg.libs.heimdall.SlideBase import SlideBase
from cyborg.modules.slice.domain.value_objects import SliceImageType
from cyborg.seedwork.domain.entities import BaseDomainEntity
from cyborg.seedwork.domain.value_objects import AIType

logger = logging.getLogger(__name__)


class SliceEntity(BaseDomainEntity):
    slide: Optional[SlideBase] = None

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
        return f'{Settings.IMAGE_SERVER}/files/getImage2?caseid={self.caseid}&fileid={self.fileid}&type={image_type.value}'

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

    @classmethod
    def fix_ai_name(cls, ai_name: str) -> str:
        if ai_name.startswith('fish'):
            return 'fish'
        return ai_name

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

    @property
    def parsed_ai_suggest(self) -> dict:
        """将tct lct dna模块的ai_suggest字符串解析成字典"""
        ai_suggest_dict = {
            "diagnosis": [],
            "microbe": [],
            "dna_diagnosis": "",
            "flag": 1
        }
        ai_suggest = self.ai_suggest
        try:
            diagnosis_microbe = ai_suggest.split(";")[0].replace("  ", " ")
            if ";" in ai_suggest:
                ai_suggest_dict["dna_diagnosis"] = ai_suggest.split(";")[-1]
            if "阴性" in ai_suggest:
                if "-样本不满意" in ai_suggest:
                    temp_list = diagnosis_microbe.split(" ")
                    if len(temp_list) == 2:
                        ai_suggest_dict["diagnosis"] = ["阴性", "-样本不满意"]
                        ai_suggest_dict["microbe"] = []
                    elif len(temp_list) == 3:
                        ai_suggest_dict["diagnosis"] = ["阴性", "-样本不满意"]
                        ai_suggest_dict["microbe"] = diagnosis_microbe.split(" ")[-1].split(",")
                    else:
                        ai_suggest_dict["flag"] = 0
                        print(f"解析失败: {ai_suggest}")
                else:
                    temp_list = diagnosis_microbe.split(" ")
                    if len(temp_list) == 1:
                        ai_suggest_dict["diagnosis"] = ["阴性", ""]
                        ai_suggest_dict["microbe"] = []
                    elif len(temp_list) == 2:
                        ai_suggest_dict["diagnosis"] = ["阴性", ""]
                        ai_suggest_dict["microbe"] = diagnosis_microbe.split(" ")[-1].split(",")
                    else:
                        ai_suggest_dict["flag"] = 0
                        print(f"解析失败: {ai_suggest}")
            elif "阳性" in ai_suggest:
                temp_list = diagnosis_microbe.split(" ")
                if len(temp_list) == 2:
                    ai_suggest_dict["diagnosis"] = [temp_list[0], temp_list[1]]
                    ai_suggest_dict["microbe"] = []
                elif len(temp_list) == 3:
                    ai_suggest_dict["diagnosis"] = [temp_list[0], temp_list[1]]
                    ai_suggest_dict["microbe"] = diagnosis_microbe.split(" ")[-1].split(",")
                else:
                    ai_suggest_dict["flag"] = 0
                    print(f"解析失败: {ai_suggest}")
            else:
                ai_suggest_dict["flag"] = 0
                print(f"ai建议(tct)格式非法: {ai_suggest}")
        except Exception as e:
            ai_suggest_dict["flag"] = 0
            print(f"解析 {ai_suggest} 失败: {e}")
        return ai_suggest_dict

    @property
    def is_ai_suggest_hacked(self):
        return self.origin_ai_sugguest and self.ai_suggest != self.origin_ai_sugguest

    @property
    def is_slide_quality_hacked(self):
        return self.origin_slide_quality and self.slide_quality != self.origin_slide_quality

    @property
    def is_imported_from_ai(self):
        """
        是否是导入了ai结果的标注
        :return:
        """
        return self.import_ai_templates is None or self.template_id in self.import_ai_templates

    def get_clarity_level(self, clarity_standards_max: float, clarity_standards_min: float):
        if not isinstance(self.clarity, (int, float, str)) or self.clarity == '-':
            level = ''
        else:
            if float(self.clarity) > clarity_standards_max:
                level = '良好'
            elif float(self.clarity) < clarity_standards_min:
                level = '较差'
            else:
                level = '中等'
        return level

    def get_cell_num_tips(self, algor_type: AIType, cell_num_threshold: int) -> List[str]:
        cell_num_tips = []
        if algor_type in [AIType.bm, AIType.tct, AIType.lct, AIType.dna]:
            if cell_num_threshold and self.cell_num and self.cell_num < cell_num_threshold:
                cell_num_tips = ['有效检测细胞量不足，请谨慎参考诊断建议。']
        return cell_num_tips

    def to_dict(self, all_fields: bool = False):
        d = {
            'id': self.fileid,
            'uid': self.id,
            'ai_suggest': self.ai_suggest or '',
            'parsed_ai_suggest': self.parsed_ai_suggest,
            'alg': self.alg,
            'check_result': self.check_result or '',
            'filename': self.filename,
            'fileid': self.fileid,
            'caseid': self.caseid,
            'operator': self.operator,
            'slice_number': self.slice_number,
            'started': self.started,
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
            'update_time': self.update_time,
            'aiDiagnosisState': self.ai_diagnosis_state,
            'checkedDiagnosisState': self.checked_diagnosis_state,
            'isImportedFromAi': self.is_imported_from_ai,
            'labels': self.labels or [],
            'isMark': self.is_marked,
            'clarityLevel': self.clarity_level,
            'aiTips': self.ai_tips or [],
            'errCode': self.err_code or 0,
            'errMessage': self.err_message or '',
            'cellNumTips': self.cell_num_tips or None,
            'company': self.company
        }
        d.update(self.data_paths)
        if all_fields:
            tool_type = self.tool_type
            template_selected = self.template_id
            if (not tool_type or tool_type in ('human', 'auto')) and (not template_selected or template_selected == 1):
                template_id = cache.get(f'{self.company}:last_selected_template_id')
                if template_id:
                    tool_type = 'tagging'
                    template_selected = int(template_id)

            # if self.slide:
            #     d['tileSize'] = self.slide.tile_size
            d.update({
                'stain': self.stain,
                'mppx': self.mppx,
                'mppy': self.mppy,
                'toolType': tool_type,
                'objective_rate': self.objective_rate,
                "ajaxToken": json.loads(self.ajax_token),
                "path": self.path,
                "type": self.type,
                "clarity": self.clarity,
                "position": json.loads(self.position) if self.position else {},
                "roilist": self.roilist,
                "ai_dict": self.ai_dict,
                "slide_quality": int(self.slide_quality) if self.slide_quality else None,
                "cell_num": self.cell_num,
                'templateId': template_selected,
            })
        return d


class CaseRecordEntity(BaseDomainEntity):
    slices: List[SliceEntity] = []
    attachments: List[SliceEntity] = []
    opinion: str = ''
    dna_opinion: str = ''

    @classmethod
    def get_all_display_columns(cls) -> List[str]:

        all_columns = [
            '样本号', '姓名', '性别', '年龄', '取样部位', '样本类型', '切片数量', '标注状态', '处理状态', '切片文件夹', '切片标签',
            '切片编号', '文件名', '自定义标签', 'AI模块', '切片质量', '扫描倍数', '清晰度', '细胞量', 'AI建议', '复核结果', '创建时间', '最后更新', '操作人', '报告'
        ]
        plugins = Settings.PLUGINS
        if 'logene' in plugins:
            all_columns.append('导出状态')
        return all_columns

    @classmethod
    def get_disabled_columns(cls) -> List[str]:
        return ['样本号']

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
    def reports_dir_v2(self) -> str:
        """
        基于报告系统快照生成的报告文件夹
        :return:
        """
        return os.path.join(self.data_dir, 'reports_v2')

    @property
    def reports(self) -> List[dict]:
        items = []
        report_ids = []
        reports_dir = ''
        use_report_service = False
        if os.path.exists(self.reports_dir_v2):
            reports_dir = self.reports_dir_v2
            report_ids = os.listdir(self.reports_dir_v2)
            use_report_service = True
        elif os.path.exists(self.reports_dir):
            reports_dir = self.reports_dir
            report_ids = os.listdir(self.reports_dir)

        for r_id in report_ids:
            path = os.path.join(reports_dir, r_id)
            if not os.path.isdir(path):
                continue
            timestamp = datetime.datetime.utcfromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%dT%H:%M:%SZ")
            item = {
                'id': r_id,
                'date': timestamp,
                'url': f'{Settings.REPORT_SERVER}/report/#/report?snapshot_id={r_id}' if use_report_service else None
            }
            items.append(item)
        items.sort(key=lambda x: x['date'], reverse=True)
        return items

    def get_report_templates(self, template_config: List[dict], file_id: str) -> List[dict]:
        if not self.slices:
            return []

        template_codes = []
        for item in template_config:
            template_id = item['templateId']
            if template_id == 'reg':
                template_codes.insert(0, item)

            for slic in self.slices:
                ai_type = AIType.get_by_value(slic.alg)
                if template_id == ai_type and item.get('templateCode'):
                    if slic.fileid == file_id:
                        item['active'] = True
                    template_codes.append(item)
                    break
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
                '霉菌', '').replace('滴虫', '').replace('线索', '').replace('放线菌', '').replace('疱疹', '').replace(
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
                replace('疱疹', '').replace('巨细胞病毒', '').replace('HPV', '').replace('萎缩', '').replace('炎症', ''). \
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
                report_info['isStatisfied'] = 1 if slice_entity.slide_quality == '1' else 0
                if slice_entity.cell_num:
                    report_info['cellNum'] = 'greater' if (slice_entity.cell_num or 0) > 5000 else 'less'
                if slice_entity.check_result:
                    items = slice_entity.check_result.split(' ')
                    report_info['pathogen'] = items[2].split(',') if len(items) >= 3 else []
                elif slice_entity.ai_suggest:
                    ai_suggest = slice_entity.parsed_ai_suggest
                    report_info['pathogen'] = ai_suggest.get('microbe', [])

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
        update_time = [slice.update_time for slice in self.slices if slice.update_time]
        update_time.append(self.update_time)
        d = {
            'id': self.caseid,
            'caseid': self.caseid,
            'basic': self.basic_info,
            'attachments': [entity.to_dict() for entity in self.attachments],
            'slices': [entity.to_dict(all_fields=True) for entity in self.slices],
            'reports': self.reports if self.slices else [],
            'slices_count': self.slice_count,
            'create_time': self.create_time,
            'update': max(update_time),
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
                AIType.tct, AIType.lct, AIType.dna, AIType.pdl1, AIType.ki67, AIType.ki67hot, AIType.er, AIType.pr,
                AIType.her2, AIType.fish_tissue, AIType.np, AIType.cd30
            ]])
        return template_config

    @classmethod
    def format_template_config_item(cls, template_config_item: dict):
        ai_type = AIType.get_by_value(template_config_item['templateId'])
        template_config_item['templateName'] = ai_type.display_name if ai_type else '通用模板'

    @property
    def formatted_template_config(self):
        template_config = self.template_config
        for item in template_config:
            self.format_template_config_item(item)
        return template_config

    def to_dict(self):
        for item in self.template_config:
            self.format_template_config_item(item)
        return super().to_dict()
