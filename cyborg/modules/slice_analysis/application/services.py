import json
import logging
import os
import shutil
import time
from functools import wraps
from inspect import getfullargspec
from typing import Optional, List

from cyborg.app.request_context import request_context
from cyborg.modules.slice.application.services import SliceService
from cyborg.modules.slice_analysis.domain.consts import AI_TYPE_MANUAL_MARK_TABLE_MAPPING
from cyborg.modules.slice_analysis.domain.entities import MarkEntity, MarkGroupEntity

from cyborg.modules.slice_analysis.domain.services import SliceAnalysisDomainService
from cyborg.modules.slice_analysis.domain.value_objects import AIType, TiledSlice, SliceMarkConfig, CellCount
from cyborg.seedwork.application.responses import AppResponse


logger = logging.getLogger(__name__)


def connect_slice_db(need_template_db: bool = False):
    def deco(f):

        @wraps(f)
        def wrapped(*args, **kwargs):
            _self: SliceAnalysisService = args[0]
            case_id = request_context.case_id
            file_id = request_context.file_id
            if not (case_id and file_id):
                return AppResponse(message='no case_id or file_id')

            data_paths = _self.slice_service.get_slice_data_paths(case_id=case_id, file_id=file_id).data
            if not data_paths:
                return AppResponse(message='db file not found')

            db_doc_path = data_paths['db_file_path']
            db_template_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'slice.db')
            if not os.path.exists(db_doc_path):
                shutil.copyfile(db_template_path, db_doc_path)

            request_context.connect_slice_db(db_doc_path)
            if need_template_db:
                request_context.connect_template_db(db_template_path)

            ai_type = request_context.ai_type

            if ai_type:
                template_id = 0
                if ai_type == AIType.label:
                    slice_info = _self.slice_service.get_slice_info(case_id=case_id, file_id=file_id).data
                    template_id = slice_info['templateId'] if slice_info else 0
                    func_args = getfullargspec(f)[0]
                    if 'template_id' in func_args and not kwargs.get('template_id'):
                        kwargs['template_id'] = template_id

                mark_table_suffix = _self.domain_service.get_mark_table_suffix(ai_type=ai_type, template_id=template_id)
                _self.domain_service.repository.mark_table_suffix = mark_table_suffix
                _self.domain_service.repository.create_mark_tables(ai_type=ai_type)

            manual_table_suffix = AI_TYPE_MANUAL_MARK_TABLE_MAPPING.get(ai_type, 'human')
            _self.domain_service.repository.manual.mark_table_suffix = manual_table_suffix
            _self.domain_service.repository.manual.create_mark_tables(ai_type=ai_type)

            r = f(*args, **kwargs)

            request_context.close_slice_db()

            return r

        return wrapped

    return deco


class SliceAnalysisService(object):

    def __init__(
            self, domain_service: SliceAnalysisDomainService, slice_service: SliceService
    ):
        super(SliceAnalysisService, self).__init__()
        self.domain_service = domain_service
        self.slice_service = slice_service

    def _get_tiled_slice(self, case_id: str, file_id: str) -> Optional[TiledSlice]:
        info = self.slice_service.get_slice_file_info(case_id=case_id, file_id=file_id).data
        return TiledSlice(**info) if info else None

    def _fetch_mark_area(self, ai_type: AIType, case_id: str, file_id: str, marks: List[MarkEntity]):
        slice_info = self.slice_service.get_slice_file_info(case_id=case_id, file_id=file_id).data
        mpp = slice_info.get('mpp', 0.242042) if slice_info else 0.242042
        for mark in marks:
            if mark.method in ['rectangle', 'freepen'] and mark.position and ai_type != AIType.fish_tissue:
                mark.area = mark.cal_polygon_area(mpp)

    @connect_slice_db()
    def create_mark(
            self,
            group_id: Optional[int] = None,
            path: Optional[dict] = None,
            area_id: Optional[int] = None,
            stroke_color: Optional[str] = None,
            fill_color: Optional[str] = None,
            radius: Optional[float] = None,
            method: Optional[str] = None,
            editable: Optional[int] = None,
            type: Optional[int] = None,
            mark_type: Optional[int] = None,
            dashed: Optional[int] = None,
            doctor_diagnosis: Optional[dict] = None,
    ) -> AppResponse[dict]:

        ai_type = request_context.ai_type

        # TODO need review
        # ai_id = self.domain_service.repository.get_ai_id_by_type(ai_type)

        if area_id == -1:
            area_id = None

        case_id = request_context.case_id
        file_id = request_context.file_id

        tiled_slice = self._get_tiled_slice(case_id=case_id, file_id=file_id)
        err_msg, new_mark = self.domain_service.create_mark(
            ai_type=ai_type, group_id=group_id, position=path, area_id=area_id,
            stroke_color=stroke_color, fill_color=fill_color, radius=radius, method=method, editable=editable,
            diagnosis_type=type, mark_type=mark_type, dashed=dashed, doctor_diagnosis=doctor_diagnosis,
            tiled_slice=tiled_slice, op_name=request_context.current_user.username
        )

        logger.info(new_mark.id)
        if err_msg:
            return AppResponse(err_code=1, message=err_msg)

        self._fetch_mark_area(ai_type=ai_type, case_id=case_id, file_id=file_id, marks=[new_mark, ])

        if ai_type not in [AIType.human, AIType.label] and area_id:
            # 更新算法结果。PS：人工打点, 标注以及算法框不需要更新
            self.domain_service.update_ai_result(
                marks=[new_mark, ], option=1, ai_type=ai_type, tiled_slice=tiled_slice
            )

        logger.info(new_mark.id)
        return AppResponse(message='create mark succeed', data={'new_mark_id': str(new_mark.id)})

    @connect_slice_db()
    def create_ai_marks(
            self, cell_marks: List[dict], roi_marks: List[dict], skip_mark_to_tile: bool = False
    ) -> AppResponse[dict]:

        case_id = request_context.case_id
        file_id = request_context.file_id
        ai_type = request_context.ai_type

        tiled_slice = self._get_tiled_slice(case_id=case_id, file_id=file_id) if not skip_mark_to_tile else None

        err_msg, new_marks = self.domain_service.create_ai_marks(
            cell_marks=cell_marks, roi_marks=roi_marks, tiled_slice=tiled_slice)

        if err_msg:
            return AppResponse(err_code=1, message=err_msg)

        self._fetch_mark_area(ai_type=ai_type, case_id=case_id, file_id=file_id, marks=new_marks)

        # if ai_type not in [AIType.human, AIType.label]:
        #     self.domain_service.update_ai_result(
        #         marks=new_marks, option=1, ai_type=ai_type, tiled_slice=tiled_slice
        #     )

        return AppResponse(
            message='create marks succeed', data={'latest_new_mark_id': new_marks[-1].id if new_marks else None})

    @connect_slice_db()
    def get_marks(self, view_path: dict) -> AppResponse:
        case_id = request_context.case_id
        file_id = request_context.file_id
        ai_type = request_context.ai_type

        slice_info = self.slice_service.get_slice_info(case_id=case_id, file_id=file_id).data
        tiled_slice = self._get_tiled_slice(case_id=case_id, file_id=file_id)
        radius = float(format(slice_info['radius'] / tiled_slice.mpp, '.5f'))
        mark_config = SliceMarkConfig(radius=radius, is_solid=slice_info['is_solid'] == 1)

        marks = self.domain_service.get_marks(
            ai_type=ai_type, view_path=view_path, tiled_slice=tiled_slice, mark_config=mark_config)
        return AppResponse(message='query succeed', data={'marks': marks})

    @connect_slice_db()
    def get_wsa_marks(self) -> AppResponse:
        _, marks = self.domain_service.repository.get_marks()
        return AppResponse()

    @connect_slice_db()
    def get_default_area(self):
        err_msg, mark = self.domain_service.get_or_create_default_area(ai_type=request_context.ai_type)
        return AppResponse(message=err_msg, data=mark.to_dict() if mark else None)

    @connect_slice_db()
    def count_marks_in_scope(self, scope: dict) -> AppResponse:
        tiled_slice = self._get_tiled_slice(case_id=request_context.case_id, file_id=request_context.file_id)
        err_code, message, count = self.domain_service.count_marks_in_scope(
            scope=scope, tiled_slice=tiled_slice, ai_type=request_context.ai_type)
        return AppResponse(err_code=err_code, message=message, data={'select_count': count})

    @connect_slice_db()
    def update_marks(
            self, marks_data: Optional[List[dict]], scope: Optional[dict], target_group_id: int,
    ) -> AppResponse:
        ai_type = request_context.ai_type
        tiled_slice = self._get_tiled_slice(request_context.case_id, request_context.file_id)
        if scope and 'x' in scope and 'y' in scope:
            err_code, message = self.domain_service.update_marks_by_scope(
                scope, target_group_id, tiled_slice, ai_type, op_name=request_context.current_user.username)
        else:
            err_code, message = self.domain_service.update_marks(marks_data, target_group_id, tiled_slice, ai_type)
        return AppResponse(err_code=err_code, message=message)

    @connect_slice_db()
    def delete_marks(self, mark_ids: Optional[List[int]], scope: Optional[dict]):
        tiled_slice = self._get_tiled_slice(case_id=request_context.case_id, file_id=request_context.file_id)
        op_name = request_context.current_user.username
        ai_type = request_context.ai_type
        if scope is not None:
            err_code, message = self.domain_service.delete_marks_by_scope(
                scope=scope, tiled_slice=tiled_slice, ai_type=ai_type, op_name=op_name)
        else:
            err_code, message = self.domain_service.delete_marks(
                mark_ids=mark_ids, tiled_slice=tiled_slice, ai_type=ai_type, op_name=op_name)

        slice_info = self.slice_service.get_slice_info(
            case_id=request_context.case_id, file_id=request_context.file_id).data
        if slice_info and ai_type == slice_info['alg']:
            if self.domain_service.repository.get_mark_count() == 0:
                self.slice_service.reset_ai_status()
        return AppResponse(err_code=err_code, message=message)

    @connect_slice_db(need_template_db=True)
    def import_ai_marks(self, template_id: int = 0):
        ai_type = request_context.ai_type
        err_msg = self.domain_service.import_ai_marks(template_id)
        if err_msg:
            return AppResponse(err_code=1, message=err_msg)

        _, marks = self.domain_service.repository.get_marks(per_page=1)
        if marks and ai_type in [AIType.fish_tissue]:
            self.slice_service.update_mark_config(radius=marks[0].radius, is_solid=marks[0].is_solid)

        return AppResponse(message='import ai result succeed')

    @connect_slice_db()
    def backup_ai_marks(self):
        if not self.domain_service.repository.backup_ai_mark_tables():
            return AppResponse(err_code=1, message='backup mark table failed')

        return AppResponse(message='backup mark table succeed')

    @connect_slice_db()
    def export_marks(self):
        mark_infos = list()
        _, marks = self.domain_service.repository.get_marks()
        for mark in marks:
            group = self.domain_service.repository.get_mark_group_by_id(mark.group_id)
            position = mark.position
            position_list = list()
            for i in range(len(position.get('x'))):
                x_coord_list = position.get('x')
                y_coord_list = position.get('y')
                position_list.append({'x': x_coord_list[i], 'y': y_coord_list[i]})
            mark_info = {
                'position': position_list,
                'color': group.color if group else '',
                'groupName': group.groupName if group else '',
            }
            mark_infos.append(mark_info)

        data_paths = self.slice_service.get_slice_data_paths(
            case_id=request_context.case_id, file_id=request_context.file_id).data
        slice_path = data_paths.get('slice_dir')
        filename = 'slice.json'
        file_path = os.path.join(slice_path, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(mark_infos, f, ensure_ascii=False)
        return file_path

    @connect_slice_db()
    def clear_ai_result(self, exclude_area_marks: Optional[List[int]] = None) -> AppResponse:
        self.domain_service.clear_ai_result(ai_type=request_context.ai_type, exclude_area_marks=exclude_area_marks)
        res = self.slice_service.delete_ai_image_files(case_id=request_context.case_id, file_id=request_context.file_id)
        if res.err_code:
            return res
        return AppResponse()

    @connect_slice_db()
    def get_rois(self) -> AppResponse:
        res = self.slice_service.get_slice_info(
            case_id=request_context.case_id, file_id=request_context.file_id)
        if res.err_code:
            return res
        slice_info = res.data

        rois = self.domain_service.get_rois(
            ai_type=request_context.ai_type, ai_suggest=slice_info['ai_suggest'], ai_status=slice_info['ai_status'])
        return AppResponse(message='query succeed', data={'ROIS': rois})

    @connect_slice_db()
    def create_mark_group(self, template_id: int) -> AppResponse:
        current_time = time.time()
        mark_group = MarkGroupEntity(raw_data=dict(
            default_color='#FF0000',
            color='#FF0000',
            op_time=current_time,
            template_id=template_id,
            is_empty=1,
            is_show=1,
            is_import=0,
            is_ai=0,
            create_time=current_time
        ))

        self.domain_service.repository.save_mark_group(mark_group)

        return AppResponse(message='create mark group succeed', data={'new_group_id': mark_group.id})

    @connect_slice_db()
    def delete_mark_group(self, group_id: int) -> AppResponse:
        if not self.domain_service.delete_mark_group(group_id):
            return AppResponse(message='delete mark failed')
        return AppResponse(message='delete mark succeed')

    @connect_slice_db()
    def update_mark_groups(self, groups: List[dict]) -> AppResponse:
        for group_info in groups:
            self.domain_service.update_mark_group(**group_info)
        return AppResponse(message='modify group succeed')

    @connect_slice_db()
    def select_mark_group(self, group_id: int, page: int = 0, per_page: int = 20):

        case_id = request_context.case_id
        file_id = request_context.file_id

        total, marks = self.domain_service.get_marks_in_group(group_id=group_id, page=page, per_page=per_page)

        slice = self.slice_service.get_slice_info(case_id=case_id, file_id=file_id).data
        file_info = self.slice_service.get_slice_file_info(case_id=case_id, file_id=file_id).data

        mpp = file_info['mpp'] if file_info else None
        radius = float(format(slice['radius'] / mpp, '.5f'))
        mark_config = SliceMarkConfig(radius=radius, is_solid=slice['is_solid'] == 1)

        show_marks = self.domain_service.show_marks(
            ai_type=request_context.ai_type, marks=marks, mark_config=mark_config, show_groups=[group_id, ])

        if not self.domain_service.repository.update_mark_group_selected(group_id=group_id):
            return AppResponse(message='change group failed')

        return AppResponse(message='query succeed', data={'marks': show_marks, 'total': total})

    @connect_slice_db()
    def get_selected_mark_group(self) -> AppResponse[dict]:
        mark_group = self.domain_service.repository.get_selected_mark_group()
        return AppResponse(data=mark_group.to_dict() if mark_group else None)

    @connect_slice_db(need_template_db=True)
    def get_mark_groups(self, template_id: int) -> AppResponse[dict]:
        mark_groups = self.domain_service.repository.get_mark_groups_by_template_id(template_id=template_id)
        return AppResponse(data=[mark_group.to_dict() for mark_group in mark_groups])

    @connect_slice_db()
    def switch_mark_group_show_status(self, group_id: int) -> AppResponse:
        message = self.domain_service.switch_mark_group_show_status(group_id=group_id)
        return AppResponse(message=message)

    @connect_slice_db()
    def get_group_info(self, group_id) -> AppResponse:
        group = self.domain_service.repository.get_mark_group_by_id(group_id)
        return AppResponse(data=self.domain_service.show_mark_groups([group])[0])

    @connect_slice_db()
    def select_template(self, template_id: int) -> AppResponse:

        res = self.slice_service.update_template_id(template_id=template_id)
        if res.err_code:
            return res

        groups = self.domain_service.repository.get_mark_groups_by_template_id(
            template_id=template_id, primary_only=True, is_import=0, is_ai=0)
        data = self.domain_service.show_mark_groups(groups)
        return AppResponse(message='operation succeed', data=data)

    def get_all_templates(self) -> AppResponse[dict]:
        templates = self.domain_service.config_repository.get_all_templates()
        return AppResponse(message='query succeed', data={'templates': templates})

    @connect_slice_db()
    def get_cell_count_in_quadrant(self, view_path: dict) -> AppResponse:
        case_id = request_context.case_id
        file_id = request_context.file_id
        ai_type = request_context.ai_type

        tiled_slice = self._get_tiled_slice(case_id=case_id, file_id=file_id)

        data = self.domain_service.get_cell_count_in_quadrant(
            view_path=view_path, tiled_slice=tiled_slice, ai_type=ai_type)

        return AppResponse(message='query succeed', data=data)

    def get_dna_info(self) -> AppResponse:
        request_context.ai_type = AIType.dna
        dna_statics = self._get_dna_info()
        return AppResponse(data=dna_statics)

    @connect_slice_db()
    def _get_dna_info(self) -> dict:
        dna_statics = {
            'num_normal': 0, 'num_abnormal_high': 0, 'num_abnormal_low': 0,
            'normal_ratio': 0, 'abnormal_high_ratio': 0, 'abnormal_low_ratio': 0,
            'mean_di_normal': 0, 'mean_di_abnormal_high': 0, 'mean_di_abnormal_low': 0,
            'std_di_normal': 0, 'std_di_abnormal_high': 0, 'std_di_abnormal_low': 0
        }
        _, marks = self.domain_service.repository.get_marks(mark_type=3)
        if marks:
            mark = marks[0]
            if mark.ai_result:
                dna_statics = mark.ai_result.get('dna_statics')
        return dna_statics

    def get_report_roi(self) -> AppResponse:
        slices = self.slice_service.get_slices(case_ids=[request_context.case_id]).data
        data = {"human": [], 'lct': [], 'tct': [], 'dna': []}

        for slice_info in slices:
            request_context.file_id = slice_info['fileid']
            request_context.ai_type = AIType.get_by_value(slice_info['alg'])
            if request_context.ai_type:
                res = self._get_slice_report_roi(slice_info)
                for ai_type in [AIType.human, AIType.lct, AIType.tct, AIType.dna]:
                    data[ai_type.value].extend(res[ai_type.value])
                data['dnaStatics'] = res.get('dnaStatics')

        return AppResponse(message='query succeed', data=data)

    @connect_slice_db()
    def _get_slice_report_roi(self, slice_info: dict):
        ai_type = request_context.ai_type
        company = request_context.current_company
        res = {"human": [], 'lct': [], 'tct': [], 'dna': []}

        _, manual_marks = self.domain_service.repository.manual.get_marks(is_export=1)
        for manual_mark in manual_marks:
            d = manual_mark.to_roi(ai_type=ai_type)
            d['type'] = ai_type.value
            d['image_url'] = MarkEntity.make_image_url(caseid=slice_info['caseid'], company=company, **d)
            res['human'].append(d)

        _, marks = self.domain_service.repository.get_marks(mark_type=[2, 3])

        ai_result = marks[0].ai_result if marks else None
        cells = ai_result.get('cells') if ai_result else None
        nuclei = ai_result.get('nuclei') if ai_result else None
        dna_statics = ai_result.get('dna_statics') if ai_result else None
        if cells:
            for d in cells:
                roi_list = cells[d]['data']
                for temp_roi in roi_list:
                    if temp_roi['image'] == 1:
                        temp_dict = {}
                        temp_dict['id'] = temp_roi.get('id')
                        temp_dict['path'] = temp_roi.get('path')
                        temp_dict['filename'] = slice_info['filename']
                        temp_dict['fileid'] = slice_info['fileid']
                        temp_dict['remark'] = temp_roi.get('remark', '')
                        temp_dict['ai_type'] = ai_type.value
                        temp_dict['image_url'] = MarkEntity.make_image_url(
                            caseid=slice_info['caseid'], company=company, **temp_dict)
                        if ai_type in [AIType.lct, AIType.tct]:
                            res['tct'].append(temp_dict)
                        else:
                            res[ai_type.value].append(temp_dict)
        if nuclei:
            for nucleus in nuclei:
                if nucleus.get('image') == 1:
                    temp_dict = {}
                    temp_dict['id'] = nucleus.get('id')
                    temp_dict['path'] = nucleus.get('path')
                    temp_dict['filename'] = slice_info['filename']
                    temp_dict['fileid'] = slice_info['fileid']
                    temp_dict['remark'] = nucleus.get('remark', '')
                    temp_dict['ai_type'] = ai_type.value
                    temp_dict['iconType'] = 'dnaIcon'
                    temp_dict['di'] = nucleus.get('dna_index')
                    temp_dict['image_url'] = MarkEntity.make_image_url(
                        caseid=slice_info['caseid'], company=company, **temp_dict)
                    res[ai_type.value].append(temp_dict)

        if dna_statics:
            res['dnaStatics'] = dna_statics

        return res
