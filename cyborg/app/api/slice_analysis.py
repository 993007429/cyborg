import json
import logging

from cyborg.app.api.json import orjsonify
from flask import request, jsonify, send_file, redirect

from cyborg.app.api import api_blueprint
from cyborg.app.limiter import limiter
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.infra.oss import oss
from cyborg.modules.slice_analysis.domain.value_objects import AIType
from cyborg.seedwork.application.responses import AppResponse
from cyborg.utils.strings import camel_to_snake
from cyborg.app.settings import Settings
from cyborg.infra.fs import fs


logger = logging.getLogger(__name__)


@api_blueprint.route('/slice/createMark', methods=['get', 'post'])
@limiter.limit("2/second", override_defaults=False)
def create_mark():
    mark_params = json.loads(request.form.get('mark'))

    if not mark_params.get('ai_type'):
        mark_params['ai_type'] = request.form.get('ai_type', 'human')
    if mark_params.get('area_id') == -1:
        mark_params['area_id'] = None
    request_context.ai_type = AIType.get_by_value(mark_params.pop('ai_type')) or AIType.unknown

    res = AppServiceFactory.new_slice_analysis_service().create_mark(
        **{camel_to_snake(k): v for k, v in mark_params.items()})
    if request_context.ai_type == AIType.label:
        AppServiceFactory.slice_service.update_slice_mark(is_marked=1)
    return jsonify(res.dict())


@api_blueprint.route('/slice/getMarks', methods=['get', 'post'])
def get_marks():
    view_path = json.loads(request.form.get('position'))  # 视野区域
    ai_type = request.form.get('ai_type')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human
    res = AppServiceFactory.new_slice_analysis_service().get_marks(view_path=view_path)
    return orjsonify(res.dict())


@api_blueprint.route('/slice/selectCount', methods=['get', 'post'])
def select_count():
    scope = request.form.get('scope')
    ai_type = request.form.get('ai_type')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human

    res = AppServiceFactory.new_slice_analysis_service().count_marks_in_scope(scope=scope)
    return jsonify(res.dict())


@api_blueprint.route('/slice/markShow', methods=['get', 'post'])
@limiter.limit("2/second", override_defaults=False)
def mark_show():
    group_id = int(request.form.get('group_id'))
    request_context.ai_type = AIType.label

    res = AppServiceFactory.new_slice_analysis_service().switch_mark_group_show_status(group_id)
    return jsonify(res.dict())


@api_blueprint.route('/slice/modifyMark', methods=['get', 'post'])
def modify_marks():
    import os
    caseid = request.form.get('caseid')
    fileid = request.form.get('fileid')
    ai_type = request.form.get('ai_type', '')
    scope = request.form.get('scope')
    target_group_id = request.form.get('target_group_id')
    marks = json.loads(request.form.get('marks')) if request.form.get('marks') is not None else None

    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human

    if scope == 'null':
        scope = None

    res = AppServiceFactory.new_slice_analysis_service().update_marks(
        marks_data=marks, scope=scope, target_group_id=int(target_group_id) if target_group_id else None)

    audio = request.files['audio']
    if audio and audio.filename.split('.')[1] not in ['wav', 'mp3']:
        res.message = 'File is not an audio file'
        res.code = 1
        return jsonify(res.dict)

    if audio:
        data = []
        slice_doc_path = fs.path_join(Settings.DATA_DIR, request_context.company, 'data', caseid, 'slices', fileid)
        for mark in marks:
            item = {}
            mark_id, file_name = int(mark.get('id')), str(mark.get('id')) + '.wav'
            audio.save(os.path.join(slice_doc_path, file_name))
            audio_url = f'/files/getAudio?caseid={caseid}&fileid={fileid}&markid={mark_id}&company={request_context.company}'
            item['markId'] = mark_id
            item['audioUrl'] = audio_url
            data.append(item)
        res.data = data
    return jsonify(res.dict())


@api_blueprint.route('/slice/deleteMark', methods=['get', 'post'])
def delete_mark():
    scope = request.form.get('scope')
    ai_type = request.form.get('ai_type')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human
    mark_ids = json.loads(request.form.get('marks')) if request.form.get('marks') else None

    if scope == 'null':
        scope = None

    res = AppServiceFactory.new_slice_analysis_service().delete_marks(mark_ids=mark_ids, scope=scope)
    return jsonify(res.dict())


@api_blueprint.route('/slice/getROIList', methods=['get', 'post'])
def get_roi_list():
    ai_type = request.form.get('ai_type')
    is_deleted = int(request.form.get('is_deleted', 0))  # 0 未删除 1 已删除
    lesion_type = request.form.get('lesion_type', 'normal')  # normal 正常二倍体细胞 abnormal_low 疑似病变细胞 abnormal_high 病变细胞
    page = int(request.form.get('page', 1))
    page_size = int(request.form.get('page_size', 100))

    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human
    res = AppServiceFactory.new_slice_analysis_service().get_rois(
        is_deleted=is_deleted,
        lesion_type=lesion_type,
        page=page,
        page_size=page_size
    )
    return jsonify(res.dict())


@api_blueprint.route('/slice/deleteDnaRoi', methods=['get', 'post'])
def delete_dna_roi():
    ai_type = request.form.get('ai_type')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human
    mark_id_list = json.loads(request.form.get('marks')) if request.form.get('marks') else []
    deleted = int(request.form.get('deleted'))

    res = AppServiceFactory.new_slice_analysis_service().delete_dna_roi(mark_id_list=mark_id_list, deleted=deleted)
    return jsonify(res.dict())


@api_blueprint.route('/slice/getStatistics', methods=['get', 'post'])
def get_statistics():
    ai_type = request.form.get('ai_type')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human

    res = AppServiceFactory.new_slice_analysis_service().get_statistics()
    return jsonify(res.dict())


@api_blueprint.route('/slice/getFeat', methods=['get', 'post'])
def get_feat():
    ai_type = request.form.get('ai_type')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human

    res = AppServiceFactory.new_slice_analysis_service().get_feat()
    return jsonify(res.dict())


@api_blueprint.route('/slice/getHistplot', methods=['get', 'post'])
def get_histplot():
    ai_type = request.form.get('ai_type')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human

    res = AppServiceFactory.new_slice_analysis_service().get_histplot()
    return jsonify(res.dict())


@api_blueprint.route('/slice/getScatterplot', methods=['get', 'post'])
def get_scatterplot():
    ai_type = request.form.get('ai_type')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human

    res = AppServiceFactory.new_slice_analysis_service().get_scatterplot()
    return jsonify(res.dict())


@api_blueprint.route('/slice/importAiResult', methods=['get', 'post'])
def import_ai_result():
    request_context.ai_type = AIType.label

    res = AppServiceFactory.new_slice_analysis_service().import_ai_marks()
    return jsonify(res.dict())


@api_blueprint.route('/slice/queryRadius', methods=['get', 'post'])
def query_radius():
    res = AppServiceFactory.slice_service.get_slice_info(
        case_id=request_context.case_id, file_id=request_context.file_id)
    if res.err_code:
        return res

    res = AppResponse(message='query succeed', data={'radius': res.data['radius']})
    return jsonify(res.dict())


@api_blueprint.route('/slice/querySolidStatus', methods=['get', 'post'])
def query_solid_status():
    res = AppServiceFactory.slice_service.get_slice_info(
        case_id=request_context.case_id, file_id=request_context.file_id)
    if res.err_code:
        return res

    res = AppResponse(message='query succeed', data={'is_solid': res.data['is_solid']})
    return jsonify(res.dict())


@api_blueprint.route('/slice/modifyRadius', methods=['get', 'post'])
def modify_radius():
    radius = float(request.form.get('radius'))
    res = AppServiceFactory.slice_service.update_mark_config(radius=radius)
    return jsonify(res.dict())


@api_blueprint.route('/slice/modifySolidStatus', methods=['get', 'post'])
@limiter.limit("2/second", override_defaults=False)
def modify_solid_status():
    is_solid = int(request.form.get('is_solid'))
    res = AppServiceFactory.slice_service.update_mark_config(is_solid=is_solid)
    return jsonify(res.dict())


@api_blueprint.route('/slice/createGroup', methods=['get', 'post'])
@limiter.limit("2/second", override_defaults=False)
def create_group():
    template_id = int(request.form.get('template_id'))
    res = AppServiceFactory.new_slice_analysis_service().create_mark_group(template_id=template_id)
    return jsonify(res.dict())


@api_blueprint.route('/slice/modifyGroup', methods=['get', 'post'])
def modify_group():
    groups = json.loads(request.form.get('groups'))
    request_context.ai_type = AIType.label
    res = AppServiceFactory.new_slice_analysis_service().update_mark_groups(groups)
    return jsonify(res.dict())


@api_blueprint.route('/slice/deleteGroup', methods=['get', 'post'])
def delete_group():
    group_id = int(request.form.get('group_id'))

    request_context.ai_type = AIType.label

    res = AppServiceFactory.new_slice_analysis_service().delete_mark_group(group_id)
    return jsonify(res.dict())


@api_blueprint.route('/slice/selectGroup', methods=['get', 'post'])
@limiter.limit("2/second", override_defaults=False)
def select_group():
    group_id = int(request.form.get('group_id'))
    page = int(request.form.get('page'))
    per_page = int(request.form.get('per_page'))

    request_context.ai_type = AIType.label

    res = AppServiceFactory.new_slice_analysis_service().select_mark_group(
        group_id=group_id, page=page - 1, per_page=per_page)

    return jsonify(res.dict())


@api_blueprint.route('/slice/queryGroup', methods=['get', 'post'])
def query_group():
    group_id = int(request.form.get('group_id'))
    request_context.ai_type = AIType.label

    res = AppServiceFactory.new_slice_analysis_service().get_group_info(group_id)
    return jsonify(res.dict())


@api_blueprint.route('/slice/selectTemplate', methods=['get', 'post'])
def select_template():
    request_context.ai_type = AIType.label
    template_id = int(request.form.get('id'))
    res = AppServiceFactory.slice_service.update_template_id(template_id=template_id)
    if res.err_code:
        return res
    res = AppServiceFactory.new_slice_analysis_service().select_template(template_id=template_id)
    return jsonify(res.dict())


@api_blueprint.route('/slice/templateList', methods=['get', 'post'])
def template_list():
    request_context.ai_type = AIType.label
    res = AppServiceFactory.new_slice_analysis_service().get_all_templates()
    return jsonify(res.dict())


@api_blueprint.route('/slice/share_templates', methods=['get', 'post'])
def share_templates():
    res = AppServiceFactory.new_slice_analysis_service().get_share_templates()
    return jsonify(res.dict())


@api_blueprint.route('/slice/getScreenCount', methods=['get', 'post'])
def get_screen_count():
    ai_type = request.form.get('ai_type')
    if ai_type.startswith('lct') or ai_type.startswith('tct'):
        ai_type = ai_type[0:3]
    request_context.ai_type = AIType.get_by_value(ai_type)

    view_path = json.loads(request.form.get('position'))

    res = AppServiceFactory.new_slice_analysis_service().get_cell_count_in_quadrant(view_path=view_path)
    return jsonify(res.dict())


@api_blueprint.route('/files/downloadTemplate', methods=['get', 'post'])
def download_template():
    template_name = request.args.get('name')
    if template_name.startswith('调参模板'):
        oss_key = oss.path_join('AI', 'PDL1', template_name)
    elif template_name.startswith('病例列表'):
        oss_key = oss.path_join('record', 'template', template_name)
    else:
        return '', 404

    oss_url = oss.generate_sign_url('GET', oss_key)

    return redirect(oss_url, code=302)


@api_blueprint.route('/files/exportJson', methods=['get', 'post'])
def export_json():
    file_path = AppServiceFactory.new_slice_analysis_service().export_marks()
    return send_file(file_path)


@api_blueprint.route('/slice/switchAI', methods=['get', 'post'])
def switch_ai():
    ai_type = request.form.get('ai_type')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human
    res = AppServiceFactory.new_slice_analysis_service().switch_ui()
    return jsonify(res.dict())


@api_blueprint.route('/slice/addLabel', methods=['get', 'post'])
def add_label():
    ids = json.loads(request.form.get('ids'))
    name = request.form.get('name')
    res = AppServiceFactory.slice_service.add_label(ids=ids, name=name)
    return jsonify(res.dict())


@api_blueprint.route('/slice/delLabel', methods=['get', 'post'])
def del_label():
    file_id = request.form.get('id')
    names = json.loads(request.form.get('names'))
    res = AppServiceFactory.slice_service.del_label(id=file_id, name=names)
    return jsonify(res.dict())


@api_blueprint.route('/slice/getLabels', methods=['get', 'post'])
def get_labels():
    res = AppServiceFactory.slice_service.get_labels()
    return jsonify(res.dict())


@api_blueprint.route('/slice/get_template', methods=['get', 'post'])
def get_template():
    body = request.get_json()
    template_id = body.get('id')
    if not template_id or not isinstance(template_id, int):
        return jsonify(AppResponse(err_code=11, message='参数格式错误').dict())
    res = AppServiceFactory.new_slice_analysis_service().get_template(template_id)
    return jsonify(res.dict())


@api_blueprint.route('/slice/add_template', methods=['get', 'post'])
def add_template():
    body = request.get_json()
    name = body.get('name')
    ai_name = body.get('aiName')
    is_multi_mark = body.get('isMultiMark')
    mark_groups = body.get('markGroups')
    res = AppServiceFactory.new_slice_analysis_service().add_templates(name, ai_name, is_multi_mark, mark_groups)
    return jsonify(res.dict())


@api_blueprint.route('/slice/edit_template', methods=['get', 'post'])
def edit_template():
    body = request.get_json()
    template_id = body.get('id')
    name = body.get('name')
    ai_name = body.get('aiName', '')
    is_multi_mark = body.get('isMultiMark')
    mark_groups = body.get('markGroups')
    res = AppServiceFactory.new_slice_analysis_service().edit_templates(
        template_id, name, ai_name, is_multi_mark, mark_groups
    )
    return jsonify(res.dict())


@api_blueprint.route('/slice/del_template', methods=['get', 'post'])
def del_template():
    body = request.get_json()
    res = AppServiceFactory.new_slice_analysis_service().del_templates(body.get('id'))
    return jsonify(res.dict())
