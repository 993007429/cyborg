import json

from flask import request, jsonify

from cyborg.app.api import api_blueprint
from cyborg.app.limiter import limiter
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.modules.slice_analysis.domain.value_objects import AIType
from cyborg.seedwork.application.responses import AppResponse
from cyborg.utils.strings import camel_to_snake


@api_blueprint.route('/slice/createMark', methods=['get', 'post'])
@limiter.limit("2/second", override_defaults=False)
def create_mark():
    mark_params = json.loads(request.form.get('mark'))

    if not mark_params.get('ai_type'):
        mark_params['ai_type'] = request.form.get('ai_type', 'human')
    if mark_params.get('area_id') == -1:
        mark_params['area_id'] = None

    request_context.ai_type = AIType.get_by_value(mark_params.pop('ai_type')) or AIType.unknown

    res = AppServiceFactory.slice_analysis_service.create_mark(
        **{camel_to_snake(k): v for k, v in mark_params.items()})

    return jsonify(res.dict())


@api_blueprint.route('/slice/getMarks', methods=['get', 'post'])
def get_marks():
    view_path = json.loads(request.form.get('position'))  # 视野区域

    ai_type = request.form.get('ai_type')

    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human

    res = AppServiceFactory.slice_analysis_service.get_marks(view_path=view_path)

    return jsonify(res.dict())


@api_blueprint.route('/slice/selectCount', methods=['get', 'post'])
def select_count():
    scope = request.form.get('scope')
    ai_type = request.form.get('ai_type')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human

    res = AppServiceFactory.slice_analysis_service.count_marks_in_scope(scope=scope)
    return jsonify(res.dict())


@api_blueprint.route('/slice/markShow', methods=['get', 'post'])
@limiter.limit("2/second", override_defaults=False)
def mark_show():
    group_id = int(request.form.get('group_id'))
    request_context.ai_type = AIType.label

    res = AppServiceFactory.slice_analysis_service.switch_mark_group_show_status(group_id)
    return jsonify(res.dict())


@api_blueprint.route('/slice/modifyMark', methods=['get', 'post'])
def modify_marks():
    scope = request.form.get('scope')
    target_group_id = request.form.get('target_group_id')
    marks = json.loads(request.form.get('marks')) if request.form.get('marks') is not None else None

    ai_type = request.form.get('ai_type', '')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human

    if scope == 'null':
        scope = None

    res = AppServiceFactory.slice_analysis_service.update_marks(
        marks_data=marks, scope=scope, target_group_id=int(target_group_id) if target_group_id else None)
    return jsonify(res.dict())


@api_blueprint.route('/slice/deleteMark', methods=['get', 'post'])
def delete_mark():
    scope = request.form.get('scope')
    ai_type = request.form.get('ai_type')
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human

    mark_ids = json.loads(request.form.get('marks')) if request.form.get('marks') else None

    if scope == 'null':
        scope = None

    res = AppServiceFactory.slice_analysis_service.delete_marks(mark_ids=mark_ids, scope=scope)
    return jsonify(res.dict())


@api_blueprint.route('/slice/getROIList', methods=['get', 'post'])
def get_roi_list():
    ai_type = request.form.get('ai_type')
    print(ai_type)
    request_context.ai_type = AIType.get_by_value(ai_type) or AIType.human
    print(request_context.ai_type)
    res = AppServiceFactory.slice_analysis_service.get_rois()
    return jsonify(res.dict())


@api_blueprint.route('/slice/importAiResult', methods=['get', 'post'])
def import_ai_result():
    request_context.ai_type = AIType.label

    res = AppServiceFactory.slice_analysis_service.import_ai_marks()
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
    res = AppServiceFactory.slice_analysis_service.create_mark_group(template_id=template_id)
    return jsonify(res.dict())


@api_blueprint.route('/slice/modifyGroup', methods=['get', 'post'])
def modify_group():
    groups = json.loads(request.form.get('groups'))
    res = AppServiceFactory.slice_analysis_service.update_mark_groups(groups)
    return jsonify(res.dict())


@api_blueprint.route('/slice/deleteGroup', methods=['get', 'post'])
def delete_group():
    group_id = int(request.form.get('group_id'))

    request_context.ai_type = AIType.label

    res = AppServiceFactory.slice_analysis_service.delete_mark_group(group_id)
    return jsonify(res.dict())


@api_blueprint.route('/slice/selectGroup', methods=['get', 'post'])
@limiter.limit("2/second", override_defaults=False)
def select_group():
    group_id = int(request.form.get('group_id'))
    page = int(request.form.get('page'))
    per_page = int(request.form.get('per_page'))

    request_context.ai_type = AIType.label

    res = AppServiceFactory.slice_analysis_service.select_mark_group(
        group_id=group_id, page=page - 1, per_page=per_page)

    return jsonify(res.dict())


@api_blueprint.route('/slice/queryGroup', methods=['get', 'post'])
def query_group():
    group_id = int(request.form.get('group_id'))
    request_context.ai_type = AIType.label

    res = AppServiceFactory.slice_analysis_service.get_group_info(group_id)
    return jsonify(res.dict())


@api_blueprint.route('/slice/selectTemplate', methods=['get', 'post'])
def select_template():
    request_context.ai_type = AIType.label
    res = AppServiceFactory.slice_analysis_service.select_template(template_id=int(request.form.get('id')))
    return jsonify(res.dict())


@api_blueprint.route('/slice/templateList', methods=['get', 'post'])
def template_list():
    res = AppServiceFactory.slice_analysis_service.get_all_templates()
    return jsonify(res.dict())


@api_blueprint.route('/slice/getScreenCount', methods=['get', 'post'])
def get_screen_count():
    ai_type = request.form.get('ai_type')
    if ai_type.startswith('lct') or ai_type.startswith('tct'):
        ai_type = ai_type[0:3]
    request_context.ai_type = AIType.get_by_value(ai_type)

    view_path = json.loads(request.form.get('position'))

    res = AppServiceFactory.slice_analysis_service.get_cell_count_in_quadrant(view_path=view_path)
    return jsonify(res.dict())
