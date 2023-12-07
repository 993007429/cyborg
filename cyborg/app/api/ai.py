import json
from flask import jsonify, request

from cyborg.app.api import api_blueprint
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.modules.slice_analysis.domain.value_objects import AIType
from cyborg.seedwork.application.responses import AppResponse


@api_blueprint.route('/ai/start', methods=['get', 'post'])
def start():
    other = request.form.get('other')  # 前端只在调用start接口时会传入参数 判断AI是否生成拼图供复核
    ai_name = request.form.get('type')
    try:
        rois = json.loads(other) if other else []
    except ValueError:
        rois = []

    upload_batch_number = request.form.get('uploadBatchNumber')  # 高通量批次号
    remote_ip = request.environ.get('REMOTE_ADDR')

    res = AppServiceFactory.ai_service.start_ai(
        ai_name=ai_name, rois=rois, upload_batch_number=upload_batch_number, ip_address=remote_ip)

    return jsonify(res.dict())


@api_blueprint.route('/ai/batchCalculation', methods=['get', 'post'])
def batch_calculation():
    case_ids = json.loads(request.form.get('caseid_list'))  # 病例id列表
    ai_name = request.form.get('algor_type')

    res = AppServiceFactory.ai_service.batch_start_ai(ai_name=ai_name, case_ids=case_ids)
    return jsonify(res.dict())


@api_blueprint.route('/ai/modelCalibration', methods=['get', 'post'])
def model_calibration():
    case_ids = json.loads(request.form.get('caseid_list'))  # 病例id列表
    ai_name = request.form.get('algor_type')  # 算法类型

    res = AppServiceFactory.ai_service.do_model_calibration(ai_name=ai_name, case_ids=case_ids)
    return jsonify(res.dict())


@api_blueprint.route('/ai/resetModelCalibration', methods=['get', 'post'])
def reset_model_calibration():
    """
    重置所有模块阈值到默认值
    :return:
    """
    res = AppServiceFactory.user_service.update_company_ai_threshold(model_name=None, use_default_threshold=True)
    return jsonify(res.dict())


@api_blueprint.route('/ai/cancelCalibration', methods=['get', 'post'])
def cancel_calibration():
    res = AppServiceFactory.ai_service.cancel_calibration()
    return jsonify(res.dict())


@api_blueprint.route('/ai/getresult', methods=['get', 'post'])
def get_result():
    ai_type = request.form.get('type')
    request_context.ai_type = AIType.get_by_value(ai_type)
    res = AppServiceFactory.ai_service.get_ai_task_result()
    return jsonify(res.dict())


@api_blueprint.route('/ai/revokeTask', methods=['get', 'post'])
def revoke_task():
    ai_type = request.form.get('type')
    request_context.ai_type = AIType.get_by_value(ai_type)

    res = AppServiceFactory.ai_service.cancel_task()
    return jsonify(res.dict())


@api_blueprint.route('/ai/analyseThreshold', methods=['get', 'post'])
def analyse_threshold():
    request_context.ai_type = AIType.get_by_value(request.form.get('alg_type'))
    search_key = json.loads(request.form.get('search_key')) if request.form.get('search_key') is not None else {} #筛选条件
    params = {
        'threshold_range': int(request.form.get('threshold_range', 0)),  #0:只改asc-h asc-us  1: 改全部
        'slice_range': int(request.form.get('slice_range', 1)), # 0 只改篩選  1: 改全部,
        'threshold_value': float(request.form.get('threshold_value')),
        'min_pos_cell': int(request.form.get('min_pos_cell', -1))
    }
    res = AppServiceFactory.ai_service.get_analyze_threshold(params=params, search_key=search_key)
    return jsonify(res.dict())


@api_blueprint.route('/ai/aiStatistics', methods=['get', 'post'])
def ai_statistics():
    request_context.ai_type = AIType.get_by_value(request.form.get('aiType'))
    start_date = request.form.get('startTime')
    end_date = request.form.get('endTime')
    res = AppServiceFactory.ai_service.get_ai_statistics(start_date=start_date, end_date=end_date)
    return jsonify(res.dict())


@api_blueprint.route('/files/save_info', methods=['get', 'post'])
def hack_ai_suggest():
    mode = request.form.get('mode')
    if mode == '1':
        res = AppServiceFactory.ai_service.hack_slide_quality()
    elif mode == '2':
        diagnosis = request.form.get('index1')
        microbe_list = json.loads(request.form.get("index2_list"))
        res = AppServiceFactory.ai_service.hack_ai_suggest(diagnosis=diagnosis, microbe_list=microbe_list)
    else:
        res = AppResponse()
    return jsonify(res.dict())
