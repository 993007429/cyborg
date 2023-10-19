# index蓝图的视图
import os
import sys
import json
from flask import request, jsonify

from cyborg.app.api import api_blueprint
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.seedwork.application.responses import AppResponse
from cyborg.seedwork.domain.value_objects import AIType


@api_blueprint.route('/manage/get_users', methods=['get', 'post'])
def get_users():
    company = request.form.get('grp_name')
    res = AppServiceFactory.user_service.get_users_by_company(company=company)
    return jsonify(res.dict())


@api_blueprint.route('/manage/get_user', methods=['get', 'post'])
def get_user():
    company = request.form.get('grp_name')
    username = request.form.get('user_name')
    res = AppServiceFactory.user_service.get_user(username=username, company=company)
    return jsonify(res.dict())


@api_blueprint.route('/manage/add_user', methods=['get', 'post'])
def add_user():
    company = request.form.get('grp_name')
    username = request.form.get('user_name')
    init_password = request.form.get('password')
    role = request.form.get("role")

    res = AppServiceFactory.user_service.create_user(
        username=username, password=init_password, company_id=company, role=role)

    return jsonify(res.dict())


@api_blueprint.route('/manage/alter_user', methods=['get', 'post'])
def update_user():
    new_user_name = request.form.get('new_user_name')
    new_password = request.form.get('password')
    new_role = request.form.get('role')
    user_id = request.form.get("id")
    res = AppServiceFactory.user_service.update_user(user_id=user_id, username=new_user_name, password=new_password, role=new_role)
    return jsonify(res.dict())


@api_blueprint.route('/manage/del_user', methods=['get', 'post'])
def delete_user():
    company = request.form.get('grp_name')
    username = request.form.get('user_name')
    res = AppServiceFactory.user_service.delete_user(username=username, company=company)
    return jsonify(res.dict())


@api_blueprint.route('/manage/get_attr', methods=['get', 'post'])
def get_company_attrs():
    company = request.form.get('grp_name')
    res = AppServiceFactory.user_service.get_company_detail(company_id=company)
    return jsonify(res.dict())


@api_blueprint.route('/manage/alter_attr', methods=['get', 'post'])
def update_company_attrs():
    is_test = request.form.get('isTest', type=int)
    res = AppServiceFactory.user_service.update_company(
        uid=request.form.get("id", type=int),
        old_company_name=request.form.get("old_grp_name"),
        new_company_name=request.form.get("new_grp_name"),
        model_lis=request.form.get("model_lis"),
        volume=request.form.get("volume"),
        remark=request.form.get("remark"),
        default_ai_threshold=request.form.get('defaultAiThreshold'),
        on_trial=int(request.form.get('onTrial')) if request.form.get('onTrial') else None,
        importable=int(request.form.get('importable')) if request.form.get('importable') else None,
        export_json=int(request.form.get('exportJson')) if request.form.get('exportJson') else None,
        trial_times=request.form.get('trialTimes'),
        is_test=is_test,
        end_time=request.form.get('endTime') if is_test else None
    )
    return jsonify(res.dict())


@api_blueprint.route('/manage/get_grp', methods=['get', 'post'])
def get_all_companies():
    res = AppServiceFactory.user_service.get_all_comanies()
    return jsonify(res.dict())


@api_blueprint.route('/manage/add_grp', methods=['get', 'post'])
def create_company():
    default_ai_threshold = request.form.get('defaultAiThreshold')
    res = AppServiceFactory.user_service.create_company(
        company_id=request.form.get('grp_name'),
        model_lis=request.form.get('model_lis'),
        volume=request.form.get('volume'),
        remark=request.form.get('remark'),
        ai_threshold=json.loads(default_ai_threshold),
        default_ai_threshold=default_ai_threshold,
        on_trial=int(request.form.get('onTrial')) if request.form.get('onTrial') else None,
        trial_times=request.form.get('trialTimes'),
        importable=int(request.form.get('importable')) if request.form.get('importable') else None,
        export_json=int(request.form.get('exportJson')) if request.form.get('exportJson') else None,
        is_test=request.form.get('isTest', type=int),
        end_time=request.form.get('endTime')
    )
    return jsonify(res.dict())


@api_blueprint.route('/manage/del_grp', methods=['get', 'post'])
async def delete_company():
    company = request.form.get('grp_name')
    res = await AppServiceFactory.user_service.delete_company(company_id=company)
    return jsonify(res.dict())


@api_blueprint.route('/manage/shutdown', methods=['get', 'post'])
def shutdown():
    os.system('shutdown /s /t 0')
    os.system('shutdown 0')
    return jsonify(AppResponse(message='即将关机').dict())


@api_blueprint.route('/manage/reboot', methods=['get', 'post'])  # 重启
def reboot():
    if sys.platform == 'linux':
        os.system('reboot')
    else:
        os.system('shutdown -r')
    return jsonify(AppResponse(message='重启服务器').dict())


@api_blueprint.route('/manage/saveAiThreshold', methods=['get', 'post'])
def save_ai_threshold():
    """
    保存算法参数
    """
    request_context.ai_type = AIType.get_by_value(request.form.get('algor_type'))
    threshold_range = int(request.form.get('threshold_range')) if request.form.get(
        'threshold_range') else None  # 0 只改asc-h asc-us  1: 改全部
    threshold_value = float(request.form.get('threshold_value'))
    all_use = request.form.get('all_use') == 'true'

    res = AppServiceFactory.user_service.save_ai_threshold(
        threshold_range=threshold_range, threshold_value=threshold_value, all_use=all_use)
    return jsonify(res.dict())


@api_blueprint.route('/manage/getAiThreshold', methods=['get', 'post'])
def get_ai_threshold():
    """
    获取算法参数
    """
    request_context.ai_type = AIType.get_by_value(request.form.get('algor_type'))
    res = AppServiceFactory.user_service.get_ai_threshold()
    return jsonify(res.dict())


@api_blueprint.route('/manage/getDefaultAiThreshold', methods=['get', 'post'])
def get_default_ai_threshold():
    request_context.ai_type = AIType.get_by_value(request.form.get('algor_type'))
    res = AppServiceFactory.user_service.get_default_ai_threshold()
    return jsonify(res.dict())


@api_blueprint.route('/manage/recordCount', methods=['get', 'post'])  # 输入截止日期，返回截止日期前的病例数量
def get_record_count():
    end_time = request.form.get('closing_date')  # 截止日期
    res = AppServiceFactory.slice_service.get_record_count(end_time=end_time)
    return jsonify(res.dict())


@api_blueprint.route('/manage/freeUpSpace', methods=['get', 'post'])
async def free_up_space():
    end_time = request.form.get('closing_date')  # 截止日期
    res = await AppServiceFactory.slice_service.free_up_space(end_time=end_time)
    return jsonify(res.dict())


@api_blueprint.route('/manage/config', methods=['get', 'post'])
def get_config():
    res = AppServiceFactory.slice_service.get_config()
    return jsonify(res.dict())
