from flask import request, send_file, make_response, jsonify

from cyborg.app.api import api_blueprint
from cyborg.app.auth import login_required, LoginUser
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.infra.fs import fs
from cyborg.utils.strings import dict_camel_to_snake


@api_blueprint.route('/user/haveSign', methods=['get', 'post'])
@login_required
def is_sign():
    res = AppServiceFactory.user_service.is_signed()
    return jsonify(res.dict())


@api_blueprint.route('/user/login', methods=['get', 'post'])
def login():
    username = request.form['username']
    password = request.form['password']
    client_ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    res = AppServiceFactory.user_service.login(username=username, password=password, client_ip=client_ip)
    resp = make_response(jsonify(res.dict()))
    if not res.err_code:
        login_user = LoginUser.from_dict(dict_camel_to_snake(res.data))
        resp.set_cookie(key='jwt', value=login_user.jwt_token, expires=login_user.expire_time)
    return resp


@api_blueprint.route('/user/sign2')
@api_blueprint.route('/user/sign')
def sign():  # 电子签名
    res = AppServiceFactory.user_service.get_current_user(user_name=request.args.get('id'))
    user_info = res.data
    sign_image_path = user_info.get('sign_image_path') if user_info else ''
    if sign_image_path and fs.path_exists(sign_image_path):
        return send_file(sign_image_path)
    return '', 404


@api_blueprint.route('/user/updateSign', methods=['post'])
@login_required
def update_sign():
    user = request_context.current_user
    res = AppServiceFactory.user_service.update_signed(
        username=user.username, company=user.company, file=request.files['sign'])
    return jsonify(res.dict())


@api_blueprint.route('/user/updatePassword', methods=['post'])
def update_password():
    res = AppServiceFactory.user_service.update_password(
        old_password=request.form['password'],
        new_password=request.form['newPassword']
    )
    return jsonify(res.dict())


@api_blueprint.route('/manage/get_trial_times', methods=['get', 'post'])
def get_trial_times():
    res = AppServiceFactory.user_service.get_company_trail_info()
    return jsonify(res.dict())


@api_blueprint.route('/manage/remainingSpace', methods=['get', 'post'])
def remaining_space():
    res = AppServiceFactory.user_service.get_company_storage_info()
    return jsonify(res.dict())


@api_blueprint.route('/ai/getLabel', methods=['get', 'post'])
def get_company_label():
    res = AppServiceFactory.user_service.get_company_label()
    return jsonify(res.dict())


@api_blueprint.route('/ai/modifyLabel', methods=['get', 'post'])
def modify_label():
    label = int(request.form.get('label'))
    clarity_standards_max = float(request.form.get('clarityStandardsUpper', 0.6))
    clarity_standards_min = float(request.form.get('clarityStandardsLower', 0.2))
    res = AppServiceFactory.user_service.update_company_label(label=label,
                                                              clarity_standards_min=clarity_standards_min,
                                                              clarity_standards_max=clarity_standards_max)
    return jsonify(res.dict())


@api_blueprint.route("/manage/xfyun-websocket-service-url", methods=['POST'], endpoint="get_ws_url")
def get_ws_url():
    url = AppServiceFactory.user_service.get_ws_url()
    return jsonify({"code": 0, "data": {"url": url}, "message": "生成websocket url成功"})
