import logging
from flask import request, send_file, make_response, jsonify

from cyborg.app.api import api_blueprint
from cyborg.app.auth import login_required, LoginUser
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.app.settings import Settings
from cyborg.infra.fs import fs

logger = logging.getLogger(__name__)


@api_blueprint.route('/user/haveSign', methods=['get', 'post'])
@login_required
def is_sign():
    res = AppServiceFactory.user_service.is_signed()
    return jsonify(res.dict())


@api_blueprint.route('/user/login', methods=['get', 'post'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username.find("@") > -1:
        username, company = username.split("@")
    else:
        company = "company1"

    client_ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    res = AppServiceFactory.user_service.login(
        username=username, password=password, company=company, client_ip=client_ip)

    if res.err_code:
        return jsonify(res.dict())

    res.data['cloud'] = Settings.CLOUD
    del res.data['id']
    login_user = LoginUser.from_dict(res.data)

    resp = make_response(jsonify(res.dict()))
    resp.set_cookie(key='jwt', value=login_user.jwt_token, expires=login_user.expire_time)

    logger.info('%s组织下的%s用户登录' % (company, username))
    return resp


@api_blueprint.route('/user/sign')
def sign():  # 电子签名
    res = AppServiceFactory.user_service.get_current_user()
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
    res = AppServiceFactory.user_service.update_company_label(label=label)
    return jsonify(res.dict())
