import logging
import re

from flask import Blueprint, request, jsonify

from cyborg.app.auth import LoginUser
from cyborg.app.request_context import request_context
from cyborg.app.settings import Settings


logger = logging.getLogger(__name__)

api_blueprint = Blueprint('api', __name__)


def api_before_request():
    request_context.begin_request()

    request_context.case_id = request.args.get('caseid') or request.form.get('caseid')
    request_context.file_id = request.args.get('fileid') or request.form.get('fileid')

    if request.path == '/' or request.path.startswith('/static'):  # 首页和静态文件不验证
        pass

    elif request.path == '/info':

        times = 'times.txt not exist'
        sn = 'sn.txt not exist'
        return jsonify(times=times, sn=sn)

    elif re.match('|'.join(Settings.WHITE_LIST), request.path):  # 这些接口不需要鉴权 直接放行了
        companyid = request.args.get('companyid')
        if companyid:
            request_context.company = companyid
    else:
        try:
            login_user = LoginUser.get_from_cookie()
            if login_user.role in ['check', 'sa'] and request.args.get('companyid') is not None:
                request_context.company = request.args.get('companyid')
            else:
                request_context.company = login_user.company
            request_context.current_user = login_user
        except Exception as e:
            return jsonify(code=400, message='unauthorized access')


def api_after_request(response):
    request_context.end_request(commit=True)
    return response


@api_blueprint.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'code': 1,
        'message': '请降低操作频率'
    }), 200


@api_blueprint.errorhandler(Exception)
def server_error_handler(e):
    logger.exception(e)
    request_context.end_request()
    request_context.close_slice_db()
    return jsonify({
        'code': 1,
        'message': '服务器错误'
    }), 500


api_blueprint.before_request(api_before_request)
api_blueprint.after_request(api_after_request)
