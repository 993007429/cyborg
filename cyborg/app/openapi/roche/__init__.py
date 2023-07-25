import logging

from flask import Blueprint, jsonify

from cyborg.app.request_context import request_context

logger = logging.getLogger(__name__)
roche_blueprint = Blueprint('roche', __name__)


def api_before_request():
    request_context.begin_request()


def api_after_request(response):
    request_context.end_request(commit=True)
    return response


@roche_blueprint.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'code': 1,
        'message': '请降低操作频率'
    }), 200


@roche_blueprint.errorhandler(Exception)
def server_error_handler(e):
    logger.exception(e)
    request_context.end_request()
    request_context.close_slice_db()
    return jsonify({
        'code': 1,
        'message': '服务器错误'
    }), 500


roche_blueprint.before_request(api_before_request)
roche_blueprint.after_request(api_after_request)
