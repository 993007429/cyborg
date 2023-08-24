import logging

from flask import Blueprint, jsonify, request

from cyborg.app.oauth import CurrentOauthClient
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.modules.oauth.application.responses import InvalidAccessTokenResponse, UnregisteredOAuthClientResponse

logger = logging.getLogger(__name__)
roche_blueprint = Blueprint('roche_bp', __name__)

WHITELIST_API_PATH = ['/roche/oauth2/token']


def api_before_request():
    request_context.begin_request()

    if request.path in WHITELIST_API_PATH:
        return

    authorization = request.headers.get('Authorization', '')
    access_token = authorization.replace('Bearer', '').strip() if 'Bearer' in authorization else ''
    if not access_token:
        return jsonify(InvalidAccessTokenResponse().dict())

    client = AppServiceFactory.oauth_service.get_client_by_access_token(token=access_token).data
    request_context.oauth_client = CurrentOauthClient.from_dict(client) if client else None
    if not request_context.oauth_client:
        return jsonify(UnregisteredOAuthClientResponse().dict())


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
