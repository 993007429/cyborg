from flask import request, jsonify

from cyborg.app.openapi import openapi_blueprint
from cyborg.app.service_factory import AppServiceFactory


@openapi_blueprint.route('/oauth2/token', methods=['post'])
def get_access_token():
    """通过code获取access_token
    """
    client_id = request.form.get('client_id', '')
    client_secret = request.form.get('client_secret', '')
    grant_type = request.form.get('grant_type', '')
    res = AppServiceFactory.oauth_service.get_access_token(
        client_id=client_id, client_secret=client_secret, grant_type=grant_type)
    return jsonify(res.dict())
