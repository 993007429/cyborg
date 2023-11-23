from flask import request, jsonify

from cyborg.app.openapi.motic import motic_blueprint
from cyborg.app.service_factory import AppServiceFactory
from cyborg.seedwork.application.responses import AppResponse


@motic_blueprint.route('/oauth2/token', methods=['post'])
def get_access_token():
    """通过code获取access_token
    """
    client_id = request.form.get('client_id', '')
    client_secret = request.form.get('client_secret', '')
    grant_type = request.form.get('grant_type', '')
    res = AppServiceFactory.oauth_service.get_access_token(
        client_id=client_id, client_secret=client_secret, grant_type=grant_type)
    res = AppResponse(err_code=res.err_code, message=res.message, data=res.data)
    return jsonify(res.dict())
