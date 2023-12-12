import os

from flask import jsonify, make_response, send_from_directory

from cyborg.app.api import api_blueprint
from cyborg.app.service_factory import PartnerAppServiceFactory


@api_blueprint.route('/analysis/<string:analysis_id>/result-file', methods=['get'])
def get_analysis_result_file(analysis_id: str):

    res = PartnerAppServiceFactory.roche_service.get_result_file_path(analysis_id)
    if res.err_code:
        return jsonify(res.dict())

    resp = make_response(send_from_directory(
        directory=os.path.split(res.data)[0],
        path=os.path.split(res.data)[1],
        mimetype='.h5',
        # as_attachment=True
    ))
    return resp
