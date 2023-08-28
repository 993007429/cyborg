import logging

from flask import jsonify, request

from cyborg.app.openapi.roche import roche_blueprint
from cyborg.app.service_factory import RocheAppServiceFactory

logger = logging.getLogger(__name__)


@roche_blueprint.route('/openapi/v1/algorithms/<string:algorithm_id>', methods=['get'])
def get_algorithm_detail(algorithm_id: str):
    res = RocheAppServiceFactory.roche_service.get_algorithm_detail(algorithm_id)
    return jsonify(res.dict())


@roche_blueprint.route('/openapi/v1/algorithms', methods=['get'])
def get_algorithms():
    locale = request.args.get('locale')
    logger.info(f'locale: {locale}')
    res = RocheAppServiceFactory.roche_service.get_algorithm_list()
    return jsonify(res.dict())
