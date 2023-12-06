import json
import logging

from flask import jsonify, request, Response

from cyborg.app.openapi.roche import roche_blueprint
from cyborg.app.service_factory import PartnerAppServiceFactory

logger = logging.getLogger(__name__)


@roche_blueprint.route('/openapi/v1/analysis', methods=['post'])
def analysis():
    algorithm_id = request.json.get('algorithm_id')
    image_url = request.json.get('image_url')

    logger.info('>>>>>>>>>>')
    logger.info(request.json)

    res = PartnerAppServiceFactory.roche_service.start_ai(
        algorithm_id=algorithm_id, slide_url=image_url, input_info=request.json)

    return jsonify(res.dict())


@roche_blueprint.route('/openapi/v1/analysis/<string:analysis_id>/secondary', methods=['post'])
def secondary_analysis(analysis_id: str):
    regions = request.json.get('regions')

    res = PartnerAppServiceFactory.roche_service.start_secondary_ai(
        analysis_id=analysis_id, regions=regions)

    return jsonify(res.dict())


@roche_blueprint.route('/openapi/v1/analysis/<string:analysis_id>/roi', methods=['post'])
def roi_analysis(analysis_id: str):
    regions = request.json.get('regions')
    # algorithm_id = request.json.get('algorithm_id')

    res = PartnerAppServiceFactory.roche_service.rescore(
        analysis_id=analysis_id, regions=regions)

    logger.info(res.dict())
    # 由于使用jsonify会对dict进行自动排序，这样可以避免
    return Response(json.dumps(res.dict()), mimetype='application/json')


@roche_blueprint.route('/openapi/v1/analysis/<string:analysis_id>/status', methods=['get'])
def get_analysis_status(analysis_id: str):
    res = PartnerAppServiceFactory.roche_service.get_task_status(analysis_id)
    return jsonify(res.dict())


@roche_blueprint.route('/openapi/v1/analysis/<string:analysis_id>/stop', methods=['post'])
def stop_analysis(analysis_id: str):
    res = PartnerAppServiceFactory.roche_service.cancel_task(analysis_id)
    return jsonify(res.dict())


@roche_blueprint.route('/openapi/v1/analysis/<string:analysis_id>/close', methods=['post'])
def close_analysis(analysis_id: str):
    res = PartnerAppServiceFactory.roche_service.close_task(analysis_id)
    return jsonify(res.dict())


@roche_blueprint.route('/openapi/v1/analysis/<string:analysis_id>/result', methods=['get'])
def get_analysis_result(analysis_id: str):
    res = PartnerAppServiceFactory.roche_service.get_task_result(analysis_id)
    # 由于使用jsonify会对dict进行自动排序，这样可以避免
    return Response(json.dumps(res.dict()), mimetype='application/json')
