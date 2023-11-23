import logging

from flask import jsonify, request

from cyborg.app.openapi.motic import motic_blueprint
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import PartnerAppServiceFactory
from cyborg.seedwork.domain.value_objects import AIType

logger = logging.getLogger(__name__)


@motic_blueprint.route('/openapi/v1/task/<string:motic_task_id>/analysis', methods=['post'])
def analysis(motic_task_id: str):
    request_context.ai_type = AIType.get_by_value(request.json.get('ai_type'))
    res = PartnerAppServiceFactory.motic_service.start_analysis(motic_task_id=motic_task_id)
    return jsonify(res.dict())


@motic_blueprint.route('/openapi/v1/task/<string:motic_task_id>/status', methods=['get'])
def get_analysis_status(motic_task_id: str):
    res = PartnerAppServiceFactory.motic_service.get_analysis_status(motic_task_id)
    return jsonify(res.dict())


@motic_blueprint.route('/openapi/v1/task/<string:motic_task_id>/result', methods=['get'])
def get_analysis_result(motic_task_id: str):
    res = PartnerAppServiceFactory.motic_service.get_analysis_result(motic_task_id)
    return jsonify(res.dict())


@motic_blueprint.route('/openapi/v1/task/<string:motic_task_id>/cancel', methods=['post'])
def stop_analysis(motic_task_id: str):
    res = PartnerAppServiceFactory.motic_service.cancel_analysis(motic_task_id)
    return jsonify(res.dict())
