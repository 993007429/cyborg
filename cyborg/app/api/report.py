from flask import request, send_file, jsonify

from cyborg.app.api import api_blueprint
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.seedwork.domain.value_objects import AIType


@api_blueprint.route('/report/create', methods=['get', 'post'])
def create_report():
    report_id = request.form.get('reportid')
    report_data = request.form['data']
    jwt = request.cookies.get('jwt')
    res = AppServiceFactory.slice_service.create_report(report_id=report_id, report_data=report_data, jwt=jwt)
    return jsonify(res.dict())


@api_blueprint.route('/report/getReportROI', methods=['get', 'post'])
def get_report_roi():
    alg_type = request.form.get('alg')
    request_context.ai_type = AIType.get_by_value(alg_type)
    res = AppServiceFactory.slice_analysis_service.get_report_roi()
    return jsonify(res.dict())


@api_blueprint.route('/report/getDNAInfo', methods=['get', 'post'])
def get_dna_info():
    request_context.ai_type = AIType.dna
    res = AppServiceFactory.slice_analysis_service.get_dna_info()
    return jsonify(res.dict())


@api_blueprint.route('/report/get', methods=['get', 'post'])
def get_report():
    report_id = request.form.get('reportid')
    res = AppServiceFactory.slice_service.get_report_data(report_id=report_id)
    return jsonify(res.dict)


@api_blueprint.route('/report/view', methods=['get', 'post'])
def view_report():
    report_id = request.args.get('reportid')
    res = AppServiceFactory.slice_service.get_report_file(report_id=report_id)
    return send_file(res.data)
