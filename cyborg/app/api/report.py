from flask import request, send_file, jsonify

from cyborg.app.api import api_blueprint
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.infra.celery import async_get_result, AsyncGetResultTimeoutException
from cyborg.modules.slice.application import tasks
from cyborg.seedwork.application.responses import AppResponse
from cyborg.seedwork.domain.value_objects import AIType


@api_blueprint.route('/report/create', methods=['get', 'post'])
async def create_report():
    case_id = request.form.get('caseid')
    report_id = request.form.get('reportid')
    report_data = request.form['data']
    jwt = request.cookies.get('jwt')
    task_result = tasks.create_report(case_id=case_id, report_id=report_id, report_data=report_data, jwt=jwt)
    try:
        res = await async_get_result(task_result, polling_params=(150, 0.2))
    except AsyncGetResultTimeoutException as e:
        res = AppResponse(err_code=11, message="create pdf error: %s" % e)
    return jsonify(res.dict())


@api_blueprint.route('/report/getReportROI', methods=['get', 'post'])
def get_report_roi():
    alg_type = request.form.get('alg')
    request_context.ai_type = AIType.get_by_value(alg_type)
    res = AppServiceFactory.new_slice_analysis_service().get_report_roi()
    return jsonify(res.dict())


@api_blueprint.route('/report/getDNAInfo', methods=['get', 'post'])
def get_dna_info():
    request_context.ai_type = AIType.dna
    res = AppServiceFactory.new_slice_analysis_service().get_dna_info()
    return jsonify(res.dict())


@api_blueprint.route('/report/get', methods=['get', 'post'])
def get_report():
    report_id = request.form.get('reportid') or request.args.get('reportid')
    res = AppServiceFactory.slice_service.get_report_data(report_id=report_id)
    return jsonify(res.dict())


@api_blueprint.route('/report/view', methods=['get', 'post'])
def view_report():
    report_id = request.args.get('reportid')
    res = AppServiceFactory.slice_service.get_report_file(report_id=report_id)
    return send_file(res.data)


@api_blueprint.route('/report/config', methods=['get', 'post'])
def report():
    company = request.args.get('company')
    if request.method == 'GET':
        res = AppServiceFactory.slice_service.get_report_config(company=company)
        return jsonify(res.dict())
    elif request.method == 'POST':
        res = AppServiceFactory.slice_service.save_report_config(
            company=company,
            template_config=request.json
        )
        return jsonify(res.dict())
