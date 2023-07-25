import json
from flask import jsonify, request

from cyborg.app.openapi.roche import roche_blueprint
from cyborg.app.service_factory import AppServiceFactory


@roche_blueprint.route('/ai/start', methods=['get', 'post'])
def start():
    other = request.form.get('other')  # 前端只在调用start接口时会传入参数 判断AI是否生成拼图供复核
    ai_name = request.form.get('type')
    try:
        rois = json.loads(other) if other else []
    except ValueError:
        rois = []

    upload_batch_number = request.form.get('uploadBatchNumber')  # 高通量批次号
    remote_ip = request.environ.get('REMOTE_ADDR')

    res = AppServiceFactory.ai_service.start_ai(
        ai_name=ai_name, rois=rois, upload_batch_number=upload_batch_number, ip_address=remote_ip)

    return jsonify(res.dict())