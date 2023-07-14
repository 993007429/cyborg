import base64
import json
import os
from datetime import datetime

from flask import request, jsonify, send_file
from werkzeug import formparser

from cyborg.app.api import api_blueprint
from cyborg.app.auth import login_required
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.seedwork.application.responses import AppResponse


@api_blueprint.route('/records/getChecked', methods=['get', 'post'])
@login_required
def get_customized_record_fields():
    res = AppServiceFactory.user_service.get_customized_record_fields()
    return jsonify(res.dict())


@api_blueprint.route('/records/setChecked', methods=['get', 'post'])
@login_required
def set_customized_record_fields():
    res = AppServiceFactory.user_service.set_customized_record_fields(record_fields=request.form.get('tableChecked'))
    return jsonify(res.dict())


def _get_query_records_params() -> dict:
    page = request.form.get('page', 1, type=int) - 1   # 服务端下标统一从0开始
    limit = request.form.get('limit', 20, type=int)
    search_key = request.form.get("search_key")
    search_value = request.form.get("search_value")

    ai_suggest = json.loads(request.form.get("ai_suggest")) if (request.form.get("ai_suggest") and request.form.get(
        "ai_suggest") != '[]') else None
    check_result = json.loads(request.form.get("check_result")) if (request.form.get("check_result") and
                                                                    request.form.get("check_result") != '[]') else None
    user_file_folder = json.loads(request.form.get("userFileFolder")) if (
            request.form.get("userFileFolder") and request.form.get(
        "userFileFolder") != '[]') else None
    operator = json.loads(request.form.get("operator")) if (request.form.get("operator") and request.form.get(
        "operator") != '[]') else None

    report = json.loads(request.form.get("reports")) if (request.form.get("reports") and request.form.get(
        "reports") != '[]') else None  # 报告的有无   1有  2无   [1, 2]
    update_min = request.form.get("update_min").strip('"') if request.form.get("update_min") else None
    update_max = request.form.get("update_max").strip('"') if request.form.get("update_max") else None
    create_time_min = request.form.get("create_min").strip('"') if request.form.get("create_min") else None
    create_time_max = request.form.get("create_max").strip('"') if request.form.get("create_max") else None
    gender = json.loads(request.form.get("gender")) if (
            request.form.get("gender") and request.form.get("gender") != '[]') else None
    age_min = request.form.get('age_min', type=int)
    age_max = request.form.get('age_max', type=int)
    sample_part = json.loads(request.form.get("samplePart")) if (
            request.form.get("samplePart") and request.form.get("samplePart") != '[]') else None  # 多选
    sample_type = json.loads(request.form.get("sampleType")) if (
            request.form.get("sampleType") and request.form.get("sampleType") != '[]') else None
    statuses = json.loads(request.form.get("status")) if (
            request.form.get("status") and request.form.get("status") != '[]') else None  # 0未处理 1处理中 2已处理 3处理异常
    alg = json.loads(request.form.get("alg_list")) if (
            request.form.get("alg_list") and request.form.get("alg_list") != '[]') else None  # ['tct','ki67']
    seq_key = request.form.get("seq_key") if request.form.get(
        "seq_key") else "create_time"  # 年龄 切片数量  更新时间 创建时间 ['age','slice_num','update_time','sampleNum','create_time']
    seq = request.form.get("seq") if request.form.get("seq") else '1'  # 默认顺序  1倒序  2正序

    slice_no = json.loads(request.form.get("slice_no")) if (request.form.get("slice_no") and request.form.get(
        "slice_no") != '[]') else None
    is_has_label = json.loads(request.form.get("label")) if (request.form.get("label") and request.form.get(
        "label") != '[]') else None
    return locals()


@api_blueprint.route('/records/search', methods=['get', 'post'])
def search():
    res = AppServiceFactory.slice_service.search_records(**_get_query_records_params())
    return jsonify(res.dict())


@api_blueprint.route('/records/all', methods=['get', 'post'])
def get_all_values_in_fields():
    # 告诉前端 取样部位和样本类型有哪些 以供前端搜索用
    res = AppServiceFactory.slice_service.get_all_values_in_fields()
    return jsonify(res.dict())


@api_blueprint.route('/records/polling', methods=['get', 'post'])
def get_new_records():
    updated_after = request.form.get('updatedAfter')
    res = AppServiceFactory.slice_service.get_new_slices(
        start_id=int(request.form.get('startId', 0)),
        updated_after=datetime.strptime(updated_after, '%Y-%m-%d %H:%M:%S') if updated_after else None,
        upload_batch_number=request.form.get('uploadBatchNumber')
    )
    return jsonify(res.dict())


@api_blueprint.route('/records/export', methods=['get', 'post'])
def export_records():
    res = AppServiceFactory.slice_service.export_records(**_get_query_records_params())
    if res.data:
        return send_file(res.data)
    else:
        return jsonify(res.dict())


@api_blueprint.route('/records/delete', methods=['get', 'post'])
def delete_records():
    res = AppServiceFactory.slice_service.delete_records(case_ids=json.loads(request.form["id"]))
    return jsonify(res.dict())


@api_blueprint.route('/records/get', methods=['get', 'post'])
@login_required
def get_record():  # 前端传来caseid 看数据库是否存在该病例
    res = AppServiceFactory.slice_service.get_record_by_case_id(
        case_id=request.form.get('caseid')
    )
    return jsonify(res.dict())


@api_blueprint.route('/records/setCaseid', methods=['get', 'post'])
@login_required
def update_sample_num():
    res = AppServiceFactory.slice_service.update_sample_num(
        case_id=request.form.get('id'),
        sample_num=request.form.get('caseid')
    )
    return jsonify(res.dict())


@api_blueprint.route('/records/update', methods=['get', 'post'])
@login_required
def update_case_record():
    case_id = request.form.get('id')
    company = request.args.get('companyid')
    content = request.form.get('content')
    content = json.loads(content)  # content={'basic':{},'reports':xxx,'slices':xxx,....}  basic存储的是病例基础信息
    basic = content.get('basic', None)
    reports = content.get('reports', [])

    res = AppServiceFactory.slice_service.update_record(
        case_id, company,
        company_report_name=content.get('report_name', None),
        company_report_info=content.get('report_info', None),
        logo=content.get('logo', None),
        attachments=content.get('attachments', []),
        record_name=basic.get('name', None),
        age=basic.get('age', None) or None,
        gender=basic.get('gender', None),
        cancer_type=basic.get('cancerType', None),
        family_history=basic.get('familyHistory', None),
        medical_history=basic.get('medicalHistory', None),
        generally_seen=basic.get('generallySeen', None),
        sample_num=basic.get('caseid', None),
        sample_part=basic.get('samplePart', None),
        sample_time=basic.get('sampleTime', None),
        sample_collect_date=basic.get('sampleCollectDate', None),
        sample_type=basic.get('sampleType', None),
        inspection_hospital=basic.get('inspectionHospital', None),
        inspection_doctor=basic.get('inspectionDoctor', None),
        report_info=content.get('reportInfo', None),
        opinion=content.get('opinion', None),
        stage=int(content.get('stage', 0) or 0),
        started=0,
        state=1,
        report=json.dumps(reports[0]) if reports else 2
    )
    return jsonify(res.dict())


@api_blueprint.route('/records/uploadImportDoc', methods=['get', 'post'])
def upload_import_doc():
    # 上传病例导入文件
    res = AppResponse()
    import_cache_path = os.path.join(request_context.current_user.data_dir, 'importCache.xlsx')
    if os.path.exists(import_cache_path):
        os.remove(import_cache_path)

    def stream_factory(total_content_length, content_type, filename, content_length=None, start=0):
        return open(import_cache_path, "wb+")

    formparser.parse_form_data(request.environ, stream_factory=stream_factory)
    return jsonify(res.dict())


@api_blueprint.route('/records/importRecords', methods=['get', 'post'])
@login_required
def import_records():
    res = AppServiceFactory.slice_service.import_records()
    return jsonify(res.dict())


@api_blueprint.route('/records/getReportOpinion', methods=['get', 'post'])
def get_report_opinion():
    res = AppServiceFactory.slice_service.get_report_opinion()
    return jsonify(res.dict())


@api_blueprint.route('/records/report/sync', methods=['get', 'post'])
async def sync_report():
    res = await AppServiceFactory.slice_service.sync_report()
    return jsonify(res.dict())


@api_blueprint.route('/records/getLogo', methods=['get', 'post'])
def get_logo():
    target_path = os.path.join(request_context.current_user.data_dir, "logo.jpg")
    if os.path.exists(target_path):
        with open(target_path, "rb") as f:  # 转为二进制格式
            base64_data = base64.b64encode(f.read())  # 使用base64进行加密
            res = AppResponse(data='data:image/jpeg;base64,' + str(base64_data, encoding="utf-8"))
            return jsonify(res.dict())
    else:
        return jsonify(AppResponse().dict())
