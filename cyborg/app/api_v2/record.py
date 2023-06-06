import os

from flask import request, jsonify
from werkzeug import formparser

from cyborg.app.api_v2 import api_v2_blueprint
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.infra.fs import fs
from cyborg.utils.strings import camel_to_snake


@api_v2_blueprint.route('/records', methods=['post', 'delete'])
def records_handler():
    if request.method == 'post':
        record_data = request.json

        res = AppServiceFactory.slice_service.create_record(
            **{camel_to_snake(k): v for k, v in record_data}
        )
        return jsonify(res.dict())
    elif request.method == 'delete':
        case_ids = request.json.get('case_ids')
        res = AppServiceFactory.slice_service.delete_records(case_ids=case_ids)
        return jsonify(res.dict())


@api_v2_blueprint.route('/slices/upload', methods=['get', 'post'])
def upload_slice():
    slide_type = request.json.get('slideType')
    case_id = request.json.get('caseId')  # 病例id
    file_id = request.json.get('fileId')  # 文件id，一个切片文件对应一个id
    cover_slice_number = request.json.get('coverSliceNumber')  # 是否用病例号覆盖样本号，覆盖是true，不覆盖为false
    user_file_path = request.json.get('userFilePath')  # 上传端切片所在文件夹名称
    file_name = request.json.get('filename')  # 切片文件名
    upload_type = request.json.get('other')  # 上传模式，若为众包模式下上传此值为'cs'
    tool_type = request.json.get('toolType')
    total_upload_size = request.json.get('total')
    high_through = request.json.get('highThrough')

    upload_path = fs.path_join(request_context.current_user.data_dir, 'data', case_id, slide_type, file_id)
    if not fs.path_exists(upload_path):  # 切片文件不存在，即新上传切片文件
        os.makedirs(upload_path)

    def stream_factory(total_content_length, content_type, filename, content_length=None, start=0):
        return open(os.path.join(upload_path, filename), "wb+")

    formparser.parse_form_data(request.environ, stream_factory=stream_factory)

    res = AppServiceFactory.slice_service.upload_slice(
        case_id=case_id, file_id=file_id, company_id=request_context.current_company, file_name=file_name,
        slide_type=slide_type, upload_type=upload_type, upload_path=upload_path,
        total_upload_size=total_upload_size, tool_type=tool_type,
        user_file_path=user_file_path, cover_slice_number=cover_slice_number, high_through=high_through,
        operator=request_context.current_user.username
    )

    return jsonify(res.dict())
