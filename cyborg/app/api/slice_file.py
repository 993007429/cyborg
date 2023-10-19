import ast
import base64
import json
import logging
import mimetypes
import os

from celery.result import AsyncResult
from celery.exceptions import TimeoutError as CeleryTimeoutError
from flask import request, jsonify, make_response, send_from_directory, send_file
from werkzeug import formparser

from cyborg.app.api import api_blueprint
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.app.settings import Settings
from cyborg.infra.fs import fs
from cyborg.modules.slice.application import tasks
from cyborg.modules.slice.domain.value_objects import SliceImageType
from cyborg.seedwork.application.responses import AppResponse
from cyborg.celery.app import app as celery_app

logger = logging.getLogger(__name__)

download_thread_dict = dict()


@api_blueprint.route('/files/upload2', methods=['get', 'post'])
def upload_slice_by_dir():
    slide_type = request.args.get('type') + 's'
    uploadid = request.args.get('uploadid')
    fileid = request.args.get('fileid')
    filename = os.sep.join(request.args.get('filename').split('\\'))

    slide_save_path = os.path.join(
        request_context.current_user.data_dir, 'upload_data', uploadid, slide_type, fileid, filename)
    os.makedirs(os.path.dirname(slide_save_path), exist_ok=True)
    with open(slide_save_path, "wb+") as f:
        while True:
            chunk = request.stream.read(4096)
            if len(chunk) == 0:
                break
            f.write(chunk)
    res = AppResponse()
    return jsonify(res.dict())


@api_blueprint.route('/files/upload', methods=['get', 'post'])
def upload_slice():
    slide_type = request.args.get('type') + 's'
    case_id = request.args.get('caseid')  # 病例id
    upload_id = request.args.get('uploadid')  # 病例id
    file_id = request.args.get('fileid')  # 文件id，一个切片文件对应一个id
    cover_slice_number = request.args.get('cover_slice_number') == 'true'  # 是否用病例号覆盖样本号，覆盖是true，不覆盖为false
    user_file_path = request.args.get('userFilePath')  # 上传端切片所在文件夹名称
    file_name = request.args.get('filename')  # 切片文件名
    tool_type = request.args.get('toolType')
    total_upload_size = int(request.args.get('total', 0))
    high_through = bool(request.args.get('high_through'))
    upload_batch_number = request.args.get('uploadBatchNumber')

    if high_through:
        upload_path = os.path.join(request_context.current_user.data_dir, 'upload_data', upload_id, slide_type, file_id)
    else:
        upload_path = os.path.join(request_context.current_user.data_dir, 'data', case_id, slide_type, file_id)

    if not os.path.exists(upload_path):  # 切片文件不存在，即新上传切片文件
        os.makedirs(upload_path)

    def stream_factory(total_content_length, content_type, filename, content_length=None, start=0):
        return open(os.path.join(upload_path, filename), "wb+")

    formparser.parse_form_data(request.environ, stream_factory=stream_factory)

    res = AppServiceFactory.slice_service.upload_slice(
        upload_id=upload_id, case_id=case_id, file_id=file_id, company_id=request_context.current_company,
        file_name=file_name, slide_type=slide_type, upload_path=upload_path,
        total_upload_size=total_upload_size, tool_type=tool_type,
        user_file_path=user_file_path, cover_slice_number=cover_slice_number, high_through=high_through,
        upload_batch_number=upload_batch_number, operator=request_context.current_user.username
    )

    return jsonify(res.dict())


@api_blueprint.route('/files/saveInfo', methods=['get', 'post'])
def update_slice_info():
    case_id = request.form.get('caseid')
    file_id = request.form.get('fileid')
    high_through = bool(request.form.get('high_through'))  # 是否是高通量上传
    content = request.form.get('content')
    content = json.loads(content)
    content.pop('labels', '')
    res = AppServiceFactory.slice_service.update_slice_info(
        case_id=case_id, file_id=file_id, high_through=high_through, info=content)
    return jsonify(res.dict())


@api_blueprint.route('/files/getInfo', methods=['get', 'post'])
def get_slice_info():
    company_id = request.args.get('companyid')
    res = AppServiceFactory.slice_service.get_slice_info(
        case_id=request_context.case_id, file_id=request_context.file_id, company_id=company_id)
    if res.err_code:
        return jsonify(res.dict())

    res.data['group'] = AppServiceFactory.new_slice_analysis_service().get_selected_mark_group().data

    return jsonify(res.dict())


@api_blueprint.route('/files/delSlice', methods=['get', 'post'])
def delete_slice():
    case_id = request.form.get('caseid')
    file_id = request.form.get('fileid')
    res = AppServiceFactory.slice_service.delete_slice(case_id, file_id)
    return jsonify(res.dict())


@api_blueprint.route('/files/delJunkFile', methods=['get', 'post'])
def del_junk_file():
    res = AppResponse(message='operation succeed')
    return jsonify(res.dict())


@api_blueprint.route('/files/getLabel', methods=['get', 'post'])
def get_label():
    caseid = request.args.get('caseid')
    fileid = request.args.get('fileid')

    content_full_path = os.path.join(request_context.current_user.data_dir, "data", caseid, "slices", fileid)

    if os.path.exists(os.path.join(content_full_path, 'label.png')):
        resp = make_response(send_from_directory(
            directory=content_full_path,
            path='label.png',
            mimetype="images/png",
            as_attachment=False
        ))
        return resp
    else:
        return jsonify(AppResponse(message='label not exist').dict())


@api_blueprint.route('/files/ocrlabel', methods=['get', 'post'])
def ocr_label():
    caseid = request.args.get('caseid')
    fileid = request.args.get('fileid')

    slice_label_path = fs.path_join(
        request_context.current_user.data_dir, 'data', caseid, 'slices', fileid, 'slice_label.jpg')

    if fs.path_exists(slice_label_path) and fs.get_file_size(slice_label_path):
        resp = make_response(send_from_directory(
            directory=fs.path_dirname(slice_label_path),
            path='slice_label.jpg',
            mimetype="image/png",
            as_attachment=False
        ))
        return resp
    else:
        resp = make_response(send_from_directory(
            directory=os.path.join(Settings.PROJECT_DIR, 'resources', 'ocr'),
            path='default.png',
            mimetype="image/png",
            as_attachment=False
        ))
        return resp


@api_blueprint.route('/files/saveImage', methods=['get', 'post'])
def save_image():
    caseid = request.form.get('caseid')
    fileid = request.form.get('fileid')
    img_type = request.form.get('img_type', type=str)
    img_base64 = request.form.get('img_base64')

    if img_type == SliceImageType.histplot:
        img_path = os.path.join(request_context.current_user.data_dir, 'data', caseid, 'slices', fileid, 'dna_index.png')
    elif img_type == SliceImageType.scatterplot:
        img_path = os.path.join(request_context.current_user.data_dir, 'data', caseid, 'slices', fileid, 'scatterplot.png')
    else:
        return jsonify(AppResponse(err_code=1).dict())

    binary_data = base64.b64decode(img_base64.split(',')[1])
    with open(img_path, 'wb') as output_file:
        output_file.write(binary_data)

    return jsonify(AppResponse(err_code=0, message='success').dict())


@api_blueprint.route('/files/getImage', methods=['get', 'post'])
@api_blueprint.route('/files/getImage2', methods=['get', 'post'])
def get_image():
    caseid = request.args.get('caseid')
    fileid = request.args.get('fileid')
    company = request.args.get('company')
    img_type = request.args.get('type', type=str)

    if company:
        if img_type == SliceImageType.histplot:
            img_path = fs.path_join(Settings.DATA_DIR, company, 'data', caseid, 'slices', fileid, 'dna_index.png')
        elif img_type == SliceImageType.scatterplot:
            img_path = fs.path_join(Settings.DATA_DIR, company, 'data', caseid, 'slices', fileid, 'scatterplot.png')
        else:
            img_path = ''
    else:
        if img_type == SliceImageType.histplot:
            img_path = fs.path_join(
                request_context.current_user.data_dir, 'data', caseid, 'slices', fileid, 'dna_index.png')
        elif img_type == SliceImageType.scatterplot:
            img_path = fs.path_join(
                request_context.current_user.data_dir, 'data', caseid, 'slices', fileid, 'scatterplot.png')
        else:
            img_path = ''

    logger.info(img_path)
    if os.path.exists(img_path) and os.path.getsize(img_path):
        resp = make_response(send_from_directory(
            directory=fs.path_dirname(img_path),
            path=fs.path_basename(img_path),
            mimetype="image/png",
            as_attachment=False
        ))
        return resp
    else:
        resp = make_response(send_from_directory(
            directory=os.path.join(Settings.PROJECT_DIR, 'static'),
            path='default.png',
            mimetype="image/png",
            as_attachment=False
        ))
        return resp


@api_blueprint.route('/files/thumbnail', methods=['get', 'post'])
def thumbnail():
    case_id = request.args.get('caseid')
    file_id = request.args.get('fileid')

    res = AppServiceFactory.slice_service.get_slice_thumbnail(case_id=case_id, file_id=file_id)

    if res.err_code:
        return jsonify(res.dict())

    resp = make_response(res.data.getvalue())
    resp.mimetype = 'image/jpeg'
    res.data.close()
    return resp


@api_blueprint.route('/files/attachment', methods=['get', 'post'])
def attachment():
    case_id = request.args.get('caseid')
    file_id = request.args.get('fileid')

    res = AppServiceFactory.slice_service.get_attachment_file_path(case_id=case_id, file_id=file_id)
    if res.err_code:
        return jsonify(res.dict())

    resp = make_response(send_from_directory(
        directory=os.path.split(res.data)[0],
        path=os.path.split(res.data)[1],
        mimetype=mimetypes.guess_type(res.data)[0],
        as_attachment=True
    ))
    return resp


@api_blueprint.route('/files/ROI', methods=['get', 'post'])
@api_blueprint.route('/files/ROI2', methods=['get', 'post'])
def get_roi():
    case_id = request.args.get('caseid')
    file_id = request.args.get('fileid')
    roi = request.args.get('roi')
    roi = ast.literal_eval(roi)
    roi_id = request.args.get('roiid')

    res = AppServiceFactory.slice_service.get_roi(case_id=case_id, file_id=file_id, roi=roi, roi_id=roi_id)
    if res.err_code:
        return jsonify(res.dict())

    resp = make_response(res.data.getvalue())
    resp.mimetype = 'image/jpeg'
    res.data.close()
    return resp


@api_blueprint.route('/files/SEG', methods=['get', 'post'])
def seg():
    case_id = request.args.get('caseid')
    file_id = request.args.get('fileid')
    dna_index = float(request.args.get('dna_index', 0))
    roi = request.args.get('roi')
    roi = ast.literal_eval(roi)

    res = AppServiceFactory.slice_service.get_roi_and_segment(case_id=case_id, file_id=file_id, roi=roi, dna_index=dna_index)
    if res.err_code:
        return jsonify(res.dict())

    resp = make_response(res.data.getvalue())
    resp.mimetype = 'image/jpeg'
    res.data.close()
    return resp


@api_blueprint.route('/screenshot', methods=['get', 'post'])
def screenshot():
    case_id = request.args.get('caseid')
    file_id = request.args.get('fileid')
    roi = request.args.get('roi')
    roi = ast.literal_eval(roi)
    roi_id = request.args.get('roiid')

    res = AppServiceFactory.slice_service.get_screenshot(case_id=case_id, file_id=file_id, roi=roi, roi_id=roi_id)
    if res.err_code:
        return jsonify(res.dict())

    dir_path, file_name = os.path.split(res.data)

    resp = make_response(send_from_directory(
        directory=dir_path,
        path=file_name,
        mimetype=mimetypes.guess_type(file_name)[0],
        as_attachment=True
    ))
    return resp


@api_blueprint.route('/files/slice', methods=['get', 'post'])
def get_slice_image():
    x: int = request.args.get('x', type=int)
    y: int = request.args.get('y', type=int)
    z: int = request.args.get('z', type=int)
    case_id = request.args.get('caseid')
    file_id = request.args.get('fileid')

    res = AppServiceFactory.slice_service.get_slice_image(case_id=case_id, file_id=file_id, axis=(x, y, z))
    if res.err_code:
        return jsonify(res.dict())

    if 'tile_path' in res.data:
        resp = make_response(send_from_directory(
            directory=os.path.split(res.data['tile_path'])[0],
            path=os.path.split(res.data['tile_path'])[1],
            mimetype='image/jpeg',
            as_attachment=False
        ))
        return resp
    elif 'image_data' in res.data:
        buf = res.data['image_data']
        resp = make_response(buf.getvalue())
        resp.mimetype = 'image/jpeg'
        buf.close()
        return resp
    else:
        return jsonify(res.dict())


@api_blueprint.route('/files/importDb', methods=['get', 'post'])
def import_db():
    case_id = request.args.get('caseid')
    file_id = request.args.get('fileid')

    res = AppServiceFactory.slice_service.get_slice_data_paths(case_id=case_id, file_id=file_id)
    if res.err_code:
        return jsonify(AppResponse(err_code=res.err_code, message=res.message).dict())

    def stream_factory(total_content_length, content_type, filename, content_length=None, start=0):
        return open(os.path.join(res.data['slice_dir'], filename), "wb+")

    formparser.parse_form_data(request.environ, stream_factory=stream_factory)

    return jsonify(AppResponse().dict())


@api_blueprint.route('/files/exportDb', methods=['get', 'post'])
def export_db():
    case_id = request.form.get('caseid')
    file_id = request.form.get('fileid')
    res = AppServiceFactory.slice_service.get_slice_data_paths(case_id=case_id, file_id=file_id)
    if res.err_code:
        return jsonify(AppResponse(err_code=res.err_code, message=res.message).dict())

    try:
        return send_file(res.data['db_file_path'])
    except Exception as e:
        logger.warning(e)
        return jsonify(AppResponse(err_code=1).dict())


@api_blueprint.route('/files/getAngle', methods=['get', 'post'])
def get_angle():
    file_id = request.form.get('fileid')
    res = AppServiceFactory.slice_service.get_slice_angle(file_id=file_id)
    return jsonify(res.dict())


@api_blueprint.route('/files/alterAngle', methods=['get', 'post'])
def update_angle():
    case_id = request.form.get('caseid')
    file_id = request.form.get('fileid')
    current_angle = float(request.form.get('currentAngle', 0))
    res = AppServiceFactory.slice_service.update_slice_angle(case_id=case_id, file_id=file_id, current_angle=current_angle)
    return jsonify(res.dict())


@api_blueprint.route('/files/capture', methods=['get', 'post'])
def capture():
    capture_file = request.files.get('capture')
    res = AppServiceFactory.slice_service.capture(capture=capture_file)
    return jsonify(res.dict())


@api_blueprint.route('/files/stagePos', methods=['get', 'post'])
def stage_pos():
    case_id = request.args.get("caseid")
    file_id = request.args.get("fileid")
    x_position = request.form.get("xPos")
    y_position = request.form.get("yPos")

    res = AppServiceFactory.slice_service.stage_position(
        case_id=case_id, file_id=file_id, x_position=x_position, y_position=y_position)

    # TODO HACK
    res_dict = res.dict()
    if res.data:
        res_dict['xPos'] = res.data['x']
        res_dict['yPos'] = res.data['y']
        res_dict['zPos'] = res.data['z']

    return jsonify(res_dict)


@api_blueprint.route('/files/download', methods=['get', 'post'])  # 前端起服务，后端上传文件
def download():
    need_db_file = bool(int(request.form.get('want_db')))  # 是否需要导出db文件，需要导出为1反之为0
    case_ids = json.loads(request.form.get('caseid'))  # 需要导出文件的病例id列表
    path = request.form.get('path')
    ip = request.form.get('ip')

    async_result = tasks.export_slice_files(
        client_ip=ip, case_ids=case_ids, path=path, need_db_file=need_db_file)

    res = AppResponse(message='下载请求发送成功', data={'key': async_result.id})
    return jsonify(res.dict())


@api_blueprint.route('/files/getDownloadResult', methods=['get', 'post'])
def get_download_result():
    res = AppResponse()
    key = request.form.get('key')

    try:
        result = AsyncResult(key, app=celery_app)
        if result.ready():
            async_res = result.get(timeout=0.1)
            if async_res:
                res = async_res
                res.data = {'done': True}
        else:
            res = AppResponse(data={'done': False})
    except CeleryTimeoutError:
        res = AppResponse(err_code=1, message='下载超时')
    except Exception as e:
        logger.exception(e)
        res = AppResponse(err_code=1, message='下载发生异常')

    return jsonify(res.dict())


@api_blueprint.route('/files/cancelDownload', methods=['get', 'post'])
def cancel_download():
    key = request.form.get('key')
    try:
        result = AsyncResult(key, app=celery_app)
        if not result.ready():
            result.revoke(terminate=True)
        res = AppResponse(message='取消下载成功')
    except CeleryTimeoutError:
        res = AppResponse(err_code=1, message='取消下载超时')
    except Exception as e:
        logger.exception(e)
        res = AppResponse(err_code=1, message='取消下载发生异常')

    return jsonify(res.dict())
