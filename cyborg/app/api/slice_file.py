import ast
import json
import logging
import mimetypes
import os
from threading import Thread

import requests
from flask import request, jsonify, make_response, send_from_directory, send_file
from werkzeug import formparser

from cyborg.app.api import api_blueprint
from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory
from cyborg.app.settings import Settings
from cyborg.infra.fs import fs
from cyborg.seedwork.application.responses import AppResponse
from cyborg.utils.thread import stop_thread

logger = logging.getLogger(__name__)

download_thread_dict = dict()


@api_blueprint.route('/files/upload2', methods=['get', 'post'])
def upload_slice_by_dir():
    slide_type = request.args.get('type') + 's'
    caseid = request.args.get('caseid')
    fileid = request.args.get('fileid')
    filename = os.sep.join(request.args.get('filename').split('\\'))

    slide_save_path = os.path.join(request_context.current_user.data_dir, 'data', caseid, slide_type, fileid, filename)
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
    file_id = request.args.get('fileid')  # 文件id，一个切片文件对应一个id
    cover_slice_number = request.args.get('cover_slice_number')  # 是否用病例号覆盖样本号，覆盖是true，不覆盖为false
    user_file_path = request.args.get('userFilePath')  # 上传端切片所在文件夹名称
    file_name = request.args.get('filename')  # 切片文件名
    upload_type = request.args.get('other')  # 上传模式，若为众包模式下上传此值为'cs'
    tool_type = request.args.get('toolType')
    total_upload_size = request.args.get('total')
    high_through = request.args.get('high_through')

    upload_path = fs.path_join(request_context.current_user.data_dir, 'data', case_id, slide_type, file_id)
    if not os.path.exists(upload_path):  # 切片文件不存在，即新上传切片文件
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


@api_blueprint.route('/files/saveInfo', methods=['get', 'post'])
def update_slice_info():
    case_id = request.form.get('caseid')
    file_id = request.form.get('fileid')
    high_through = bool(request.form.get('high_through'))  # 是否是高通量上传
    content = request.form.get('content')
    content = json.loads(content)
    res = AppServiceFactory.slice_service.update_slice_info(
        case_id=case_id, file_id=file_id, high_through=high_through, info=content)
    return jsonify(res.dict())


@api_blueprint.route('/files/getInfo', methods=['get', 'post'])
def get_slice_info():
    case_id = request.form.get('caseid')
    file_id = request.form.get('fileid')
    res = AppServiceFactory.slice_service.get_slice_info(case_id=case_id, file_id=file_id)

    res.data['group'] = AppServiceFactory.slice_analysis_service.get_selected_mark_group().data

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


@api_blueprint.route('/files/getImage', methods=['get', 'post'])
def get_image():
    caseid = request.args.get('caseid')
    fileid = request.args.get('fileid')
    img_type = request.args.get('type', type=str)
    if img_type == 'histplot':
        img_path = fs.path_join(
            request_context.current_user.data_dir, 'data', caseid, 'slices', fileid, 'dna_index.png')
    elif img_type == 'scatterplot':
        img_path = fs.path_join(
            request_context.current_user.data_dir, 'data', caseid, 'slices', fileid, 'scatterplot.png')
    else:
        img_path = ''
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
    file_id = request.form.get('fileid')
    current_angle = request.form.get('currentAngle')
    res = AppServiceFactory.slice_service.update_slice_angle(file_id=file_id, current_angle=current_angle)
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


@api_blueprint.route('/download', methods=['get', 'post'])  # 前端起服务，后端上传文件
def download():
    need_db_file = bool(int(request.form.get('want_db')))  # 是否需要导出db文件，需要导出为1反之为0
    case_ids = json.loads(request.form.get('caseid'))  # 需要导出文件的病例id列表
    path = request.form.get('path')
    ip = request.form.get('ip')
    url = 'http://{}:3000/download'.format(ip)
    key = request.form.get('key')

    def send_files(file_path_list, url, company, key):
        """
        发送文件到前端
        :param file_path_list: 待发送的文件路径及请求参数列表，格式为
        [[文件1路径, 文件1params请求参数字典, 是否为切片文件标志位（1：切片, 0: db文件）],
         [文件2路径, 文件2params请求参数字典, 标志位], ...]
        :param url: 请求url
        :param company: 组织
        :param key: 唯一值，用于区分下载任务，取消下载任务使用
        """
        message = '导出完成'
        from cyborg.app.init import app
        with app.app_context():
            for i in file_path_list:
                file = {'file': open(i[0], 'rb')}
                try:
                    res = requests.request("POST", url=url, files=file, timeout=600, params=i[1])
                    if res.text == '保存成功':
                        if i[2] == 1:
                            logger.info('caseid为<{}>下名为<{}>切片文件下载成功'.format(i[-2], i[-1]))
                        else:
                            logger.info('caseid为<{}>下名为<{}>切片数据库文件下载成功'.format(i[-2], i[-1]))
                    elif res.text == '磁盘空间不足':
                        if i[2] == 1:
                            logger.info('caseid为<{}>下名为<{}>切片文件下载失败'.format(i[-2], i[-1]))
                        else:
                            logger.info('caseid为<{}>下名为<{}>切片数据库文件下载失败'.format(i[-2], i[-1]))
                        message = '磁盘空间不足'
                        break
                except requests.exceptions.ConnectionError:
                    message = '导出失败。请检查本机ip地址是否正常。'
                    break
        inform_data = {
            'company': company,
            'type': 'download',
            'message': message,
        }
        requests.post(
            url='http://127.0.0.1:{}/ws/inform'.format(app.config.get('PORT') + 1),
            data=inform_data)
        download_thread_dict.pop(key)

    file_path_list = AppServiceFactory.slice_service.get_slice_files(
        case_ids=case_ids, path=path, need_db_file=need_db_file).data
    t = Thread(target=send_files, args=(file_path_list, url, request_context.current_company, key))
    download_thread_dict[key] = t
    t.start()

    res = AppResponse(message='下载请求发送成功')
    return jsonify(res.dict())


@api_blueprint.route('/cancelDownload', methods=['get', 'post'])  # 取消文件下载
def cancel_download():
    res = AppResponse()
    try:
        key = request.form.get('key')
        t = download_thread_dict.get(key)
        stop_thread(t)
        download_thread_dict.pop(key)
        res.message = '取消下载任务成功'
    except Exception as e:
        logger.error(e)
        res.message = '取消下载任务失败'
    finally:
        return jsonify(res.dict())
