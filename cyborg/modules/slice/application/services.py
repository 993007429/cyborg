import base64
import datetime
import json
import logging
import os
import random
import string
from io import BytesIO
from typing import Optional, List, Tuple, Union

import aiohttp
import requests
import xlsxwriter
from PIL import Image
from werkzeug.datastructures import FileStorage

from cyborg.app.request_context import request_context
from cyborg.app.settings import Settings
from cyborg.consts.common import Consts
from cyborg.infra.cache import cache
from cyborg.infra.fs import fs
from cyborg.libs.heimdall.dispatch import open_slide

from cyborg.modules.slice.domain.services import SliceDomainService
from cyborg.modules.slice.domain.value_objects import SliceStartedStatus

from cyborg.modules.user_center.user_core.application.services import UserCoreService
from cyborg.seedwork.application.responses import AppResponse
from cyborg.seedwork.domain.value_objects import AIType
from cyborg.utils.pagination import Pagination


logger = logging.getLogger(__name__)


class SliceService(object):

    def __init__(self, domain_service: SliceDomainService, user_service: UserCoreService):
        super(SliceService, self).__init__()
        self.domain_service = domain_service
        self.user_service = user_service

    def get_record_by_case_id(self, case_id: str) -> AppResponse[dict]:
        record = self.domain_service.repository.get_record_by_case_id(case_id, request_context.current_company)
        if not record:
            return AppResponse(err_code=2, message='no such case')

        record.slices = self.domain_service.repository.get_slices_by_case_id(case_id=case_id, company=request_context.current_company)
        return AppResponse(data=record.to_dict())

    def get_all_values_in_fields(self) -> AppResponse[dict]:
        company_id = request_context.current_company
        return AppResponse(data={
            'sampleType': self.domain_service.repository.get_all_sample_types(company_id=company_id),
            'samplePart': self.domain_service.repository.get_all_sample_parts(company_id=company_id),
            'folders': self.domain_service.repository.get_all_user_folders(company_id=company_id),
            'operators': self.domain_service.repository.get_all_operators(company_id=company_id)
        })

    def get_display_columns(self) -> AppResponse:
        all_columns = self.domain_service.get_all_record_columns()
        checked_columns = self.user_service.get_customized_record_fields().data
        disabled_columns = self.domain_service.get_disabled_record_columns()

        data = [{
            'name': column,
            'checked': column in checked_columns,
            'disabled': column in disabled_columns
        } for column in all_columns]
        return AppResponse(data=data)

    def search_records(
            self,
            **kwargs
    ) -> AppResponse[dict]:
        total, records = self.domain_service.repository.search_records(request_context.current_company, **kwargs)
        data = [record.to_dict() for record in records]
        return AppResponse(data=Pagination(locals()).to_dict())

    def export_records(self, **kwargs) -> AppResponse[str]:
        total, records = self.domain_service.repository.search_records(request_context.current_company, **kwargs)

        headers = self.user_service.get_customized_record_fields().data + ['最终结果']
        if '报告' in headers:
            headers.remove('报告')

        file_path = os.path.join(request_context.current_user.data_dir, 'znbl.xls')
        workbook = xlsxwriter.Workbook(file_path)
        try:
            err_msg = self.domain_service.write_records_to_excel(records, headers, workbook)
            if err_msg:
                return AppResponse(err_code=1, message=err_msg)
        except Exception as e:
            logger.exception(f'write records to excel fail: {e}')
            return AppResponse(err_code=1, message='写excel文件发生错误')
        finally:
            workbook.close()
        return AppResponse(data=file_path)

    def update_sample_num(self, case_id: str, sample_num: str) -> AppResponse[str]:
        record = self.domain_service.repository.get_record_by_case_id(
            case_id=case_id, company=request_context.current_company)
        record.update_data(sample_num=sample_num)
        if not self.domain_service.repository.save(record):
            return AppResponse(message='更新失败')
        return AppResponse(data='更新成功')

    def update_record(
            self, case_id: str, company_id: str,
            company_report_name: Optional[str] = None,
            company_report_info: Optional[str] = None,
            logo: Optional[str] = None,
            attachments: Optional[List[dict]] = None,
            **kwargs
    ) -> AppResponse[str]:

        if company_report_name is not None or company_report_info is not None:
            self.user_service.update_report_settings(report_name=company_report_name, report_info=company_report_info)

        if logo is not None:
            with open(os.path.join(request_context.current_user.data_dir, 'logo.jpg'), 'wb') as f:
                f.write(base64.b64decode(logo.split(',')[1]))

        record = self.domain_service.save_record(
            case_id=case_id,
            user_name=request_context.current_user.username,
            company_id=company_id,
            attachments=attachments,
            **kwargs
        )
        if not record:
            return AppResponse(message='保存失败')
        return AppResponse(message='保存成功', data=record.to_dict())

    def delete_records(self, case_ids: List[str]) -> AppResponse[str]:
        freed_size = self.domain_service.delete_records(case_ids, request_context.current_company) or 0
        if freed_size <= 0:
            return AppResponse(message='删除失败')

        return AppResponse(data='删除成功')

    def import_records(self) -> AppResponse[dict]:
        result = self.domain_service.import_records(company_id=request_context.current_company)
        return AppResponse(data=result)

    def _check_space_usage(self) -> Tuple[int, str]:
        if Settings.CLOUD:
            company_info = self.user_service.get_company_info(company_id=request_context.current_company).data
            volume = company_info['volume']
            storage_info = self.user_service.get_company_storage_info().data
            usage = storage_info['usage']
            if volume and usage and float(usage) >= float(volume):
                return 5, f'volume: {volume}, usage: {usage}'
        else:
            free_space = fs.get_free_space(Settings.DATA_DIR)
            if free_space < 10:  # 校验物 理磁盘剩余空间（留10个g兜底）
                return 5, f'剩余磁盘空间不足,剩余{free_space}G'

        return 0, ''

    def update_clarity(self, slice_id: int, slice_file_path: str) -> AppResponse[str]:
        self.domain_service.update_clarity(slice_id, slice_file_path)
        return AppResponse(data='更新成功')

    def get_slices(
            self, case_ids: List[int], page: int = 0, per_page: int = 20
    ) -> AppResponse:
        slices = self.domain_service.repository.get_slices(
            case_ids=case_ids, slice_type='slice', company=request_context.current_company,
            page=page, per_page=per_page
        )
        return AppResponse(data=[s.to_dict() for s in slices])

    def upload_slice(
            self, upload_id: str, case_id: str, file_id: str, company_id: str, file_name: str, slide_type: str,
            upload_path: str, total_upload_size: int, tool_type: str, user_file_path: str, cover_slice_number: bool,
            high_through: bool, upload_batch_number: str, operator: str
    ) -> AppResponse[dict]:

        err_code, message = self._check_space_usage()
        if err_code:
            return AppResponse(err_code=err_code, message=message)

        if high_through:
            case_id = cache.get(f'uploadid:{upload_id}:caseid')
            if not case_id:
                case_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_' + str(random.randint(0, 1000000))
                cache.set(f'uploadid:{upload_id}:caseid', case_id, ex=3600 * 24 * 3)

            # 上传文件重复性校验
            slice = self.domain_service.repository.get_slice_by_local_filename(
                user_file_path=user_file_path, file_name=file_name, company=request_context.current_company)
            if slice:
                return AppResponse(err_code=2, message='2')

            # 上传切片文件完整性校验
            if Settings.INTEGRITY_CHECK:
                uploaded_size = fs.get_dir_size(upload_path)
                if uploaded_size < total_upload_size:
                    return AppResponse(err_code=1, message='1')

            is_valid = self.domain_service.create_slice_link_file(upload_path, file_name, check_slide=True)
            if not is_valid:
                return AppResponse(err_code=3, message='3')

        company = self.user_service.get_company_info(company_id).data

        if slide_type == 'slices':
            file_list = os.listdir(upload_path)
            for local_file_name in file_list:
                slice_info = self.domain_service.create_slice(
                    case_id=case_id, file_id=file_id, company_id=company_id, upload_path=upload_path,
                    file_name=file_name, local_file_name=local_file_name, slide_type=slide_type, tool_type=tool_type,
                    label_rec_mode=company['label'], user_file_path=user_file_path,
                    cover_slice_number=cover_slice_number, operator=operator, upload_batch_number=upload_batch_number,
                    sample_num=upload_id if high_through else case_id, high_through=high_through,
                )
                if slice_info:
                    return AppResponse(data=slice_info)
            return AppResponse(err_code=3, message='3')
        else:
            return AppResponse(data={'path': os.path.join(Settings.DATA_DIR, company_id, 'data')})

    def update_slice_info(
            self, case_id: str, file_id: str, high_through: bool, info: dict) -> AppResponse:
        updated = self.domain_service.update_slice_info(
            case_id=case_id, file_id=file_id, high_through=high_through, info=info)
        return AppResponse(data=updated)

    def update_slice_ai_suggest(self, abnormal_high: int, abnormal_low: int, total: int) -> AppResponse:
        updated = self.domain_service.update_slice_ai_suggest(abnormal_high=abnormal_high,
                                                              abnormal_low=abnormal_low,
                                                              total=total)
        return AppResponse(data=updated)

    def finish_ai(
            self, status: SliceStartedStatus, ai_suggest: Optional[str] = None, slide_quality: Optional[int] = None,
            cell_num: Optional[int] = None, as_id: Optional[int] = None
    ) -> AppResponse:
        err_code, message, slice_entity = self.domain_service.get_slice(
            case_id=request_context.case_id, file_id=request_context.file_id, company_id=request_context.current_company)
        if err_code:
            return AppResponse(err_code=err_code, message=message)

        slice_entity.update_data(started=status)
        if ai_suggest is not None:
            slice_entity.update_data(ai_suggest=ai_suggest.strip())
        if slide_quality is not None:
            slice_entity.update_data(slide_quality=slide_quality)
        if cell_num is not None:
            slice_entity.update_data(cellNum=cell_num)
        if as_id is not None:
            slice_entity.update_data(ai_id=as_id)
        slice_entity.update_data(update_time=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))

        if status == SliceStartedStatus.success:
            slice_entity.update_data(ai_status=1)

        if not self.domain_service.repository.save_slice(slice_entity):
            return AppResponse(err_code=1, message='update ai result failed')
        return AppResponse()

    def update_mark_config(self, radius: Optional[float] = None, is_solid: Optional[int] = None) -> AppResponse:
        updated = self.domain_service.update_mark_config(
            case_id=request_context.case_id, file_id=request_context.file_id, radius=radius, is_solid=is_solid)
        return AppResponse(message='update succeed' if updated else 'update failed')

    def update_template_id(self, template_id: int) -> AppResponse:
        err_code, message = self.domain_service.update_template_id(
            case_id=request_context.case_id, file_id=request_context.file_id, template_id=template_id)
        return AppResponse(err_code=err_code, message=message)

    def reset_ai_status(self) -> AppResponse[dict]:
        slice_entity = self.domain_service.reset_ai_status(
            case_id=request_context.case_id, file_id=request_context.file_id,
            company_id=request_context.current_company,
        )
        return AppResponse(
            message='reset succeed' if slice_entity else 'reset failed',
            data=slice_entity.to_dict() if slice_entity else None
        )

    def update_ai_status(
            self, status: SliceStartedStatus, ai_name: Optional[str] = None, upload_batch_number: Optional[str] = None,
            template_id: Optional[int] = None, ip_address: Optional[str] = None
    ) -> AppResponse[dict]:
        err_msg, slice_entity = self.domain_service.update_ai_status(
            case_id=request_context.case_id, file_id=request_context.file_id,
            company_id=request_context.current_company,
            status=status,
            ai_name=ai_name,
            upload_batch_number=upload_batch_number,
            template_id=template_id,
            ip_address=ip_address
        )
        return AppResponse(
            err_code=1 if err_msg else 0,
            message=err_msg,
            data=slice_entity.to_dict() if slice_entity else None
        )

    def get_slice_info(self, case_id: str, file_id: str, company_id: str = '') -> AppResponse[dict]:
        err_code, message, entity = self.domain_service.get_slice(
            case_id=case_id, file_id=file_id, company_id=company_id or request_context.current_company)
        if entity:
            entity.slide = open_slide(entity.slice_file_path)
        return AppResponse(err_code=err_code, message=message, data=entity.to_dict(all_fields=True) if entity else None)

    def get_slice_file_info(self, case_id: str, file_id: str) -> AppResponse[dict]:
        info = self.domain_service.get_slice_file_info(
            case_id=case_id, file_id=file_id, company_id=request_context.current_company)
        return AppResponse(data=info)

    def get_slice_data_paths(self, case_id: str, file_id: str) -> AppResponse[dict]:
        err_code, message, entity = self.domain_service.get_slice(
            case_id=case_id, file_id=file_id, company_id=request_context.current_company)
        return AppResponse(err_code=err_code, message=message, data=entity.data_paths if entity else None)

    def export_slice_files(self, client_ip: str, case_ids: List[str], path: str, need_db_file: bool = False) -> AppResponse[list]:
        url = Settings.ELECTRON_UPLOAD_SERVER.format(client_ip)
        file_path_list = list()  # 待发送的文件路径列表
        for case_id in case_ids:
            slices = self.domain_service.repository.get_slices_by_case_id(
                case_id=case_id, company=request_context.current_company)
            if len(slices) != 1:
                logger.info('caseid为<{}>下包含多张切片，不进行下载！'.format(case_id))
            else:
                slice = slices[0]
                slice_file_path = fs.path_join(slice.slice_dir, slice.filename.split('.')[0])  # 可能存在的多切片文件夹存放路径
                if os.path.exists(slice_file_path) and not slice.filename.endswith('.svs'):
                    logger.info('caseid为<{}>下文件名为<{}>的切片为多文件切片，不进行下载！'.format(case_id, slice.filename))
                else:
                    params = {
                        'path': path,
                        'caseid': case_id,
                    }
                    if need_db_file:  # 下载切片db文件
                        file_path_list.append([slice.db_file_path, params, 0, case_id, slice.filename])
                    file_path_list.append([slice.slice_file_path, params, 1, case_id, slice.filename])

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
                    return AppResponse(err_code=1, message='磁盘空间不足')
            except requests.exceptions.ConnectionError:
                return AppResponse(err_code=1, message='导出失败。请检查本机ip地址是否正常。')

        return AppResponse(message='导出完成')

    def get_slice_angle(self, file_id: str):
        err_code, message, entity = self.domain_service.get_slice(
            file_id=file_id, company_id=request_context.current_company)
        if err_code:
            return AppResponse(err_code=err_code, message=message)

        assert entity is not None
        data = {'aiAngle': entity.ai_angle, 'currentAngle': entity.current_angle}
        return AppResponse(data=data)

    def update_slice_angle(self, case_id: str, file_id: str, current_angle: float) -> AppResponse:
        err_code, message, entity = self.domain_service.get_slice(
            case_id=case_id, file_id=file_id, company_id=request_context.current_company)
        if err_code:
            return AppResponse(err_code=err_code, message=message)

        entity.update_data(current_angle=current_angle)
        if not self.domain_service.repository.save_slice(entity):
            return AppResponse(err_code=1, message='update angle failed')
        return AppResponse()

    def get_slice_image(self, case_id: str, file_id: str, axis: Tuple[int, int, int]) -> AppResponse[dict]:
        slice = self.domain_service.repository.get_slice(
            case_id=case_id, file_id=file_id, company=request_context.current_company)
        if not slice:
            return AppResponse(err_code=1, message='slice does not exist')

        tile_path = slice.get_tile_path(*axis)

        if os.path.isfile(tile_path):
            return AppResponse(data={'tile_path': tile_path})
        else:
            if not fs.path_exists(slice.slice_file_path):
                return AppResponse(err_code=1, message='slice does not exist')
            try:
                slide = open_slide(slice.slice_file_path)
                tile_image = slide.get_tile(*axis)
                buf = BytesIO()
                tile_image.save(buf, 'jpeg')
                return AppResponse(data={'image_data': buf})
            except Exception as e:
                logger.exception(e)
                return AppResponse(err_code=2, message='image encounter error!')

    def get_slice_thumbnail(self, case_id: str, file_id: str) -> AppResponse[BytesIO]:
        slice = self.domain_service.repository.get_slice(
            case_id=case_id, file_id=file_id, company=request_context.current_company)
        if not (slice and os.path.isfile(slice.thumb_file_path)):
            return AppResponse(err_code=1, message='Thumbnail does not exist')
        thumbnail = Image.open(slice.thumb_file_path)
        buf = BytesIO()
        thumbnail.save(buf, 'jpeg')
        return AppResponse(data=buf)

    def get_attachment_file_path(self, case_id: str, file_id: str) -> AppResponse[str]:
        slice = self.domain_service.repository.get_slice(
            case_id=case_id, file_id=file_id, company=request_context.current_company)
        if not (slice and fs.path_exists(slice.attachment_file_path) and fs.path_isfile(slice.attachment_file_path)):
            return AppResponse(err_code=1, message='attachment does not exist')

        return AppResponse(data=slice.attachment_file_path)

    def get_roi(self, case_id: str, file_id: str, roi: list, roi_id: str) -> AppResponse[BytesIO]:
        slice = self.domain_service.repository.get_slice(
            case_id=case_id, file_id=file_id, company=request_context.current_company)
        if not (slice and fs.path_exists(slice.slice_file_path)):
            return AppResponse(err_code=1, message='slice does not exist')

        roi_file_path = slice.get_roi_path(roi_id=roi_id)

        roi_img = open_slide(slice.slice_file_path).get_roi(roi)
        os.makedirs(slice.roi_dir, exist_ok=True)
        roi_img.save(roi_file_path)
        buf = BytesIO()
        roi_img.save(buf, 'jpeg')

        return AppResponse(data=buf)

    def get_roi_and_segment(self, case_id: str, file_id: str, roi: list, dna_index: float) -> AppResponse[BytesIO]:
        slice = self.domain_service.repository.get_slice(
            case_id=case_id, file_id=file_id, company=request_context.current_company)
        if not (slice and fs.path_exists(slice.slice_file_path)):
            return AppResponse(err_code=1, message='slice does not exist')

        _, white_background = open_slide(slice.slice_file_path).get_roi_and_segment(roi, dna_index, None)
        os.makedirs(slice.roi_dir, exist_ok=True)
        buf = BytesIO()
        white_background.save(buf, 'jpeg')

        return AppResponse(data=buf)

    def get_screenshot(self, case_id: str, file_id: str, roi: list, roi_id: str) -> AppResponse[str]:
        slice = self.domain_service.repository.get_slice(
            case_id=case_id, file_id=file_id, company=request_context.current_company)

        if not (slice and fs.path_exists(slice.slice_file_path)):
            return AppResponse(err_code=1, message='slice does not exist')

        screenshot_file_path = slice.get_screenshot_path(roi_id=roi_id)

        if not fs.path_exists(screenshot_file_path):
            roi_img = open_slide(slice.slice_file_path).get_roi(roi, is_bounding_resize=False, standard_mpp=0.242042)
            os.makedirs(slice.screenshot_dir, exist_ok=True)
            roi_img.save(screenshot_file_path)

        return AppResponse(data=screenshot_file_path)

    def delete_slice(self, case_id: str, file_id: str) -> AppResponse:
        slice = self.domain_service.repository.get_slice(
            case_id=case_id, file_id=file_id, company=request_context.current_company)
        if not slice:
            return AppResponse(err_code=1, message='slice does not exist')

        self.domain_service.delete_slice(slice)
        return AppResponse(message='delete slice succeed')

    def delete_ai_image_files(self, case_id: str, file_id: str):
        entity = self.domain_service.repository.get_slice(
            case_id=case_id, file_id=file_id, company=request_context.current_company)
        if not entity:
            return AppResponse(err_code=1, message='slice does not exist')

        self.domain_service.remove_ai_image_files(entity)
        return AppResponse()

    def capture(self, capture: FileStorage):
        filename = ''.join(random.sample(string.ascii_letters + string.digits, 8)) + '.png'
        capture_path = fs.path_join(request_context.current_user.data_dir, filename)
        capture.save(capture_path)

        text = self.domain_service.recognize_label_text(capture_path, None, label_rec_mode=4)

        if not text:
            return AppResponse(data={'total': 0, 'data': [], 'text': ''})

        total, records = self.domain_service.repository.search_records(
            request_context.current_company, search_key='sample_num', search_value=text)

        data = [record.to_dict() for record in records]
        res_data = Pagination(locals()).to_dict()
        res_data['text'] = text
        return AppResponse(data=res_data)

    def stage_position(
            self, case_id: str, file_id: str, x_position: Union[str, float, int], y_position: Union[str, float, int]
    ) -> AppResponse:
        slice = self.domain_service.repository.get_slice(
            case_id=case_id, file_id=file_id, company=request_context.current_company)
        if not slice:
            return AppResponse(err_code=1, message='slice does not exist')
        slide = open_slide(slice.slice_file_path)
        try:
            x, y, z = slide.convert_pos(float(x_position), float(y_position))
            return AppResponse(data={'x': x, 'y': y, 'z': z})
        except AttributeError as e:
            logger.exception(e)
            return AppResponse(err_code=2, message="移动失败，找不到切片文件的扫描坐标信息。")
        except Exception as e:
            logger.exception(e)
            return AppResponse(err_code=1, message="移动失败，找不到切片文件的扫描坐标信息。")

    def get_analyzed_slices(self) -> AppResponse:
        slices = self.domain_service.repository.get_slices(
            ai_type=request_context.ai_type, started=SliceStartedStatus.success,
            company=request_context.current_company)
        return AppResponse(data=[s.to_dict() for s in slices])

    def get_report_opinion(self) -> AppResponse:
        record = self.domain_service.repository.get_record_by_case_id(
            case_id=request_context.case_id, company=request_context.current_company)
        slices = self.domain_service.repository.get_slices_by_case_id(
            case_id=request_context.case_id, company=request_context.current_company)
        record.slices = slices

        data = record.report_opinion
        return AppResponse(data=data)

    async def sync_report(self):
        """同步报表业务数据到报表中心
        :return:
        """
        record = self.domain_service.repository.get_record_by_case_id(
            case_id=request_context.case_id, company=request_context.current_company)
        if not record:
            return AppResponse(err_code=1, message='病例不存在')
        record.slices = self.domain_service.repository.get_slices_by_case_id(
            case_id=request_context.case_id, company=request_context.current_company)
        report_config = self.domain_service.get_report_config(company=request_context.current_company)
        report_templates = record.get_report_templates(
            template_config=report_config.template_config, file_id=request_context.file_id)
        alg_type = self.domain_service.get_alg_type(case_id=request_context.case_id, file_id=request_context.file_id,
                                                    company_id=request_context.current_company)

        report_data = record.to_dict_for_report()
        report_data['histplot'] = report_data['histplot'] + f'&company={request_context.current_company}'
        report_data['scatterplot'] = report_data['scatterplot'] + f'&company={request_context.current_company}'

        # TODO 设计上这里不应该调用slice_analysis模块的服务，勾选的roi应该保存在病例模块，后续需要优化这个逻辑
        from cyborg.app.service_factory import AppServiceFactory
        roi_data = AppServiceFactory.new_slice_analysis_service().get_report_roi().data
        roi_images = []
        for k in [AIType.human, AIType.tct, AIType.lct, AIType.dna]:
            roi_images.extend([roi.get('image_url', '') for roi in roi_data[k] if roi.get('iconType') != 'dnaIcon'])

        if alg_type == AIType.dna_ploidy:
            rois_list = roi_data[alg_type]
            sorted_rois = sorted(rois_list, key=lambda item: item['di'], reverse=True)
            sorted_rois_50 = sorted_rois[:50]
            dnaIcons = [{'id': roi['id'], 'di': roi['di'], 'image_url': roi['image_url']} for roi in sorted_rois_50]
        else:
            dnaIcons = [roi for roi in roi_data[k] if roi.get('iconType') == 'dnaIcon']

        report_data['roiImages'] = roi_images
        report_data.update(AppServiceFactory.new_slice_analysis_service().get_capture_images().data)
        report_data['dnaStatics'] = roi_data.get('dnaStatics')
        report_data['dnaIcons'] = dnaIcons
        report_data['signUrl'] = f'{Settings.IMAGE_SERVER}/user/sign2?id={request_context.current_user.username}&companyid={request_context.current_company}' # noqa
        report_data['reportTime'] = datetime.datetime.now().strftime('%Y-%m-%d')

        if alg_type == AIType.dna_ploidy:
            report_data['dna_statistics'] = AppServiceFactory.new_slice_analysis_service().get_dna_statistics().data
            report_data['dna_suggest'], report_data['dna_diagnose'] = AppServiceFactory.slice_service.get_dna_check_result().data
        elif alg_type == AIType.np:
            cell_statics, area_statics = AppServiceFactory.new_slice_analysis_service().get_np_info().data
            report_data['cellTypeTable'] = {
                'header': ['', '总数', '指数（%）'],
                'data': [[
                    {'color': item['stats']['color'], 'val': item['name']},
                    {'color': item['stats']['color'], 'val': item['stats']['count']},
                    {'color': item['stats']['color'], 'val': item['stats']['index']}] for item in cell_statics]
            }
            report_data['cellTypePie'] = self.domain_service.gen_pie_chart_data(
                [{'name': item['name'], 'value': item['stats']['count'], 'color': item['stats']['color']}
                 for item in cell_statics[:-1]])
            report_data['areaTypeTable'] = {
                'header': ['', '总面积(mm²)', '指数(%)'],
                'data': [[
                    {'color': item['stats']['color'], 'val': item['name']},
                    {'color': item['stats']['color'], 'val': item['stats']['area']},
                    {'color': item['stats']['color'], 'val': item['stats']['index']}] for item in area_statics]
            }
            report_data['areaTypePie'] = self.domain_service.gen_pie_chart_data(
                [{'name': item['name'], 'value': item['stats']['area'], 'color': item['stats']['color']}
                 for item in area_statics[:-1]])

        # 拼接report name
        template_name = '默认模板'
        for item in report_templates:
            if item.get('active') is True:
                template_name = item.get('templateName')
                break

        prefix = record.sample_num if record.sample_num else record.caseid
        report_name = prefix + '_' + template_name

        async with aiohttp.ClientSession() as client:
            params = {
                'bizType': Consts.REPORT_BIZ_TYPE,
                'bizUid': record.id,
                'reportName': report_name,
                'reportData': report_data
            }
            async with client.post(f'{Settings.REPORT_SERVER}/report/api/reports', json=params, verify_ssl=False) as resp:
                if resp.status == 200:
                    data = json.loads(await resp.read())
                    if data.get('code') == 0:
                        return AppResponse(message='发送成功', data={
                            'reportTemplates': report_templates,
                            'reportServer': Settings.REPORT_SERVER
                        })
                    else:
                        return AppResponse(err_code=1, message=data.get('message', '发送报告失败'))
                else:
                    return AppResponse(err_code=1, message='报告中心服务异常')

    async def save_report_snapshot(self, template_code: str):
        record = self.domain_service.repository.get_record_by_case_id(
            case_id=request_context.case_id, company=request_context.current_company)
        if not record:
            return AppResponse(err_code=1, message='病例不存在')

        async with aiohttp.ClientSession() as client:
            params = {
                'bizType': Consts.REPORT_BIZ_TYPE,
                'bizUid': record.id,
                'templateCode': template_code
            }
            async with client.post(
                    f'{Settings.REPORT_SERVER}/report/api/reports/snapshots', json=params, verify_ssl=False) as resp:
                if resp.status == 200:
                    data = json.loads(await resp.read())
                    if data.get('code') == 0:
                        snapshot_id = data.get('data', {}).get('id', None)
                        if snapshot_id:
                            self.domain_service.update_report_snapshot(record=record, snapshot_id=snapshot_id)
                        return AppResponse(message='发送成功')
                    else:
                        return AppResponse(err_code=1, message=data.get('message', '发送报告失败'))
                else:
                    return AppResponse(err_code=1, message='报告中心服务异常')

    def create_report(self, case_id: str, report_id: str, report_data: str, jwt: str) -> AppResponse:
        record = self.domain_service.repository.get_record_by_case_id(
            case_id=case_id, company=request_context.current_company)
        if not record:
            return AppResponse(err_code=1, message='病例不存在')

        report_dir = fs.path_join(record.data_dir, 'reports', report_id)
        os.makedirs(report_dir)
        pdf = fs.path_join(report_dir, 'index.pdf')

        with open(fs.path_join(report_dir, "report.json"), 'w',
                  encoding='UTF-8') as report:
            report.write(report_data)

        header_target = "http://%s/report.html?caseid=%s&reportid=%s&username=%s#/viewheader" % (
            Settings.REPORT_SERVER, record.caseid, report_id, request_context.current_user.username)
        index_target = "http://%s/report.html?caseid=%s&reportid=%s&username=%s" % (
            Settings.REPORT_SERVER, record.caseid, report_id, request_context.current_user.username)
        footer_target = "http://%s/report.html?caseid=%s&reportid=%s&username=%s#/viewfooter" % (
            Settings.REPORT_SERVER, record.caseid, report_id, request_context.current_user.username)
        try:
            _to_pdf_command = 'wkhtmltopdf'
            params = '--header-html "%s"  --footer-html "%s" --margin-left 0 --margin-top 25 --window-status 1 --cookie "jwt" "%s"' % (
                header_target, footer_target, jwt)
            command = '%s %s "%s" "%s"' % (_to_pdf_command, params, index_target, pdf)
            logger.info(command)
            os.system(command)
        except Exception as e:
            AppResponse(err_code=1, message="create pdf error: %s" % e)

        return AppResponse(message=report_id, data=report_id)

    def get_report_file(self, report_id: str) -> AppResponse:
        record = self.domain_service.repository.get_record_by_case_id(
            case_id=request_context.case_id, company=request_context.current_company)
        if not record:
            return AppResponse(err_code=1, message='病例不存在')

        return AppResponse(data=os.path.join(record.data_dir, 'reports', report_id, 'index.pdf'))

    def get_report_data(self, report_id: str) -> AppResponse:
        record = self.domain_service.repository.get_record_by_case_id(
            case_id=request_context.case_id, company=request_context.current_company)
        if not record:
            return AppResponse(err_code=1, message='病例不存在')

        report_path = os.path.join(record.data_dir, 'reports', report_id, 'report.json')
        if fs.path_exists(report_path):
            with open(report_path, 'r', encoding='UTF-8') as file:
                data = file.read()
                return AppResponse(data=data)
        else:
            return AppResponse(err_code=2, message='找不到报告数据')

    def apply_ai_threshold(self, threshold_range: int, threshold_value: float) -> bool:
        return self.domain_service.apply_ai_threshold(
            company=request_context.current_company, ai_type=request_context.ai_type,
            threshold_range=threshold_range, threshold_value=threshold_value
        )

    def get_record_count(self, end_time: str) -> AppResponse:
        records = self.domain_service.repository.get_records(end_time=end_time, company=request_context.current_company)
        return AppResponse(data={'recordCount': len(records)})

    def get_new_slices(
            self, start_id: int, updated_after: Optional[datetime.datetime] = None, upload_batch_number: Optional[str] = None
    ) -> AppResponse:
        last_id, increased, slices = self.domain_service.repository.get_new_slices(
            company=request_context.current_company, start_id=start_id, upload_batch_number=upload_batch_number)
        updated, updated_slices = self.domain_service.repository.get_new_updated_slices(
            company=request_context.current_company, updated_after=updated_after,
            upload_batch_number=upload_batch_number)
        last_modified = updated_slices[-1]['last_modified'] if updated_slices else updated_after
        data = {
            'lastId': last_id,
            'lastModified': last_modified.strftime('%Y-%m-%d %H:%M:%S.%f') if last_modified else None,
            'increased': increased,
            'updated': updated
        }
        if upload_batch_number:
            data.update({'newRecords': slices, 'updatedRecords': updated_slices})
            if slices or updated_slices:
                pending_count = self.domain_service.repository.get_pending_slices_count(
                    company=request_context.company,
                    upload_batch_number=upload_batch_number
                )
                data['pendingAnalysis'] = pending_count
        return AppResponse(data=data)

    async def free_up_space(self, end_time: str) -> AppResponse:
        success = await self.domain_service.free_up_space(end_time, request_context.current_company)
        if not success:
            return AppResponse(err_code=1, message='free up space failed')
        return AppResponse()

    def get_report_config(self, company: str) -> AppResponse:
        company = company or request_context.current_company
        company_info = self.user_service.get_company_info(company_id=company).data if company else None
        if not company_info:
            return AppResponse(err_code=1, message='找不到组织')
        config = self.domain_service.get_report_config(company=company)
        return AppResponse(data=config.to_dict_v2() if config else None)

    def save_report_config(self, company: str, template_config: List[dict]):
        company = company or request_context.current_company
        company_info = self.user_service.get_company_info(company_id=company).data if company else None
        if not company_info:
            return AppResponse(err_code=1, message='找不到组织')
        config = self.domain_service.get_report_config(company=company)
        if config:
            config.update_data(template_config=template_config)
            if self.domain_service.report_config_repository.save(config):
                return AppResponse(message='保存成功')
        return AppResponse(err_code=1, message='保存失败')

    def get_config(self) -> AppResponse:
        plugins = Settings.PLUGINS or []
        config = {
            'hasPIS': 'logene' in plugins,
            'syncOperations': Settings.SYNC_OPERATIONS or [],
            'autoSelectAI': 'logene' in plugins
        }

        return AppResponse(data=config)

    def get_dna_check_result(self) -> AppResponse:
        case_id = request_context.case_id
        file_id = request_context.file_id
        company = request_context.current_company

        data = self.domain_service.get_dna_check_result(case_id=case_id, file_id=file_id, company=company)

        return AppResponse(data=data)
