import base64
import datetime
import logging
import os
import random
import string
import sys
from io import BytesIO
from typing import Optional, List, Tuple, Union

import xlsxwriter
from PIL import Image
from werkzeug.datastructures import FileStorage

from cyborg.app.request_context import request_context
from cyborg.app.settings import Settings
from cyborg.infra.fs import fs
from cyborg.libs.slide.dispatch import open_slide

from cyborg.modules.slice.domain.services import SliceDomainService
from cyborg.modules.slice.domain.value_objects import SliceStartedStatus

from cyborg.modules.user_center.user_core.application.services import UserCoreService
from cyborg.seedwork.application.responses import AppResponse
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
        return AppResponse(data=record.to_dict())

    def get_all_values_in_fields(self) -> AppResponse[dict]:
        company_id = request_context.current_company
        return AppResponse(data={
            'sampleType': self.domain_service.repository.get_all_sample_types(company_id=company_id),
            'samplePart': self.domain_service.repository.get_all_sample_parts(company_id=company_id),
            'folders': self.domain_service.repository.get_all_user_folders(company_id=company_id),
            'operators': self.domain_service.repository.get_all_operators(company_id=company_id)
        })

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
                return AppResponse(message=err_msg)
        except Exception as e:
            logger.error(f'write records to excel fail: {e}')
            return AppResponse(message='写excel文件发生错误')
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

        if not self.domain_service.update_record(
                case_id=case_id,
                user_name=request_context.current_user.username,
                company_id=company_id,
                attachments=attachments,
                **kwargs
        ):
            return AppResponse(message='更新失败')
        return AppResponse(data='更新成功')

    def delete_records(self, case_ids: List[str]) -> AppResponse[str]:
        freed_size = self.domain_service.delete_records(case_ids, request_context.current_company)
        if freed_size > 0:
            self.user_service.update_company_storage_usage(
                company_id=request_context.current_company,
                increased_size=- freed_size)
        else:
            return AppResponse(message='删除失败')

        # inform_data = {
        #     'company': request_context.current_company,
        #     'type': 'inform',
        #     'message': 'delete record',
        # }
        # requests.post(
        #     url='http://127.0.0.1:{}/ws/inform'.format(Settings.PORT + 1),
        #     data=inform_data)
        return AppResponse(data='删除成功')

    def import_records(self) -> AppResponse[dict]:
        result = self.domain_service.import_records(company_id=request_context.current_company)
        return AppResponse(data=result)

    def _check_space_usage(self) -> Tuple[int, str]:
        free_space = fs.get_free_space(Settings.DATA_DIR)
        if free_space < 10:  # 校验物 理磁盘剩余空间（留10个g兜底）
            return 5, f'剩余磁盘空间不足,剩余{free_space}G'

        company_info = self.user_service.get_company_info(company_id=request_context.current_company).data
        volume = company_info['volume']
        usage = company_info['usage']
        if volume and usage and float(usage) >= float(volume):
            return 5, f'volume: {volume}, usage: {usage}'

        return 0, ''

    def update_clarity(self, slice_id: int, slice_file_path: str) -> AppResponse[str]:
        self.domain_service.update_clarity(slice_id, slice_file_path)
        return AppResponse(data='更新成功')

    def get_slices(self, case_ids: List[int], page: int = 0, per_page: int = 20) -> AppResponse:
        slices = self.domain_service.repository.get_slices(
            case_ids=case_ids, company=request_context.current_company, page=page, per_page=per_page)
        return AppResponse(data=[s.to_dict() for s in slices])

    def upload_slice(
            self, case_id: str, file_id: str, company_id: str, file_name: str, slide_type: str, upload_type: str,
            upload_path: str, total_upload_size: int, tool_type: str, user_file_path: str, cover_slice_number: bool,
            high_through: bool, operator: str
    ) -> AppResponse[dict]:

        if high_through:
            # 上传文件重复性校验
            slice = self.domain_service.repository.get_slice_by_local_filename(
                user_file_path=user_file_path, file_name=file_name, company=request_context.current_company)
            if slice:
                return AppResponse(err_code=2, message='2')

            # 上传切片文件完整性校验
            if Settings.INTEGRITY_CHECK:
                uploaded_size = fs.get_dir_size(upload_path)
                if uploaded_size < int(total_upload_size):
                    return AppResponse(err_code=1, message='1')
            # 切片文件损坏校验
                is_valid = self.domain_service.validate_slice_file(upload_path, file_name)
                if not is_valid:
                    return AppResponse(err_code=3, message='3')

        company = self.user_service.get_company_info(company_id).data

        if slide_type == 'slices':
            file_list = os.listdir(upload_path)
            for local_file_name in file_list:
                slice_info = self.domain_service.create_slice(
                    case_id=case_id, file_id=file_id, company_id=company_id, slide_path=upload_path, file_name=file_name,
                    local_file_name=local_file_name, upload_type=upload_type, tool_type=tool_type,
                    label_rec_mode=company['label'], user_file_path=user_file_path, cover_slice_number=cover_slice_number,
                    operator=operator, high_through=high_through
                )
                return AppResponse(data=slice_info)
        else:
            slide_save_name = os.path.join(upload_path, file_name)
            slide_size = fs.get_file_size(slide_save_name) / 1024 ** 3
            self.user_service.update_company_storage_usage(
                company_id=company_id,
                increased_size=slide_size
            )
            return AppResponse(data={'path': os.path.join(Settings.DATA_DIR, company_id, 'data')})

        return AppResponse(err_code=6, message='上传发生异常')

    def update_slice_info(
            self, case_id: str, file_id: str, high_through: bool, info: dict) -> AppResponse:
        updated = self.domain_service.update_slice_info(
            case_id=case_id, file_id=file_id, high_through=high_through, info=info)
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
        slice_entity = self.domain_service.update_ai_status(
            case_id=request_context.case_id, file_id=request_context.file_id,
            company_id=request_context.current_company,
            status=status,
            ai_name=ai_name,
            upload_batch_number=upload_batch_number,
            template_id=template_id,
            ip_address=ip_address
        )
        return AppResponse(
            err_code=0 if slice_entity else 1,
            message='update succeed' if slice_entity else 'update failed',
            data=slice_entity.to_dict() if slice_entity else None
        )

    def get_slice_info(self, case_id: str, file_id: str) -> AppResponse[dict]:
        err_code, message, entity = self.domain_service.get_slice(
            case_id=case_id, file_id=file_id, company_id=request_context.current_company)
        return AppResponse(err_code=err_code, message=message, data=entity.to_dict(all_fields=True) if entity else None)

    def get_slice_file_info(self, case_id: str, file_id: str) -> AppResponse[dict]:
        info = self.domain_service.get_slice_file_info(
            case_id=case_id, file_id=file_id, company_id=request_context.current_company)
        return AppResponse(data=info)

    def get_slice_data_paths(self, case_id: str, file_id: str) -> AppResponse[dict]:
        err_code, message, entity = self.domain_service.get_slice(
            case_id=case_id, file_id=file_id, company_id=request_context.current_company)
        return AppResponse(err_code=err_code, message=message, data=entity.data_paths if entity else None)

    def get_slice_files(self, case_ids: List[str], path: str, need_db_file: bool = False) -> AppResponse[list]:
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
        return AppResponse(data=file_path_list)

    def get_slice_angle(self, file_id: str):
        err_code, message, entity = self.domain_service.get_slice(
            file_id=file_id, company_id=request_context.current_company)
        if err_code:
            return AppResponse(err_code=err_code, message=message)

        assert entity is not None
        data = {'aiAngle': entity.ai_angle, 'currentAngle': entity.current_angle}
        return AppResponse(data=data)

    def update_slice_angle(self, file_id: str, current_angle: float) -> AppResponse:
        err_code, message, entity = self.domain_service.get_slice(
            file_id=file_id, company_id=request_context.current_company)
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
                logger.error(e)
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

        freed_size = self.domain_service.delete_slice(slice)
        if freed_size > 0:
            self.user_service.update_company_storage_usage(
                company_id=request_context.current_company,
                increased_size=freed_size
            )
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
        except AttributeError:
            return AppResponse(err_code=2, message="移动失败，找不到切片文件的扫描坐标信息。")
        except Exception:
            return AppResponse(err_code=1, message="移动失败，找不到切片文件的扫描坐标信息。")

    def get_analyzed_slices(self) -> AppResponse:
        slices = self.domain_service.repository.get_slices(
            ai_type=request_context.ai_type, started=SliceStartedStatus.success,
            company=request_context.current_company)
        return AppResponse(data=[s.to_dict() for s in slices])
