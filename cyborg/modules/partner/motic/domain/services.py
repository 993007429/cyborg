import logging
import os
from typing import Tuple, Any

from cyborg.modules.partner.motic.application.settings import MoticSettings

try:
    from cyborg.modules.partner.motic.libs.MGCore import (
        MG_SetCredential, MG_GetSlideTileSVS, MG_GetTaskROI, MG_SetTaskStatus, MG_TaskStatus_PROCESSING,
        MG_TaskStatus_STARTED, MG_TaskStatus_FINISHED, MG_TaskStatus_FAILED, MG_TaskStatus_ABORTED)

    MG_SetCredential(MoticSettings.API_ACCESS_KEY, MoticSettings.API_ACCESS_SECRET)
except ImportError:
    pass

logger = logging.getLogger(__name__)


class MoticDomainService(object):

    def get_task_rois(self, motic_task_id: str) -> list:
        rois = MG_GetTaskROI(motic_task_id)
        return rois or []

    def download_slide(self, motic_task_id: str, roi: Any, slide_path: str) -> Tuple[str, str, int]:
        file_name = f'{roi.slideId}.svs'
        file_path = f'{slide_path}/{file_name}'
        success = MG_GetSlideTileSVS(motic_task_id, roi, file_path)
        if not success:
            return '', '', 0

        file_size = os.path.getsize(file_path)
        return file_name, file_path, file_size

    def callback_analysis_status(self, motic_task_id: str, ai_status: int):
        motic_task_status = {
            0: MG_TaskStatus_STARTED,
            1: MG_TaskStatus_PROCESSING,
            2: MG_TaskStatus_FINISHED,
            3: MG_TaskStatus_FAILED,
            4: MG_TaskStatus_ABORTED
        }.get(ai_status)
        logger.info(f'>>>>>>>>>{motic_task_id} >>>>> {ai_status} >>>>> {motic_task_status}')
        MG_SetTaskStatus(motic_task_id, motic_task_status)
