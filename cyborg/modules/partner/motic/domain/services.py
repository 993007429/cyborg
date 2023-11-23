import logging
import os
from typing import Tuple, Any

from cyborg.modules.partner.motic.application.settings import MoticSettings

try:
    from cyborg.modules.partner.motic.libs.MGCore import MG_SetCredential, MG_GetSlideTileSVS, MG_GetTaskROI
    MG_SetCredential(MoticSettings.API_ACCESS_KEY, MoticSettings.API_ACCESS_SECRET)
except Exception as e:
    pass

logger = logging.getLogger(__name__)


class MoticDomainService(object):

    def get_task_rois(self, motic_task_id: str) -> list:
        rois = MG_GetTaskROI(motic_task_id)
        return rois or []

    def download_slide(self, motic_task_id, roi: Any, slide_path: str) -> Tuple[str, str, int]:
        file_name = f'{roi.slideId}.svs'
        file_path = f'{slide_path}/{file_name}'
        success = MG_GetSlideTileSVS(motic_task_id, roi, file_path)
        if not success:
            return '', '', 0

        file_size = os.path.getsize(file_path)
        return file_name, file_path, file_size
