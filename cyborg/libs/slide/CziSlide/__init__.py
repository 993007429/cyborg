import math
from typing import Optional, List

import cv2
import numpy as np

from PIL import Image
from ..SlideBase import SlideBase
from .pylibCZIrw import czi as pyczi
from .pylibCZIrw.czi import CziReader
from ctypes import *
from threading import *

lock = Lock()


class CziSlide(SlideBase):

    def __init__(self, filename):
        self.filename = filename
        # self.slide = CziReader(self.filename)
        self.slide = CziReader(self.filename)
        self.width = self.slide.total_bounding_rectangle.w
        self.height = self.slide.total_bounding_rectangle.h
        self.num_channels = self.slide.total_bounding_box['C'][1]
        self.slide.close()
        SlideBase.__init__(self)

    def read(
        self, location: Optional[List[int]] = None, size: Optional[List[Union[int, float], Union[int, float]]] = None,
        scale: float = 1.0, greyscale: bool = False
    ):
        if size is None:
            width, height = self.width, self.height
        else:
            width, height = size

        crop_start_x, crop_start_y = location
        crop_start_x, crop_start_y = min(max(crop_start_x, 0), self.width), \
            min(max(crop_start_y, 0), self.height)

        crop_end_x = math.ceil(min(max(crop_start_x + width, 0), self.width))
        crop_end_y = math.ceil(min(max(crop_start_y + height, 0), self.height))

        crop_width = crop_end_x - crop_start_x
        crop_height = crop_end_y - crop_start_y

        roi = (crop_start_x, crop_start_y, crop_width, crop_height)
        zoom_level = 1.0 / scale
        with pyczi.open_czi(self.filename) as slide_f:
            crop_region = slide_f.read_color(pixel_type='Gray16', roi=roi, zoom=zoom_level)
        crop_region = cv2.cvtColor(crop_region, cv2.COLOR_RGB2BGR)

        return crop_region

    def save_label(self, path):
        with pyczi.open_czi(self.filename) as slide_f:
            attchmnt_names = slide_f._czi_reader.GetAttachmentNames()
            if 'Label' in attchmnt_names:
                label_idx = attchmnt_names.index('Label')
                label_image = np.array(slide_f._czi_reader.GetLabelImage(label_idx), copy=False)
                # label_image = cv2.cvtColor(label_image,cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, label_image)
            else:
                super().save_label(path)

    def get_tile(self, x, y, z):
        try:
            scale = math.pow(2, self.maxlvl - z)
            r = int(1024 * scale)
            with lock:
                tile = self.read([x * r, y * r], [r, r], scale)
            return Image.fromarray(tile, mode='RGB')
        except Exception as e:
            pass

    @property
    def mpp(self):
        mpp = None
        return mpp

    def get_max_value(self, slide, channel):
        max_value_dict = {
            'Gray8': 255,
            'Gray16': 65535,
            'Bgr24': 255,
            'Bgr48': 65535
        }
        return max_value_dict[slide.get_channel_pixel_type(channel)]
