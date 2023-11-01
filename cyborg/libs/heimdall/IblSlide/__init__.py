# -*- coding:utf-8 -*-

import os, math, sys, io
from ..SlideBase import SlideBase
from PIL import Image
from .ibl_py_sdk import IblWsi
from io import BytesIO
import numpy as np


class IblSlide(SlideBase):

    def __init__(self, filename):
        self.filename = filename
        self.slide = IblWsi(self.filename.encode())
        self.width = self.slide.width.value
        self.height = self.slide.height.value
        self.scanScale = self.slide.scanScale.value
        try:
            SlideBase.__init__(self)
        except:
            pass

    @property
    def mpp(self):
        if self.scanScale == 40:
            mpp = 0.25
        elif self.scanScale == 20:
            mpp = 0.5
        else:
            mpp = None
        return mpp

    def read(self, location=[0, 0], size=None, scale=1, greyscale=False):
        try:
            if size == None:
                width, height = self.width, self.height
            else:
                width, height = size

            if scale > 256:
                crop_region = np.full((1, 1, 3), 255, np.uint8)
                return crop_region

            zoom_rate = 1 / scale
            crop_start_x, crop_start_y = location
            crop_start_x, crop_start_y = min(max(crop_start_x, 0), self.width), min(max(crop_start_y, 0), self.height)
            crop_end_x = math.ceil(min(max(width + crop_start_x, 0), self.width))
            crop_end_y = math.ceil(min(max(height + crop_start_y, 0), self.height))

            crop_width = math.ceil((crop_end_x - crop_start_x) / scale)
            crop_height = math.ceil((crop_end_y - crop_start_y) / scale)

            crop_start_x = math.ceil(crop_start_x / scale)
            crop_start_y = math.ceil(crop_start_y / scale)

            crop_region_data = self.slide.GetRoiData(self.scanScale * zoom_rate, crop_start_x, crop_start_y, crop_width,
                                                     crop_height)
            crop_region_io = BytesIO(crop_region_data)
            crop_region = Image.open(crop_region_io)

            if greyscale:
                crop_region = crop_region.convert('L')
            return np.array(crop_region)
        except Exception as e:
            print(e)

    def save_label(self, path):
        label = self.slide.GetLabelData()
        if label.__len__() > 0:
            with open(path, 'wb') as f:
                f.write(label)
        else:
            pass

    def get_thumbnail(self, size=500):
        scale = size / self.height
        macro_width = int(scale * self.width)
        macro_io = BytesIO(self.slide.GetRoiData(self.scanScale * scale, 0, 0, macro_width, size))
        return Image.open(macro_io)

    def get_tile(self, x, y, z):
        scale = math.pow(2, self.maxlvl - z)
        size = int(1024 * scale)
        tile = self.read([x * size, y * size], [size, size], scale, greyscale=False)
        return Image.fromarray(tile, 'RGB')

    def __del__(self):
        self.slide.CloseIBL()
