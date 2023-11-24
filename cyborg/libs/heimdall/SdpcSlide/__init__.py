#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
import os
import sys
import cv2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(base_dir)
from ..SlideBase import SlideBase
import numpy as np
from PIL import Image
from ctypes import *
from pystruct import *


cur_encoding='utf-8'
lib = cdll.LoadLibrary('libDecodeSdpc.so')


class SdpcSlide(SlideBase):

    def __init__(self, filename):

        self.filename = filename
        self.bfilename = self.filename.encode(cur_encoding)
        lib.SqOpenSdpc.restype = POINTER(SqSdpcInfo)
        self.slide = lib.SqOpenSdpc(c_char_p(self.bfilename))

        slide_contents = self.slide.contents
        self.width = slide_contents.picHead.contents.srcWidth
        self.height = slide_contents.picHead.contents.srcHeight
        self.scale = slide_contents.picHead.contents.scale
        self.correct = True
        if self.slide.contents.extra is not None and self.correct:
            lib.InitColorCollectTable.restype = POINTER(SqColorTable)
            self.colorTable = lib.InitColorCollectTable(self.slide)
        else:
            self.colorTable = None

        SlideBase.__init__(self)

    def read(self, location=[0, 0], size=None, scale=1, greyscale=False):
        '''
        :param location: (x, y) at level=0
        :param size: (width, height)
        :param scale: resize scale, scale>1 -> zoom out, scale<1 -> zoom in
        :param greyscale: if True, convert image to greyscale
        :return: a numpy image,  np_img.shape=[height, width, channel=1 or 3]
        '''
        if size == None:
            width, height = self.width, self.height
        else:
            width, height = size

        crop_start_x, crop_start_y = location
        if self.scale == 0.5:
            crop_level = math.floor(math.log2(scale))
        else: #for scale==0.25
            crop_level = math.floor(math.log(scale, 4))

        crop_level = max(0, min(crop_level, self.slide.contents.picHead.contents.hierarchy-1))
        level_ratio = 2 ** crop_level if self.scale == 0.5 else 4**crop_level
        resize_ratio = level_ratio / scale

        # make sure the crop region is inside the slide
        crop_start_x, crop_start_y = min(max(crop_start_x, 0), self.width-1), min(max(crop_start_y, 0), self.height-1)
        crop_end_x = math.floor(min(max(width + crop_start_x, 0), self.width-1))
        crop_end_y = math.floor(min(max(height + crop_start_y, 0), self.height-1))

        crop_width = math.floor((crop_end_x - crop_start_x) / level_ratio)
        crop_height = math.floor((crop_end_y - crop_start_y) / level_ratio)

        if crop_height == 0 or crop_width == 0:
            return None
        layer = c_int(crop_level)
        # rgb = pointer(pointer(c_ubyte()))
        rgb = pointer(c_ubyte())
        w = c_int(crop_width)
        h = c_int(crop_height)
        xp = c_uint(int(crop_start_x/level_ratio))
        yp = c_uint(int(crop_start_y/level_ratio))



        ret = lib.SqGetRoiRgbOfSpecifyLayer(self.slide, byref(rgb), w, h, xp, yp, layer)
        if ret < 0:  # 图像缺块
            if layer != c_int(0):
                img = self.read(location=location, size=size, scale=1)
                img = cv2.resize(img, dsize=None, fx=1 / scale, fy=1 / scale)
                return img
            raise ValueError(f'Image is damaged at ({xp},{yp}), ({w}, {h}), {layer}')

        if self.slide.contents.extra is not None and self.correct:

            lib.RgbColorCorrect.restype = POINTER(c_ubyte)
            colorCorrectRgb = lib.RgbColorCorrect(rgb, w, h, 3, self.colorTable)
            bits = np.ctypeslib.as_array(colorCorrectRgb, shape=(crop_width * crop_height * 3,)).copy()
            lib.Dispose(colorCorrectRgb)
        else:
            bits = np.ctypeslib.as_array(rgb, shape=(crop_width * crop_height * 3,))

        bits = bits.reshape((crop_height, crop_width, 3))
        finalwidth=int(crop_width * resize_ratio)
        finalheight=int(crop_height * resize_ratio)
        if finalwidth==0 and finalheight==0:
            crop_region=cv2.resize(bits,(1,1))
        elif finalwidth==0 and finalheight!=0:
            crop_region=cv2.resize(bits,(1,finalheight))
        elif finalwidth!=0 and finalheight==0:
            crop_region=cv2.resize(bits,(finalwidth,1))
        else:
            crop_region=cv2.resize(bits,(finalwidth,finalheight))

        lib.Dispose(rgb)
        crop_region = crop_region[:, :, ::-1]

        return crop_region


    def get_tile(self, x, y, z):
        scale = math.pow(2, self.maxlvl - z)
        r = 1024 * scale
        r = int(r)
        tile = self.read([x * r, y * r], [r, r], scale, greyscale=False)
        return Image.fromarray(tile, mode='RGB')


    def save_label(self, path):
        size = self.slide.contents.macrograph.contents.contents.streamSize
        data = self.slide.contents.macrograph.contents.contents.stream
        bits = np.ctypeslib.as_array(data, shape=(size,))
        with open(path, 'wb') as f:
            f.write(bits)


    def get_thumbnail(self, *args, **kwargs):
        width = self.slide.contents.thumbnail.contents.width
        height = self.slide.contents.thumbnail.contents.height
        data = self.slide.contents.thumbnail.contents.bgr
        size = width * height * 3
        bits = np.ctypeslib.as_array(data, shape=(size,))
        bits = bits.reshape(height, width, 3)
        bits = bits[:, :, ::-1]

        return Image.fromarray(bits, mode='RGB')


    @property
    def mpp(self):
        return self.slide.contents.picHead.contents.ruler

    def __del__(self):
        if self.colorTable:
            lib.DisposeColorCorrectTable(self.colorTable)
        lib.SqCloseSdpc(self.slide)

