import sys
import math
from ctypes import *

import cv2
import numpy as np
from PIL import Image

from .pystruct import SqSdpcInfo
from ..SlideBase import SlideBase

if sys.platform == 'win32':
    cur_encoding = 'gbk'
    lib = windll.LoadLibrary('DecodeSdpcDll.dll')
else:
    cur_encoding = 'utf-8'
    lib = cdll.LoadLibrary('libDecodeSdpc.so')


class SdpcSlide(SlideBase):

    def __init__(self, filename):

        self.filename = filename
        self.bfilename = self.filename.encode(cur_encoding)
        lib.SqOpenSdpc.restype = POINTER(SqSdpcInfo)
        self.slide = lib.SqOpenSdpc(c_char_p(self.bfilename))
        self.width = self.slide.contents.picHead.contents.srcWidth
        self.height = self.slide.contents.picHead.contents.srcHeight
        self.scale = self.slide.contents.picHead.contents.scale
        # if self.scale!=0.5:
        #     raise Exception('scale is not 0.5 that we need!')
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
        crop_level = math.floor(math.log2(scale))
        crop_level = max(0, min(crop_level, self.slide.contents.picHead.contents.hierarchy - 1))

        level_ratio = 2 ** crop_level if self.scale == 0.5 else 4 ** crop_level
        resize_ratio = level_ratio / scale

        # make sure the crop region is inside the slide
        crop_start_x, crop_start_y = min(max(crop_start_x, 0), self.width), min(max(crop_start_y, 0), self.height)
        crop_end_x = math.ceil(min(max(width + crop_start_x, 0), self.width))
        crop_end_y = math.ceil(min(max(height + crop_start_y, 0), self.height))

        crop_width = math.ceil((crop_end_x - crop_start_x) / level_ratio)
        crop_height = math.ceil((crop_end_y - crop_start_y) / level_ratio)

        if crop_height == 0 or crop_width == 0:
            return None
        layer = c_int(crop_level)
        # rgb = pointer(pointer(c_ubyte()))
        rgb = pointer(c_ubyte())
        w = c_int(crop_width)
        h = c_int(crop_height)
        xp = c_uint(int(crop_start_x / level_ratio))
        yp = c_uint(int(crop_start_y / level_ratio))

        lib.SqGetRoiRgbOfSpecifyLayer(self.slide, byref(rgb), w, h, xp, yp, layer)

        # if self.slide.contents.extra is not None:
        #     gamma = self.slide.contents.extra.contents.ccmGamma
        #     if self.slide.contents.extra.contents.cameraGamma > 0.000001:
        #         gamma = self.slide.contents.extra.contents.cameraGamma
        #     lib.InitColorCollectTable.restype = POINTER(SqColorTable)
        #     # lib.InitColorCollectTable.argtypes = [POINTER(c_float), POINTER(c_float), c_float, POINTER(c_float)]
        #     colorTable = lib.InitColorCollectTable(self.slide)
        #     colorCorrectRgb = pointer((c_ubyte*crop_width*crop_height*3)())
        #     lib.RgbColorCorrect(rgb, colorCorrectRgb, w, h,3, colorTable)
        #     bits = np.ctypeslib.as_array(colorCorrectRgb.contents, shape=(crop_width * crop_height * 3,))
        #     # lib.Dispose(colorCorrectRgb)
        #     lib.DisposeColorCorrectTable(colorTable)
        # else:
        #     bits = np.ctypeslib.as_array(rgb, shape=(crop_width * crop_height * 3,))

        bits = np.ctypeslib.as_array(rgb, shape=(crop_width * crop_height * 3,))

        bits = bits.reshape((crop_height, crop_width, 3))

        finalwidth = int(crop_width * resize_ratio)
        finalheight = int(crop_height * resize_ratio)

        if finalwidth == 0 and finalheight == 0:
            crop_region = cv2.resize(bits, (1, 1))
        elif finalwidth == 0 and finalheight != 0:
            crop_region = cv2.resize(bits, (1, finalheight))
        elif finalwidth != 0 and finalheight == 0:
            crop_region = cv2.resize(bits, (finalwidth, 1))
        else:
            crop_region = cv2.resize(bits, (finalwidth, finalheight))

        lib.Dispose(rgb)
        crop_region = crop_region[:, :, ::-1]

        return crop_region

    def get_tile(self, x, y, z):
        scale = math.pow(2, self.maxlvl - z)
        r = 1024 * scale
        r = int(r)
        tile = self.read([x * r, y * r], [r, r], scale, greyscale=False)
        return Image.fromarray(tile, mode='RGB')
        # return tile

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
        lib.SqCloseSdpc(self.slide)


if __name__ == "__main__":
    # slide = SdpcSlide(r"C:\Users\Admin\Desktop\\20191129_140040.sdpc")
    # import cv2

    slide = SdpcSlide(r"C:\Users\Admin\Desktop\\图片1_0.5.sdpc")

    res = slide.get_tile(0, 0, 17)
    # print(slide.height,slide.width,slide.mpp)
    print(slide.mpp)
    # cv2.imshow('image', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(help(slide.info))
    # print(slide.slide.contents.picHead.contents.hierarchy)
    # slide.save_label('1.png')

    # for z in range(0,1):
    #     os.makedirs("test/{}".format(z), exist_ok= True)
    #     for x in range(100):
    #         for y in range(100):
    #             try:
    #                 img = slide.get_tile(x,y,z)
    #                 img.save("test/{}/{}_{}.jpg".format(z,y,x))
    #                 print(x,y,z,'生成成功')
    #             except Exception as e:
    #                 # import traceback
    #                 # traceback.print_exc()
    #                 pass
