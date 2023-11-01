# -*- coding: utf-8 -*-
import os
import sys
import ctypes
if sys.platform == 'win32':
    raise("Format is not supported now")
    # lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'win_libs/libiblsdk.dll')
elif sys.platform == 'linux':
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'linux_lib/libiblsdk.so')
# print(lib_path)
lib = ctypes.cdll.LoadLibrary(lib_path)
print('load libiblsdk.so success')


class IblWsi:
    def __init__(self, ibl_path):
        self.filePath = ibl_path
        self.width = ctypes.c_int()
        self.height = ctypes.c_int()
        self.depth = ctypes.c_int()
        self.scanScale = ctypes.c_int()
        self.jpegQuality = ctypes.c_int()
        self.focusNumber = ctypes.c_int()
        self.bkColor = ctypes.c_int()
        self.fPixelSize = ctypes.c_double(0.0)
        self.ret = lib.OpenIBL(self.filePath)
        self.ret = lib.GetInfo(self.filePath, ctypes.byref(self.width), ctypes.byref(self.height), ctypes.byref(self.depth), ctypes.byref(self.scanScale), ctypes.byref(self.jpegQuality), ctypes.byref(self.focusNumber), ctypes.byref(self.bkColor), ctypes.byref(self.fPixelSize))

    def GetMacroData(self):
        MacroDataBuffer = ctypes.create_string_buffer(2500*800*3)
        lib.GetMacroData(self.filePath, MacroDataBuffer)
        return MacroDataBuffer

    def GetLabelData(self):
        LabelDataBuffer = ctypes.create_string_buffer(800*800*3)
        lib.GetLabelData(self.filePath, LabelDataBuffer)
        return LabelDataBuffer

    def DecodeTile(self, layer, posX, posY, focus=0):
        """
        layer starts from 0, posX and posY means the tile's index, not the pixel position
        pucData should be allocated by user,please Make sure the memory is large enough
        Tile width/height 256*256
        return TileDataBuffer
        """
        TILE_LEN = 256
        TileDataBuffer = ctypes.create_string_buffer(TILE_LEN * TILE_LEN * 3)
        lib.DecodeTile(self.filePath, layer, posX, posY, TileDataBuffer, 1)
        return TileDataBuffer


    def GetRoiData(self, zoomRate, x, y, w, h, focus=0):
        """
        zoomRate, current scale 40/20/10/5,...
        (x, y) start pixel point of ROI in current scale
        (w, h) pixel size of ROI in current scale
        data should be allocated by user and make sure it's large enough
        return RoiDataBuffer
        """
        RoiDataBuffer = ctypes.create_string_buffer(w * h * 3)
        lib.GetRoiData(self.filePath, ctypes.c_float(zoomRate), x, y, w, h, RoiDataBuffer)
        return RoiDataBuffer

    def CloseIBL(self):
        lib.CloseIBL(self.filePath)


if __name__ == '__main__':
    # 定义WSI路径
    wsi_path = b"./1-13-1-20221113-094504.ibl"

    # 初始化
    wsi = IblWsi(wsi_path)

    # 打印WSI相关信息
    '''
    fileName is local file
    width:图像宽度
    height：图像高度
    depth: 图像深度
    scanScale: 扫描倍率20X/40x
    jpegQuality:jpeg压缩比
    focusNumber：3D扫描>=0 其余的模式为0
    bkColor：背景值
    fPixelSize
    '''
    print('WSI INFO, width:{}, height:{}, depth:{}, scanScale:{}, jpegQuality:{}, focusNumber:{}, bkColor:{}, fPixelSize:{}.'.format(wsi.width.value, wsi.height.value, wsi.depth.value,
                                                                                                                                     wsi.scanScale.value,
                                                                                                                                     wsi.jpegQuality.value, wsi.focusNumber.value, wsi.bkColor.value,
                                                                                                                                     wsi.fPixelSize.value))

    # 保存标签图
    with open('./save/label_img.jpg', "w+b") as fo:
        fo.write(wsi.GetLabelData())

    # 保存宏观图
    with open('./save/macro_img.jpg', "w+b") as fo:
        fo.write(wsi.GetMacroData())

    # DecodeTile
    with open('./save/tile_img.jpg', "w+b") as fo:
        fo.write(wsi.DecodeTile(layer=4, posX=2, posY=2))

    # 以2048*2048块大小切分并保存整张WSI
    patch_size = 2048
    n_width = wsi.width.value // patch_size
    n_height = wsi.height.value // patch_size
    for i in range(n_width):
        for j in range(n_height):
            with open('./save/img_{}_{}.jpg'.format(i, j), "w+b") as fo:
                fo.write(wsi.GetRoiData(float(wsi.scanScale.value), i * patch_size, j * patch_size, patch_size, patch_size))
    wsi.CloseIBL()