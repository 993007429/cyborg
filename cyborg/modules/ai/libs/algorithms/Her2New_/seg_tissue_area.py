#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 22:53
# @Author  : Can Cui
# @File    : seg_tissue_area.py
# @Software: PyCharm
# @Comment:

import cv2
import numpy as np

def sort_edge(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0],binary

def auto_threshold(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img[np.where(img == 0)] = 255
    img = cv2.medianBlur(img, 5)
    th2_ori = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th2 = 255 - th2_ori
    #cv2.imwrite('D:\znbl3_1020/th_1.png', th2)
    kernel_2 = np.ones((10, 10), np.uint8)
    th2 = cv2.dilate(th2, kernel_2)
    #cv2.imwrite('D:\znbl3_1020/th_2.png', th2)
    k = 30
    print(f'Img:{img.shape}')
    hh,ww = img.shape
    if hh < 3000 and ww < 3000:
        k = 10
    kernel = np.ones((k, k), np.uint8)  # 设置小
    th2 = cv2.erode(th2, kernel)  # 做一个连tong
    th2 = cv2.medianBlur(th2, 51)
    contours,binary = sort_edge(th2)
    return contours,binary

def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut)
    output_img = np.uint8(output_img + 0.5)  # 这句一定要加上
    return output_img

def find_tissue_countours(slide,scale):
    try:
        scale = scale
        print(f'Scale:{scale}')
        img = slide.read((0,0), (slide.width, slide.height), scale=scale)
        print(f'image:{img.shape}')
        out = gamma(img, 0.00000005, 4.0)
        contours,binary = auto_threshold(out)
        new_contours = [binary]
        #new_contours = [cnt*scale for cnt in new_contours]
        new_contours = binary * scale
        flg = True
    except Exception as e:
        print('countours error')
        print(e)
        new_contours = None
        flg = False
    return new_contours,flg


# def find_tissue_countours_new(slide):
#     try:
#         ww = math.ceil(slide.width / 16)
#         hh = math.ceil(slide.width / 16)
#         st_x = 0
#         st_y = 0
#         for idx in range(16):




