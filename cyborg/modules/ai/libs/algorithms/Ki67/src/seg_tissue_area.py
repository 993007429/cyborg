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
    return contours[0]

def auto_threshold(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img[np.where(img == 0)] = 255
    img = cv2.medianBlur(img, 5)
    th2_ori = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th2 = 255 - th2_ori
    kernel_2 = np.ones((10, 10), np.uint8)
    th2 = cv2.dilate(th2, kernel_2)
    kernel = np.ones((30, 30), np.uint8)  # 设置小
    th2 = cv2.erode(th2, kernel)  # 做一个连tong
    th2 = cv2.dilate(th2, kernel)
    contours = sort_edge(th2)
    area_list = []
    for roi_list in contours:
        area = cv2.contourArea(roi_list)
        area_list.append(area)
    area_list = np.array(area_list, dtype=int)

    new_contours = []
    pass_contours = []
    for i in range(area_list.size):
        #threshold = area_list[i] / np.max(area_list)
        #if threshold >=0.05:
        new_contours.append(contours[i])
    return new_contours

def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut)
    output_img = np.uint8(output_img + 0.5)  # 这句一定要加上
    return output_img

def find_tissue_countours(slide):
    try:
        scale=16
        img = slide.read((0,0), (slide.width, slide.height), scale)
        out = gamma(img, 0.00000005, 4.0)
        new_contours = auto_threshold(out)
        new_contours = [cnt*scale for cnt in new_contours]
    except Exception as e:
        print(e)
        new_contours = []
    return new_contours

