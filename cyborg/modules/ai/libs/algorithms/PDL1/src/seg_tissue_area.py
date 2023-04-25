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

def median_kernel_selection(shape):
    median_kernel_shape = ((min(shape[0] // 1000, shape[1] // 1000) // 2) + 1) * 10 + 1
    max_median_shape = 51
    if median_kernel_shape > max_median_shape:
        median_kernel_shape = max_median_shape
    return median_kernel_shape

def auto_threshold(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img[np.where(img == 0)] = 255
    img = cv2.medianBlur(img, 5)
    #cv2.imwrite('./thumbnail.jpg',img)

    th2_ori = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th2 = 255 - th2_ori
    kernel_2 = np.ones((10, 10), np.uint8)
    th2 = cv2.dilate(th2, kernel_2)
    kernel = np.ones((30, 30), np.uint8)  # 设置小
    th2 = cv2.erode(th2, kernel)  # 做一个连tong
    
    #cv2.imwrite('./th2.jpg',th2)
    median_kernel_shape = median_kernel_selection(img.shape)
    th2 = cv2.medianBlur(th2,median_kernel_shape)
    #cv2.imwrite('./median.jpg',th2)
    
    contours,binary = sort_edge(th2)
    area_list = []

    return contours, binary

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
        new_contours,th2 = auto_threshold(out)

        new_contours = [cnt*scale for cnt in new_contours]
    except Exception as e:
       print(e)
       new_contours = []
       th2 = np.zeros((slide.width//16,slide.height//16))
    return new_contours,th2

