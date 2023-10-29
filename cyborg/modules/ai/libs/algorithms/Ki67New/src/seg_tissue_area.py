#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 22:53
# @Author  : Can Cui
# @File    : seg_tissue_area.py
# @Software: PyCharm
# @Comment:

import cv2
import numpy as np
import time
import os
import torch
from skimage import io
from skimage.color import rgb2hed, hed2rgb
from cyborg.modules.ai.libs.algorithms.Ki67New.src.kmeans_pytorch import kmeans
from torchvision import transforms as T

def sort_edge(image,thresh):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(image, thresh+1, 255, cv2.THRESH_BINARY_INV)
    contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0],binary

def auto_threshold(image):
    #time_1 = time.time()
    #io.imsave(f'./log_imgs/thumbnail_{time_1}.jpg',image)
    status = False
    to_tensor = T.ToTensor()
    ihc_hed = rgb2hed(image)
    null = np.zeros_like(ihc_hed[:,:,0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:,:,0],null,null),axis=-1))
    ihc_h = (ihc_h/(ihc_h.max()-ihc_h.min()))*255
    ihc_h = ihc_h.astype(np.uint8)
    km_array = to_tensor(ihc_h).permute((1,2,0)).reshape((-1,3))
    
    while not status: 
        cluster_idx,cluster_centers,status = kmeans(X=km_array,num_clusters=2,device=torch.device('cuda:0'))
    cluster_idx = cluster_idx.reshape((ihc_h.shape[0:2])).numpy()
    torch.cuda.empty_cache()

    cluster_centers = (cluster_centers*255).numpy().astype(np.uint8)
    
    ihc_h[cluster_idx==0] = cluster_centers[0]
    ihc_h[cluster_idx==1] = cluster_centers[1]

    #io.imsave(f'./log_imgs/h_ch_{time_1}.jpg',bin_image)
    gray_img = cv2.cvtColor(ihc_h,cv2.COLOR_RGB2GRAY)
    #io.imsave(f'./log_imgs/gray_{time_1}.jpg',bin_image)
    contours,binary = sort_edge(gray_img,gray_img.min())
    #io.imsave(f'./log_imgs/bin_{time_1}.jpg',binary)
    #os.system(f'taskkill /t /f /pid {os.getpid()}')
    
    return contours,binary


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
       #logger.info(e)
       new_contours = []
       th2 = np.zeros((slide.width//16,slide.height//16))
    return new_contours,th2
