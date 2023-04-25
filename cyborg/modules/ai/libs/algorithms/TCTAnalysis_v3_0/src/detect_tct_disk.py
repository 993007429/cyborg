#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 19:30
# @Author  : Can Cui
# @File    : detect_tct_disk.py
# @Software: PyCharm
# @Comment:

from scipy.ndimage import binary_fill_holes
from skimage import filters, img_as_ubyte
import cv2
import os, sys
from PIL import Image
import numpy as np
from skimage.morphology import remove_small_objects
# from scipy import misc
from imageio import imread, imsave


def walk_dir(data_dir, file_types= ['.jpg']):
    # file_types = ['.txt', '.kfb']
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:

                if f.lower().endswith(this_type):
                    path_list.append(os.path.join(dirpath, f))
                    break
    return path_list



def get_thumbnail(slide, img_size=1024):

    thumbnail_img = None
    try:
        thumbnail_img = slide.get_thumbnail(size=(img_size, img_size))
    except:
        for k, v in slide.associated_images.items():
            if 'thumbnail' in str(k):
                thumbnail_img = slide.associated_images[k]

    if thumbnail_img:
        if thumbnail_img.mode == 'RGBA':
            thumbnail_img = thumbnail_img.convert('RGB')
    thumbnail_arr = np.array(thumbnail_img)
    ori_w, ori_h = slide.dimensions
    new_h, new_w = thumbnail_arr.shape[0], thumbnail_arr.shape[1]
    ratio_x, ratio_y = ori_w/new_w, ori_h/new_h
    return thumbnail_arr, ratio_x, ratio_y


# resize thumbnail to the proper size
def _resize(image, bounding, padding=True):
    if not isinstance(image, np.ndarray):
        w, h = image.size
        if w <= bounding and h <= bounding:
            if padding:
                new_img = Image.new(mode="RGB", size=(bounding, bounding), color=(255, 255, 255))
                new_img.paste(image, ((bounding-w)//2, (bounding-h)//2))
            else:
                new_img = image
            return new_img
        else:
            scale1, scale2 = w/bounding, h/bounding
            if(scale1 >= scale2):
                image = image.resize((int(w/scale1), int(h/scale1)))
            else:
                image = image.resize(((int(w/scale2), int(h/scale2))))
            return image

    else:
        h, w = image.shape[0], image.shape[1]
        if w <= bounding and h <= bounding:
            if padding:
                new_img = np.ones((bounding, bounding, 3))*255
                new_img[int(((bounding-h)/2)):int(((bounding+h)/2)), int(((bounding-w)/2)):int(((bounding+w)/2)), :] = image
            else:
                new_img = image
            return new_img
        else:
            scale1, scale2 = w/bounding, h/bounding
            if(scale1 >= scale2):
                image = cv2.resize(image, (int(w/scale1), int(h/scale1)))
            else:
                image = cv2.resize(image, (int(w/scale2), int(h/scale2)))
            return image


def rgb2gray(img):
    gray = np.dot(img, [0.299, 0.587, 0.114])
    return gray.astype(np.uint8)


def thresh_slide(gray, thresh_val, sigma=5):
    # Smooth
    # smooth = filters.gaussian(gray, sigma=sigma)
    # smooth /= np.amax(smooth)
    smooth = cv2.blur(gray, (7,7))

    # import pdb; pdb.set_trace()
    # Threshold
    # bw_img = smooth < thresh_val
    # s1 = (smooth[:, :, np.newaxis]*255).astype(np.uint8)
    s1 = smooth.astype(np.uint8)
    thread, bw_img = cv2.threshold(s1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(thread)
    # import pdb; pdb.set_trace()
    # for i in range(255):
    #     _, bw_img = cv2.threshold(s1, i, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     imsave("result1/{}.jpg".format(i), bw_img)
    # _, bw_img = cv2.threshold(s1, 0, 255, cv2.THRESH_OTSU)

    # bw_img = cv2.adaptiveThreshold(s1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


    bw_img = ~bw_img

    return bw_img.astype(np.uint8)


def fill_tissue_holes(bw_img):
    # Fill holes
    bw_fill = binary_fill_holes(bw_img)
    return bw_fill.astype(np.uint8)

def remove_small_tissue(bw_img, min_size=100000000000000):
    bw_img = bw_img.astype(np.bool)
    bw_remove = remove_small_objects(bw_img, min_size=min_size, connectivity=1)
    imsave("bw_img.png", bw_img*255)
    imsave("remove.png", bw_remove*255)

    import pdb; pdb.set_trace()
    return bw_remove.astype(np.uint8)


def find_tissue_cnts(bw_img):
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, cnts, _ = cv2.findContours(img_as_ubyte(bw_img),
                                      mode=cv2.RETR_EXTERNAL,
                                      method=cv2.CHAIN_APPROX_NONE)
    else:
        cnts, _ = cv2.findContours(img_as_ubyte(bw_img),
                                      mode=cv2.RETR_EXTERNAL,
                                      method=cv2.CHAIN_APPROX_NONE)
    return cnts

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(32,32))
    clahe_img = clahe.apply(gray_img)
    return clahe_img

def apply_closing(bw_remove, iterations=3):
    # kernel = np.ones((5, 5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    bw_dilation = cv2.dilate(bw_remove, kernel, iterations=iterations)
    bw_erosion = cv2.erode(bw_dilation, kernel, iterations=iterations)
    # closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
    return bw_erosion


def add_mask(img, mask):
    W, H, C = img.shape
    assert (W,H) == mask.shape
    for w in range(W):
        for h in range(H):
            if mask[w,h] == 1:
                img[w,h,0] = 255
    return img

def apply_dilation(bw_remove, iterations=3):
    kernel = np.ones((5, 5), np.uint8)
    bw_dilation = cv2.dilate(bw_remove, kernel, iterations=iterations)
    return  bw_dilation.astype(np.uint8)


def locate_tct_disk(slide, iterations=2):
    # Step1. get small slide image
    H, W = slide.height, slide.width
    slide_img = np.array(slide.getThumbnail(1024))
    h, w = slide_img.shape[0:2]
    ratio_x, ratio_y = W/w, H/h

    hsv = cv2.cvtColor(slide_img, cv2.COLOR_RGB2HSV)
    noise = hsv[:, :, 2] <120
    slide_img[noise] = [255, 255, 255]
    gray_img = rgb2gray(slide_img)
    gray_img = cv2.blur(gray_img, (5,5))
    gray_img = apply_clahe(gray_img)
    s1 = gray_img.astype(np.uint8)
    thread, bw_img = cv2.threshold(s1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_img = ~bw_img
    bw_img = cv2.blur(bw_img, (9,9))
    bw_closing = apply_closing(bw_img, iterations)

    cnts = find_tissue_cnts(bw_closing)
    if len(cnts) == 0:
        x_coords, y_coords  = [0,0,W,W], [0,0,H,H]
    else:
        max_idx = 0
        max_area = 0
        for idx, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_idx = idx
        max_cnt = cnts[max_idx]
        x_coords = list(max_cnt[:,0,0]*ratio_x)
        y_coords = list(max_cnt[:,0,1]*ratio_y)
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))

        if (xmax-xmin)*(ymax-ymin) < 2e4**2:
            x_coords = [0,0,W, W]
            y_coords = [0,0,H, H]

        # cv2.drawContours(slide_img, [cnts[max_idx]], -1, (0, 255, 0), 5)
        # imsave("test.jpg", slide_img)

    return x_coords, y_coords


def disc_detect(slide, is_detect=True):
    if is_detect:
        try:
            x_coords, y_coords = locate_tct_disk(slide)
        except:
            x_coords, y_coords = [], []
    else:
        x_coords, y_coords = [], []
    return x_coords, y_coords

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from Slide.dispatch import openSlide
    slide = openSlide('E:\TCT_TRAIN\BJH301CP40X024_228.kfb')
    locate_tct_disk(slide)