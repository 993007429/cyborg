import logging
import math
import os
import random
from threading import Thread

import cv2
import joblib
import numpy as np
from skimage.feature import hog

logger = logging.getLogger(__name__)


def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(32, 32))
    clahe_img = clahe.apply(gray_img)
    return clahe_img


def seg_tissue(slide):
    height, width = slide.height, slide.width
    slide_img = np.array(slide.get_thumbnail(1024))
    h, w = slide_img.shape[0:2]
    ratio_x, ratio_y = width / w, height / h

    hsv = cv2.cvtColor(slide_img, cv2.COLOR_RGB2HSV)
    noise = hsv[:, :, 2] < 120
    slide_img[noise] = [255, 255, 255]
    gray_img = cv2.cvtColor(slide_img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.blur(gray_img, (5, 5))
    gray_img = apply_clahe(gray_img)
    s1 = gray_img.astype(np.uint8)
    thread, bw_img = cv2.threshold(s1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_img = ~bw_img
    return bw_img, ratio_x, ratio_y


def slide_cls(slide):
    model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clf.pkl'))
    slide_img = np.array(slide.get_thumbnail(512))
    slide_img = cv2.resize(slide_img, (512, 512))
    gray = cv2.cvtColor(slide_img, cv2.COLOR_RGB2GRAY)
    fd = hog(gray, orientations=8, pixels_per_cell=(64, 64),
             cells_per_block=(3, 3))
    slide_type = model.predict(fd.reshape(1, -1))[0]
    # {1:'TCT', 0:'IHC'}
    return slide_type


def blur_check_worker(slide, coords_list, res_list, size=256, scale=1):
    for coord in coords_list:
        try:
            img = slide.read(coord, (size, size), scale=scale)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            res_list.append(score)
        except Exception as e:
            logger.info(e)
    return None


def blur_check(slide):
    pick_num = 400
    num_worker = 10
    if slide.mpp:
        scale = round(0.48 / slide.mpp)
    else:
        scale = 1
    patch_size = 256 * scale
    stride = 512 * scale
    random.seed(1024)

    slide_type = slide_cls(slide)
    defual_thres = 20 if slide_type == 0 else 60

    logger.info(f'slide type: {slide_type}  thres: {defual_thres}')

    mask, ratio_x, ratio_y = seg_tissue(slide)
    mask_patch_size = math.ceil(patch_size / ratio_x)
    coord_list = []
    for x in range(0, slide.width, stride):
        for y in range(0, slide.height, stride):
            try:
                xs, ys = math.floor(x / ratio_x), math.floor(y / ratio_y)
                patch_mask = mask[ys:ys + mask_patch_size, xs:xs + mask_patch_size]
                if patch_mask.sum() / patch_mask.size >= 0.5:
                    coord_list.append((x, y))
            except Exception as e:
                logger.info(e)
    random.shuffle(coord_list)
    pick_list = coord_list[:pick_num]

    clarity_list = []
    partition = math.ceil(len(pick_list) / num_worker)
    t_list = []
    for i in range(num_worker):
        this_partition_list = pick_list[i * partition:(i + 1) * partition]
        t = Thread(target=blur_check_worker, args=(slide, this_partition_list, clarity_list, patch_size, scale))
        t.start()
        t_list.append(t)
    for t in t_list:
        t.join()
    clarity_list = sorted(clarity_list)[::-1][:pick_num // 2]
    if np.min(clarity_list) >= 400:
        clarity_score = 1
    else:
        clarity_list = np.array(clarity_list)
        thres = max(np.mean(clarity_list[:10]) * 0.2, defual_thres)
        clarity_score = float(np.sum(np.array(clarity_list) > thres)) / max((len(clarity_list), 1e-10))

    if clarity_score < 0.1:
        clarity_score = 0

    return round(clarity_score, 3)
