#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/12 14:55
# @Author  : Can Cui
# @File    : utils.py
# @Software: PyCharm
import os, sys
import yaml
import numpy as np
import cv2
import math
import traceback

alg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def load_cfg(cfg_path):
    model_map = load_yaml(os.path.join(alg_dir, 'deploy.yaml'))
    base_config_path = os.path.join(alg_dir, 'configs', 'base.yaml')
    if cfg_path in model_map:
        cfg_path = model_map[cfg_path]
    print(cfg_path)
    base_config_dict = load_yaml(base_config_path)
    if not cfg_path.endswith('.yaml'): cfg_path+='.yaml'
    if os.path.exists(os.path.join(alg_dir, 'configs', cfg_path)):
        config_dict = load_yaml(os.path.join(alg_dir, 'configs', cfg_path))
    elif os.path.exists(cfg_path):
        config_dict = load_yaml(cfg_path)
    else:
        raise FileNotFoundError(f'Config file <<{cfg_path}>> does not exist')

    # only support two levels of sub dict
    for k,v in config_dict.items():
        if not isinstance(v, dict):
            base_config_dict.update({k:v})
        else:
            for sub_k, sub_v in v.items():
                if k not in base_config_dict:
                    base_config_dict[k]={}
                base_config_dict[k].update({sub_k:sub_v})
    base_config_dict.update({'model_name':os.path.splitext(cfg_path)[0]})
    print(base_config_dict)
    return base_config_dict

class SlideRegion:
    def __init__(self, location=(0,0), size=(0,0), scale=1.0, image=None, ori_image=None, image2=None, sub_coords=None):
        self.location = location
        self.size = size
        self.scale = scale
        self.image = image
        self.ori_image = ori_image
        self.image2 = image2
        self.sub_coords = sub_coords

def match_bboxes_with_pathces(bboxes, crop_size):
    '''
      ### Assign ROIs to bigger patches
    :param bboxes:
    :param crop_size:
    :return:
    '''

    bboxes = bboxes.copy()
    box_w, box_h = bboxes[0,2]-bboxes[0,0], bboxes[0,3]-bboxes[0,1]
    xmin, ymin, xmax, ymax = np.min(bboxes[:,0]), np.min(bboxes[:,1]), np.max(bboxes[:,2])+1, np.max(bboxes[:,3])+1

    stride_x, stride_y = max(crop_size-box_w, box_w), max(crop_size-box_h, box_h)
    patch_bboxes_list = list()
    total_num = bboxes.shape[0]
    search_space = np.ones((bboxes.shape[0],), dtype=np.bool)

    for x in range(xmin, xmax, stride_x):
        for y in range(ymin, ymax, stride_y):

            search_idx = np.where(search_space==1)[0]
            search_bboxes = bboxes[search_idx]
            # print("search_box shape = {}  num_rois = {}".format(search_bboxes.shape, total_num))
            pick_idx = np.where(np.logical_and(
                np.logical_and(search_bboxes[:,0]>=x,search_bboxes[:,1]>=y),
                np.logical_and(search_bboxes[:,2]<x+crop_size, search_bboxes[:,3]<y+crop_size)))[0]

            if pick_idx.shape[0] >0:
                picked_bboxes = search_bboxes[pick_idx]
                crop_x1, crop_y1, crop_x2, crop_y2 = np.min(picked_bboxes[:,0]), np.min(picked_bboxes[:,1]),\
                                                     np.max(picked_bboxes[:,2]), np.max(picked_bboxes[:,3])
                crop_patch = np.array([crop_x1, crop_y1, crop_x2, crop_y2])

                picked_bboxes-=np.array([crop_x1, crop_y1, crop_x1, crop_y1])
                patch_bboxes_list.append((crop_patch, picked_bboxes))

                search_space[search_idx[pick_idx]] = 0
                total_num -= pick_idx.shape[0]

    return patch_bboxes_list

def read_region_worker(slide, in_queue, out_queue, window_size=320, crop_scale=1):
    '''Read big patch, then crop smaller patch from big patch'''
    while not in_queue.empty():
        slideRegion = in_queue.get()
        # print
        patch_x, patch_y = slideRegion.location
        crop_w, crop_h = math.ceil(window_size * (slideRegion.scale / crop_scale)), math.ceil(window_size * (slideRegion.scale / crop_scale)) # scale to level0 then rescale to crop level
        try:
            image = slide.read(slideRegion.location, slideRegion.size, crop_scale)
            crop_img_list = []
            sub_coords_list = []
            for i in range(slideRegion.sub_coords.shape[0]):
                x, y = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                xs, ys = math.ceil(x/crop_scale), math.ceil(y/crop_scale)  # rescale from level0 to crop level
                crop_img = image[ys:ys+crop_h, xs:xs+crop_w]
                h, w = crop_img.shape[0:2]
                pad_h, pad_w = crop_h - h, crop_w - w
                if pad_h > 0 or pad_w > 0:
                    crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant',constant_values=255)
                if crop_img.shape != (window_size, window_size):
                    crop_img = cv2.resize(crop_img, dsize=(window_size, window_size))
                crop_img_list.append(crop_img)
                sub_coords_list.append(slideRegion.sub_coords[i]+np.array([patch_x,patch_y,patch_x,patch_y]))
            for crop_img, sub_coords in zip(crop_img_list, sub_coords_list):
                out_queue.put(SlideRegion(image=crop_img,sub_coords=sub_coords))

        except Exception: # read big patch failed
            print(traceback.format_exc())
            sub_coords = slideRegion.sub_coords
            sub_coords+= np.array([patch_x, patch_y, patch_x, patch_y])
            num_coords = sub_coords.shape[0]
            for i in range(num_coords):
                try:
                    x1, y1, x2, y2 = sub_coords[i]
                    crop_img = slide.read((x1,x2), (x2-x1, y2-y1), crop_scale)
                    h, w = crop_img.shape[:2]
                    pad_h, pad_w = crop_h - h, crop_w - w
                    if pad_h > 0 or pad_w > 0:
                        crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant',constant_values=114)
                    if crop_img.shape != (window_size,window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))
                    out_queue.put(SlideRegion(
                        image=crop_img,sub_coords=sub_coords[i]
                    ))
                except:
                    print(traceback.format_exc())
                    out_queue.put(SlideRegion(
                        image=None, sub_coords=sub_coords[i]
                    ))
    return None

def read_patch_worker(slide, in_queue, out_queue, window_size=224, crop_scale=1):
    while not in_queue.empty():
        slideRegion = in_queue.get()
        crop_w, crop_h = math.ceil(window_size * (slideRegion.scale / crop_scale)), math.ceil(window_size * (slideRegion.scale / crop_scale)) # scale to level0 then rescale to crop level
        try:
            crop_img = slide.read(slideRegion.location, slideRegion.size, crop_scale)
            h, w = crop_img.shape[:2]
            pad_h, pad_w = crop_h - h, crop_w - w
            if pad_h > 0 or pad_w > 0:
                crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=114)
            if crop_img.shape != (window_size, window_size):
                crop_img = cv2.resize(crop_img, (window_size, window_size))
            slideRegion.image = crop_img
            out_queue.put(slideRegion)
        except:
            print(traceback.format_exc())
            slideRegion.image = None
            out_queue.put(slideRegion)
    return None

def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)

def generate_colors(i, bgr=False):
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = []
    for iter in hex:
        h = '#' + iter
        palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
    num = len(palette)
    color = palette[int(i) % num]
    return (color[2], color[1], color[0]) if bgr else color


if __name__ == '__main__':
    info = load_cfg('base1.yaml')