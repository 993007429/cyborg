#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/7/11 21:30
# @Author  : Can Cui
# @File    : cell_cls.py
# @Software: PyCharm
# @Comment:

from queue import Queue
from threading import Thread
import math
from .utils import *
import torch
import time

def read_patch_worker_20x(slide, in_queue, out_queue, window_size=320):
    while not in_queue.empty():
        #
        slideRegion = in_queue.get()
        patch_x, patch_y = slideRegion.location

        try:
            image = slide.read(slideRegion.location, slideRegion.size, slideRegion.scale)
            image = image * (2. / 255) - 1.
            image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

            for i in range(slideRegion.sub_coords.shape[0]):
                x, y = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                xs, ys = math.ceil(x/slideRegion.scale), math.ceil(y/slideRegion.scale)
                crop_img = image[ys:ys+window_size, xs:xs+window_size]

                if crop_img.shape != (window_size,window_size):
                    crop_img = cv2.resize(crop_img, (window_size, window_size))

                crop_img = crop_img[None]
                out_queue.put(SlideRegion(
                    image=crop_img, sub_coords=slideRegion.sub_coords[i]+np.array([patch_x,patch_y,patch_x,patch_y])
                ))
        except Exception as e: # read big patch failed
            print(e)
            sub_coords = slideRegion.sub_coords
            sub_coords+= np.array([patch_x, patch_y, patch_x, patch_y])
            num_coords = sub_coords.shape[0]
            for i in range(num_coords):
                try:
                    x1, y1, x2, y2 = sub_coords[i]
                    crop_img = slide.read((x1,x2), (x2-x1, y2-y1), slideRegion.scale)
                    crop_img = crop_img * (2. / 255) - 1.
                    crop_img = 0.2989 * crop_img[:, :, 0] + 0.5870 * crop_img[:, :, 1] + 0.1140 * crop_img[:, :, 2]

                    if crop_img.shape != (window_size,window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))
                    crop_img = crop_img[None]

                    out_queue.put(SlideRegion(
                        image=crop_img, sub_coords=sub_coords[i]
                    ))
                except Exception as e:
                    print(e)
                    out_queue.put(SlideRegion(
                        image=None, sub_coords=sub_coords[i]
                    ))
    return None

def detect_cells_microbes20x(slide, center_coords, cell_classifier=None, microbe_classifier=None,
                          patch_size=160, overlap=40, batch_size=128, num_workers=4):

    crop_queue = Queue()
    process_queue = Queue(batch_size*8)

    standard_mpp = 0.242042*2
    this_mpp = slide.mpp
    scale_ratio = standard_mpp/this_mpp

    lvl0_crop_size = int(patch_size*scale_ratio)
    lvl0_overlap = int(overlap*scale_ratio)
    lvl0_stride = lvl0_crop_size-lvl0_overlap

    bboxes = sampling_bboxes(center_coords, lvl0_stride, lvl0_crop_size, border=[0,0,slide.width, slide.height])
    num_bboxes = bboxes.shape[0]
    patch_bboxes_list = match_bboxes_with_pathces(bboxes=bboxes, crop_size=5120)

    for i in range(len(patch_bboxes_list)):
        patch_coord, bbox_coord = patch_bboxes_list[i]
        xs, ys =  patch_coord[0], patch_coord[1]
        w , h  = patch_coord[2]-patch_coord[0], patch_coord[3]-patch_coord[1]
        crop_queue.put(SlideRegion(location=(xs, ys), size=(w, h), scale=scale_ratio, sub_coords=bbox_coord))

    for i in range(num_workers):
        t = Thread(target=read_patch_worker_20x, args=(slide, crop_queue, process_queue, patch_size))
        t.start()

    batch_data_list = []
    bbox_list, feat_list, cell_soft_list, microbe_soft_list = [], [], [], []
    with torch.no_grad():
        for i in range(num_bboxes):
            # print("process_queue = {}".format(process_queue.qsize()))
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)
                bbox_list.append(slide_region.sub_coords)

            if len(batch_data_list) == batch_size or (i == num_bboxes - 1 and len(batch_data_list)>0):

                data_variable = torch.tensor(np.array(batch_data_list), dtype=torch.float, device=torch.device('cuda'))

                _, feat, cell_soft = cell_classifier(data_variable)
                feat_list.append(feat)
                cell_soft_list.append(cell_soft)
                if microbe_classifier is not None:
                    _, _, microbe_soft = microbe_classifier(data_variable)
                    microbe_soft_list.append(microbe_soft)
                batch_data_list = []

        bboxes = np.array(bbox_list)
        cell_soft = torch.cat(cell_soft_list, 0).cpu().numpy()
        microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy() if len(microbe_soft_list)>0 else np.empty_like(cell_soft, dtype=float)
        feat = torch.cat(feat_list,0).cpu().numpy()

    return bboxes, cell_soft, microbe_soft, feat



def read_patch_worker_20x_scale1(slide, in_queue, out_queue, window_size=320):
    while not in_queue.empty():
        #
        slideRegion = in_queue.get()
        patch_x, patch_y = slideRegion.location

        try:
            image = slide.read(slideRegion.location, slideRegion.size, 1)

            for i in range(slideRegion.sub_coords.shape[0]):
                xs, ys = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                crop_img = image[ys:ys+int(window_size *  slideRegion.scale), xs:xs+int(window_size *  slideRegion.scale)]

                if crop_img.shape != (window_size,window_size):
                    crop_img = cv2.resize(crop_img, (window_size, window_size))
                ##add
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                crop_img = crop_img * (2. / 255) - 1.

                crop_img = crop_img[None]

                out_queue.put(SlideRegion(
                    image=crop_img,sub_coords=slideRegion.sub_coords[i]+np.array([patch_x,patch_y,patch_x,patch_y])
                ))
        except Exception as e: # read big patch failed
            print(e)
            sub_coords = slideRegion.sub_coords
            sub_coords+= np.array([patch_x, patch_y, patch_x, patch_y])
            num_coords = sub_coords.shape[0]
            for i in range(num_coords):
                try:
                    x1, y1, x2, y2 = sub_coords[i]
                    crop_img = slide.read((x1,x2), (x2-x1, y2-y1), 1)
                    ori_img = crop_img.copy()
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

                    crop_img = crop_img * (2. / 255) - 1.

                    if crop_img.shape != (window_size,window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))
                        ori_img = cv2.resize(ori_img, (window_size, window_size))
                    crop_img = crop_img[None]

                    out_queue.put(SlideRegion(
                        image=crop_img,ori_image=ori_img,sub_coords=sub_coords[i]
                    ))
                except Exception as e:
                    print(e)
                    out_queue.put(SlideRegion(
                        image=None, image2=None, sub_coords=sub_coords[i]
                    ))
    return None

def detect_cells_microbes20x_scale1(slide, center_coords, cell_classifier=None, microbe_classifier=None,
                          patch_size=160, overlap=40, batch_size=128, num_workers=4):

    crop_queue = Queue()
    process_queue = Queue(batch_size*8)

    standard_mpp = 0.242042*2
    this_mpp = slide.mpp
    scale_ratio = standard_mpp/this_mpp

    lvl0_crop_size = int(patch_size*scale_ratio)
    lvl0_overlap = int(overlap*scale_ratio)
    lvl0_stride = lvl0_crop_size-lvl0_overlap

    bboxes = sampling_bboxes(center_coords, lvl0_stride, lvl0_crop_size, border=[0,0,slide.width, slide.height])
    num_bboxes = bboxes.shape[0]
    patch_bboxes_list = match_bboxes_with_pathces(bboxes=bboxes, crop_size=5120)

    for i in range(len(patch_bboxes_list)):
        patch_coord, bbox_coord = patch_bboxes_list[i]
        xs, ys =  patch_coord[0], patch_coord[1]
        w , h  = patch_coord[2]-patch_coord[0], patch_coord[3]-patch_coord[1]
        crop_queue.put(SlideRegion(location=(xs, ys), size=(w, h), scale=scale_ratio, sub_coords=bbox_coord))

    for i in range(num_workers):
        t = Thread(target=read_patch_worker_20x_scale1, args=(slide, crop_queue, process_queue, patch_size))
        t.start()

    batch_data_list = []
    bbox_list, feat_list, cell_soft_list, microbe_soft_list = [], [], [], []

    with torch.no_grad():
        for i in range(num_bboxes):
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)
                bbox_list.append(slide_region.sub_coords)

            if len(batch_data_list) == batch_size or (i == num_bboxes - 1 and len(batch_data_list)>0):
                data_variable = torch.tensor(np.array(batch_data_list), dtype=torch.float, device=torch.device('cuda'))
                _, feat, cell_soft = cell_classifier(data_variable)
                feat_list.append(feat)
                cell_soft_list.append(cell_soft)

                if microbe_classifier is not None:
                    _, _, microbe_soft = microbe_classifier(data_variable)
                    microbe_soft_list.append(microbe_soft)
                batch_data_list = []

        bboxes = np.array(bbox_list)
        cell_soft = torch.cat(cell_soft_list, 0).cpu().numpy()
        microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy() if len(microbe_soft_list)>0 else np.empty_like(cell_soft, dtype=float)
        feat = torch.cat(feat_list,0).cpu().numpy()

    return bboxes, cell_soft, microbe_soft, feat


def read_patch_worker_10x_20x(slide, in_queue, out_queue, window_size=320):
    while not in_queue.empty():
        #
        slideRegion = in_queue.get()
        patch_x, patch_y = slideRegion.location

        try:
            image = slide.read(slideRegion.location, slideRegion.size, slideRegion.scale)
            image_gray = image * (2. / 255) - 1.
            image_gray = 0.2989 * image_gray[:, :, 0] + 0.5870 * image_gray[:, :, 1] + 0.1140 * image_gray[:, :, 2]

            for i in range(slideRegion.sub_coords.shape[0]):
                x, y = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                xs, ys = math.ceil(x/slideRegion.scale), math.ceil(y/slideRegion.scale)
                crop_img = image_gray[ys:ys+window_size, xs:xs+window_size]
                ori_image = image[ys:ys+window_size, xs:xs+window_size]
                if crop_img.shape != (window_size,window_size):
                    crop_img = cv2.resize(crop_img, (window_size, window_size))
                    ori_image = cv2.resize(ori_image, (window_size, window_size))
                crop_img10x = cv2.resize(crop_img, (window_size//2, window_size//2))
                # 不是应该在10倍下crop320*320吗
                # ori_img不需要缩减一个维度
                crop_img = crop_img[None]
                crop_img10x = crop_img10x[None]
                out_queue.put(SlideRegion(
                    image=crop_img,ori_image=ori_image,image2=crop_img10x, sub_coords=slideRegion.sub_coords[i]+np.array([patch_x,patch_y,patch_x,patch_y])
                ))
        except Exception as e: # read big patch failed
            print(e)
            sub_coords = slideRegion.sub_coords
            sub_coords+= np.array([patch_x, patch_y, patch_x, patch_y])
            num_coords = sub_coords.shape[0]
            for i in range(num_coords):
                try:
                    x1, y1, x2, y2 = sub_coords[i]
                    crop_img = slide.read((x1,x2), (x2-x1, y2-y1), slideRegion.scale)
                    crop_img = crop_img * (2. / 255) - 1.
                    crop_img = 0.2989 * crop_img[:, :, 0] + 0.5870 * crop_img[:, :, 1] + 0.1140 * crop_img[:, :, 2]

                    if crop_img.shape != (window_size,window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))
                    crop_img10x = cv2.resize(crop_img, (window_size // 2, window_size // 2))

                    crop_img = crop_img[None]
                    crop_img10x = crop_img10x[None]

                    out_queue.put(SlideRegion(
                        image=crop_img, image2=crop_img10x, sub_coords=sub_coords[i]
                    ))
                except Exception as e:
                    print(e)
                    out_queue.put(SlideRegion(
                        image=None, sub_coords=sub_coords[i]
                    ))
    return None

def detect_cells20x_microbes10x(slide, center_coords, cell_classifier=None, microbe_classifier=None,
                          patch_size=160, overlap=40, batch_size=128, num_workers=4):

    crop_queue = Queue()
    process_queue = Queue(batch_size*8)

    standard_mpp = 0.242042*2
    this_mpp = slide.mpp
    scale_ratio = standard_mpp/this_mpp

    lvl0_crop_size = int(patch_size*scale_ratio)
    lvl0_overlap = int(overlap*scale_ratio)
    lvl0_stride = lvl0_crop_size-lvl0_overlap

    bboxes = sampling_bboxes(center_coords, lvl0_stride, lvl0_crop_size, border=[0,0,slide.width, slide.height])
    num_bboxes = bboxes.shape[0]
    patch_bboxes_list = match_bboxes_with_pathces(bboxes=bboxes, crop_size=5120)

    for i in range(len(patch_bboxes_list)):
        patch_coord, bbox_coord = patch_bboxes_list[i]
        xs, ys =  patch_coord[0], patch_coord[1]
        w , h  = patch_coord[2]-patch_coord[0], patch_coord[3]-patch_coord[1]
        crop_queue.put(SlideRegion(location=(xs, ys), size=(w, h), scale=scale_ratio, sub_coords=bbox_coord))

    for i in range(num_workers):
        t = Thread(target=read_patch_worker_10x_20x, args=(slide, crop_queue, process_queue, patch_size))
        t.start()

    batch_data_list = []
    batch_data10x_list = []
    bbox_list, feat_list, cell_soft_list, microbe_soft_list = [], [], [], []
    with torch.no_grad():
        for i in range(num_bboxes):
            # print("process_queue = {}".format(process_queue.qsize()))
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)
                batch_data10x_list.append(slide_region.image2)
                bbox_list.append(slide_region.sub_coords)

            if len(batch_data_list) == batch_size or (i == num_bboxes - 1 and len(batch_data_list)>0):

                data_variable = torch.tensor(np.array(batch_data_list), dtype=torch.float, device=torch.device('cuda'))
                data_variable10x = torch.tensor(np.array(batch_data10x_list), dtype=torch.float, device=torch.device('cuda'))

                _, feat, cell_soft = cell_classifier(data_variable)
                feat_list.append(feat)
                # if len(feat_list)==1024:
                #     break
                cell_soft_list.append(cell_soft)
                if microbe_classifier is not None:
                    _, _, microbe_soft = microbe_classifier(data_variable10x)
                    microbe_soft_list.append(microbe_soft)
                batch_data_list = []
                batch_data10x_list = []

        bboxes = np.array(bbox_list)
        cell_soft = torch.cat(cell_soft_list, 0).cpu().numpy()
        microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy() if len(microbe_soft_list)>0 else np.empty_like(cell_soft, dtype=float)
        feat = torch.cat(feat_list,0).cpu().numpy()

    return bboxes, cell_soft, microbe_soft, feat

def detect_cells_microbes20x_scale1_bi_head(slide, center_coords, cell_classifier=None, microbe_classifier=None,
                          patch_size=160, overlap=40, batch_size=128, num_workers=4):

    crop_queue = Queue()
    process_queue = Queue(batch_size*8)

    standard_mpp = 0.242042*2
    this_mpp = slide.mpp
    scale_ratio = standard_mpp/this_mpp

    lvl0_crop_size = int(patch_size*scale_ratio)
    lvl0_overlap = int(overlap*scale_ratio)
    lvl0_stride = lvl0_crop_size-lvl0_overlap

    bboxes = sampling_bboxes(center_coords, lvl0_stride, lvl0_crop_size, border=[0,0,slide.width, slide.height])
    num_bboxes = bboxes.shape[0]
    patch_bboxes_list = match_bboxes_with_pathces(bboxes=bboxes, crop_size=5120)

    for i in range(len(patch_bboxes_list)):
        patch_coord, bbox_coord = patch_bboxes_list[i]
        xs, ys =  patch_coord[0], patch_coord[1]
        w , h  = patch_coord[2]-patch_coord[0], patch_coord[3]-patch_coord[1]
        crop_queue.put(SlideRegion(location=(xs, ys), size=(w, h), scale=scale_ratio, sub_coords=bbox_coord))

    for i in range(num_workers):
        t = Thread(target=read_patch_worker_20x_scale1, args=(slide, crop_queue, process_queue, patch_size))
        t.start()

    batch_data_list = []
    bbox_list, feat_list, cell_soft_list, microbe_soft_list = [], [], [], []

    with torch.no_grad():
        for i in range(num_bboxes):
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)
                bbox_list.append(slide_region.sub_coords)

            if len(batch_data_list) == batch_size or (i == num_bboxes - 1 and len(batch_data_list)>0):
                data_variable = torch.tensor(np.array(batch_data_list), dtype=torch.float, device=torch.device('cuda'))
                # import pdb;pdb.set_trace()
                cell_pred, microbe_pred, feat = cell_classifier(data_variable)
                cell_soft = torch.softmax(cell_pred, dim=1)
                microbe_soft = torch.softmax(microbe_pred, dim=1)
                feat_list.append(feat)
                cell_soft_list.append(cell_soft)
                microbe_soft_list.append(microbe_soft)

                batch_data_list = []
                # print(cell_soft,microbe_soft)
        bboxes = np.array(bbox_list)
        cell_soft = torch.cat(cell_soft_list, 0).cpu().numpy()
        microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy() if len(microbe_soft_list)>0 else np.empty_like(cell_soft, dtype=float)
        feat = torch.cat(feat_list,0).cpu().numpy()
    return bboxes, cell_soft, microbe_soft, feat