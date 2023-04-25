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

import numpy as np

from .utils import *
import torch

def detect_cells_microbes20x(slide, center_coords, cell_classifier=None, microbe_classifier=None,
                          patch_size=160, overlap=40, batch_size=128, num_workers=4):

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
                    xs, ys = math.ceil(x / slideRegion.scale), math.ceil(y / slideRegion.scale)
                    crop_img = image[ys:ys + window_size, xs:xs + window_size]

                    if crop_img.shape != (window_size, window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))

                    crop_img = crop_img[None]
                    out_queue.put(SlideRegion(
                        image=crop_img,
                        sub_coords=slideRegion.sub_coords[i] + np.array([patch_x, patch_y, patch_x, patch_y])
                    ))
            except Exception as e:  # read big patch failed
                print(e)
                sub_coords = slideRegion.sub_coords
                sub_coords += np.array([patch_x, patch_y, patch_x, patch_y])
                num_coords = sub_coords.shape[0]
                for i in range(num_coords):
                    try:
                        x1, y1, x2, y2 = sub_coords[i]
                        crop_img = slide.read((x1, x2), (x2 - x1, y2 - y1), slideRegion.scale)
                        crop_img = crop_img * (2. / 255) - 1.
                        crop_img = 0.2989 * crop_img[:, :, 0] + 0.5870 * crop_img[:, :, 1] + 0.1140 * crop_img[:, :, 2]

                        if crop_img.shape != (window_size, window_size):
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

def detect_cells_microbes20x_scale1(slide, center_coords, cell_classifier=None, microbe_classifier=None,
                          patch_size=160, overlap=40, batch_size=32, num_workers=4):
    def read_patch_worker_20x_scale1(slide, in_queue, out_queue, window_size=320):
        while not in_queue.empty():
            #
            slideRegion = in_queue.get()
            patch_x, patch_y = slideRegion.location

            try:
                image = slide.read(slideRegion.location, slideRegion.size, 1)

                for i in range(slideRegion.sub_coords.shape[0]):
                    xs, ys = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                    crop_img = image[ys:ys + int(window_size * slideRegion.scale),
                               xs:xs + int(window_size * slideRegion.scale)]

                    if crop_img.shape != (window_size, window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))

                    ##add
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                    crop_img = crop_img * (2. / 255) - 1.

                    crop_img = crop_img[None]

                    out_queue.put(SlideRegion(
                        image=crop_img,
                        sub_coords=slideRegion.sub_coords[i] + np.array([patch_x, patch_y, patch_x, patch_y])
                    ))
            except Exception as e:  # read big patch failed
                print(e)
                sub_coords = slideRegion.sub_coords
                sub_coords += np.array([patch_x, patch_y, patch_x, patch_y])
                num_coords = sub_coords.shape[0]
                for i in range(num_coords):
                    try:
                        x1, y1, x2, y2 = sub_coords[i]
                        crop_img = slide.read((x1, y1), (x2 - x1, y2 - y1), 1)
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

                        crop_img = crop_img * (2. / 255) - 1.

                        if crop_img.shape != (window_size, window_size):
                            crop_img = cv2.resize(crop_img, (window_size, window_size))
                        crop_img = crop_img[None]

                        out_queue.put(SlideRegion(
                            image=crop_img, sub_coords=sub_coords[i]
                        ))
                    except Exception as e:
                        print(e)
                        out_queue.put(SlideRegion(
                            image=None, image2=None, sub_coords=sub_coords[i]
                        ))
        return None

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
                # X, feat = cell_classifier(data_variable)
                # cell_soft = torch.softmax(X, dim=1)
                feat_list.append(feat)
                cell_soft_list.append(cell_soft)

                if microbe_classifier is not None:
                    _, _, microbe_soft = microbe_classifier(data_variable)
                    # microbe = microbe_classifier(data_variable)
                    # microbe_soft = torch.softmax(microbe, dim=1)
                    microbe_soft_list.append(microbe_soft)
                batch_data_list = []

        bboxes = np.array(bbox_list)
        cell_soft = torch.cat(cell_soft_list, 0).cpu().numpy()
        microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy() if len(microbe_soft_list)>0 else np.zeros_like(cell_soft, dtype=float)
        feat = torch.cat(feat_list,0).cpu().numpy()

    return bboxes, cell_soft, microbe_soft, feat

def detect_cells20x_microbes10x(slide, center_coords, cell_classifier=None, microbe_classifier=None,
                          patch_size=160*2, overlap=40*2, batch_size=32, num_workers=4):
    def read_patch_worker_10x_20x(slide, in_queue, out_queue, window_size=320):
        while not in_queue.empty():

            slideRegion = in_queue.get()
            patch_x, patch_y = slideRegion.location

            try:
                image = slide.read(slideRegion.location, slideRegion.size, slideRegion.scale)
                print(image.shape)
                image = image * (2. / 255) - 1.
                image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

                for i in range(slideRegion.sub_coords.shape[0]):
                    x, y = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                    xs, ys = math.ceil(x / slideRegion.scale), math.ceil(y / slideRegion.scale)
                    crop_img = image[ys:ys + window_size, xs:xs + window_size]

                    if crop_img.shape != (window_size, window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))

                    if microbe_classifier:
                        crop_img10x = cv2.resize(crop_img, (window_size // 2, window_size // 2))
                        crop_img10x = crop_img10x[None]
                    else:
                        crop_img10x = None

                    crop_img = crop_img[None]

                    out_queue.put(SlideRegion(
                        image=crop_img, image2=crop_img10x,
                        sub_coords=slideRegion.sub_coords[i] + np.array([patch_x, patch_y, patch_x, patch_y])
                    ))
            except Exception as e:  # read big patch failed
                print(e)
                sub_coords = slideRegion.sub_coords
                sub_coords += np.array([patch_x, patch_y, patch_x, patch_y])
                num_coords = sub_coords.shape[0]
                for i in range(num_coords):
                    try:
                        x1, y1, x2, y2 = sub_coords[i]
                        crop_img = slide.read((x1, x2), (x2 - x1, y2 - y1), slideRegion.scale)
                        crop_img = crop_img * (2. / 255) - 1.
                        crop_img = 0.2989 * crop_img[:, :, 0] + 0.5870 * crop_img[:, :, 1] + 0.1140 * crop_img[:, :, 2]

                        if crop_img.shape != (window_size, window_size):
                            crop_img = cv2.resize(crop_img, (window_size, window_size))

                        if microbe_classifier:
                            crop_img10x = cv2.resize(crop_img, (window_size // 2, window_size // 2))
                            crop_img10x = crop_img10x[None]
                        else:
                            crop_img10x = None
                        crop_img = crop_img[None]

                        out_queue.put(SlideRegion(
                            image=crop_img, image2=crop_img10x, sub_coords=sub_coords[i]
                        ))
                    except Exception as e:
                        print(e)
                        out_queue.put(SlideRegion(
                            image=None, sub_coords=sub_coords[i]
                        ))
        return None

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
                _, feat, cell_soft = cell_classifier(data_variable)
                feat_list.append(feat)
                cell_soft_list.append(cell_soft)
                if microbe_classifier is not None:
                    data_variable10x = torch.tensor(np.array(batch_data10x_list), dtype=torch.float,
                                                    device=torch.device('cuda'))
                    _, _, microbe_soft = microbe_classifier(data_variable10x)
                    microbe_soft_list.append(microbe_soft)
                batch_data_list = []
                batch_data10x_list = []

        bboxes = np.array(bbox_list)
        cell_soft = torch.cat(cell_soft_list, 0).cpu().numpy()
        microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy() if len(microbe_soft_list)>0 else np.empty_like(cell_soft, dtype=float)
        feat = torch.cat(feat_list,0).cpu().numpy()

    return bboxes, cell_soft, microbe_soft, feat

def detect_mix20x_scale1(slide, center_coords, cell_classifier=None, microbe_classifier=None,
                          patch_size=160, overlap=40, batch_size=128, num_workers=4):
    def read_patch_worker_20x_scale1_rgb(slide, in_queue, out_queue, window_size=320):
        while not in_queue.empty():
            #
            slideRegion = in_queue.get()
            patch_x, patch_y = slideRegion.location

            try:
                image = slide.read(slideRegion.location, slideRegion.size, 1)

                for i in range(slideRegion.sub_coords.shape[0]):
                    xs, ys = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                    crop_img = image[ys:ys + int(window_size * slideRegion.scale),
                               xs:xs + int(window_size * slideRegion.scale)]
                    if crop_img.shape[:2] != (window_size, window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))

                    crop_img = crop_img * (2. / 255) - 1.
                    crop_img = crop_img.transpose(2, 0, 1)
                    out_queue.put(SlideRegion(
                        image=crop_img,
                        sub_coords=slideRegion.sub_coords[i] + np.array([patch_x, patch_y, patch_x, patch_y])
                    ))
            except Exception as e:  # read big patch failed
                print(e)
                sub_coords = slideRegion.sub_coords
                sub_coords += np.array([patch_x, patch_y, patch_x, patch_y])
                num_coords = sub_coords.shape[0]
                for i in range(num_coords):
                    try:
                        x1, y1, x2, y2 = sub_coords[i]
                        crop_img = slide.read((x1, x2), (x2 - x1, y2 - y1), 1)

                        if crop_img.shape[:2] != (window_size, window_size):
                            crop_img = cv2.resize(crop_img, (window_size, window_size))

                        crop_img = crop_img * (2. / 255) - 1.
                        crop_img = crop_img.transpose(2, 0, 1)

                        out_queue.put(SlideRegion(
                            image=crop_img, sub_coords=sub_coords[i]
                        ))
                    except Exception as e:
                        print(e)
                        out_queue.put(SlideRegion(
                            image=None, image2=None, sub_coords=sub_coords[i]
                        ))
        return None

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
        t = Thread(target=read_patch_worker_20x_scale1_rgb, args=(slide, crop_queue, process_queue, patch_size))
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
                # data_variable = torch.cat(batch_data_list).cuda()
                # feature, cell_output, microbe_output = cell_classifier(data_variable)
                cell_output, feature = cell_classifier(data_variable)
                # import pdb; pdb.set_trace()
                cell_output = torch.softmax(cell_output, dim=1)
                # microbe_output = torch.zeros_like(cell_output)
                if microbe_classifier is not None:
                    microbe_output, _ = microbe_classifier(data_variable)
                    microbe_output = torch.softmax(microbe_output, dim=1)
                    microbe_soft_list.append(microbe_output)

                feat_list.append(feature)
                cell_soft_list.append(cell_output)

                batch_data_list = []

        bboxes = np.array(bbox_list)
        cell_soft = torch.cat(cell_soft_list, 0).cpu().numpy()
        microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy() if len(microbe_soft_list)>0 else np.zeros_like(cell_soft, dtype=float)
        feat = torch.cat(feat_list,0).cpu().numpy()
    return bboxes, cell_soft, microbe_soft, feat

def detect_cells_microbes20x_scale1_rgb_gray(slide, center_coords, cell_classifier=None, microbe_classifier=None,
                          patch_size=160, overlap=40, batch_size=32, num_workers=4):
    def read_patch_worker_20x_scale1_rgb_gray(slide, in_queue, out_queue, window_size=320):
        while not in_queue.empty():
            slideRegion = in_queue.get()
            patch_x, patch_y = slideRegion.location

            try:
                image = slide.read(slideRegion.location, slideRegion.size, 1)

                for i in range(slideRegion.sub_coords.shape[0]):
                    xs, ys = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                    crop_img = image[ys:ys + int(window_size * slideRegion.scale),
                               xs:xs + int(window_size * slideRegion.scale)]
                    if crop_img.shape[:2] != (window_size, window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))

                    if microbe_classifier:
                        rgb = crop_img * (2. / 255) - 1.
                        rgb = rgb.transpose(2, 0, 1)
                    else:
                        rgb = None

                    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                    gray = gray * (2. / 255) - 1.
                    gray = gray[None]

                    out_queue.put(SlideRegion(
                        image=gray, image2=rgb,
                        sub_coords=slideRegion.sub_coords[i] + np.array([patch_x, patch_y, patch_x, patch_y])
                    ))
            except Exception as e:  # read big patch failed
                print(e)
                sub_coords = slideRegion.sub_coords
                sub_coords += np.array([patch_x, patch_y, patch_x, patch_y])
                num_coords = sub_coords.shape[0]
                for i in range(num_coords):
                    try:
                        x1, y1, x2, y2 = sub_coords[i]
                        crop_img = slide.read((x1, x2), (x2 - x1, y2 - y1), 1)
                        if crop_img.shape != (window_size, window_size):
                            crop_img = cv2.resize(crop_img, (window_size, window_size))

                        if microbe_classifier:
                            rgb = crop_img * (2. / 255) - 1.
                            rgb = rgb.transpose(2, 0, 1)
                        else:
                            rgb = None

                        gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                        gray = gray * (2. / 255) - 1.
                        gray = gray[None]

                        out_queue.put(SlideRegion(
                            image=gray, image2=rgb, sub_coords=sub_coords[i]
                        ))
                    except Exception as e:
                        print(e)
                        out_queue.put(SlideRegion(
                            image=None, image2=None, sub_coords=sub_coords[i]
                        ))
        return None

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
        t = Thread(target=read_patch_worker_20x_scale1_rgb_gray, args=(slide, crop_queue, process_queue, patch_size))
        t.start()

    batch_gray_list, batch_rgb_list = [], []
    bbox_list, feat_list, cell_soft_list, microbe_soft_list = [], [], [], []

    with torch.no_grad():
        for i in range(num_bboxes):
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_gray_list.append(slide_region.image)
                batch_rgb_list.append(slide_region.image2)
                bbox_list.append(slide_region.sub_coords)

            if len(batch_gray_list) == batch_size or (i == num_bboxes - 1 and len(batch_gray_list)>0):
                data_variable = torch.tensor(np.array(batch_gray_list), dtype=torch.float, device=torch.device('cuda'))
                _, feat, cell_soft = cell_classifier(data_variable)
                # X, feat = cell_classifier(data_variable)
                # cell_soft = torch.softmax(X, dim=1)
                feat_list.append(feat)
                cell_soft_list.append(cell_soft)

                if microbe_classifier is not None:
                    data_variable = torch.tensor(np.array(batch_rgb_list), dtype=torch.float,
                                                 device=torch.device('cuda'))

                    # _, _, microbe_soft = microbe_classifier(data_variable)
                    microbe = microbe_classifier(data_variable)
                    microbe_soft = torch.softmax(microbe, dim=1)
                    microbe_soft_list.append(microbe_soft)
                batch_gray_list, batch_rgb_list = [], []

        bboxes = np.array(bbox_list)
        cell_soft = torch.cat(cell_soft_list, 0).cpu().numpy()
        microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy() if len(microbe_soft_list)>0 else np.zeros_like(cell_soft, dtype=float)
        feat = torch.cat(feat_list,0).cpu().numpy()

    return bboxes, cell_soft, microbe_soft, feat


def detect_mix20x_scale1_qc(slide, center_coords, cell_classifier=None, microbe_classifier=None, qc_model=None,
                          patch_size=160, overlap=40, batch_size=128, num_workers=4):
    def read_patch_worker_20x_scale1_rgb(slide, in_queue, out_queue, window_size=320):
        while not in_queue.empty():
            #
            slideRegion = in_queue.get()
            patch_x, patch_y = slideRegion.location

            try:
                image = slide.read(slideRegion.location, slideRegion.size, 1)

                for i in range(slideRegion.sub_coords.shape[0]):
                    xs, ys = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                    crop_img = image[ys:ys + int(window_size * slideRegion.scale),
                               xs:xs + int(window_size * slideRegion.scale)]
                    if crop_img.shape[:2] != (window_size, window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))

                    crop_img = crop_img * (2. / 255) - 1.
                    crop_img = crop_img.transpose(2, 0, 1)
                    out_queue.put(SlideRegion(
                        image=crop_img,
                        sub_coords=slideRegion.sub_coords[i] + np.array([patch_x, patch_y, patch_x, patch_y])
                    ))
            except Exception as e:  # read big patch failed
                print(e)
                sub_coords = slideRegion.sub_coords
                sub_coords += np.array([patch_x, patch_y, patch_x, patch_y])
                num_coords = sub_coords.shape[0]
                for i in range(num_coords):
                    try:
                        x1, y1, x2, y2 = sub_coords[i]
                        crop_img = slide.read((x1, x2), (x2 - x1, y2 - y1), 1)

                        if crop_img.shape[:2] != (window_size, window_size):
                            crop_img = cv2.resize(crop_img, (window_size, window_size))

                        crop_img = crop_img * (2. / 255) - 1.
                        crop_img = crop_img.transpose(2, 0, 1)

                        out_queue.put(SlideRegion(
                            image=crop_img, sub_coords=sub_coords[i]
                        ))
                    except Exception as e:
                        print(e)
                        out_queue.put(SlideRegion(
                            image=None, image2=None, sub_coords=sub_coords[i]
                        ))
        return None
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
        t = Thread(target=read_patch_worker_20x_scale1_rgb, args=(slide, crop_queue, process_queue, patch_size))
        t.start()

    batch_data_list = []
    bbox_list, feat_list, cell_soft_list, microbe_soft_list, qc_soft_list = [], [], [], [], []
    this_batch_bbox = []
    with torch.no_grad():
        for i in range(num_bboxes):
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)
                this_batch_bbox.append(slide_region.sub_coords)

            if len(batch_data_list) == batch_size or (i == num_bboxes - 1 and len(batch_data_list)>0):
                data_variable = torch.tensor(np.array(batch_data_list), dtype=torch.float, device=torch.device('cuda'))

                if qc_model is not None:
                    qc_output = qc_model(data_variable)
                    qc_output = torch.softmax(qc_output, dim=1)
                    qc_soft_list.append(qc_output)
                    data_variable = data_variable[qc_output.argmax(1)==1]
                    bbox_list += list(np.array(this_batch_bbox)[(qc_output.argmax(1)==1).cpu().numpy()])
                else:
                    bbox_list += this_batch_bbox

                if data_variable.size()[0] > 0:
                    cell_output, feature = cell_classifier(data_variable)
                    cell_output = torch.softmax(cell_output, dim=1)
                    feat_list.append(feature)
                    cell_soft_list.append(cell_output)

                    if microbe_classifier is not None:
                        microbe_output, _ = microbe_classifier(data_variable)
                        microbe_output = torch.softmax(microbe_output, dim=1)
                        microbe_soft_list.append(microbe_output)

                batch_data_list = []
                this_batch_bbox = []

        bboxes = np.array(bbox_list)
        cell_soft = torch.cat(cell_soft_list, 0).cpu().numpy()
        microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy() if len(microbe_soft_list)>0 else np.zeros_like(cell_soft, dtype=float)
        qc_soft = torch.cat(qc_soft_list, 0).cpu().numpy() if len(qc_soft_list)>0 else np.ones((num_bboxes, 2), dtype=float)
        feat = torch.cat(feat_list,0).cpu().numpy()

    return bboxes, cell_soft, microbe_soft, feat, qc_soft


def detect_mix20x_scale1_qc2(slide, center_coords, cell_classifier=None, microbe_classifier=None, qc_model=None, microbe_qc=None,
                          patch_size=160, overlap=40, batch_size=128, num_workers=4):
    def read_patch_worker_20x_scale1_rgb(slide, in_queue, out_queue, window_size=320):
        while not in_queue.empty():
            #
            slideRegion = in_queue.get()
            patch_x, patch_y = slideRegion.location

            try:
                image = slide.read(slideRegion.location, slideRegion.size, 1)

                for i in range(slideRegion.sub_coords.shape[0]):
                    xs, ys = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                    crop_img = image[ys:ys + int(window_size * slideRegion.scale),
                               xs:xs + int(window_size * slideRegion.scale)]
                    if crop_img.shape[:2] != (window_size, window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))

                    crop_img = crop_img * (2. / 255) - 1.
                    crop_img = crop_img.transpose(2, 0, 1)
                    out_queue.put(SlideRegion(
                        image=crop_img,
                        sub_coords=slideRegion.sub_coords[i] + np.array([patch_x, patch_y, patch_x, patch_y])
                    ))
            except Exception as e:  # read big patch failed
                print(e)
                sub_coords = slideRegion.sub_coords
                sub_coords += np.array([patch_x, patch_y, patch_x, patch_y])
                num_coords = sub_coords.shape[0]
                for i in range(num_coords):
                    try:
                        x1, y1, x2, y2 = sub_coords[i]
                        crop_img = slide.read((x1, x2), (x2 - x1, y2 - y1), 1)

                        if crop_img.shape[:2] != (window_size, window_size):
                            crop_img = cv2.resize(crop_img, (window_size, window_size))

                        crop_img = crop_img * (2. / 255) - 1.
                        crop_img = crop_img.transpose(2, 0, 1)

                        out_queue.put(SlideRegion(
                            image=crop_img, sub_coords=sub_coords[i]
                        ))
                    except Exception as e:
                        print(e)
                        out_queue.put(SlideRegion(
                            image=None, image2=None, sub_coords=sub_coords[i]
                        ))
        return None
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
        t = Thread(target=read_patch_worker_20x_scale1_rgb, args=(slide, crop_queue, process_queue, patch_size))
        t.start()

    batch_data_list = []
    bbox_list, feat_list, cell_soft_list, microbe_soft_list, qc_soft_list = [], [], [], [], []
    microbe_qc_list = []
    this_batch_bbox = []
    with torch.no_grad():
        for i in range(num_bboxes):
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)
                this_batch_bbox.append(slide_region.sub_coords)

            if len(batch_data_list) == batch_size or (i == num_bboxes - 1 and len(batch_data_list)>0):
                data_variable = torch.tensor(np.array(batch_data_list), dtype=torch.float, device=torch.device('cuda'))

                if qc_model is not None:
                    qc_output = qc_model(data_variable)
                    qc_output = torch.softmax(qc_output, dim=1)
                    qc_soft_list.append(qc_output)
                    data_variable = data_variable[qc_output.argmax(1)==1]
                    bbox_list += list(np.array(this_batch_bbox)[(qc_output.argmax(1)==1).cpu().numpy()])
                else:
                    bbox_list += this_batch_bbox

                if data_variable.size()[0] > 0:
                    cell_output, feature = cell_classifier(data_variable)
                    cell_output = torch.softmax(cell_output, dim=1)
                    feat_list.append(feature)
                    cell_soft_list.append(cell_output)

                    if microbe_classifier is not None:
                        if microbe_qc is not None:
                            microbe_qc_output = microbe_qc(data_variable)
                            microbe_qc_output = torch.softmax(microbe_qc_output, dim=1)
                            # microbe_qc_list.append(microbe_qc_output)
                            # microbe_data = data_variable[microbe_qc_output.argmax(1)==1]
                            # bbox_list += list(np.array(this_batch_bbox)[(microbe_qc_output.argmax(1) == 1).cpu().numpy()])
                            microbe_output, _ = microbe_classifier(data_variable)
                            microbe_output = torch.softmax(microbe_output, dim=1)
                            microbe_output[microbe_qc_output.argmax(1)==0, 0] = 1

                            microbe_soft_list.append(microbe_output)
                        else:
                            microbe_output, _ = microbe_classifier(data_variable)
                            microbe_output = torch.softmax(microbe_output, dim=1)
                            microbe_soft_list.append(microbe_output)
                    #
                    # if microbe_classifier is not None:
                    #     microbe_output, _ = microbe_classifier(data_variable)
                    #     microbe_output = torch.softmax(microbe_output, dim=1)
                    #     microbe_soft_list.append(microbe_output)

                batch_data_list = []
                this_batch_bbox = []

        bboxes = np.array(bbox_list)
        # microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy()
        cell_soft = torch.cat(cell_soft_list, 0).cpu().numpy()
        microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy() if len(microbe_soft_list)>0 else np.zeros_like(cell_soft, dtype=float)
        qc_soft = torch.cat(qc_soft_list, 0).cpu().numpy() if len(qc_soft_list)>0 else np.ones((num_bboxes, 2), dtype=float)
        feat = torch.cat(feat_list,0).cpu().numpy()

    return bboxes, cell_soft, microbe_soft, feat, qc_soft


def detect_mix20x_scale1_qc3(slide, center_coords, cell_classifier=None, microbe_classifier=None, qc_model=None, microbe_qc=None,
                          patch_size=160, overlap=40, batch_size=128, num_workers=4):
    def read_patch_worker_20x_scale1_rgb(slide, in_queue, out_queue, window_size=320):
        while not in_queue.empty():
            slideRegion = in_queue.get()
            patch_x, patch_y = slideRegion.location

            try:
                image = slide.read(slideRegion.location, slideRegion.size, 1)

                for i in range(slideRegion.sub_coords.shape[0]):
                    xs, ys = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                    crop_img = image[ys:ys + int(window_size * slideRegion.scale),
                               xs:xs + int(window_size * slideRegion.scale)]
                    if crop_img.shape[:2] != (window_size, window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))

                    crop_img = crop_img * (2. / 255) - 1.
                    crop_img = crop_img.transpose(2, 0, 1)
                    out_queue.put(SlideRegion(
                        image=crop_img,
                        sub_coords=slideRegion.sub_coords[i] + np.array([patch_x, patch_y, patch_x, patch_y])
                    ))
            except Exception as e:  # read big patch failed
                print(e)
                sub_coords = slideRegion.sub_coords
                sub_coords += np.array([patch_x, patch_y, patch_x, patch_y])
                num_coords = sub_coords.shape[0]
                for i in range(num_coords):
                    try:
                        x1, y1, x2, y2 = sub_coords[i]
                        crop_img = slide.read((x1, x2), (x2 - x1, y2 - y1), 1)

                        if crop_img.shape[:2] != (window_size, window_size):
                            crop_img = cv2.resize(crop_img, (window_size, window_size))

                        crop_img = crop_img * (2. / 255) - 1.
                        crop_img = crop_img.transpose(2, 0, 1)

                        out_queue.put(SlideRegion(
                            image=crop_img, sub_coords=sub_coords[i]
                        ))
                    except Exception as e:
                        print(e)
                        out_queue.put(SlideRegion(
                            image=None, image2=None, sub_coords=sub_coords[i]
                        ))
        return None
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
        t = Thread(target=read_patch_worker_20x_scale1_rgb, args=(slide, crop_queue, process_queue, patch_size))
        t.start()

    batch_data_list = []
    bbox_list, feat_list, cell_soft_list, microbe_soft_list, qc_soft_list = [], [], [], [], []
    this_batch_bbox = []
    with torch.no_grad():
        for i in range(num_bboxes):
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)
                this_batch_bbox.append(slide_region.sub_coords)

            if len(batch_data_list) == batch_size or (i == num_bboxes - 1 and len(batch_data_list)>0):
                data_variable = torch.tensor(np.array(batch_data_list), dtype=torch.float, device=torch.device('cuda'))

                if qc_model is not None:
                    qc_output = qc_model(data_variable)
                    qc_output = torch.softmax(qc_output, dim=1)
                    qc_soft_list.append(qc_output)
                    data_variable = data_variable[qc_output.argmax(1)==1]
                    bbox_list += list(np.array(this_batch_bbox)[(qc_output.argmax(1)==1).cpu().numpy()])
                else:
                    bbox_list += this_batch_bbox

                if data_variable.size()[0] > 0:
                    cell_output, feature = cell_classifier(data_variable)
                    cell_output = torch.softmax(cell_output, dim=1)
                    feat_list.append(feature)
                    cell_soft_list.append(cell_output)

                    if microbe_classifier is not None:
                        if microbe_qc is not None:
                            microbe_qc_output = microbe_qc(data_variable)
                            microbe_qc_output = torch.softmax(microbe_qc_output, dim=1)

                            microbe_output= microbe_classifier(data_variable)
                            microbe_output = torch.softmax(microbe_output, dim=1)
                            microbe_output[microbe_qc_output.argmax(1)==0, 0] = 1

                            microbe_soft_list.append(microbe_output)
                        else:
                            microbe_output= microbe_classifier(data_variable)
                            microbe_output = torch.softmax(microbe_output, dim=1)
                            microbe_soft_list.append(microbe_output)


                batch_data_list = []
                this_batch_bbox = []

        bboxes = np.array(bbox_list)
        cell_soft = torch.cat(cell_soft_list, 0).cpu().numpy()
        microbe_soft = torch.cat(microbe_soft_list, 0).cpu().numpy() if len(microbe_soft_list)>0 else np.zeros_like(cell_soft, dtype=float)
        qc_soft = torch.cat(qc_soft_list, 0).cpu().numpy() if len(qc_soft_list)>0 else np.ones((num_bboxes, 2), dtype=float)
        feat = torch.cat(feat_list,0).cpu().numpy()

    return bboxes, cell_soft, microbe_soft, feat, qc_soft