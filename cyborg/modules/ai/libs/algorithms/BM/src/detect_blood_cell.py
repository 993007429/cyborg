#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 14:46
# @Author  : Can Cui
# @File    : detect_blood_cell.py
# @Software: PyCharm
import math
import os
from queue import  Queue
from threading import Thread
from .utils import SlideRegion
import cv2
import numpy as np
import torch
import torchvision
from shapely.geometry import Polygon, box
from traceback import format_exc


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

@torch.no_grad()
def non_max_suppression_nuclei(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, multi_label=False, max_det=5000):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
    return output


@torch.no_grad()
def nuclei_det(slide, model, is_half, standard_mpp, patch_size=1024, crop_size=(1024,1024), overlap=128, num_workers=4, batch_size=4, x_coords=[], y_coords=[]):
    def read_region_worker(slide, in_queue, out_queue):
        while not in_queue.empty():
            slideRegion = in_queue.get()
            try:
                image = slide.read(slideRegion.location, slideRegion.size, slideRegion.scale)
                image = np.ascontiguousarray(image).astype(np.uint8)
                # image = image[:patch_size, :patch_size,:]
                h, w = image.shape[0:2]
                # print(image.shape)
                pad_h, pad_w = patch_size - h, patch_size-w
                if pad_h > 0 and pad_w > 0:
                    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
                if image.shape[0:2] != (patch_size, patch_size):
                    image = cv2.resize(image, (patch_size, patch_size))

                out_queue.put(SlideRegion(
                    image=image, location=slideRegion.location
                ))
            except Exception as e:  # read big patch failed
                print(e)
                print(format_exc())
                out_queue.put(SlideRegion(
                    image=None, location=slideRegion.location
                ))
        return None


    slide_mpp = slide.mpp if slide.mpp else standard_mpp
    scale_ratio = max(1, round(standard_mpp/slide_mpp))
    # scale_ratio = standard_mpp/slide_mpp
    lvl0_crop_size =  (math.ceil(crop_size[0])*scale_ratio, math.ceil(crop_size[1])*scale_ratio)
    # print(lvl0_crop_size)
    stride = (math.ceil((crop_size[0]-overlap)*scale_ratio), math.ceil((crop_size[1]-overlap)*scale_ratio))
    fx, fy = crop_size[0]/patch_size, crop_size[1]/patch_size

    if len(x_coords)>0:
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))
        disc_ploy = Polygon(zip(x_coords, y_coords))
    else:
        xmin, xmax = 0, slide.width
        ymin, ymax = 0, slide.height
        disc_ploy = None
    crop_queue = Queue()
    process_queue = Queue(batch_size*4)

    for x in range(xmin, xmax, stride[0]):
        for y in range(ymin, ymax, stride[1]):
            if disc_ploy is not None:
                tile_box = box(x, y, x+lvl0_crop_size[0], y+lvl0_crop_size[1])
                if not disc_ploy.intersects(tile_box):
                    continue
            crop_queue.put(SlideRegion(location=(x, y), size=(lvl0_crop_size[0], lvl0_crop_size[1]), scale=scale_ratio))
    num_patch = crop_queue.qsize()

    for i in range(num_workers):
        t = Thread(target=read_region_worker ,args=(slide, crop_queue, process_queue))
        t.start()


    bboxes_list, scores_list, labels_list = [], [], []

    batch_data_list = []
    coords_shift_list = []

    for i in range(num_patch):
        slide_region = process_queue.get()
        if slide_region.image is not None:
            batch_data_list.append(slide_region.image)
            coords_shift_list.append(slide_region.location)

        if len(batch_data_list) == batch_size or (i == num_patch - 1 and len(batch_data_list) > 0):
            if is_half:
                data_variable = torch.as_tensor(np.array(batch_data_list), dtype=torch.half, device=torch.device('cuda'))
            else:
                data_variable = torch.as_tensor(np.array(batch_data_list), dtype=torch.float,
                                                device=torch.device('cuda'))

            data_variable = data_variable/255.
            data_variable = data_variable.permute((0,3,1,2))

            out = model(data_variable)[0]
            preds = non_max_suppression_nuclei(out, conf_thres=0.05, iou_thres=0.6, agnostic=True, max_det=1000)

            for idx, (pred, coords_shift) in enumerate(zip(preds, coords_shift_list)):
                if len(pred)>0:
                    bboxes, scores, labels = pred[:, :4].int().cpu().numpy(), pred[:, 4].cpu().numpy(), pred[:,5].cpu().numpy().astype(int)
                    bboxes = bboxes*scale_ratio*np.array([fx, fy, fx, fy]) + np.array([coords_shift, coords_shift]).flatten()
                    bboxes_list.append(bboxes)
                    scores_list.append(scores)
                    labels_list.append(labels)

            batch_data_list = []
            coords_shift_list = []

    if len(labels_list)>0:
        labels_np = np.concatenate(labels_list, 0)
        scores_np = np.concatenate(scores_list, 0)
        bboxes_np = np.concatenate(bboxes_list, 0)
        pick_idx = torchvision.ops.nms(torch.from_numpy(bboxes_np).float().cuda(), torch.from_numpy(scores_np).float().cuda(), 0.3).cpu().numpy()
        labels_np = labels_np[pick_idx]
        bboxes_np = bboxes_np[pick_idx]
        scores_np = scores_np[pick_idx]
    else:
        labels_np = np.empty((0,), dtype=np.float)
        scores_np = np.empty((0,), dtype=np.float)
        bboxes_np = np.empty((0,4), dtype=np.int32)

    return bboxes_np.astype(int), scores_np.round(4), labels_np.astype(int)
