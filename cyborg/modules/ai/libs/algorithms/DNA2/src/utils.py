#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 17:35
# @Author  : Can Cui
# @File    : utils.py
# @Software: PyCharm
# @Comment:

import numpy as np
import torch
from skimage.feature import peak_local_max
import cv2
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class SlideRegion:
    def __init__(self, location=(0,0), size=(0,0), scale=1.0,
                 cell_img = None, microbe_img = None, nuclei_img=None,
                 image=None, ori_image=None, image2=None, sub_coords=None):
        self.location = location
        self.size = size
        self.scale = scale
        self.image = image
        self.ori_image = ori_image
        self.image2 = image2
        self.sub_coords = sub_coords
        self.cell_img = cell_img
        self.microbe_img = microbe_img
        self.nuclei_img = nuclei_img


def split_image(image, patch_size, overlap):
    h, w = image.shape[0:2]
    stride = patch_size - overlap
    patch_list = []
    num_y, num_x = 0, 0
    # import pdb; pdb.set_trace()

    for y in range(0, h, stride):
        num_x = 0
        for x in range(0, w, stride):
            crop_img = image[y:y + patch_size, x:x + patch_size, :]
            crop_h, crop_w = crop_img.shape[0:2]
            pad_h, pad_w = patch_size - crop_h, patch_size - crop_w

            if pad_h > 0 or pad_w > 0:
                crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')

            patch_list.append(crop_img)
            num_x += 1
        num_y += 1
    patch_image = np.array(patch_list)

    return patch_image, num_y, num_x

def split_image_cuda(image, patch_size, overlap):

    h,w = image.size()[0:2]
    stride = patch_size - overlap
    patch_list = []
    num_y, num_x = 0,0

    for y in range(0, h, stride):
        num_x = 0
        for x in range(0, w, stride):
            crop_img = image[y:y+patch_size, x:x+patch_size, :]
            crop_h, crop_w = crop_img.size()[0:2]
            pad_h, pad_w = patch_size-crop_h, patch_size-crop_w
            crop_img = crop_img.permute(2,0,1).unsqueeze(0)
            if pad_h>0 or pad_w>0:
                # import pdb; pdb.set_trace()
                crop_img = torch.nn.functional.pad(crop_img, (0, pad_w ,0, pad_h), 'constant', torch.mean(crop_img).cpu().item())
                # pad_func = torch.nn.ReflectionPad2d((0,pad_w, 0, pad_h))
                # print(crop_img.size())
                # crop_img = pad_func(crop_img)
                # print(crop_img.size())
            patch_list.append(crop_img)
            num_x+=1
        num_y+=1
    patch_image = torch.cat(patch_list)
    return patch_image, num_y, num_x

def reconstruct_mask(masks, patch_size, overlap, num_y, num_x):
    num_channel = masks.shape[1]
    stride = patch_size - overlap
    mask_h, mask_w = patch_size + (num_y - 1) * stride, patch_size + (num_x - 1) * stride
    result_mask = np.zeros((num_channel, mask_h, mask_w))
    mask_count = np.zeros((mask_h, mask_w, 1), dtype=np.uint8)
    for y in range(num_y):
        for x in range(num_x):
            i = y * num_x + x

            ys, ye = y * stride, y * stride + patch_size
            xs, xe = x * stride, x * stride + patch_size

            result_mask[:, ys:ye, xs:xe] += masks[i]
            mask_count[ys:ye, xs:xe, :] += 1
    result_mask = result_mask.transpose(1, 2, 0)
    result_mask /= mask_count
    return result_mask

def reconstruct_mask_cuda(masks,  patch_size, overlap, num_y, num_x):
    num_channel = masks.shape[1]
    stride = patch_size - overlap
    mask_h, mask_w = patch_size+(num_y-1)*stride, patch_size+(num_x-1)*stride
    result_mask = torch.zeros((num_channel, mask_h, mask_w)).cuda()
    mask_count = torch.zeros((mask_h, mask_w, 1)).cuda()

    for y in range(num_y):
        for x in range(num_x):
            i = y*num_x + x
            ys, ye = y*stride, y*stride+patch_size
            xs, xe = x*stride, x*stride+patch_size
            result_mask[:, ys:ye, xs:xe] += masks[i]
            mask_count[ys:ye, xs:xe, :] += 1
    result_mask = result_mask.permute(1,2,0)
    result_mask /= mask_count
    return result_mask

def generate_result_mask(image, net, patch_size=512, overlap=0, batch_size=4):
    img_h, img_w = image.shape[0:2]
    patch_imgs, num_y, num_x = split_image(image, patch_size, overlap)
    num_patches = patch_imgs.shape[0]
    patch_imgs = patch_imgs.transpose((0, 3, 1, 2))
    patch_imgs = patch_imgs * (2. / 255) - 1.
    # patch_imgs = patch_imgs /255
    results = []
    for i in range(0, num_patches, batch_size):
        this_batch = patch_imgs[i:i + batch_size]
        with torch.no_grad():
            data_variable = torch.from_numpy(this_batch).float().cuda()
            result = net(data_variable)
            results.append(result.cpu().numpy())

    results = np.concatenate(results)
    result_masks = reconstruct_mask(results, patch_size, overlap, num_y, num_x)
    result_masks = result_masks[0:img_h, 0:img_w, :]
    return result_masks

def generate_result_mask_cuda(image, net, patch_size=512, overlap=128, batch_size=16):
    with torch.no_grad():
        img_tensor = torch.from_numpy(image).float().cuda()
        img_h, img_w = img_tensor.size()[0:2]
        patch_tensor, num_y, num_x = split_image_cuda(img_tensor, patch_size,  overlap)
        num_patches = patch_tensor.size()[0]
        # patch_tensor = patch_tensor * (2. / 255) - 1.
        # patch_tensor = patch_tensor/255
        results = []

        for i in range(0, num_patches, batch_size):
            this_batch = patch_tensor[i:i + batch_size]
            result = net(this_batch)
            sigmoid_result = torch.sigmoid(result)
            results.append(sigmoid_result)

        results = torch.cat(results)
        result_masks = reconstruct_mask_cuda(results, patch_size, overlap, num_y, num_x)
        result_masks = result_masks[0:img_h, 0:img_w, :]
        return result_masks

def plot_dna(di, areas, save_dir='', abnormal_high_thres=2.5, abnormal_low_thres=2.3):

    os.makedirs(save_dir, exist_ok=True)
    # di_round = np.round(di, 1)
    di_round = di
    f, ax = plt.subplots(figsize=(4, 4))
    sns.set_theme(style="ticks")
    df1 = pd.DataFrame({'DNA_Index': di_round})
    mean = np.round(np.mean(di), 3)
    std = np.round(np.std(di), 3)
    cv = round(std / mean, 3)
    cnt = di.shape[0]
    left_position = 0.50
    top_position = 0.9
    margin = 0.07
    image_dpi = 200
    # print(df1)
    ax = sns.histplot(data=df1, x='DNA_Index', binwidth=0.04, linewidth=0.04, color='g')
    # ax.text(left_position, top_position, f"MEAN {mean}", transform=ax.transAxes, color='r', fontsize='x-large')
    # ax.text(left_position, top_position - margin, f"SD       {std}", transform=ax.transAxes, color='black',
    #         fontsize='x-large')
    # ax.text(left_position, top_position - 2 * margin, f"CV       {cv}", transform=ax.transAxes, color='black',
    #         fontsize='x-large')
    # ax.text(left_position, top_position - 3 * margin, f"Cnt      {cnt}", transform=ax.transAxes, color='black',
    #         fontsize='x-large')

    # ax.set(title='DNA Index', ylabel='Count', xlabel='')
    ax.set(title='', xlabel='DNA Index', ylabel='Count')
    ax.set_xticks(range(np.max(di).astype(int)))
    plt.savefig(os.path.join(save_dir, 'dna_index.png'), dpi=image_dpi, bbox_inches='tight')


    f, ax = plt.subplots(figsize=(4, 4))
    sns.set_theme(style="ticks")

    aux_array = np.ones_like(di)
    aux_array[np.where(np.logical_and(di > abnormal_low_thres, di < abnormal_high_thres))[0]] = 2
    aux_array[np.where(di >= abnormal_high_thres)[0]] = 4

    df1 = pd.DataFrame({'DNA_Index': di, 'Area': areas, 'aux': aux_array})
    markers = {1: ".", 2: "p", 4: 'X'}

    sns.color_palette("pastel")

    sns.scatterplot(x="DNA_Index", y="Area",
                    hue='aux',
                    style='aux',
                    sizes=(1, 8),
                    linewidth=0.0,
                    legend=False,
                    palette={1: 'green', 2: 'orange', 4: 'red'},
                    markers=markers,
                    alpha=0.3,
                    data=df1, ax=ax)

    # ax.set(title='DNA Index vs Area', ylabel='Area', xlabel='DNA Index')
    ax.set(title='', ylabel='Area', xlabel='DNA Index')
    ax.set_xticks(range(np.max(di).astype(int)))
    plt.savefig(os.path.join(save_dir,'scatterplot.png'), dpi=image_dpi, bbox_inches='tight')

    return True


# def get_coordinate(voting_map, min_len=10):
#     coordinates = peak_local_max(voting_map, min_distance=min_len)  # N by 2
#     if coordinates.size == 0:
#         coordinates = None  # np.asarray([])
#     else:
#         coordinates = coordinates[:, ::-1]
#     return coordinates


def get_coordinate( voting_map, min_len=5):
    #abs_val = max( threshhold*np.max(voting_map[:]),  abs_threshold)
    # abs_val = max( threshhold*0.95,  abs_threshold)
    # voting_map[ voting_map < abs_val ] = 0
    try:
        import cupy
        from cupyimg.skimage.feature import peak_local_max as peak_local_max_gpu
        cupy_array =  cupy.asarray(voting_map)
        coordinates = peak_local_max_gpu(cupy_array, min_distance= min_len,exclude_border=False) # N by 2
        coordinates = coordinates.get()
    except:
        coordinates = peak_local_max(voting_map, min_distance=min_len, exclude_border=False)

    if coordinates.size == 0:
        coordinates = None  # np.asarray([])
    else:
        coordinates = coordinates[:, ::-1]
    return coordinates


def post_process_mask(masks, threshold=0.1):
    masks[masks < max(threshold * np.max(masks), 0.1)] = 0
    bboxes = get_coordinate(masks, min_len=15)
    if bboxes is None:
        bboxes = np.empty((0,2), dtype=int)

    return bboxes

def draw_center(img, center):
    img = img.copy()
    if center is None:
        return  img
    # img = cv2.resize(img, fx=0.25, fy=0.25, dsize=None)
    num_center = center.shape[0]

    radius = 3
    thickness = -1

    for i in range(num_center):
        img = cv2.circle(img, (center[i,0], center[i,1]), radius,(255,0,0), thickness)
    return img

def draw_rect(img, bboxes):
    # bboxes = N*4  [xmin, ymin, xmax, ymax]
    img = img.copy()
    num_bbox = bboxes.shape[0]

    for i in range(num_bbox):
        xmin, ymin, xmax, ymax = bboxes[i]
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
    return img

def sampling_bboxes(center_coords, stride, patch_size, border=[]):
    xmin, ymin, xmax, ymax = border
    bin_center_coords = np.floor(center_coords/stride)
    unique_bin_center_coords, inverse_idx = np.unique(bin_center_coords, return_inverse=True, axis=0)
    num_bin = unique_bin_center_coords.shape[0]

    for i in range(num_bin):
        coords_in_bin = center_coords[np.where(inverse_idx==i)[0], :]
        unique_bin_center_coords[i] = np.mean(coords_in_bin, axis=0)
    bboxes_start = unique_bin_center_coords - patch_size/2
    bboxes_start = np.clip(bboxes_start, [xmin, ymin], [xmax-1, ymax-1])
    bboxes_end = bboxes_start + patch_size
    overflow =  bboxes_end-[xmax-1, ymax-1]
    overflow[overflow<0] = 0
    bboxes_start-=overflow
    bboxes_end-=overflow

    bboxes = np.column_stack([bboxes_start, bboxes_end])

    return bboxes.astype(np.int)

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

def get_testing_feat(pos_ratio, logits, feat, testing_num=300, use_mixed=False):
    feat = np.squeeze(feat)
    if use_mixed:
        mix_feat = np.concatenate([feat, logits], axis=1)
    else:
        mix_feat = feat

    total_ind = np.array(range(0, logits.shape[0]))
    chosen_total_ind_ = total_ind[0:testing_num]
    chosen_total_ind_ = chosen_total_ind_.reshape((chosen_total_ind_.shape[0],))
    chosen_feat = mix_feat[chosen_total_ind_]
    pos_ratio = np.asarray([pos_ratio])
    return chosen_feat[None], pos_ratio


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

