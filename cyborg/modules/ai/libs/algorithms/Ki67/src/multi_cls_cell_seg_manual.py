#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:44:26 2020

@author: af2o
"""

import os
import json

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max

from .seg_tissue_area import find_tissue_countours
from .utils import split_patches


def del_duplicate(outputs_points, outputs_scores, interval):
    n = len(outputs_points)
    fused = np.full(n, False)
    filtered_points = []
    filtered_labels = []
    for i, point in enumerate(outputs_points):
        if fused[i] == False:
            distance = np.linalg.norm(point - outputs_points, 2, axis=1)
            distance_bool = np.where((distance < interval))[0]
            max_score = 0
            for index in distance_bool:
                fused[index] = True
                if np.max(outputs_scores[index]) > max_score:
                    max_score = np.max(outputs_scores[index])
                    max_index = index
            filtered_points.append(outputs_points[max_index])
            filtered_labels.append(np.argmax(outputs_scores[max_index]))
    return np.array(filtered_points), np.array(filtered_labels)


def split_image(image, patch_size, overlap):
    h, w = image.shape[0:2]
    stride = patch_size - overlap
    patch_list = []
    num_y, num_x = 0, 0

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


def split_image_P2P(image, coord_range, patch_size, overlap):
    stride = patch_size - overlap
    patch_list = []
    coord_list = []
    for x in range(coord_range[0], coord_range[2], stride):
        for y in range(coord_range[1], coord_range[3], stride):
            coord_list.append(np.array([x, y]))
    return coord_list


def reconstruct_mask(masks, patch_size, overlap, num_y, num_x):
    num_channel = masks.shape[1]
    stride = patch_size - overlap
    mask_h, mask_w = patch_size + (num_y - 1) * stride, patch_size + (num_x - 1) * stride
    result_mask = np.zeros((num_channel, mask_h, mask_w))
    mask_count = np.zeros((mask_h, mask_w, 1))
    for y in range(num_y):
        for x in range(num_x):
            i = y * num_x + x
            ys, ye = y * stride, y * stride + patch_size
            xs, xe = x * stride, x * stride + patch_size
            # import pdb; pdb.set_trace()
            result_mask[:, ys:ye, xs:xe] += masks[i]
            mask_count[ys:ye, xs:xe, :] += 1
    result_mask = result_mask.transpose(1, 2, 0)
    result_mask /= mask_count
    return result_mask


def get_pred_results(outputs):
    outputs_scores = F.softmax(outputs['pred_logits'][0], dim=-1).cpu().numpy()
    outputs_points = outputs['pred_points'][0].cpu().numpy()
    argmax_label = torch.argmax(outputs['pred_logits'][0], dim=-1).cpu().numpy()
    valid_pred_bool = (argmax_label > 0)
    pred_sum = valid_pred_bool.sum()

    if pred_sum > 0:
        outputs_points = outputs_points[valid_pred_bool]
        outputs_scores = outputs_scores[valid_pred_bool]
        pred_points, pred_labels = del_duplicate(outputs_points, outputs_scores, 16)
        pred_points = np.array(pred_points)
        pred_labels = np.array(pred_labels)
    else:
        pred_points = None
        pred_labels = None

    return pred_points, pred_labels


def generate_result_mask(image, net, patch_size=512, overlap=128, batch_size=4):
    img_h, img_w = image.shape[0:2]
    patch_imgs, num_y, num_x = split_image(image, patch_size, overlap)
    num_patches = patch_imgs.shape[0]
    patch_imgs = patch_imgs.transpose((0, 3, 1, 2))
    patch_imgs = patch_imgs * (2. / 255) - 1.
    results = []
    for i in range(0, num_patches, batch_size):
        # import pdb; pdb.set_trace()
        this_batch = patch_imgs[i:i + batch_size]
        with torch.no_grad():
            data_variable = torch.from_numpy(this_batch).float()
            if net.parameters().__next__().is_cuda:
                data_variable = data_variable.cuda(net.parameters().__next__().get_device())
                result = net(data_variable)
                # print('result_shape', result)
                sigmoid_result = torch.sigmoid(result[:, :7, :, :]).cpu()
                # final_result = sigmoid_result * weight
                final_result = sigmoid_result
                results.append(final_result.numpy())

    results = np.concatenate(results)
    result_masks = reconstruct_mask(results, patch_size, overlap, num_y, num_x)
    result_masks = result_masks[0:img_h, 0:img_w, :]

    return result_masks


def deduplicate(points, scores, interval):
    n = len(points)
    fused = np.full(n, False)
    result = np.zeros((0, 2))
    classes = np.array([])
    for i in range(n):
        if not fused[i]:
            fused_index = np.where(np.linalg.norm(points[[i]] - points[i:], 2, axis=1) < interval)[0] + i
            fused[fused_index] = True

            r_, c_ = np.where(scores[fused_index] == np.max(scores[fused_index]))
            r_, c_ = [r_[0]], [c_[0]]
            result = np.append(result, points[fused_index[r_]], axis=0)
            classes = np.append(classes, c_)
    return result, classes


def predict(model, images, num_classes, apply_deduplication: bool = False):
    outputs = model(images)

    points = outputs['pnt_coords'][0].cpu().numpy()
    scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()
    classes = np.argmax(scores, axis=-1)
    reserved_index = classes < num_classes
    torch.cuda.empty_cache()
    if apply_deduplication:
        return deduplicate(points[reserved_index], scores[reserved_index], 12)
    else:
        return points[reserved_index], classes[reserved_index]


def test_p2p_plus_ki67(slide, coord_range, net, patch_size=512):
    mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])

    slide_width = slide.width
    slide_height = slide.height
    # print(coord_range)
    coord_range_test = [0, 0, slide_width - 1, slide_height - 1]

    if coord_range_test == coord_range:
        contours = find_tissue_countours(slide)
        region_info_dict = split_patches(slide, patch_size=patch_size, contours=contours)
        coord_list = region_info_dict.values()
    else:
        coord_list = split_image_P2P(slide, coord_range, patch_size=patch_size, overlap=0)

    # print('patch_imgs', patch_imgs.shape)
    final_pred_labels = []
    final_pred_points = []
    zoom_rate = 768 / patch_size
    base_trans = transforms.Compose([transforms.ToTensor()])
    with torch.no_grad():
        for coord in coord_list:
            x = coord[0]
            y = coord[1]
            crop_img = slide.read((x, y), (patch_size, patch_size), 1)

            crop_img = cv2.resize(crop_img, None, fx=768 / patch_size, fy=768 / patch_size)
            crop_w, crop_h = crop_img.shape[0:2]
            pad_h, pad_w = 768 - crop_h, 768 - crop_w
            if pad_h > 0 or pad_w > 0:
                crop_img = np.pad(crop_img, ((0, pad_w), (0, pad_h), (0, 0)), 'constant')
            image = base_trans(crop_img)
            for t, m, s in zip(image, mean, std):
                t.sub_(m).div_(s)
            image = image[None].cuda()
            pred_points, pred_labels = predict(net, image, num_classes=6, apply_deduplication=True)

            if pred_points is not None:
                pred_points = (pred_points / zoom_rate) + coord
                pred_points = list(pred_points)
                pred_labels = list(pred_labels)
                final_pred_points.extend(pred_points)
                final_pred_labels.extend(pred_labels)
    return np.array(final_pred_points), np.array(final_pred_labels)


def generate_result_mask_P2P(image, net, patch_size=512, overlap=128):
    patch_imgs, coord_list = split_image_P2P(image, patch_size, overlap)
    patch_imgs = patch_imgs.transpose((0, 3, 1, 2))
    patch_imgs = patch_imgs * (2. / 255) - 1.
    final_pred_labels = []
    final_pred_points = []
    for this_batch, coord in zip(patch_imgs, coord_list):
        with torch.no_grad():
            data_variable = torch.from_numpy(this_batch).float()
            if net.parameters().__next__().is_cuda:
                data_variable = data_variable.cuda(net.parameters().__next__().get_device())
                data_variable = data_variable.unsqueeze(dim=0)
                outputs = net(data_variable)
                pred_points, pred_labels = get_pred_results(outputs)
                if pred_points is not None:
                    pred_points = pred_points + coord
                    pred_points = list(pred_points)
                    pred_labels = list(pred_labels)
                    final_pred_points.extend(pred_points)
                    final_pred_labels.extend(pred_labels)

    final_pred_points = np.array(final_pred_points)
    final_pred_labels = np.array(final_pred_labels)
    # final_pred_points, final_pred_labels = del_duplicate(final_pred_points, final_pred_labels, 16)

    return final_pred_points, final_pred_labels


def get_coordinate(voting_map, min_len=6):
    voting_map = cv2.GaussianBlur(voting_map, (49, 49), 0)
    coordinates = peak_local_max(voting_map, min_distance=min_len, exclude_border=min_len // 2)  # N by 2
    if coordinates.size == 0:
        coordinates = None
        return coordinates

    boxes_list = [coordinates[:, 1:2], coordinates[:, 0:1], coordinates[:, 1:2], coordinates[:, 0:1]]
    coordinates = np.concatenate(boxes_list, axis=1)
    return coordinates


def write_added_mask(img, rgb, folder_name, img_name, func_name):
    img_grey = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    img_grey = np.stack([img_grey, img_grey, img_grey], axis=2)

    pred_masks = cv2.addWeighted(img_grey, 0.6, rgb, 0.4, 0)
    cv2.imwrite(os.path.join(folder_name, img_name + func_name), pred_masks)

    return


def post_process_mask(img, masks, threshold, resize_ratio):
    voting_map = np.sum(masks, axis=2)
    voting_map[voting_map < threshold * np.max(voting_map)] = 0

    bboxes = get_coordinate(voting_map, min_len=int(10 * resize_ratio))
    # print('emmm', bboxes)
    x_coords = bboxes[:, 0]
    y_coords = bboxes[:, 1]
    pred_center_coords = bboxes[:, 0:2]

    label_map = np.argmax(masks, axis=2)
    pred_label = label_map[y_coords, x_coords]
    # img_ = img.copy()
    # B, G, R = cv2.split(img_)

    # center_label_dict = dict(zip(tuple(map(tuple, pred_center_coords)), pred_label))
    # center_label_dict, center_coords_voted, label_voted = \
    #     voting(img, R, masks, center_label_dict, pred_center_coords, pred_label)

    # seed = np.zeros_like(masks)
    # x_coords = pred_center_coords[:, 0]
    # y_coords = pred_center_coords[:, 1]
    # seed[y_coords, x_coords, label_voted] = 1

    return pred_center_coords, pred_label


def get_anno(point_anno_path):
    with open(point_anno_path, "r", encoding="utf-8") as f:
        index_info = json.load(f)
        if "roilist" in index_info:
            roilist = index_info['roilist']
            for roi in roilist:
                path = roi.get("path")
                if 'x' in path and 'y' in path:
                    x_coord = path['x'][0]
                    y_coord = path['y'][0]

    return x_coord, y_coord


def cal_standard_shade(point_anno_path, R):
    shallow_x_coord, shallow_y_coord = get_anno(point_anno_path)
    std_seed = np.zeros_like(R)
    std_seed[int(shallow_y_coord), int(shallow_x_coord)] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    std_seed = cv2.dilate(std_seed, kernel, iterations=1)

    mask_label, _ = ndimage.label(std_seed)
    index = np.unique(mask_label)
    standard_shade = ndimage.mean(R, labels=mask_label, index=index)[1]

    return standard_shade


def cal_ki67_np(ori_img, net, resize_ratio=1):
    img = ori_img.copy()
    result_masks = generate_result_mask(img, net, patch_size=512, overlap=64, batch_size=1)
    pred_center_coords, pred_label = post_process_mask(ori_img, masks=result_masks, threshold=0.1,
                                                       resize_ratio=resize_ratio)
    return pred_center_coords, pred_label
