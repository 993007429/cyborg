#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 1 14:44:26 2020

@author: af2o
"""
import numpy as np
import cv2
import pdb
import time
from scipy.ndimage import label as label_func
from scipy.ndimage import mean as mean_func
from imageio import imsave

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def P_tumor_voting(d1, d2, alpha = 0.2, beta = 3, std_dist = 100, std_R_dist = 10):

    Euclid_dist = np.exp(-d1/std_dist)
    R_dist = np.exp(-d2/std_R_dist)
    # print('Euclid_dist', np.mean(Euclid_dist), np.median(Euclid_dist), np.max(Euclid_dist))
    # print('R_dist', np.mean(R_dist), np.median(R_dist), np.max(R_dist))
    distance = alpha * (Euclid_dist + beta * R_dist)

    return distance

def compute_matrix(a, b):
    return np.abs(np.array(a).reshape(-1, 1) - np.array(b).reshape(1, -1))

def compute_d1(lymph_ycoords, p_tumor_ycoords, lymph_xcoords, p_tumor_xcoords):
    # index is given by label()
    y_distance = compute_matrix(lymph_ycoords, p_tumor_ycoords)
    x_distance = compute_matrix(lymph_xcoords, p_tumor_xcoords)
    d1 = np.sqrt(y_distance ** 2 + x_distance ** 2)
    return d1

def compute_d2(lymph_shade, p_tumor_shade):
    # index is given by label()
    d2 = np.abs(np.array(lymph_shade).reshape(-1, 1) - np.array(p_tumor_shade).reshape(1, -1))
    return d2

def watershed(img, seed, label_id = 5):

    # erode might discard some marginal coords

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed = cv2.dilate(seed, kernel, iterations=1)
    seed_temp = (seed * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_copy = img.copy()
    img_bin = cv2.morphologyEx(seed_temp, cv2.MORPH_OPEN, np.ones((3, 3), dtype=int))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    if label_id == 2:
        border_outside = cv2.dilate(img_bin, kernel, iterations = 15)
        border = border_outside-cv2.erode(border_outside, kernel, 3)##10
    else:
        border_outside = cv2.dilate(img_bin, kernel, iterations = 30)
        border = border_outside-cv2.erode(border_outside, kernel, 3)##10

    lbl, ncc = label_func(seed_temp)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    cv2.watershed(img_copy, lbl)

    lbl[lbl == -1] = 0
    lbl = 255 - lbl.astype(np.uint8)

    mask_temp = lbl.copy()

    mask_temp[(mask_temp != 0) & (mask_temp != 255)] = 1
    mask_temp[mask_temp == 255] = 0
    return mask_temp

def dict2list(center_label_dict):
    center_coords = np.array(list(center_label_dict)).astype(np.int)
    labels = list(center_label_dict.values())
    labels = np.array(labels).astype(np.int)
    return center_coords, labels

def extract_paras(img, seed, center_coords, center_label_dict, R):

    index_labels, num_regions = label_func(watershed(img, seed))
    index = np.unique(index_labels)
    average_shade_list = mean_func(R, labels=index_labels, index=index)
    all_coords = list(zip(list(center_coords[:, 0]), list(center_coords[:, 1])))
    lympha_paras = []
    p_tumor_paras = []

    for i, (x_coord, y_coord) in enumerate(all_coords):

        label = center_label_dict[(x_coord, y_coord)]


        if (label == 1) or (label == 4):
            index_label = index_labels[y_coord, x_coord]
            shade = average_shade_list[index_label]
            lympha_paras.append((x_coord, y_coord, shade))

        if label == 5:
            index_label = index_labels[y_coord, x_coord]
            shade = average_shade_list[index_label]
            p_tumor_paras.append((x_coord, y_coord, shade))

    return lympha_paras, p_tumor_paras

def change_label(center_label_dict, lymph_ycoords, lymph_xcoords, masks, addition_matrix):

    changed_num = 0
    for lymph_index in range(len(lymph_ycoords)):

        y_coord = lymph_ycoords[lymph_index]
        x_coord = lymph_xcoords[lymph_index]
        xy_coords = (x_coord, y_coord)

        prob_p_tumor = masks[y_coord, x_coord, 5]
        prob_lymph = max(masks[y_coord, x_coord, 1], masks[y_coord, x_coord, 4])
        voting_res = sum(addition_matrix[lymph_index, :])
        addition = sigmoid(voting_res)
        voting_result = prob_p_tumor + addition


        '''
        I unchanged prob_lymph'''

        if voting_result > prob_lymph:
            changed_num += 1

            center_label_dict[xy_coords] = 5

    return center_label_dict, changed_num

def voting(img, R, masks, center_label_dict, center_coords, label):

    seed = np.zeros_like(masks)
    seed[center_coords[:, 1], center_coords[:, 0], label] = 1

    seed_in_use = seed[:, :, 1] + seed[:, :, 4] + seed[:, :, 5]
    lympha_paras, p_tumor_paras = \
        extract_paras(img, seed_in_use, center_coords, center_label_dict, R)

    if (not lympha_paras) or (not p_tumor_paras):
        return center_label_dict, center_coords, label


    lymph_xcoords, lymph_ycoords,lymph_shade = list(zip(*lympha_paras))[:]
    p_tumor_xcoords, p_tumor_ycoords, p_tumor_shade = list(zip(*p_tumor_paras))[:]

    # if len(lymph_shade) == 0 or len(p_tumor_shade) == 0:
    #     return center_coords, label

    d1 = compute_d1(lymph_ycoords, p_tumor_ycoords, lymph_xcoords, p_tumor_xcoords)
    d2 = compute_d2(lymph_shade, p_tumor_shade)
    addition_matrix = P_tumor_voting(d1, d2)
    center_label_dict, changed_num = \
        change_label(center_label_dict, lymph_ycoords, lymph_xcoords, masks, addition_matrix)

    center_coords_voted, labels = dict2list(center_label_dict)

    return center_label_dict, center_coords_voted, labels

def delete_label(center_coords, n_tumor_seed_dilated, center_label_dict, B, outlier_size = 1.12, max_shade = 240):

    index_labels, num_regions = label_func(n_tumor_seed_dilated)
    index = np.unique(index_labels)
    average_shade_list = mean_func(B, labels=index_labels, index=index)
    pic_ave_shade = np.mean(average_shade_list[1:])
    all_coords = list(zip(list(center_coords[:, 0]), list(center_coords[:, 1])))
    deleted_num = 0

    for i, (x_coord, y_coord) in enumerate(all_coords):

        label = center_label_dict[(x_coord, y_coord)]
        if label != 2:
            continue
        index_label = index_labels[y_coord, x_coord]
        blue_shade = average_shade_list[index_label]
        if (blue_shade > pic_ave_shade * outlier_size) or (blue_shade > max_shade):
            del center_label_dict[(x_coord, y_coord)]
            deleted_num += 1

    return center_label_dict, deleted_num

def delete_n_tumor(center_coords, label, masks, B):

    center_label_dict = dict(zip(tuple(map(tuple, center_coords)), label))
    seed = np.zeros_like(masks)
    seed[center_coords[:, 1], center_coords[:, 0], label] = 1

    n_tumor_seed = seed[:, :, 2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    seed_dilated = cv2.dilate(n_tumor_seed, kernel, iterations=1)

    center_label_dict, deleted_num = delete_label(center_coords, seed_dilated, center_label_dict, B)

    center_coords_deleted, pred_label_deleted = dict2list(center_label_dict)


    return center_label_dict, center_coords_deleted, pred_label_deleted
