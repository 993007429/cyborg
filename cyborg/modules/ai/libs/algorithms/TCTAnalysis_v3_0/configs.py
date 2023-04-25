#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 18:18
# @Author  : Can Cui
# @File    : configs.py
# @Software: PyCharm
# @Comment:

class TctBaseConfig(object):

    multi_wsi_cls_dict = {
        'NILM': 0,
        'ASC-US': 1,
        'LSIL': 2,
        'ASC-H': 3,
        'HSIL': 4,
        'AGC':5
    }
    multi_wsi_cls_dict_reverse = {v:k for k,v in multi_wsi_cls_dict.items()}

    multi_cell_cls_dict = {
        'neg': 0,
        'ASC-UC':1,
        'LSIL':2,
        'HSIL':4,
        'ASC-H':3,
        'AGC':5
    }
    multi_cell_cls_dict_reverse = {v:k for k,v in multi_cell_cls_dict.items()}

    multi_microorganism_cls_dict = {
        'neg':0,
        '放线菌':1,
        '滴虫':2,
        '真菌':3,
        '疱疹':4,
        '线索细胞':5
    }
    multi_microorganism_cls_dict_reverse = {v:k for k,v in multi_microorganism_cls_dict.items()}

    quality_control_cls_dict = {
        'good': 0,
        'bad': 1
    }
    quality_control_cls_dict_reverse = {v:k for k,v in quality_control_cls_dict.items()}

    is_disc_detect = True
    is_nms = True
    is_qc = True
    is_microbe_detect = False
    num_workers = 5
    min_cellnum = 5000 # if cell num < min_cellnum: quality -> bad
    qc_threshold = 0.15 # quality control model threshold
    min_return_pos_cell = 36  # in case of positive slide has no positive cell
    min_pos_cell_threshold = -1 # if pos cell num < min_pos_cell_threshold:  slide -> negative

