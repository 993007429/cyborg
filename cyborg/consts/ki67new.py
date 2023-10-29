#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/29 18:19
# @Author  : Can Cui
# @File    : ki67new.py
# @Software: PyCharm


class Ki67NewConsts(object):

    map_dict = {0: 1, 1: 0, 2: 2, 3: 2}
    type_color_dict = {0: 'green', 1: 'red', 2: '#C280FF'}
    annot_clss_map_dict = {
        '阴性肿瘤细胞': 0,
        '阳性肿瘤细胞': 1,
        '阴性组织细胞': 2,
        '阳性组织细胞': 3,
    }
    ori_clss_map_dict = {
        '阳性肿瘤细胞': 0,
        '阴性肿瘤细胞': 1,
        '阳性淋巴细胞': 2,
        '阴性淋巴细胞': 3,
    }

    reversed_annot_clss_map_dict = {v: k for k, v in ori_clss_map_dict.items()}