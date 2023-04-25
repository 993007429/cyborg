#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 14:37
# @Author  : Can Cui
# @File    : bm_alg.py
# @Software: PyCharm
import torch
import os, sys
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model')
from .configs import BMBaseConfig
from .src.detect_blood_cell import nuclei_det



class BMAlgBase(object):
    def __init__(self):
        self.config = BMBaseConfig()
        self.model = None
        self.result = {}

        self.patch_size=1024
        self.standard_mpp = 0.242042*2
        self.crop_size = (1920, 1440)



    def load_bm_detector(self, model_name='yolov5', weights_name='best.torchscript'):
        det_model_path = os.path.join(model_dir, model_name, weights_name)
        model = torch.jit.load(det_model_path)
        model = model if not torch.cuda.is_available() else model.cuda()
        model.eval()
        if self.config.is_half:
            model.half()
            model(torch.zeros(1,3,1024,1024).half().cuda())
        else:
            model(torch.zeros(1,3,1024,1024).float().cuda())
        return model

    def cal_bm(self, slide):
        bboxes, scores, labels = nuclei_det(slide, self.model, self.config.is_half, self.standard_mpp, patch_size=self.patch_size, crop_size=self.crop_size, overlap=0, num_workers=4, batch_size=4,
                   x_coords=[], y_coords=[])


        return bboxes, scores, labels



class BM1115(BMAlgBase):
    def __init__(self):
        super(BM1115, self).__init__()
        self.model = self.load_bm_detector()
