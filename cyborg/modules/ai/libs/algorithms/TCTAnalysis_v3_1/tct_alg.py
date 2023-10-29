#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/11 11:55
# @Author  : Can Cui
# @File    : tct_alg.py
# @Software: PyCharm

import os, sys
import numpy as np
import yaml
from cyborg.infra.oss import oss

from .src.utils import load_cfg
from .src.detect_tct_disk import disc_detect
from .src.cell_counting import CellCounter
from .src.cell_detection import CellDetector
from .src.wsi_cls import WSIClassifier
import torch
from collections import Counter


model_dir = oss.path_join('AI', 'TCTAnalysis_v3_1', 'Model')

class TCT_ALG2:
    def __init__(self, config_path='test', threshold=None):
        self.cfg = load_cfg(config_path)
        self.model_name = self.cfg['model_name']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pos_threshold = 0.33 if threshold is None else threshold

        if self.cfg.get('cell_count'):
            self.cell_counter = CellCounter(
                model_path=oss.path_join(model_dir, self.cfg['count']['model'], self.cfg['count']['weights']),
                device=self.device,**self.cfg['count'])
        else:
            self.cell_counter = None

        # types of cells to detect
        self.pos_det_idx = [idx for idx, label in enumerate(self.cfg['cell_det_labels']) if label in self.cfg['pos_labels']]
        self.neg_det_idx = [idx for idx, label in enumerate(self.cfg['cell_det_labels']) if label in self.cfg['neg_labels']]
        self.microbe_det_idx = [idx for idx, label in enumerate(self.cfg['cell_det_labels']) if label in self.cfg['microbe_labels']]
        self.det_cls = []
        self.det_cls+=self.pos_det_idx
        if self.cfg.get('microbe_det'):
            self.det_cls += self.microbe_det_idx
        if self.cfg.get('neg_det'):
            self.det_cls += self.neg_det_idx

        self.cell_detector = CellDetector(
            model_path=oss.path_join(model_dir, self.cfg['det']['model'], self.cfg['det']['weights']),
            device=self.device, classes=self.det_cls, **self.cfg['det'])

        self.wsi_classifier = WSIClassifier(
            backbone_path=oss.path_join(model_dir, self.cfg['wsi']['cell_model'], self.cfg['wsi']['cell_weights']),
            wsi_model_path=oss.path_join(model_dir, self.cfg['wsi']['wsi_model'], self.cfg['wsi']['wsi_weights']),
            device=self.device,**self.cfg['wsi'])

        self.qc_classifier = None
        self.microbe_classifier = None

    def filter_cells(self, det_result, wsi_soft):
        return_cells = []
        watch_list = []
        microbe_pred = []
        background_pred = []
        count_list = []
        sort_idx = det_result[:,4].argsort(descending=True)
        det_result = det_result[sort_idx]

        pos_det_result = det_result[(det_result[:, 5:6] == torch.tensor(self.pos_det_idx, device=det_result.device)).any(1)]

        wsi_soft = wsi_soft.cpu().numpy()
        pos_det_result = pos_det_result.cpu().numpy()
        for idx in range(pos_det_result.shape[0]):
            coords = pos_det_result[idx, :4].astype(int).tolist()
            cls_idx = int(pos_det_result[idx,5])
            cls_name = self.cfg['cell_det_labels'][cls_idx]
            prob = float(pos_det_result[idx, 4])
            if prob >= self.cfg['cell_thres'][cls_name]:
                return_cells.append({'bbox':coords, 'label':cls_name, 'prob':prob})
            else:
                watch_list.append({'bbox':coords, 'label':cls_name, 'prob':prob})
        num_pos_cell = len(return_cells)
        if num_pos_cell < self.cfg['min_return_cell']:
            append_num = self.cfg['min_return_cell'] - num_pos_cell
            return_cells += watch_list[:append_num]

        other_det_cls = []
        if self.cfg['microbe_det']:
            other_det_cls += self.microbe_det_idx
        if self.cfg['neg_det']:
            other_det_cls += self.neg_det_idx

        other_det_result = det_result[(det_result[:, 5:6] == torch.tensor(other_det_cls, device=det_result.device)).any(1)]

        other_det_result = other_det_result.cpu().numpy()
        for idx in range(other_det_result.shape[0]):
            coords = other_det_result[idx, :4].astype(int).tolist()
            cls_idx = int(other_det_result[idx,5])
            cls_name = self.cfg['cell_det_labels'][cls_idx]
            prob = float(other_det_result[idx, 4])
            if prob >= self.cfg['cell_thres'][cls_name]:
                return_cells.append({'bbox':coords, 'label':cls_name, 'prob':prob})
                count_list.append(cls_name)

        cell_counter = Counter(count_list)
        for k, v in cell_counter.items():
            if v >= self.cfg['wsi_report_num'][k]:
                if k in self.cfg['microbe_labels']:
                    microbe_pred.append(k)
                if k in self.cfg['neg_labels']:
                    background_pred.append(k)
        return return_cells, microbe_pred, background_pred

    @torch.no_grad()
    def cal_tct(self, slide):

        x_coords, y_coords = disc_detect(slide, self.cfg.get('disc_det'))

        # if os.path.exists('det_res.pth'):
        # if False:
        #     det_result = torch.load('det_res.pth')
        # else:
        det_result = self.cell_detector.detect_wsi(slide, x_coords=x_coords, y_coords=y_coords)
        if self.cfg.get('cell_count'):
            cell_num = self.cell_counter.count_wsi(slide, x_coords=x_coords, y_coords=y_coords)
        else:
            cell_num = det_result.shape[0]*5


        pos_det_result = det_result[(det_result[:, 5:6] == torch.tensor(self.pos_det_idx, device=det_result.device)).any(1)]
        bboxes, cell_soft, wsi_soft = self.wsi_classifier.infer_wsi(slide, pos_det_result)


        slide_diagnosis = 'positive' if 1 - wsi_soft[0] > float(self.pos_threshold) else 'negative'
        tbs_label = self.cfg['wsi_labels'][wsi_soft.argmax().item()]
        if tbs_label=='NILM' and slide_diagnosis=='positive':
            tbs_label = self.cfg['wsi_labels'][wsi_soft[:1].argmax().item()+1]
        if slide_diagnosis == 'negative':
            tbs_label = ''

        if cell_num<self.cfg['min_cellnum']:
            quality=0
        else:
            quality = 1
        return_cells, microbe_pred, background_pred = self.filter_cells(det_result, wsi_soft)

        result = dict(
            cells=return_cells,
            background=background_pred,
            microbe=microbe_pred,
            slide_pos_prob=wsi_soft.cpu().numpy(),
            diagnosis=slide_diagnosis,
            tbs_label=tbs_label,
            quality=quality,
            cell_num=cell_num
        )

        return result




