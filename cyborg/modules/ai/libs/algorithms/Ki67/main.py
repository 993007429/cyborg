import logging
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import time
from torch import nn
from glob import glob
from skimage.segmentation import slic

from cyborg.infra.oss import oss
from cyborg.libs.heimdall.dispatch import open_slide
from cyborg.modules.ai.libs.algorithms.Ki67.models.Network import resnet50
from cyborg.modules.ai.libs.algorithms.Ki67.models.detr import build_model
from cyborg.modules.ai.libs.algorithms.Ki67.src.multi_cls_cell_seg_manual import test_p2p_plus_ki67
from cyborg.modules.ai.libs.algorithms.Ki67.utils import roi_filter, count_test_summary_P2P, threading_classification

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

logger = logging.getLogger(__name__)


class WSITester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        # delete the patch that the number of tumor less than thre.
        self.thre = 0
        self.standard_mpp = 0.479784

        # hyperparameters of cell counting
        self.label_dict = {'阴性肿瘤': 0, '阳性肿瘤': 1, '阴性淋巴': 2, '阳性淋巴': 3, '纤维细胞': 4, '其它细胞': 5}
        self.color_dict = {0: (0, 255, 255), 1: (0, 0, 255), 2: (26, 26, 139), 3: (0, 255, 0), "0": (0, 0, 255),
                           "1": (255, 255, 0), "2": (0, 255, 0), "3": (0, 0, 255),
                           "4": (255, 0, 0), "5": (255, 0, 0), "6": (128, 0, 255)}

        # load cell detection model
        self.detnet = build_model()
        model_file = oss.get_object_to_io(oss.path_join('AI', 'Ki67', 'Model', 'ki67_mix_0.9.pth'))
        ckpt = torch.load(model_file, map_location='cpu')
        model_dict = self.detnet.state_dict()
        pretrained_dict = {k: v for k, v in ckpt['model'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.detnet.load_state_dict(model_dict)
        self.detnet.cuda(self.device)
        self.detnet.eval()

        # load classification checkpoint
        self.cnet = resnet50(pretrained=False, classes=5)
        self.cnet = nn.DataParallel(self.cnet)
        net_weightspath = '5class_classification_model_best1.pth'
        checkpoint_file = oss.get_object_to_io(oss.path_join('AI', 'Ki67', 'Model', net_weightspath))
        checkpoint = torch.load(checkpoint_file)
        self.cnet.load_state_dict(checkpoint['Classifier'])
        self.cnet = self.cnet.module.to(self.device)
        self.cnet.eval()

    def _collect_info(self, pred_result_summary):
        df = pd.DataFrame(pred_result_summary)
        filtered_df = df.drop(df[df['肿瘤细胞'] < self.thre].index)
        ki67_total = 0
        pos_total = 0
        neg_total = 0
        for idx, row in filtered_df.iterrows():
            pos, neg, ki67 = row['阳性肿瘤'], row['阴性肿瘤'], row['ki67指数']
            ki67_total += ki67
            pos_total += pos
            neg_total += neg

        logger.info('patch size Ki67 index: %.2f' % (ki67_total / max(filtered_df.shape[0], 1)))
        logger.info('cell size Ki67 index: %.2f' % (pos_total / max((pos_total + neg_total), 1)))
        return filtered_df

    def _calculate_ki67(self, cur_slide, roi_list):
        pred_result_summary = {'阴性肿瘤': [], '阳性肿瘤': [], '阴性淋巴': [],
                               '阳性淋巴': [], '纤维细胞': [], '其它细胞': [],
                               '细胞总数': [], '肿瘤细胞': [], 'ki67指数': [], '类别': [], '检测坐标': [], 'id': [],
                               'x_coords': [], 'y_coords': []}

        for idx, roi in enumerate(roi_list):
            cls = roi.get('type') if roi.get('type') is not None else 3
            roiid = roi.get('id') if roi.get('id') is not None else -1

            x_coords = roi.get('x')
            y_coords = roi.get('y')
            xmin, ymin = max(0, int(min(min(x_coords), cur_slide.width - 1))), max(0, int(min(min(y_coords),
                                                                                              cur_slide.height - 1)))
            xmax, ymax = min(cur_slide.width - 1, int(max(0, max(x_coords)))), min(cur_slide.height - 1,
                                                                                   int(max(0, max(y_coords))))
            coord_range = [xmin, ymin, xmax, ymax]
            center_coords_pred, labels_pred = test_p2p_plus_ki67(cur_slide, coord_range, self.detnet,
                                                                 patch_size=self.patch_size)
            if labels_pred.shape[0] > 0:
                labels_pred = labels_pred.astype(int)
                selected_index = np.logical_and(center_coords_pred[:, 0] < xmax, center_coords_pred[:, 1] < ymax)
                center_coords_pred = center_coords_pred[selected_index]
                labels_pred = labels_pred[selected_index]
            if labels_pred.shape[0] > 0:
                center_coords_pred, labels_pred = roi_filter(center_coords_pred, labels_pred, x_coords,
                                                             y_coords)
                pred_count_dict = count_test_summary_P2P(center_coords_pred, labels_pred, self.label_dict)
                for k, v in pred_count_dict.items():
                    pred_result_summary[k].append(pred_count_dict[k])
                pred_result_summary['类别'].append(cls)
                pred_result_summary['检测坐标'].append(
                    np.concatenate((center_coords_pred, labels_pred[:, np.newaxis]), axis=1))
                pred_result_summary['id'].append(roiid)
                pred_result_summary['x_coords'].append(x_coords)
                pred_result_summary['y_coords'].append(y_coords)

        return pred_result_summary

    def _get_patch_number(self, maxconfs, threshold=0.5):
        max_number = 1000
        min_number = 3
        return np.clip((np.array(maxconfs) > threshold).sum(), min_number, max_number)

    def run(self, slide, compute_wsi: bool, roi_list: List[dict]):
        """classification evaluation phase
        :param slide:
        :param compute_wsi:
        :param roi_list:
        :return:
        """
        mpp = slide.mpp if slide.mpp else self.standard_mpp
        levels = [768, 1536, 3072]
        _, self.patch_size = min(enumerate(levels), key=lambda x: abs(x[1] - 384 / mpp))
        logger.info("Mpp is %.4f, and the patch size is %d." % (mpp, self.patch_size))

        if compute_wsi:
            mode = 'WSI'
        elif roi_list and roi_list[0].get('x'):
            mode = 'SEMI'
        else:
            mode = 'AUTO'

        if mode == 'AUTO':
            start_time = time.time()
            excutor = threading_classification(slide, self.cnet, self.patch_size, batch_size=64, device=self.device)
            excutor.execute()
            self.heatmap = excutor.heatmap
            self.width = slide.width
            self.height = slide.height
            self.W, self.H = excutor.W, excutor.H
            self.indexs = excutor.processed_index
            logger.info('Hot spots selection processing time: %.2f' % (time.time() - start_time))

            # selecting highly-confident patches
            full_heatmap = np.zeros((self.H * self.W))
            full_heatmap[self.indexs] = torch.softmax(self.heatmap, dim=1)[:, :3].sum(dim=1).cpu().numpy()
            full_heatmap = full_heatmap.reshape((self.W, self.H))

            seg = slic(full_heatmap, n_segments=self.H * self.W // 16, compactness=0.5, start_label=1)
            maxconfs = [np.median(full_heatmap[seg == i]) for i in range(seg.max() + 1)]
            self.patch_number = self._get_patch_number(maxconfs)
            kth = len(maxconfs) - self.patch_number
            inds = np.argpartition(np.array(maxconfs), kth)[kth:]

            for ind in inds:
                ind = np.argmax(np.where(seg == ind, full_heatmap, -10000))
                x, y = ind // self.H * self.patch_size, ind % self.H * self.patch_size
                xmax, ymax = x + self.patch_size, y + self.patch_size
                roi_list.append({'x': [x, xmax, xmax, x], 'y': [y, y, ymax, ymax]})

        for roi in roi_list:
            if not roi.get('x'):
                roi['x'] = [0.0, float(slide.width), float(slide.width), 0.0]
            if not roi.get('y'):
                roi['y'] = [0.0, 0.0, float(slide.height), float(slide.height)]

        pred_result_summary = self._calculate_ki67(slide, roi_list=roi_list)
        if mode == 'AUTO':
            filtered_df = self._collect_info(pred_result_summary)
        else:
            filtered_df = pd.DataFrame(pred_result_summary)
        return filtered_df


if __name__ == "__main__":
    pths = []
    ext = ['mrxs', 'kfb', 'sdpc']
    # [pths.extend(glob('E:/ki67_data/*/' + '*.' + e)) for e in ext]
    [pths.extend(glob('./data/*.' + e)) for e in ext]
    # pths.sort(key=lambda x: int(x.split('\\')[1][:9]))
    ki67_dict = {}
    for pth in pths:
        # if '1209618' not in pth:
        #     continue
        tic = time.time()
        print("slide name: %s" % pth)
        cur_slide = open_slide(pth)
        filtered_df = WSITester().run(cur_slide, compute_wsi=False, roi_list=[])
        ki67_dict[pth.split('\\')[1][:-5]] = filtered_df['阳性肿瘤'].sum() / filtered_df['肿瘤细胞'].sum() * 100.0
        print(time.time() - tic)
    pd.DataFrame(ki67_dict.items()).to_csv('Daan_Ki67_prediction.csv')
