from __future__ import unicode_literals

import copy
import logging
from typing import Optional

import numpy as np
from collections import Counter
import pandas as pd
from csaps import csaps
import heapq

from cyborg.consts.pdl1 import Pdl1Consts
from cyborg.modules.ai.libs.algorithms.PDL1.run_pdl1 import cal_pdl1

logger = logging.getLogger(__name__)

ptc = Pdl1Consts.annot_clss_map_dict['阳性肿瘤细胞']
ntc = Pdl1Consts.annot_clss_map_dict['阴性肿瘤细胞']
phc = Pdl1Consts.annot_clss_map_dict['阳性组织细胞']
nlc = Pdl1Consts.annot_clss_map_dict['阴性组织细胞']


def refine_tps(combine_test_labels, combine_test_prob, fitting_model):
    combine_test_labels_all = combine_test_labels
    combine_test_prob_all = combine_test_prob

    total_num_ptc = combine_test_labels_all.count(ptc)
    total_num_ntc = combine_test_labels_all.count(ntc)

    tps = total_num_ptc / (total_num_ptc + total_num_ntc + 1e-10)

    target_tps = fitting_model(tps * 100) / 100

    if target_tps < 0:
        target_tps = 0

    matrix_alpha_value, matrix_beta_value = cell_refine_matrix(tps, target_tps)

    cell_prob_dict = {
        'ptc_prob': [],
        'ntc_prob': [],
        'pnc_prob': [],
        'nnc_prob': [],
    }

    cell_pos_dict = {
        'ptc_pos': [],
        'ntc_pos': [],
        'pnc_pos': [],
        'nnc_pos': [],
    }

    for i in range(combine_test_labels_all.__len__()):
        if combine_test_labels_all[i] == ptc:
            cell_prob_dict['ptc_prob'].extend([combine_test_prob_all[i]])
            cell_pos_dict['ptc_pos'].extend([i])
        if combine_test_labels_all[i] == ntc:
            cell_prob_dict['ntc_prob'].extend([combine_test_prob_all[i]])
            cell_pos_dict['ntc_pos'].extend([i])
        if combine_test_labels_all[i] == phc:
            cell_prob_dict['pnc_prob'].extend([combine_test_prob_all[i]])
            cell_pos_dict['pnc_pos'].extend([i])
        if combine_test_labels_all[i] == nlc:
            cell_prob_dict['nnc_prob'].extend([combine_test_prob_all[i]])
            cell_pos_dict['nnc_pos'].extend([i])
    combine_test_labels_all_change = change_cell_result(cell_prob_dict, cell_pos_dict, combine_test_labels_all[:],
                                                        matrix_alpha_value, matrix_beta_value,
                                                        target_tps)
    return combine_test_labels_all_change


def cell_refine_matrix(tps, target_tps):
    matrix_alpha = [[1, 0.561, 0.3162, 0.2123, 0.1492, 0.1139, 0.07907, 0.04929, 0.02499, 0.01],
                    [1.369, 1, 0.6637, 0.5081, 0.3585, 0.2522, 0.174, 0.09276, 0.04768, 0.15],
                    [1.5, 1.273, 1, 0.7077, 0.6045, 0.4274, 0.2839, 0.1753, 0.09642, 0.25],
                    [1.402, 1.227, 1.076, 1, 0.7898, 0.5414, 0.4086, 0.2813, 0.1499, 0.35],
                    [1.406, 1.37, 1.15, 1.144, 1, 0.7786, 0.552, 0.3401, 0.2213, 0.45],
                    [1.398, 1.41, 1.163, 1.263, 1.092, 1, 0.7792, 0.5212, 0.3183, 0.55],
                    [1.37, 1.209, 1.379, 1.23, 1.175, 1.085, 1, 0.7375, 0.4688, 0.65],
                    [1.473, 1.429, 1.225, 1.38, 1.334, 1.219, 1.279, 1, 0.8207, 0.75],
                    [1.453, 1.456, 1.446, 1.429, 1.422, 1.178, 1.311, 1.185, 1, 0.90],
                    [1.246, 1.468, 1.464, 1.311, 1.411, 1.326, 1.416, 1.344, 1.351, 1]
                    ]

    matrix_beta = [[1, 1.232, 1.183, 1.203, 1.249, 1.399, 1.465, 1.482, 1.491, 1.05],
                   [0.6275, 1, 1.11, 1.297, 1.347, 1.388, 1.44, 1.241, 1.252, 1.05],
                   [0.5, 0.7675, 1, 1.074, 1.344, 1.387, 1.382, 1.374, 1.468, 1.05],
                   [0.267, 0.498, 0.7424, 1, 1.151, 1.147, 1.294, 1.429, 1.459, 1.05],
                   [0.194, 0.3952, 0.5331, 0.7906, 1, 1.127, 1.188, 1.167, 1.439, 1.05],
                   [0.1452, 0.2975, 0.3851, 0.6135, 0.7604, 1, 1.152, 1.221, 1.397, 1.05],
                   [0.1098, 0.188, 0.3273, 0.4184, 0.564, 0.7401, 1, 1.159, 1.362, 1.05],
                   [0.092, 0.1655, 0.2033, 0.3192, 0.4265, 0.5434, 0.8238, 1, 1.11, 1.05],
                   [0.07275, 0.1236, 0.1627, 0.2108, 0.2767, 0.3093, 0.4837, 0.6646, 1, 1.05],
                   [0.0499, 0.0893, 0.1021, 0.1057, 0.1347, 0.1445, 0.2144, 0.2865, 0.4875, 1]
                   ]

    int_tps = int(tps * 10)
    int_target_tps = int(target_tps * 10)
    matrix_alpha_value = matrix_alpha[int_target_tps][int_tps]
    matrix_beta_value = matrix_beta[int_target_tps][int_tps]

    if target_tps < 0.01 and target_tps != 0:
        matrix_alpha_value = 0.1
        matrix_beta_value = 1.01

    if target_tps == 0:
        matrix_alpha_value = 0
        matrix_beta_value = 1.01
    return matrix_alpha_value, matrix_beta_value


def change_cell_result(cell_prob_dict, cell_pos_dict, combine_test_labels_all, matrix_alpha_value, matrix_beta_value,
                       target_tps):
    total_num_ptc = combine_test_labels_all.count(ptc)
    total_num_ntc = combine_test_labels_all.count(ntc)
    total_num_phc = combine_test_labels_all.count(phc)
    total_num_nlc = combine_test_labels_all.count(nlc)

    if matrix_alpha_value > 1:
        n_ptc = int(total_num_ptc * abs(1 - matrix_alpha_value))

        if n_ptc < total_num_phc * 0.5:
            min_pos = list(map(cell_prob_dict['pnc_prob'].index, heapq.nsmallest(n_ptc, cell_prob_dict['pnc_prob'])))

            for i in range(min_pos.__len__()):
                combine_test_labels_all[cell_pos_dict['pnc_pos'][min_pos[i]]] = ptc

            n_ntc = int(total_num_ntc * abs(1 - matrix_beta_value))

            min_pos = list(map(cell_prob_dict['ntc_prob'].index, heapq.nsmallest(n_ntc, cell_prob_dict['ntc_prob'])))

            for i in range(min_pos.__len__()):
                combine_test_labels_all[cell_pos_dict['ntc_pos'][min_pos[i]]] = nlc

        else:

            min_pos = list(map(cell_prob_dict['pnc_prob'].index,
                               heapq.nsmallest(int(total_num_phc * 0.5), cell_prob_dict['pnc_prob'])))

            for i in range(min_pos.__len__()):
                combine_test_labels_all[cell_pos_dict['pnc_pos'][min_pos[i]]] = ptc

            ntc_change_num = int(abs((total_num_ptc + int(total_num_phc * 0.5)) / (target_tps + 1e-10) - (
                    total_num_ptc + int(total_num_phc * 0.5)) - total_num_ntc))

            min_pos = list(
                map(cell_prob_dict['ntc_prob'].index, heapq.nsmallest(ntc_change_num, cell_prob_dict['ntc_prob'])))

            for i in range(min_pos.__len__()):
                combine_test_labels_all[cell_pos_dict['ntc_pos'][min_pos[i]]] = nlc

    if matrix_beta_value > 1:
        n_ntc = int(total_num_ntc * abs(1 - matrix_beta_value))

        if n_ntc < total_num_nlc * 0.5:
            min_pos = list(map(cell_prob_dict['nnc_prob'].index, heapq.nsmallest(n_ntc, cell_prob_dict['nnc_prob'])))

            for i in range(min_pos.__len__()):
                combine_test_labels_all[cell_pos_dict['nnc_pos'][min_pos[i]]] = ntc

            n_ptc = int(total_num_ptc * abs(1 - matrix_alpha_value))

            min_pos = list(map(cell_prob_dict['ptc_prob'].index, heapq.nsmallest(n_ptc, cell_prob_dict['ptc_prob'])))

            for i in range(min_pos.__len__()):
                combine_test_labels_all[cell_pos_dict['ptc_pos'][min_pos[i]]] = phc

        else:

            min_pos = list(map(cell_prob_dict['nnc_prob'].index,
                               heapq.nsmallest(int(total_num_nlc * 0.5), cell_prob_dict['nnc_prob'])))

            for i in range(min_pos.__len__()):
                combine_test_labels_all[cell_pos_dict['nnc_pos'][min_pos[i]]] = ntc

            ptc_change_num = int(abs(
                target_tps * (total_num_ntc + int(total_num_nlc * 0.5)) / (1 - target_tps) - total_num_ptc))

            min_pos = list(
                map(cell_prob_dict['ptc_prob'].index, heapq.nsmallest(ptc_change_num, cell_prob_dict['ptc_prob'])))

            for i in range(min_pos.__len__()):
                combine_test_labels_all[cell_pos_dict['ptc_pos'][min_pos[i]]] = phc

    return combine_test_labels_all


def fitting_target_tps_update(excel_path: str = None, smooth_value: Optional[float] = None):
    try:
        smooth_value = float(smooth_value)
        tps_csv = pd.read_excel(excel_path)
        tps_csv = tps_csv.dropna(axis=0, how='any')
        tps_csv = tps_csv.drop_duplicates(subset=['AI_TPS'], keep='first')  # 将x重复的drop,保留第一项
        orginal_tps_np = tps_csv['AI_TPS']
        target_tps_np = tps_csv['Target_tps']
        zip_tps = list(map(list, zip(list(orginal_tps_np), list(target_tps_np))))
        zip_tps.sort(key=lambda x: x[0])
        np_tps = np.array(zip_tps, dtype=float)
        orginal_tps_np = np.array(np_tps[:, 0:1].flatten())
        target_tps_np = np.array(np_tps[:, 1:2].flatten())
        fitting_model = csaps(orginal_tps_np, target_tps_np, smooth=smooth_value)
        return fitting_model
    except Exception as e:
        logger.error(str(e))
        return None


def compute_pdl1_s(slide_path=None, x_coords=None, y_coords=None, fitting_model=None, smooth=None):
    count_summary_dict = {'neg_norm': 0, 'neg_tumor': 0, 'pos_norm': 0, 'pos_tumor': 0, 'total': 0, 'tps': 0}
    roi_center = []
    cls_labels = []
    ori_labels = []
    annot_cls_labels = []
    probs = []

    try:
        center_coords_np, cls_labels_np, probs_np = cal_pdl1(
            slide_path, x_coords=x_coords or [], y_coords=y_coords or [])

        if fitting_model and smooth:
            changed_cls_labels = refine_tps(combine_test_labels=copy.deepcopy(cls_labels_np.tolist()),
                                            combine_test_prob=probs_np.tolist(),
                                            fitting_model=fitting_model)
            changed_cls_labels = np.array(changed_cls_labels)
        else:
            changed_cls_labels = cls_labels_np
        remap_changed_cls_labels = np.vectorize(Pdl1Consts.label_to_diagnosis_type.get)(changed_cls_labels)
        remap_ori_cls_labels = np.vectorize(Pdl1Consts.label_to_diagnosis_type.get)(cls_labels_np)
        if remap_changed_cls_labels is not None:
            cell_count = Counter(remap_changed_cls_labels)
            count_summary_dict = {
                'neg_norm': cell_count[0], 'neg_tumor': cell_count[1],
                'pos_norm': cell_count[2], 'pos_tumor': cell_count[3],
                'total': int(remap_changed_cls_labels.size),
                'tps': round(float(cell_count[Pdl1Consts.cell_label_dict['pos_tumor']] / (
                        cell_count[Pdl1Consts.cell_label_dict['pos_tumor']] +
                        cell_count[Pdl1Consts.cell_label_dict['neg_tumor']] + 1e-10)), 4)}

            roi_center = center_coords_np.tolist()
            cls_labels = remap_changed_cls_labels.tolist()
            probs = probs_np.tolist()
            annot_cls_labels = changed_cls_labels.tolist()
            ori_labels = remap_ori_cls_labels.tolist()
        return count_summary_dict, roi_center, ori_labels, cls_labels, probs, annot_cls_labels

    except Exception as e:
        logger.exception(e)
        return count_summary_dict, roi_center, ori_labels, cls_labels, probs, annot_cls_labels
