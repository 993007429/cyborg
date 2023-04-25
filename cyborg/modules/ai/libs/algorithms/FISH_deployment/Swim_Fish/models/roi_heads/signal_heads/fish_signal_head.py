# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import HEADS, build_loss
# from mmcv.runner import BaseModule
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch
import numpy as np
import random
from imageio import imsave
import cv2
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead

from Algorithms.FISH_deployment.Swim_Fish.utils.signal_head_utils import abs_img_point_to_rel_roi_point

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        # print('let me see your size', x.size())
        out = self.double_conv(x)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.relu(out)
        return out


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(RegressionModel, self).__init__()

        self.conv1 = DoubleConv(num_features_in, feature_size)
        self.conv2 = DoubleConv(feature_size, feature_size)


    def forward(self, x):
        out = self.conv1(x)
        # print('out1-get')
        out = self.conv2(out)
        # print('out3-get')

        return out

class Conv_Delta(nn.Module):
    def __init__(self, num_anchor_points, feature_size):
        super(Conv_Delta, self).__init__()
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out = self.output(x)
        return out

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, feature_size):
        super(ClassificationModel, self).__init__()

        self.conv1 = DoubleConv(num_features_in, feature_size)
        self.conv2 = DoubleConv(feature_size, feature_size)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out

class Conv_logits(nn.Module):
    def __init__(self, num_anchor_points, signal_classes, feature_size):
        super(Conv_logits, self).__init__()

        self.signal_classes = signal_classes
        self.num_anchor_points = num_anchor_points

        self.output = nn.Conv2d(feature_size, num_anchor_points * signal_classes, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out = self.output(x)
        return out


class HungarianMatcher_Crowd():

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def matcher(self, pred_coords, pred_logits, tgt_points, tgt_ids):
        # print('########### pred_coords size ##########', pred_logits.size())
        roi_num, num_queries = pred_logits.shape[:2]
        out_prob = pred_logits.flatten(0, 1).softmax(-1)  # [roi_num * num_queries, num_classes]
        out_points = pred_coords.flatten(0, 1)  # [roi_num * num_queries, 2]
        tgt_points_cat = torch.cat([point.cpu() for point in tgt_points])
        tgt_ids_cat = torch.cat([label.cpu() for label in tgt_ids])
        sizes = [len(label) for label in tgt_ids]
        assert len(sizes) == roi_num

        tgt_ids_cat = tgt_ids_cat.type(torch.long)
        cost_class = -out_prob[:, tgt_ids_cat]
        cost_point = torch.cdist(out_points.float(), tgt_points_cat.float(), p=2)

        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(roi_num, num_queries, -1)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices_results = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        # print('########### indices_results ##########', len(indices_results))

        return indices_results


@HEADS.register_module()
class HTCSignalUNetHead(FCNMaskHead):
    def __init__(self, return_logit, num_convs, roi_feat_size, in_channels, conv_kernel_size, conv_out_channels,
                 num_classes, signal_classes, loss_points_coef, loss_cls_coef, signal_loss_weight, eos_coef, red_coef, *args, **kwargs):
        super(HTCSignalUNetHead, self).__init__(num_convs, roi_feat_size, in_channels, conv_kernel_size, conv_out_channels, num_classes, *args, **kwargs)

        if return_logit:
            self.num_anchor_points = 1
            self.eos_coef = eos_coef
            self.red_coef = red_coef
            self.signal_classes = signal_classes + 1
            empty_weight = torch.ones(self.signal_classes)
            empty_weight[0] = self.eos_coef
            empty_weight[1] = self.red_coef
            self.register_buffer('empty_weight', empty_weight)
            self.signal_loss_weight = signal_loss_weight
            self.loss_signal_points_dict = loss_points_coef
            self.loss_signal_labels_dict = loss_cls_coef

            self.loss_signal_points = build_loss(self.loss_signal_points_dict)
            self.loss_signal_labels = build_loss(self.loss_signal_labels_dict)
            self.matcher = HungarianMatcher_Crowd(cost_class=1, cost_point=0.05)

            self.regression_branch = RegressionModel(num_features_in=conv_out_channels, feature_size=conv_out_channels)
            self.conv_delta = Conv_Delta(num_anchor_points=self.num_anchor_points, feature_size=conv_out_channels)

            self.classification_branch = ClassificationModel(num_features_in=conv_out_channels, feature_size=conv_out_channels)
            self.conv_logits = Conv_logits(num_anchor_points=self.num_anchor_points, signal_classes=self.signal_classes,
                                           feature_size=conv_out_channels)

    def forward(self, x):

        for conv in self.convs:
            x = conv(x)
        res_feat = x

        return res_feat


    def forward_signal(self, x):

        x_delta_feat = self.regression_branch(x)
        deltas = self.conv_delta(x_delta_feat)
        deltas = deltas.permute(0, 2, 3, 1).contiguous()
        deltas = deltas.view(deltas.shape[0], -1, 2)

        x_logits_feat = self.classification_branch(x)
        pred_logits = self.conv_logits(x_logits_feat)
        pred_logits = pred_logits.permute(0, 2, 3, 1).contiguous()
        pred_logits = pred_logits.view(pred_logits.shape[0], -1, self.signal_classes)

        return deltas, pred_logits


    def get_targets(self, sampling_results, gt_signal_points, gt_signal_labels, gt_signal_ignore_area):
        """Get training targets of MaskPointHead for all images.
        Returns:
            Tensor: Point target, shape (num_rois, num_points).
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        # rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
        if gt_signal_ignore_area:
            signal_targets = list(map(self._get_target_single, pos_proposals, pos_assigned_gt_inds_list,
                                      gt_signal_points, gt_signal_labels, gt_signal_ignore_area))
        else:
            signal_targets = list(map(self._get_target_single, pos_proposals, pos_assigned_gt_inds_list,
                                      gt_signal_points, gt_signal_labels))

        return signal_targets


    def _get_target_single(self, rois, pos_assigned_gt_inds, gt_signal_points, gt_signal_labels):
        """Get training target of MaskPointHead for each image."""
        num_pos = rois.size(0)
        if num_pos > 0:
            signal_points_within_masks = [gt_signal_points[index] for index in list(np.array(pos_assigned_gt_inds.cpu()))]
            signal_labels_within_masks = [gt_signal_labels[index] for index in list(np.array(pos_assigned_gt_inds.cpu()))]

            signal_points_within_masks = abs_img_point_to_rel_roi_point(rois, signal_points_within_masks)
            all_signal_points = signal_points_within_masks
            all_signal_labels = signal_labels_within_masks

        else:
            all_signal_points = []
            all_signal_labels = []

        signal_target = {}
        signal_target['points'] = all_signal_points
        signal_target['labels'] = all_signal_labels

        vis_abs_points(all_target_points=all_signal_points, all_target_labels=all_signal_labels, rois= rois)

        return signal_target

    # def _get_target_single(self, rois, pos_assigned_gt_inds, gt_signal_points, gt_signal_labels, gt_signal_ignore_area):
    #     """Get training target of MaskPointHead for each image."""
    #     num_pos = rois.size(0)
    #     if num_pos > 0:
    #         signal_points_within_masks = [gt_signal_points[index] for index in list(np.array(pos_assigned_gt_inds.cpu()))]
    #         signal_labels_within_masks = [gt_signal_labels[index] for index in list(np.array(pos_assigned_gt_inds.cpu()))]
    #         signal_ignore_area_within_masks = [gt_signal_ignore_area[index] for index in list(np.array(pos_assigned_gt_inds.cpu()))]
    #
    #         signal_points_within_masks = abs_img_point_to_rel_roi_point(rois, signal_points_within_masks)
    #
    #         signal_ignore_rel_roi_area = abs_ignore_to_rel_roi_point(rois, signal_ignore_area_within_masks)
    #         dense_signal_points = dense_sampling(rois, signal_ignore_rel_roi_area)
    #         dense_signal_labels = [torch.ones(signal_points.size(0)).cuda(rois.device) \
    #                                           if len(signal_points) > 0 else torch.tensor([]).cuda(rois.device) \
    #                                       for signal_points in dense_signal_points]
    #
    #         all_signal_points = [torch.cat([sparse_signals, dense_signals]) for sparse_signals, dense_signals in zip(signal_points_within_masks, dense_signal_points)]
    #         all_signal_labels = [torch.cat([sparse_labels, dense_labels]) for sparse_labels, dense_labels in zip(signal_labels_within_masks, dense_signal_labels)]
    #
    #     else:
    #         all_signal_points = []
    #         all_signal_labels = []
    #
    #     signal_target = {}
    #     signal_target['points'] = all_signal_points
    #     signal_target['labels'] = all_signal_labels
    #
    #     vis_abs_points(all_target_points=all_signal_points, all_target_labels=all_signal_labels, rois= rois)
    #
    #     return signal_target


    def _get_src_permutation_idx(self, indices):

        roi_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return roi_idx, src_idx


    def signal_points(self, pred_coords, tgt_points, indices):

        idx = self._get_src_permutation_idx(indices)
        src_points = pred_coords[idx]
        target_points = torch.cat([target_point[t_id] for target_point, (_, t_id) in zip(tgt_points, indices)], dim=0)

        return src_points, target_points


    def signal_labels(self, pred_logits, tgt_labels, indices):

        src_logits = pred_logits
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([target_label[t_id] for target_label, (_, t_id) in zip(tgt_labels, indices)]).long()
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        return src_logits, target_classes


    def loss(self, pred_coords, pred_logits, tgt_points, tgt_labels, indices):

        src_points, target_points = self.signal_points(pred_coords, tgt_points, indices)
        src_logits, target_classes = self.signal_labels(pred_logits, tgt_labels, indices)
        loss_signal_p = self.loss_signal_points(src_points.float(), target_points.float())
        loss_signal_l = self.loss_signal_labels(src_logits.transpose(1, 2), target_classes, self.empty_weight.cuda(target_classes.device))

        loss_signal = (loss_signal_p + loss_signal_l) * self.signal_loss_weight
        print('########### loss_signal_point #############', loss_signal_p)
        print('########### loss_signal_class ##########', loss_signal)

        return loss_signal


def vis_abs_points(all_target_points, all_target_labels, rois):
    # print('==================== len =================', len(rois), len(center_heatmap_target), len(ignore_region_target))
    assert len(all_target_points) == len(all_target_labels)
    vis_points_image = np.uint8(np.zeros((800, 1280, 3)))
    randfloat = random.randint(1, 100)
    label_color_dict = {'1': (255, 0, 0), '2': (0, 255, 0)}
    for target_rel_points, traget_labels, roi in zip(all_target_points, all_target_labels, rois):
        for target_rel_point, target_label in zip(target_rel_points, traget_labels):
            if int(target_label) > 0:
                vis_points_image = cv2.circle(vis_points_image, (int(target_rel_point[0] + int(roi[0])), int(target_rel_point[1])+int(roi[1])),
                                              radius = 1, color = label_color_dict[str(int(target_label))], thickness = -1)
    for roi in rois:
        vis_points_image = cv2.rectangle(vis_points_image, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])),
                                         (255, 255, 0), 2)
    imsave('/data2/Caijt/FISH_mmdet/check_sanity/target_vis/' + str(randfloat) + '.jpg', vis_points_image)
