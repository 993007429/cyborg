#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 23:02
# @Author  : Can Cui
# @File    : cell_det_cls.py
# @Software: PyCharm
# @Comment:
import numpy as np
import torch
from torchvision import transforms


@torch.no_grad()
def predict(model, images):
    outputs = model(images)

    points = outputs['pnt_coords'][0].cpu().numpy()
    scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()
    select_idx = np.where(np.logical_and(points[:,0]<images.shape[-1],np.logical_and(points[:,1]<images.shape[-2],np.logical_and(points[:,0]>0,points[:,1]>0))))[0]
    points = points[select_idx]
    scores = scores[select_idx]

    classes = np.argmax(scores, axis=-1)
    reserved_index = classes < 4
    torch.cuda.empty_cache()
    return deduplicate(points[reserved_index], scores[reserved_index], 12)


def deduplicate(points, scores, interval):
    n = len(points)
    fused = np.full(n, False)
    result = np.zeros((0, 2))
    classes = np.array([])
    probs = np.array([])
    for i in range(n):
        if not fused[i]:
            fused_index = np.where(np.linalg.norm(points[[i]] - points[i:], 2, axis=1) < interval)[0] + i
            fused[fused_index] = True
            r_, c_ = np.where(scores[fused_index] == np.max(scores[fused_index]))
            p_ = np.max(scores[fused_index], axis=1)
            r_, c_, p_ = [r_[0]], [c_[0]], [p_[0]]
            result = np.append(result, points[fused_index[r_]], axis=0)
            classes = np.append(classes, c_)
            probs = np.append(probs, p_)
    return result, classes, probs


def cal_pdl1_np(patch_img: np.ndarray,
                mean: np.ndarray,
                std: np.ndarray,
                net: torch.nn.Module,
                device: int = 0):
    """
    :param patch_img: 一张PDL1 patch图像 H×W×3
    :param mean: 预处理均值
    :param std: 预处理方差
    :param net: 网络模型
    :param device: GPU设备号
    :param threshold: 检测的置信度阈值
    :return: 预测点的位置及类别
    """
    trans = transforms.Compose([transforms.ToTensor()])
    inp = trans(patch_img)
    for t, m, s in zip(inp, mean, std):
        t.sub_(m).div_(s)
    inp = inp[None].cuda(device)
    with torch.no_grad():
        pd_points, pd_classes, pd_probs = predict(net, inp)
    return pd_points.astype(int), pd_classes.astype(int), pd_probs.astype(float)


def predict_trt(pred_result):
    img_shape = [3,1024,1024]
    points = pred_result[0]
    scores = pred_result[1]
    select_idx = np.where(np.logical_and(points[:,0]<img_shape[-1],np.logical_and(points[:,1]<img_shape[-2],np.logical_and(points[:,0]>0,points[:,1]>0))))[0]
    points = points[select_idx]
    scores = scores[select_idx]

    classes = np.argmax(scores, axis=-1)
    reserved_index = classes < 10
    #torch.cuda.empty_cache()
    pd_points, pd_classes, pd_probs = deduplicate(points[reserved_index], scores[reserved_index], 12)
    return pd_points.astype(int), pd_classes.astype(int), pd_probs.astype(float)
