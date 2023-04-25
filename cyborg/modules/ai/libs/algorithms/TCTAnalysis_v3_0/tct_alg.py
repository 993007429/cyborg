#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/19 19:25
# @Author  : Can Cui
# @File    : tct_alg.py
# @Software: PyCharm
# @Comment:

import os
import pdb
import traceback

from .configs import TctBaseConfig
from .src.detect_tct_disk import disc_detect
from .src.cell_counting import count_cells_slide_thread
from .src.cell_cls import *
from .src.nms import non_max_suppression
from .src.utils import get_testing_feat,box_iou
from .src.cell_detect_from_patch import Detect_Cell,Dataset
from .src.cell_cls_after_detec import Classify_Cell, Classify_Cell1,image_transfer,crop,crop_padd
from .src.wsi_cls import Classify_Wsi
import torch
import numpy as np
# from torchvision.ops import roi_pool
import torchvision
from .models.resnest.resnest import resnest50
from .models.resnest.resnet_torch import resnet34_custom
from .models.ghostnet.ghostnet import ghostnet
from .models.cell_det.unet import UNetPap
from .models.dense_wsi.wsinet_lhl import denseWsiNet
from .models.dense_wsi_lct.wsinet_lhl import denseWsiNet as denseWsiNet_ln
from .models.dense_wsi_lct_nolayernorm.wsinet_lhl import denseWsiNet as denseWsiNet_noln
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model')
from torchvision import transforms
from PIL import Image
import cv2


class AlgBase:
    def __init__(self):
        self.config = TctBaseConfig()
        self.result =  {'bboxes': np.empty((0, 4), dtype=int),
                       'cell_pred': np.empty((0,), dtype=int),
                       'microbe_pred': np.empty((0,), dtype=int),
                       'cell_prob': np.empty((0,), dtype=float),
                       'microbe_prob': np.empty((0,), dtype=float),
                       'diagnosis': '',
                       'tbs_label': '',
                       'quality': 1,
                       'clarity': 1.0,
                       'cell_num': 0,
                       # zhong bao result
                       'microbe_bboxes1': np.empty((0, 4), dtype=int),
                       'microbe_pred1': np.empty((0,), dtype=int),
                       'microbe_prob1': np.empty((0,), dtype=float),
                       'cell_bboxes1': np.empty((0, 4), dtype=int),
                       'cell_prob1': np.empty((0,), dtype=float)
                       }
        self.pos_threshold = 0.0
        self.cell_det_net = None
        self.cell_net = None
        self.wsi_net = None
        self.qc_net = None
        self.microbe_net = None
        self.cell_det_func = count_cells_slide_thread
        self.cell_cls_func = detect_cells20x_microbes10x

    def load_ghostnet_qc_model(self, model_name='qc_ghostnet_5x_256', weights_name='best_epoch_90_0.0343785428626683.pth'):
        qc_net = ghostnet(in_channels=3, class_num=2)
        qc_model_path = os.path.join(model_dir, model_name, weights_name)
        qc_net_weights_dict = torch.load(qc_model_path, map_location=lambda storage, loc: storage)
        qc_net.load_state_dict(qc_net_weights_dict)
        qc_net.eval()
        return qc_net if not torch.cuda.is_available() else qc_net.cuda()

    def load_celldet_model(self, model_name='tct_unet_tiny', weights_name='weights_epoch_8100.pth'):
        cell_det_net = UNetPap(in_channels=1, n_classes=1)
        cell_det_model_path = os.path.join(model_dir, model_name, weights_name)
        cell_det_net_weights_dict = torch.load(cell_det_model_path, map_location=lambda storage, loc: storage)
        cell_det_net.load_state_dict(cell_det_net_weights_dict)
        cell_det_net.eval()
        return cell_det_net if not torch.cuda.is_available() else cell_det_net.cuda()

    def load_res34_cell_model(self, model_name='h5_gray_model_lct', weights_name='cell_weights_epoch_13.pth'):
        cell_net = resnet34_custom(in_channels=1, class_num=6)
        cell_cls_model_path = os.path.join(model_dir, model_name, weights_name)
        cell_net_weights_dict = torch.load(cell_cls_model_path, map_location=lambda storage, loc: storage)
        cell_net.load_state_dict(cell_net_weights_dict)
        cell_net.eval()
        return cell_net if not torch.cuda.is_available() else cell_net.cuda()

    def load_transformer_cell_model(self, model_name='h5_gray_model_lct', weights_name='cell_weights_epoch_13.pth'):
        cell_net = resnet34_custom(in_channels=1, class_num=6)
        cell_cls_model_path = os.path.join(model_dir, model_name, weights_name)
        cell_net_weights_dict = torch.load(cell_cls_model_path, map_location=lambda storage, loc: storage)
        cell_net.load_state_dict(cell_net_weights_dict)
        cell_net.eval()
        return cell_net if not torch.cuda.is_available() else cell_net.cuda()

    def load_resnest50_microbe_model(self, model_name='microorganism_10x_resnest50_2021-7-6', weights_name='weights_epoch_48.pth'):
    # def load_resnest50_microbe_model(self, model_name='microbe_20x_mix_resnest50_20211224', weights_name='weights_epoch_13.pth'):
        microbe_net = resnest50(in_channels=1, class_num=6)
        microbe_cls_model_path = os.path.join(model_dir, model_name, weights_name)
        microbe_cls_weights_dict = torch.load(microbe_cls_model_path, map_location=lambda storage, loc: storage)
        microbe_net.load_state_dict(microbe_cls_weights_dict)
        microbe_net.eval()
        return microbe_net if not torch.cuda.is_available() else microbe_net.cuda()

    def load_wsi_model(self, model_name='h5_gray_model_lct', weights_name='wsi_weights_epoch_13.pth'):
        wsi_net_noln = denseWsiNet_noln(class_num=6, in_channels=512, use_self='global', use_aux=False)
        wsi_net = denseWsiNet(class_num=6, in_channels=512, use_self='global', use_aux=False)
        wsi_net_ln = denseWsiNet_ln(class_num=6, in_channels=512, use_self='global', use_aux=False)
        wsi_model_path = os.path.join(model_dir, model_name, weights_name)
        wsi_net_weights_dict = torch.load(wsi_model_path, map_location=lambda storage, loc: storage)
        try:
            wsi_net_noln.load_state_dict(wsi_net_weights_dict)
            wsi_net_noln.eval()
            return wsi_net_noln if not torch.cuda.is_available() else wsi_net_noln.cuda()
        except:
            try:
                wsi_net.load_state_dict(wsi_net_weights_dict)
                wsi_net.eval()
                return wsi_net if not torch.cuda.is_available() else wsi_net.cuda()
            except:
                wsi_net_ln.load_state_dict(wsi_net_weights_dict)
                wsi_net_ln.eval()
                return wsi_net_ln if not torch.cuda.is_available() else wsi_net_ln.cuda()

    def cal_tct(self, slide, is_cs=False):
        with torch.no_grad():
            x_coords, y_coords = disc_detect(slide, self.config.is_disc_detect)

            center_coords, quality = self.cell_det_func(slide, self.cell_det_net, self.qc_net, patch_size=512, num_workers=4,
                                                        batch_size=4, x_coords=x_coords, y_coords=y_coords)

            bboxes, cell_soft, microbe_soft, feat = self.cell_cls_func(slide, center_coords,
                                                                      cell_classifier=self.cell_net,
                                                                      microbe_classifier=self.microbe_net,
                                                                      patch_size=160 * 2, overlap=40 * 2,
                                                                      batch_size=32)
            # object detect + 64*64 cell cls
            # import pdb;pdb.set_trace()

            cell_pred = np.argmax(cell_soft, axis=1)
            microbe_pred = np.argmax(microbe_soft, axis=1) if self.config.is_microbe_detect else np.zeros_like(cell_pred)

            sort_ind = np.argsort(cell_soft[:, 0])
            feat = feat[sort_ind]
            testing_feat, ratio_feat = get_testing_feat(0.0, cell_soft, feat, testing_num=128)
            testing_feat = torch.from_numpy(testing_feat).cuda().float()
            ratio_feat = torch.from_numpy(ratio_feat).cuda().float()
            _, slide_pos_prob, att_w = self.wsi_net(testing_feat, ratio_feat)
            slide_pos_prob = slide_pos_prob.cpu().data.numpy()
            slide_pred = np.argmax(slide_pos_prob)

            slide_diagnosis = '阳性' if 1 - slide_pos_prob[0, 0] > self.pos_threshold else '阴性'
            tbs_label = self.config.multi_wsi_cls_dict_reverse[slide_pred]
            if tbs_label == 'NILM' and slide_diagnosis == '阳性': tbs_label = 'ASC-US'
            if np.sum(cell_pred>0) <= self.config.min_pos_cell_threshold: slide_diagnosis = '阴性'
            if slide_diagnosis == '阴性': tbs_label = ''
            if quality == 0: tbs_label += '-样本不满意'

            if np.sum(cell_pred>0)<self.config.min_return_pos_cell:#取最阳性的前config个预测
                cell_pred[sort_ind[:self.config.min_return_pos_cell]] = np.argmax(cell_soft[sort_ind[:self.config.min_return_pos_cell], 1:], axis=1)+1

            pick_idx = np.where(np.logical_or(cell_pred > 0, microbe_pred > 0))[0]
            pick_bboxes = bboxes[pick_idx]
            cell_pred = cell_pred[pick_idx]
            microbe_pred = microbe_pred[pick_idx]

            cell_prob = np.max(cell_soft[pick_idx], axis=1)
            microbe_prob = np.max(microbe_soft[pick_idx], axis=1)

            # TBS result
            self.result['diagnosis'] = slide_diagnosis
            self.result['tbs_label'] = tbs_label
            self.result['cell_num'] = int(center_coords.shape[0])
            self.result['quality'] =  quality
            self.result['bboxes'] = pick_bboxes
            self.result['cell_pred'] = cell_pred
            self.result['cell_prob'] = cell_prob
            self.result['microbe_pred'] = microbe_pred
            self.result['microbe_prob'] = microbe_prob
            self.result['slide_pos_prob'] = slide_pos_prob
            # SORTING FOR ROI PATCH
            if is_cs:
                top_pos_idx = np.argsort(cell_soft[:, 0])[:1000]
                top_pos_prob = 1 - cell_soft[:, 0][top_pos_idx]
                top_pos_cell_bboxes = bboxes[top_pos_idx]

                top_microbe_idx = np.where(microbe_pred > 0)[0]
                top_microbe_bboxes = pick_bboxes[top_microbe_idx]
                top_microbe_pred = microbe_pred[top_microbe_idx]
                top_microbe_prob = microbe_prob[top_microbe_idx]

                if self.config.is_nms:
                    if top_microbe_bboxes.shape[0] > 0:
                        microbe_pick_idx = non_max_suppression(top_microbe_bboxes, top_microbe_prob, 0.0)
                        top_microbe_bboxes = top_microbe_bboxes[microbe_pick_idx]
                        top_microbe_pred = top_microbe_pred[microbe_pick_idx]
                        top_microbe_prob = top_microbe_prob[microbe_pick_idx]
                    if top_pos_cell_bboxes.shape[0] > 0:
                        cell_pick_idx = non_max_suppression(top_pos_cell_bboxes, top_pos_prob, 0.0)
                        top_pos_cell_bboxes = top_pos_cell_bboxes[cell_pick_idx]
                        top_pos_prob = top_pos_prob[cell_pick_idx]
                # ZHONG BAO result
                self.result['microbe_bboxes1'] = top_microbe_bboxes
                self.result['microbe_prob1'] = top_microbe_prob
                self.result['microbe_pred1'] = top_microbe_pred
                self.result['cell_bboxes1'] = top_pos_cell_bboxes
                self.result['cell_prob1'] = top_pos_prob
            return self.result


'''结合细胞检测的 通用模型'''
class KWW_ALG(AlgBase):
    def __init__(self, threshold=None):
        super(KWW_ALG, self).__init__()

        self.pos_threshold = 0.15 if threshold is None else threshold

        self.device = torch.device('cuda')

        self.patch_det_net = self.load_celldet_model()
        self.patch_net = self.load_res34_cell_model(model_name='h5_gray_model_lct',
                                                   weights_name='cell_weights_epoch_13.pth')
        self.qc_net = self.load_ghostnet_qc_model() if self.config.is_qc else None
        self.patch_det_func = self.cell_det_func

        self.cell_detector = Detect_Cell(os.path.join(model_dir, "base_with_objdete", "cell_detector.torchscript"), device=self.device)
        self.cell_classification = Classify_Cell(os.path.join(model_dir, "base_with_objdete", "cell_fm.torchscript"),os.path.join(model_dir, "base_with_objdete", "cell_fc.torchscript"),device=self.device)
        self.wsi_classification = Classify_Wsi(os.path.join(model_dir, "base_with_objdete", "wsi_class.torchscript"),device=self.device)
        del self.cell_det_func

    def cal_tct(self, slide, is_cs=False):
        with torch.no_grad():
            x_coords, y_coords = disc_detect(slide, self.config.is_disc_detect)
            center_coords, quality = self.patch_det_func(slide, self.patch_det_net, self.qc_net, patch_size=512,
                                                        num_workers=4,
                                                        batch_size=4, x_coords=x_coords, y_coords=y_coords)
            # 返回的是 ROI在全场图坐标，ROI阳性置信度，20倍下crop出来的ROI; 有1024个
            roi_bbox,roi_conf_bag,roi_bag = self.roi_cls_func(slide, center_coords,
                                        cell_classifier=self.patch_net,
                                        microbe_classifier=self.microbe_net,
                                        patch_size=160 * 2, overlap=40 * 2,
                                        batch_size=32)
            # import cv2;cv2.imwrite(r"D:/tmp.jpg",np.uint8(roi_bag[0].cpu().numpy()))
            roi_bag = roi_bag.permute(0,3,1,2) #slide.read读出来是rgb

            topK = 256
            cell_bboxes, cells_conf = self.cell_detector.infer(roi_bag, conf_thres=0.01)

            #FIXME using roi_pool to resize cells
            if False:
                crop_cells = roi_pool(roi_bag.float(), cell_bboxes, output_size=64)  # [k,3,64,64]
            #TODO using crop&pad to resize cells
            if True:
                output_size = 64
                crop_cells = torch.ones((len(cells_conf), 3, output_size, output_size), dtype=torch.uint8,device=roi_bag.device)
                crop_cells *= 255
                count = 0
                for img_id,bboxes in enumerate(cell_bboxes):
                    for bbox in bboxes:
                        bbox = bbox.int()
                        crop_cell = roi_bag[img_id, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        _, h, w = crop_cell.shape
                        center_y, center_x = h // 2, w // 2
                        new_h, new_w = min(h, output_size), min(w, output_size)
                        output_center_y, output_center_x = output_size // 2, output_size // 2
                        crop_cells[count, :, output_center_y - new_h // 2:output_center_y - new_h // 2 + new_h,\
                            output_center_x - new_w // 2:output_center_x - new_w // 2 + new_w] = \
                            crop_cell[:, center_y - new_h // 2:center_y - new_h // 2 + new_h,\
                            center_x - new_w // 2:center_x - new_w // 2 + new_w]
                        count+=1
            # import cv2;cv2.imwrite(r"D:/tmp.jpg",wsi_input[0].cpu().numpy().transpose((1, 2, 0))[:,:,::-1])

            # 用于检测结果可视化，先把检测的cell坐标从roi映射回全场图
            # 根据selected_idx 取前topK个，再根据后续的分类结果取


            # FIXME: 用于可视化在ROI上的细胞检测结果
            if False:
                for roi_id,roi_bboxes in enumerate(cell_bboxes):
                    image = roi_bag[roi_id]
                    image = image.cpu().numpy().transpose((1,2,0))[:,:,::-1].astype(np.uint8)
                    roi_bboxes = roi_bboxes.int().cpu().numpy()
                    for roi_bbox in roi_bboxes:
                        image = cv2.rectangle(image, (roi_bbox[0], roi_bbox[1]), (roi_bbox[2], roi_bbox[3]), (255, 0, 255), 2)
                    cv2.imwrite(r"D:/tmp.jpg",image)
                    a = input()

            # FIXME:计算细胞框映射回wsi的位置
            standard_mpp = 0.242042
            this_mpp = slide.mpp
            scale_ratio = standard_mpp * 2 / this_mpp #从20倍映射回40倍
            for roi_idx in range(len(cell_bboxes)):
                roi_x, roi_y = roi_bbox[roi_idx, :2]
                cell_bboxes[roi_idx] = cell_bboxes[roi_idx]*scale_ratio
                cell_bboxes[roi_idx][:,::2] = cell_bboxes[roi_idx][:,::2]+roi_x
                cell_bboxes[roi_idx][:, 1::2] = cell_bboxes[roi_idx][:, 1::2] + roi_y
            bboxes = torch.cat(cell_bboxes)

            # FIXME: 对ROI间的细胞作nms抑制后挑前topK 的细胞以及全场图坐标 or 不作nms抑制
            #  只有当训练的时候对ROI间的细胞作了nms抑制后才使用
            if False:
                i = torchvision.ops.nms(bboxes, cells_conf, 0.3)  # NMS
                i = i[:topK]
                bboxes = bboxes[i]
                wsi_input = crop_cells[i]
            else:
                select_count = min(topK, len(crop_cells))
                _, selected_idx = cells_conf.topk(select_count, dim=0)
                crop_cells = crop_cells[selected_idx]
                bboxes = bboxes[selected_idx]  # 用于后续的每个细胞的框定位
                if False: # 特征提取前补全到256
                    wsi_input = torch.zeros((topK, 3, 64, 64), device=crop_cells.device, dtype=crop_cells.dtype) #不足补0,把补0也放在特征提取前，可以在后续wsi分类的时候补
                    wsi_input[:len(crop_cells)] = crop_cells
                else:
                    wsi_input = crop_cells


            # FIXME: 用于可视化在wsi上的细胞检测结果
            if False:
                bboxes = bboxes.int().cpu().numpy()
                for bbox in bboxes:
                    location = (bbox[0],bbox[1])
                    bbox_size = (bbox[2]-bbox[0],bbox[3]-bbox[1]) #(w,h)
                    image = slide.read(location, bbox_size, scale_ratio)
                    cv2.imwrite(r"D:/tmp1.jpg",image)
                    a=input()


            fm, probs = self.cell_classification.infer(wsi_input)


            if True:# 挑选置信度>0.5的细胞
                selected_idx = (1-probs[:, 0]) > 0.5
                fm = fm[selected_idx]
                probs = probs[selected_idx]
                bboxes = bboxes[selected_idx]


            select_count = min(topK//2, len(probs))
            _, selected_idx = (1-probs[:, 0]).topk(select_count, dim=0, sorted=False)

            fm = fm[selected_idx]
            probs = probs[selected_idx]
            bboxes = bboxes[selected_idx]

            # FIXME 在wsi分类前才做padding
            padding_num = topK//2 - select_count
            if padding_num>0:
                tmp_fm = torch.zeros((padding_num,*fm.shape[1:]),device=self.device)
                fm = torch.cat((fm,tmp_fm))

            fm = torch.unsqueeze(fm, 0)
            _, slide_pos_prob, att_w = self.wsi_classification.infer(fm)
            slide_pos_prob = slide_pos_prob.cpu().data.numpy()
            slide_pred = np.argmax(slide_pos_prob)

            slide_diagnosis = '阳性' if 1 - slide_pos_prob[0, 0] > self.pos_threshold else '阴性'
            tbs_pred_name = self.config.multi_wsi_cls_dict_reverse[slide_pred]
            if tbs_pred_name == 'NILM' and slide_diagnosis == '阳性': tbs_pred_name = 'ASC-US'

            # 细胞检测后，分类的阳性细胞个数（因为是2分类）,不取padding的分类结果
            if False: # FIXME 在特征提取前就padding，如果是在wsi_input输入的时候padding就不需要取前n个了
                probs = probs[:len(bboxes)]
                _, selected_idx = (1-probs[:, 0]).topk(min(len(bboxes),128), dim=0, sorted=False) #去除padding后的prob_ID
                probs = probs[selected_idx]   #根据阳性前128的置信度分布
                bboxes = bboxes[selected_idx] #根据分类结果取前128个框

            cell_pred = torch.argmax(probs,dim=1) #最阳性的类别
            # import pdb;pdb.set_trace()
            # torch.save(wsi_input,r"D:/p%N_l%H.pt")
            positive_cell_preds_nums = (cell_pred>0).sum().cpu().item()
            if  positive_cell_preds_nums<= self.config.min_pos_cell_threshold: slide_diagnosis = '阴性'
            if slide_diagnosis == '阴性': tbs_label = ''
            if quality == 0: tbs_label += '-样本不满意' #这个可以提前，节省后续的时间

            # 取最阳性的前config个细胞？
            # if positive_cell_preds_nums < self.config.min_return_pos_cell:

            pick_index = cell_pred>0
            cell_pred = cell_pred[pick_index].cpu().numpy()
            pick_bboxes = bboxes[pick_index].int().cpu().numpy()

            cell_prob = torch.max(probs[pick_index],dim=1)[0].cpu().numpy()

            # TBS result
            self.result['diagnosis'] = slide_diagnosis
            self.result['tbs_label'] = tbs_pred_name
            self.result['cell_num'] = int(center_coords.shape[0]) #返回了有多少个ROI
            self.result['quality'] = quality
            self.result['bboxes'] = pick_bboxes
            self.result['cell_pred'] = cell_pred
            self.result['cell_prob'] = cell_prob
            self.result['slide_pos_prob'] = slide_pos_prob

            # if is_cs:
            #TODO: 取前1000，然后做nms抑制
        return self.result


    def roi_cls_func(self,slide, center_coords, cell_classifier=None, microbe_classifier=None,
                          patch_size=160, overlap=40, batch_size=128, num_workers=4):
        crop_queue = Queue()
        process_queue = Queue(batch_size * 8)

        # 都是40倍，不同扫描仪的视野大小也有区别，需要重新计算
        standard_mpp = 0.242042 * 2
        this_mpp = slide.mpp
        scale_ratio = standard_mpp / this_mpp
        lvl0_crop_size = int(patch_size * scale_ratio)
        lvl0_overlap = int(overlap * scale_ratio)
        lvl0_stride = lvl0_crop_size - lvl0_overlap
        # ROI的patch坐标 ， openslide 先读一个大区域放到内存，再从大区域里读取patch
        bboxes = sampling_bboxes(center_coords, lvl0_stride, lvl0_crop_size, border=[0, 0, slide.width, slide.height])
        num_bboxes = bboxes.shape[0]
        patch_bboxes_list = match_bboxes_with_pathces(bboxes=bboxes, crop_size=5120)

        for i in range(len(patch_bboxes_list)):
            patch_coord, bbox_coord = patch_bboxes_list[i]
            xs, ys = patch_coord[0], patch_coord[1]
            w, h = patch_coord[2] - patch_coord[0], patch_coord[3] - patch_coord[1]
            crop_queue.put(SlideRegion(location=(xs, ys), size=(w, h), scale=scale_ratio, sub_coords=bbox_coord))

        for i in range(num_workers):
            t = Thread(target=read_patch_worker_10x_20x, args=(slide, crop_queue, process_queue, patch_size))
            t.start()

        batch_data_list = []
        batch_oridata20x_list = []
        batch_oridata40x_list = []
        bbox_list = []
        out_bbox = None
        out_conf = None
        out_bags = None
        topK = 1024
        for i in range(num_bboxes):
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)
                batch_oridata20x_list.append(slide_region.ori_image)
                bbox_list.append(slide_region.sub_coords)
            if len(batch_data_list) == batch_size or (i == num_bboxes - 1 and len(batch_data_list)>0):
                data_variable = torch.tensor(np.ascontiguousarray(batch_data_list), dtype=torch.float, device=torch.device('cuda'))
                tmp_bags = torch.tensor(np.ascontiguousarray(batch_oridata20x_list), dtype=torch.float, device=torch.device('cuda'))
                tmp_bboxes = torch.tensor(np.ascontiguousarray(bbox_list), dtype=torch.float, device=torch.device('cuda'))
                with torch.no_grad():
                    _, feat, cell_soft = cell_classifier(data_variable)
                if out_conf is None:
                    out_conf = 1-cell_soft[:,0]
                    out_bags = tmp_bags
                    out_bbox = tmp_bboxes
                else:
                    out_conf = torch.cat((out_conf,1-cell_soft[:,0]))
                    out_bags = torch.cat((out_bags,tmp_bags))
                    out_bbox = torch.cat((out_bbox, tmp_bboxes))
                if len(out_conf)>topK and len(out_conf)==len(out_bags):
                    out_conf,selected_idx = out_conf.topk(topK,dim=0)
                    out_bags = out_bags[selected_idx]
                    out_bbox = out_bbox[selected_idx]
                batch_data_list = []
                batch_oridata20x_list = []
                bbox_list = []
                # print(i,"/",num_bboxes,"====",out_bags.shape)
        return out_bbox,out_conf,out_bags

#FIXME 结合细胞检测 通用模型2.2版本
class KWW_ALG_2(object):
    def __init__(self, threshold=None):
        super(KWW_ALG_2, self).__init__()

        self.pos_threshold = 0.15 if threshold is None else threshold

        self.device = torch.device('cuda')
        self.crop_len = 1024  #20倍下的大小
        self.overlap = 256
        self.detec_topK = 256
        self.max_detec_cells = 10000
        self.detect_batch_size = 1
        self.num_workers = 0
        self.cell_shape = 224

        self.cell_detector = Detect_Cell(os.path.join(model_dir, "base_with_objdete", "cell_detector.torchscript"), input_size=1024,device=self.device)
        self.cell_classification = Classify_Cell(os.path.join(model_dir, "base_with_objdete", "cell_fm.torchscript"),os.path.join(model_dir, "base_with_objdete", "cell_fc.torchscript"),device=self.device)
        self.wsi_classification = Classify_Wsi(os.path.join(model_dir, "base_with_objdete", "wsi_class.torchscript"),input_size=256,device=self.device)

    def detec_cell_from_slide_after(self,slide,disc_coords,scale_ratio):

        crop_len = int(self.crop_len * scale_ratio)
        overlap = int(self.overlap * scale_ratio)
        stride = crop_len - overlap

        xmin, ymin, xmax, ymax = disc_coords
        bboxes = []
        for y in range(ymin, ymax, stride):
            for x in range(xmin, xmax, stride):
                bbox_x1, bbox_y1 = x, y
                bbox_x2, bbox_y2 = min(xmax, x + crop_len), min(y + crop_len, ymax)
                bboxes.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])
        roi_dataset = Dataset(slide, bboxes, scale_ratio, (self.crop_len, self.crop_len))
        sampler = None
        roi_loader = torch.utils.data.DataLoader(
            roi_dataset,
            batch_size=self.detect_batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=True,
            drop_last=False
        )
        wsi_bboxes = []
        wsi_confs = []
        wsi_cells = []

        for iter_id, batch in enumerate(roi_loader):
            # if RANK in [-1, 0]:
            #     print( iter_id, "/", len(roi_loader))
            roi_bag = batch['cur_region'].to(device=self.cell_detector.device, non_blocking=True)
            cell_bboxes, cells_conf = self.cell_detector.infer(roi_bag, conf_thres=0.01)

            # FIXME:计算细胞框映射回wsi的位置
            roi_bbox = batch['tile_coords'].to(device=self.device, non_blocking=True)
            for roi_idx in range(len(cell_bboxes)):
                roi_x, roi_y = roi_bbox[roi_idx, :2]
                cell_bboxes[roi_idx] = cell_bboxes[roi_idx] * scale_ratio
                cell_bboxes[roi_idx][:, ::2] = cell_bboxes[roi_idx][:, ::2] + roi_x
                cell_bboxes[roi_idx][:, 1::2] = cell_bboxes[roi_idx][:, 1::2] + roi_y
                # 怕有在padding范围内的细胞框
                cell_bboxes[roi_idx][:, 0].clamp_(roi_bbox[roi_idx][0], roi_bbox[roi_idx][2])  # x1
                cell_bboxes[roi_idx][:, 1].clamp_(roi_bbox[roi_idx][1], roi_bbox[roi_idx][3])  # y1
                cell_bboxes[roi_idx][:, 2].clamp_(roi_bbox[roi_idx][0], roi_bbox[roi_idx][2])  # x2
                cell_bboxes[roi_idx][:, 3].clamp_(roi_bbox[roi_idx][1], roi_bbox[roi_idx][3])  # y2

            bboxes = torch.cat(cell_bboxes)
            bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            index = bboxes_area != 0  # 去掉padding范围的细胞框
            bboxes = bboxes[index]
            # crop_cells = crop_cells[index]
            cells_conf = cells_conf[index]

            # FIXME 没有多卡通信
            # if WORLD_SIZE > 1:
            #     wsi_bboxes_ = [None for _ in range(WORLD_SIZE)]
            #     wsi_confs_ = [None for _ in range(WORLD_SIZE)]
            #     wsi_cells_ = [None for _ in range(WORLD_SIZE)]
            #     dist.all_gather_object(wsi_bboxes_, bboxes.cpu())
            #     dist.all_gather_object(wsi_cells_, crop_cells.cpu())
            #     dist.all_gather_object(wsi_confs_, cells_conf.cpu())
            # else:
            wsi_bboxes_ = [bboxes]
            wsi_confs_ = [cells_conf]
            # wsi_cells_ = [crop_cells]

            wsi_bboxes.extend(wsi_bboxes_)
            wsi_confs.extend(wsi_confs_)
            # wsi_cells.extend(wsi_cells_)

        # if RANK in [-1, 0]:
        wsi_bboxes = torch.cat(wsi_bboxes).to(self.device)
        wsi_confs = torch.cat(wsi_confs).to(self.device)
        # wsi_cells = torch.cat(wsi_cells).to(self.device)
        n = wsi_confs.shape[0]
        if not n:
            print("detect not cell on the wsi！！！！")
        elif n > self.max_detec_cells:
            i = wsi_confs.argsort(descending=True)[:self.max_detec_cells]
            wsi_confs = wsi_confs[i]
            wsi_bboxes = wsi_bboxes[i]

        i = torchvision.ops.nms(wsi_bboxes, wsi_confs, 0.3)  # NMS
        if False: # just select the max 256
            i = i[:self.detec_topK]
        wsi_bboxes = wsi_bboxes[i]
        wsi_confs = wsi_confs[i]
        # wsi_cells = wsi_cells[i] #[TOPK,4,h,w] BGR img = img.cpu().numpy().transpose((1,2,0))[...,::-1]

        return wsi_bboxes,wsi_confs

    def cal_tct(self, slide, is_cs=False):

        RANK = -1
        standard_mpp = 0.242042 * 2
        this_mpp = slide.mpp
        scale_ratio = standard_mpp / this_mpp

        wsi_pt_path = os.path.join(os.path.dirname(slide.filename), "detect_result.pt")
        # wsi_pt_path = r"E:/tmp.pt"
        if not os.path.exists(wsi_pt_path) or True:
            w, h = slide.width, slide.height
            try: # disc_detect 有时候会检测不到圆盘，返回空的x_coords,y_coords
                x_coords, y_coords = disc_detect(slide, self.config.is_disc_detect) # 圆盘检测 #最大层的坐标
                xmin, xmax = int(min(x_coords)), int(max(x_coords))
                ymin, ymax = int(min(y_coords)), int(max(y_coords))
                xmin, xmax = min(max(0, xmin), w), min(max(0, xmax), w)
                ymin, ymax = min(max(0, ymin), h), min(max(0, ymax), h)
            except:
                xmin,ymin,xmax,ymax = 0,0,w,h

            wsi_bboxes,wsi_confs = self.detec_cell_from_slide_after(slide,[xmin,ymin,xmax,ymax],scale_ratio)

            if RANK in [-1, 0]:
                #TODO 保存检测的中间结果：
                wsi_pt_path = os.path.join(os.path.dirname(slide.filename),"detect_result.pt")
                # dete_result = {"bboxes":wsi_bboxes.cpu(),"confs":wsi_confs.cpu(),"wsi_cells":wsi_cells.cpu()}
                dete_result = {"bboxes": wsi_bboxes.cpu(), "confs": wsi_confs.cpu()}
                torch.save(dete_result,wsi_pt_path)
                print(slide.filename," finished!!!!")
        else:
            dete_result = torch.load(wsi_pt_path,map_location=torch.device('cpu'))
            wsi_bboxes = dete_result['bboxes']
            wsi_confs = dete_result['confs'] #检测的置信度

        match_with_label_bboxes = None
        if False: # cal the iou between all pred bboxes and the doctor's label result。
            from diagnosis.tasks.utils.tct_utils import get_human_tl  # 用于读取已有的标签
            label_coords = get_human_tl(slide.filename)
            label_coords = torch.from_numpy(np.array(label_coords))
            if len(wsi_bboxes)>self.detec_topK and len(label_coords):
                match_with_label_bboxes = wsi_bboxes[self.detec_topK:]
                ious = box_iou(match_with_label_bboxes,label_coords)
                max_ious = torch.max(ious,dim=1)[0]
                match_idx = max_ious>0.1
                match_with_label_bboxes = match_with_label_bboxes[match_idx,:]



        # save all bboxes coords to pt file and select here
        wsi_bboxes = wsi_bboxes[:self.detec_topK]
        wsi_confs = wsi_confs[:self.detec_topK]

        if match_with_label_bboxes is not None:
            wsi_bboxes = torch.cat((wsi_bboxes,match_with_label_bboxes))

        # 制片不及格+检测器检测到太少的细胞，直接返回
        if wsi_bboxes.shape[0]<10:
            self.result['diagnosis'] = '样本不满意'
            self.result['tbs_label'] = "样本不满意"
            self.result['cell_num'] = 0
            self.result['quality'] = 0
            self.result['slide_pos_prob'] = [[1,0,0,0,0,0]]
            return self.result



        #FIXME 把40or20倍下的细胞crop出来保存，转化为transformer的输入shape
        wsi_cells_path = os.path.join(os.path.dirname(slide.filename), "wsi_cells_20X.pt") #20be
        if not os.path.exists(wsi_cells_path) or True:
            # 把测试集的细胞图像保存下来，放到linux上测试
            save_deploy_img_flag = False
            if save_deploy_img_flag:
                import re
                wsi_name = os.path.basename(slide.filename)
                fname = os.path.splitext(wsi_name)[0]
                # fname = re.sub('[\u4e00-\u9fa5]', '', fname)
                # if "G219978" in fname:
                #     fname = "G219978"
                # if fname in ["10","17","18","24","26","28","32","39","40","49","545","560","624","632","70","72"]:
                #     fname = slide.filename.split("\\")[-2]+"_"+fname
                # save_dir = os.path.join(os.path.dirname(slide.filename),"cell_detection_imgs_20X")
                fname = slide.filename.split("\\")[-2] + "_" + fname
                save_dir = r"D:/new_test_img_20X"
                os.makedirs(save_dir, exist_ok=True)
                # the name of the middle result of a wsi without wsi_label
                if False:
                    category_change = {"N":"0","ASC-US":"1","LSIL":"2","ASC-H":"3","HSIL":"4","AGC":"5"}
                    labels_path = r"D:/data/适配/新疆标签.xlsx"
                    import pandas as pd
                    labels_dic = pd.read_excel(labels_path)
                    wsi_names = list(labels_dic["切片名称"])
                    tmp_labels = labels_dic["阳性分级"]
                    labels_dic = {}
                    for iter_id, wsi_name in enumerate(wsi_names):
                        labels_dic[wsi_name] = category_change[tmp_labels[iter_id]]
                    # fname = os.path.splitext(wsi_name)[0].replace("Result-","")
                    label = labels_dic[wsi_name]
                    output_dir = label + "_" + fname
                else:
                    output_dir = fname
                hospital_name = slide.filename.split("\\")[2]
                # hospital_name = re.sub('[\u4e00-\u9fa5]', 'C', hospital_name)
                save_dir = os.path.join(save_dir,hospital_name)
                os.makedirs(save_dir, exist_ok=True)
                save_dir = os.path.join(save_dir,output_dir)
                os.makedirs(save_dir, exist_ok=True)

            wsi_cells = []
            masks = []
            data = {}
            for tmp_id,bbox in enumerate(wsi_bboxes):
                xmin, ymin, xmax, ymax = bbox.int().cpu().numpy().tolist()
                cell_w, cell_h = xmax - xmin, ymax - ymin
                try:
                    crop_cell = np.array(slide.read((xmin, ymin), (cell_w, cell_h), scale_ratio))  # RGB
                    cell_h,cell_w = crop_cell.shape[:2] #20X
                    if save_deploy_img_flag and tmp_id<self.detec_topK:
                        cv2.imencode(".png",crop_cell.copy()[:,:,::-1])[1].tofile(os.path.join(save_dir,str(tmp_id)+".png"))
                        # cv2.imwrite(os.path.join(save_dir,str(tmp_id)+".png"),crop_cell.copy()[:,:,::-1])

                    cell_h,cell_w = cell_h * 2, cell_w * 2 #40X
                    crop_cell = cv2.resize(crop_cell, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)#20X->40X

                    # crop_cell = crop(crop_cell, self.cell_shape, self.cell_shape)
                    # import pdb;
                    # pdb.set_trace()
                    crop_cell = crop_padd(crop_cell, self.cell_shape, self.cell_shape, "white")[..., :3]
                    data[tmp_id] = torch.from_numpy(crop_cell)
                    # crop_cell = crop_cell[:, :, ::-1]  # RGB->BGR
                    crop_cell = (crop_cell.astype(np.float32) / 255.)
                    crop_cell = (crop_cell - 0.5) / 0.5
                    crop_cell = crop_cell.transpose(2, 0, 1)

                    # crop_cell, mask = image_transfer(crop_cell, (self.cell_shape, self.cell_shape), "white")
                    crop_cell = crop_cell[np.newaxis, ...]
                    crop_cell = torch.from_numpy(crop_cell)
                    # mask = mask[np.newaxis, ...]
                    wsi_cells.append(crop_cell)
                    # masks.append(mask)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    continue
            wsi_cells = torch.cat(wsi_cells)

            # masks = torch.cat(masks)

            # data = {"wsi_cells":wsi_cells,"masks":masks}
            data["wsi_cells"] = wsi_cells
            # data["masks"] = masks
            torch.save(data, wsi_cells_path)
        else:
            data = torch.load(wsi_cells_path, map_location=torch.device('cpu'))
            wsi_cells = data['wsi_cells']
            # masks = data['masks']  # 检测的置信度

        # FIXME 细胞分类
        two_step_flag = True
        fm_list,probs_list = [],[]
        if two_step_flag:
            postive_probs_list = []
        for wsi_cells_index in range(0,wsi_cells.shape[0],256):
            batch_wsi_cells = wsi_cells[wsi_cells_index:wsi_cells_index + 256].to(self.cell_classification.device)
            # batch_masks = masks[wsi_cells_index:wsi_cells_index + 256].to(self.cell_classification.device)
            # fm, probs = self.cell_classification.infer(batch_wsi_cells, batch_masks, normal=False)
            fm, probs = self.cell_classification.infer(batch_wsi_cells, normal=False)
            fm_list.append(fm)
            if two_step_flag:
                probs, postive_probs = probs
                probs_list.append(probs)
                postive_probs_list.append(postive_probs)
            else:
                probs_list.append(probs)

        fm = torch.cat(fm_list, 0)
        probs = torch.cat(probs_list, 0)
        if two_step_flag:
            postive_probs = torch.cat(postive_probs_list, 0)

        match_bboxes,match_cell_pred = None,None
        if len(probs)>self.detec_topK:
            match_probs = probs[self.detec_topK:]
            probs = probs[:self.detec_topK]
            if two_step_flag:
                match_postive_probs = postive_probs[self.detec_topK:]
                postive_probs = postive_probs[:self.detec_topK]
            match_bboxes = wsi_bboxes[self.detec_topK:]
            wsi_bboxes = wsi_bboxes[:self.detec_topK]
            fm = fm[:self.detec_topK]

            # 对匹配到标签的预测进行阳性筛选：
            # 导致假阴的情况（1. 细胞检测没有检测到，2.细胞分类分成阴性了）
            selected_idx = (1 - match_probs[:, 0]) > 0.5
            if two_step_flag:
                match_postive_probs = match_postive_probs[selected_idx]
                match_bboxes= match_bboxes[selected_idx]
                if len(match_bboxes):
                    if two_step_flag :
                        match_cell_pred = torch.argmax(match_postive_probs, dim=1) + 1
                    else:
                        match_cell_pred = torch.argmax(match_probs[:, 1:], dim=1) + 1

        if True:  # 挑选置信度>0.5的细胞
            selected_idx = (1-probs[:, 0]) > 0.5
            fm = fm[selected_idx]
            probs = probs[selected_idx]
            bboxes = wsi_bboxes[selected_idx]
            if two_step_flag:
                postive_probs = postive_probs[selected_idx]

        select_count = min(self.detec_topK // 2, len(probs))
        _, selected_idx = (1 - probs[:, 0]).topk(select_count, dim=0, sorted=False)
        fm = fm[selected_idx]
        probs = probs[selected_idx]
        if two_step_flag:
            postive_probs = postive_probs[selected_idx]
        bboxes = bboxes[selected_idx]

        padding_num = self.detec_topK // 2 - select_count
        if padding_num > 0:
            tmp_fm = torch.zeros((padding_num, *fm.shape[1:]), device=self.device)
            fm = torch.cat((fm, tmp_fm))
        fm = torch.unsqueeze(fm, 0)

        _, slide_pos_prob, att_w = self.wsi_classification.infer(fm)
        # import pdb;pdb.set_trace()

        slide_pos_prob = slide_pos_prob.cpu().data.numpy()
        slide_pred = np.argmax(slide_pos_prob)

        slide_diagnosis = '阳性' if 1 - slide_pos_prob[0, 0] > self.pos_threshold else '阴性'
        tbs_label = self.config.multi_wsi_cls_dict_reverse[slide_pred]
        if tbs_label == 'NILM' and slide_diagnosis == '阳性': tbs_label = 'ASC-US'

        # 细胞分类后没有任何阳性细胞
        if int(bboxes.shape[0])==0:
            self.result['diagnosis'] = slide_diagnosis

        if match_bboxes is not None and match_cell_pred is not None:
            bboxes = torch.cat((bboxes,match_bboxes))
        bboxes = bboxes.cpu().numpy()
        if len(bboxes):
            if two_step_flag:
                cell_pred = torch.argmax(postive_probs, dim=1) + 1
                # asc-h 与 hsil合并后的处理
                agc_index = (cell_pred==4)
                cell_pred[agc_index] += 1
                hsil_index = (postive_probs[:,2]>0.9)
                cell_pred[hsil_index]+=1
            else:
                cell_pred = torch.argmax(probs[:,1:],dim=1)+1
            if match_cell_pred is not None:
                cell_pred = torch.cat((cell_pred,match_cell_pred))
                if two_step_flag:
                    postive_probs = torch.cat((postive_probs,match_postive_probs))
                else:
                    probs = torch.cat((probs,match_probs))
            cell_pred = cell_pred.cpu().numpy()
            pick_idx = np.where(cell_pred > 0)[0]
            pick_bboxes = bboxes[pick_idx]
            cell_pred = cell_pred[pick_idx]
            if two_step_flag:
                cell_prob = np.max(postive_probs.cpu().numpy()[pick_idx], axis=1)
            else:
                cell_prob = np.max(probs[:, 1:].cpu().numpy()[pick_idx], axis=1)

            self.result['cell_num'] = int(bboxes.shape[0])
            self.result['bboxes'] = pick_bboxes
            self.result['cell_pred'] = cell_pred
            self.result['cell_prob'] = cell_prob

        # if np.sum(cell_pred > 0) <= self.config.min_pos_cell_threshold: slide_diagnosis = '阴性'
        if slide_diagnosis == '阴性': tbs_label = ''
        quality = 1


        # TBS result
        self.result['diagnosis'] = slide_diagnosis
        self.result['tbs_label'] = tbs_label
        self.result['quality'] = quality
        self.result['slide_pos_prob'] = slide_pos_prob

        return self.result

