#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/19 19:25
# @Author  : Can Cui
# @File    : tct_alg.py
# @Software: PyCharm
# @Comment:

import os
from .configs import TctBaseConfig
from .src.detect_tct_disk import disc_detect
from .src.cell_counting import *
from .src.cell_cls import *
from .src.nms import non_max_suppression
from .src.utils import get_testing_feat
import torch
import numpy as np
import traceback
from .models.cell_det.unet import UNetPap
from .models.dense_wsi.wsinet_lhl import denseWsiNet
from .models.dense_wsi_lct.wsinet_lhl import denseWsiNet as denseWsiNet_ln
from .models.dense_wsi_lct_nolayernorm.wsinet_lhl import denseWsiNet as denseWsiNet_noln
from .models.convnext.convnext_microbe import ConvNeXt as convnext_microbe
from torchvision.models.mobilenetv3 import mobilenet_v3_small
use_trt = True
try:
    from torch2trt import torch2trt
    from torch2trt import TRTModule
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
except:
    print(traceback.format_exc())
    print("[WARNING!]  Loading TensorRT failed, use Pytorch to do the inference.")
    use_trt = False

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model')

class AlgBase:
    def __init__(self):
        self.config = TctBaseConfig()
        self.result = {'bboxes': np.empty((0, 4), dtype=int),
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
        self.cell_det_net=None
        self.wsi_net=None
        self.microbe_net=None
        self.cell_net=None
        self.qc_net1=None
        self.microbe_qc_net=None
        self.cell_cls_func = detect_mix20x_scale1_qc3
        self.cell_det_func = count_cells_slide_thread_noqc
        self.pos_threshold = None

    def load_celldet_model(self, model_name='tct_unet_tiny', weights_name='weights_epoch_8100.pth'):

        model_path = os.path.join(model_dir, model_name, weights_name)
        trt_model_path = os.path.splitext(model_path)[0]+'.cc'
        if os.path.exists(trt_model_path) and use_trt:
            print('load:{}'.format(trt_model_path))
            model = TRTModule()
            model.load_state_dict(torch.load(trt_model_path))
            return model
        else:
            model = UNetPap(in_channels=1, n_classes=1)
            model_weights_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(model_weights_dict)
            model.eval()
            model.cuda()
            if use_trt and not os.path.exists(trt_model_path):
                model =  torch2trt(model, [torch.zeros(1,1,512,512).cuda()], max_batch_size=8, fp16_mode=True)
                torch.save(model.state_dict(), trt_model_path)
            return model

    def load_wsi_model_768_global(self, model_name='convnext_2fc', weights_name='wsi_weights_epoch_2.pth'):
        wsi_net_noln = denseWsiNet_noln(class_num=6, in_channels=768, use_aux=False, use_self='global')
        wsi_net = denseWsiNet(class_num=6, in_channels=768, use_aux=False, use_self='global')
        wsi_net_ln = denseWsiNet_ln(class_num=6, in_channels=768, use_aux=False, use_self='global')
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

    def load_cell0921_model(self, model_name='microbe_20x_convnext_rgb_20220704',
                            weights_name='checkpoint-best-ema.pth'):
        model_path = os.path.join(model_dir, model_name, weights_name)
        trt_model_path = os.path.splitext(model_path)[0]+'.cc'
        # print(trt_model_path)
        if os.path.exists(trt_model_path) and use_trt:
            print('load:{}'.format(trt_model_path))
            model = TRTModule()
            model.load_state_dict(torch.load(trt_model_path))
            return model
        else:
            model = convnext_microbe(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], in_chans=3, num_classes=5)
            model_weights_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            if 'model' in model_weights_dict:
                # torch.save(model_weights_dict['model_ema'], model_path)
                model_weights_dict = model_weights_dict['model']
            elif 'model_ema' in model_weights_dict:
                # torch.save(model_weights_dict['model'], model_path)
                model_weights_dict = model_weights_dict['model_ema']

            # if 'model_ema' in model_weights_dict:
            #     # torch.save(model_weights_dict['model_ema'], model_path)
            #     model_weights_dict = model_weights_dict['model_ema']
            # elif 'model' in model_weights_dict:
            #     # torch.save(model_weights_dict['model'], model_path)
            #     model_weights_dict = model_weights_dict['model']
            # else:
            #     pass
            model.load_state_dict(model_weights_dict)
            model.eval()
            model.cuda()
            if use_trt and not os.path.exists(trt_model_path):
                model =  torch2trt(model, [torch.zeros(1,3,224,224).cuda()], max_batch_size=64,fp16_mode=True)
                torch.save(model.state_dict(), trt_model_path)
            return model

    def load_qc_net(self, model_name='', weights_name='.pth'):

        model_path = os.path.join(model_dir, model_name, weights_name)
        trt_model_path = os.path.splitext(model_path)[0]+'3.cc'
        if os.path.exists(trt_model_path):
            print('load:{}'.format(trt_model_path))
            model = TRTModule()
            model.load_state_dict(torch.load(trt_model_path))
            return model
        else:
            model = mobilenet_v3_small(num_classes=2)
            model_weights_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            if 'model' in model_weights_dict:
                model_weights_dict = model_weights_dict['model']
            elif 'model_ema' in model_weights_dict:
                model_weights_dict = model_weights_dict['model_ema']

            model.load_state_dict(model_weights_dict)
            model.eval()
            model.cuda()
            return model

    def load_qc_net1(self, model_name='', weights_name='.pth'):
        model_path = os.path.join(model_dir, model_name, weights_name)
        model = mobilenet_v3_small(num_classes=2)
        print('load:{}'.format(model_path))
        model_weights_dict = torch.load(model_path, map_location=lambda storage, loc:storage)
        if 'model_ema' in model_weights_dict:
            model_weights_dict = model_weights_dict['model_ema']
        elif 'model' in model_weights_dict:
            model_weights_dict = model_weights_dict['model']
        model.load_state_dict(model_weights_dict)
        model.eval()
        model.cuda()
        return model

    def load_mobile_net(self, model_name='', weights_name='.pth'):
        model_path = os.path.join(model_dir, model_name, weights_name)
        model = mobilenet_v3_small(num_classes=6)
        print('load:{}'.format(model_path))
        model_weights_dict = torch.load(model_path, map_location=lambda storage, loc:storage)
        if 'model_ema' in model_weights_dict:
            model_weights_dict = model_weights_dict['model_ema']
        elif 'model' in model_weights_dict:
            model_weights_dict = model_weights_dict['model']
        model.load_state_dict(model_weights_dict)
        model.eval()
        model.cuda()
        return model

    def cal_tct(self, slide, is_cs=False, save_result=True):
        with torch.no_grad():

            save_result_path = os.path.join(os.path.dirname(slide.filename), os.path.basename(slide.filename)+'_'+self.__class__.__name__+'.npy')
            if save_result and os.path.exists(save_result_path):
                result = np.load(save_result_path, allow_pickle=True).item()
                bboxes = result['bboxes']
                cell_soft = result['cell_soft']
                microbe_soft = result['microbe_soft']
                quality = result['quality']
                sort_ind = result['sort_ind']
                slide_pred = result['slide_pred']
                slide_pos_prob = result['slide_pos_prob']
                cell_num = result['cell_num']

            else:
                x_coords, y_coords = disc_detect(slide, self.config.is_disc_detect)

                center_coords = self.cell_det_func(slide, self.cell_det_net, patch_size=512, num_workers=4,
                                                            batch_size=4, x_coords=x_coords, y_coords=y_coords)

                bboxes, cell_soft, microbe_soft, feat, qc_soft = self.cell_cls_func(slide, center_coords,
                                                                          cell_classifier=self.cell_net,
                                                                          microbe_classifier=self.microbe_net,
                                                                          qc_model=self.qc_net1,
                                                                          microbe_qc=self.microbe_qc_net,
                                                                          patch_size=224, overlap=56,
                                                                          batch_size=64)

                cell_num = int(center_coords.shape[0])

                if center_coords.shape[0]<self.config.min_cellnum or bboxes.shape[0] < self.config.min_roi or \
                        qc_soft.argmax(1).sum()/qc_soft.shape[0] < self.config.qc_threshold:
                    quality = 0
                else:
                    quality = 1

                cell_soft = np.hstack([cell_soft, cell_soft[:,-1:]])
                cell_soft[:,4]=0

                sort_ind = np.argsort(cell_soft[:, 0])
                feat = feat[sort_ind]
                testing_feat, ratio_feat = get_testing_feat(0.0, cell_soft, feat, testing_num=128)
                testing_feat = torch.from_numpy(testing_feat).cuda().float()
                ratio_feat = torch.from_numpy(ratio_feat).cuda().float()
                _, slide_pos_prob, att_w = self.wsi_net(testing_feat, ratio_feat)
                slide_pos_prob = slide_pos_prob.cpu().data.numpy()
                slide_pred = np.argmax(slide_pos_prob)

                if save_result:
                    result = {
                        'bboxes':bboxes,
                        'cell_soft':cell_soft,
                        'microbe_soft':microbe_soft,
                        'quality':quality,
                        'sort_ind':sort_ind,
                        'slide_pred':slide_pred,
                        'slide_pos_prob':slide_pos_prob,
                        'cell_num':cell_num
                    }
                    np.save(save_result_path, result)


            # improve cell sensitivity
            if self.config.cell_calibrate:
                cell_weights = slide_pos_prob.copy()
                cell_weights[0, 0] = 0
                cell_weights[0, 3] += cell_weights[0, 4]
                cell_weights = np.sqrt(np.clip(cell_weights, 0.3 ** 2, 1))
                cell_soft = cell_soft * cell_weights
                cell_pred = np.argmax(cell_soft, axis=1)
                cell_pred[np.logical_and(cell_pred == 3, cell_soft[:, 3] / cell_soft[:, 0] > 1.5)] = 4
                microbe_pred = np.argmax(microbe_soft, axis=1) if self.config.is_microbe_detect else np.zeros_like(cell_pred)
            else:
                cell_pred = np.argmax(cell_soft, axis=1)
                cell_pred[cell_soft[:,3]>=0.95] = 4
                microbe_pred = np.argmax(microbe_soft, axis=1) if self.config.is_microbe_detect else np.zeros_like(cell_pred)


            slide_diagnosis = '阳性' if 1 - slide_pos_prob[0, 0] > float(self.pos_threshold) else '阴性'
            tbs_label = self.config.multi_wsi_cls_dict_reverse[slide_pred]
            if tbs_label == 'NILM' and slide_diagnosis == '阳性':
                slide_pred = np.argmax(slide_pos_prob[:,1:])+1
                tbs_label = self.config.multi_wsi_cls_dict_reverse[slide_pred]

            if np.sum(cell_pred>0) <= self.config.min_pos_cell_threshold: slide_diagnosis = '阴性'
            if slide_diagnosis == '阴性': tbs_label = ''
            if quality == 0: tbs_label += '-样本不满意'

            if slide_diagnosis == '阳性' and np.sum(cell_pred>0)<self.config.min_return_pos_cell:
                cell_pred[sort_ind[:self.config.min_return_pos_cell]] = np.argmax(cell_soft[sort_ind[:self.config.min_return_pos_cell], 1:], axis=1)+1

            if slide_diagnosis == '阴性' and np.sum(cell_pred>0)<self.config.min_return_neg_cell:
                cell_pred[sort_ind[:self.config.min_return_neg_cell]] = np.argmax(cell_soft[sort_ind[:self.config.min_return_neg_cell], 1:], axis=1)+1

            pick_idx = np.where(np.logical_or(cell_pred > 0, microbe_pred > 0))[0]
            pick_bboxes = bboxes[pick_idx]
            cell_pred = cell_pred[pick_idx]
            microbe_pred = microbe_pred[pick_idx]

            cell_prob = np.max(cell_soft[pick_idx][:,1:], axis=1)
            microbe_prob = np.max(microbe_soft[pick_idx][:,1:], axis=1)

            # TBS result
            self.result['diagnosis'] = slide_diagnosis
            self.result['tbs_label'] = tbs_label
            self.result['cell_num'] = cell_num
            self.result['quality'] = quality
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

class LCT40k_convnext_nofz(AlgBase):
    def __init__(self, threshold=None):
        super(LCT40k_convnext_nofz, self).__init__()
        global use_trt
        use_trt = False
        self.cell_det_net = self.load_celldet_model()
        self.wsi_net = self.load_wsi_model_768_global(model_name='lct_convnext_nofz', weights_name='wsi_weights_epoch_4.pth')
        self.microbe_net = self.load_mobile_net(model_name='microbe_mobilenet_0324', weights_name='checkpoint-best-ema.pth') if self.config.is_microbe_detect else None
        self.cell_net = self.load_cell0921_model(model_name='lct_convnext_nofz', weights_name='cell_weights_epoch_4.pth')
        self.qc_net1 = self.load_qc_net(model_name='qc_mobilenetv3_0220', weights_name='checkpoint-best.pth') if self.config.is_qc else None
        self.microbe_qc_net = self.load_qc_net1(model_name='microbe_qc_20230324', weights_name='checkpoint-best-ema.pth') if self.config.is_microbe_qc else None
        self.cell_cls_func = detect_mix20x_scale1_qc3
        self.cell_det_func = count_cells_slide_thread_noqc
        self.pos_threshold = 0.33 if threshold is None else threshold

class LCT40k_convnext_HDX(AlgBase):
    def __init__(self, threshold=None):
        super(LCT40k_convnext_HDX, self).__init__()
        global use_trt
        use_trt = False
        self.cell_det_net = self.load_celldet_model()
        self.wsi_net = self.load_wsi_model_768_global(model_name='海德星', weights_name='wsi_weights_epoch_4.pth')
        self.microbe_net = self.load_mobile_net(model_name='microbe_mobilenet_0324', weights_name='checkpoint-best-ema.pth') if self.config.is_microbe_detect else None
        self.cell_net = self.load_cell0921_model(model_name='海德星', weights_name='cell_weights_epoch_4.pth')
        self.qc_net1 = self.load_qc_net(model_name='qc_mobilenetv3_0220', weights_name='checkpoint-best.pth') if self.config.is_qc else None
        self.microbe_qc_net = self.load_qc_net1(model_name='microbe_qc_20230324', weights_name='checkpoint-best-ema.pth') if self.config.is_microbe_qc else None
        self.cell_cls_func = detect_mix20x_scale1_qc3
        self.cell_det_func = count_cells_slide_thread_noqc
        self.pos_threshold = 0.33 if threshold is None else threshold

class LCT_mobile_micro0324(AlgBase):
    def __init__(self, threshold=None):
        super(LCT_mobile_micro0324, self).__init__()
        self.cell_det_net = self.load_celldet_model()
        self.wsi_net = self.load_wsi_model_768_global(model_name='mix80k0303', weights_name='wsi_weights_epoch_11.pth')
        self.microbe_net = self.load_mobile_net(model_name='microbe_mobilenet_0324', weights_name='checkpoint-best-ema.pth') if self.config.is_microbe_detect else None
        self.cell_net = self.load_cell0921_model(model_name='mix80k0303', weights_name='checkpoint-10.pth')
        self.qc_net1 = self.load_qc_net(model_name='qc_mobilenetv3_0220', weights_name='checkpoint-best.pth') if self.config.is_qc else None
        self.microbe_qc_net = self.load_qc_net1(model_name='microbe_qc_20230324', weights_name='checkpoint-best-ema.pth') if self.config.is_microbe_qc else None
        self.cell_cls_func = detect_mix20x_scale1_qc3
        self.cell_det_func = count_cells_slide_thread_noqc
        self.pos_threshold = 0.33 if threshold is None else threshold

class LCT_mix80k0417_8(AlgBase):
    def __init__(self, threshold=None):
        super(LCT_mix80k0417_8, self).__init__()
        self.cell_det_net = self.load_celldet_model()
        self.wsi_net = self.load_wsi_model_768_global(model_name='mix80k0417', weights_name='wsi_weights_epoch_8.pth')
        self.microbe_net = self.load_mobile_net(model_name='microbe_mobilenet_0324', weights_name='checkpoint-best-ema.pth') if self.config.is_microbe_detect else None
        self.cell_net = self.load_cell0921_model(model_name='mix80k0417', weights_name='cell_weights_epoch_8.pth')
        self.qc_net1 = self.load_qc_net(model_name='qc_mobilenetv3_0220', weights_name='checkpoint-best.pth') if self.config.is_qc else None
        self.microbe_qc_net = self.load_qc_net1(model_name='microbe_qc_20230324', weights_name='checkpoint-best-ema.pth') if self.config.is_microbe_qc else None
        self.cell_cls_func = detect_mix20x_scale1_qc3
        self.cell_det_func = count_cells_slide_thread_noqc
        self.pos_threshold = 0.33 if threshold is None else threshold


