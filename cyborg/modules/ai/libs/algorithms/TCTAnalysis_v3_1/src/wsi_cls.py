#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/14 13:32
# @Author  : Can Cui
# @File    : cell_cls.py
# @Software: PyCharm
import torch
from queue import Queue
from threading import Thread
import os
import traceback
from cyborg.infra.oss import oss
from .utils import SlideRegion, read_patch_worker
import numpy as np

class WSIClassifier:
    def __init__(self, backbone_path, wsi_model_path, device=None, use_trt=True, rerank=False, topk=128, img_size=224,
                 mpp=0.242042*2, batch_size=32, workers=8, crop_scale=1, merge_hsil=True, cell_calibrate=True, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.backbone = self.load_backbone(backbone_path, use_trt)
        self.wsi_classifier = self.load_wsi_model(wsi_model_path)
        self.rerank=rerank
        self.topk=topk
        self.img_size=img_size
        self.mpp=mpp
        self.batch_size=batch_size
        self.workers=workers
        self.crop_scale=crop_scale

    def load_backbone(self, model_path, use_trt):
        trt_env = False
        try:
            from torch2trt import torch2trt
            from torch2trt import TRTModule
            os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        except:
            print(traceback.format_exc())
            print("[WARNING!]  Loading TensorRT failed, use Pytorch to do the inference.")
            trt_env = False

        trt_model_path = os.path.splitext(model_path)[0] + '.cc'
        if oss.object_exists(trt_model_path) and use_trt and trt_env:
            model = TRTModule()
            trt_model_file = oss.get_object_to_io(trt_model_path)
            model.load_state_dict(torch.load(trt_model_file, map_location=self.device))
        else:
            model_file = oss.get_object_to_io(model_path)
            model = torch.jit.load(model_file, map_location=self.device)
            model.eval()
        return model

    def load_wsi_model(self, model_path):
        model_file = oss.get_object_to_io(model_path)
        model = torch.jit.load(model_file, map_location=self.device)
        model.eval()
        return model


    @torch.no_grad()
    def infer_wsi(self, slide, det_result):
        if slide.mpp is not None:
            this_mpp = slide.mpp
        else:
            this_mpp = 0.242042*2
        scale_ratio = self.mpp / this_mpp

        if isinstance(self.crop_scale, str):
            self.crop_scale = scale_ratio

        if len(det_result)<=self.topk or self.rerank:
            selected_dets = det_result
        else:
            topk_idx = det_result[:,4].topk(k=self.topk, dim=0)[1]
            selected_dets = det_result[topk_idx]

        crop_queue = Queue()
        process_queue = Queue(self.batch_size * 8)
        lvl0_patch_size = int(self.img_size * scale_ratio)
        num_bboxes = selected_dets.size()[0]
        for xyxy in selected_dets[:,:4]:
            x1,y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cent_x, cent_y = (x1+x2)/2 , (y1+y2)/2
            xs,ys = int(cent_x-lvl0_patch_size/2), int(cent_y-lvl0_patch_size/2)
            crop_queue.put(SlideRegion(location=(xs,ys), size=(lvl0_patch_size,lvl0_patch_size), scale=scale_ratio, sub_coords=torch.tensor([x1,y1,x2,y2]).to(self.device)))

        for i in range(self.workers):
            t = Thread(target=read_patch_worker, args=(slide, crop_queue, process_queue, self.img_size, self.crop_scale))
            t.start()

        batch_data_list = []
        bbox_list, embedding_list, cell_soft_list =[], [], []

        for i in range(num_bboxes):
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)
                bbox_list.append(slide_region.sub_coords)

            if len(batch_data_list) == self.batch_size or (i == num_bboxes - 1 and len(batch_data_list) > 0):
                data_variable = torch.tensor(np.array(batch_data_list), dtype=torch.float, device=torch.device('cuda'))
                data_variable = data_variable * (2. / 255) - 1.
                data_variable = data_variable.permute(0,3,1,2)

                result = self.backbone(data_variable)
                if len(result)==2:
                    output, embedding = result
                    cell_soft = torch.softmax(output, dim=1)
                else:
                    output, embedding, cell_soft = result
                cell_soft_list.append(cell_soft)
                embedding_list.append(embedding)
                batch_data_list= []


        wsi_embedding = torch.cat(embedding_list, 0)
        bboxes = torch.cat(bbox_list, 0).view(-1,4)
        cell_soft = torch.cat(cell_soft_list, 0)
        sort_idx = cell_soft[:,0].argsort()
        wsi_embedding = wsi_embedding[sort_idx][:self.topk].unsqueeze(0)
        # import pdb; pdb.set_trace()
        output, wsi_soft, att_score=  self.wsi_classifier(wsi_embedding)

        # TODO re-classify cells
        # if self.rerank:
        #     # improve cell sensitivity
        #     if self.cell_calibrate:
        #         cell_weights = slide_pos_prob.copy()
        #         cell_weights[0, 0] = 0
        #         cell_weights[0, 3] += cell_weights[0, 4]
        #         cell_weights = np.sqrt(np.clip(cell_weights, 0.1 ** 2, 1))
        #         cell_soft = cell_soft * cell_weights
        #         cell_pred = np.argmax(cell_soft, axis=1)
        #         cell_pred[np.logical_and(cell_pred == 3, cell_soft[:, 3] / cell_soft[:, 0] > 1.5)] = 4
        #
        #     else:
        #         cell_pred = np.argmax(cell_soft, axis=1)
        #         cell_pred[cell_soft[:, 3] >= 0.95] = 4
        return bboxes, cell_soft, wsi_soft[0]

class MicrobeClassifier:
    def __init__(self, model_path, device=None, use_trt=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = self.load_model(model_path, use_trt=use_trt)

    def load_model(self, model_path, use_trt):
        pass

    @torch.no_grad()
    def microbe_cls_wsi(self):
        pass

if __name__ == '__main__':
    from Slide.dispatch import openSlide
    det_res = torch.load('det_res.pth')
    import time
    tic = time.time()
    classifier = WSIClassifier(backbone_path='/opt/projects/znbl/znbl3/alg/Algorithms/TCTAnalysis_v3_1/Model/cell_cls_model/cell_weights_epoch_7.torchscript',
                               wsi_model_path='/opt/projects/znbl/znbl3/alg/Algorithms/TCTAnalysis_v3_1/Model/cell_cls_model/wsi_weights_epoch_7.torchscript')
    # slide= openSlide('/mnt/data_xsky/private_cloud_data/河南省人民2022P/data/2023_05_08_22_51_49_576221/slices/20230508225149755636/2314368.sdpc')
    slide= openSlide('/mnt/data_xsky/private_cloud_data/河南省人民2022P/data/2023_05_08_21_49_14_214707/slices/20230508214914835199/2314013.sdpc')

    res = classifier.infer_wsi(slide, det_res)
    print(res[2])
    print(time.time()-tic)

