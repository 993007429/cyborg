#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/11 11:44
# @Author  : Can Cui
# @File    : cell_detection.py
# @Software: PyCharm
import os
import torch
import traceback
from threading import Thread
from queue import Queue
import numpy as np
import torchvision

from cyborg.infra.oss import oss
from .utils import match_bboxes_with_pathces, SlideRegion, read_region_worker
from .nms import non_max_suppression


def rescale(det, scale_ratio, shift):
    det[:, :4] *= scale_ratio
    det[:,:4] += torch.tensor([shift[0], shift[1], shift[0], shift[1]]).to(det.device)
    return det


class CellDetector:
    def __init__(self, model_path, device=None, use_trt=True, img_size=1024, overlap=128, crop_size=5120*8, batch_size=16,
                   mpp=0.242042*4, workers=8, crop_scale='adapt',
                   conf_thres=0.1, iou_thres=0.45, agnostic=True, max_det=300,
                   drop_edge=True, clamp=False, classes=None, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = self.load_model(model_path, use_trt=use_trt)
        self.img_size = img_size
        self.overlap=overlap
        self.crop_size=crop_size
        self.batch_size=batch_size
        self.mpp=mpp
        self.workers=workers
        self.crop_scale=crop_scale
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic=agnostic
        self.max_det=max_det
        self.drop_edge=drop_edge
        self.clamp = clamp
        self.classes = classes

    def load_model(self, model_path, use_trt):
        trt_env = False
        try:
            import tensorrt
            os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        except:
            print(traceback.format_exc())
            print("[WARNING!]  Loading TensorRT failed, use Pytorch to do the inference.")
            trt_env = False

        trt_model_path = os.path.splitext(model_path)[0] + '.cc'
        if os.path.exists(trt_model_path) and use_trt and trt_env:
            model = None
            #TODO add trt model load
            # model = TRTModule()
            # model.load_state_dict(torch.load(trt_model_path, map_location=self.device))
        else:
            model_file = oss.get_object_to_io(model_path)
            model = torch.jit.load(model_file, map_location=self.device)
            # model.eval()
        return model

    def remove_overlap_bbox(self, bboxes):
        box_area = torchvision.ops.box_area(bboxes)
        lt = torch.max(bboxes[:, None, :2], bboxes[:, :2])  # [N,M,2]
        rb = torch.min(bboxes[:, None, 2:], bboxes[:, 2:])  # [N,M,2]
        wh = (rb-lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        temp = (inter>=box_area*0.95).sum(dim=0)
        keep_box_idx = (temp==1).nonzero(as_tuple=True)[0]
        return keep_box_idx

    @torch.no_grad()
    def detect_wsi(self, slide, x_coords=[], y_coords=[]):

        if slide.mpp is not None:
            this_mpp = slide.mpp
        else:
            this_mpp = 0.242042*2
        scale_ratio = self.mpp / this_mpp

        if len(x_coords) > 0:
            xmin, xmax = int(min(x_coords)), int(max(x_coords))
            ymin, ymax = int(min(y_coords)), int(max(y_coords))
        else:
            xmin, xmax = 0, slide.width
            ymin, ymax = 0, slide.height

        if isinstance(self.crop_scale, str):
            if self.crop_scale.isdigit():
                self.crop_scale = float(self.crop_scale)
            else:
                self.crop_scale = scale_ratio

        crop_queue = Queue()
        process_queue = Queue(self.batch_size * 2)

        lvl0_patch_size = int(self.img_size * scale_ratio)
        lvl0_stride = int((self.img_size-self.overlap)*scale_ratio)
        bboxes_list = []
        for y in range(ymin, ymax, lvl0_stride):
            for x in range(xmin, xmax, lvl0_stride):
                lvl0_crop_w, lvl0_crop_h = int(min(xmax - x, lvl0_patch_size)), int(min(ymax - y, lvl0_patch_size))
                bboxes_list.append([x, y, x + lvl0_crop_w, y + lvl0_crop_h])
        bboxes = np.array(bboxes_list)
        num_bboxes = bboxes.shape[0]
        #crop large patch from wsi is faster than crop small one by one
        patch_bboxes_list = match_bboxes_with_pathces(bboxes=bboxes, crop_size=self.crop_size)

        for i in range(len(patch_bboxes_list)):
            patch_coord, bbox_coord = patch_bboxes_list[i]
            xs, ys = patch_coord[0], patch_coord[1]
            w, h = patch_coord[2] - patch_coord[0], patch_coord[3] - patch_coord[1]
            crop_queue.put(SlideRegion(location=(xs, ys), size=(w, h), scale=scale_ratio, sub_coords=bbox_coord))

        for i in range(self.workers):
            t = Thread(target=read_region_worker, args=(slide, crop_queue, process_queue, self.img_size, self.crop_scale))
            t.start()

        batch_data_list = []
        coords_shift_list = []
        result_list = []

        for j in range(num_bboxes):
            # print("{}/{}".format(j+1, num_bboxes))
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)
                coords_shift_list.append(slide_region.sub_coords)

            if len(batch_data_list) == self.batch_size or (j==num_bboxes-1 and len(batch_data_list)>0):
                data_variable = torch.as_tensor(np.array(batch_data_list), dtype=torch.float, device=self.device)
                data_variable /= 255.
                data_variable = data_variable.permute((0,3,1,2)).contiguous()
                preds = self.model(data_variable)
                dets = non_max_suppression(preds, self.conf_thres, self.iou_thres, self.classes, self.agnostic, max_det=self.max_det, img_size=self.img_size, drop_edge=self.drop_edge, clamp=self.clamp)
                for idx, (coord_shift, det) in enumerate(zip(coords_shift_list, dets)):
                    if len(det):
                        if False:
                            from utils import plot_box_and_label, generate_colors
                            label_list = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC', 'CC', 'TRI', 'CAN', 'ACT', 'HSV', 'INF', 'UNSURE']
                            img_ori = batch_data_list[idx]
                            for *xyxy, conf, cls in reversed(det[:,:6]):

                                class_num = int(cls)
                                label = f'{label_list[class_num]} {conf:.2f}'
                                plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy,  label,
                                                   color=generate_colors(class_num, True))
                            save_dir = '/mnt/data_alg/TCT2/cell_det_test'
                            os.makedirs(save_dir, exist_ok=True)
                            from imageio import imsave
                            imsave(os.path.join(save_dir, f'{j-idx}.jpg'), img_ori)
                            print(j-idx, det)

                        det = rescale(det, scale_ratio, coord_shift)
                        result_list.append(det)

                batch_data_list = []
                coords_shift_list = []
        det_res = torch.cat(result_list, 0)
        keep_box_idx = torchvision.ops.nms(det_res[:,:4], det_res[:, 5], self.iou_thres)  # NMS
        # keep_box_idx = self.remove_overlap_bbox(det_res[keep_box_idx][:,:4]) # 大框套小框
        det_res = det_res[keep_box_idx]


        return det_res





if __name__ == '__main__':
    from Slide.dispatch import openSlide
    import time
    tic = time.time()

    cfg = dict(img_size=1024, overlap=256, crop_size=5120*2, batch_size=16,
                   mpp=0.242042*2, workers=8, crop_scale='adapt',
                   conf_thres=0.2, iou_thres=0.45, classes=[0,1,2,3,4], agnostic=True, max_det=300,
                   drop_edge=True, clamp=False)
    # import pdb; pdb.set_trace()

    celldetector = CellDetector(model_path='/opt/projects/znbl/znbl3/alg/Algorithms/TCTAnalysis_v3_1/Model/cell_det/best.torchscript', **cfg)
    # slide= openSlide('/data/2314364.sdpc')
    slide= openSlide('/mnt/data_xsky/private_cloud_data/河南省人民2022P/data/2023_05_08_21_49_14_214707/slices/20230508214914835199/2314013.sdpc')
    result = celldetector.detect_wsi(slide)
    print(time.time()-tic)
    torch.save(result, 'det_res.pth')
    print(result.shape)
    print(torch.unique(result[:,5],return_counts=True))


