#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2023/7/8 20:20
# @Author  : Can Cui
# @File    : cell_counting.py
# @Software: PyCharm
# @Comment:

from threading import Thread
from queue import Queue
import os, sys
import math
import numpy as np
import torch
import traceback
from skimage.feature import peak_local_max
import cv2
from cyborg.infra.oss import oss

from .utils import match_bboxes_with_pathces, SlideRegion, read_region_worker

def get_coordinate( voting_map, min_len=5):
    #abs_val = max( threshhold*np.max(voting_map[:]),  abs_threshold)
    # abs_val = max( threshhold*0.95,  abs_threshold)
    # voting_map[ voting_map < abs_val ] = 0
    try:
        import cupy
        from cupyimg.skimage.feature import peak_local_max as peak_local_max_gpu
        cupy_array =  cupy.asarray(voting_map)
        coordinates = peak_local_max_gpu(cupy_array, min_distance= min_len,exclude_border=False) # N by 2
        coordinates = coordinates.get()
    except:
        coordinates = peak_local_max(voting_map, min_distance=min_len, exclude_border=False)

    if coordinates.size == 0:
        coordinates = None  # np.asarray([])
    else:
        coordinates = coordinates[:, ::-1]
    return coordinates

def post_process_mask(masks, threshold=0.1):
    masks[masks < max(threshold * np.max(masks), 0.1)] = 0
    bboxes = get_coordinate(masks, min_len=15)
    if bboxes is None:
        bboxes = np.empty((0,2), dtype=int)

    return bboxes


class CellCounter:
    def __init__(self, model_path, device=None, use_trt=True, img_size=1024,
                 crop_size=5120 * 8, batch_size=8, mpp=0.242042 * 8,
                 threshold=0.1, workers=8, crop_scale='adapt', **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = self.load_model(model_path, use_trt=use_trt)
        self.img_size = img_size
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.mpp = mpp
        self.threshold = threshold
        self.workers = workers
        self.crop_scale = crop_scale


    def load_model(self, model_path, use_trt):
        trt_env = True
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

    @torch.no_grad()
    def count_wsi(self, slide, x_coords=[], y_coords=[]):

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
        process_queue = Queue(self.batch_size * 8)

        lvl0_patch_size = int(self.img_size * scale_ratio)
        bboxes_list = []
        for y in range(ymin, ymax, lvl0_patch_size):
            for x in range(xmin, xmax, lvl0_patch_size):
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
            t = Thread(target=read_region_worker, args=(slide, crop_queue, process_queue, self.img_size,  self.crop_scale))
            t.start()

        batch_data_list = []
        cell_counts = 0
        for j in range(num_bboxes):
            # print("{}/{}".format(j+1, num_bboxes))
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)

            if len(batch_data_list) == self.batch_size or (j==num_bboxes-1 and len(batch_data_list)>0):
                data_variable = torch.as_tensor(np.array(batch_data_list), dtype=torch.float, device=self.device)
                data_variable = data_variable * (2. / 255) - 1.
                gray_data_variable = (0.2989*data_variable[:,:,:,0] + 0.5870*data_variable[:,:,:,1] + 0.1140*data_variable[:,:,:,2]).unsqueeze(1)
                masks = self.model(gray_data_variable)
                masks = masks.cpu().numpy()

                for idx in range(len(batch_data_list)):
                    mask = masks[idx, 0, :, :]
                    center_coords_this_patch = post_process_mask(mask, threshold=self.threshold)
                    cell_counts+=center_coords_this_patch.shape[0]

                    # from imageio import imsave
                    # save_dir = '/mnt/data_alg/cell_counting_test'
                    # imsave(os.path.join(save_dir, f'{j-idx}.jpg'), batch_data_list[idx])

                batch_data_list = []
        return cell_counts



if __name__ == '__main__':
    from Slide.dispatch import openSlide
    import time
    args = dict(
        img_size=512, crop_size=5120 * 8, batch_size=8, mpp=0.242042 * 8, threshold=0.1, workers=8, crop_scale='adapt', model_name='', model_size='123'
    )
    tic = time.time()
    cell_counter = CellCounter(model_path='/opt/projects/znbl/znbl3/alg/Algorithms/TCTAnalysis_v3_1/Model/cell_counting/weights_epoch_8100.pth',  **args)
    # slide= openSlide('/data/2314364.sdpc')
    slide= openSlide('/mnt/data_xsky/private_cloud_data/河南省人民2022P/data/2023_05_08_22_51_49_576221/slices/20230508225149755636/2314368.sdpc')
    print(cell_counter.count_wsi(slide))
    print(time.time()-tic)





# http://192.168.119.20:9000/#/share?caseid=2023_05_08_22_51_49_576221&fileid=20230508225149755636&companyid=河南省人民2022P&filename=2314368.sdpc
