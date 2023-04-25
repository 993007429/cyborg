#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 20:20
# @Author  : Can Cui
# @File    : cell_counting.py
# @Software: PyCharm
# @Comment:


from .utils import *
from threading import Thread
from queue import Queue
import os, sys
import numba
from numba import jit, njit
import math
import numba
if numba.__version__.startswith('0.5'):
    from numba.typed import List
    ls = List
else:
    ls = list

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def read_region_worker(slide, in_queue, out_queue, window_size=320):
    while not in_queue.empty():
        #
        slideRegion = in_queue.get()
        patch_x, patch_y = slideRegion.location

        try:
            image = slide.read(slideRegion.location, slideRegion.size, slideRegion.scale)
            image = image * (2. / 255) - 1.

            for i in range(slideRegion.sub_coords.shape[0]):
                x, y = slideRegion.sub_coords[i][0], slideRegion.sub_coords[i][1]
                xs, ys = math.ceil(x/slideRegion.scale), math.ceil(y/slideRegion.scale)
                crop_img = image[ys:ys+window_size, xs:xs+window_size]
                h, w = crop_img.shape[0:2]
                pad_h, pad_w = window_size - h, window_size - w
                if pad_h > 0 or pad_w > 0:
                    crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
                if crop_img.shape != (window_size, window_size):
                    crop_img = cv2.resize(crop_img, dsize=(window_size, window_size))

                out_queue.put(SlideRegion(
                    image=crop_img,sub_coords=slideRegion.sub_coords[i]+np.array([patch_x,patch_y,patch_x,patch_y])
                ))
        except Exception as e: # read big patch failed
            print(e)
            sub_coords = slideRegion.sub_coords
            sub_coords+= np.array([patch_x, patch_y, patch_x, patch_y])
            num_coords = sub_coords.shape[0]
            for i in range(num_coords):
                try:
                    x1, y1, x2, y2 = sub_coords[i]
                    crop_img = slide.read((x1,x2), (x2-x1, y2-y1), slideRegion.scale)
                    h, w = crop_img.shape[:2]
                    crop_img = crop_img * (2. / 255) - 1.
                    pad_h, pad_w = window_size - h, window_size - w
                    if pad_h > 0 or pad_w > 0:
                        crop_img = np.pad(crop_img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')

                    if crop_img.shape != (window_size,window_size):
                        crop_img = cv2.resize(crop_img, (window_size, window_size))

                    out_queue.put(SlideRegion(
                        image=crop_img,sub_coords=sub_coords[i]
                    ))
                except:
                    out_queue.put(SlideRegion(
                        image=None, sub_coords=sub_coords[i]
                    ))
    return None

@jit(nopython=True)
def pointinpolygon(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in numba.prange(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


@njit(parallel=True)
def parallelpointinpolygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean)
    for i in numba.prange(0, len(D)):
        D[i] = pointinpolygon(points[i,0], points[i,1], polygon)
    return D


def count_cells_slide_thread(slide, net, qc_net=None, patch_size=512,
                             batch_size=8, num_workers=4, x_coords=[], y_coords=[],
                             min_cell_num=5000, qc_threshold=0.15):
    standard_mpp = 0.242042*8
    this_mpp = slide.mpp
    scale_ratio = standard_mpp/this_mpp
    if len(x_coords)>0:
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))
    else:
        xmin, xmax = 0, slide.width
        ymin, ymax = 0, slide.height

    crop_queue = Queue()
    process_queue = Queue(batch_size*8)

    lvl0_patch_size =  int(patch_size * scale_ratio)
    bboxes_list = []
    for y in range(ymin, ymax, lvl0_patch_size):
        for x in range(xmin, xmax, lvl0_patch_size):
            lvl0_crop_w, lvl0_crop_h = int(min(xmax-x, lvl0_patch_size)), int(min(ymax-y, lvl0_patch_size))
            bboxes_list.append([x,y,x+lvl0_crop_w,y+lvl0_crop_h])
    bboxes = np.array(bboxes_list)
    num_bboxes = bboxes.shape[0]
    patch_bboxes_list = match_bboxes_with_pathces(bboxes=bboxes, crop_size=math.ceil(5120*8))
    # print(len(patch_bboxes_list))

    for i in range(len(patch_bboxes_list)):
        patch_coord, bbox_coord = patch_bboxes_list[i]
        xs, ys =  patch_coord[0], patch_coord[1]
        w , h  = patch_coord[2]-patch_coord[0], patch_coord[3]-patch_coord[1]
        crop_queue.put(SlideRegion(location=(xs, ys), size=(w, h), scale=scale_ratio, sub_coords=bbox_coord))

    for i in range(num_workers):
        t = Thread(target=read_region_worker ,args=(slide, crop_queue, process_queue, patch_size))
        t.start()

    center_coords_list = []
    coords_shift_list = []
    batch_data_list = []
    qc_score_list = []
    with torch.no_grad():
        for j in range(num_bboxes):
            # print("{}/{}".format(j+1, crop_count))
            slide_region = process_queue.get()
            if slide_region.image is not None:
                batch_data_list.append(slide_region.image)
                coords_shift_list.append(slide_region.sub_coords)

            if len(batch_data_list) == batch_size or (j==num_bboxes-1 and len(batch_data_list)>0):
                data_variable = torch.as_tensor(np.array(batch_data_list), dtype=torch.float, device=torch.device('cuda'))
                gray_data_variable = (0.2989*data_variable[:,:,:,0] + 0.5870*data_variable[:,:,:,1] + 0.1140*data_variable[:,:,:,2]).unsqueeze(1)
                # import pdb; pdb.set_trace()
                masks = net(gray_data_variable)
                masks = masks.cpu().numpy()

                for idx in range(len(batch_data_list)):
                    mask = masks[idx, 0, :, :]
                    center_coords_this_patch = post_process_mask(mask, threshold=0.05)
                    if center_coords_this_patch.size > 0:
                        center_coords_this_patch = center_coords_this_patch*scale_ratio + np.array(coords_shift_list[idx][:2])
                        if qc_net is not None:
                            this_var = data_variable[idx].unsqueeze(0)
                            qc_batch_data = torch.cat([this_var[:,:256,:256,:], this_var[:,256:,:256, :],
                                                       this_var[:,:256,256:,:], this_var[:,256:,256:,:]], 0).permute(0,3,1,2)
                            _, _, cls_pred = qc_net(qc_batch_data)
                            qc_score_list.append(cls_pred)

                    center_coords_list.append(center_coords_this_patch.astype(int))
                batch_data_list=[]
                coords_shift_list=[]

        center_coords = np.concatenate(center_coords_list)


        # REMOVE CELLS OUTSIDE THE DISC AREA
        if center_coords.shape[0]>0 and len(x_coords)>0:
            try:
                pick_idx = parallelpointinpolygon(center_coords, ls(zip(x_coords, y_coords)))
                if pick_idx.sum()>1e4:
                    center_coords = center_coords[pick_idx]
            except:
                pass

        quality=1
        if qc_net is not None and len(qc_score_list)>0:
            qc_score_np = np.argmin(torch.cat(qc_score_list).cpu().numpy(), axis=1)
            qc_score = float(np.sum(qc_score_np) / qc_score_np.size)
            if center_coords.shape[0] < min_cell_num or qc_score < qc_threshold:
                quality = 0


    return center_coords, quality



















