import torch.utils.data as data
# from parse_embolus import read_region_kfb
import numpy as np
import cv2
import os
import sys

import os
import sys
import json
#import albumentations as A

from tqdm import tqdm
from skimage import io
#from wsi_infer.transforms import *
import math

from torch.utils.data import Dataset



# import kfb.kfbslide as kfbslide

def split_image(image, patch_size, overlap):
    h, w = image.shape[0:2]
    stride = patch_size - overlap
    patch_list = []
    patch_coords = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            crop_img = image[y:y + patch_size, x:x + patch_size, :]

            crop_h, crop_w = crop_img.shape[0:2]
            new_crop_img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            new_crop_img[:crop_h, :crop_w] = crop_img

            patch_list.append(new_crop_img)
            patch_coords.append(np.array([x, y, x + crop_w, y + crop_h]))
    patch_image = np.array(patch_list)
    patch_coords = np.array(patch_coords)

    return patch_image, patch_coords


class Dataset(data.Dataset):
    def __init__(self, slide, crop_coord, wsi_type='mrxs', overlap=128,crop_len=1024,lvl=8,mask=None,mask_lvl = 16,has_Mask = False,):
        self.slide = slide #original
        self.crop_coord = crop_coord
        self.nF = len(self.crop_coord)
        self.overlap = overlap
        self.wsi_type = wsi_type
        self.crop_len = crop_len
        self.lvl = lvl
        self.has_Mask = has_Mask
        self.slide_h, self.slide_w = slide.height,slide.width
        if self.has_Mask:
            self.mask = mask
            self.mask_height,self.mask_width,_= self.mask.shape
            if (self.mask_height<2000 and self.mask_width<2000):
                self.mask = np.ones((self.mask_width,self.mask_height,3))
            self.mask_lvl = mask_lvl

    def __len__(self):
        return self.nF

    def __getitem__(self, index):

        xmin, ymin, xmax, ymax = self.crop_coord[index]
        xmin = min(self.slide_w - 1, xmin)
        ymin = min(self.slide_h - 1, ymin)
        xmax = min(self.slide_w - 1, xmax)
        ymax = min(self.slide_h - 1, ymax)

        if self.has_Mask:
            mask_xmin = min(self.mask_width-1, xmin//self.mask_lvl)
            mask_ymin = min(self.mask_height-1, ymin//self.mask_lvl)
            mask_xmax = min(self.mask_width-1, xmax//self.mask_lvl)
            mask_ymax = min(self.mask_height-1, ymax//self.mask_lvl)



        h, w = ymax - ymin, xmax - xmin


        new_cur_region = np.zeros((self.crop_len, self.crop_len, 3), dtype=np.uint8)
        new_mask_region = np.ones((self.crop_len, self.crop_len, 3), dtype=np.uint8)

        # self.slide.read([xmin+int(self.roi_xy[0]), ymin+ int(self.roi_xy[1])], (w, h), scale = (1 / self.mpp_ratio)))  # RGBA

        try:
            h = min(h, self.crop_len * self.lvl)
            w = min(w, self.crop_len * self.lvl)
            # new_patch[0:(ed_y-st_y),0:(ed_x-st_x),:] = slide[st_y:ed_y,st_x:ed_x,:]
            sc = self.lvl
            if self.wsi_type in ['ndpi','mrxs',"kfb"]:
                if(self.lvl==1):
                    sc = 1 / self.lvl


            tmp = self.slide.read([xmin, ymin], (w, h),scale=sc)

            #cur_region = cur_region[:, :, :3].astype(np.uint8)
            tmp = tmp[:,:,:3].astype(np.uint8)
            hh,ww,cc = tmp.shape
            # print(f'HH:{hh},WW:{ww}')
            # print(f'new_cur_region.shape: {new_cur_region.shape}')
            h,w,c = new_cur_region.shape
            if hh > new_cur_region.shape[0] or ww > new_cur_region.shape[1]:
                tmp = tmp[0:h,0:w,:]
                hh,ww,cc = tmp.shape
            new_cur_region[0:hh, 0:ww,:] = tmp
            new_cur_region_bgr = new_cur_region[..., ::-1].copy()
            # print(f'new_cur_region.shape2: {new_cur_region.shape}')
            #cv2.imwrite('3.png',new_cur_region_bgr)
            if self.has_Mask:
                tmp_mask = self.mask[mask_ymin:mask_ymax, mask_xmin:mask_xmax,:]        
                new_mask_region[0:hh,0:ww,:] = cv2.resize(tmp_mask, (ww,hh),interpolation=cv2.INTER_LINEAR)
                #cv2.imwrite('3.png',new_mask_region)
            new_mask_region = new_mask_region.transpose(2, 0, 1)

            new_cur_region = new_cur_region.transpose(2, 0, 1)
            new_cur_region_bgr = new_cur_region_bgr.transpose(2,0,1)


            #new_cur_region = new_cur_region.transpose(2, 0, 1)
        except Exception as e:
            print(str(e))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(exc_obj)
            print(e)
            new_cur_region_bgr = new_cur_region[..., ::-1].copy()
            new_cur_region = new_cur_region.transpose(2, 0, 1)
            new_mask_region = new_mask_region.transpose(2,0,1)
            new_cur_region_bgr = new_cur_region_bgr.transpose(2, 0, 1)


        ret = {'cur_region': new_cur_region,
               'cur_region_bgr':new_cur_region_bgr,
               'xmin': xmin,
               'ymin': ymin,
               'xmax': xmin+w,
               'ymax': ymin+h,
               'mask': new_mask_region,
               'st_xy':[int(xmin/self.lvl),int(ymin/self.lvl)]
               }
        return ret


def change_lvl(st_x=0,st_y=0,ed_x=0,ed_y=0,clvl = 8):
    st_x = int(st_x*clvl)
    st_y = int(st_y*clvl)
    ed_x = int(ed_x*clvl)
    ed_y = int(ed_y*clvl)

    return[st_x,st_y,ed_x,ed_y]





# def build_dataset(args, image_set):
#     mean, std = np.load(f'./datasets/{args.dataset}/mean_std.npy')
#     additional_targets = {}
#     for i in range(1, args.num_classes):
#         additional_targets.update({'keypoints%d' % i: 'keypoints'})
#     if image_set == 'train':
#         augmentor = A.Compose([
#             # A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=0, shift_limit=0, border_mode=0, value=0, p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomBrightnessContrast(p=0.5),
#             #A.RandomCrop(height=1080, width=1080, always_apply=True),
#             A.RandomCrop(height=1024, width=1024, always_apply=True),
#         ], p=1, keypoint_params=A.KeypointParams(format='xy'), additional_targets=additional_targets)
#         transform = Preprocessing(mean, std, augmentor)
#     elif image_set == 'test':
#         transform = Preprocessing(mean, std)
#     else:
#         raise NotImplementedError
#     data_folder = DataFolder(args.dataset, args.num_classes, image_set, transform)
#     return data_folder
