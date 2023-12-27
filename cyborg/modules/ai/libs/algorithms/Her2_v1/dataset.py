# from parse_embolus import read_region_kfb
import numpy as np
from PIL import Image
import cv2
import os
import sys
import os
import sys
import json
from tqdm import tqdm
from skimage import io
#from wsi_infer.transforms import *
import math
import torch.utils.data as data
from torchvision import transforms

from PIL import Image,ImageDraw

transform_image = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization values for pre-trained PyTorch models
])

transform_mask = transforms.Compose([
    transforms.ToTensor()
])

class region_dataset(data.Dataset):
    def __init__(self, slide,crop_coord,scale,window_size):
        self.slide = slide #original
        self.crop_coord = crop_coord
        self.window_size = window_size
        self.slide_h, self.slide_w = slide.height,slide.width
        self.scale = scale
    def __len__(self):
        return len(self.crop_coord)
    def __getitem__(self, index):
        xmin, ymin, xmax, ymax = self.crop_coord[index]
        new_cur_region =  np.zeros((self.window_size[1],self.window_size[0],3),dtype=np.uint8)
        tmp = self.slide.read([xmin,ymin],(xmax-xmin,ymax-ymin), scale = self.scale)
        tmp_h,tmp_w,_ = tmp.shape
        tmp_h = np.clip(tmp_h,0,self.window_size[1])
        tmp_w = np.clip(tmp_w,0,self.window_size[0])
        tmp = Image.fromarray(tmp)
        tmp = np.array(tmp.resize((tmp_w,tmp_h)))
        new_cur_region[:tmp_h,:tmp_w,:]=tmp
        new_cur_region = new_cur_region.transpose(2, 0, 1)
        r = {
            'cur_region': new_cur_region,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'lvl':self.scale
            }
        return r

class cell_dataset(data.Dataset):
    def __init__(self, slide,crop_coord,region_map,cell_dict):
        self.slide = slide
        self.crop_coord = crop_coord
        self.region_map = region_map
        self.cell_dict = cell_dict
    def __len__(self):
        return len(self.crop_coord)

    def __getitem__(self, index):
        x,x_,y,y_ = self.crop_coord[index]
        crop_len = int (self.cell_dict['cell_crop_len'] * self.cell_dict['mpp_ratio'])
        new_cur_region = np.zeros((crop_len, crop_len, 3), dtype=np.uint8)
        new_map_region = np.ones((crop_len, crop_len, 3), dtype=np.uint8)
        tmp_region = self.slide.read([x, y], (x_-x, y_-y)).astype(np.uint8)
        hh,ww,_ = tmp_region.shape
        hh = np.clip(hh,0,crop_len)
        ww = np.clip(ww,0,crop_len)
        new_cur_region[0:hh,0:ww,:] = tmp_region
        #tmp_map = cv2.resize(tmp_map,(ww,hh))
        new_cur_region= cv2.resize(new_cur_region,(self.cell_dict['cell_crop_len'],self.cell_dict['cell_crop_len']))
        if self.region_map is not None:
            scale = self.cell_dict['thresh_scale']
            h,w,_ = self.region_map.shape
            xx = np.clip(int(x/scale),0,w)
            xx_ = np.clip(int(x_/scale),0,w)
            yy = np.clip(int(y/scale),0,h)
            yy_ = np.clip(int(y_/scale),0,h)
            tmp_map = self.region_map[yy:yy_,xx:xx_,:] /255
            tmp_map = tmp_map.astype(np.uint8)
            tmp_map = cv2.resize(tmp_map,(ww,hh))
            new_map_region[0:hh,0:ww,:] = tmp_map
        new_map_region= cv2.resize(new_map_region,(self.cell_dict['cell_crop_len'],self.cell_dict['cell_crop_len']))
        #cv2.imwrite(r'C:\znbl3_230220\alg\Algorithms\Her2_v1\tmp_data\new.png',new_cur_region)
        new_cur_region = new_cur_region.transpose(2,0,1)
        new_map_region = new_map_region.transpose(2,0,1)
        ret = {'cur_region': new_cur_region,
               'xmin': x,
               'ymin': y,
               'xmax': x_,
               'ymax': y_,
               'mask': new_map_region,
               }
        return ret

        
        




    


class Dataset(data.Dataset):
    def __init__(self, slide,crop_coord,mask_pth='',crop_len=1024):
        self.slide = slide #original
        self.crop_coord = crop_coord
        self.nF = len(self.crop_coord)
        self.crop_len = crop_len
        self.slide_h, self.slide_w = slide.height,slide.width
        self.mask_pth = mask_pth
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        if self.mask_pth != '':
            self.has_mask = True
            self.mask = np.array(Image.open(self.mask_pth))
            self.mask_h,self.mask_w,_ = self.mask.shape
            self.mask_lvl = int(self.slide_h/self.mask_h)
    def __len__(self):
        return self.nF

    def check_within_range(number, lower_bound, upper_bound):
        if lower_bound <= number <= upper_bound:
            return number
        elif lower_bound > number:
            return lower_bound
        elif upper_bound < number:
            return upper_bound

    def __getitem__(self, index):
        xmin, ymin, xmax, ymax = self.crop_coord[index]
        # new_cur_region
        new_cur_region =  np.zeros((self.crop_len,self.crop_len,3),dtype=np.uint8)
        new_mask_region = np.ones((self.crop_len,self.crop_len,3),dtype=np.uint8)
        tmp = self.slide.read([xmin,ymin],(xmax-xmin,ymax-ymin))
        tmp_h,tmp_w = tmp.shape
        new_cur_region[:tmp_h,:tmp_w,:]=tmp

        if self.has_Mask:
            mask_xmin = self.check_within_range(int(xmin / self.mask_lvl),0,self.mask_w)
            mask_ymin = self.check_within_range(int(ymin / self.mask_lvl),0,self.mask_h)
            mask_xmax = self.check_within_range(int(xmax / self.mask_lvl),0,self.mask_w)
            mask_ymax = self.check_within_range(int(ymax / self.mask_lvl),0,self.mask_h)
            tmp_mask = self.mask[mask_ymin:mask_ymax,mask_xmin:mask_xmax,:]
            tmp_mask = cv2.resize(tmp_mask,(tmp_w,tmp_h),interpolation=cv2.INTER_LINEAR)
            new_mask_region[:tmp_h,:tmp_w,:]=tmp_mask
        
        # new_mask_region = new_mask_region.transpose(2, 0, 1)
        # new_cur_region = new_cur_region.transpose(2, 0, 1)
        new_mask_region = self.transform_mask(new_mask_region)
        new_cur_region = self.transform_image(new_cur_region)

        ret = {
            'cur_region': new_cur_region,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'mask': new_mask_region,
            'st_xy':[int(xmin/self.lvl),int(ymin/self.lvl)]
            }
        return ret



            

      