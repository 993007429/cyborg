import os,sys
import numpy as np
from dataset import region_dataset,cell_dataset
import math
import torch
from models import experimental as e
from models.general import non_max_suppression
#sys.path.insert(0,r'C:\znbl3_230220\alg\python_lib_38')
from PIL import Image,ImageDraw
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import region_process as region
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#os.environ['RANK'] = '0'
#os.environ['WORLD_SIZE'] = '2'
import psutil
import cv2

def check_parent_pid(pid):
    pid = int(pid)
    if sys.platform == 'linux':
        return os.path.exists(f"/proc/{pid}")
    else:
        return psutil.pid_exists(pid)

def load_model(checkpth,device):
    return e.attempt_load(checkpth,device)

def non_max_suppression_(preds,region_):
    conf_thres = region_['region_threshold'] 
    iou_thres = region_['region_iou_threshold'] 
    return non_max_suppression(preds,conf_thres,iou_thres)

def merge_mask(final_mask,current_mask):
    hh,ww,_ = final_mask.shape
    current_mask = cv2.resize(current_mask,(ww,hh))
    merged_mask = np.minimum(current_mask, final_mask)
    merged_mask = np.clip(merged_mask, 0, 255)
    return merged_mask

def produce_roi_threshold_map(roi_x,roi_y,slide,scale):
    img = slide.read(scale=scale)
    h,w,_ = img.shape
    img = np.zeros_like(img)
    x_min,x_max,y_min,y_max = np.min(roi_x),np.max(roi_x),np.min(roi_y),np.max(roi_y)
    x_min = np.clip(int(x_min / scale),0,w)
    x_max = np.clip(int(x_max / scale),0,w)
    y_min = np.clip(int(y_min / scale),0,h)
    y_max = np.clip(int(y_max / scale),0,h)
    
    
    threshold_map = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
    
    return threshold_map

def check_mask(x,x_,y,y_,map,scale):
    if map is None:
        return True
    else:
        h,w,_ = map.shape
        x = np.clip(int(x/scale),0,w)
        x_ = np.clip(int(x_/scale),0,w)
        y = np.clip(int(y/scale),0,h)
        y_ = np.clip(int(y_/scale),0,h)
        amount = np.sum(map[y:y_,x:x_])
        return amount > 0




class region_processer:
    def __init__(self,slide_path,region_threshold,window_size,stride_ratio,checkpth,scale,iou_thres):
        self.slide_path = slide_path
        self.region_threshold = region_threshold
        self.iou_thres = iou_thres
        self.checkpth = checkpth
        self.scale = scale
        self.window_size = window_size
        self.stride_ratio = stride_ratio
    def start(self):
        region.main_function(slice_path)

class region_dataset_processer:
    def __init__(self, slide,region_threshold,window_size,stride_ratio,checkpth,scale,iou_thres):
        self.slide = slide
        self.region_threshold = region_threshold
        self.iou_thres = iou_thres
        self.checkpth = checkpth
        self.scale = scale
        self.window_size = window_size
        self.stride_ratio = stride_ratio
    def start(self):
        dataset_ = dataset_producer(self.slide,self.window_size,self.stride_ratio,self.scale).start()
        args = self.yolo_args()
        port_id = 10000+np.random.randint(0,1000)
        args.dist_url = 'tcp://127.0.0.1:'+str(port_id)
        args.num_gpus = torch.cuda.device_count()
        args.backend = 'gloo'

        # if self.gpu_num > 1:
        #          sampler = DistributedSampler(dataset_)
        # print(sampler)
        # device_ids = list(range(0, self.gpu_num))
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #  model = e.attempt_load(self.checkpth,device)
        # model.half()
        # self.model = model

        ###
    
       

        if args.num_gpus == 1:
             region_process(gpu=0,args=args)
        else:
            #mp.set_start_method('spawn')
            print('mp')
            print(args.num_gpus)
            torch.multiprocessing.set_start_method('spawn')
            mp.spawn(region_process, nprocs=args.num_gpus, args=(args,))



        # dataset_ = dataset_producer(self.slide,self.window_size,self.stride_ratio,self.scale).start()

        # #print(model)
        # if gpu_num > 1:
        #     sampler = DistributedSampler(dataset_)
        #     print(sampler)
        #     model = DDP(model,device_ids=device_ids)
          
        # patch_loader_ = torch.utils.data.DataLoader(dataset=dataset_,batch_size=1,shuffle=False,num_workers=0,sampler=sampler)
       


        # gpu,backend,model,dataset = args
        # rank = gpu
        # dist.init_process_group(backend=backend,init_method=self.dist_url,
        # world_size=self.gpu_num,rank=rank)
        # torch.cuda.set_device(gpu)
        # model = DDP(model,device_ids=[gpu],find_unused_parameters=True)
        # sampler = DistributedSampler(dataset)
        # patch_loader_ = torch.utils.data.DataLoader(dataset=dataset_,batch_size=1,shuffle=False,num_workers=0,sampler=sampler)
        # print('Start region detection')
        # model.eval()
        # sum_num = 0
        # print(len(patch_loader_))
        # for idx,patch in enumerate(patch_loader_):
        #     inputs = patch['cur_region'][0].half()
        #     inputs /= 255.0
        #     if inputs.ndimension() == 3:
        #         inputs = inputs.unsqueeze(0)
        #     inputs = inputs.to(device)
        #     with torch.no_grad():
        #         preds = model(inputs)[0]
        #     sum_num = sum_num + ((preds[..., 4] > self.region_threshold).cpu().numpy()).sum()
        
        # print('end')
        # print(sum_num)

    
class dataset_producer:
    def __init__(self, slide,window_size=None,stride_ratio=None,scale=None):
        self.slide = slide
        self.window_size = window_size
        if window_size is not None:
            self.scaled_window_size = (window_size[0]*scale, window_size[1]*scale)
            self.stride = int(stride_ratio * self.scaled_window_size[1])
            self.scale = scale

    def start(self):
        self.coords = self.get_coords(self.scaled_window_size,[self.slide.width,self.slide.height],self.stride)
        return region_dataset(self.slide,self.coords,self.scale,self.window_size)   
    
    def set_cell_dataset_config(self,threshold_map,region_map,config_dict):
        self.threshold_map = threshold_map
        self.region_map = region_map
        self.config_dict = config_dict

    def start_cell(self):
        self.cell_coords = self.get_coords_with_map()
        return cell_dataset(self.slide,self.cell_coords,self.region_map,self.config_dict)


    def get_coords_with_map(self):
        mpp_ratio = self.config_dict['mpp_ratio']
        #croplen = int(croplen * lvl * mpp_ratio)
        crop_len = int (self.config_dict['cell_crop_len'] * mpp_ratio)
        w,h = self.slide.width,self.slide.height
        num_y = math.ceil(h / crop_len)
        num_x = math.ceil(w / crop_len)
        crop_coords = []
        for i in range(num_y):
            y = int(i*crop_len)
            y_ = np.clip(y + crop_len,0,h)
            for j in range(num_x):
                x = int(j*crop_len)
                x_ = np.clip(x+crop_len,0,w)
                if check_mask(x,x_,y,y_,self.region_map,self.config_dict['thresh_scale']) and check_mask(x,x_,y,y_,self.threshold_map,self.config_dict['thresh_scale']):
                    crop_coords.append([x,x_,y,y_])
        return  crop_coords



    
    def get_coords(self,window_size=[1000,1000],slide_size=[10000,10000],stride=0):
            w,h = slide_size
            wh,ww = window_size
            coords = []
            num_h = (h-wh+stride) // stride
            num_w = (w-ww+stride) // stride
            print(num_h)
            print(num_w)
            # h_remainder = h - (num_h*stride-stride+wh)
            # w_remainder = w - (num_w*stride-stride+ww)
            for i in range(max(num_h+1,1)):
                for j in range(max(num_w+1,1)):
                    x = j * stride
                    y = i * stride
                    x_ = x+ww
                    y_ = y+wh
                    if x_ > w:
                        x_ = w
                    if y_ > h:
                        y_ = h
                    coords.append([x,y,x_,y_])
            return coords


def calculate_cell(slice_path,roi_list,config):
    cls_labels_with_id = {}
    center_coords_with_id = {}
    for idx, roi in enumerate(roi_list):
        roiid, x_coords, y_coords = roi['id'], roi['x'], roi['y']
        name = os.path.splitext(os.path.basename(slide_path))[0]
        region_activate = config.get('region', 'region_activate')

def nearest_power_of_two(number):
    power = round(math.log2(number))
    return int(math.pow(2, power))


