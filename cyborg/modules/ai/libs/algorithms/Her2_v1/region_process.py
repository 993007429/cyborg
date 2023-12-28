import logging
import sys,os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import utils
import configparser
import numpy as np
import torch
from cyborg.libs.heimdall.dispatch import open_slide
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
current_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_root)
config_path = os.path.join(current_root,'config.ini')

logger = logging.getLogger(__name__)


def scale_coords(img_len,coord,padding,mask):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    mask_h,mask_w,_ = mask.shape
    coord = coord.numpy()
    x_min,y_min,x_max,y_max,confidence,class_num,st_x,st_y,lvl = coord
    st_x = st_x / lvl
    st_y = st_y / lvl
    xmin_abs = x_min + st_x
    ymin_abs = y_min + st_y
    xmax_abs = x_max + st_x
    ymax_abs = y_max + st_y
   
    x_min = xmin_abs- padding
    y_min = ymin_abs - padding 
    x_max = xmax_abs + padding
    y_max = ymax_abs + padding
    x_min = np.clip(x_min,0,mask_w)
    y_min = np.clip(y_min,0,mask_h)
    x_max = np.clip(x_max,0,mask_w)
    y_max = np.clip(y_max,0,mask_h)
    return [int(x_min),int(y_min),int(x_max),int(y_max)]

def gather_together(data):
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return torch.cat(gather_data,dim=0)

def main_worker(gpu,world_size,region_dict):
    dist.init_process_group(backend=region_dict['backend'],
    init_method=region_dict['dist_url'],
    world_size=world_size,
    rank=gpu)
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    '''
    load model 

    '''
    logger.info(region_dict['region_model'])
    model = utils.load_model(region_dict['region_model'],device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],
                                                      find_unused_parameters=True)
    model = model.half()
    '''
    dataset 

    '''
    slide = open_slide(region_dict['slice_path'])
    mask_scale_list = region_dict['mask_scale']
    thresh_scale = region_dict['thresh_scale']
    

    slide_draw = np.zeros((int(slide.height/thresh_scale), int(slide.width/thresh_scale), 3), dtype=np.uint8)
    final_mask = np.ones(slide_draw.shape, dtype=np.uint8) * 255
    image = slide.read(scale=thresh_scale) 

    for mask_scale in mask_scale_list:
        print('mask_scale:',mask_scale)

        slide_draw = np.zeros((int(slide.height/mask_scale), int(slide.width/mask_scale), 3), dtype=np.uint8)
        window_size = (region_dict['region_crop_len'],region_dict['region_crop_len'])
        dataset = utils.dataset_producer(slide,window_size,
        region_dict['region_stride_ratio'],
        mask_scale).start()

        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        patch_loader = torch.utils.data.DataLoader(dataset,batch_size=1,
                    num_workers=0, pin_memory=True, sampler=sampler)
        print('patch_loader',len(patch_loader))
        gathered_list = None
        for idx,patch in enumerate(patch_loader):
            if not utils.check_parent_pid(region_dict['pid']):
                return
            inputs = patch['cur_region'][0].half()
            x = patch['xmin'][0]
            y = patch['ymin'][0]
            sc = patch['lvl'][0]
            inputs /= 255.0
            if inputs.ndimension() == 3:
                inputs = inputs.unsqueeze(0)
            inputs = inputs.to(gpu)
            model.eval()
            with torch.no_grad():
                preds = model(inputs)[0]
            preds = utils.non_max_suppression_(preds,region_dict)[0].cpu()
            num_row = preds.shape[0]
            x_col = torch.full((num_row,1),x)
            y_col = torch.full((num_row,1),y)
            sc_col = torch.full((num_row,1),sc)
            preds = torch.cat((preds,x_col,y_col,sc_col),dim=1)
            preds = gather_together(preds)
            #print(preds.shape)
            if gathered_list == None:
                gathered_list = preds
            else:
                gathered_list = torch.cat((gathered_list,preds),dim=0)

        if gpu == 0:
            mask = np.ones(slide_draw.shape, dtype=np.uint8) * 255
            for coord in gathered_list:
                new_coords = scale_coords(region_dict['region_crop_len'],coord,10,mask)
                #[int(x_min),int(y_min),int(x_max),int(y_max)]
                xmin,ymin,xmax,ymax = new_coords
                #print(new_coords)
                cur_mask = np.zeros((int(ymax-ymin),int(xmax-xmin),3),dtype=np.uint8)
                mask[ymin:ymax, xmin:xmax] = cv2.bitwise_and(mask[ymin:ymax, xmin:xmax], cur_mask)
            final_mask = utils.merge_mask(final_mask,mask)
    if gpu == 0:           
        name = os.path.basename(region_dict['slice_path']).split('.')[0]
        mask_save_path = os.path.join(current_root,'tmp_data' , name + '_mask.png')
        cv2.imwrite(mask_save_path,final_mask)

        if region_dict['region_vis']:
            mask_save_path = os.path.join(current_root,'tmp_data', name + '_vis.png')
            alpha = 0.5
            final_mask = cv2.resize(final_mask,(image.shape[1],image.shape[0]))
            masked_image = cv2.addWeighted(image, 1 - alpha,final_mask, alpha, 0)
            cv2.imwrite(mask_save_path,masked_image)         