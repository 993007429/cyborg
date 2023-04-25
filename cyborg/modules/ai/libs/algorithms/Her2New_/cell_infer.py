import os
import sys
import argparse
import random
import math
import torch
import time
import cv2
from dataset import Dataset
from utils import *
from tqdm import tqdm
import numpy as np
from progress.bar import Bar
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
from logger import logger
global label_to_idx_dict,idx_to_label_dict
from torchvision import transforms
import random
import psutil
import signal
#from tensorrt_utils import load_engine_and_infer,reshape_outputs
label_to_idx_dict = {
        '微弱的不完整膜阳性肿瘤细胞': 0,
        '弱-中等的完整细胞膜阳性肿瘤细胞': 1,
        '阴性肿瘤细胞': 2,
        '强度的完整细胞膜阳性肿瘤细胞': 3,
        '中-强度的不完整细胞膜阳性肿瘤细胞': 4,
        "其他": 5,
    }

color_card_hex={2:'#00FF00',
                        0:'#FF3399',
                        3:'#FF0033',
                        1:'#FF6633',
                        4:'#4051B5',
                        5: "#AAEA63",
                }

idx_to_label_dict =  {v: k for k, v in label_to_idx_dict.items()}
import torch.distributed as dist
def init_device(LOCAL_RANK,world_size):
    # DDP设置
    torch.backends.cudnn.benchmark = True
    if LOCAL_RANK != -1:
        # assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        # assert 4 % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="gloo", rank=LOCAL_RANK, world_size=world_size)  # nccl & gloo
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device

def deduplicate(points, scores, interval):
    n = len(points)
    fused = np.full(n, False)
    result = np.zeros((0, 2))
    classes = np.array([])
    probs = np.array([])
    for i in range(n):
        if not fused[i]:
            fused_index = np.where(np.linalg.norm(points[[i]] - points[i:], 2, axis=1) < interval)[0] + i
            fused[fused_index] = True
            r_, c_ = np.where(scores[fused_index] == np.max(scores[fused_index]))
            p_ = np.max(scores[fused_index], axis=1)
            r_, c_, p_ = [r_[0]], [c_[0]], [p_[0]]
            result = np.append(result, points[fused_index[r_]], axis=0)
            classes = np.append(classes, c_)
            probs = np.append(probs, p_)
    return result, classes


@torch.no_grad()
def predict(model, images, apply_deduplication: bool = False,opt=None):
    points_batch_list = [0, ] * images.shape[0]
    classes_batch_list = [0, ] * images.shape[0]
    h, w = images.shape[-2:]
    # logger.info(f'SHAPPPPPE, images:{images.shape}')
    outputs = model(images)
    for i in range(images.shape[0]):
        points = outputs['pnt_coords'][i].cpu().numpy()
        scores = torch.softmax(outputs['cls_logits'][i], dim=-1).cpu().numpy()
        not_select_idx = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
        # select_idx = np.where(points[:, 0] < w and points[:,0]>=0 and points[:,1]>=0 and points[:, 1] < h)
        points = points[~not_select_idx]
        scores = scores[~not_select_idx]

        classes = np.argmax(scores, axis=-1)

        reserved_index = classes < 6
        # print('before deduplication')
        # print(opt.num_classes)
        # print(points[reserved_index].shape)

        points,labels = deduplicate(points[reserved_index], scores[reserved_index], 16)

        # print('after deduplication')
        # print(points.shape)


        points_batch_list[i] = points.astype(np.int)
        classes_batch_list[i] = labels.astype(np.int)

    return points_batch_list,classes_batch_list

def predict_image(model,image,apply_deduplication: bool = False,opt=None):
    outputs = model(image)
    points = outputs['pnt_coords'][0].cpu().numpy()
    scores = torch.softmax(outputs['cls_logits'][0], dim=-1).cpu().numpy()
    select_idx = np.where(np.logical_and(points[:, 0] < image.shape[-1], points[:, 1] < image.shape[-2]))[0]
    points = points[select_idx]
    scores = scores[select_idx]

    classes = np.argmax(scores, axis=-1)
    reserved_index = classes < opt.num_classes
    if apply_deduplication:
        points, labels = deduplicate(points[reserved_index], scores[reserved_index], 12)
    else:
        points, labels = points[reserved_index], scores[reserved_index]

    return points,labels





        #cell_count_dict,wsi_cell_center_coords,wsi_cell_labels,masks_label_list,masks_coords_list

def Hex_to_RGB(hex):
    r = int(hex[1:3],16)
    g = int(hex[3:5],16)
    b = int(hex[5:7], 16)
    rgb = (r,g,b)
    return rgb

def is_process_running(pid):
    pl = psutil.pids()
    for id in pl:
        if id == pid:
            return True
    else:
        return False

def kill(pid):
    os.kill(pid, signal.SIGINT)

@torch.no_grad()
def cell_infer_(C_net,crop_coords_wsi,ext,lvl_list,mask,slide,device,opt_,batch_size,model_name):
    #mask_lvl = int(slide.height//mask.shape[0])
    mask_lvl = 16
    count = 0
    randint = random.randint(5, 50)
    logger.info(f'Mask_lvl:  {mask_lvl}')
    has_mask = True #if mask.any()!=None else False

    wsi_dataset = Dataset(slide, crop_coords_wsi, wsi_type=ext, crop_len=1024, lvl=lvl_list[0], mask_lvl=mask_lvl,
                                             mask=mask, has_Mask=has_mask)

    # print('a')
    # print('crop_coords')
    # print(crop_coords_wsi)
    logger.info('C_net0412')

    # train_size = int(0.8  *  len(full_dataset))
    # test_size = len(full_dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # train_size = int((1/6)*wsi_dataset.nF)
    # others = wsi_dataset.nF - train_size
    # train_dataset,_ = torch.utils.data.random_split(wsi_dataset,[train_size,others])

    patch_loader = torch.utils.data.DataLoader(
        wsi_dataset,
        batch_size=batch_size,
        num_workers = 0,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    num_iters = len(patch_loader)
    C_net.to(device)
    if RANK in [-1, 0]:  # 显示进度
        bar = Bar('{}:'.format("cell_detection"), max=num_iters)

    for idx , batch in enumerate(patch_loader):
        #slogger.info(f'Batch_SIIIIIIIZZZZZEE: {batch.shape}')
        if not is_process_running(opt_.ppid):
            return wsi_cell_labels,wsi_cell_center_coords,wsi_patch_xy,False

        count=count+1
        torch.cuda.empty_cache()
        t0 = time.time()
        image = batch['cur_region']
        mask = batch['mask']# bchw
        xmins,ymins,xmaxs,ymaxs = batch['xmin'].cpu().numpy(),\
                                  batch['ymin'].cpu().numpy(),\
                                  batch['xmax'].cpu().numpy(),\
                                  batch['ymax'].cpu().numpy()
        image = image.permute(0,2,3,1)
        image_draw = image
        # print('image shape3')
        # print(image.shape)

        image = image.float() / 255
        # image(b,h,w,c) -> image(c,h,w,b) -> image(b,c,h,w)
        image = image.permute(3, 1, 2, 0)
        mean, std = np.load(f'wsi_infer/C_net{model_name}/mean_std.npy')
        for t, m, s in zip(image, mean, std):
            t.sub_(m).div_(s)
        image = image.permute(3, 0, 1, 2)
        image = image.cuda(device)[:, :, :, :]
        t1 = time.time()
        with torch.no_grad():
            pd_batch_points, pd_batch_classes = predict(C_net, image, apply_deduplication=True,opt=opt_)
        t2 = time.time()

        # print('len pd')
        # print(len(pd_batch_points))
        for iidx in range(len(pd_batch_points)):
            points, labels = pd_batch_points[iidx], pd_batch_classes[iidx]
            for index in range(points.shape[0]):
                x = int(points[index,0])
                y = int(points[index,1])

                if mask[iidx,0,y,x] == 0:
                    # logger.info('mask 0')
                    labels[index] = opt_.num_classes
                    # logger.info(labels[index])
            reserved_index = labels < opt_.num_classes
            points, labels = points[reserved_index], labels[reserved_index]

            if opt_.vis!='none':
                patch = image_draw[iidx].cpu().numpy().astype(np.uint8).copy()
                for idxxx, point in enumerate(points):
                    pd_class = labels[idxxx]
                    color_card = Hex_to_RGB(color_card_hex[pd_class])
                    patch = cv2.circle(patch, (int(point[0]), int(point[1])), radius=0, color=color_card, thickness=10)
                patch = patch[...,::-1].copy()
                cv2.imwrite(opt_.vis+'/' + 'result'+str(randint)+'_'+str(count)+'_'+ str(iidx)+ '.png', patch)


            points[:,0]  =  points[:,0]+ xmins[iidx]
            points[:, 1] = points[:, 1] + ymins[iidx]
            patch_xy = [xmins[iidx],ymins[iidx]] * points.shape[0]


            wsi_patch_xy.extend(patch_xy)
            wsi_cell_center_coords.extend(points)
            wsi_cell_labels.extend(labels)

        # print('center_coords shape')
        # print(wsi_cell_center_coords[0])


        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} |Time:{time:.3f}'.format(idx, num_iters, total=bar.elapsed_td,
                                                                   eta=bar.eta_td,time=time.time()-t0)
        logger.info('[{0}/{1}]|Tot: {total:} |ETA: {eta:} |Time:{time:.3f}'.format(idx, num_iters, total=bar.elapsed_td,
                                                                   eta=bar.eta_td,time=time.time()-t0))
        bar.next()

    return wsi_cell_labels,wsi_cell_center_coords,wsi_patch_xy,True

wsi_cell_center_coords = []
wsi_cell_labels = []
wsi_patch_xy = []

def cell_infer__(crop_coords_wsi,ext,lvl_list,mask,slide,opt_,batch_size):
    mask_lvl = 16 #math.ceil(slide.height//mask.shape[0])
    count = 0
    randint = random.randint(5, 50)
    logger.info(f'Mask_lvl:  {mask_lvl}')
    wsi_dataset = Dataset(slide, crop_coords_wsi, wsi_type=ext, crop_len=1024, lvl=lvl_list[0], mask_lvl=mask_lvl,mask=mask, has_Mask=True)
    patch_loader = torch.utils.data.DataLoader(
        wsi_dataset,
        batch_size=batch_size,
        num_workers = 0,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    num_iters = len(patch_loader)
    if RANK in [-1, 0]:  # 显示进度
        bar = Bar('{}:'.format("cell_detection"), max=num_iters)
    for idx , batch in enumerate(patch_loader):
        if not is_process_running(opt_.ppid):
            return wsi_cell_labels,wsi_cell_center_coords,wsi_patch_xy,False
        count=count+1
        torch.cuda.empty_cache()
        t0 = time.time()
        image = batch['cur_region']
        mask = batch['mask']# bchw
        xmins,ymins,xmaxs,ymaxs = batch['xmin'].cpu().numpy(),\
                                  batch['ymin'].cpu().numpy(),\
                                  batch['xmax'].cpu().numpy(),\
                                  batch['ymax'].cpu().numpy()
        image = image.permute(0,2,3,1)
        image_draw = image
        image = image.float() / 255
        image = image.permute(3, 1, 2, 0)
        mean, std = np.load('wsi_infer/C_net1025/mean_std.npy')
        for t, m, s in zip(image, mean, std):
            t.sub_(m).div_(s)
        image = image.permute(3, 0, 1, 2)
        h, w = image.shape[-2:]
        # image = image.cuda(device)[:, :, :, :]
        t1 = time.time()
        image = np.array(image)
        pnt_coords,cls_logits = load_engine_and_infer('her2_.engine',image)
        pnt_coords,cls_logits = reshape_outputs(pnt_coords,cls_logits)
        t2 = time.time()

        for iidx in range(xmins.shape[0]):
            points, labels = pnt_coords[iidx], cls_logits[iidx]
            scores = torch.softmax(torch.from_numpy(labels), dim=-1).cpu().numpy()
            not_select_idx = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
            points = points[~not_select_idx]
            scores = scores[~not_select_idx]
            classes = np.argmax(scores, axis=-1)
            reserved_index = classes < 6
            points,labels = deduplicate(points[reserved_index], scores[reserved_index], 16)
            for index in range(points.shape[0]):
                x = int(points[index,0])
                y = int(points[index,1])
                if mask[iidx,0,y,x] == 0:
                    labels[index] = opt_.num_classes
                    # logger.info(labels[index])
            reserved_index = labels < opt_.num_classes
            points, labels = points[reserved_index], labels[reserved_index]

            if opt_.vis!='none':
                patch = image_draw[iidx].cpu().numpy().astype(np.uint8).copy()
                for idxxx, point in enumerate(points):
                    pd_class = labels[idxxx]
                    color_card = Hex_to_RGB(color_card_hex[pd_class])
                    patch = cv2.circle(patch, (int(point[0]), int(point[1])), radius=0, color=color_card, thickness=10)
                patch = patch[...,::-1].copy()
                cv2.imwrite(opt_.vis+'/' + 'result'+str(randint)+'_'+str(count)+'_'+ str(iidx)+ '.png', patch)


            points[:,0] = points[:,0] + xmins[iidx]
            points[:,1] = points[:,1] + ymins[iidx]
            patch_xy = [xmins[iidx],ymins[iidx]] * points.shape[0]


            wsi_patch_xy.extend(patch_xy)
            wsi_cell_center_coords.extend(points)
            wsi_cell_labels.extend(labels)

        # print('center_coords shape')
        # print(wsi_cell_center_coords[0])


        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} |Time:{time:.3f}'.format(idx, num_iters, total=bar.elapsed_td,
                                                                   eta=bar.eta_td,time=time.time()-t0)
        logger.info('[{0}/{1}]|Tot: {total:} |ETA: {eta:} |Time:{time:.3f}'.format(idx, num_iters, total=bar.elapsed_td,
                                                                   eta=bar.eta_td,time=time.time()-t0))
        bar.next()


    return wsi_cell_labels,wsi_cell_center_coords,wsi_patch_xy,True





