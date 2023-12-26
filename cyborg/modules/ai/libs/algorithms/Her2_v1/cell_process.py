import sys
import os
import utils
import configparser
import numpy as np
import torch
import cv2
import json
import argparse
import time

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from seg_tissue_area import find_tissue_countours
from models.pa_p2pnet import build_model

from cyborg.libs.heimdall.dispatch import open_slide
from cyborg.modules.ai.utils.file import load_alg_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
current_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_root)
config_path = os.path.join(current_root, 'config.ini')


def gather_together(data):
    dist.barrier()
    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)
    return np.concatenate(gather_data, axis=0)


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
    return result, classes, probs


@torch.no_grad()
def predict(model, images, apply_deduplication: bool = False, class_num=6):
    points_batch_list = [0, ] * images.shape[0]
    classes_batch_list = [0, ] * images.shape[0]
    scores_batch_list = [0, ] * images.shape[0]
    h, w = images.shape[-2:]
    outputs = model(images)
    for i in range(images.shape[0]):
        points = outputs['pnt_coords'][i].cpu().numpy()
        scores = torch.softmax(outputs['cls_logits'][i], dim=-1).cpu().numpy()
        not_select_idx = (points[:, 0] < 0) | (points[:, 0] >= w) | (points[:, 1] < 0) | (points[:, 1] >= h)
        points = points[~not_select_idx]
        scores = scores[~not_select_idx]

        classes = np.argmax(scores, axis=-1)

        reserved_index = classes < class_num

        points, labels, probs = deduplicate(points[reserved_index], scores[reserved_index], 16)

        points_batch_list[i] = points.astype(np.int)
        classes_batch_list[i] = labels.astype(np.int)
        scores_batch_list[i] = probs.astype(np.float)

    return points_batch_list, classes_batch_list, scores_batch_list


def get_opt():
    parser = argparse.ArgumentParser(description='manual to this script')

    parser.add_argument('--num_classes', type=int, default=6,
                        help="Number of cell categories")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--row', default=2, type=int, help="number of anchor points per row")
    parser.add_argument('--col', default=2, type=int, help="number of anchor points per column")

    parser.add_argument('--ppid', default=1000)
    parser.add_argument('--roi_x', default=[])
    parser.add_argument('--roi_y', default=[])
    parser.add_argument('--slice_path', default='')

    args = parser.parse_args()
    return args


def main_worker(gpu, world_size, cell_dict):
    dist.init_process_group(backend=cell_dict['backend'],
                            init_method=cell_dict['dist_url'],
                            world_size=world_size,
                            rank=gpu)
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    slide = open_slide(cell_dict['slice_path'])
    '''
    find Foreground

    '''
    if len(cell_dict['roi_x']) == 0:
        if cell_dict['foreground_activate']:
            threshold_map, _ = find_tissue_countours(slide, cell_dict['thresh_scale'])
            threshold_map = np.stack((threshold_map,) * 3, axis=-1)
        else:
            threshold_map = None
    else:
        threshold_map = utils.produce_roi_threshold_map(cell_dict['roi_x'], cell_dict['roi_y'], slide,
                                                        cell_dict['thresh_scale'])

    # save threshold map
    if gpu == 0 and threshold_map is not None:
        name = os.path.basename(cell_dict['slice_path']).split('.')[0]
        mask_save_path = current_root + '/tmp_data/' + name + '_test.png'
        cv2.imwrite(mask_save_path, threshold_map)
    '''
    load region map

    '''
    name = os.path.basename(cell_dict['slice_path']).split('.')[0]
    mask_save_path = current_root + '/tmp_data/' + name + '_mask.png'
    if os.path.exists(mask_save_path):
        region_map = cv2.imread(mask_save_path)
    else:
        region_map = None
    '''
    load model 

    '''
    opt = get_opt()
    torch.cuda.set_device(gpu)
    torch.cuda.empty_cache()
    C_net = build_model(opt).cuda(gpu)
    model_file = load_alg_model(cell_dict['cell_model'])
    checkpoint = torch.load(model_file, map_location={'cuda:0': f'cuda:{gpu}'})
    model_dict = C_net.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    C_net.load_state_dict(model_dict)
    '''
    dataset 

    '''
    dataset_producer = utils.dataset_producer(slide=slide)
    dataset_producer.set_cell_dataset_config(threshold_map=threshold_map, region_map=region_map, config_dict=cell_dict)
    dataset = dataset_producer.start_cell()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    patch_loader = torch.utils.data.DataLoader(dataset, batch_size=10,
                                               num_workers=0, pin_memory=True, sampler=sampler)
    # print('cell_patch_loader',len(patch_loader))
    gathered_list = None

    # load config#

    cell_deduplicate_activate = cell_dict['cell_deduplicate_activate']
    cell_deduplicate_range = cell_dict['cell_deduplicate_range']
    region_activate = cell_dict['region_activate']

    gathered_wsi_cell_center_coords = []
    gathered_wsi_cell_labels = []
    gathered_wsi_patch_xy = []
    gathered_wsi_cell_probs = []

    if gpu == 0:
        time_1 = time.time()

    #    wsi_cell_center_coords = []
    #    wsi_cell_labels = []
    #    wsi_patch_xy = []
    #    wsi_cell_probs = []

    for idx, batch in enumerate(patch_loader):
        if not utils.check_parent_pid(cell_dict['pid']):
            return
        img = batch['cur_region'].permute(0, 2, 3, 1)  # bchw -> bhwc
        mask = batch['mask']  # bchw
        xmin_list = batch['xmin'].cpu().numpy()
        ymin_list = batch['ymin'].cpu().numpy()
        img = img.float() / 255
        img = img.permute(3, 1, 2, 0)  # bhwc -> chwb
        mean = [0.81301117, 0.78663919, 0.77938885]
        std = [0.1537656, 0.18297883, 0.21775213]
        for t, m, s in zip(img, mean, std):
            t.sub_(m).div_(s)
        img = img.permute(3, 0, 1, 2)  # chwb -> bchw
        img = img.cuda(gpu)
        with torch.no_grad():
            batch_points, batch_cls, batch_prob = predict(C_net, img, apply_deduplication=True, class_num=6)
        # batch_size = len(pd_batch_points) 1
        batch_size = len(batch_points)
        for batch_num in range(batch_size):
            _, mask_h, mask_w = mask[batch_num].shape
            deduplicate_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

            pts, labels, probs = batch_points[batch_num], batch_cls[batch_num], batch_prob[batch_num]
            for pt_idx in range(pts.shape[0]):
                x, y = int(pts[pt_idx, 0]), int(pts[pt_idx, 1])
                if mask[batch_num, 0, y, x] == 0 and region_activate:
                    labels[pt_idx] = opt.num_classes
                if cell_deduplicate_activate:
                    sum = np.sum(deduplicate_mask[y - cell_deduplicate_range:y + cell_deduplicate_range,
                                 x - cell_deduplicate_range:x + cell_deduplicate_range])
                    if sum > 0:
                        labels[pt_idx] = opt.num_classes
                    else:
                        deduplicate_mask[y, x] = 1
            reserved_index = labels < opt.num_classes
            pts, labels, probs = pts[reserved_index], labels[reserved_index], probs[reserved_index]

            pts[:, 0] = (pts[:, 0] * cell_dict['mpp_ratio']) + xmin_list[batch_num]
            pts[:, 1] = (pts[:, 1] * cell_dict['mpp_ratio']) + ymin_list[batch_num]
            patch_xy = np.array([xmin_list[batch_num], ymin_list[batch_num]] * pts.shape[0])
            # patch_xy = [ptx,pty,ptx,pty,....]
            # pts = [[x,y],....]

            labels = gather_together(labels)
            probs = gather_together(probs)
            pts = gather_together(pts)
            patch_xy = gather_together(patch_xy)
            gathered_wsi_cell_labels.extend(labels)
            gathered_wsi_cell_probs.extend(probs)
            gathered_wsi_patch_xy.extend(patch_xy)
            gathered_wsi_cell_center_coords.extend(pts)

        # if idx > 5:
        #     break
        # gathered_wsi_cell_labels = torch.cat((gathered_wsi_cell_labels,labels),dim=0)
        # gathered_wsi_cell_probs = torch.cat((gathered_wsi_cell_probs,probs),dim=0)
        # gathered_wsi_patch_xy = torch.cat((gathered_wsi_patch_xy,patch_xy),dim=0)
        # gathered_wsi_cell_center_coords = torch.cat((gathered_wsi_cell_center_coords,pts),dim=0)
    if gpu == 0:
        time_2 = time.time()
        print('cell process time: {:.2f} mins'.format((time_2 - time_1) / 60))
        print(len(gathered_wsi_cell_labels))
        result_to_npz(gathered_wsi_cell_center_coords, gathered_wsi_cell_labels, gathered_wsi_patch_xy,
                      gathered_wsi_cell_probs, cell_dict['slice_path'])
        # print(gathered_wsi_patch_xy)
        # print(len(gathered_wsi_cell_labels))
        # print(gathered_wsi_cell_labels)
        # print(len(gathered_wsi_cell_center_coords))
        # print(gathered_wsi_cell_center_coords)


def result_to_npz(points, labels, patch_xy, probs, slide_path=''):
    base_folder = os.path.dirname(slide_path)
    npz_name = 'her2_ai_result.npz'
    save_path = os.path.join(base_folder, npz_name)
    if os.path.exists(save_path):
        os.remove(save_path)
    print(save_path)
    time_1 = time.time()
    np.savez(save_path, points=points, labels=labels, patch_xy=patch_xy, probs=probs)
    time_2 = time.time()
    print('save process time: {:.2f} mins'.format((time_2 - time_1) / 60))


if __name__ == '__main__':
    # command = ['python', cell_python,str(main_pid),slice_path,roi_x,roi_y]
    cell_dict = {}
    opt = get_opt()
    cell_dict['pid'] = opt.ppid
    cell_dict['slice_path'] = opt.slice_path
    cell_dict['roi_x'] = json.loads(opt.roi_x)
    cell_dict['roi_y'] = json.loads(opt.roi_y)

    print('main_pid: ', cell_dict['pid'])
    config = configparser.ConfigParser()
    config.read(config_path)
    '''
    cell config 

    '''
    cell_dict['cell_crop_len'] = int(config.get('cell', 'crop_len'))
    cell_model = config.get('cell', 'cell_model')
    cell_dict['cell_model'] = os.path.join('AI', 'Her2_v1', 'Model', cell_model)
    cell_dict['cell_deduplicate_activate'] = config.get('cell', 'cell_deduplicate_activate').strip().lower() == 'true'
    cell_dict['cell_deduplicate_range'] = int(config.get('cell', 'cell_deduplicate_range'))

    '''
    common

    '''
    mpp_std = float(config.get('common', 'mpp_std'))
    cur_mpp = open_slide(cell_dict['slice_path']).mpp
    if cur_mpp is None:
        cur_mpp = mpp_std
    cur_mpp = 0.25
    # mpp_ratio = 0.25/round(cur_mpp,3)
    mpp_ratio = mpp_std / round(cur_mpp, 3)
    cell_dict['mpp_ratio'] = mpp_ratio
    cell_dict['cur_mpp'] = cur_mpp
    cell_dict['foreground_activate'] = config.get('common', 'foreground_activate').strip().lower() == 'true'
    cell_dict['thresh_scale'] = int(config.get('common', 'thresh_scale'))
    cell_dict['region_activate'] = config.get('region', 'region_activate').strip().lower() == 'true'

    port_id = 10000 + np.random.randint(0, 1000)
    cell_dict['dist_url'] = 'tcp://127.0.0.1:' + str(port_id)
    num_gpus = torch.cuda.device_count()
    cell_dict['num_gpus'] = num_gpus
    cell_dict['backend'] = 'gloo'

    if num_gpus == 1:
        preds = main_worker(gpu=0, world_size=num_gpus, cell_dict=cell_dict)
    else:
        torch.multiprocessing.set_start_method('spawn')
        mp.spawn(main_worker, nprocs=num_gpus, args=(num_gpus, cell_dict))
