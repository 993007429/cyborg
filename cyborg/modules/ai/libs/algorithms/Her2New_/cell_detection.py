import logging
import os
import json
import argparse
import sys

import cv2
import torch
import numpy as np
import mpi4py.MPI as MPI

her2_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(her2_root)

from cyborg.infra.oss import oss
from models.pa_p2pnet import build_model
import cell_utils
from cell_infer import cell_infer_
from seg_tissue_area import find_tissue_countours

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

logger = logging.getLogger(__name__)

segmentation_cls = 4


@torch.no_grad()
def detect(slice_path='', opt=None):
    logger.info('调用her2new_new compute_process')
    comm = MPI.COMM_WORLD
    comm_rank, comm_size = comm.Get_rank(), comm.Get_size()
    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
    else:
        gpu_num = 1

    int_device = int(comm_rank % gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int_device)
    # # C_net1025 original 
    model_name = '0505'

    # model
    torch.cuda.set_device(int_device)
    torch.cuda.empty_cache()
    c_net = build_model(opt).cuda(int_device)

    model_file = oss.get_object_to_io(oss.path_join('AI', 'Her2New_', 'wsi_infer', f'C_net{model_name}/her2.pth'))
    checkpoint = torch.load(model_file, map_location={'cuda:0': f'cuda:{int_device}'})

    model_dict = c_net.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    c_net.load_state_dict(model_dict)

    # slice
    slide, lvl_list, size_list, ext = cell_utils.get_slice(slice_path)
    logger.info(f'finished load model {model_name}')
    roi_coords = opt.roi

    # wsi lvl
    h0, w0 = size_list[0]
    thresh_scale = 16
    maskw, maskh = 0, 0
    if opt.mask:
        mask = cv2.imread(opt.mask)
        maskh, maskw, _ = mask.shape
        logger.info(f'Mask_shape:{mask.shape}')
        threshold_map = np.zeros((maskh, maskw))
    else:
        threshold_map = np.zeros((slide.height // thresh_scale, slide.width // thresh_scale))
        mask = None

    if roi_coords is not None:
        x_coords, y_coords = json.loads(roi_coords[0]), json.loads(roi_coords[1])
    else:
        x_coords, y_coords = None, None
    if not x_coords:
        logger.info('Do Not Have ROI')
        threshold_map, flg = find_tissue_countours(slide, thresh_scale)
        if not flg:
            threshold_map, flg = find_tissue_countours(slide, 32)
        logger.info(f'Threshold_map2:{threshold_map.shape}')
    else:
        logger.info('Have ROI')
        x_coords_ = np.array(x_coords)
        y_coords_ = np.array(y_coords)
        x_coords_ = x_coords_ / thresh_scale
        y_coords_ = y_coords_ / thresh_scale
        contours = [np.expand_dims(np.stack([x_coords_.astype(int), y_coords_.astype(int)], axis=1), axis=1)]
        threshold_map = cv2.fillPoly(threshold_map, contours, 1)

    if opt.mask:
        threshold_map = cv2.resize(threshold_map, (maskw, maskh))
        logger.info(f'Threshold_map:{threshold_map.shape}')

    lvl_list[3] = 16
    logger.info(f'lvl_list : {lvl_list}')

    crop_coords_wsi = cell_utils.get_patch_with_contours(slide_w=w0, slide_h=h0, save=False, lvl=lvl_list[0],
                                                         map=threshold_map, scale=thresh_scale)

    crop_list = np.array_split(crop_coords_wsi, comm_size)
    local_data = comm.scatter(crop_list, root=0)
    logger.info('start cell detection')

    wsi_cell_labels, wsi_cell_center_coords, wsi_cell_patch_xy, flg = cell_infer_(c_net, local_data, ext, lvl_list,
                                                                                  mask, slide, int_device, opt, 6,
                                                                                  model_name=model_name)

    if not flg:
        comm.gather([], root=0)
        logger.info('end cell detection')
        return

    combine_test_coords = comm.gather(wsi_cell_center_coords, root=0)
    combine_test_labels = comm.gather(wsi_cell_labels, root=0)
    combine_patch_xy = comm.gather(wsi_cell_patch_xy, root=0)
    final_coords = []
    final_labels = []
    final_xy = []

    if combine_test_labels is not None:
        for coords in combine_test_coords:
            final_coords.extend(coords)
        for labels in combine_test_labels:
            final_labels.extend(labels)
        for xy in combine_patch_xy:
            final_xy.extend(xy)

    result_to_json(final_coords, final_labels, final_xy, slide_path=slice_path)
    logger.info('Cell Detection Finished')


def result_to_json(points, labels, stps, slide_path):
    points = np.array(points)
    labels = np.array(labels)
    stps = np.array(stps)
    x_coords = [float(coord[0]) for coord in points]
    y_coords = [float(coord[1]) for coord in points]
    stp = [pp.tolist() for pp in stps]
    dict2 = {'class': labels.tolist()}
    dict1 = {'x': x_coords, 'y': y_coords, 'stxy': stp}
    result_root = os.path.dirname(slide_path)
    coord_json_name = 'her2_coords_wsi.json'
    label_json_name = 'her2_label_wsi.json'
    with open(os.path.join(str(result_root), coord_json_name), 'w', encoding="utf-8") as result_file:
        json.dump(dict1, result_file)
    with open(os.path.join(str(result_root), label_json_name), 'w', encoding="utf-8") as result_file:
        json.dump(dict2, result_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--slice_root', default='data/her2Slice', action='store_true', help='don`t trace model')
    parser.add_argument('--save_mask', default=True)
    parser.add_argument('--slide', type=str, default="/data1/Caijt/PDL1_Parallel/A080 PD-L1 V+.kfb",
                        help='Slide Path')

    # * Optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # * Train
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--start_eval', default=0, type=int)

    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='her2_pure.pth',
                        help='resume from checkpoint')
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--num_classes', type=int, default=6,
                        help="Number of cell categories")

    # * Loss
    parser.add_argument('--reg_loss_coef', default=2e-3, type=float)
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.55, type=float,
                        help="Relative classification weight of the no-object class")

    # * Matcher
    parser.add_argument('--set_cost_point', default=0.1, type=float,
                        help="L2 point coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    # * Model
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
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

    # * Dataset
    parser.add_argument('--dataset', default='test', type=str)
    parser.add_argument('--num_workers', default=8, type=int)

    # * Evaluator
    parser.add_argument('--match_dis', default=12, type=int)

    # * Distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--num_process', default=2)
    parser.add_argument('--mask', default='')
    parser.add_argument('--vis', default='')
    parser.add_argument('--roi', type=str, nargs=2, default=None)
    parser.add_argument('--ppid', default=12345, type=int)

    args = parser.parse_args()
    slice = args.slide
    opt = args

    detect(slice_path=slice, opt=opt)
