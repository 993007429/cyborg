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
