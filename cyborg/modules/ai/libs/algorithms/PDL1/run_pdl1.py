import os
import sys
import stat
from typing import List

import cv2
import torch
import numpy as np
import subprocess
import json
import argparse
import mpi4py.MPI as MPI
from itertools import chain
from configparser import ConfigParser

from cyborg.infra.oss import oss
from cyborg.libs.heimdall.dispatch import open_slide
from cyborg.modules.ai.libs.algorithms.PDL1.src.cell_det_cls import cal_pdl1_np
from cyborg.modules.ai.libs.algorithms.PDL1.src.utils import delete_prev_json, split_patches, split2groups, \
    dump_results, map_results, roi_filter, split_patches_map
from cyborg.modules.ai.libs.algorithms.PDL1.src.seg_tissue_area import find_tissue_countours
from cyborg.modules.ai.libs.algorithms.PDL1.models.detr import build_model


libs_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_dir = os.path.dirname(os.path.abspath(__file__))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def model_selection():
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alg_config.ini')
    config = ConfigParser()
    config.read(config_file, encoding='utf-8')
    model_type = config.get('custom', 'customName')
    if model_type == 'general':
        model_path = config.get(model_type, 'modelName')
        net_weights_model = oss.get_object_to_io(oss.path_join('AI', 'PDL1', 'Model', model_path))
        net_weights_dict = torch.load(net_weights_model, map_location=lambda storage, loc: storage)
        mean_std_data = oss.get_object_to_io(
            oss.path_join('AI', 'PDL1', 'Model', 'mean_std', config.get(model_type, 'meanstdVersion')))
        mean_std = np.load(mean_std_data)
        return net_weights_dict, mean_std
    else:
        model_root = model_type
        model_version = config.get(model_type, 'modelVersion')
        model_time = config.get(model_type, 'modelTime')
        model_path = os.path.join(model_root, f'{model_version}_{model_time}.pth')
        net_weights_model = oss.get_object_to_io(oss.path_join('AI', 'PDL1', 'Model', model_path))
        net_weights_dict = torch.load(net_weights_model, map_location=lambda storage, loc: storage)['model']
        mean_std_data = oss.get_object_to_io(
            oss.path_join('AI', 'PDL1', 'Model', 'mean_std', config.get(model_type, 'meanstdVersion')))
        mean_std = np.load(mean_std_data)
        return net_weights_dict, mean_std


def load_model(default_device):
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    net = build_model(args)

    net_weights_dict, mean_std = model_selection()
    net.load_state_dict(net_weights_dict)
    net.eval()

    if torch.cuda.is_available():
        net.cuda(default_device)
        print('{} are using GPU - {}'.format(os.getpid(), default_device))
    return net, mean_std


def cal_pdl1(slide_path: str, x_coords: List[float], y_coords: List[float]):
    result_root = os.path.dirname(slide_path)
    slide_path = '"' + slide_path + '"'
    coord_json_name = 'pdl1_coords_wsi.json'
    label_json_name = 'pdl1_label_wsi.json'
    prob_json_name = 'pdl1_prob_wsi.json'

    delete_prev_json(result_root, coord_json_name, label_json_name, prob_json_name)
    gpu_num = torch.cuda.device_count()
    num_process_per_gpu = 1
    command = ['mpiexec', '-np', str(gpu_num * num_process_per_gpu), sys.executable, '-m',
               'algorithms.PDL1.run_pdl1']
    command.insert(1, '--allow-run-as-root')
    command.append('--slide_path {}'.format(slide_path))
    command.append('--roi_coords')
    command.append(json.dumps(x_coords, separators=(',', ':')))
    command.append(json.dumps(y_coords, separators=(',', ':')))
    command_str = ' '.join(command)

    bat_name = 'run_{}.sh'.format(os.path.splitext(os.path.basename(slide_path))[0])
    bat_name = bat_name.replace(' ', '_')
    with open(os.path.join(current_dir, bat_name), 'w', encoding='utf-8') as f:
        f.write(command_str)
    os.chmod(os.path.join(current_dir, bat_name), stat.S_IRWXU)

    status = subprocess.Popen(
        os.path.join(current_dir, bat_name), cwd=libs_root, shell=True, env=os.environ.copy(), preexec_fn=os.setsid)

    returncode = status.wait()
    if returncode == 0:
        if os.path.exists(os.path.join(current_dir, bat_name)):
            os.remove(os.path.join(current_dir, bat_name))
        with open(os.path.join(result_root, coord_json_name), 'r') as result_file:
            coords_dict = json.load(result_file)
            center_x_coords = np.array(coords_dict["x_coords"])
            center_y_coords = np.array(coords_dict["y_coords"])
            center_coords_np = np.concatenate(
                (np.expand_dims(center_x_coords, axis=-1), np.expand_dims(center_y_coords, axis=-1)), axis=-1)
        with open(os.path.join(result_root, label_json_name), 'r') as result_file:
            cls_labels_np = json.load(result_file)
            cls_labels_np = np.array(cls_labels_np)
        with open(os.path.join(result_root, prob_json_name), 'r') as result_file:
            probs_np = json.load(result_file)
            probs_np = np.array(probs_np)
    else:
        raise ValueError('run subprocess failed')
    return center_coords_np, cls_labels_np, probs_np


def compute_process(slide_path: str, x_coords: List[float], y_coords: List[float], patch_size=1024):
    comm = MPI.COMM_WORLD
    comm_rank, comm_size = comm.Get_rank(), comm.Get_size()
    gpu_num = torch.cuda.device_count()
    int_device = int(comm_rank % gpu_num)
    net, mean_std = load_model(int_device)
    mean, std = mean_std

    slide = open_slide(slide_path)
    slide_mpp = slide.mpp
    standard_mpp = 0.25
    if slide_mpp is not None:
        resize_ratio = slide_mpp / standard_mpp
    else:
        resize_ratio = 1

    patch_suffix = ['png', 'jpg', 'jpeg', 'bmp']
    if comm_rank == 0:
        suffix = slide_path.split('.')[-1]
        if len(x_coords) > 0:
            contours = [np.expand_dims(np.stack([x_coords, y_coords], axis=1), axis=1)]
            region_info_dict = split_patches(slide, patch_size=int(patch_size / resize_ratio), contours=contours)
        elif suffix in patch_suffix:
            contours = []
            region_info_dict = split_patches(slide, patch_size=int(patch_size / resize_ratio), contours=contours)
        else:
            contours, threshold_map = find_tissue_countours(slide)
            if not contours:
                region_info_dict = split_patches(slide, patch_size=int(patch_size / resize_ratio), contours=contours)
            else:
                region_info_dict = split_patches_map(slide, patch_size=int(patch_size / resize_ratio),
                                                     threshold_map=threshold_map)
        region_info_dict = split2groups(region_info_dict=region_info_dict, comm_size=comm_size)
    else:
        region_info_dict = None
    local_data_dict = comm.scatter(region_info_dict, root=0)

    center_coords_list = []
    test_labels_list = []
    test_prob_list = []

    for key, value in local_data_dict.items():
        try:
            region_start = value
            cur_region = slide.read(([region_start[0], region_start[1]]),
                                    (int(patch_size / resize_ratio), int(patch_size / resize_ratio)), 1)
            cur_shape = cur_region.shape
            cur_region = cv2.resize(cur_region, (int(cur_shape[1] * resize_ratio), int(cur_shape[0] * resize_ratio)))
            test_center_coords, test_labels, test_probs = cal_pdl1_np(cur_region.astype(np.uint8), mean, std, net,
                                                                      int_device)

            if len(test_center_coords) > 0:
                test_center_coords = test_center_coords / resize_ratio + np.asarray([region_start[0], region_start[1]])
                test_center_coords = test_center_coords.astype(int)
                test_center_coords = list(test_center_coords)

                test_labels = list(test_labels)
                test_probs = list(test_probs)

                center_coords_list.extend(test_center_coords)
                test_labels_list.extend(test_labels)
                test_prob_list.extend(test_probs)
        except Exception as e:
            print(e, value)

    combine_test_coords = comm.gather(center_coords_list, root=0)
    combine_test_labels = comm.gather(test_labels_list, root=0)
    combine_test_probs = comm.gather(test_prob_list, root=0)
    # combine_test_labels = refine_tps(combine_test_labels, combine_test_prob, comm_size)
    if combine_test_coords:
        combine_test_coords = list(chain(*combine_test_coords))
        combine_test_labels = list(chain(*combine_test_labels))
        combine_test_probs = list(chain(*combine_test_probs))
    else:
        combine_test_coords = []
        combine_test_labels = []
        combine_test_probs = []

    return combine_test_coords, combine_test_labels, combine_test_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--slide_path', type=str, default="/data1/Caijt/PDL1_Parallel/A080 PD-L1 V+.kfb",
                        help='Slide Path')
    parser.add_argument('--roi_coords', type=str, nargs=2, default=None)

    # * Model
    parser.add_argument('--num_classes', type=int, default=10, help="Number of cell categories")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
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
    parser.add_argument('--space', default=16, type=int)
    args = parser.parse_args()
    slide_path = args.slide_path
    # mean, std = np.load('mean_std.npy')
    # slide_path = r'D:\迈杰PDL1已分析\30301-F3749-IHC.kfb'

    roi_coords = args.roi_coords
    if roi_coords is not None:
        x_coords, y_coords = json.loads(roi_coords[0]), json.loads(roi_coords[1])
    else:
        x_coords, y_coords = None, None

    # slide_path = r'D:\data\company1\data\2022_08_04_13_04_12_753492\slices\3477816\2022-07-27-185633.svs'
    # x_coords, y_coords = [], []
    center_coords_ls, label_ls, probs_ls = compute_process(slide_path, x_coords=x_coords, y_coords=y_coords)
    center_coords_all, labels_all, prob_all = map_results(center_coords_ls, label_ls, probs_ls)
    center_coords_all, labels_all, prob_all = roi_filter(center_coords_all, labels_all, prob_all, x_coords=x_coords,
                                                         y_coords=y_coords)
    dump_results(slide_path, center_coords_all, labels_all, prob_all)
