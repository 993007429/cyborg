import logging
import os
import sys
import stat
from typing import List, Tuple

import mpi4py.MPI as MPI
import torch
import numpy as np
import subprocess
import json
from itertools import chain
import argparse
import cv2

from cyborg.libs.slide.dispatch import open_slide
from cyborg.modules.ai.libs.algorithms.NP.models.waternet.detr import build_model
from cyborg.modules.ai.libs.algorithms.NP.src.utils import delete_prev_json, split_patches, split2groups, \
    filter_points, filter_contours
from cyborg.modules.ai.libs.algorithms.NP.src.seg_tissue_area import find_tissue_countours
from cyborg.modules.ai.libs.algorithms.NP.src.multi_cls_cell_det import cal_region_deeplab, cal_bxr_np
from cyborg.modules.ai.libs.algorithms.NP.models.deeplab.deeplab import DeepLab

libs_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

logger = logging.getLogger(__name__)


def load_model(default_device, slide_mpp):
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model')

    # 加载模型
    c_net_weights_path = os.path.join(model_dir, 'bxr_pure_mpp=0.25.pth')  # label_map
    c_net = build_model(args)
    c_net_weights_dict = torch.load(c_net_weights_path, map_location=lambda storage, loc: storage)
    c_net.load_state_dict(c_net_weights_dict)
    c_net.eval()

    if slide_mpp == 0.250001:
        r_net_weights_path = os.path.join(model_dir, 'bxr_deeplab_mpp=0.25.pth')
    else:
        r_net_weights_path = os.path.join(model_dir, 'bxr_deeplab_mpp=0.5.pth')
    r_net = DeepLab(num_classes=4,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=True,
                    freeze_bn=False)
    r_net_weights_dict = torch.load(r_net_weights_path, map_location=lambda storage, loc: storage)
    r_net.load_state_dict(r_net_weights_dict)
    r_net.eval()

    if torch.cuda.is_available():
        c_net.cuda(default_device)
        r_net.cuda(default_device)
        logger.info('{} are using CUDA - {}'.format(os.getpid(), default_device))
    elif torch.has_mps:
        c_net.to('mps')
        r_net.to('mps')
        logger.info('{} are using Metal on Mac OS - {}'.format(os.getpid(), default_device))

    return c_net, r_net


def cal_np(slide_path: str, x_coords: List[float], y_coords: List[float]):
    result_root = os.path.dirname(slide_path)
    slide_path = '"' + slide_path + '"'
    result_file = 'np_result.json'

    delete_prev_json(result_root, result_file)
    gpu_num = torch.cuda.device_count() or 1
    num_process_per_gpu = 1
    command = ['mpiexec', '-np', str(gpu_num * num_process_per_gpu), 'python', '-m', 'algorithms.NP.run_np']
    if sys.platform == 'linux':
        command[3] = 'python3'
        command.insert(1, '--allow-run-as-root')
    command.append('--slide_path {}'.format(slide_path))
    command.append('--roi_coords')
    command.append(json.dumps(x_coords, separators=(',', ':')))
    command.append(json.dumps(y_coords, separators=(',', ':')))
    command_str = ' '.join(command)
    if sys.platform == 'win32':
        bat_name = 'run_{}.bat'.format(os.path.splitext(os.path.basename(slide_path))[0])
        with open(os.path.join(current_dir, bat_name), 'w', encoding='gbk') as f:
            f.write(command_str)
    else:
        bat_name = 'run_{}.sh'.format(os.path.splitext(os.path.basename(slide_path))[0])
        bat_name = bat_name.replace(' ', '_')
        with open(os.path.join(current_dir, bat_name), 'w', encoding='utf-8') as f:
            f.write(command_str)
        os.chmod(os.path.join(current_dir, bat_name), stat.S_IRWXU)

    if sys.platform == 'win32':
        status = subprocess.Popen(os.path.join(current_dir, bat_name), cwd=libs_root, shell=True, env=os.environ.copy())
    else:
        status = subprocess.Popen(os.path.join(current_dir, bat_name), cwd=libs_root, shell=True, env=os.environ.copy(),
                                  preexec_fn=os.setsid)

    status.communicate()
    returncode = status.returncode

    if returncode == 0:
        if os.path.exists(os.path.join(current_dir, bat_name)):
            os.remove(os.path.join(current_dir, bat_name))

        with open(os.path.join(result_root, result_file), 'r') as res_f:
            result = json.load(res_f)

            center_x_coords = np.array(result['centers']['x'])
            center_y_coords = np.array(result['centers']['y'])
            cls_labels_np = np.array(result['cls_labels'])
            region_contour_ls = result['region_contours']
            region_label_ls = result['region_labels']
            total_area = result['total_area']

            center_coords_np = np.concatenate(
                (np.expand_dims(center_x_coords, axis=-1), np.expand_dims(center_y_coords, axis=-1)), axis=-1)

    else:
        raise ValueError('run subprocess failed')

    return center_coords_np, cls_labels_np, region_contour_ls, region_label_ls, total_area


def compute_process(
        slide_path: str, x_coords: List[float], y_coords: List[float], patch_size: Tuple[int, int] = (1920, 1080)):
    comm = MPI.COMM_WORLD
    comm_rank, comm_size = comm.Get_rank(), comm.Get_size()
    gpu_num = torch.cuda.device_count() or 1
    int_device = int(comm_rank % gpu_num)

    slide = open_slide(slide_path)
    slide_mpp = slide.mpp
    standard_cell_mpp = 0.25
    standard_region_mpp = 0.5
    if slide_mpp is not None:
        cell_resize_ratio = slide_mpp / standard_cell_mpp
        region_resize_ratio = slide_mpp / standard_region_mpp
    else:
        slide_mpp = 0.250001
        cell_resize_ratio = 1
        region_resize_ratio = 1
    c_net, r_net = load_model(int_device, slide_mpp)
    patch_size_ori = patch_size
    cell_patch_size = (int(patch_size[0] / cell_resize_ratio), int(patch_size[1] / cell_resize_ratio))
    region_patch_size = (int(patch_size[0] / region_resize_ratio), int(patch_size[1] / region_resize_ratio))

    contours = []
    if comm_rank == 0:
        if len(x_coords) > 0:
            contours = [np.expand_dims(np.stack([x_coords, y_coords], axis=1), axis=1)]
        else:
            contours = find_tissue_countours(slide)
        c_region_info_dict = split_patches(slide, patch_w=cell_patch_size[0], patch_h=cell_patch_size[1],
                                           contours=contours)
        c_region_info_dict = split2groups(region_info_dict=c_region_info_dict, comm_size=comm_size)
        r_region_info_dict = split_patches(slide, patch_w=region_patch_size[0], patch_h=region_patch_size[1],
                                           contours=contours)
        r_region_info_dict = split2groups(region_info_dict=r_region_info_dict, comm_size=comm_size)

    else:
        c_region_info_dict = None
        r_region_info_dict = None

    c_local_data_dict = comm.scatter(c_region_info_dict, root=0)
    r_local_data_dict = comm.scatter(r_region_info_dict, root=0)
    mean, std = np.load(os.path.join(current_dir, 'mean_std.npy'))

    center_coords_list = []
    test_labels_list = []
    region_contours_list = []
    region_labels_list = []

    for key, value in c_local_data_dict.items():
        try:
            region_start = value
            cur_region = slide.read(([region_start[0], region_start[1]]), (cell_patch_size[0], cell_patch_size[1]),
                                    1 / cell_resize_ratio)
            cur_region = cur_region.astype(np.uint8)
            cur_shape = (cur_region.shape[1], cur_region.shape[0])
            if (cur_shape[0] != patch_size_ori[0] or cur_shape[1] != patch_size_ori[1]) \
                    and abs(cur_shape[0] - patch_size_ori[0]) < 2 and abs(cur_shape[1] - patch_size_ori[1]) < 2:
                cur_region = cv2.resize(cur_region, dsize=patch_size_ori, interpolation=cv2.INTER_CUBIC)
            test_center_coords, test_labels = cal_bxr_np(cur_region, c_net, mean, std, int_device)
            if len(test_center_coords) > 0:
                test_center_coords = test_center_coords / cell_resize_ratio + np.asarray(
                    [region_start[0], region_start[1]])
                test_center_coords = test_center_coords.astype(int)
                test_center_coords = list(test_center_coords)
                test_labels = list(test_labels)

                # label_map [浆细胞0, 淋巴细胞1, 嗜酸性粒细胞2, 中性粒细胞3] -> {'淋巴细胞': 3, '浆细胞': 2, '中性粒细胞': 0, '嗜酸性粒细胞': 1}
                label_map = list(np.array([2, 3, 1, 0], dtype=np.int32))
                test_labels = [label_map[x] for x in test_labels]

                center_coords_list.extend(test_center_coords)
                test_labels_list.extend(test_labels)
        except Exception as e:
            logger.exception(e)

    for key, value in r_local_data_dict.items():
        try:
            region_start = value
            cur_region = slide.read(([region_start[0], region_start[1]]), (region_patch_size[0], region_patch_size[1]),
                                    1 / region_resize_ratio)
            cur_region = cur_region.astype(np.uint8)
            cur_shape = (cur_region.shape[1], cur_region.shape[0])
            if (cur_shape[0] != patch_size_ori[0] or cur_shape[1] != patch_size_ori[1]) \
                    and abs(cur_shape[0] - patch_size_ori[0]) < 2 and abs(cur_shape[1] - patch_size_ori[1]) < 2:
                cur_region = cv2.resize(cur_region, dsize=patch_size_ori, interpolation=cv2.INTER_CUBIC)
            result_contours, result_contours_labels = cal_region_deeplab(cur_region, r_net, int_device, patch_size=512)
            if len(result_contours) > 0:
                for contour_idx in range(len(result_contours)):
                    for cell_idx in range(len(result_contours[contour_idx])):
                        result_contours[contour_idx][cell_idx][0] = \
                            result_contours[contour_idx][cell_idx][0] / region_resize_ratio + region_start[0]
                        result_contours[contour_idx][cell_idx][1] = \
                            result_contours[contour_idx][cell_idx][1] / region_resize_ratio + region_start[1]
                region_contours_list.extend(result_contours)
                region_labels_list.extend(result_contours_labels)
        except Exception as e:
            logger.exception(e)

    combine_test_coords = comm.gather(center_coords_list, root=0)
    combine_test_labels = comm.gather(test_labels_list, root=0)
    combine_region_contours = comm.gather(region_contours_list, root=0)
    combine_region_labels = comm.gather(region_labels_list, root=0)
    comm.Barrier()

    if comm_rank == 0:

        if combine_test_coords:
            combine_test_coords = list(chain(*combine_test_coords))
            combine_test_labels = list(chain(*combine_test_labels))
        else:
            combine_test_coords, combine_test_labels = [], []
        if combine_region_contours:
            combine_region_contours = list(chain(*combine_region_contours))
            combine_region_labels = list(chain(*combine_region_labels))
        else:
            combine_region_contours, combine_region_labels = [], []

        merge_region_contours = []
        merge_region_labels = []

        # 合并区域
        roi_list = contours[0]
        x_min_parent_region, x_max_parent_region = roi_list[:, :, 0].min(), roi_list[:, :, 0].max()
        y_min_parent_region, y_max_parent_region = roi_list[:, :, 1].min(), roi_list[:, :, 1].max()
        for roi_list in contours:
            x_min_parent_region = min(roi_list[:, :, 0].min(), x_min_parent_region)
            x_max_parent_region = max(roi_list[:, :, 0].max(), x_max_parent_region)
            y_min_parent_region = min(roi_list[:, :, 1].min(), y_min_parent_region)
            y_max_parent_region = max(roi_list[:, :, 1].max(), y_max_parent_region)

        x_min_parent_region = max(0, x_min_parent_region - 100)
        x_max_parent_region = min(slide.width, x_max_parent_region + 100)
        y_min_parent_region = max(0, y_min_parent_region - 100)
        y_max_parent_region = min(slide.height, y_max_parent_region + 100)

        w_parent_region = x_max_parent_region - x_min_parent_region
        h_parent_region = y_max_parent_region - y_min_parent_region

        area = int(w_parent_region) * int(h_parent_region)
        if area < 0:
            scale = 16
        elif area < 10 ** 9 * 4:  # 4GB
            scale = 1
        elif area < 10 ** 9 * 16:  # 16GB
            scale = 2
        elif area < 10 ** 9 * 64:  # 64GB
            scale = 4
        elif area < 10 ** 9 * 128:  # 128GB
            scale = 8
        else:
            scale = 16
        # 防止区域重叠，前面的区域抑制后面的区域
        last_contours = []
        # [1, 3, 2] {'上皮区域':1, '血管区域':2, '腺体区域':3, }
        for i in [1, 3, 2]:
            target_region_contours = [combine_region_contours[index] for index, x in
                                      enumerate(combine_region_labels) if x == i]
            target_region_contours = [np.array(x)[:, np.newaxis, :] for x in target_region_contours]
            # 将WSI坐标映射为局部mask坐标
            for j in range(len(target_region_contours)):
                target_region_contours[j][:, :, 0] = target_region_contours[j][:, :, 0] - x_min_parent_region
                target_region_contours[j][:, :, 1] = target_region_contours[j][:, :, 1] - y_min_parent_region
            target_region_contours = [(x / scale).astype(np.int32) for x in target_region_contours]

            mask = np.zeros((int(h_parent_region / scale), int(w_parent_region / scale)), dtype=np.uint8)
            mask = cv2.drawContours(mask, target_region_contours, -1, (255, 0, 0), -1)
            # 前面的区域抑制后面的区域
            mask = cv2.drawContours(mask, last_contours, -1, (0, 0, 0), -1)
            # kernel[i] {'上皮区域':1, '血管区域':2, '腺体区域':3, }
            kernel = [0, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))]
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel[i], iterations=7)
            # 前面的区域抑制后面的区域
            mask = cv2.drawContours(mask, last_contours, -1, (0, 0, 0), -1)
            new_contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

            # 过滤面积小于120微米^2的区域
            new_contours = [roi_list for roi_list in new_contours if
                            cv2.contourArea(roi_list) * scale ** 2 * slide_mpp ** 2 >= 120]
            last_contours += new_contours[:]

            # 将过大区域减少插值
            large_contours_threshold = 3000
            new_contours_temp = []
            for i_contours in new_contours:
                contours_len = int(i_contours.shape[0])
                if contours_len > large_contours_threshold * 2:
                    interval = int(contours_len / large_contours_threshold)
                    interval = interval * 3 - 3  # 3,6,9,12,15,18,21,24,27,30。越大的区域插值的点越少
                    mask = np.zeros((int(h_parent_region / scale / interval), int(w_parent_region / scale / interval)),
                                    dtype=np.uint8)
                    i_contours = (i_contours / interval).astype(np.int32)
                    mask = cv2.drawContours(mask, [i_contours], -1, (255, 0, 0), -1)
                    i_contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                    i_contours = (i_contours[0] * interval).astype(np.int32)
                new_contours_temp.append(i_contours)
            new_contours = new_contours_temp

            mask = np.zeros((int(h_parent_region / scale), int(w_parent_region / scale)), dtype=np.uint8)
            mask = cv2.drawContours(mask, new_contours, -1, (255, 0, 0), -1)

            # 用区域抑制细胞
            filter_test_coords_boolean = [True] * len(combine_test_coords)

            for index, v in enumerate(combine_test_coords):
                y_test_coords, x_test_coords = int((v[1] - y_min_parent_region) / scale), int(
                    (v[0] - x_min_parent_region) / scale)
                if (0 <= x_test_coords < mask.shape[1] and
                        0 <= y_test_coords < mask.shape[0] and mask[y_test_coords, x_test_coords] == 255):
                    filter_test_coords_boolean[index] = False
            combine_test_coords = [combine_test_coords[idx] for idx, v in enumerate(filter_test_coords_boolean) if v]
            combine_test_labels = [combine_test_labels[idx] for idx, v in enumerate(filter_test_coords_boolean) if v]

            # 将局部mask坐标映射回WSI坐标
            new_contours = [x * scale for x in new_contours]
            for j in range(len(new_contours)):
                new_contours[j][:, :, 0] = new_contours[j][:, :, 0] + x_min_parent_region
                new_contours[j][:, :, 1] = new_contours[j][:, :, 1] + y_min_parent_region

            new_contours = [x[:, 0, :].tolist() for x in new_contours]
            merge_region_contours += new_contours
            merge_region_labels += [i] * len(new_contours)

        combine_test_coords = [x.astype(np.int64) for x in combine_test_coords]
        center_coords_all, labels_all = filter_points(np.array(combine_test_coords), np.array(combine_test_labels),
                                                      x_coords=x_coords,
                                                      y_coords=y_coords)
        region_coords_all, region_labels_all = filter_contours(merge_region_contours, merge_region_labels,
                                                               x_coords=x_coords,
                                                               y_coords=y_coords)

        total_area = 0
        for contour in contours:
            total_area += cv2.contourArea(np.array(contour).astype(np.int32))

        if len(center_coords_all) > 0:
            centers = {'x': list(map(int, center_coords_all[:, 0])), 'y': list(map(int, center_coords_all[:, 1]))}
        else:
            centers = {'x': [], 'y': []}
        result = {
            'centers': centers,
            'cls_labels': list(map(int, labels_all)),
            'region_contours': region_coords_all,
            'region_labels': region_labels_all,
            'total_area': total_area
        }

        result_root = os.path.dirname(slide_path)
        os.makedirs(result_root, exist_ok=True)
        with open(os.path.join(str(result_root), 'np_result.json'), 'w', encoding="utf-8") as result_file:
            json.dump(result, result_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--slide_path', type=str, default="/data1/Caijt/PDL1_Parallel/A080 PD-L1 V+.kfb",
                        help='Slide Path')
    parser.add_argument('--roi_coords', type=str, nargs=2, default=None)
    # * Model
    parser.add_argument('--num_classes', type=int, default=4,
                        help="Number of cell categories")
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

    args = parser.parse_args()
    slide_path = args.slide_path

    roi_coords = args.roi_coords
    if roi_coords is not None:
        x_coords, y_coords = json.loads(roi_coords[0]), json.loads(roi_coords[1])
    else:
        x_coords, y_coords = [], []

    compute_process(slide_path, x_coords=x_coords, y_coords=y_coords)
