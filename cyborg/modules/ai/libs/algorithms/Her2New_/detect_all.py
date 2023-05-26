import logging
import os
import sys
import stat

import subprocess
import json
import argparse
import time
import math
from typing import Any

import cv2
from PIL import Image
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from cyborg.infra.oss import oss
from cyborg.libs.heimdall.dispatch import open_slide
from cyborg.modules.ai.libs.algorithms.Her2New_.cell_utils import analysis_wsi
from cyborg.modules.ai.libs.algorithms.Her2New_.dataset import Dataset
from cyborg.modules.ai.libs.algorithms.Her2New_.models.experimental import attempt_load
from cyborg.modules.ai.libs.algorithms.Her2New_.utils.general import non_max_suppression
from cyborg.modules.ai.libs.algorithms.Her2New_.utils.torch_utils import select_device
from cyborg.seedwork.domain.value_objects import BaseValueObject

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
current_root = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)

segmentation_cls = 4


def get_patch(slide_w, slide_h, croplen, stride, lvl, roi_x, roi_y):
    if len(roi_x) > 0:
        xmin, xmax, ymin, ymax = min(roi_x), max(roi_x), min(roi_y), max(roi_y)
    else:
        xmin = 0
        ymin = 0
        xmax = np.inf
        ymax = np.inf
    stride = int(stride * lvl)
    croplen = int(croplen * lvl)
    h, w = slide_h, slide_w
    num_y = math.ceil(h // stride)
    num_x = math.ceil(w // stride)
    crop_x_coords = []
    crop_y_coords = []
    crop_coords = []
    for i in range(num_y):
        st_y = int(i * stride)
        ed_y = st_y + croplen
        if ymin < st_y < ymax:
            for j in range(num_x):
                st_x = int(j * stride)
                ed_x = st_x + croplen
                if xmin < st_x < xmax:
                    if ed_x > w:
                        ed_x = w
                    if ed_y > h:
                        ed_y = h
                    crop_x_coords.append(st_x)
                    crop_y_coords.append(st_y)
                    crop_coords.append([st_x, st_y, ed_x, ed_y])

    return crop_coords


def load_model(model, weight_path='R_net_seg.pth.tar'):
    checkpoint = torch.load(weight_path)
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    # 获取保存的模型的权重
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()  # 获取当前模型的权重
    for k in state_dict:
        if k in model_state_dict:
            # 加载节点权重尺寸与当前模型节点权重尺寸是否一致，若不一致，则以当前模型权重为准。
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.info('Skip loading parameter {}, required shape{}, loaded shape{}. '.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:  # 若当前网络缺少该算子，则抛弃需要加载的多余算子权重
            logger.info('Drop parameter {}.'.format(k))
    # 加载的模型权重缺少相应的算子权重，则不变。
    for k in model_state_dict:
        if not (k in state_dict):
            logger.info('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]

    # 模型加载权重信息
    model.load_state_dict(state_dict, strict=False)
    return model


def get_level_dim_dict(slide_path):
    level_dim_dict = {}
    ext = os.path.splitext(slide_path)[1][1:].lower()
    if ext == 'kfb' or ext == 'ndpi':
        slide = open_slide(slide_path)
    else:
        slide = open_slide(slide_path)
    h = slide.height
    w = slide.width
    maxlvl = slide.maxlvl
    downsamples = [pow(2, i) for i in range(maxlvl)]
    dims = [[h / i, w / i] for i in downsamples]
    # dims = slide.level_dimensions
    # downsamples = slide.level_downsamples
    for i in range(len(downsamples)):
        level_dim_dict[i] = (dims[i], downsamples[i])
    return level_dim_dict, ext


def get_slice(slice_path):
    # slice list , 3 lvl slide
    slide_original = open_slide(slice_path)
    lvl0 = 1
    h0, w0, maxlvl = slide_original.height, slide_original.width, slide_original.maxlvl
    d, ext = get_level_dim_dict(slice_path)

    idx = 3  # i-1 if(i-1)>0 else 0
    hw, lvl1 = d[idx]
    h1, w1 = hw
    idx = 4  # i
    hw, lvl2 = d[idx]
    h2, w2 = hw
    idx = 5  # idx+1 if idx+1<(maxlvl-1) else maxlvl-1
    hw, lvl3 = d[idx]
    h3, w3 = hw

    # original slice ,
    # lvl_list from zero to high,
    # size list corresponding to the lvl list
    # type of extension
    return [slide_original, [lvl0, lvl1, lvl2, lvl3], [[h0, w0], [h1, w1], [h2, w2], [h3, w3]], ext]


def init_device(local_rank, world_size):
    # DDP设置
    torch.backends.cudnn.benchmark = True
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(backend="gloo", rank=local_rank, world_size=world_size)  # nccl & gloo
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device


def check_merge_bbox(bbox1, bbox2, thre):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    flg = False

    if abs(x1 - x2_) < thre and abs(y1 - y1_) < thre:
        flg = True
    if abs(x1 - x1_) < thre and abs(y1 - y1_) < thre:
        flg = True
    if abs(x2 - x1_) < thre and abs(y1 - y1_) < thre:
        flg = True
    if abs(x1 - x1_) < thre and abs(y2 - y2_) < thre:
        flg = True
    if abs(x1 - x1_) < thre and abs(y2 - y1_) < thre:
        flg = True
    if abs(x1 - x1_) < thre and abs(y1 - y2_) < thre:
        flg = True
    if abs(x2 - x2_) < thre and abs(y1 - y1_) < thre:
        flg = True
    if abs(x2 - x2_) < thre and abs(y2 - y2_) < thre:
        flg = True

    return flg, min(x1, x1_), min(y1, y1_), max(x2, x2_), max(y2, y2_)


def generate_seg(img, r_seg_net):
    dsize = (256, 256)
    img = img[..., ::-1].copy()  # to rgb
    h, w, c = img.shape
    img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(np.uint8(img))

    transform = transforms.ToTensor()
    img = transform(img)
    norm_ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img = norm_(img)
    img = img.unsqueeze(0)
    mask = np.ones((dsize[0], dsize[1], 3)) * 255
    with torch.no_grad():
        img = img.cuda()
        output = r_seg_net(img)
        output = torch.argmax(output, dim=1)
        output = output.squeeze(0)
        output = output.data.cpu().numpy()
        counter = []
        for i in range(segmentation_cls - 1):
            counter.append(np.count_nonzero(output == i))

        mask[output != segmentation_cls - 1] = np.array((0, 0, 0))

        mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        mask = np.uint8(mask)
        return mask


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, st_x=0, st_y=0, ratio_x=1, ratio_y=1):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape, st_x, st_y, ratio_x, ratio_y)
    return coords


def clip_coords(boxes, img_shape, st_x, st_y, ratio_x, ratio_y):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 0] = (boxes[:, 0] + st_x) / ratio_x
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 1] = (boxes[:, 1] + st_y) / ratio_y
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 2] = (boxes[:, 2] + st_x) / ratio_x
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
    boxes[:, 3] = (boxes[:, 3] + st_y) / ratio_y


@torch.no_grad()
def region_process(slice_path, opt, name, roi_x, roi_y):
    slide, lvl_list, size_list, ext = get_slice(slice_path)
    ####################
    # get models
    ####################

    # models
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    weights = opt.weights
    weights = oss.path_join('AI', 'Her2New_', weights)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16

    ####################
    # get datasets
    ####################
    # wsi lvl
    h0, w0 = size_list[0]
    # lvl1
    h1, w1 = size_list[1]
    # lvl2 / visual lvl
    h2, w2 = size_list[2]

    # each lvl to visual lvl
    ratio_x = w1 / w0
    ratio_y = h1 / h0
    ratio_x_ = w2 / w0
    ratio_y_ = h2 / h0

    # vis slide
    try:
        slide_draw = slide.read(scale=16)  # gbr
    except Exception as e:
        logger.exception(e)
        slide_draw = np.zeros((int(w0 / 16), int(h0 / 16), 3), dtype=np.uint8)

    logger.info(f'Slide_draw.shape:{slide_draw.shape}')
    slide_draw = slide_draw.astype(np.uint8)

    crop_coords = get_patch(slide_w=w0, slide_h=h0, croplen=2560, stride=2560 / 4, lvl=lvl_list[1],
                            roi_x=roi_x, roi_y=roi_y)
    crop_coords_ = get_patch(slide_w=w0, slide_h=h0, croplen=2560, stride=2560 / 4, lvl=lvl_list[2],
                             roi_x=roi_x, roi_y=roi_y)

    patch_dataset = Dataset(slide, crop_coords, wsi_type=ext, crop_len=2560, lvl=lvl_list[1])
    patch_dataset_ = Dataset(slide, crop_coords_, wsi_type=ext, crop_len=2560, lvl=lvl_list[2])
    patch_loader = torch.utils.data.DataLoader(dataset=patch_dataset, batch_size=1, shuffle=False, num_workers=0)
    patch_loader_ = torch.utils.data.DataLoader(dataset=patch_dataset_, batch_size=1, shuffle=False, num_workers=0)

    patch_loaders = [patch_loader, patch_loader_]
    ratio_xs = [ratio_x, ratio_x_]
    ratio_ys = [ratio_y, ratio_y_]

    t0 = time.time()
    logger.info('Start region detection and segmentation')
    final_preds = None
    for idx, patch_loader in enumerate(patch_loaders):
        ratio_x = ratio_xs[idx]
        ratio_y = ratio_ys[idx]
        for iter_id, batch in enumerate(patch_loader):
            st_x, st_y = batch['st_xy'][0].to(device), batch['st_xy'][1].to(device)
            imgs = batch['cur_region'][0]
            imgs = imgs.half() if half else imgs.float()  # uint8 to fp16/32
            imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
            if imgs.ndimension() == 3:
                imgs = imgs.unsqueeze(0)
            imgs = imgs.to(device)

            # Inference
            with torch.no_grad():
                preds = model(imgs, augment=opt.augment)[0]

            pred = non_max_suppression(preds, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)

            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], imgs.shape[2:],
                                              st_x=st_x, st_y=st_y,
                                              ratio_x=ratio_x, ratio_y=ratio_y).round()
                    if final_preds is None:
                        final_preds = det
                    else:
                        final_preds = torch.cat((final_preds, det), 0)

    mask = np.ones(slide_draw.shape, dtype=np.uint8) * 255
    # padding
    padding = 15
    if final_preds is not None:
        print(len(final_preds))
        hh, ww, cc = slide_draw.shape
        final_preds = torch.unsqueeze(final_preds, 0).cpu().numpy()[0]
        for *xyxy, conf, cls in reversed(final_preds):
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            x1_ = int(x1 / 16 - padding)
            x2_ = int(x2 / 16 + padding)
            y1_ = int(y1 / 16 - padding)
            y2_ = int(y2 / 16 + padding)
            w = x2_ - x1_ if (x2_ < ww) else ww - x1_
            h = y2_ - y1_ if (y2_ < hh) else hh - y1_
            cur_mask = np.zeros((h, w, 3), dtype=np.uint8)
            mask[y1_:y1_ + h, x1_:x1_ + w] = cv2.bitwise_and(mask[y1_:y1_ + h, x1_:x1_ + w], cur_mask)
            slide_draw[y1_:y1_ + h, x1_:x1_ + w] = cv2.bitwise_and(slide_draw[y1_:y1_ + h, x1_:x1_ + w],
                                                                   cur_mask)
    mask_save_path3 = ''
    if opt.save_mask:
        name = name.replace(" ", '')
        mask_save_path3 = current_root + '/data/mask_result/' + name + '_mask.png'

        cv2.imwrite(mask_save_path3, mask)

    logger.info(f'End region. total:{time.time() - t0:.3f}s ')

    return mask, mask_save_path3


def cal_cell(o_slide_path, roi_list, opt):
    cls_labels_np_with_id = {}
    center_coords_np_with_id = {}
    for idx, roi in enumerate(roi_list):
        roiid, x_coords, y_coords = roi['id'], roi['x'], roi['y']
        name = os.path.splitext(os.path.basename(o_slide_path))[0]
        mask, mask_path = region_process(o_slide_path, opt, name, x_coords, y_coords)
        result_root = os.path.dirname(o_slide_path)
        slide_path = '"' + o_slide_path + '"'
        coord_json_name = 'her2_coords_wsi.json'
        label_json_name = 'her2_label_wsi.json'
        vis = 'none'
        pid = os.getpid()

        delete_prev_json(result_root, coord_json_name, label_json_name)
        delete_prev_json(result_root, coord_json_name, label_json_name)
        gpu_num = torch.cuda.device_count()

        num_process_per_gpu = 1
        command = ['mpiexec', '-np', str(gpu_num * num_process_per_gpu),
                   sys.executable,
                   '-m', 'cell_detection']
        command.insert(1, '--allow-run-as-root')
        command.append('--ppid {}'.format(pid))
        command.append('--slide {}'.format(slide_path))
        command.append('--mask {}'.format(mask_path))
        command.append('--roi')
        command.append(json.dumps(x_coords, separators=(',', ':')))
        command.append(json.dumps(y_coords, separators=(',', ':')))
        command.append('--vis {}'.format(vis))
        command_str = ' '.join(command)

        bat_name = '{}.sh'.format(
            os.path.splitext(
                os.path.basename(slide_path))[0].strip().replace(
                " ", '').replace("(", '').replace(")", '').replace("+", ''))
        with open(os.path.join(current_root, bat_name), 'w', encoding='utf-8') as f:
            f.write(command_str)
        os.chmod(os.path.join(current_root, bat_name), stat.S_IRWXU)
        status = subprocess.Popen(os.path.join(current_root, bat_name), cwd=current_root, shell=True,
                                  env=os.environ.copy())
        returncode = status.wait()

        if returncode == 0:
            if os.path.exists(os.path.join(current_root, bat_name)):
                os.remove(os.path.join(current_root, bat_name))

            with open(os.path.join(result_root, coord_json_name), 'r') as result_file:
                coords_dict = json.load(result_file)
                center_x_coords = np.array(coords_dict["x"])
                center_y_coords = np.array(coords_dict["y"])
                center_coords_np = np.concatenate(
                    (np.expand_dims(center_x_coords, axis=-1), np.expand_dims(center_y_coords, axis=-1)), axis=-1)

            with open(os.path.join(result_root, label_json_name), 'r') as result_file:
                cls_labels_np = json.load(result_file)
                cls_labels_np = np.array(cls_labels_np["class"])
        else:
            raise ValueError('run subprocess failed')

        center_coords_np_with_id[roiid] = center_coords_np.tolist()
        cls_labels_np_with_id[roiid] = cls_labels_np.tolist()
        if os.path.exists(mask_path):
            os.remove(mask_path)

    return center_coords_np_with_id, cls_labels_np_with_id


def delete_prev_json(result_root, coord_json_name, label_json_name):
    if os.path.exists(os.path.join(result_root, coord_json_name)):
        os.remove(os.path.join(result_root, coord_json_name))
    if os.path.exists(os.path.join(result_root, label_json_name)):
        os.remove(os.path.join(result_root, label_json_name))
    if not os.path.exists(os.path.join(result_root, coord_json_name)):
        open(os.path.join(result_root, coord_json_name), 'x')
    if not os.path.exists(os.path.join(result_root, label_json_name)):
        open(os.path.join(result_root, label_json_name), 'x')


def pointinpolygon(center_coords, labels, x_coords, y_coords):
    validate_center_coords = []
    validate_labels = []
    polygon_list = []
    for idx, coord in enumerate(x_coords):
        point = (coord, y_coords[idx])
        polygon_list.append(point)
    polygon = Polygon(polygon_list)
    for iidx, coord in enumerate(center_coords):
        point = Point(coord[0], coord[1])
        if polygon.contains(point):
            validate_center_coords.append(coord)
            validate_labels.append(labels[iidx])
    return validate_center_coords, validate_labels


def roi_filter(center_coords, labels, x_coords, y_coords):
    if len(y_coords) > 0 and len(x_coords) > 0:
        center_coords, labels = pointinpolygon(center_coords, labels, x_coords, y_coords)

    return center_coords, labels


class Her2Args(BaseValueObject):
    device: str = ''
    weights: str = 'best1215.pt'
    augment: bool = False
    conf_thres: float = 0.35
    iou_thres: float = 0.45
    classes: Any = None
    num_classes: int = 6
    agnostic_nms: bool = False
    save_mask: bool = True


# best
def run_her2_alg(slide_path, roi_list):
    args = Her2Args()
    center_coords_np_with_id, cls_labels_np_with_id = cal_cell(slide_path, roi_list=roi_list, opt=args)
    cls_labels_np_wsi = []
    for k in cls_labels_np_with_id:
        # extend
        cls_labels_np_wsi.extend(cls_labels_np_with_id[k])
    r1, r2, r3 = analysis_wsi(args, cls_labels_np_wsi)
    return center_coords_np_with_id, cls_labels_np_with_id, r2, r1, r3
