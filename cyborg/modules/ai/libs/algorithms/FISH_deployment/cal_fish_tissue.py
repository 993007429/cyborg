import logging
import os
import re
import time

from typing import Union
import cv2
import numpy as np
from mmcv import Config
from shapely import geometry
from argparse import ArgumentParser
from mmdet.apis import inference_detector
from mmdet.apis.inference import init_detector

from cyborg.infra.oss import oss

pattern = re.compile(r"\$\{[a-zA-Z\d_.]*\}")

current_dir = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


def get_match_segment_results(center_coords, labels, contours, type, dist_threshold):
    red_list = []
    green_list = []
    sample_num = center_coords.shape[0]
    poly_context = {'type': 'MULTIPOLYGON',
                    'coordinates': [[contours]]}
    poly = geometry.shape(poly_context)
    for i in range(sample_num):
        each_coord = center_coords[i]
        point = geometry.Point(each_coord[0], each_coord[1])
        if point.within(poly):
            signal_label = labels[i]
            if signal_label == 0:
                red_list.append(each_coord)
            if signal_label == 1:
                green_list.append(each_coord)

    if not red_list or not green_list:
        yellow_list = []
    elif type == 'Amplification' or type == 'Deletion':
        yellow_list = []
    elif type == 'Fracture' or type == 'Fusion':
        red_list, green_list, yellow_list = match_fusion_signals(red_list, green_list, dist_threshold=dist_threshold)
    else:
        raise ValueError('The FISH type is not available')

    red_count = len(red_list)
    green_count = len(green_list)
    yellow_count = len(yellow_list)

    summary_str = ""
    if red_count > 0:
        summary_str = summary_str + '{}R'.format(red_count)
    if green_count > 0:
        summary_str = summary_str + '{}G'.format(green_count)
    if yellow_count > 0:
        summary_str = summary_str + '{}Y'.format(yellow_count)

    return red_list, green_list, yellow_list, summary_str


def iqu(pred_box, gt_box):
    x1, y1, x2, y2 = pred_box[0], pred_box[1], pred_box[2], pred_box[3]
    x1_g, y1_g, x2_g, y2_g = gt_box[0], gt_box[1], gt_box[2], gt_box[3]

    shadow_x1 = max(x1, x1_g)
    shadow_y1 = max(y1, y1_g)
    shadow_x2 = max(min(x2, x2_g), shadow_x1)
    shadow_y2 = max(min(y2, y2_g), shadow_y1)

    bbox_w = shadow_x2 - shadow_x1
    bbox_h = shadow_y2 - shadow_y1

    intersection = bbox_h * bbox_w
    union = (x2 - x1) * (y2 - y1) + (x2_g - x1_g) * (y2_g - y1_g) - intersection
    small_area = min((x2 - x1) * (y2 - y1), (x2_g - x1_g) * (y2_g - y1_g))

    iou = intersection / union
    ots = intersection / small_area

    return iou, ots


def remove_array(from_arr, arr):
    ind = 0
    size = len(from_arr)
    while ind != size and not np.array_equal(from_arr[ind], arr):
        ind += 1
    if ind != size:
        from_arr.pop(ind)


def match_fusion_signals(red_list, green_list, dist_threshold):
    yellow_list = []
    red_list_copy = red_list.copy()
    green_list_copy = green_list.copy()
    for i, red_signal in enumerate(red_list_copy):
        for j, green_signal in enumerate(green_list_copy):
            distance = ((red_signal[0] - green_signal[0]) ** 2 + (red_signal[1] - green_signal[1]) ** 2) ** 0.5
            if distance < dist_threshold:
                yellow_coord = np.array(
                    [int((red_signal[0] + green_signal[0]) * 0.5), int((red_signal[1] + green_signal[1]) * 0.5)])
                yellow_list.append(tuple(yellow_coord))
                try:
                    remove_array(red_list, red_signal)
                    remove_array(green_list, green_signal)
                except Exception:
                    pass

    return red_list, green_list, yellow_list


def get_match_bbox_results(center_coords, labels, bbox, type, dist_threshold):
    red_list = []
    green_list = []
    sample_num = center_coords.shape[0]
    x1, x2, y1, y2 = bbox[0], bbox[2], bbox[1], bbox[3]
    for i in range(sample_num):
        each_coord = center_coords[i]
        x, y = each_coord[0], each_coord[1]
        if x1 < x < x2:
            if y1 < y < y2:
                signal_label = labels[i]
                if signal_label == 0:
                    red_list.append(each_coord)
                if signal_label == 1:
                    green_list.append(each_coord)

    if len(red_list) == 0 or len(green_list) == 0:
        yellow_list = []
    elif type == 'Amplification' or type == 'Deletion':
        yellow_list = []
    elif type == 'Fracture' or type == 'Fusion':
        red_list, green_list, yellow_list = \
            match_fusion_signals(red_list, green_list, dist_threshold=dist_threshold)
    else:
        raise ValueError('The FISH type is not available')

    red_count = len(red_list)
    green_count = len(green_list)
    yellow_count = len(yellow_list)

    summary_str = ""
    if red_count > 0:
        summary_str = summary_str + '{}R'.format(red_count)
    if green_count > 0:
        summary_str = summary_str + '{}G'.format(green_count)
    if yellow_count > 0:
        summary_str = summary_str + '{}Y'.format(yellow_count)

    return red_list, green_list, yellow_list, summary_str


def get_all_pred_points(signal_result):
    all_center_coords = []
    all_center_labels = []
    for roi_signals in signal_result:
        center_coords = roi_signals[:, :2]
        center_labels = roi_signals[:, 2]
        for center_coord, center_label in zip(list(center_coords), list(center_labels)):
            center_label = int(center_label)
            if center_label == 1 or center_label == 2:
                all_center_coords.append(center_coord)
                all_center_labels.append(center_label)

    return all_center_coords, all_center_labels


def generate_bbox_summary(cell_or_tissue, result, score_thr, overlap_thr, ots_thr):
    if cell_or_tissue == 'CELL':
        class_name_dict = {0: "Primordial cell", 1: "Promyelocytic cells", 2: "Late immature cells",
                           3: "Other cell", 4: "Dead cell", 5: "Autofluorescent cells",
                           6: "Non-cellular components", 7: "bubble"}
        num_classes = 8
    else:
        class_name_dict = {0: "Nucleus"}
        num_classes = 1

    bbox_summary_dict = {}
    bbox_id = 0
    if isinstance(result, tuple):
        bbox_pred, segm_result, signal_result = result
        signal_result = signal_result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_pred, segm_result, signal_result = result, None, None

    red_count_all = 0
    green_count_all = 0
    cell_count = 0
    center_coords, center_labels = [], []

    for class_idx in range(num_classes):
        class_name = class_name_dict[class_idx]
        bboxes = bbox_pred[class_idx]
        segments = segm_result[class_idx]
        for i, bbox in enumerate(bboxes):
            bbox_id += 1
            bbox_score = bbox[4]
            segment = segments[i]
            roi_signals = signal_result[i]

            if bbox_score < score_thr:
                continue
            if segment is not None:
                segment = np.uint8(segment * 255)
                ret, thresh = cv2.threshold(segment, 127, 255, 0)
                cv_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = np.array(
                    [np.array(cv_contour).squeeze() for cv_contour in cv_contours[0] if len(cv_contour) > 0])
                if len(contours) > 2:
                    red_list = []
                    green_list = []
                    yellow_list = []
                    center_coords = roi_signals[:, :2]
                    center_labels = roi_signals[:, 2]
                    for center_coord, center_label in zip(list(center_coords), list(center_labels)):
                        center_label = int(center_label)
                        if center_label == 1:
                            red_list.append(center_coord)
                        if center_label == 2:
                            green_list.append(center_coord)
                    summary_str = ""
                    red_count = len(red_list)
                    green_count = len(green_list)
                    if red_count > 0:
                        summary_str = summary_str + '{}R'.format(red_count)
                    if green_count > 0:
                        summary_str = summary_str + '{}G'.format(green_count)

                    area = (max(contours[:, 0]) - min(contours[:, 0])) * (max(contours[:, 1]) - min(contours[:, 1]))
                    if area > 300:
                        bbox_dict = {
                            'bbox_id': bbox_id, 'class_idx': 0, 'class_name': class_name, 'bbox': bbox,
                            'area': area, 'segment': contours, 'show_segment': contours.reshape(-1, 1, 2),
                            'score': bbox_score, 'summary': summary_str,
                            'coords': {'red': red_list, 'green': green_list, 'yellow': yellow_list}
                        }

                        overlap = False
                        for record_idx in bbox_summary_dict.keys():
                            record = bbox_summary_dict[record_idx]
                            record_bbox = record['bbox']
                            iou, ots = iqu(record_bbox, bbox)
                            if iou > overlap_thr:
                                overlap = True
                                record_score = record['score']
                                if bbox_score > record_score:
                                    bbox_summary_dict[record_idx] = bbox_dict
                            if ots > ots_thr:
                                overlap = True
                                record_area = record['area']
                                if area < record_area:
                                    bbox_summary_dict[record_idx] = bbox_dict

                        if not overlap:
                            bbox_summary_dict[bbox_id] = bbox_dict
                            red_count = len(bbox_dict['coords']['red'])
                            green_count = len(bbox_dict['coords']['green'])
                            if red_count > 0 and green_count > 0:
                                red_count_all += red_count
                                green_count_all += green_count
                                cell_count += 1

    return bbox_summary_dict, center_coords, center_labels


def generate_final_results(bbox_summary_dict):
    red_coords_list = []
    green_coords_list = []
    cell_boundary_list = []

    for idx in bbox_summary_dict.keys():
        bbox_dict = bbox_summary_dict[idx]
        contours = bbox_dict['segment']
        signal_dict = bbox_dict['coords']
        red_coords = signal_dict['red']
        green_coords = signal_dict['green']

        if len(red_coords) > 0:
            red_coords_list.extend(red_coords[:])
        if len(green_coords) > 0:
            green_coords_list.extend(green_coords[:])
        cell_boundary_list.append(contours)

    return cell_boundary_list, red_coords_list, green_coords_list


def prediction_show(img_, bbox_summary_dict, instance_color_dict):
    color_dict = {"red": (255, 0, 0), "green": (0, 255, 0), "yellow": (255, 255, 0)}
    img = img_.copy()
    radius = 5
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx in bbox_summary_dict.keys():
        bbox_dict = bbox_summary_dict[idx]
        bbox = bbox_dict['bbox']
        contours = bbox_dict['show_segment']
        class_name = bbox_dict['class_name']
        summary_str = bbox_dict['summary']
        signal_dict = bbox_dict['coords']
        red_coords = signal_dict['red']
        green_coords = signal_dict['green']
        yellow_coords = signal_dict['yellow']
        x1, x2, y1, y2 = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
        cv2.drawContours(img, [contours], 0, instance_color_dict[class_name], 2)

        # draw signals
        for red_coord in red_coords:
            img = cv2.circle(img, (int(red_coord[0]), int(red_coord[1])), radius, color_dict["red"], thickness)
        for green_coord in green_coords:
            img = cv2.circle(img, (int(green_coord[0]), int(green_coord[1])), radius, color_dict["green"], thickness)
        for yellow_coord in yellow_coords:
            img = cv2.circle(img, (int(yellow_coord[0]), int(yellow_coord[1])), radius, color_dict["yellow"], thickness)
        # draw summary
        if summary_str != "":
            summary = '(' + summary_str + ')'
            img = cv2.putText(img, summary, (x1, y2), font, 1, (255, 255, 255), 2)

    return img


def filter_bbox(bbox_summary_dict):
    bbox_summary_dict_copy = bbox_summary_dict.copy()
    final_bbox_summary_dict = {}
    for bbox_key, bbox_dict in bbox_summary_dict_copy.items():
        signal_dict = bbox_dict['coords']
        red_coords = signal_dict['red']
        green_coords = signal_dict['green']
        if len(green_coords) >= 1 and len(red_coords) >= 1:
            final_bbox_summary_dict[bbox_key] = bbox_dict
    return final_bbox_summary_dict


def save_both_results(cell_or_tissue, pred, score_thr, overlap_thr, ots_thr):
    if cell_or_tissue not in ('TISSUE', 'CELL'):
        raise ValueError('You can only choose CELL or TISSUE for calculation')

    bbox_start_time = time.time()
    bbox_summary_dict, center_coords, center_labels = generate_bbox_summary(
        cell_or_tissue, pred,
        score_thr=score_thr,
        overlap_thr=overlap_thr, ots_thr=ots_thr
    )
    logger.info(f'generate_bbox_summary time: {time.time() - bbox_start_time}')
    cell_boundary_list, red_coords_list, green_coords_list = generate_final_results(bbox_summary_dict)

    return cell_boundary_list, red_coords_list, green_coords_list


def parse_args():
    parser = ArgumentParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config",
                        default=os.path.join(current_dir, 'config_and_weights/config.py'),
                        help="Config file")
    parser.add_argument(
        "--checkpoint",
        default=oss.generate_sign_url('GET', oss.path_join(
            'AI', 'FISH_deployment', 'config_and_weights', 'mul0.2_coef0.1_redcoef0.8_epoch_300.pth')),
        help="checkpoint file")
    parser.add_argument("--CELL_OR_TISSUE",
                        default="TISSUE",
                        help="CELL_OR_TISSUE")
    parser.add_argument("--img_dir",
                        default="/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Her2_Images_40X_szl_dense/",
                        help="Image file")
    parser.add_argument(
        "--output", type=str,
        default='/data2/Caijt/FISH_mmdet/data/FISH_TISSUE/Test_Results/HER2_TEST_1011_dense_coef0.1_size14_mul0.5/',
        help="specify the directory to save visualization results.")

    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--dist_threshold", type=float, default=15, help="Distance threshold for fusion recognition")
    parser.add_argument("--FISH_type", default="Amplification", help="Amplification, Deletion, Fracture, Fusion")

    parser.add_argument("--score-thr", type=float, default=0.3, help="bbox score threshold")
    parser.add_argument("--overlap-thr", type=float, default=0.4, help="bbox score threshold")
    parser.add_argument("--ots-thr", type=float, default=0.85, help="3 bbox overlap threshold")

    args = parser.parse_args()
    return args


def get_value(cfg: dict, chained_key: str):
    keys = chained_key.split(".")
    if len(keys) == 1:
        return cfg[keys[0]]
    else:
        return get_value(cfg[keys[0]], ".".join(keys[1:]))


def resolve(cfg: Union[dict, list], base=None):
    if base is None:
        base = cfg
    if isinstance(cfg, dict):
        return {k: resolve(v, base) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [resolve(v, base) for v in cfg]
    elif isinstance(cfg, tuple):
        return tuple([resolve(v, base) for v in cfg])
    elif isinstance(cfg, str):
        # process
        var_names = pattern.findall(cfg)
        if len(var_names) == 1 and len(cfg) == len(var_names[0]):
            return get_value(base, var_names[0][2:-1])
        else:
            vars = [get_value(base, name[2:-1]) for name in var_names]
            for name, var in zip(var_names, vars):
                cfg = cfg.replace(name, str(var))
            return cfg
    else:
        return cfg


def setup_env(cfg):
    os.environ["WORK_DIR"] = cfg.work_dir


def patch_config(cfg):
    cfg_dict = super(Config, cfg).__getattribute__("_cfg_dict").to_dict()
    cfg_dict["cfg_name"] = os.path.splitext(os.path.basename(cfg.filename))[0]
    cfg_dict = resolve(cfg_dict)
    cfg = Config(cfg_dict, filename=cfg.filename)
    # enable environment variables
    setup_env(cfg)
    return cfg


def run_fish(slide):
    cell_start_time = time.time()
    cfg = Config.fromfile(os.path.join(current_dir, 'config_and_weights/config.py'))
    cfg = patch_config(cfg)

    oss_key = oss.path_join(
        'AI', 'FISH_deployment', 'config_and_weights', 'mul0.2_coef0.1_redcoef0.8_epoch_300.pth')
    file_path = os.path.join(current_dir, 'config_and_weights', 'mul0.2_coef0.1_redcoef0.8_epoch_300.pth')
    oss.get_object_to_file(oss_key, file_path)

    model = init_detector(cfg, checkpoint=file_path, device='cuda:0')
    middle_time = time.time()
    logger.info(f'cell_init_time: {middle_time - cell_start_time}')
    img = slide
    result = inference_detector(model, img)
    cell_cal_time = time.time() - middle_time
    logging.info(f'cell_inf_time: {cell_cal_time}')
    cell_boundary_list, red_coords_list, green_coords_list = save_both_results(
        cell_or_tissue='TISSUE',
        pred=result,
        score_thr=0.3,
        overlap_thr=0.4,
        ots_thr=0.85
    )

    return cell_boundary_list, red_coords_list, green_coords_list


if __name__ == "__main__":
    # print('proj_root', proj_root)
    # args = parse_args()
    # slide_path = r'C:\znbl3_branch\alg\Algorithms\FISH_deployment\data\FISH_TISSUE\Her2_Images_40X_szl_dense\2022-8-2_202227312-1_01.jpg'
    # img_result_path = r"C:\Users\Administrator\Desktop\ai\Algorithms\FishTissue\1965452\2019-32158B_result.jpg"
    # start_time = time.time()
    # cell_boundary_list, red_coords_list, green_coords_list = run_fish(slide_path)
    # print(cell_boundary_list[0].shape)
    pass
