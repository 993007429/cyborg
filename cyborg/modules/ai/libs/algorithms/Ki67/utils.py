#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from imageio import imread, imsave
import shutil
import json
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from glob import glob
import torch
import skimage
import os
import torchvision.transforms as transforms
import time
from threading import Thread
from queue import Queue
import cv2
from numba import jit, njit
import numba
if numba.__version__.startswith('0.5'):
    from numba.typed import List
    ls = List
else:
    ls = list

anno_label_dict = {'阴性肿瘤': 0, '阳性肿瘤': 1, '阴性淋巴': 2, '阳性淋巴': 3,
                       '阴性纤维': 4, '阳性纤维': 4, '其它': 5}
label_dict = {'0': '阴性肿瘤', '1': '阳性肿瘤', '2': '阴性淋巴',
                  '3': '阳性淋巴', '4': '纤维细胞', '5': '其它细胞'}
color_dict = {"0": (0, 255, 0), "1": (255, 0, 0), "2": (255, 255, 0),
                  "3": (139, 105, 20), "4": (0, 0, 255), "5": (128, 0, 255)}

def get_metrics(stat_dict):
    eps = 1e-5
    all_pred_num_per_class = np.array(stat_dict['pred_num'])
    all_target_num_per_class = np.array(stat_dict['target_num'])
    match_cls_num_per_class = np.array(stat_dict['match_cls_num'])
    print('pred, target, match', all_pred_num_per_class, all_target_num_per_class, match_cls_num_per_class)

    true_positive_cls = match_cls_num_per_class
    Precision_cls = (true_positive_cls + eps) / (all_pred_num_per_class + eps)
    Recall_cls = (true_positive_cls + eps) / (all_target_num_per_class + eps)
    F1_score_cls = 2 * Precision_cls * Recall_cls / (Precision_cls + Recall_cls + eps)
    cls_metrics = {'Precision': Precision_cls, 'Recall': Recall_cls,
                   'F1_score': F1_score_cls}

    for i in range(len(all_pred_num_per_class)):
        print('class {}: Precision_cls, Recall_cls, F1_score_cls'.format(label_dict[str(i)]),
              Precision_cls[i], Recall_cls[i], F1_score_cls[i], '\n')
    return cls_metrics

def walk_dir(data_dir, file_types):
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:
                if f.endswith(this_type):
                    path_list.append( os.path.join(dirpath, f)  )
                    break
    return path_list

def draw_center(img, center, label):
    img = img.copy()
    num_center = center.shape[0]
    radius = 4
    thickness = -1
    for i in range(num_center):
        if str(int(label[i])) in color_dict:
            img = cv2.circle(img, (int(center[i, 0]), int(center[i, 1])), radius, color_dict[str(int(label[i]))], thickness)
    return img

def save_anno_img(img_path, center_coords, labels, result_save_dir):
    image = imread(img_path)
    img = image.copy()
    test_res_img = draw_center(img, center_coords, labels)
    test_save_path = os.path.join(result_save_dir, os.path.basename(img_path))
    print('save_path', test_save_path)
    imsave(test_save_path, test_res_img)

def save_pred_img(img_path, id_name, center_coords_pred, labels_pred, result_save_dir):
    image = imread(img_path)
    img = image.copy()
    test_res_img = draw_center(img, center_coords_pred, labels_pred)
    test_save_path = os.path.join(result_save_dir, id_name + '_' + os.path.basename(img_path))
    print('test_save_path', test_save_path)
    imsave(test_save_path, test_res_img)
    return


def concat_images(images):
    """Generate composite of all supplied images."""
    # Get the widest width.
    height, width = images[0].shape[0], images[0].shape[1]
    # Add up all the heights.
    padding = 40
    compo_width = width * 3 + padding * 2
    composite = np.ones((height,compo_width,3), dtype=np.uint8) * 255
    for id, image in enumerate(images):
        composite[:,id*(width+padding):id*(width+padding)+width,:] = image
    return composite

def my_save_img(img_path, base_name, annotation_path, center_coords_pred, labels_pred, result_save_dir):
    center_coords, labels = parse_anno(annotation_path)
    image = imread(img_path)
    pred = draw_center(image, center_coords_pred, labels_pred)
    anno = draw_center(image, center_coords, labels)
    saved_img = concat_images([image, pred, anno])
    test_save_path = os.path.join(result_save_dir, base_name + '.png')
    print('test_save_path', test_save_path)
    imsave(test_save_path, saved_img)

def save_img(img_path, annotation_path, center_coords_pred, labels_pred, result_save_dir):
    id_name = str(os.path.basename(os.path.dirname(img_path))[0])
    center_coords, labels = my_parse_anno(annotation_path)
    assert np.amax(labels) <= 5, 'The max value of labels is %s'%(np.amax(labels))
    image = imread(img_path)
    pred = draw_center(image, center_coords_pred, labels_pred)
    anno = draw_center(image, center_coords, labels)
    saved_img = concat_images([image, pred, anno])
    test_save_path = os.path.join(result_save_dir, id_name + '_' + os.path.basename(img_path))
    print('test_save_path', test_save_path)
    imsave(test_save_path, saved_img)
    return

def parse_pred(pred_path):
    with open(pred_path, "r", encoding='utf-8') as f:
        pred_info = json.load(f)
        if 'y_coords_pred' in pred_info:
            y_coords_pred = map(int, pred_info['y_coords_pred'])
        if 'x_coords_pred' in pred_info:
            x_coords_pred = map(int, pred_info['x_coords_pred'])
        if 'labels_pred' in pred_info:
            pred_labels = list(map(int, pred_info['labels_pred']))
    pred_center_coords = list(zip(y_coords_pred, x_coords_pred))
    pred_center_coords = np.array(pred_center_coords).astype(np.int)
    pred_labels = np.array(pred_labels).astype(np.int)
    return pred_center_coords, pred_labels

def my_parse_anno(pred_path):
    with open(pred_path, "r", encoding='utf-8') as f:
        pred_info = json.load(f)
        points = pred_info['points']
        labels = pred_info['labels']
        # annotation labels range in [1,2,3,4,5,6] and should be mapped into [0,1,2,3,4,5]
    return np.array(points), np.array(labels) - 1

def parse_anno(annotation_path):
    annotation_dict = {'阴性纤维': 0, '阴性淋巴': 0, '阴性肿瘤': 0, '阳性纤维': 0, '阳性淋巴': 0, '阳性肿瘤': 0,
                       '其它': 0, '细胞总数': 0, 'ki67指数': 0}
    total_count = 0
    center_coords = []
    labels = []

    with open(annotation_path, "r", encoding='utf-8') as f:
        annotation_info = json.load(f)
        if "roilist" in annotation_info:
            roilist = annotation_info['roilist']
            for roi in roilist:
                if "remark" in roi:
                    remark = roi['remark']
                    if remark in anno_label_dict.keys():
                        annotation_dict[remark] += 1
                        total_count+=1
                        x, y = roi['path']['x'][0], roi['path']['y'][0]
                        center_coords.append([x,y])
                        labels.append(anno_label_dict[remark])

    center_coords = np.array(center_coords).astype(np.int)
    labels = np.array(labels).astype(np.int)
    return center_coords, labels

def my_save_json(img_path, base_name, annotation_path, center_coords_pred, labels_pred, eval_root):
    anno_dst_name = os.path.join(eval_root, base_name + '.json')
    pred_dst_name = os.path.join(eval_root, base_name + '_pred.json')
    img_dst_name = os.path.join(eval_root, base_name + '.png')
    shutil.copy(annotation_path, anno_dst_name)
    shutil.copy(img_path, img_dst_name)
    # print(pred_dst_name)
    res_dict = {}
    res_dict['y_coords_pred'] = list((center_coords_pred[:, 0]).astype(np.float))
    res_dict['x_coords_pred'] = list((center_coords_pred[:, 1]).astype(np.float))
    res_dict['labels_pred'] = list(labels_pred.astype(np.float))
    with open(pred_dst_name, encoding='utf-8', mode='w') as f:
        json.dump(res_dict, f)

def save_json(img_path, id_name, annotation_path, center_coords_pred, labels_pred, eval_root):

    anno_dst_name = os.path.join(eval_root, id_name + '_' + os.path.basename(annotation_path).split('.')[0] + '.json')
    pred_dst_name = os.path.join(eval_root, id_name + '_' + os.path.basename(annotation_path).split('.')[0] + '_pred.json')
    img_dst_name = os.path.join(eval_root, id_name + '_' + os.path.basename(img_path))
    shutil.copy(annotation_path, anno_dst_name)
    shutil.copy(img_path, img_dst_name)
    # print(pred_dst_name)
    res_dict = {}
    res_dict['y_coords_pred'] = list((center_coords_pred[:, 0]).astype(np.float))
    res_dict['x_coords_pred'] = list((center_coords_pred[:, 1]).astype(np.float))
    res_dict['labels_pred'] = list(labels_pred.astype(np.float))
    with open(pred_dst_name, encoding='utf-8', mode='w') as f:
        json.dump(res_dict, f)

def add2summary(image_file, cls_metrics, summary_dict):
    Precision_cls = cls_metrics['Precision']
    Recall_cls = cls_metrics['Recall']
    F1_score = cls_metrics['F1_score']
    summary_dict['slide name'].append(image_file),
    summary_dict['阴性肿瘤_RECALL'].append(Recall_cls[0]),
    summary_dict['阳性肿瘤_RECALL'].append(Recall_cls[1]),
    summary_dict['阴性淋巴_RECALL'].append(Recall_cls[2]),
    summary_dict['阳性淋巴_RECALL'].append(Recall_cls[3]),
    summary_dict['纤维细胞_RECALL'].append(Recall_cls[4]),
    summary_dict['其它细胞_RECALL'].append(Recall_cls[5]),

    summary_dict['阴性肿瘤_PRECISION'].append(Precision_cls[0]),
    summary_dict['阳性肿瘤_PRECISION'].append(Precision_cls[1]),
    summary_dict['阴性淋巴_PRECISION'].append(Precision_cls[2]),
    summary_dict['阳性淋巴_PRECISION'].append(Precision_cls[3]),
    summary_dict['纤维细胞_PRECISION'].append(Precision_cls[4]),
    summary_dict['其它细胞_PRECISION'].append(Precision_cls[5]),

    summary_dict['阴性肿瘤_F1_SCORE'].append(F1_score[0]),
    summary_dict['阳性肿瘤_F1_SCORE'].append(F1_score[1]),
    summary_dict['阴性淋巴_F1_SCORE'].append(F1_score[2]),
    summary_dict['阳性淋巴_F1_SCORE'].append(F1_score[3]),
    summary_dict['纤维细胞_F1_SCORE'].append(F1_score[4]),
    summary_dict['其它细胞_F1_SCORE'].append(F1_score[5]),

    return summary_dict

def get_match_coords_np(pred_center_coords, pred_labels, anno_center_coords, anno_labels, num_classes):

    cost_point = cdist(pred_center_coords, anno_center_coords)
    indices = linear_sum_assignment(cost_point)
    pred_idx, target_id = indices

    dist_threshold = 24
    dist = np.linalg.norm(pred_center_coords[pred_idx] - anno_center_coords[target_id], ord = 2, axis = 1)
    dist_bool = (dist < dist_threshold)

    ## get pred
    all_pred_num_per_class_list = []
    for i in range(num_classes):
        num_per_class = (pred_labels == i).sum()
        all_pred_num_per_class_list.append(num_per_class)
        print('all_pred_num_{}: '.format(label_dict[str(i)]), num_per_class)

    ## get target
    all_target_num_per_class_list = []
    for i in range(num_classes):
        num_per_class = (anno_labels == i).sum()
        all_target_num_per_class_list.append(num_per_class)
        print('target_num_{}: '.format(label_dict[str(i)]), num_per_class)

    pred_match_points = pred_center_coords[pred_idx]
    pred_match_labels = pred_labels[pred_idx]
    target_match_labels = anno_labels[target_id]
    label_match_bool = (pred_match_labels == target_match_labels)

    match_bool = label_match_bool * dist_bool
    match_bool = np.array(match_bool)
    # match_bool = np.array(label_match_bool)

    if match_bool.sum() > 0:
        print('Yeah! I got points')
        pred_match_points = pred_match_points[match_bool]
        pred_match_labels = pred_match_labels[match_bool]
        match_cls_num_per_class_list = []
        for i in range(num_classes):
            num_match_per_class = (pred_match_labels == i).sum()
            match_cls_num_per_class_list.append(num_match_per_class)
            print('match_cls_num_{}: '.format(label_dict[str(i)]), num_match_per_class)
    else:
        pred_match_points = None
        pred_match_labels = None
        match_cls_num_per_class_list = [0, 0, 0, 0, 0, 0]

    stat_dict = {'pred_num': np.array(all_pred_num_per_class_list),
                 'target_num': np.array(all_target_num_per_class_list),
                 'match_cls_num': np.array(match_cls_num_per_class_list)}

    return pred_match_points, pred_match_labels, stat_dict

def get_match_results(img_file, num_classes):
    anno_file = img_file[:-4] + '.json'
    pred_file = img_file[:-4] + '_pred.json'
    anno_center_coords, anno_labels = my_parse_anno(anno_file)
    pred_center_coords, pred_labels = parse_pred(pred_file)
    _, _, stat_dict = get_match_coords_np(pred_center_coords, pred_labels,
                                          anno_center_coords, anno_labels, num_classes)
    print('stat_dict', stat_dict)
    cls_metrics = get_metrics(stat_dict)
    print('cls_metrics', cls_metrics)

    return cls_metrics, stat_dict

class threading_classification:
    def __init__(self, slide, cnet, patch_size=1536, batch_size=8, num_workers=4, device='cpu'):
        self.slide = slide
        self.device = device
        self.cnet = cnet
        self.patch_size = patch_size
        # in my code, H denotes the first dimension, W denotes the second one. In the Openslide, they are converse.
        self.W, self.H = slide.width // patch_size +1, slide.height // patch_size +1
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.batch_size = batch_size
        self.start_time = time.time()
        self.num_workers = num_workers
        self.preds = torch.Tensor().to(self.device)
        self.processed_index = []

    def _get_index_queue(self):

        if self.H * self.W > 100:
            thumb = self.slide.read((0, 0), (self.W * self.patch_size, self.H * self.patch_size),
                                                  scale=self.patch_size//3, greyscale=True)
            min_pooling = skimage.measure.block_reduce(thumb, (3, 3), np.min)
            min_pooling = np.transpose(min_pooling)
            self.read_region = thumb
            self.indexs = np.where((min_pooling < np.max(min_pooling) * 0.98) & (min_pooling > 80))
            self.indexs = [i * self.H + j for i,j in zip(self.indexs[0], self.indexs[1])]
        else:
            self.indexs = [i*self.H + j for i in np.arange(0, self.W, 1) for j in np.arange(0, self.H, 1)]
        in_queue = Queue(len(self.indexs))
        [in_queue.put(i) for i in self.indexs]
        return in_queue

    def _load_data(self, in_queue, out_queue):
        # self._lock.acquire()
        item = None
        while True:
            if item is None:
                index = in_queue.get()
                region_start = [index // self.H * self.patch_size, index % self.H * self.patch_size]
                # print(region_start)
                scale = self.patch_size/224
                try:
                    region = self.slide.read(region_start, (self.patch_size, self.patch_size), scale)
                    if region.shape != [224,224,3]:
                        region = cv2.resize(region, (224,224))
                except Exception as e:
                    print(e)
                    region = np.zeros((224, 224, 3))
                item = self.transform(region.copy())

            if not out_queue.full():
                out_queue.put([item, index])
                # print(index)
                item = None
            else:
                time.sleep(0.01)
                continue

    def _process_data(self, out_queue):
        collected_tensors = []
        count = 0
        with torch.no_grad():
            while True:
                if len(collected_tensors) == self.batch_size or (count == len(self.indexs) and count>0):
                    batch = torch.stack(collected_tensors, dim=0)
                    batch = batch.to(self.device).float()
                    p = self.cnet(batch)
                    self.preds = torch.cat((self.preds, p), dim=0)
                    collected_tensors = []

                    if count == len(self.indexs):
                        self.heatmap = self.preds
                        print("Time consume: %s"%(time.time()-self.start_time))
                        break

                if not out_queue.empty():
                    [item,index] = out_queue.get()
                    self.processed_index.append(index)
                    count += 1
                    # print(index, count)
                    collected_tensors.append(item)
                else:
                    time.sleep(0.01)
                    continue

    def execute(self):
        in_queue = self._get_index_queue()
        out_queue = Queue(3 * self.num_workers)
        # with concurrent.futures.ThreadPoolExecutor(self.num_workers) as executor:
        #     for index in range(self.num_workers):
        #         executor.submit(self._load_data, in_queue, out_queue)
        #     self._process_data(out_queue)
            # executor.submit(self._process_data, out_queue)
        for i in range(self.num_workers):
            t = Thread(target=self._load_data, args=(in_queue, out_queue))
            t.daemon = True
            t.start()
        self._process_data(out_queue)

def count_test_summary(center_coords, labels, label_dict):
    cell_count_dict = {'阴性纤维': 0, '阴性淋巴': 0, '阴性肿瘤': 0, '阳性纤维': 0, '阳性淋巴': 0, '阳性肿瘤': 0,
                       '其它': 0, '细胞总数': 0, '肿瘤细胞': 0, 'ki67指数': 0}
    # labels = np.array(labels).astype(np.int)
    total_count = 0
    reverse_dict = {}

    for k, v in label_dict.items():
        reverse_dict[v] = k

    for i in range(np.max(labels) + 1):
        this_num = np.sum(labels == i)
        cell_count_dict[reverse_dict[i]] = this_num
        total_count += this_num

    cell_count_dict["细胞总数"] = total_count
    cell_count_dict["ki67指数"] = \
        round(cell_count_dict["阳性肿瘤"] / (cell_count_dict["阴性肿瘤"] + cell_count_dict["阳性肿瘤"] + 1e-10), 4)
    cell_count_dict["肿瘤细胞"] = cell_count_dict["阴性肿瘤"] + cell_count_dict["阳性肿瘤"]

    return cell_count_dict


def count_test_summary_P2P(center_coords, labels, label_dict):
    cell_count_dict = {'阴性肿瘤': 0, '阳性肿瘤': 0, '阴性淋巴': 0, '阳性淋巴': 0, '纤维细胞': 0, '其它细胞': 0,
                       '细胞总数': 0, '肿瘤细胞': 0, 'ki67指数': 0}
    # labels = np.array(labels).astype(np.int)
    total_count = 0
    reverse_dict = {}

    for k, v in label_dict.items():
        reverse_dict[v] = k

    try:
        for i in range(np.max(labels) + 1):
            this_num = np.sum(labels == i)
            cell_count_dict[reverse_dict[i]] = this_num
            total_count += this_num
    except Exception as e:
        print(e)
        
    cell_count_dict["细胞总数"] = total_count
    cell_count_dict["ki67指数"] = \
        round(cell_count_dict["阳性肿瘤"] / (cell_count_dict["阴性肿瘤"] + cell_count_dict["阳性肿瘤"] + 1e-10), 4)
    cell_count_dict["肿瘤细胞"] = cell_count_dict["阴性肿瘤"] + cell_count_dict["阳性肿瘤"]

    return cell_count_dict



@jit(nopython=True)
def pointinpolygon(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in numba.prange(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


@njit(parallel=True)
def parallelpointinpolygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean)
    for i in numba.prange(0, len(D)):
        D[i] = pointinpolygon(points[i,0], points[i,1], polygon)
    return D


def roi_filter(center_coords, labels, x_coords, y_coords):

    if center_coords.shape[0] > 0 and len(x_coords) > 0:
        try:
            pick_idx = parallelpointinpolygon(center_coords, ls(zip(x_coords, y_coords)))
            center_coords = center_coords[pick_idx]
            labels = labels[pick_idx]
        except Exception as e:
            print(e)
    return center_coords, labels




if __name__ == '__main__':

    # anno_label_dict = {'阴性肿瘤': 0, '阳性肿瘤': 1, '阴性淋巴': 2, '阳性淋巴': 3,
    #                    '阴性纤维': 4, '阳性纤维': 4, '其它': 5}
    # label_dict = {'0': '阴性肿瘤', '1': '阳性肿瘤', '2': '阴性淋巴',
    #               '3': '阳性淋巴', '4': '纤维细胞', '5': '其它细胞'}
    #
    # get_match_results("/data2/Caijt/KI67_under_going/Densitymap_P2P_compare/Test_Results/Evaluation_0406_p2p_0.06_test/2_32.png",
    #                   anno_label_dict = anno_label_dict,
    #                   label_dict = label_dict, num_classes = 6)

    train_dir = glob('/data2/Caijt/KI67_under_going/ki67_deployment_P2P/Data/KI67_data_V2_json_split/*_train')
    save_dir = './saved_images'

    for split_dir in train_dir:
        img_pths = glob(split_dir + '/*.png')
        save_split_dir = os.path.join(save_dir, split_dir.split('/')[-1])
        os.makedirs(save_split_dir, exist_ok=True)
        for img_pth in img_pths:
            anno_pth = img_pth[:-4] + '.json'
            with open(anno_pth, 'r') as f:
                data = json.load(f)
                points = np.array(data['points'])
                labels = np.array(data['labels']) - 1
            save_anno_img(img_pth, points, labels, save_split_dir)
