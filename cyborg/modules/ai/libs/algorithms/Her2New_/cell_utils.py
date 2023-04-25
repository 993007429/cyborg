import cv2
import numpy as np

import math
import os

from collections import defaultdict

from cyborg.libs.slide.dispatch import open_slide

label_dict = {
    '微弱的不完整膜阳性肿瘤细胞': 0,
    '弱-中等的完整细胞膜阳性肿瘤细胞': 1,
    '阴性肿瘤细胞': 2,
    # '纤维细胞': 3,
    # '淋巴细胞': 4,
    # '难以区分的非肿瘤细胞': 5,
    # '组织细胞': 6,
    '强度的完整细胞膜阳性肿瘤细胞': 3,
    '中-强度的不完整细胞膜阳性肿瘤细胞': 4,
    "其他": 5,
}


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
    dict, ext = get_level_dim_dict(slice_path)
    for i in dict.keys():
        size, sc = dict[i]
        h, w = size
        if w < 6000 or h < 6000:
            break
    idx = i - 1 if (i - 1) > 0 else 0
    hw, lvl1 = dict[idx]
    h1, w1 = hw
    idx = i
    hw, lvl2 = dict[idx]
    h2, w2 = hw
    idx = idx + 1 if idx + 1 < (maxlvl - 1) else maxlvl - 1
    hw, lvl3 = dict[idx]
    h3, w3 = hw

    # original slice ,
    # lvl_list from zero to high,
    # size list corresponding to the lvl list
    # type of extension
    return [slide_original, [lvl0, lvl1, lvl2, lvl3], [[h0, w0], [h1, w1], [h2, w2], [h3, w3]], ext]


def check_map(bbox=[], map=[], scale=32):
    st_x, st_y, ed_x, ed_y = list((np.array(bbox) / scale).astype(int))
    return np.sum(map[st_y:ed_y, st_x:ed_x]) > 0


def check_map2(bbox=[], map=[], crop_len=1024, scale=32):
    st_x, st_y, ed_x, ed_y = list((np.array(bbox) / scale).astype(int))
    crop_len = crop_len / scale
    print(crop_len ** 2)
    return np.sum(map[st_y:ed_y, st_x:ed_x]) == int(crop_len ** 2)

    # crop coords ：xmin, ymin, xmax, ymax


def get_patch_with_roi(slide_w=0, slide_h=0, croplen=1024, save=False, lvl=8, map=None, scale=32):
    croplen = int(croplen * lvl)
    # crop coords ：xmin, ymin, xmax, ymax
    croplen = int(croplen * lvl)
    print('map size')
    print(map.shape)
    h, w = slide_h, slide_w
    num_y = math.ceil(h // croplen)
    num_x = math.ceil(w // croplen)
    crop_x_coords = []
    crop_y_coords = []
    crop_coords = []
    index = 0
    for i in range(num_y):
        st_y = int(i * croplen)
        ed_y = st_y + croplen
        for j in range(num_x):
            st_x = int(j * croplen)
            ed_x = st_x + croplen
            if ed_x > w:
                ed_x = w
            if ed_y > h:
                ed_y = h
            if check_map2([st_x, st_y, ed_x, ed_y], map, scale=scale):
                crop_x_coords.append(st_x)
                crop_y_coords.append(st_y)
                crop_coords.append([st_x, st_y, ed_x, ed_y])
                if (save):
                    new_patch = np.zeros((croplen, croplen, 3))
                    new_patch[0:(ed_y - st_y), 0:(ed_x - st_x), :] = slide[st_y:ed_y, st_x:ed_x, :]
                    cv2.imwrite(dir + '\\' + str(index) + '.png', new_patch)
                    index = index + 1

    # else:
    valid_crop_coords = crop_coords
    return valid_crop_coords


def get_patch_with_contours(slide_w=0, slide_h=0, croplen=1024, save=False, lvl=8, map=None, scale=32):
    croplen = int(croplen * lvl)
    # crop coords ：xmin, ymin, xmax, ymax
    croplen = int(croplen * lvl)
    h, w = slide_h, slide_w
    num_y = math.ceil(h / croplen)
    num_x = math.ceil(w / croplen)
    crop_x_coords = []
    crop_y_coords = []
    crop_coords = []
    index = 0
    for i in range(num_y):
        st_y = int(i * croplen)
        ed_y = st_y + croplen
        for j in range(num_x):
            st_x = int(j * croplen)
            ed_x = st_x + croplen
            if ed_x > w:
                ed_x = w
            if ed_y > h:
                ed_y = h
            if check_map([st_x, st_y, ed_x, ed_y], map, scale=scale):
                crop_x_coords.append(st_x)
                crop_y_coords.append(st_y)
                crop_coords.append([st_x, st_y, ed_x, ed_y])
                if (save):
                    new_patch = np.zeros((croplen, croplen, 3))
                    new_patch[0:(ed_y - st_y), 0:(ed_x - st_x), :] = slide[st_y:ed_y, st_x:ed_x, :]
                    cv2.imwrite(dir + '\\' + str(index) + '.png', new_patch)
                    index = index + 1

    # else:
    valid_crop_coords = crop_coords
    return valid_crop_coords


# <class 'list'>, {'all': 207714, 0: 10901, 1: 0, 2: 45020, 3: 1, 4: 21, 5: 151771}
def cal_score_reverse(a_wsi_result):
    # np.save(r'D:\result_test\result1.npy', a_wsi_result)
    cell_count1 = float(a_wsi_result[label_dict['微弱的不完整膜阳性肿瘤细胞']])
    cell_count2 = float(a_wsi_result[label_dict['弱-中等的完整细胞膜阳性肿瘤细胞']])
    cell_count3 = float(a_wsi_result[label_dict['中-强度的不完整细胞膜阳性肿瘤细胞']])
    cell_count4 = float(a_wsi_result[label_dict['强度的完整细胞膜阳性肿瘤细胞']])
    if (cell_count1) >= (cell_count4):
        cell_count2 = cell_count3 + cell_count2
    if (cell_count1) < (cell_count4):
        cell_count1 = cell_count3 + cell_count1
    cell_count3 = 0
    cell_count5 = cell_count2 + cell_count4  # 完整细胞膜阳性肿瘤细胞
    cell_count6 = cell_count1 + cell_count2 + cell_count3 + cell_count4  # 阳性肿瘤细胞
    cell_count7 = float(a_wsi_result[label_dict['阴性肿瘤细胞']])
    cell_count8 = cell_count6 + cell_count7 + 1e-16

    if (cell_count4 / cell_count8) > 0.1:
        return 3, False
    elif ((cell_count4 / cell_count8) <= 0.1 and (cell_count4 / cell_count8) >= 0.005) or (
            (cell_count5 / cell_count8) > 0.1 and (cell_count4 / cell_count8) < 0.1):
        return 2, False
    elif ((cell_count2 / cell_count8) >= 0.02 and (cell_count2 / cell_count8) <= 0.1) or (
            ((cell_count1 + cell_count3) / cell_count8) > 0.1 and (cell_count5 / cell_count8) <= 0.1):
        # if(cell_count5 < 100):
        #     return 0,True
        return 1, False
    elif ((cell_count1 + cell_count3) / cell_count8 <= 0.1) or (cell_count6 / cell_count8) < 0.01:
        return 0, False
    return -1, False


def analysis_wsi(opt, wsi_cell_labels):
    flg = False
    cell_count_dict = defaultdict(list)
    cell_count_dict['all'] = len(wsi_cell_labels)
    for i in range(opt.num_classes):
        cell_count_dict[i] = 0
    for idx, data in enumerate(wsi_cell_labels):
        cell_count_dict[data] += 1

    score, flg = cal_score_reverse(cell_count_dict)

    print(cell_count_dict)
    print(score)
    return score, cell_count_dict, flg


def vis(wsi_cell_coords, wsi_cell_labes, slice_path):
    wsi_cell_center_coords = np.concatenate(wsi_cell_center_coords, axis=0)
    wsi_cell_center_coords = wsi_cell_center_coords // scale
    wsi_cell_center_coords = wsi_cell_center_coords.astype(np.int)
    wsi_cell_labels = np.concatenate(wsi_cell_labels, axis=0)
    wsi_cell_labels = wsi_cell_labels.astype(np.int)
    wsi_cell_image = draw_center(thumbnail, wsi_cell_center_coords, wsi_cell_labels, radius=1, thickness=1)
    wsi_cell_image = wsi_cell_image[:, :, ::-1]
