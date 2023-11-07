import logging
import os
import numpy as np
import random
import imageio
from PIL import Image
import json
from copy import deepcopy
from cyborg.libs.heimdall.dispatch import open_slide

logger = logging.getLogger(__name__)

multi_cell_cls_dict = {
    'neg': 0,
    'ASCUS': 1,
    'LSIL': 2,
    'HSIL': 4,
    'ASC-H': 3,
    'AGC': 5
}

multi_microorganism_cls_dict = {
    'neg': 0,
    '放线菌': 1,
    '滴虫': 2,
    '霉菌': 3,
    '疱疹': 4,
    '线索': 5
}

cells_result_template = {
    'ASCUS': {'num': 0, 'data': []},
    'ASC-H': {'num': 0, 'data': []},
    'LSIL': {'num': 0, 'data': []},
    'HSIL': {'num': 0, 'data': []},
    'AGC': {'num': 0, 'data': []},
    '滴虫': {'num': 0, 'data': []},
    '霉菌': {'num': 0, 'data': []},
    '线索': {'num': 0, 'data': []},
    '疱疹': {'num': 0, 'data': []},
    '放线菌': {'num': 0, 'data': []},
    "萎缩性改变": {"num": 0, "data": []},
    "修复细胞": {"num": 0, "data": []},
    "化生细胞": {"num": 0, "data": []},
    "腺上皮细胞": {"num": 0, "data": []},
    "炎性细胞": {"num": 0, "data": []}
}

translate_map = {
    'ASC-US': 'ASCUS',
    'ASC-H': 'ASC-H',
    'LSIL': 'LSIL',
    'HSIL': 'HSIL',
    'AGC': 'AGC',

    'TRI': '滴虫',
    'CAN': '霉菌',
    'CC': '线索',
    'HSV': '疱疹',
    'ACT': '放线菌',

    'ATR': '萎缩性改变',
    'RAP': '修复细胞',
    'META': '化生细胞',
    'GC': '腺上皮细胞',
    'INF': '炎性细胞',

    'positive': '阳性',
    'negative': '阴性'
}

color_map = {
    'ASC-US': 'red',
    'ASC-H': 'red',
    'LSIL': 'red',
    'HSIL': 'red',
    'AGC': 'red',

    'TRI': '#00FFFB',
    'CAN': '#00FFFB',
    'CC': '#00FFFB',
    'HSV': '#00FFFB',
    'ACT': '#00FFFB',

    'ATR': '#B8FFDB',
    'RAP': '#B8FFDB',
    'META': '#B8FFDB',
    'GC': '#B8FFDB',
    'INF': '#B8FFDB',
}

multi_cell_cls_dict_reverse = {v: k for k, v in multi_cell_cls_dict.items()}
multi_microorganism_cls_dict_reverse = {v: k for k, v in multi_microorganism_cls_dict.items()}


def process_cell_result(result):
    cell_list = []

    microbe_bboxes = result['microbe_bboxes1']
    microbe_pred = result['microbe_pred1']
    cell_bboxes = result['cell_bboxes1']
    cell_prob = result['cell_prob1']

    pick_idx_m = np.where(microbe_pred == multi_microorganism_cls_dict['霉菌'])[0][:5]
    pick_idx_xs = np.where(microbe_pred == multi_microorganism_cls_dict['线索'])[0][:3]
    pick_idx_t = np.where(microbe_pred == multi_microorganism_cls_dict['滴虫'])[0][:4]
    picked_microbe_idx = np.hstack([pick_idx_m, pick_idx_xs, pick_idx_t])

    microbe_idx_list = list(np.hstack(
        [picked_microbe_idx, np.setdiff1d(np.arange(microbe_bboxes.shape[0]), picked_microbe_idx)]).astype(np.int))
    cell_idx_list = list(np.arange(cell_bboxes.shape[0]).astype(np.int))

    for i in range(min(100, len(cell_idx_list) + len(microbe_idx_list))):
        if len(microbe_idx_list) > 0 and (i % 7 > 4 or len(cell_idx_list) == 0):
            cur_idx = microbe_idx_list.pop(0)
            cur_box = microbe_bboxes[cur_idx]
            # import pdb; pdb.set_trace()
            cur_label = multi_microorganism_cls_dict_reverse[int(microbe_pred[cur_idx])]
            cur_type = int(microbe_pred[cur_idx])
        else:
            cur_idx = cell_idx_list.pop(0)
            cur_box = cell_bboxes[cur_idx]
            cur_label = str(round(cell_prob[cur_idx] * 100, 2)) + '%'
            cur_type = 7

        xmin, ymin, xmax, ymax = map(int, cur_box.tolist())
        roi_center = [int((xmin + xmax) / 2), int((ymin + ymax) / 2)]
        contour_point = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
        cell_list.append(
            {'contourPoints': contour_point, 'type': cur_type, 'center': roi_center, 'label': cur_label}
        )
    return cell_list


def save_roi(slide_path, result, num_row=6, num_col=7, patch_size=2048, scale=1, border_thickness=15,
             is_save_label=False, label_size=500, alg_type='lct'):
    patch_size *= scale
    result_dict = {}

    num_patch = num_row * num_col
    slide = open_slide(slide_path)

    slide_mpp = slide.mpp
    cells = result['cells']
    num_cells = len(cells)

    height, width = slide.height, slide.width
    h, w = (num_row + 1) * border_thickness + num_row * patch_size, \
           (num_col + 1) * border_thickness + num_col * patch_size
    combined_image = np.ones((h, w, 3), dtype=np.uint8) * 255

    patch_cells = []

    cur_roi_indx = 0  # current index in N rois
    cur_cell_idx = 0  # current index in total num of cells

    while cur_roi_indx < num_patch and cur_cell_idx < num_cells:
        try:
            this_cell = cells[cur_cell_idx]
            center_coords = this_cell['center']
            contour_coords = this_cell['contourPoints']
            if 'label' in this_cell:
                label = this_cell['label']
            else:
                label = ''
            x_center = int(center_coords[0])
            y_center = int(center_coords[1])
            crop_x, crop_y = x_center - patch_size // 2, y_center - patch_size // 2
            crop_x = int(min(max(crop_x, 0), width - patch_size - 1))
            crop_y = int(min(max(crop_y, 0), height - patch_size - 1))
            patch_img = slide.read(location=(crop_x, crop_y), size=(patch_size, patch_size), scale=scale)
            col = cur_roi_indx % num_col
            row = cur_roi_indx // num_col

            start_x = (col + 1) * border_thickness + col * patch_size
            start_y = (row + 1) * border_thickness + row * patch_size

            xmin, ymin = contour_coords[0]
            xmax, ymax = contour_coords[2]
            xmin = int((xmin - crop_x) / scale + start_x)
            xmax = int((xmax - crop_x) / scale + start_x)
            ymin = int((ymin - crop_y) / scale + start_y)
            ymax = int((ymax - crop_y) / scale + start_y)
            x_coords = [xmin, xmax, xmax, xmin]
            y_coords = [ymin, ymin, ymax, ymax]

            x_patch_coords = [start_x, start_x + patch_size, start_x + patch_size, start_x]
            y_patch_coords = [start_y, start_y, start_y + patch_size, start_y + patch_size]

            combined_image[start_y:start_y + patch_size, start_x:start_x + patch_size, :] = patch_img
            patch_cells.append(
                {
                    'id': str(int(random.random() * 100000)),
                    'path': {'x': x_coords, 'y': y_coords},
                    'remark': label,
                    'patch_path': {'x': x_patch_coords, 'y': y_patch_coords},
                    'microorganism': False,
                    'positive': False
                })

            cur_roi_indx += 1
            cur_cell_idx += 1
        except Exception as e:
            logger.error(e)
            cur_cell_idx += 1
    slide_path = slide.filename

    if is_save_label:
        label_path = os.path.join(os.path.dirname(slide_path), 'label.png')
        slide.saveLabel(label_path)
        if os.path.exists(label_path):
            slide_label = Image.open(label_path)
            if slide_label.mode != 'RGB':
                slide_label = slide_label.convert('RGB')

            label_width, label_height = slide_label.size
            if label_height > label_width:
                slide_label = slide_label.rotate(90)
                label_width, label_height = label_height, label_width

            resize_ratio = label_size / label_height
            slide_label = slide_label.resize((round(label_width * resize_ratio), label_size))
            slide_label = np.array(slide_label, dtype=np.uint8)
            label_resize_h, label_resize_w = slide_label.shape[0:2]
            upper_label = np.ones((label_resize_h, w, 3), dtype=np.uint8) * 255
            upper_label[0:label_resize_h, 0:label_resize_w, :] = slide_label
            combined_image = np.vstack((combined_image, upper_label))
        combined_image = combined_image.astype(np.uint8)
    os.makedirs(os.path.join(os.path.split(slide_path)[0], 'ai', alg_type), exist_ok=True)
    imageio.imsave(
        os.path.join(os.path.split(slide_path)[0], 'ai', alg_type, 'rois.jpg'),
        combined_image)
    # TODO decode error
    # imageio.imsave(os.path.join(res_path.decode(), 'rois.jpg'), combined_image)
    # algor_type = os.path.splitext(os.path.basename(__file__))[0]
    result_dict[alg_type] = {'diagnosis': result['diagnosis'], 'result': patch_cells}
    result_dict['mpp'] = slide_mpp

    with open(os.path.join(os.path.split(slide_path)[0], 'ai', alg_type, 'rois.json'), 'w', encoding='utf-8') as f:
        json.dump(result_dict, f)


def save_empty_roi(slide_path, alg_type='lct', message=''):
    empty_img = np.zeros((10240, 10240, 3), dtype=np.uint8)
    imageio.imsave(os.path.join(os.path.split(slide_path)[0], 'ai', alg_type, 'rois.jpg'), empty_img)
    with open(os.path.join(os.path.split(slide_path)[0], 'ai', alg_type, 'rois.json'), 'w', encoding='utf-8') as f:
        json.dump({alg_type: {'diagnosis': message, 'result': []}}, f)
    logger.error(f'{alg_type} -- {slide_path} generating rois failed,  a black image is generated, err_msg: {message}')


def generate_ai_result(result: dict, roiid: int):
    cells = deepcopy(cells_result_template)
    if len(result) > 0:
        cell_id = 0
        diagnosis = [result['diagnosis'], result['tbs_label']]
        clarity_score = result['clarity']
        quality = result['quality']
        wsi_cell_num = result['cell_num']
        bboxes = result['bboxes']
        cell_prob = result['cell_prob']
        cell_pred = result['cell_pred']
        microbe_prob = result['microbe_prob']
        microbe_pred = result['microbe_pred']
        microbe_diagnosis = []
        cell_prob_sort_idx = np.argsort(1 - cell_prob)
        microbe_prob_sort_idx = np.argsort(1 - microbe_prob)
        sorted_cell_pred = cell_pred[cell_prob_sort_idx]
        sorted_microbe_pred = microbe_pred[microbe_prob_sort_idx]
        sorted_microbe_prob = microbe_prob[microbe_prob_sort_idx]

        for k, v in multi_cell_cls_dict.items():
            if v > 0:
                pick_idx = np.where(sorted_cell_pred == v)[0]
                this_type_cell_num = int(pick_idx.size)
                cell_list = []

                for idx in cell_prob_sort_idx[pick_idx[:100]]:
                    xmin, ymin, xmax, ymax = bboxes[idx]
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    cell_list.append({'id': cell_id,
                                      'path': {'x': [xmin, xmax, xmax, xmin], 'y': [ymin, ymin, ymax, ymax]},
                                      'image': 0,
                                      'editable': 0,
                                      'dashed': 0,
                                      'fillColor': '',
                                      'mark_type': 2,
                                      'area_id': roiid,
                                      'method': 'rectangle',
                                      'strokeColor': 'red',
                                      'radius': 0,
                                      'cell_pos_prob': float(cell_prob[idx])
                                      })
                    cell_id += 1
                cells[k] = {'num': this_type_cell_num, 'data': cell_list}

        for k, v in multi_microorganism_cls_dict.items():
            if k != 'neg':
                pick_idx = np.where(sorted_microbe_pred == v)[0]
                if pick_idx.size > 1000:
                    pick_idx = np.where(np.logical_and(sorted_microbe_pred == v, sorted_microbe_prob > 0.90))[0]

                this_type_cell_num = int(pick_idx.size)
                if k not in microbe_diagnosis:
                    if k == '线索':
                        if this_type_cell_num > 100:
                            microbe_diagnosis.append(k)
                    elif k == '滴虫':
                        if this_type_cell_num > 50:
                            microbe_diagnosis.append(k)
                    elif k == '霉菌':
                        if this_type_cell_num > 5:
                            microbe_diagnosis.append(k)
                    else:
                        if this_type_cell_num > 1:
                            microbe_diagnosis.append(k)

                cell_list = []
                for idx in microbe_prob_sort_idx[pick_idx[:100]]:
                    xmin, ymin, xmax, ymax = bboxes[idx]
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    cell_list.append({'id': cell_id,
                                      'path': {'x': [xmin, xmax, xmax, xmin], 'y': [ymin, ymin, ymax, ymax]},
                                      'image': 0,
                                      'editable': 0,
                                      'dashed': 0,
                                      'fillColor': '',
                                      'mark_type': 2,
                                      'area_id': roiid,
                                      'method': 'rectangle',
                                      'strokeColor': 'red',
                                      'radius': 0
                                      })
                    cell_id += 1
                cells[k] = {'num': this_type_cell_num, 'data': cell_list}
        ai_result = {
            'cell_num': wsi_cell_num,
            'clarity': clarity_score,
            'slide_quality': quality,
            'diagnosis': diagnosis,
            'microbe': microbe_diagnosis,
            'cells': cells,
            'whole_slide': 1
        }

    else:
        ai_result = {
            'cell_num': 0,
            'clarity': 0.0,
            'slide_quality': '',
            'diagnosis': ['', ''],
            'microbe': [''],
            'cells': cells,
            'whole_slide': 1,
        }
    return ai_result


def generate_ai_result2(result: dict, roiid: int):
    cells = deepcopy(cells_result_template)

    if len(result) > 0:
        cell_id = 0
        diagnosis = [translate_map[result['diagnosis']], result['tbs_label']]
        quality = result['quality']
        wsi_cell_num = result['cell_num']

        for cell in result['cells']:
            box, label, prob = cell['bbox'], cell['label'], cell['prob']
            color = 'red' if label not in color_map else color_map[label]
            cells[translate_map[label]]['num'] += 1
            if cells[translate_map[label]]['num'] <= 100:
                xmin, ymin, xmax, ymax = box
                cells[translate_map[label]]['data'].append(
                    {"id": cell_id, "path": {"x": [xmin, xmax, xmax, xmin], "y": [ymin, ymin, ymax, ymax]},
                     "image": 0, "editable": 0, "dashed": 0, "fillColor": "", "mark_type": 2, "area_id": roiid,
                     "method": "rectangle", "strokeColor": color, "radius": 0, "cell_pos_prob": prob})
                cell_id += 1
        microbe = [translate_map[k] for k in result['microbe']]
        background = [translate_map[k] for k in result['background']]
        aiResult = {
            'cell_num': wsi_cell_num,
            'clarity': 1,
            'slide_quality': quality,
            'diagnosis': diagnosis,
            'microbe': microbe,
            'background': background,
            'cells': cells,
            'whole_slide': 1
        }

    else:
        aiResult = {'cell_num': 0, 'slide_quality': 0, 'diagnosis': [], 'microbe': [], 'background': [], 'cells': [],
                    'whole_slide': 1}
    return aiResult


def generate_dna_ai_result(result: dict, roiid: int):
    nuclei_list = []
    cell_id = 1000
    for idx in range(min(100, result['nuclei_bboxes'].shape[0])):
        xmin, ymin, xmax, ymax = result['nuclei_bboxes'][idx]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        nuclei_list.append({
            'id': cell_id,
            'path': {'x': [xmin, xmax, xmax, xmin], 'y': [ymin, ymin, ymax, ymax]},
            'image': 1,
            'editable': 0,
            'dashed': 0,
            'fillColor': '',
            'mark_type': 2,
            'area_id': roiid,
            'method': 'rectangle',
            'strokeColor': '#00FFFE',
            'radius': 0,
            'dna_index': round(float(result['dna_index_values'][idx]), 2),
            'dna_amount': round(float(result['iod_values'][idx]), 2),
            'area': round(float(result['area'][idx]), 2)
        })
        cell_id += 1

    diagnosis_dict = {
        'insufficient_nuclei': '有效检测细胞不足',
        'no_abnormal_nucleus': '未见DNA倍体异常细胞',
        'a_few_abnormal_nuclei': '可见少量DNA倍体异常细胞（1-2个）',
        'plenty_of_abnormal_nuclei': '可见DNA倍体异常细胞（≥3个）',
        'normal_proliferation': '可见少量细胞增生（5%-10%）',
        'abnormal_proliferation': '可见细胞异常增生（≥10%）',
        'abnormal_nuclei_peak': '可见异倍体细胞峰'
    }

    ai_result = {
        'nuclei': nuclei_list,
        'num_abnormal_low': int(result['num_abnormal_low']),
        'num_abnormal_high': int(result['num_abnormal_high']),
        'num_normal': int(result['num_normal']),
        'dna_diagnosis': diagnosis_dict[result['dna_diagnosis']],
        'nuclei_num': int(result['num_nuclei']),
        'control_iod': round(float(result['control_iod']), 2),
        'dna_statics': result['dna_statics'],
        'cell_num': int(result['num_nuclei'])
    }

    return ai_result


def generate_dna_ploidy_aiResult(result, roiid):
    nuclei_list = []
    # print(result['nuclei_bboxes'])
    cell_id = 1
    for idx in range(result['nuclei_bboxes'].shape[0]):
        xmin, ymin, xmax, ymax = result['nuclei_bboxes'][idx]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        dna_index = round(float(result['dna_index_values'][idx]), 2)
        lesion_type = "normal"
        strokeColor = "rgb(112,182,3)"  # green
        if dna_index >= 1.25:
            lesion_type = "abnormal_low"
            strokeColor = "rgb(245,154,35)"  # yellow
        if dna_index >= 2.5:
            lesion_type = "abnormal_high"
            strokeColor = "rgb(217,0,27)"  # red
        nuclei_list.append({
            "id": cell_id,
            "path": {"x": [xmin, xmax, xmax, xmin], "y": [ymin, ymin, ymax, ymax]},
            "image": 1,
            "editable": 0,
            "dashed": 0,
            "fillColor": "",
            "mark_type": 2,
            "area_id": roiid,
            "method": "rectangle",
            "strokeColor": strokeColor,
            "radius": 0,
            "dna_iod": round(float(result['iod_values'][idx]), 2),
            "dna_index": round(float(result['dna_index_values'][idx]), 2),
            "dna_amount": round(float(result['iod_values'][idx]), 2),
            "area": round(float(result['area'][idx]), 2),
            "is_deleted": 0,
            "lesion_type": lesion_type
        })
        cell_id += 1

    diagnosis_dict = {
        'insufficient_nuclei': '有效检测细胞不足',
        'no_abnormal_nucleus': '未见DNA倍体异常细胞',
        'a_few_abnormal_nuclei': '可见少量DNA倍体异常细胞（1-2个）',
        'plenty_of_abnormal_nuclei': '可见DNA倍体异常细胞（≥3个）',
        'normal_proliferation': '可见少量细胞增生（5%-10%）',
        'abnormal_proliferation': '可见细胞异常增生（≥10%）',
        'abnormal_nuclei_peak': '可见异倍体细胞峰'
    }

    aiResult = {
        'nuclei': nuclei_list,
        'num_abnormal_low': int(result['num_abnormal_low']),
        'num_abnormal_high': int(result['num_abnormal_high']),
        'num_normal': int(result['num_normal']),
        'dna_diagnosis': diagnosis_dict[result['dna_diagnosis']],
        'nuclei_num': int(result['num_nuclei']),
        "control_iod": round(float(result['control_iod']), 2),
        'dna_statics': result['dna_statics'],
        "cell_num": int(result['num_nuclei'])
    }

    return aiResult
