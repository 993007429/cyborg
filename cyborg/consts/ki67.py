class Ki67Consts(object):
    cell_label_dict = {'阴性肿瘤细胞': 0, '阳性肿瘤细胞': 1, '阴性淋巴细胞': 2, '阳性淋巴细胞': 3, '纤维细胞': 4,
                       '其他': 5}

    reversed_cell_label_dict = {v: k for k, v in cell_label_dict.items()}

    roi_label_dict = {'肿瘤区域（弱）': 0, '肿瘤区域（中）': 1, '肿瘤区域（强）': 2, '其他': 3, 'ROI': 4}

    reversed_roi_label_dict = {v: k for k, v in roi_label_dict.items()}

    label_to_diagnosis_type = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2}

    display_cell_dict = {'阴性肿瘤': 0, '阳性肿瘤': 1, '非肿瘤': 2}

    cell_color_dict = {0: 'green', 1: 'red', 2: '#C280FF'}

    roi_color_dict = {0: 'grey', 1: 'grey', 2: 'grey', 3: 'grey'}
