class Pdl1Consts(object):

    cell_label_dict = {'neg_norm': 0, 'neg_tumor': 1, 'pos_norm': 2, 'pos_tumor': 3}

    cell_type_list = sorted(cell_label_dict.keys(), key=lambda k: k[1], reverse=True)

    reversed_cell_label_dict = {v: k for k, v in cell_label_dict.items()}

    display_color_dict = {0: 'purple', 1: 'green', 2: 'yellow', 3: 'red'}

    label_to_diagnosis_type = {0: 3, 1: 1, 2: 2, 3: 0, 4: 2, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

    annot_clss_map_dict = {
        '阳性肿瘤细胞': 0,
        '阴性肿瘤细胞': 1,
        '阳性组织细胞': 2,
        '阴性组织细胞': 3,
        '阳性淋巴细胞': 4,
        '阴性淋巴细胞': 5,
        '纤维细胞': 6,
        '其他炎性细胞': 7,
        '正常肺泡细胞': 8,
        '碳末沉渣': 9,
        '其他': 10
    }

    label_to_en = {
        0: 'Negative normal cells',
        1: 'Negative tumor cells',
        2: 'Positive normal cells',
        3: 'Positive tumor cells',
    }

    sorted_labels = [0, 2, 1, 3]

    reversed_annot_clss_map_dict = {v: k for k, v in annot_clss_map_dict.items()}
