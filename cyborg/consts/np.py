class NPConsts(object):

    cell_label_dict = {'淋巴细胞': 3, '浆细胞': 2, '中性粒细胞': 0, '嗜酸性粒细胞': 1}

    roi_label_dict = {'上皮区域': 1, '腺体区域': 3, '血管区域': 2}

    cell_type_list = sorted(cell_label_dict.keys(), key=lambda k: k[1], reverse=True)

    return_diagnosis_type = {
        '嗜酸性粒细胞': 1,
        '淋巴细胞': 2,
        '浆细胞': 3,
        '中性粒细胞': 4,
        '上皮区域': 5,
        '腺体区域': 6,
        '血管区域': 7
    }

    reversed_cell_label_dict = {v: k for k, v in cell_label_dict.items()}
    reversed_roi_label_dict = {v: k for k, v in roi_label_dict.items()}

    display_color_dict = {
        "嗜酸性粒细胞": "#FF0000",
        "中性粒细胞": "#66FF00",
        "淋巴细胞": "#00EEFF",
        "浆细胞": "#EAFF00",
        "上皮区域": "#00FFBB",
        "腺体区域": "#3D1CE1",
        "血管区域": "#B821BB",
    }
