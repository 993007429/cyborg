
class Her2Consts(object):

    label_to_idx_dict = {
        '微弱的不完整膜阳性肿瘤细胞': 0,
        '弱-中等的完整细胞膜阳性肿瘤细胞': 1,
        '阴性肿瘤细胞': 2,
        '强度的完整细胞膜阳性肿瘤细胞': 3,
        '中-强度的不完整细胞膜阳性肿瘤细胞': 4,
        '纤维细胞': 5,
    }

    idx_to_label = {v: k for k, v in label_to_idx_dict.items()}

    label_to_diagnosis_type = {0: 1, 1: 2, 2: 7, 3: 4, 4: 3, 5: 12}

    cell_label_dict = {
        '0': '微弱的不完整膜阳性肿瘤细胞',
        '1': '弱中等的完整膜阳性肿瘤细胞',
        '2': '阴性肿瘤细胞',
        '3': '强度的完整膜阳性肿瘤细胞',
        '4': '中强度的不完整膜阳性肿瘤细胞',
        '5': '组织细胞',
    }

    sorted_labels = ['5', '2', '0', '1', '4', '3']

    type_color_dict = {
        1: '#FFC1C1', 2: '#FC7915', 3: '#9974FE', 4: '#FF0000', 7: '#3D901F', 9: '#A8FC8E', 10: '#A8FC8E',
        11: '#A8FC8E', 12: '#A8FC8E', 13: '#A8FC8E'}

    rois_summary_dict = {
        '微弱的不完整膜阳性肿瘤细胞': 0, '弱中等的完整膜阳性肿瘤细胞': 0, '中强度的不完整膜阳性肿瘤细胞': 0,
        '强度的完整膜阳性肿瘤细胞': 0, '阴性肿瘤细胞': 0, '组织细胞': 0, '淋巴细胞': 0, '纤维细胞': 0, '其他非肿瘤细胞': 0}

    level = {0: "HER-2 0", 1: "HER-2 1+", 2: "HER-2 2+", 3: "HER-2 3+"}

    display_cell_types = [
        '微弱的不完整膜阳性肿瘤细胞',
        '弱中等的完整膜阳性肿瘤细胞',
        '中强度的不完整膜阳性肿瘤细胞',
        '强度的完整膜阳性肿瘤细胞',
        '完整膜阳性肿瘤细胞',
        '阳性肿瘤细胞',
        '阴性肿瘤细胞',
        '肿瘤细胞总数',
        '其他',
        '阳性肿瘤细胞占比'
    ]
