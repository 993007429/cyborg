class TCTConsts:

    tct_multi_wsi_cls_dict = {
        'NILM': 0,
        'ASC-US': 1,
        'LSIL': 2,
        'ASC-H': 3,
        'HSIL': 4,
        'AGC': 5
    }
    tct_multi_wsi_cls_dict_reverse = {v: k for k, v in tct_multi_wsi_cls_dict.items()}

    cells_types = [
        'ASCUS', 'ASC-H', 'LSIL', 'HSIL', 'AGC', '滴虫', '霉菌', '线索', '疱疹', '放线菌', '萎缩性改变', '修复细胞', '化生细胞'
        '腺上皮细胞', '炎性细胞']
