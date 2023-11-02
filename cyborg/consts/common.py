class Consts:

    ALGOR_DICT = {
        'tct': 'TCT',
        'pdl1': 'PD-L1',
        'ki67': 'Ki-67',
        'Ki-67': 'Ki-67',
        'er': 'ER',
        'pr': 'PR',
        'fish': 'FISH',
        'fishTissue': 'FISH',
        'cellseg': '细胞分割',
        'celldet': '细胞检测',
        'lct': 'LCT',
        'ki67hot': 'Ki-67热区',
        'her2': 'Her-2',
        'np': '鼻息肉',
        'dna': 'TBS+DNA',
        'bm': '骨髓血细胞',
        'cd30': 'CD30'
    }

    # 系统可用的模型占用显存大小  单位G
    MODEL_SIZE = {
        'tct': (1, 5),
        'lct': (1, 5),
        'model_calibrate_lct': (1, 10),
        'model_calibrate_tct': (1, 10),
        'dna': (1, 5),
        'dna_ploidy': (1, 5),
        'bm': (1, 5),
        'pdl1': ((1, 5), 10),
        'fishTissue': (1, 5),
        'ki67': (1, 10),
        'ki67hot': (1, 10),
        'er': (1, 10),
        'pr': (1, 10),
        'her2': ((1, 5), 10),
        'np': ((1, 5), 10),
        'celldet': (1, 10),
        'cellseg': (1, 10),
    }

    # 算法超时时间  单位秒
    ALGOR_OVERTIME = {
        'tct': 1800,
        'lct': 1800,
        'ki67': 1800,
        'er': 1800,
        'pr': 1800,
        'celldet': 1800,
        'cellseg': 1800,
        'pdl1': 1800,
        'fish': 1800,
        'fishTissue': 1800,
        'ki67hot': 1800,
        'her2': 6400,
        'np': 6400,
        'dna': 1800,
        'bm': 1800,
        'model_calibrate_lct': 6400,
        'model_calibrate_tct': 6400,
    }

    ALLOWED_EXTENSIONS = [
        '.svs', '.tiff', '.tif', '.jpeg', '.vmu', '.ndpi', '.png', '.mrxs', '.svslide', '.kfb', '.scn',
        '.vms', '.svslide', '.bif', '.jpg', '.bmp', '.czi', '.sdpc', '.mdsx', '.hdx', '.zyp', '.tmap', '.tron'
    ]

    OBJECTIVE_RATE_DICT = {
        '.hdx': ['40X', '20X', '10X', '5X'],
        '.svs': ['40X', '20X', '10X', '5X'],
        '.sdpc': ['40X', '20X', '10X', '5X'],
        'other': ['80X', '40X', '20X', '10X'],
    }

    REPORT_BIZ_TYPE = 'znbl'   # 在报告中心注册的业务代码，用于识别业务方身份
