from cyborg.seedwork.domain.value_objects import BaseEnum


class SliceStartedStatus(BaseEnum):
    default = 0
    analyzing = 1
    success = 2
    failed = 3

    @property
    def description(self):
        return {
            SliceStartedStatus.default: "待处理",
            SliceStartedStatus.analyzing: "处理中",
            SliceStartedStatus.success: "已处理",
            SliceStartedStatus.failed: "处理异常",
        }[self]


class SliceImageType(BaseEnum):
    histplot = 'histplot'
    scatterplot = 'scatterplot'


class SliceAlg(BaseEnum):
    tct1 = 'TCT1.0'
    tct2 = 'TCT2.0'
    lct1 = 'LCT1.0'
    lct2 = 'LCT2.0'
    pdl1 = 'PD-L1'
    ki67 = 'Ki-67'
    er = 'ER'
    pr = 'PR'
    fish = 'FISH'
    fishTissue = 'FISH'
    cellseg = '细胞分割'
    celldet = '细胞检测'
    lct = 'LCT'
    ki67hot = 'Ki-67热区'
    her2 = 'Her-2'
    np = '鼻息肉'
    dna = 'TBS+DNA'
    bm = '骨髓血细胞'
    cd30 = 'CD30'
    dna_ploidy = 'DNA倍体'
