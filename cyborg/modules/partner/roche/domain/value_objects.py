from cyborg.seedwork.domain.value_objects import BaseEnum


class RocheAlgorithmType(BaseEnum):
    RUO = 'RUO'
    IVD = 'IVD'
    IUO = 'IUO'


class RocheTissueType(BaseEnum):
    PROSTATE = 'Prostate'
    LUNG = 'Lung'
    HEMATOLOGICAL = 'Hematological'
    BLADDER = 'Bladder'
    COLORECTAL = 'Colorectal'
    BREAST = 'Breast'
    GASTRIC = 'Gastric'
    SKIN = 'Skin'
    LIVER = 'Liver'
    LYMPH = 'Lymph'


class RocheAnnotationType(BaseEnum):
    INCLUSION = 'INCLUSION'
    EXCLUSION = 'EXCLUSION',
    BOTH = 'BOTH'


class RocheAITaskStatus(BaseEnum):
    default = 0
    analyzing = 1
    success = 2
    failed = 3
