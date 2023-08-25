from cyborg.seedwork.domain.value_objects import BaseEnum


class RocheAlgorithmType(BaseEnum):
    RUO = 'RUO'
    IVD = 'IVD'
    IUO = 'IUO'


class RocheTissueType(BaseEnum):
    PROSTATE = 'PROSTATE'
    LUNG = 'LUNG'
    HEMATOLOGICAL = 'HEMATOLOGICAL'
    BLADDER = 'BLADDER'
    COLORECTAL = 'COLORECTAL'
    BREAST = 'BREAST'
    GASTRIC = 'GASTRIC'
    SKIN = 'SKIN'
    LIVER = 'LIVER'
    LYMPH = 'LYMPH'


class RocheAnnotationType(BaseEnum):
    INCLUSION = 'INCLUSION'
    EXCLUSION = 'EXCLUSION',
    BOTH = 'BOTH'
