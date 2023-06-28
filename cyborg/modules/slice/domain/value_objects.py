from cyborg.seedwork.domain.value_objects import BaseEnum


class SliceStartedStatus(BaseEnum):
    default = 0
    analyzing = 1
    success = 2
    failed = 3


class SliceImageType(BaseEnum):
    histplot = 'histplot'
    scatterplot = 'scatterplot'
