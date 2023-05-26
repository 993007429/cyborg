from typing import List, Optional

from cyborg.seedwork.domain.value_objects import BaseValueObject, BaseEnum


class Mark(BaseValueObject):
    id: Optional[int] = None
    position: Optional[dict] = None
    ai_result: Optional[dict] = None
    fill_color: Optional[str] = None
    mark_type: Optional[int] = None
    diagnosis: Optional[dict] = None
    radius: Optional[float] = None
    area_id: Optional[int] = None
    editable: Optional[int] = None
    group_id: Optional[int] = None
    method: Optional[str] = None

    def to_dict(self):
        d = super().to_dict()
        return {k: v for k, v in d.items() if v is not None}


class ALGResult(BaseValueObject):
    ai_suggest: str
    area_marks: List[Mark] = []
    cell_marks: List[Mark] = []
    roi_marks: List[Mark] = []
    slide_quality: Optional[int] = None
    cell_num: Optional[int] = None
    prob_dict: Optional[dict] = None
    err_msg: Optional[str] = None


class AITaskStatus(BaseEnum):
    default = 0
    analyzing = 1
    success = 2
    failed = 3
