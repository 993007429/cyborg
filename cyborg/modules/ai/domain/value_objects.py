from typing import List, Optional

from cyborg.seedwork.domain.value_objects import BaseValueObject, BaseEnum


class Mark(BaseValueObject):
    id: Optional[int] = None
    position: Optional[dict] = None
    ai_result: Optional[dict] = None
    fill_color: Optional[str] = None
    stroke_color: Optional[str] = None
    mark_type: Optional[int] = None
    diagnosis: Optional[dict] = None
    radius: Optional[float] = None
    area_id: Optional[int] = None
    editable: Optional[int] = None
    group_id: Optional[int] = None
    method: Optional[str] = None
    is_export: Optional[int] = None

    def to_dict(self):
        d = super().to_dict()
        return {k: v for k, v in d.items() if v is not None}


class ALGResult(BaseValueObject):
    ai_suggest: str
    cell_marks: List[Mark] = []
    roi_marks: List[Mark] = []
    slide_quality: Optional[int] = None
    cell_num: Optional[int] = None
    prob_dict: Optional[dict] = None
    err_msg: Optional[str] = None

    @classmethod
    def parse_ai_suggest(cls, ai_suggest: str) -> dict:
        """将tct lct dna模块的ai_suggest字符串解析成字典"""
        ai_suggest_dict = {
            "diagnosis": [],
            "microbe": [],
            "dna_diagnosis": "",
            "flag": 1
        }
        try:
            diagnosis_microbe = ai_suggest.split(";")[0].replace("  ", " ")
            if ";" in ai_suggest:
                ai_suggest_dict["dna_diagnosis"] = ai_suggest.split(";")[-1]
            if "阴性" in ai_suggest:
                if "-样本不满意" in ai_suggest:
                    temp_list = diagnosis_microbe.split(" ")
                    if len(temp_list) == 2:
                        ai_suggest_dict["diagnosis"] = ["阴性", "-样本不满意"]
                        ai_suggest_dict["microbe"] = []
                    elif len(temp_list) == 3:
                        ai_suggest_dict["diagnosis"] = ["阴性", "-样本不满意"]
                        ai_suggest_dict["microbe"] = diagnosis_microbe.split(" ")[-1].split(",")
                    else:
                        ai_suggest_dict["flag"] = 0
                        print(f"解析失败: {ai_suggest}")
                else:
                    temp_list = diagnosis_microbe.split(" ")
                    if len(temp_list) == 1:
                        ai_suggest_dict["diagnosis"] = ["阴性", ""]
                        ai_suggest_dict["microbe"] = []
                    elif len(temp_list) == 2:
                        ai_suggest_dict["diagnosis"] = ["阴性", ""]
                        ai_suggest_dict["microbe"] = diagnosis_microbe.split(" ")[-1].split(",")
                    else:
                        ai_suggest_dict["flag"] = 0
                        print(f"解析失败: {ai_suggest}")
            elif "阳性" in ai_suggest:
                temp_list = diagnosis_microbe.split(" ")
                if len(temp_list) == 2:
                    ai_suggest_dict["diagnosis"] = [temp_list[0], temp_list[1]]
                    ai_suggest_dict["microbe"] = []
                elif len(temp_list) == 3:
                    ai_suggest_dict["diagnosis"] = [temp_list[0], temp_list[1]]
                    ai_suggest_dict["microbe"] = diagnosis_microbe.split(" ")[-1].split(",")
                else:
                    ai_suggest_dict["flag"] = 0
                    print(f"解析失败: {ai_suggest}")
            else:
                ai_suggest_dict["flag"] = 0
                print(f"ai建议(tct)格式非法: {ai_suggest}")
        except Exception as e:
            ai_suggest_dict["flag"] = 0
            print(f"解析 {ai_suggest} 失败: {e}")
        return ai_suggest_dict


class AITaskStatus(BaseEnum):
    default = 0
    analyzing = 1
    success = 2
    failed = 3
    canceled = 4


class TCTDiagnosisType(BaseEnum):
    negative = 0
    HSIL = 1
    ASC_H = 2
    LSIL = 3
    ASC_US = 4
    AGC = 5

    @property
    def desc(self) -> str:
        if self == self.negative:
            return '阴性'
        else:
            return f"阳性 {self.name.replace('_', '-')}"


class MicrobeType(BaseEnum):
    trichomonas = 0  # 滴虫
    fungus = 1  # 霉菌
    filamentous = 2  # 线索
    actinomyces = 3  # 放线菌
    herpes = 4  # 疱疹

    @property
    def desc(self) -> str:
        return {
            'trichomonas': '滴虫',
            'fungus': '霉菌',
            'filamentous': '线索',
            'actinomyces': '放线菌',
            'herpes': '疱疹'
        }.get(self.name)
