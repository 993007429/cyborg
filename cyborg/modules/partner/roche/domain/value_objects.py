import math
from typing import Optional, List, Tuple, Dict

from geojson import FeatureCollection

from cyborg.modules.partner.roche.domain.consts import HER2_ALGORITHM_DISPLAY_ID
from cyborg.seedwork.domain.value_objects import BaseEnum, BaseValueObject
from cyborg.utils.strings import snake_to_camel

ROCHE_TILE_SIZE = 2048


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
    EXCLUSION = 'EXCLUSION'
    BOTH = 'BOTH'


class RocheAITaskStatus(BaseEnum):
    pending = 0
    accepted = 1
    in_progress = 2
    completed = 3
    failed = 4
    cancelled = 5
    closed = 6

    @property
    def display_name(self):
        if self == RocheAITaskStatus.in_progress:
            return 'In-Progress'
        else:
            return snake_to_camel(self.name, is_big_camel=True)


class RocheMark(BaseValueObject):
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
    # group_id: Optional[int] = None
    method: Optional[str] = None
    is_export: Optional[int] = None

    def to_dict(self):
        d = super().to_dict()
        return {k: v for k, v in d.items() if v is not None}


class RocheDiplomat(BaseValueObject):
    version: str = '1.30'
    locale: str = 'en-US'
    uuid: str
    # date: str


class RocheDeveloper(BaseValueObject):
    company_name: str
    company_logo: str
    address: str = ''
    infoURL: str = ''
    description: str = ''


class RocheAlgorithm(BaseValueObject):
    algorithm_id: str
    algorithm_display_id: str
    algorithm_name: str
    algorithm_description: str
    algorithm_type: RocheAlgorithmType
    version_number: str
    software_build: str
    status: str = 'Active'
    stain_name: str
    tissue_types: List[dict]
    indication_types: List[dict]
    vendor: str
    # supported_magnification: str
    supported_mpp_ranges: List[list]
    supported_image_formats: List[str]
    supported_scanners: List[str]
    required_slide_types: List[str]
    roi_analysis_support: bool
    primary_analysis_overlay_display: bool
    provides_primary_analysis_score: bool
    provides_navigational_heatmap_thumbnail: bool
    manual_score_mode: str
    results_parameters: List[dict]
    # clone_type: List[str]
    # supported_mpp_ranges: List[List[int]]
    # secondary_analysis_support: bool,
    # secondary_analysis_annotation_type: RocheAnnotationType,
    # max_secondary_analysis_allowed: int,
    # 'overlay_acceptance_required': False,
    # 'slide_score_acceptance_required': False,
    # 'requires_analysis_rejection_feedback': False,
    # 'provides_prognostic_score': False,


class RocheWsiInput(BaseValueObject):
    image_location: str
    image_id: str
    sha256: str
    md5: str
    scanner_name: str
    scanner_unit_number: str
    microns_per_pixel_x: float
    microns_per_pixel_y: float
    slide_width: int
    slide_height: int
    slide_magnification: int
    slide_depth: int
    number_levels: int
    dimensions: List[List[int]]


class RocheWsiScores(BaseValueObject):
    whole_slide_score: str

    def to_dict(self):
        return {
            'WholeSlideScore': self.whole_slide_score
        }


class RocheIndexItem(BaseValueObject):
    filename: str
    bbox: List[int]
    geo_json: FeatureCollection
    heatmap_points: List[list]

    @classmethod
    def locate(cls, x: int, y: int) -> Tuple[int, int]:
        tile_x = math.ceil(x / ROCHE_TILE_SIZE) - 1
        tile_y = math.ceil(y / ROCHE_TILE_SIZE) - 1
        return tile_x, tile_y

    @classmethod
    def new_geo(cls):
        return FeatureCollection(features=[])

    @classmethod
    def new_item(cls, tile_x: int, tile_y: int) -> Optional['RocheIndexItem']:
        return cls(
            filename=f'tile{tile_x}_{tile_y}',
            bbox=[
                tile_x * ROCHE_TILE_SIZE,
                tile_y * ROCHE_TILE_SIZE,
                (tile_x + 1) * ROCHE_TILE_SIZE - 1,
                (tile_y + 1) * ROCHE_TILE_SIZE - 1
            ],
            geo_json=cls.new_geo(),
            heatmap_points=[]
        )

    def to_dict(self):
        return {
            'filename': self.filename,
            'bbox': self.bbox
        }


class RocheMarkerType(BaseEnum):
    TUMOR_PLUS = 'tc+'
    TUMOR_MINUS = 'tc-'
    IMMUNE_PLUS = 'im+'
    ARTIFACT = 'ar'
    OTHER_MINUS = 'oth-'


class RocheMarkerGroup(BaseValueObject):

    label: int
    name: str
    description: str
    textgui: str
    locked: bool
    visible: bool
    type: str


class RocheMarkerPreset(BaseValueObject):

    textgui: str
    active: bool
    data: List[RocheMarkerGroup]

    def to_dict(self):
        return {
            'textgui': self.textgui,
            'active': self.active,
            'data': [marker_group.to_dict() for marker_group in self.data]
        }


class RocheMarkerShape(BaseValueObject):

    level: int
    outline_width: int
    size: int
    type: str
    color: str
    label_color: str
    outline_color: str


class RocheHeatMap(BaseValueObject):
    labels: List[dict] = []
    active: bool = True
    locked: bool = False
    level: int = -1
    colorlut: str = 'heat'
    description: str = ''
    gui: str = HER2_ALGORITHM_DISPLAY_ID
    name: str = HER2_ALGORITHM_DISPLAY_ID
    overlay: str = HER2_ALGORITHM_DISPLAY_ID
    visible: bool = True
    level_opacity: List[float] = []


class RochePanel(BaseValueObject):

    name: str
    description: str
    visible: bool = True
    exclusive: bool = False


class RocheALGResult(BaseValueObject):
    ai_results: dict = {}
    wsi_input: Optional[RocheWsiInput] = None
    wsi_scores: Optional[RocheWsiScores] = None
    cells_index_items: List[RocheIndexItem] = []
    marker_presets: List[RocheMarkerPreset] = []
    marker_shapes: Dict[str, RocheMarkerShape] = {}
    heatmaps: List[RocheHeatMap] = []
    panels: List[RochePanel] = []
    thumbnail: Optional[bytearray] = None
    err_msg: Optional[str] = None


class RocheColor(BaseEnum):
    purple: str = '#660099'
    green: str = '#00ff00'
    yellow: str = '#ffff00'
    red: str = '#ff0000'
