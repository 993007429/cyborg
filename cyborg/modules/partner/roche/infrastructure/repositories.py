from typing import Optional, List

from cyborg.consts.her2 import Her2Consts
from cyborg.infra.session import transaction
from cyborg.modules.partner.roche.domain.consts import HER2_ALGORITHM_ID, HER2_ALGORITHM_DISPLAY_ID, \
    HER2_ALGORITHM_NAME, PDL1_ALGORITHM_ID, PDL1_ALGORITHM_DISPLAY_ID, PDL1_ALGORITHM_NAME
from cyborg.modules.partner.roche.domain.value_objects import RocheAlgorithmType, RocheTissueType, RocheAlgorithm, \
    RocheAnnotationType
from cyborg.seedwork.infrastructure.repositories import SQLAlchemyRepository
from cyborg.modules.partner.roche.domain.entities import RocheAITaskEntity, RocheAlgorithmEntity
from cyborg.modules.partner.roche.domain.repositories import RocheRepository
from cyborg.modules.partner.roche.infrastructure.models import RocheAITaskModel


class SQLAlchemyRocheRepository(RocheRepository, SQLAlchemyRepository):

    algorithms = {
        HER2_ALGORITHM_ID: RocheAlgorithm(
            algorithm_id=HER2_ALGORITHM_ID,
            algorithm_display_id=HER2_ALGORITHM_DISPLAY_ID,
            algorithm_name=HER2_ALGORITHM_NAME,
            algorithm_description='Dipath Her2 algorithm',
            algorithm_type=RocheAlgorithmType.IUO.value,
            version_number='1.1.1',
            software_build='4.0.5.123',
            status='Active',
            stain_name='HER-2',
            tissue_types=[
                {'key': RocheTissueType.BREAST.name, 'name': RocheTissueType.BREAST.value}
            ],
            indication_types=[
                {'key': 'MELANOMA', 'name': 'Melanoma'},
                {'key': 'UROTHELIAL_CARCINOMA', 'name': 'Urothelial Carcinoma'}
            ],
            vendor='dipath1',
            # supported_magnification='20',
            supported_mpp_ranges=[[0.1, 1]],
            supported_image_formats=['BIF', 'TIF', 'TIFF'],
            supported_scanners=['VENTANA DP 200', 'VENTANA DP 600'],
            required_slide_types=[],
            roi_analysis_support=True,
            primary_analysis_overlay_display=True,
            provides_primary_analysis_score=True,
            manual_score_mode='INCLUSIVE',  # "EXCLUSIVE"
            provides_navigational_heatmap_thumbnail=True,
            # 'clone_type=['SP142', 'SP263'],
            secondary_analysis_support=True,
            secondary_analysis_annotation_type=RocheAnnotationType.INCLUSION.value,
            max_secondary_analysis_allowed=3,
            # 'overlay_acceptance_required=False,
            # 'slide_score_acceptance_required=False,
            # 'requires_analysis_rejection_feedback=False,
            # 'provides_prognostic_score=False,
            results_parameters=[{
                'name': 'HER-2 Level',
                'key': 'her2_level',
                'data_type': 'string',
                'primary_display': True
            }, *[{
                'name': cell_type,
                'key': cell_type,
                'data_type': 'string',
                'primary_display': True
            } for cell_type in Her2Consts.display_cell_types]]
        ),
        PDL1_ALGORITHM_ID: RocheAlgorithm(
            algorithm_id=PDL1_ALGORITHM_ID,
            algorithm_display_id=PDL1_ALGORITHM_DISPLAY_ID,
            algorithm_name=PDL1_ALGORITHM_NAME,
            algorithm_description='Dipath PD-L1 algorithm',
            algorithm_type=RocheAlgorithmType.IUO.value,
            version_number='1.1.1',
            software_build='4.0.5.123',
            status='Active',
            stain_name='PD-L1',
            tissue_types=[
                {'key': RocheTissueType.LIVER.name, 'name': RocheTissueType.LIVER.value},
                {'key': RocheTissueType.LUNG.name, 'name': RocheTissueType.LUNG.value},
            ],
            indication_types=[
                {'key': 'MELANOMA', 'name': 'Melanoma'},
                {'key': 'UROTHELIAL_CARCINOMA', 'name': 'Urothelial Carcinoma'}
            ],
            vendor='dipath1',
            # supported_magnification='20',
            supported_mpp_ranges=[[0.1, 1]],
            supported_image_formats=['BIF', 'TIF', 'TIFF'],
            supported_scanners=['VENTANA DP 200', 'VENTANA DP 600'],
            required_slide_types=[],
            roi_analysis_support=True,
            primary_analysis_overlay_display=True,
            provides_primary_analysis_score=True,
            manual_score_mode='INCLUSIVE',  # "EXCLUSIVE"
            provides_navigational_heatmap_thumbnail=True,
            # 'clone_type=['SP142', 'SP263'],
            secondary_analysis_support=True,
            secondary_analysis_annotation_type=RocheAnnotationType.INCLUSION.value,
            max_secondary_analysis_allowed=3,
            # 'overlay_acceptance_required=False,
            # 'slide_score_acceptance_required=False,
            # 'requires_analysis_rejection_feedback=False,
            # 'provides_prognostic_score=False,
            results_parameters=[{
                'name': 'tps',
                'key': 'tps',
                'data_type': 'string',
                'primary_display': True
            }]
        )
    }

    def get_algorithms(self) -> List[RocheAlgorithmEntity]:
        return [RocheAlgorithmEntity.from_dict(algo.to_dict()) for algo in self.algorithms.values()]

    def get_algorithm(self, algorithm_id: str) -> Optional[RocheAlgorithmEntity]:
        algo = self.algorithms.get(algorithm_id)
        return RocheAlgorithmEntity.from_dict(algo.to_dict()) if algo else None

    @transaction
    def save_ai_task(self, ai_task: RocheAITaskEntity) -> bool:
        model = self.convert_to_model(ai_task, RocheAITaskModel)
        if not model:
            return False
        self.session.add(model)
        self.session.flush([model])
        ai_task.update_data(**model.raw_data)
        return True

    def get_ai_task_by_id(self, task_id: int) -> Optional[RocheAITaskEntity]:
        model = self.session.get(RocheAITaskModel, task_id)
        return RocheAITaskEntity.from_dict(model.raw_data) if model else None

    def get_ai_task_by_analysis_id(self, analysis_id: str) -> Optional[RocheAITaskEntity]:
        model = self.session.query(RocheAITaskModel).filter_by(analysis_id=analysis_id).one_or_none()
        return RocheAITaskEntity.from_dict(model.raw_data) if model else None
