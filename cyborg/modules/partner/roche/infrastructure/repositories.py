from typing import Optional, List

from cyborg.infra.session import transaction
from cyborg.modules.partner.roche.domain.value_objects import RocheAlgorithmType, RocheTissueType, RocheAlgorithm
from cyborg.seedwork.infrastructure.repositories import SQLAlchemyRepository
from cyborg.modules.partner.roche.domain.entities import RocheAITaskEntity, RocheAlgorithmEntity
from cyborg.modules.partner.roche.domain.repositories import RocheRepository
from cyborg.modules.partner.roche.infrastructure.models import RocheAITaskModel


class SQLAlchemyRocheRepository(RocheRepository, SQLAlchemyRepository):

    algorithms = {
        '8bacbdb5-46d7-1578-11a0-00e2a7b7a9d1': RocheAlgorithm(
            algorithm_id='8bacbdb5-46d7-1578-11a0-00e2a7b7a9d1',
            algorithm_display_id='123456',
            algorithm_name='her2',
            algorithm_description='Dipath Her2 algorithm',
            algorithm_type=RocheAlgorithmType.IUO.value,
            version_number='1.1.0',
            software_build='4.0.3.123',
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
            supported_magnification='40',
            supported_image_formats=['BIF', 'TIF', 'TIFF'],
            supported_scanners=['VENTANA DP 200', 'VENTANA DP 600'],
            required_slide_types=['HER-2'],
            roi_analysis_support=False,
            primary_analysis_overlay_display=True,
            provides_primary_analysis_score=True,
            manual_score_mode='INCLUSIVE',  # "EXCLUSIVE"
            # 'clone_type=['SP142', 'SP263'],
            # 'supported_mpp_ranges=[[0, 1]],
            # 'secondary_analysis_support =True,
            # 'secondary_analysis_annotation_type=RocheAnnotationType.INCLUSION.value,
            # 'max_secondary_analysis_allowed=0,
            # 'overlay_acceptance_required=False,
            # 'slide_score_acceptance_required=False,
            # 'requires_analysis_rejection_feedback=False,
            # 'provides_prognostic_score=False,
            results_parameters=[{
                'name': 'HER-2 Level',
                'key': 'her2_level',
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
