import logging

from flask import jsonify, request

from cyborg.app.openapi.roche import roche_blueprint
from cyborg.modules.partner.roche.application.response import RocheAppResponse
from cyborg.modules.partner.roche.domain.value_objects import RocheAlgorithmType, RocheTissueType

logger = logging.getLogger(__name__)

ALGORITHMS = {
    '8bacbdb5-46d7-1578-11a0-00e2a7b7a9d1': {
        'algorithm_id': '8bacbdb5-46d7-1578-11a0-00e2a7b7a9d1',
        'algorithm_name': 'Her2',
        'algorithm_description': 'Dipath Her2 algorithm',
        'algorithm_type': RocheAlgorithmType.IUO.value,
        'version_number': '1.1.0',
        'status': 'Active',
        'stain_name': 'HER-2',
        'tissue_types': [RocheTissueType.BREAST.value],
        'indication_types':[{'key': 'MELANOMA','name': 'Melanoma'}, {'key': 'UROTHELIAL_CARCINOMA','name': 'Urothelial Carcinoma'}],
        'vendor': 'dipath',
        # 'clone_type': ['SP142', 'SP263'],
        'supported_magnification': '40',
        # 'supported_mpp_ranges': [[0, 1]],
        'supported_image_formats': ['BIF', 'TIF', 'TIFF', 'SVS', 'KFB'],
        'supported_scanners': ['VENTANA DP 200', 'VENTANA DP 600'],
        'required_slide_types': ['PD-L1'], 
        'roi_analysis_support': False,
        'primary_analysis_overlay_display': True,
        'provides_primary_analysis_score': True,
        # 'secondary_analysis_support':True,
        # 'secondary_analysis_annotation_type': RocheAnnotationType.INCLUSION.value,
        # 'max_secondary_analysis_allowed': 0,
        'manual_score_mode': 'INCLUSIVE',  # "EXCLUSIVE"
        # 'overlay_acceptance_required': False,
        # 'slide_score_acceptance_required': False,
        # 'requires_analysis_rejection_feedback': False,
        # 'provides_prognostic_score': False,
        'results_parameters': [{
            'name': 'HER-2 Level',
            'key': 'her2_level',
            'data_type': 'string',
            'primary_display': True
        }]
    }
}


@roche_blueprint.route('/openapi/v1/algorithms/<string:algorithm_id>', methods=['get'])
def get_algorithm_detail(algorithm_id: str):
    data = ALGORITHMS.get(algorithm_id)
    res = RocheAppResponse(data=data)
    return jsonify(res.dict())


@roche_blueprint.route('/openapi/v1/algorithms', methods=['get'])
def get_algorithms():
    locale = request.args.get('locale')
    logger.info(f'locale: {locale}')
    data = list(ALGORITHMS.values())
    res = RocheAppResponse(data=data)
    return jsonify(res.dict())
