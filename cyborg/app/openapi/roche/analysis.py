import logging

from flask import jsonify, request

from cyborg.app.openapi.roche import roche_blueprint
from cyborg.modules.partner.roche.application.response import RocheAppResponse

logger = logging.getLogger(__name__)


@roche_blueprint.route('/openapi/v1/analysis', methods=['get'])
def analysis():
    algorithm_id = request.form.get('algorithm_id')
    image_url = request.form.get('image_url')
    md5 = request.form.get('md5')
    microns_per_pixel_x = request.form.get('microns_per_pixel_x')
    microns_per_pixel_y = request.form.get('microns_per_pixel_y')
    slide_width = request.form.get('slide_width')
    slide_height = request.form.get('slide_height')
    stain = request.form.get('stain')
    tissue_type = request.form.get('tissue_type')
    clone_type = request.form.get('clone_type')
    slide_type = request.form.get('slide_type')
    indication_type = request.form.get('indication_type')

    logger.info(locals())



    return jsonify()
