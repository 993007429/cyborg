import logging
import os
from io import BytesIO

from cyborg.infra.oss import oss

logger = logging.getLogger(__name__)

ALG_MODEL_PATH = "/data/model/"


def load_alg_model(file_key) -> BytesIO:
    model_path = os.path.join(ALG_MODEL_PATH, file_key)
    if os.path.exists(model_path):
        logger.info(f'load model from local file: {model_path}')
        buffer = BytesIO()
        with open(model_path, "rb") as f:
            buffer.write(f.read())
            buffer.seek(0)
        return buffer
    logger.info(f'load model from oss: {model_path}')
    return oss.get_object_to_io(file_key)
