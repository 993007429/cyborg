import os
from io import BytesIO

from cyborg.infra.oss import oss

ALG_MODEL_PATH = "/data/model/"


def load_alg_model(file_key) -> BytesIO:
    model_path = os.path.join(ALG_MODEL_PATH, file_key)
    if os.path.exists(model_path):
        buffer = BytesIO()
        with open(model_path, "rb") as f:
            buffer.write(f.read())
            buffer.seek(0)
        return buffer
    return oss.get_object_to_io(file_key)
