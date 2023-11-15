import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def image_to_8bit_binary_stream(img: Image) -> bytearray:

    # Convert the image to grayscale
    img_gray = img.convert('L')

    # Convert the image to a NumPy array
    img_array = np.array(img_gray, dtype=np.int8)

    # Flatten the 2D array to a 1D array
    flat_array = img_array.flatten()

    # Convert the array to an 8-bit binary stream
    binary_stream = bytearray(flat_array)

    return binary_stream
