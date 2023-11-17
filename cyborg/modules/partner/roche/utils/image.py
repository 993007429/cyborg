import io
from PIL import Image


def convert_image_to_png_data(img: Image) -> bytearray:

    img_rgba = img.convert('RGBA')

    png_data = io.BytesIO()

    img_rgba.save(png_data, format='PNG')

    return bytearray(png_data.getvalue())
