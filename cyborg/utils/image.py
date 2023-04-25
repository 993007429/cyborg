from PIL import Image
from PIL.ExifTags import TAGS


def rotate_jpeg(slide_full_path):
    img = Image.open(slide_full_path)
    if hasattr(img, "_getexif"):
        exif_info = img._getexif()
        if exif_info:
            ret = {TAGS.get(key): value for key, value in exif_info.items()}
            orientation = ret.get("Orientation", None)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
            img.save(slide_full_path)
