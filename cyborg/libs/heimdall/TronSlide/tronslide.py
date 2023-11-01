import ctypes
from . import lowlevel
import PIL
import numpy as np


class TileIndex:
    def __init__(self, lodLevel: int, layer: int, row: int, column: int) -> None:
        self.lodLevel = lodLevel
        self.layer = layer
        self.row = row
        self.column = column


class Metadata:
    def __init__(self, handler) -> None:
        self._handler = handler

    @property
    def vendor(self):
        vendor = ctypes.create_string_buffer(256)
        lowlevel.get_vendor(self._handler, vendor, 256)
        return vendor.value.decode()

    @property
    def quick_hash(self):
        slide_hash = ctypes.create_string_buffer(256)
        lowlevel.get_quick_hash(self._handler, slide_hash, 256)
        return slide_hash.value.decode()

    @property
    def resolution(self):
        resolution = lowlevel.get_resolution(self._handler)
        return {
            'horizontal': resolution.horizontal,
            'vertical': resolution.vertical,
        }

    @property
    def name(self):
        name = ctypes.create_string_buffer(512)
        lowlevel.get_name(self._handler, name, 512)
        return name.value.decode()

    @property
    def maximum_zoom_level(self):
        return lowlevel.get_maximum_zoom_level(self._handler)

    @property
    def lod_level_range(self):
        levelRange = lowlevel.get_lod_level_range(self._handler)

        return {
            'minimum': levelRange.minimum,
            'maximum': levelRange.maximum,
        }

    def get_lod_gap_of(self, level):
        return lowlevel.get_lod_gap_of(self._handler, level)

    @property
    def content_region(self):
        contentRegion = lowlevel.get_content_region(self._handler)

        return {
            'left': contentRegion.left,
            'top': contentRegion.top,
            'width': contentRegion.width,
            'height': contentRegion.height,
        }

    @property
    def comments(self):
        comments = ctypes.create_string_buffer(1024)
        lowlevel.get_comments(self._handler, comments, 1024)
        return comments.value.decode()

    @property
    def background_color(self):
        backgroundColor = lowlevel.get_background_color(self._handler)

        return {
            'red': backgroundColor.red,
            'green': backgroundColor.green,
            'blue': backgroundColor.blue,
        }

    @property
    def tile_count(self):
        tileCount = lowlevel.get_tile_count(self._handler)

        return {
            'horizontal', tileCount.horizontal,
            'vertical', tileCount.vertical,
        }

    @property
    def tile_size(self):
        tileSize = lowlevel.get_tile_size(self._handler)

        return {
            'width', tileSize.width,
            'height', tileSize.height,
        }

    @property
    def version(self):
        version = lowlevel.get_version(self._handler)

        return {
            'major', version.major,
            'minor', version.minor,
        }

    @property
    def representative_layer_index(self):
        return lowlevel.get_representative_layer_index(self._handler)

    @property
    def default_layer(self):
        return self.representative_layer_index

    @property
    def minimumLODLevel(self):
        return self.lod_level_range["minimum"]

    @property
    def maximumLODLevel(self):
        return self.lod_level_range["maximum"]


class _TileFuncWrapper:
    def __init__(self,
                 slide,
                 begin,
                 stride,
                 size,
                 scaled_size,
                 row,
                 col):
        self.slide = slide
        self.begin = begin
        self.stride = stride
        self.size = size
        self.scaled_size = scaled_size
        self.col = col
        self.row = row

    def __call__(self):
        x0, y0 = (self.begin + np.array((self.col, self.row))
                  * self.stride).astype(int)
        image = self.slide.read_region(
            x0, y0, self.scaled_size[0], self.scaled_size[1], None, None)
        if image.width != self.size[0] or image.height != self.size[1]:
            image = image.resize(self.size, PIL.Image.ANTIALIAS)
        return {
            'location': [x0, y0],
            'image': image,
            'size': list(self.size),
            'step': [self.col, self.row]
        }


class TronSlide:
    def __init__(self, filename: str) -> None:
        """Open a whole-slide image."""
        self._filename = filename
        self._handler = lowlevel.open(str(filename))

    def close(self):
        lowlevel.close(self._handler)

    @property
    def metadata(self):
        return Metadata(self._handler)

    def get_tile_info(self, index: TileIndex):
        tileInfo = lowlevel.get_tile_image_info(self._handler,
                                                index.lodLevel,
                                                index.layer,
                                                index.row,
                                                index.column)
        if not tileInfo.existed:
            return None
        return {
            "width": tileInfo.width,
            "height": tileInfo.height,
            "length": tileInfo.length,
        }

    def get_tile(self, index: TileIndex):
        tileInfo = self.get_tile_info(index)
        if not tileInfo:
            return None

        length = tileInfo['length']
        w = tileInfo['width']
        h = tileInfo['height']
        buff = ctypes.create_string_buffer(length)
        lowlevel.get_tile_image_data(
            self._handler,
            index.lodLevel,
            index.layer,
            index.row,
            index.column,
            buff)
        return self._load_image(buff, (w, h))

    def read_region(self, x, y, w, h, layer=None, lod_level=None):
        layer = layer if layer else self.metadata.default_layer
        lod = lod_level if lod_level else self.metadata.minimumLODLevel
        w = int(w)
        h = int(h)
        buff = ctypes.create_string_buffer(w * h * 3)
        lowlevel.read_region(self._handler,
                             lod,
                             layer,
                             x, y,
                             w, h,
                             buff)
        return self._load_image(buff, (w, h))

    def _load_image(self, content, size):
        (w, h) = size
        content = ctypes.string_at(content, w*h*3)
        img = np.fromstring(content, np.uint8)
        return PIL.Image.frombuffer('RGB', (w, h),
                                    img, 'raw', 'RGB', 0, 0)

    def get_named_image(self, imageName: str):
        if(not isinstance(imageName, bytes)):
            imageName = imageName.encode()
        imageInfo = self.get_named_image_info(imageName)
        if not imageInfo:
            return None
        length = imageInfo['length']
        w = imageInfo['width']
        h = imageInfo['height']
        buff = ctypes.create_string_buffer(length)
        lowlevel.get_named_image_data(self._handler, imageName, buff)
        return self._load_image(buff, (w, h))

    def get_named_image_info(self, imageName: str):
        if(not isinstance(imageName, bytes)):
            imageName = imageName.encode()

        imageInfo = lowlevel.get_named_image_info(
            self._handler, imageName)
        if not imageInfo.existed:
            return None

        return {
            'width': imageInfo.width,
            'height': imageInfo.height,
            'length': imageInfo.length,
        }

    def enumerate_tiles(self, begin, stop, stride, size):
        pass

    def enumerate_content_tiles(self, stride, size):
        content_region = self.metadata.content_region
        begin = int(content_region['top']), int(content_region['left'])
        stop = begin[0] + int(self.metadata.content_region["width"]), \
            begin[1] + int(self.metadata.content_region["height"])

        scale = [1, 1]
        begin = np.array(begin)
        stop = np.array(stop)
        stride = (np.array(stride) * scale).astype(int)
        scaled_size = (np.array(size) * scale).astype(int)
        step = ((stop - begin - size)/stride+1).astype(int)

        for row in range(step[1]):
            for col in range(step[0]):
                yield(_TileFuncWrapper(self, begin, stride, size, scaled_size, row, col))


def open_slide(filename: str):
    """
    Open a whole-slide or regular image.
    Return an OpenSlide object for whole-slide images and an ImageSlide
    object for other types of images.
    """
    return TronSlide(filename)
