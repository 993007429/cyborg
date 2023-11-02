import os, sys
import cv2
from PIL import Image
import numpy as np
import math
from .tronslide import open_slide
from ..SlideBase import SlideBase
from threading import Lock

lock = Lock()


class TronSlide(SlideBase):
    def __init__(self, filename):
        self.filename = filename
        self.slide = open_slide(self.filename)
        self.left_bound = self.slide.metadata.content_region['left']
        self.top_bound = self.slide.metadata.content_region['top']
        self.width = self.slide.metadata.content_region['width']
        self.height = self.slide.metadata.content_region['height']
        self.file_max_lvl = self.slide.metadata.maximumLODLevel
        self.file_min_lvl = self.slide.metadata.minimumLODLevel
        try:
            SlideBase.__init__(self)
        except:
            pass

    def read(self, location=[0, 0], size=None, scale=1, greyscale=False):

        if size == None:
            width, height = self.width, self.height
        else:
            width, height = size

        zoom_level = min(max(math.floor(math.log2(scale)), 0), self.file_max_lvl)
        level_ratio = math.pow(2, zoom_level)
        resize_ratio = level_ratio / scale

        crop_start_x, crop_start_y = location
        crop_start_x = crop_start_x + self.left_bound
        crop_start_y = crop_start_y + self.top_bound

        crop_start_x, crop_start_y = math.ceil(min(max(crop_start_x, 0), self.width + self.left_bound)), math.ceil(
            min(max(crop_start_y, 0), self.height + self.top_bound))
        crop_end_x = math.ceil(min(max(width + crop_start_x, 0), self.width + self.left_bound))
        crop_end_y = math.ceil(min(max(height + crop_start_y, 0), self.height + self.top_bound))

        crop_width = math.ceil((crop_end_x - crop_start_x) / level_ratio)
        crop_height = math.ceil((crop_end_y - crop_start_y) / level_ratio)
        with lock:
            crop_region = self.slide.read_region(crop_start_x, crop_start_y, crop_width, crop_height, 0, zoom_level)
            crop_region = np.array(crop_region, dtype=np.uint8)
            crop_region = cv2.resize(crop_region, dsize=None, fx=resize_ratio, fy=resize_ratio)

            if greyscale:
                crop_region = cv2.cvtColor(crop_region, cv2.COLOR_RGB2GRAY)

            # import os
            # from imageio import imsave
            # data_dir = '/mnt/data_alg/tron_test/cell_count'
            # os.makedirs(data_dir, exist_ok=True)
            # imsave(f'{data_dir}/{location[1]}_{location[0]}_{scale}.jpg', crop_region)
            return crop_region

    def save_label(self, path):
        label_info = self.slide.get_named_image_info('label')
        if label_info['length'] > 0:
            label = self.slide.get_named_image('label')
            label.save(path)
        else:
            pass

    def get_thumbnail(self, size=500):
        thumbnail_info = self.slide.get_named_image_info('thumbnail')
        if thumbnail_info['length'] > 0:
            thumbnail = self.slide.get_named_image('thumbnail')
        return thumbnail

    def get_tile(self, x, y, z):
        scale = int(math.pow(2, self.maxlvl - z))
        size = int(1024 * scale)
        tile = self.read([x * size, y * size], [size, size], scale, greyscale=False)
        return Image.fromarray(tile, 'RGB')

    def is_power_of_two(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    @property
    def mpp(self):
        return self.slide.metadata.resolution['horizontal'] if self.slide else 0.242042

    def __del__(self):
        self.slide.close()
