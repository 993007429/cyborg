import abc
import math
from typing import Optional, Union, Tuple

import cv2
import numpy as np
from PIL import Image

from .consts import DEFAULT_MPP


class SlideBase(object):
    def __init__(self):
        mx = max(self.width, self.height)
        self.maxlvl = math.ceil(math.log2(mx))

    @abc.abstractmethod
    def read(
            self, location: Optional[Tuple[int, int]] = None, size: Tuple[Union[int, float], Union[int, float]] = None,
            scale: float = 1.0, greyscale: bool = False):
        """
        :param location: (x, y) at level=0
        :param size: (width, height)
        :param scale: downsampling ratio
        :param greyscale: if True, convert image to greyscale
        :return: a numpy image,  np_img.shape=[height/scale, width/scale, channel=1 or 3]
        """
        ...

    def get_tile(self, x, y, z):
        scale = math.pow(2, self.maxlvl - z)
        r = 1024*scale
        tile = self.read((x * r, y * r), (r, r), scale, greyscale=False)
        return Image.fromarray(tile, mode='RGB')

    def get_window(self, xindex, yindex, window_size=None, overlap=None, scale=1, padding=True, bbox=None):
        window_size = window_size or [100, 100]
        overlap = overlap or [50, 50]
        if bbox is None:
            x_min, y_min, x_max, y_max = 0, 0, self.width, self.height
        else:
            x_min, y_min, x_max, y_max = bbox

        window_w, window_h = window_size
        overlap_w, overlap_h = overlap

        window_w *= scale
        window_h *= scale
        overlap_w *= scale
        overlap_h *= scale

        stride_w, stride_h = window_w - overlap_w, window_h - overlap_h

        crop_start_x = x_min + xindex*stride_w
        crop_start_y = y_min + yindex*stride_h

        img = self.read((crop_start_x, crop_start_y), (window_w, window_h), scale)
        if padding:
            img = pad_img(img, window_size)

        return img

    def get_slide_window_info(self, standard_mpp, window_size=None, overlap=None, cut_leftover=None, bbox=None):

        """compute resize scale, number of rows and columns for sliding window
        :param standard_mpp: mpp of training dataset
        :param window_size: slide window size, order is in [width, height]
        :param overlap: overlaps between two adjacent window, order is in [width, height]
        :param cut_leftover: omit the leftover if leftover <= cut_leftover,  [width, height]
        :param bbox: box area to run the slide window, order is in [x_min, y_min, x_max, y_max]
        :return:
                scale: dowmsampling ratio
                (num_x, num_y):
                    num_x: number of windows in horizontal direction
                    num_y: number of windows in vertical direction
        """

        window_size = window_size or [512, 512]
        overlap = overlap or [128, 128]
        cut_leftover = cut_leftover or [0, 0]

        if standard_mpp is None or self.mpp is None:
            scale = 1
        else:
            scale = standard_mpp/self.mpp

        if bbox is None:
            x_min, y_min, x_max, y_max = 0, 0, self.width, self.height
        else:
            x_min, y_min, x_max, y_max = bbox

        height, width = y_max - y_min, x_max - x_min

        window_w, window_h = window_size
        overlap_w, overlap_h = overlap
        cut_leftover_w, cut_leftover_h = cut_leftover

        window_w *= scale
        window_h *= scale
        overlap_w *= scale
        overlap_h *= scale
        cut_leftover_h *= scale
        cut_leftover_w *= scale

        stride_h, stride_w = window_h-overlap_h, window_w-overlap_w

        num_x, num_y = 1 + math.floor((width - window_w) / stride_w), 1 + math.floor((height - window_h) / stride_h)
        num_x, num_y = max(num_x, 1), max(num_y, 1)

        leftover_w = width - window_w - (num_x-1)*stride_w
        leftover_h = height - window_h - (num_y-1)*stride_h

        if leftover_w > cut_leftover_w:
            num_x += 1
        if leftover_h > cut_leftover_h:
            num_y += 1

        return scale, (num_x, num_y)

    def get_thumbnail(self, size=500):
        try:
            k = self.slide.associated_images.keys()
            if 'thumbnail' in k:
                thumbnail_img = self.slide.associated_images[k]
            else:
                raise Exception
        except Exception:
            maxSize = max(self.height, self.width)
            scale_ratio = maxSize / size
            np_thumb = self.read(location=(0, 0), size=(self.width, self.height), scale=scale_ratio)
            thumbnail_img = Image.fromarray(np_thumb, mode='RGB')
                
        if thumbnail_img:
            if thumbnail_img.mode == 'RGBA':
                thumbnail_img = thumbnail_img.convert('RGB')

        return thumbnail_img

    def get_roi(self, roi, is_bounding_resize=True, standard_mpp=None):
        max_x, max_y = self.width, self.height
        sx, sy = list(map(int, roi[0]))
        ex, ey = list(map(int, roi[1]))
        # make sure roi is within the slide
        sx, sy, ex, ey = max(sx, 0), max(sy, 0), max(ex, 0), max(ey, 0)
        sx, sy, ex, ey = min(sx, max_x), min(sy, max_y), min(ex, max_x), min(ey, max_y)
        sx, sy, ex, ey = math.ceil(sx), math.ceil(sy), math.ceil(ex), math.ceil(ey)

        roi_w, roi_h = int(abs(ex - sx)), int(abs(ey - sy))
        scale_ratio = max(roi_w, roi_h) / 200 if is_bounding_resize else 1.0
        if standard_mpp:
            this_mpp = self.mpp if self.mpp is not None else 0.242042
            scale_ratio = float(this_mpp / standard_mpp) * scale_ratio if standard_mpp else scale_ratio

        _size = (max(roi_w, roi_h), max(roi_w, roi_h))
        _location = (sx + roi_w // 2 - _size[0] // 2, sy + roi_h // 2 - _size[1] // 2)

        img_np = self.read(location=_location, size=_size, scale=scale_ratio)

        roi_img = Image.fromarray(img_np)
        if roi_img.mode == 'RGBA':
            roi_img = roi_img.convert('RGB')

        return roi_img

    def get_roi_and_segment(self, roi, DI, standard_mpp=None):
        # 根据DI决定颜色
        if DI >= 2.5:
            selected_color = [255, 0, 0]  # 红色
        elif 1.25 <= DI < 2.5:
            selected_color = [255, 165, 0]  # 橘黄色
        else:
            selected_color = [0, 128, 0]  # 绿色
        # 从slide中获取ROI
        max_x, max_y = self.width, self.height
        sx, sy = list(map(int, roi[0]))
        ex, ey = list(map(int, roi[1]))

        # 确保roi在slide内
        sx, sy, ex, ey = max(sx, 0), max(sy, 0), max(ex, 0), max(ey, 0)
        sx, sy, ex, ey = min(sx, max_x), min(sy, max_y), min(ex, max_x), min(ey, max_y)
        sx, sy, ex, ey = math.ceil(sx), math.ceil(sy), math.ceil(ex), math.ceil(ey)

        _size = (max(ex - sx, ey - sy), max(ex - sx, ey - sy))
        _location = (sx + (ex - sx) // 2 - _size[0] // 2, sy + (ey - sy) // 2 - _size[1] // 2)

        img_np = self.read(location=_location, size=_size)
        lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2Lab)
        value_channel = lab[:, :, 0]

        pixels = value_channel.reshape((-1, 1))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.05)

        k = 2
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)

        nucleus_label = 0 if centers[0] < centers[1] else 1
        mask = np.zeros_like(value_channel)
        mask[labels.reshape(value_channel.shape) == nucleus_label] = 255

        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_iou = 0
        final_mask = np.zeros_like(mask)
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                temp_mask = np.zeros_like(mask)
                cv2.ellipse(temp_mask, ellipse, 255, -1)
                intersection = np.logical_and(mask, temp_mask)
                union = np.logical_or(mask, temp_mask)
                iou = np.sum(intersection) / np.sum(union)
                if iou > best_iou:
                    best_iou = iou
                    final_mask = temp_mask
        final_mask = np.logical_and(mask, final_mask)
        contours_final, _ = cv2.findContours(final_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
        if contours_final:
            contour_final = contours_final[0]
            # 获取最佳椭圆的中心和主轴
            center, axes, angle = cv2.fitEllipse(contour_final)
            # 重新计算主轴长度为原来的90%
            axes = (axes[0] * 0.9, axes[1] * 0.9)

            # 使用新的主轴和原始中心，画一个新的椭圆并填充
            smaller_ellipse_mask = np.zeros_like(mask)
            cv2.ellipse(smaller_ellipse_mask, (center, axes, angle), 255, -1)

            # 使用逻辑或将新的椭圆mask与final_mask合并
            final_mask = np.logical_or(final_mask, smaller_ellipse_mask)

        roi_img = Image.fromarray(img_np)
        if roi_img.mode == 'RGBA':
            roi_img = roi_img.convert('RGB')

        height, width = final_mask.shape
        white_background = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 将final_mask的目标区域设置为指定颜色
        white_background[final_mask == 1] = selected_color

        # cv2.imwrite('1.jpg', white_background)
        white_background = Image.fromarray(white_background)

        return roi_img, white_background

    @abc.abstractmethod
    def save_label(self, path: str):
        ...

    @property
    def mpp(self):
        return 0.242042

    def to_dict(self):
        return {
            'width': self.width,
            'height': self.height,
            'max_level': self.maxlvl,
            'mpp': self.mpp or DEFAULT_MPP
        }


def pad_img(img, pad_size=None):
    pad_size = pad_size or (512, 512)
    if img.shape[0:2] == pad_size:
        return img
    else:
        new_img = np.zeros((pad_size[0], pad_size[1], img.shape[2]))
        new_img[:img.shape[0], :img.shape[1], :] = img
        return new_img
