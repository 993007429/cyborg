import os
import sys

# LRU算法直接返回对象 加快读取切片效率
from .cache import LRUCache

slides = LRUCache()


def open_slide(filename):
    ext = os.path.splitext(filename)[1][1:].lower()

    if filename in slides:
        return slides[filename]

    if ext == 'kfb':  # 宁波江丰
        from .KfbSlide import KfbSlide
        slide = KfbSlide(filename)
    elif ext == 'sdpc':  # 深圳生强
        from .SdpcSlide import SdpcSlide
        slide = SdpcSlide(filename)
    elif ext == 'mdsx':  # 麦克奥迪
        from .MdsxSlide import MdsxSlide
        slide = MdsxSlide(filename)
    elif ext == 'hdx':  # 海德星5片机
        from .HdxSlide import HdxSlide
        slide = HdxSlide(filename)
    # 志盈linux版本还有些问题，后期改完再统一
    elif ext == 'zyp' and sys.platform == 'win32':  # 志盈
        from .ZYPSlide import ZYPSlide
        slide = ZYPSlide(filename)
    elif ext == 'tmap':  # 优纳
        from .TmapSlide import TmapSlide
        slide = TmapSlide(filename)
    elif ext == 'czi':  # 蔡司
        from .CziSlide import CziSlide
        slide = CziSlide(filename)
    elif ext == 'ibl':  # 爱病理
        from .IblSlide import IblSlide
        slide = IblSlide(filename)
    elif ext == 'tron':  # 英特美迪
        from .TronSlide import TronSlide
        slide = TronSlide(filename)
    else:  # openslide支持的切片格式
        from .OtherSlide import OtherSlide
        slide = OtherSlide(filename)

    slides[filename] = slide

    return slide
