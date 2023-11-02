from ctypes import (
    POINTER,
    byref,
    c_bool,
    c_ubyte,
    c_float,
    c_char_p,
    c_double,
    c_int32,
    c_int64,
    c_size_t,
    c_uint32,
    c_void_p,
    cdll,
    Structure
)
from itertools import count
import sys
import os

if sys.platform == 'win32':
    raise NotImplementedError('该格式暂不支持windows系统')
elif sys.platform == 'linux':
    _lib = cdll.LoadLibrary('libtronc.so')


class OpenSlideError(Exception):
    """An error produced by the OpenSlide library.
    Import this from openslide rather than from openslide.lowlevel.
    """


class OpenSlideVersionError(OpenSlideError):
    """This version of OpenSlide does not support the requested functionality.
    Import this from openslide rather than from openslide.lowlevel.
    """

    def __init__(self, minimum_version):
        super().__init__(f'OpenSlide >= {minimum_version} required')
        self.minimum_version = minimum_version


class OpenSlideUnsupportedFormatError(OpenSlideError):
    """OpenSlide does not support the requested file.
    Import this from openslide rather than from openslide.lowlevel.
    """


class _OpenSlide:
    """Wrapper class to make sure we correctly pass an OpenSlide handle."""

    def __init__(self, ptr):
        self._as_parameter_ = ptr
        self._valid = True
        # Retain a reference to close() to avoid GC problems during
        # interpreter shutdown
        self._close = close

    def __del__(self):
        if self._valid:
            self._close(self)

    def invalidate(self):
        self._valid = False

    @classmethod
    def from_param(cls, obj):
        if obj.__class__ != cls:
            raise ValueError("Not an OpenSlide reference")
        if not obj._as_parameter_:
            raise ValueError("Passing undefined slide object")
        if not obj._valid:
            raise ValueError("Passing closed slide object")
        return obj


class _utf8_p:
    """Wrapper class to convert string arguments to bytes."""

    @classmethod
    def from_param(cls, obj):
        if isinstance(obj, bytes):
            return obj
        elif isinstance(obj, str):
            return obj.encode('UTF-8')
        else:
            raise TypeError('Incorrect type')

# check for errors opening an image file and wrap the resulting handle


def _check_open(result, _func, _args):
    if result is None:
        raise OpenSlideUnsupportedFormatError(
            "Unsupported or missing image file")
    slide = _OpenSlide(c_void_p(result))
    err = get_error(slide)
    if err is not None:
        raise OpenSlideError(err)
    return slide


# prevent further operations on slide handle after it is closed
def _check_close(_result, _func, args):
    args[0].invalidate()


# Convert returned byte array, if present, into a string
def _check_string(result, func, _args):
    if func.restype is c_char_p and result is not None:
        return result.decode('UTF-8', 'replace')
    else:
        return result

# check if the library got into an error state after each library call


def _check_error(result, func, args):
    err = get_error(args[0])
    if err is not None:
        raise OpenSlideError(err)
    return _check_string(result, func, args)


# Convert returned NULL-terminated char** into a list of strings
def _check_name_list(result, func, args):
    _check_error(result, func, args)
    names = []
    for i in count():
        name = result[i]
        if not name:
            break
        names.append(name.decode('UTF-8', 'replace'))
    return names


# resolve and return an OpenSlide function with the specified properties
def _func(name, restype, argtypes, errcheck=_check_error, minimum_version=None):
    try:
        func = getattr(_lib, name)
    except AttributeError:
        if minimum_version is None:
            raise

        # optional function doesn't exist; fail at runtime
        def function_unavailable(*_args):
            raise OpenSlideVersionError(minimum_version)

        return function_unavailable
    func.argtypes = argtypes
    func.restype = restype
    if errcheck is not None:
        func.errcheck = errcheck
    return func


class TronBackgroundColor(Structure):
    """Represents the background color of a tron slide."""
    _fields_ = [
        ('red', c_ubyte),
        ('green', c_ubyte),
        ('blue', c_ubyte),
    ]


class TronContentRegion(Structure):
    """Represents the background color of a tron slide."""
    _fields_ = [
        ('left', c_int32),
        ('top', c_int32),
        ('width', c_int32),
        ('height', c_int32),
    ]


class TronLodLevelRange(Structure):
    """Represents the minimum and maximum LOD level of a tron slide."""
    _fields_ = [
        ('minimum', c_int32),
        ('maximum', c_int32),
    ]


class TronImageInfo(Structure):
    """Represents the dimensions of an image."""
    _fields_ = [
        ('existed', c_bool),
        ('width', c_size_t),
        ('height', c_size_t),
        ('length', c_size_t),
    ]


class TronResolution(Structure):
    """Represents the resolution information of a tron slide."""
    _fields_ = [
        ('horizontal', c_float),
        ('vertical', c_float),
    ]


class TronTileCount(Structure):
    """Represents the tile count information of a tron slide."""
    _fields_ = [
        ('horizontal', c_int32),
        ('vertical', c_int32),
    ]


class TronTileSize(Structure):
    """Represents the size of a tile."""
    _fields_ = [
        ('width', c_int32),
        ('height', c_int32),
    ]


class TronVersion(Structure):
    """Represents the version of a tron slide."""
    _fields_ = [
        ('major', c_int32),
        ('minor', c_int32),
    ]


open = _func('tron_open', c_void_p, [_utf8_p], _check_open)

close = _func('tron_close', None, [_OpenSlide], _check_close)

get_error = _func('tron_get_last_error', c_char_p, [_OpenSlide], _check_string)

get_vendor = _func('tron_get_vendor', c_size_t, [
    _OpenSlide, c_char_p, c_size_t])

get_quick_hash = _func("tron_get_quick_hash", c_size_t, [
                       _OpenSlide, c_char_p, c_size_t])

get_resolution = _func(
    'tron_get_resolution', TronResolution, [_OpenSlide])

get_name = _func('tron_get_name', c_size_t, [_OpenSlide, c_char_p, c_size_t])

get_maximum_zoom_level = _func(
    'tron_get_maximum_zoom_level', c_float, [_OpenSlide])

get_lod_level_range = _func(
    'tron_get_lod_level_range', TronLodLevelRange, [_OpenSlide])

get_lod_gap_of = _func('tron_get_lod_gap_of', c_float, [
    _OpenSlide, c_size_t])


get_content_region = _func(
    'tron_get_content_region', TronContentRegion, [_OpenSlide])

get_comments = _func('tron_get_comments', c_size_t, [
    _OpenSlide, c_char_p, c_size_t])


get_background_color = _func(
    'tron_get_background_color', TronBackgroundColor, [_OpenSlide])

get_named_image_data = _func('tron_get_named_image_data', c_size_t, [
    _OpenSlide, c_char_p, c_char_p])

get_named_image_info = _func(
    'tron_get_named_image_info', TronImageInfo, [_OpenSlide, c_char_p])

get_tile_count = _func(
    'tron_get_tile_count', TronTileCount, [_OpenSlide])
get_tile_image_data = _func('tron_get_tile_image_data', c_size_t,
                            [_OpenSlide, c_int32, c_int32, c_int32, c_int32, c_char_p])

get_tile_image_info = _func('tron_get_tile_image_info', TronImageInfo,
                            [_OpenSlide, c_int32, c_int32, c_int32, c_int32])
get_tile_size = _func(
    'tron_get_tile_size', TronTileSize, [_OpenSlide])
get_version = _func('tron_get_version', TronVersion, [_OpenSlide])

get_representative_layer_index = _func(
    'tron_get_representative_layer_index', c_int32, [_OpenSlide])


read_region = _func('tron_read_region', c_size_t,
                    [_OpenSlide, c_int32, c_int32, c_int32, c_int32, c_size_t, c_size_t, c_char_p])
