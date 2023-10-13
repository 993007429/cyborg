import logging

from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown, \
    NVMLError

logger = logging.getLogger(__name__)


def get_gpu_status():
    gpu_status = {}
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            ith_gpu_handle = nvmlDeviceGetHandleByIndex(i)
            mem = nvmlDeviceGetMemoryInfo(ith_gpu_handle)
            gpu_status[str(i)] = {'total': mem.total >> 20, 'used': mem.used >> 20, 'free': mem.free >> 20}
        nvmlShutdown()
    except NVMLError:
        logger.warning('no nvidia gpu found!')

    return gpu_status
