import time


def get_time_now():
    """
    获取当前时间戳 单位：ms
    """
    return int(round((time.time()) * 1000))


def ms_to_hours(millis):
    seconds, milliseconds = divmod(millis, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds
