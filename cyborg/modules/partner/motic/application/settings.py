from cyborg.app.settings import Settings


class MoticSettings(object):

    """默认配置"""

    SETTINGS = Settings.LOCAL_SETTINGS['motic'] if 'motic' in Settings.LOCAL_SETTINGS else None
    API_ACCESS_KEY = SETTINGS['access_key'] if SETTINGS else ''
    API_ACCESS_SECRET = SETTINGS['access_secret'] if SETTINGS else ''
