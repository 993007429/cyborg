import os
from configparser import RawConfigParser


def get_local_settings(file_path: str):
    conf = RawConfigParser()
    conf.read(file_path)
    return conf


class Settings(object):

    """默认配置"""

    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ENV = os.environ.get('CYBORG_ENV', 'DEV').upper()

    LOCAL_SETTINGS = get_local_settings(f'{PROJECT_DIR}/local_settings/cyborg-{ENV.lower()}.ini')

    LOG_LEVEL = 'INFO'

    # 设置session 秘钥
    # 可以通过 base64.b64encode(os.urandom(48)) 来生成一个指定长度的随机字符串
    SECRET_KEY = 'CF3tEA1J3hRyIOw3PWE3ZE9+hLOcUDq6acX/mABsEMTXNjRDm5YldRLIXazQviwP'

    JWT_SECRET = 'mysalt'

    PORT = int(os.environ.get('CYBORG_PORT', '8080'))

    COVER_RESULT = False  # 复核结果覆盖ai计算结果设为True，反之为False

    INTEGRITY_CHECK = True  # 高通量上传切片完整性校验，开启为True，关闭为False

    MAX_AREA = 1200000  # 多选框最大支持面积，此值关系到多选返回标注数量的速度，值越大可操作范围越大，响应会变慢，不建议更改此值(单位：平方微米)

    DATA_DIR = LOCAL_SETTINGS['default']['data_dir']

    LOG_DIR = LOCAL_SETTINGS['default']['log_dir']

    APP_LOG_FILE = os.path.join(LOG_DIR, f'cyborg-app-{PORT}')

    WHITE_LIST = [
        '/aipath/api/user/login',
        '/aipath/api/files/slice',
        '/aipath/api/files/getInfo',
        '/aipath/api/files/thumbnail',
        '/aipath/api/ai/inform',
        '/aipath/api/ai/connect',
        '/aipath/api/files/downloadTemplate',
        '/aipath/api/ai/cailibrateInform',
    ]

    LIMIT_URL = [
        'slice/createMark',
        'slice/getMarks',
        'slice/modifyMark',
        'slice/createGroup',
        'slice/selectGroup',
        'slice/modifyGroup',
        'slice/markShow',
        'slice/selectTemplate'
    ]

    MEM_PER_GPU = 12

    THUMBNAIL_BOUNDING = 500

    BLOCK_SIZE = 1024

    # last_show_groups，存储group_id，涉及到的组包含的标注会在前端图层最底层显示
    LAST_SHOW_GROUPS = [266, ]

    # 是否为公有云版本，默认为私有云版本
    CLOUD = False

    # mysql配置
    user = 'root'
    password = 'dyj123'
    host = '127.0.0.1'
    port = 3306
    database = 'dipath'
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://{}:{}@{}:{}/{}'.format(user, password, host, port, database)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False

    # redis配置
    REDIS_HOST = '127.0.0.1'
    REDIS_PORT = 6379
    LOCK_DB = 4  # 分布式锁所在db
    LOCK_EXPIRATION_TIME = 10  # 分布式锁过期时间

    # CACHE_REDIS_URI = 'redis://{}:{}'.format(REDIS_HOST, str(REDIS_PORT))
    CACHE_REDIS_URI = ''

    # celery配置
    CELERY_BROKER_URL = 'redis://{}:{}/{}'.format(REDIS_HOST, str(REDIS_PORT), LOCAL_SETTINGS['celery']['broker_db'])
    CELERY_BACKEND_URL = 'redis://{}:{}/{}'.format(REDIS_HOST, str(REDIS_PORT), LOCAL_SETTINGS['celery']['backend_db'])

    MINIO_ACCESS_KEY = LOCAL_SETTINGS['minio']['access_key']
    MINIO_ACCESS_SECRET = LOCAL_SETTINGS['minio']['access_secret']
    PRIVATE_ENDPOINT = LOCAL_SETTINGS['minio']['private_endpoint']
    PUBLIC_ENDPOINT = LOCAL_SETTINGS['minio']['public_endpoint']
    BUCKET_NAME = LOCAL_SETTINGS['minio']['bucket_name']

    IMAGE_SERVER = LOCAL_SETTINGS['default']['image_server']

    REPORT_SERVER = LOCAL_SETTINGS['default']['report_server']

    ELECTRON_UPLOAD_SERVER = 'http://{}:3000/download'

    # 需要记录操作日志的算法模块
    ai_log_list = ['tct', 'lct', 'pdl1', 'human_tl']

    # 版本号（用于算法用量统计）
    VERSION = '4.0.0'
