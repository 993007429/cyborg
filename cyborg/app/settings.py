import os
from configparser import RawConfigParser


def get_local_settings(file_path: str):
    conf = RawConfigParser()
    conf.read(file_path)
    return conf


class Settings(object):

    """默认配置"""

    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ENV = os.environ.get('CYBORG_ENV', 'default').upper()

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
        '/aipath/api/user/sign2',
        '/aipath/api/files/slice',
        '/aipath/api/files/getInfo',
        '/aipath/api/files/thumbnail',
        '/aipath/api/files/ROI2',
        '/aipath/api/files/downloadTemplate',
        '/aipath/api/files/getImage',
        '/aipath/api/ai/inform',
        '/aipath/api/ai/connect',
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

    THUMBNAIL_BOUNDING = 500

    BLOCK_SIZE = 1024

    # last_show_groups，存储group_id，涉及到的组包含的标注会在前端图层最底层显示
    LAST_SHOW_GROUPS = [266, ]

    # 是否为公有云版本，默认为私有云版本
    CLOUD = False

    # mysql配置
    MYSQL_USER = os.environ.get('mysql.user') or LOCAL_SETTINGS['mysql']['user']
    MYSQL_PASSWORD = os.environ.get('mysql.password') or LOCAL_SETTINGS['mysql']['password']
    MYSQL_HOST = os.environ.get('mysql.host') or LOCAL_SETTINGS['mysql']['host']
    MYSQL_PORT = os.environ.get('mysql.port') or LOCAL_SETTINGS['mysql']['port']
    MYSQL_DATABASE = 'dipath'
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://{}:{}@{}:{}/{}'.format(
        MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False

    # redis配置
    REDIS_HOST = os.environ.get('redis.host') or LOCAL_SETTINGS['redis']['host']
    REDIS_PORT = os.environ.get('redis.port') or LOCAL_SETTINGS['redis']['port']

    REDLOCK_CONNECTION_CONFIG = [{
        'host': REDIS_HOST,
        'port': REDIS_PORT,
        'db': 4,
        'password': None
    }]

    LOCK_EXPIRATION_TIME = 10  # 分布式锁过期时间

    CACHE_REDIS_URI = 'redis://{}:{}'.format(REDIS_HOST, str(REDIS_PORT))

    # celery配置
    CELERY_BROKER_URL = 'redis://{}:{}/{}'.format(REDIS_HOST, str(REDIS_PORT), LOCAL_SETTINGS['celery']['broker_db'])
    CELERY_BACKEND_URL = 'redis://{}:{}/{}'.format(REDIS_HOST, str(REDIS_PORT), LOCAL_SETTINGS['celery']['backend_db'])

    GPU_SETTINGS = LOCAL_SETTINGS['gpu'] if 'gpu' in LOCAL_SETTINGS else None
    MINIO_SETTINGS = LOCAL_SETTINGS['minio'] if 'minio' in LOCAL_SETTINGS else None
    ROCHE_SETTINGS = LOCAL_SETTINGS['roche'] if 'roche' in LOCAL_SETTINGS else None

    MINIO_ACCESS_KEY = MINIO_SETTINGS['access_key'] if MINIO_SETTINGS else ''
    MINIO_ACCESS_SECRET = MINIO_SETTINGS['access_secret'] if MINIO_SETTINGS else ''
    PRIVATE_ENDPOINT = MINIO_SETTINGS['private_endpoint'] if MINIO_SETTINGS else ''
    PUBLIC_ENDPOINT = MINIO_SETTINGS['public_endpoint'] if MINIO_SETTINGS else ''
    BUCKET_NAME = MINIO_SETTINGS['bucket_name'] if MINIO_SETTINGS else ''
    USE_HTTPS = MINIO_SETTINGS['use_https'].lower() == 'true' if MINIO_SETTINGS else False

    TOTAL_GPU_MEM = GPU_SETTINGS['total_gpu_mem'] if GPU_SETTINGS else 12

    IMAGE_SERVER = LOCAL_SETTINGS['default']['image_server']

    REPORT_SERVER = LOCAL_SETTINGS['default']['report_server']

    ELECTRON_UPLOAD_SERVER = 'http://{}:3000/download'

    # 需要记录操作日志的算法模块
    ai_log_list = ['tct', 'lct', 'pdl1', 'human_tl']

    PLUGINS = []

    SYNC_OPERATIONS = []

    ROCHE_API_SERVER = ROCHE_SETTINGS['api_server'] if ROCHE_SETTINGS else None
    ROCHE_IMAGE_SERVER = ROCHE_SETTINGS['image_server'] if ROCHE_SETTINGS else None

    # 版本号（用于算法用量统计）
    VERSION = '4.0.0'
