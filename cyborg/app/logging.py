_logging_config: dict = {}


def gen_logging_config(logging_filename):
    global _logging_config
    if not _logging_config:
        _logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simple': {
                    'format': '%(asctime)s-%(levelname)s-%(module)s-%(lineno)s-%(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S %z',
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'simple',
                    'stream': 'ext://sys.stdout',
                },
                'file': {
                    'class': 'logging.handlers.TimedRotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'simple',
                    'filename': logging_filename,
                    'when': 'midnight',
                    'interval': 1,
                    'backupCount': 30
                }
            },
            'loggers': {
                'cyborg': {
                    'handlers': ['console'],
                    'level': 'INFO',
                    'propagate': True,
                }
            }
        }
    return _logging_config
