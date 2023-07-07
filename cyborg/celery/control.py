import logging
import time
from logging.config import dictConfig

from celery.app.control import Control

from cyborg.app.logging import gen_logging_config
from cyborg.app.settings import Settings
from cyborg.celery.app import app, QUEUE_NAME_AI_TASK, ROUTING_KEY_AI_TASK
from cyborg.modules.ai.utils.gpu import get_gpu_status

is_debug = not Settings.ENV or Settings.ENV == 'LOCAL'
dictConfig(gen_logging_config(logging_filename=Settings.APP_LOG_FILE))
logger = logging.getLogger('cyborg.celery.control')


if __name__ == "__main__":

    control = Control(app)

    while True:
        gpu_status = get_gpu_status()
        if gpu_status:
            ins = control.inspect()
            queues = [queue for host, queues in ins.active_queues().items() for queue in queues]
            is_consuming = any(queue.get('name') == QUEUE_NAME_AI_TASK for queue in queues)
            if not any(status['free'] > 10 for status in gpu_status):
                if is_consuming:
                    logger.info(f'celery worker stop to consume from queue: {QUEUE_NAME_AI_TASK}')
                    control.cancel_consumer(QUEUE_NAME_AI_TASK)
            else:
                if not is_consuming:
                    logger.info(f'celery worker resume to consume from queue: {QUEUE_NAME_AI_TASK}')
                    control.add_consumer(QUEUE_NAME_AI_TASK, routing_key=ROUTING_KEY_AI_TASK)

        time.sleep(5)
