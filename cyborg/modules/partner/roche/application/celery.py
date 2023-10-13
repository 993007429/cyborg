import logging
from functools import wraps

from celery import Celery
from kombu import Exchange, Queue

from cyborg.app.request_context import request_context
from cyborg.infra.celery import make_default_config

logger = logging.getLogger(__name__)


QUEUE_NAME_AI_TASK = 'roche_ai_task'
ROUTING_KEY_AI_TASK = f'roche.{QUEUE_NAME_AI_TASK}'


def make_celery_config():
    d = make_default_config()
    return d


roche_exchange = Exchange('roche', type='direct')

app = Celery('cyborg.main', include=[
    'cyborg.modules.partner.roche.application.tasks',
])

app.config_from_object(make_celery_config())
app.conf.update(
    BROKER_POOL_LIMIT=None,
    CELERY_QUEUES=(
        Queue(QUEUE_NAME_AI_TASK, roche_exchange, routing_key=ROUTING_KEY_AI_TASK),
    ),
    CELERY_DEFAULT_QUEUE='default',
    CELERY_DEFAULT_EXCHANGE='default',
    CELERY_DEFAULT_ROUTING_KEY='default',
    CELERY_ROUTES=(
        {'cyborg.modules.partner.roche.application.tasks.run_ai_task': {
            'queue': QUEUE_NAME_AI_TASK,
            'routing_key': ROUTING_KEY_AI_TASK,
        }},
    )
)


def celery_task(f):

    @wraps(f)
    def _remote_func(*args, **kwargs):
        with request_context:
            result = f(*args, **kwargs)
            return result

    func = app.task(_remote_func, max_retries=3)

    @wraps(func)
    def wrapper(*args, **kwargs):
        task_options = kwargs.pop('task_options', {}) or {}
        result = func.apply_async(args, kwargs, **task_options)
        return result
    return wrapper


if __name__ == '__main__':

    app.start()
