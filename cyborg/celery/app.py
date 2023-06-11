import logging
from functools import wraps
from typing import Tuple

from celery import Celery
from celery.result import AsyncResult
from kombu import Exchange, Queue

from cyborg.app.request_context import request_context
from cyborg.infra.celery import make_default_config, async_get_result

logger = logging.getLogger(__name__)


QUEUE_NAME_DEFAULT = 'default'
QUEUE_NAME_AI_TASK = 'ai_task'
ROUTING_KEY_DEFAULT = f'host.{QUEUE_NAME_DEFAULT}'
ROUTING_KEY_AI_TASK = f'host.{QUEUE_NAME_AI_TASK}'


def make_celery_config():
    d = make_default_config()
    return d


default_exchange = Exchange('default', type='direct')
host_exchange = Exchange('host', type='direct')

app = Celery('cyborg.main', include=[
    'cyborg.modules.ai.application.tasks',
    'cyborg.modules.slice.application.tasks',
])

app.config_from_object(make_celery_config())
app.conf.update(
    BROKER_POOL_LIMIT=None,
    CELERY_QUEUES=(
        Queue(QUEUE_NAME_DEFAULT, default_exchange, routing_key='default'),
        Queue(QUEUE_NAME_AI_TASK, host_exchange, routing_key=f'host.{QUEUE_NAME_AI_TASK}'),
    ),
    CELERY_DEFAULT_QUEUE='default',
    CELERY_DEFAULT_EXCHANGE='default',
    CELERY_DEFAULT_ROUTING_KEY='default',
    CELERY_ROUTES=(
        {'cyborg.modules.ai.application.tasks.run_ai_task': {
            'queue': QUEUE_NAME_AI_TASK,
            'routing_key': ROUTING_KEY_AI_TASK,
        }},
        {'cyborg.modules.slice.application.tasks.create_report': {
            'queue': QUEUE_NAME_DEFAULT,
            'routing_key': ROUTING_KEY_DEFAULT,
        }}
    )
)


def celery_task(f):

    @wraps(f)
    def _remote_func(*args, **kwargs):
        with request_context:
            request_context.current_user = kwargs.pop('current_user')
            request_context.company = kwargs.pop('company')
            result = f(*args, **kwargs)
            return result

    func = app.task(_remote_func, max_retries=3)

    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['current_user'] = request_context.current_user
        kwargs['company'] = request_context.company
        task_options = kwargs.pop('task_options', {}) or {}
        result = func.apply_async(args, kwargs, **task_options)
        return result
    return wrapper


if __name__ == '__main__':

    app.start()
