import logging
from functools import wraps

from celery import Celery
from celery.signals import worker_process_shutdown
from kombu import Exchange, Queue

from cyborg.app.request_context import request_context
from cyborg.infra.celery import make_default_config

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
    'cyborg.modules.partner.roche.application.tasks',
])

app.config_from_object(make_celery_config())
app.conf.update(
    BROKER_POOL_LIMIT=None,
    CELERY_QUEUES=(
        Queue(QUEUE_NAME_DEFAULT, default_exchange, routing_key='default'),
        Queue(QUEUE_NAME_AI_TASK, host_exchange, routing_key=ROUTING_KEY_AI_TASK),
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
        }},
        {'cyborg.modules.slice.application.tasks.export_slice_files': {
            'queue': QUEUE_NAME_DEFAULT,
            'routing_key': ROUTING_KEY_DEFAULT,
        }},
        {'cyborg.modules.partner.roche.application.tasks.run_ai_task': {
            'queue': QUEUE_NAME_AI_TASK,
            'routing_key': ROUTING_KEY_AI_TASK,
        }},
    )
)


@worker_process_shutdown.connect
def worker_process_shutdown_handler(pid=None, **kwargs):
    from cyborg.app.service_factory import AppServiceFactory
    logger.info(f'kill task processes: {pid}')
    AppServiceFactory.ai_service.kill_task_processes(pid=pid)


def celery_task(f):

    @wraps(f)
    def _remote_func(*args, **kwargs):
        with request_context:
            request_context.current_user = kwargs.pop('current_user', None)
            request_context.company = kwargs.pop('company', None)
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
