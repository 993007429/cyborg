from functools import wraps

from celery import Celery
from kombu import Exchange, Queue

from cyborg.infra.celery import make_default_config


QUEUE_NAME_DEFAULT = 'default'
QUEUE_NAME_AI_TASK = 'ai_task'
ROUTING_KEY_AI_TASK = f'host.{QUEUE_NAME_AI_TASK}'


def make_celery_config():
    d = make_default_config()
    return d


default_exchange = Exchange('default', type='direct')
host_exchange = Exchange('host', type='direct')

app = Celery('cyborg.main', include=[
    'cyborg.modules.ai.application.tasks',
])

app.config_from_object(make_celery_config())
app.conf.update(
    BROKER_POOL_LIMIT=None,
    CELERY_QUEUES=(
        Queue('default', default_exchange, routing_key='default'),
        Queue(QUEUE_NAME_AI_TASK, host_exchange, routing_key=f'host.{QUEUE_NAME_AI_TASK}'),
    ),
    CELERY_DEFAULT_QUEUE='default',
    CELERY_DEFAULT_EXCHANGE='default',
    CELERY_DEFAULT_ROUTING_KEY='default',
    CELERY_ROUTES=(
        # 文件上传
        {'cyborg.modules.ai.application.tasks.run_ai_task': {
            'queue': QUEUE_NAME_AI_TASK,
            'routing_key': f'host.{QUEUE_NAME_AI_TASK}',
        }}
    )
)


def celery_task(f):
    func = app.task(f, max_retries=3)

    @wraps(func)
    def wrapper(*args, **kwargs):
        task_options = kwargs.pop('task_options', {}) or {}
        return func.apply_async(args, kwargs, **task_options)
    return wrapper


if __name__ == '__main__':

    app.start()
