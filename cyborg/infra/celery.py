import asyncio
from typing import Tuple

from celery.app import trace
from celery.result import AsyncResult
from redis import StrictRedis

from cyborg.app.settings import Settings

trace.LOG_SUCCESS = 'Task %(name)s[%(id)s] succeeded in %(runtime)ss'

redis_broker = None


def get_redis_broker():
    global redis_broker
    if not redis_broker:
        redis_broker = StrictRedis.from_url(Settings.CELERY_BROKER_URL)
    return redis_broker


def make_default_config():
    config = dict(
        broker_url=Settings.CELERY_BROKER_URL,
        result_backend=Settings.CELERY_BACKEND_URL,
        broker_failover_strategy='round-robin',
        task_queue_ha_policy='all',
        broker_connection_max_retries=3,
        task_serializer='pickle',
        result_serializer='pickle',
        accept_content=['pickle', 'json'],
        result_expires=1800,
        worker_prefetch_multiplier=1,
        worker_max_memory_per_child=10000000,
        worker_send_task_events=True,
        task_send_sent_event=True,
        # task_acks_late=True,
        broker_transport_options={'visibility_timeout': 60}
    )
    return config


class AsyncGetResultTimeoutException(Exception):
    pass


async def async_get_result(result: AsyncResult, polling_params: Tuple[int, float] = (50, 0.5)):
    """
    get data from celery backend result using asyncio
    :param result:
    :param polling_params: a tuple which contains polling_times and sleep_time(by_second)
    :return:
    """
    polling_times, sleep_time = polling_params
    for _ in range(polling_times):
        if result.ready():
            return result.get()
        else:
            await asyncio.sleep(sleep_time)
    raise AsyncGetResultTimeoutException
