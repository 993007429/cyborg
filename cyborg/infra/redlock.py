import inspect
import logging

from functools import wraps

from redlock import RedLockFactory, RedLockError

from redis.exceptions import ConnectionError

from cyborg.app.settings import Settings

try:
    lock_factory = RedLockFactory(connection_details=Settings.REDLOCK_CONNECTION_CONFIG)
except ConnectionError:
    lock_factory = None

logger = logging.getLogger(__name__)


def generate_key(key_pattern, arg_names, defaults, *a, **kw):
    args = dict(zip(arg_names[-len(defaults):], defaults)) if defaults else {}
    args.update(zip(arg_names, a))
    args.update(kw)

    key = key_pattern.format(**args)
    # key = re.sub('\s', '', key)
    return key.encode('utf-8'), args


def with_redlock(key_pattern, **lock_kwargs):
    """
    分布式Redis锁，https://redis.io/topics/distlock
    默认情况下：会尝试3次获得锁, 每次最多等待0.2秒，即如果程序f的执行时间小于0.4秒，则f会被重复执行
    视业务需要决定lock_kwargs，如果并发的请求只需要f被执行一次并抛弃其余的请求，请设置retry_times=1 retry_delay=0
    :param key_pattern:
    :param lock_kwargs:
         retry_times=DEFAULT_RETRY_TIMES  默认3，尝试次数，不是重试次数，故最少为1
         retry_delay=DEFAULT_RETRY_DELAY  默认200 毫秒，会取 0 ~ retry_delay 之间的随机值
         ttl=DEFAULT_TTL                  默认100000 毫秒，锁过期时间，必须设置，避免死锁和浪费redis空间
    :return:
    """

    def deco(f):
        arg_names, varargs, varkw, defaults, _, _, _ = inspect.getfullargspec(f)
        if varargs or varkw:
            raise Exception('do not support varargs')

        @wraps(f)
        def _(*a, **kw):
            try:
                key, _ = generate_key(key_pattern, arg_names, defaults, *a, **kw)
                with lock_factory.create_lock(key, **lock_kwargs):
                    return f(*a, **kw)
            except RedLockError as e:
                logger.exception('red lock error: %s', e)

        return _

    return deco


def with_redlock_kwargs(key_pattern, **lock_kwargs):
    """
    支持f(a, b, *args, **kwargs)，只支持a和b作为key_pattern中的填充字段
    """

    def deco(f):
        arg_names, _, _, defaults, _, _, _ = inspect.getfullargspec(f)

        @wraps(f)
        def _(*a, **kw):
            try:
                key, _ = generate_key(key_pattern, arg_names, defaults, *a, **kw)
                with lock_factory.create_lock(key_pattern, **lock_kwargs):
                    return f(*a, **kw)
            except RedLockError as e:
                logger.exception('red lock error: %s', e)

        return _

    return deco
