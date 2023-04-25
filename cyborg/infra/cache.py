import logging
import pickle
from functools import wraps
from inspect import getfullargspec
from typing import Iterable, Union

from redis.client import Redis

from cyborg.app.settings import Settings

logger = logging.getLogger(__name__)


class Cache(object):
    ONE_MINUTE = 60
    ONE_HOUR = 60 * ONE_MINUTE
    ONE_DAY = 24 * ONE_HOUR
    ONE_WEEK = 7 * ONE_DAY

    def __init__(self, write_node, read_node):
        self.write_node = write_node
        self.read_node = read_node

    @staticmethod
    def _get_key_generator(key_pattern, arg_names, defaults):
        args = dict(zip(arg_names[-len(defaults):], defaults)) if defaults else {}

        def gen_key(*a, **kw):
            aa = args.copy()
            aa.update(dict(zip(arg_names, a)))
            aa.update(kw)
            key = key_pattern.format(*[aa[n] for n in arg_names], **aa)
            return key and key.replace(' ', '_'), aa

        return gen_key

    @staticmethod
    def dump_object(value):
        """Dumps an object into a string for redis.  By default it serializes
        integers as regular string and pickle dumps everything else.
        """
        t = type(value)
        if t == int:
            return str(value).encode("ascii")
        return b"!" + pickle.dumps(value)

    @staticmethod
    def load_object(value):
        """The reversal of :meth:`dump_object`.
        """
        if value is None:
            return None
        if value.startswith(b"!"):
            try:
                return pickle.loads(value[1:])
            except pickle.PickleError:
                return None
        try:
            return int(value)
        except ValueError:
            return value

    def ttl(self, key):
        return self.read_node.ttl(key)

    def expire(self, key, time):
        """
        Set an expire flag on key ``name`` for ``time`` seconds. ``time``
        can be represented by an integer or a Python timedelta object.
        """
        self.write_node.expire(key, time)

    def get(self, key, default=None):
        value = self.load_object(self.read_node.get(key))
        if value is None:
            return default
        return value

    def set(self, key, value, ex=None, nx=None):
        dump = self.dump_object(value)
        return self.write_node.set(key, dump, ex=ex, nx=nx)

    # List Command
    def llen(self, key) -> int:
        return self.read_node.llen(key)

    def lrange(self, key, start, end):
        return self.read_node.lrange(key, start, end)

    def append(self, key, *values):
        return self.write_node.rpush(key, *values)

    def lindex(self, key, index):
        return self.read_node.lindex(key, index)

    def lset(self, key, index, value):
        return self.write_node.lset(key, index, value)

    # Map Command
    def hset(self, key, field, value, nx=False):
        if nx:
            return self.write_node.hsetnx(key, field, value)
        return self.write_node.hset(key, field, value)

    def hget(self, key, field):
        return self.read_node.hget(key, field)

    def hgetall(self, key):
        return self.read_node.hgetall(key)

    def hdel(self, key, *fields):
        return self.write_node.hdel(key, *fields)
    # end Map

    # Set Command
    def sadd(self, key, *values):
        return self.write_node.sadd(key, *values)

    def srem(self, key, *values):
        return self.write_node.srem(key, *values)

    def smembers(self, key):
        return self.read_node.smembers(key)

    def sismember(self, key, value):
        return self.read_node.sismember(key, value)
    # end Set

    def delete(self, key):
        return self.write_node.delete(key)

    def delete_many(self, keys: Iterable):
        if not keys:
            return
        return self.write_node.delete(*keys)

    def incrby(self, key: str, val: int = 1):
        return self.write_node.incrby(key, val)

    def has(self, key):
        return self.read_node.exists(key)

    def clear(self, asynchronous=False):
        """Delete all keys in the current database.

        ``asynchronous`` indicates whether the operation is
        executed asynchronously by the server.
        """
        return self.write_node.flushdb(asynchronous=asynchronous)

    def cache(self, key_pattern: str, expire: Union[int, float] = ONE_DAY, version: int = 0):
        """Decorator for sync function cache!
        """
        def decorator(f):
            args, varargs, keywords, defaults, _, _, _ = getfullargspec(f)
            if varargs or keywords:
                raise Exception('Varargs not supported')
            gen_key = self._get_key_generator(key_pattern, args, defaults)

            @wraps(f)
            def wrapped(*a, **kw):
                if version == -1:
                    return f(*a, **kw)

                key, _ = gen_key(*a, **kw)
                if not key:
                    return f(*a, **kw)

                if version > 0:
                    key = f'{key}:v{version}'

                try:
                    r = self.get(key)
                except Exception:
                    logger.exception('Exception when get from cache backend')
                    return f(*a, **kw)

                if r is None:
                    r = f(*a, **kw)
                    if r is not None:
                        self.set(key, r, int(expire))
                return r

            return wrapped
        return decorator

    def async_cache(self, key_pattern: str, expire=ONE_DAY):
        """Decorator for async function to cache!
        """
        def decorator(f):
            args, varargs, keywords, defaults, _, _, _ = getfullargspec(f)
            if varargs or keywords:
                raise Exception('Varargs not supported')
            gen_key = self._get_key_generator(key_pattern, args, defaults)

            @wraps(f)
            async def wrapped(*a, **kw):
                key, _ = gen_key(*a, **kw)
                if not key:
                    result = await f(*a, **kw)
                    return result

                try:
                    r = self.get(key)
                except Exception:
                    logger.exception('Exception when get from cache backend')
                    result = await f(*a, **kw)
                    return result

                if r is None:
                    r = await f(*a, **kw)
                    if r is not None:
                        self.set(key, r, expire)
                return r

            return wrapped
        return decorator

    def pcache(self, key_pattern, count=300, expire=ONE_DAY):
        def decorator(f):
            args, varargs, keywords, defaults, _, _, _ = getfullargspec(f)
            if varargs or keywords:
                raise Exception('Varargs not supported')
            if 'limit' not in args:
                raise Exception('Function must have "limit" in args')
            gen_key = self._get_key_generator(key_pattern, args, defaults)

            @wraps(f)
            def wrapped(*a, **kw):
                key, args = gen_key(*a, **kw)
                start = args.pop('start', 0)
                limit = args.pop('limit')
                if not key or limit is None or start + limit > count:
                    return f(*a, **kw)

                try:
                    r = self.get(key)
                except Exception:
                    logger.exception('Exception when pcache get from backend')
                    return f(*a, **kw)

                if r is None:
                    r = f(start=0, limit=count, **args)
                    self.set(key, r, expire)

                return r[start:start + limit]
            return wrapped
        return decorator


class NoCache(Cache):

    def __init__(self):
        super(NoCache, self).__init__(None, None)

    def ttl(self, key):
        return 0

    def expire(self, key, time):
        ...

    def get(self, key, default=None):
        return None

    def set(self, key, value, ex=None, nx=None):
        ...

    def llen(self, key) -> int:
        return 0

    def lrange(self, key, start, end):
        return []

    def append(self, key, *values):
        ...

    def hset(self, key, field, value, nx=False):
        ...

    def hget(self, key, field):
        return None

    def hgetall(self, key):
        return {}

    def hdel(self, key, *fields):
        ...

    def sadd(self, key, *values):
        ...

    def srem(self, key, *values):
        ...

    def smembers(self, key):
        return []

    def sismember(self, key, value):
        return False

    def delete(self, key):
        ...

    def delete_many(self, keys: Iterable):
        ...

    def incrby(self, key: str, val: int = 1):
        ...

    def has(self, key):
        return False

    def clear(self, asynchronous=False):
        ...


if Settings.CACHE_REDIS_URI:
    db = Redis.from_url(Settings.CACHE_REDIS_URI, db=0)
    cache = Cache(
        write_node=db,
        read_node=db,
    )
else:
    cache = NoCache()
