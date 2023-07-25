import re
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode


class UrlUtil:
    """A toolkit for web normal use
    """

    @staticmethod
    def add_url_query_params(url, qs_dict):
        bits = list(urlparse(url))
        qs = parse_qs(bits[4])
        for k, v in qs_dict.items():
            if isinstance(v, (list, tuple)):
                for i in v:
                    qs.setdefault(k, []).append(i)
            else:
                qs.setdefault(k, []).append(v)
        bits[4] = urlencode(qs, True)
        return urlunparse(bits)

    @classmethod
    def is_valid_url(cls, string: str) -> bool:
        if not string:
            return False
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return regex.match(string) is not None

    @classmethod
    def turn_url_queries_to_dict(cls, query_string: str, *, order: str = 'asc') -> dict:
        """convert url query string to dict format

        if multiple value with same key in queries,
        will convert to a list to that `key`

        :param query_string: string format like `w=13&q=35&q=899`
        :param order: sort the dict with the `key` alpha number with specify order
        :return: dict
        """
        result = parse_qs(query_string)
        return dict(
            sorted(
                {k: (v if len(v) > 1 else v[0]) for k, v in result.items()}.items(),
                key=lambda x: x,
                reverse=order == 'desc',
            )
        )

    @classmethod
    def turn_dict_to_url_queries(cls, data: dict) -> str:
        """convert a dict into url query string format

        :param data: dict data
        :return: string format like q=123&w=345&q=233
        """
        query_string = ''
        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                for i in v:
                    query_string += f'&{k}={i}'
            else:
                query_string += f'&{k}={v}'
        return query_string[1:]

    @classmethod
    def sort_query_string_by_key(cls, query_string: str, order: str = 'asc') -> str:
        return cls.turn_dict_to_url_queries(
            cls.turn_url_queries_to_dict(query_string, order=order)
        )
