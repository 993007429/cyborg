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
    def to_sorted_query_string(cls, data: dict) -> str:
        """convert a dict into url query string format

        :param data: dict data
        :return: string format like q=123&w=345&q=233
        """
        query_string = ''
        for k, v in sorted(data.items(), key=lambda x: x[0]):
            if isinstance(v, (list, tuple)):
                for i in v:
                    query_string += f'&{k}={i}'
            else:
                query_string += f'&{k}={v}'
        return query_string[1:]
