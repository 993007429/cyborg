from typing import Optional


class Pagination(object):

    def __init__(self, args_dict: dict):
        self.data: Optional[list] = None
        self.page: Optional[int] = None
        self.per_page: Optional[int] = None
        self.total: Optional[int] = None
        for k, v in args_dict.items():
            if k == 'data':
                self.data = v
            elif k == 'page':
                self.page = v
            elif k == 'per_page':
                self.per_page = v
            elif k == 'total':
                self.total = v

    def to_dict(self) -> dict:
        data = self.data or []
        d = {
            'data': data,
            'total': self.total if self.total is not None else len(data)
        }
        if self.page is not None:
            d['page'] = self.page
        if self.per_page is not None:
            d['per_page'] = self.per_page
        return d
