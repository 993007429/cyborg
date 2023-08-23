from datetime import datetime


class DatetimeUtil(object):

    @staticmethod
    def is_available_date(date: datetime) -> bool:
        if not date:
            return False
        delta = datetime.now() - date
        if delta.total_seconds() / 60 > 5:
            return False
        return True

    @staticmethod
    def dt_to_normal_text(dt: datetime) -> str:
        """将 `datetime` 对象转成易读文本

        - X秒前 < 60秒
        - X分钟前 < 60分
        - X小时前 < 24小时
        - X天前 < 1个月(30天)
        - X个月以前 < 12个月
        - 具体时间 >= 12个月
        """
        now = datetime.now()
        delta = now - dt

        seconds = delta.seconds

        if delta.days > 0:
            if delta.days < 30:  # 当月
                return f'{delta.days}天前'
            elif now.year == dt.year:  # 同年/12月内
                return f'{now.month - dt.month}月前'
            else:
                return dt.strftime('%Y-%m-%d %H:%M:%S')

        if seconds < 60:
            return f'{seconds}秒前'
        elif seconds < 60 * 60:  # 60分钟
            return f'{int(seconds / 60)}分钟前'
        elif seconds < 60 * 60 * 24:  # 24小时
            return f'{int(seconds / 60 / 60)}小时前'
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S')
