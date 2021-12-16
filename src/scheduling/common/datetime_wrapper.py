# -*- coding: utf-8 -*-

import sys
import os
from datetime import datetime, timedelta
import time
import logging
reload(sys)
sys.path.append(os.getcwd())

"""
    python中时间日期格式化符号：
    ------------------------------------
    %y 两位数的年份表示（00-99）
    %Y 四位数的年份表示（000-9999）
    %m 月份（01-12）
    %d 月内中的一天（0-31）
    %H 24小时制小时数（0-23）
    %I 12小时制小时数（01-12）
    %M 分钟数（00=59）
    %S 秒（00-59）
    %a 本地简化星期名称
    %A 本地完整星期名称
    %b 本地简化的月份名称
    %B 本地完整的月份名称
    %c 本地相应的日期表示和时间表示
    %j 年内的一天（001-366）
    %p 本地A.M.或P.M.的等价符
    %U 一年中的星期数（00-53）星期天为星期的开始
    %w 星期（0-6），星期天为星期的开始
    %W 一年中的星期数（00-53）星期一为星期的开始
    %x 本地相应的日期表示
    %X 本地相应的时间表示
    %Z 当前时区的名称  # 乱码
    %% %号本身
"""
DT_F_YEAR = "%Y"        # 四位数的年份表示（000-9999）
DT_F_MONTH = "%m"       # 月份（01-12）
DT_F_DAY = "%d"         # 月内中的一天（0-31）
DT_F_HOUR = "%H"        # 24小时制小时数（0-23）
DT_F_MIN = "%M"         # 分钟数（00=59）
DT_F_SEC = "%S"         # 秒（00-59）
DT_F_DAY_OF_WEEK = "%w" # 星期（0-6），星期天为星期的开始
DT_F_TIMEZONE = "%Z"    # 当前时区的名称

DT_F_PERFECT_DATE = "{}-{}-{}".format(DT_F_YEAR, DT_F_MONTH, DT_F_DAY)
DT_F_PERFECT_TIME = "{}:{}:{}".format(DT_F_HOUR, DT_F_MIN, DT_F_SEC)
DT_F_PERFECT_DATETIME = "{} {}".format(DT_F_PERFECT_DATE, DT_F_PERFECT_TIME)
DT_F_PERFECT_DATEHOUR = "{}-{}".format(DT_F_PERFECT_DATE, DT_F_HOUR)
DT_F_VERSION = "{}{}{}{}{}{}".format(DT_F_YEAR, DT_F_MONTH, DT_F_DAY, DT_F_HOUR, DT_F_MIN, DT_F_SEC)


class DateTime(object):

    @staticmethod
    def now(fmt):
        return datetime.now().strftime(fmt)

    @staticmethod
    def str_to_datetime(arg, fmt, verbose=True):
        try:
            d = datetime.strptime(arg, fmt)
            return d
        except Exception as ex:
            if verbose:
                logging.exception("exception occur in %s, %s", __file__, ex)
        return None

    @staticmethod
    def datetime_to_str(arg, fmt, verbose=True):
        try:
            if not isinstance(arg, datetime):
                raise Exception("input arg is not class datetime object")
            d = arg.strftime(fmt)
            return d
        except Exception as ex:
            if verbose:
                logging.exception("exception occur in %s, %s", __file__, ex)
        return None

    @staticmethod
    def to_ts(arg, fmt=DT_F_PERFECT_DATE):
        t = time.strptime(arg, fmt)
        return int(time.mktime(t)) * 1000

    def __init__(self, arg):
        self._datetime = arg

    def is_perfect_date(self):
        return DateTime.str_to_datetime(self._datetime, DT_F_PERFECT_DATE, verbose=False) is not None

    def is_perfect_datehour(self):
        return DateTime.str_to_datetime(self._datetime, DT_F_PERFECT_DATEHOUR, verbose=False) is not None

    def is_perfect_datetime(self):
        return DateTime.str_to_datetime(self._datetime, DT_F_PERFECT_DATETIME, verbose=False) is not None

    def apply_offset_by_day(self, offset):
        d = DateTime.str_to_datetime(self._datetime, DT_F_PERFECT_DATE)
        assert d is not None, "input date should be formatted as: {}".format(DT_F_PERFECT_DATE)
        d += timedelta(days=offset)
        return DateTime.datetime_to_str(d, DT_F_PERFECT_DATE)

    def apply_offset_by_hour(self, offset):
        d = DateTime.str_to_datetime(self._datetime, DT_F_PERFECT_DATEHOUR)
        assert d is not None, "input date should be formatted as: {}".format(DT_F_PERFECT_DATEHOUR)
        d += timedelta(hours=offset)
        return DateTime.datetime_to_str(d, DT_F_PERFECT_DATEHOUR)

    def apply_range_by_day(self, offset, exclude=False, reverse=False):
        assert offset != 0, "offset must not be zero"
        offset_range = range(1, abs(offset) + 1) if exclude else range(abs(offset))
        dates = [self.apply_offset_by_day(i*offset/abs(offset)) for i in offset_range]
        dates.sort(reverse=reverse)
        return dates

    def apply_range_by_hour(self, offset, exclude=False, reverse=False):
        assert offset != 0, "offset must not be zero"
        offset_range = range(1, abs(offset) + 1) if exclude else range(abs(offset))
        dates = [self.apply_offset_by_hour(i*offset/abs(offset)) for i in offset_range]
        dates.sort(reverse=reverse)
        return dates

    def distance_by_day(self, to):
        from_d = DateTime.str_to_datetime(self._datetime, DT_F_PERFECT_DATE)
        to_d = DateTime.str_to_datetime(to, DT_F_PERFECT_DATE)
        assert from_d, "input must be formatted as {}".format(DT_F_PERFECT_DATE)
        assert to_d, "input must be formatted as {}".format(DT_F_PERFECT_DATE)
        return (to_d - from_d).days

    def distance_by_hour(self, to):
        from_d = DateTime.str_to_datetime(self._datetime, DT_F_PERFECT_DATEHOUR)
        to_d = DateTime.str_to_datetime(to, DT_F_PERFECT_DATEHOUR)
        assert from_d, "input must be formatted as {}".format(DT_F_PERFECT_DATEHOUR)
        assert to_d, "input must be formatted as {}".format(DT_F_PERFECT_DATEHOUR)
        x = to_d - from_d
        return x.days * 24 + x.seconds / 3600


def test():
    assert DateTime("2019-01-10").is_perfect_date(), "fail is_perfect_date"
    assert not DateTime("2019-01-10-12").is_perfect_date(), "fail is_perfect_date"

    assert DateTime("2019-01-10-12").is_perfect_datehour(), "fail is_perfect_datehour"
    assert not DateTime("2019-01-10").is_perfect_datehour(), "fail is_perfect_datehour"

    assert DateTime("2019-01-10").apply_offset_by_day(2) == "2019-01-12", "fail apply_offset_by_day"
    assert DateTime("2019-01-10").apply_offset_by_day(-2) == "2019-01-08", "fail apply_offset_by_day"

    assert DateTime("2019-01-10-12").apply_offset_by_hour(2) == "2019-01-10-14", "fail apply_offset_by_hour"
    assert DateTime("2019-01-10-12").apply_offset_by_hour(-2) == "2019-01-10-10", "fail apply_offset_by_hour"

    assert DateTime("2019-01-10").apply_range_by_day(2, exclude=True, reverse=True) == ["2019-01-12", "2019-01-11"], "fail apply_range_by_day"
    assert DateTime("2019-01-10").apply_range_by_day(2, exclude=False, reverse=True) == ["2019-01-11", "2019-01-10"], "fail apply_range_by_day"
    assert DateTime("2019-01-10").apply_range_by_day(2, exclude=True, reverse=False) == ["2019-01-11", "2019-01-12"], "fail apply_range_by_day"
    assert DateTime("2019-01-10").apply_range_by_day(2, exclude=False, reverse=False) == ["2019-01-10", "2019-01-11"], "fail apply_range_by_day"
    assert DateTime("2019-01-10").apply_range_by_day(-2, exclude=True, reverse=True) == ["2019-01-09", "2019-01-08"], "fail apply_range_by_day"
    assert DateTime("2019-01-10").apply_range_by_day(-2, exclude=False, reverse=True) == ["2019-01-10", "2019-01-09"], "fail apply_range_by_day"
    assert DateTime("2019-01-10").apply_range_by_day(-2, exclude=True, reverse=False) == ["2019-01-08", "2019-01-09"], "fail apply_range_by_day"
    assert DateTime("2019-01-10").apply_range_by_day(-2, exclude=False, reverse=False) == ["2019-01-09", "2019-01-10"], "fail apply_range_by_day"

    assert DateTime("2019-01-10-12").apply_range_by_hour(2, exclude=True, reverse=True) == ["2019-01-10-14", "2019-01-10-13"], "fail apply_range_by_hour"
    assert DateTime("2019-01-10-12").apply_range_by_hour(2, exclude=False, reverse=True) == ["2019-01-10-13", "2019-01-10-12"], "fail apply_range_by_hour"
    assert DateTime("2019-01-10-12").apply_range_by_hour(2, exclude=True, reverse=False) == ["2019-01-10-13", "2019-01-10-14"], "fail apply_range_by_hour"
    assert DateTime("2019-01-10-12").apply_range_by_hour(2, exclude=False, reverse=False) == ["2019-01-10-12", "2019-01-10-13"], "fail apply_range_by_hour"
    assert DateTime("2019-01-10-12").apply_range_by_hour(-2, exclude=True, reverse=True) == ["2019-01-10-11", "2019-01-10-10"], "fail apply_range_by_hour"
    assert DateTime("2019-01-10-12").apply_range_by_hour(-2, exclude=False, reverse=True) == ["2019-01-10-12", "2019-01-10-11"], "fail apply_range_by_hour"
    assert DateTime("2019-01-10-12").apply_range_by_hour(-2, exclude=True, reverse=False) == ["2019-01-10-10", "2019-01-10-11"], "fail apply_range_by_hour"
    assert DateTime("2019-01-10-12").apply_range_by_hour(-2, exclude=False, reverse=False) == ["2019-01-10-11", "2019-01-10-12"], "fail apply_range_by_hour"

    assert DateTime("2019-01-10").distance_by_day("2019-01-12") == 2, "fail distance_by_day"
    assert DateTime("2019-01-10").distance_by_day("2019-01-08") == -2, "fail distance_by_day"

    assert DateTime("2019-01-10-12").distance_by_hour("2019-01-10-14") == 2, "fail distance_by_hour"
    assert DateTime("2019-01-10-12").distance_by_hour("2019-01-10-10") == -2, "fail distance_by_hour"

    print >> sys.stdout, "all unittest passed"


if __name__ == "__main__":
    test()
