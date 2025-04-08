#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文日期和时间规范化处理模块
"""

import re
from typing import Dict, List

from .constants import DIGIT_MAP, ORDINAL_MAP

# 月份名称
MONTH_NAMES = {
    '1': 'January', '2': 'February', '3': 'March', '4': 'April',
    '5': 'May', '6': 'June', '7': 'July', '8': 'August',
    '9': 'September', '10': 'October', '11': 'November', '12': 'December'
}

# 星期名称
DAY_NAMES = {
    '1': 'Monday', '2': 'Tuesday', '3': 'Wednesday', '4': 'Thursday',
    '5': 'Friday', '6': 'Saturday', '7': 'Sunday'
}

# 时间表达式
RE_TIME = re.compile(r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?', re.IGNORECASE)
RE_TIME_RANGE = re.compile(r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\s*[-~]\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?', re.IGNORECASE)

# 日期表达式 (MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD)
RE_DATE = re.compile(r'(\d{1,2})/(\d{1,2})/(\d{2,4})')
RE_DATE2 = re.compile(r'(\d{4})[-.\/](\d{1,2})[-.\/](\d{1,2})')


def get_ordinal_suffix(num: int) -> str:
    """获取序数词后缀
    
    Args:
        num: 数字
        
    Returns:
        后缀
    """
    if 10 <= num % 100 <= 20:
        return 'th'
    
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
    return suffix


def verbalize_ordinal(num: int) -> str:
    """将数字转换为序数词
    
    Args:
        num: 数字
        
    Returns:
        序数词
    """
    if str(num) in ORDINAL_MAP:
        return ORDINAL_MAP[str(num)]
    
    from .num import verbalize_number
    if num % 10 == 0 and str(num) in ORDINAL_MAP:
        return ORDINAL_MAP[str(num)]
    
    num_word = verbalize_number(str(num))
    if num_word.endswith('y'):
        return num_word[:-1] + 'ieth'
    return num_word + 'th'


def replace_time(match) -> str:
    """处理时间
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    is_range = len(match.groups()) > 4
    
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = match.group(3)
    ampm = match.group(4)
    
    if is_range:
        hour2 = int(match.group(5))
        minute2 = int(match.group(6))
        second2 = match.group(7)
        ampm2 = match.group(8)
    
    # 处理第一个时间
    result = _format_time(hour, minute, second, ampm)
    
    # 如果是时间范围，处理第二个时间
    if is_range:
        result += ' to ' + _format_time(hour2, minute2, second2, ampm2)
    
    return result


def _format_time(hour: int, minute: int, second, ampm) -> str:
    """格式化时间
    
    Args:
        hour: 小时
        minute: 分钟
        second: 秒
        ampm: 上午/下午
        
    Returns:
        格式化后的时间文本
    """
    # 处理12小时制/24小时制
    if ampm and ampm.lower() in ['pm', 'p.m.'] and hour < 12:
        hour += 12
    elif ampm and ampm.lower() in ['am', 'a.m.'] and hour == 12:
        hour = 0
    
    from .num import verbalize_number
    hour_text = verbalize_number(str(hour))
    
    if minute == 0:
        time_text = f"{hour_text} o'clock"
    elif minute == 15:
        time_text = f"quarter past {hour_text}"
    elif minute == 30:
        time_text = f"half past {hour_text}"
    elif minute == 45:
        next_hour = (hour + 1) % 24
        next_hour_text = verbalize_number(str(next_hour))
        time_text = f"quarter to {next_hour_text}"
    else:
        minute_text = verbalize_number(str(minute))
        time_text = f"{hour_text} {minute_text}"
    
    # 添加秒
    if second and second != '00':
        second_text = verbalize_number(second)
        time_text += f" and {second_text} seconds"
    
    # 添加上午/下午
    if ampm:
        if ampm.lower() in ['am', 'a.m.']:
            time_text += ' in the morning'
        else:
            if hour < 18:
                time_text += ' in the afternoon'
            else:
                time_text += ' in the evening'
    
    return time_text


def replace_date(match) -> str:
    """处理日期 (MM/DD/YYYY 美式)
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    month, day, year = match.groups()
    
    month_name = MONTH_NAMES[month.lstrip('0')]
    day_num = int(day.lstrip('0'))
    
    # 处理年份
    if len(year) == 2:
        year_num = int(year)
        if year_num < 10:
            year_text = f"two thousand {DIGIT_MAP[year]}"
        elif year_num < 20:
            year_text = f"twenty {DIGIT_MAP[year]}"
        else:
            # 21-99年按两位数读
            decade = DIGIT_MAP[year[0]] if year[0] != '0' else ''
            digit = DIGIT_MAP[year[1]] if year[1] != '0' else ''
            year_text = f"{decade} {digit}".strip()
    else:
        # 4位年份
        if year.startswith('19'):
            # 1900-1999年按"十九XX年"读
            year_text = f"nineteen {DIGIT_MAP[year[2]] + ' ' + DIGIT_MAP[year[3]]}".strip()
        elif year.startswith('20'):
            # 2000-2099年按"二千零XX年/二十XX年"读
            if year[2:] == '00':
                year_text = "two thousand"
            elif year[2] == '0':
                year_text = f"two thousand {DIGIT_MAP[year[3]]}".strip()
            else:
                from .num import verbalize_number
                year_text = f"two thousand {verbalize_number(year[2:])}"
        else:
            # 其他年份按完整数字读
            from .num import verbalize_number
            year_text = verbalize_number(year)
    
    # 组合日期
    return f"{month_name} {verbalize_ordinal(day_num)}, {year_text}"


def replace_date2(match) -> str:
    """处理ISO日期格式 (YYYY-MM-DD)
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    year, month, day = match.groups()
    
    month_name = MONTH_NAMES[month.lstrip('0')]
    day_num = int(day.lstrip('0'))
    
    # 处理年份
    from .num import verbalize_number
    if year.startswith('19'):
        # 1900-1999年按"十九XX年"读
        year_text = f"nineteen {verbalize_number(year[2:])}"
    elif year.startswith('20'):
        # 2000-2099年按"二千零XX年/二十XX年"读
        if year[2:] == '00':
            year_text = "two thousand"
        elif year[2] == '0':
            year_text = f"two thousand {DIGIT_MAP[year[3]]}".strip()
        else:
            year_text = f"two thousand {verbalize_number(year[2:])}"
    else:
        # 其他年份按完整数字读
        year_text = verbalize_number(year)
    
    # 组合日期
    return f"{month_name} {verbalize_ordinal(day_num)}, {year_text}"