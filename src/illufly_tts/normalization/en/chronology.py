#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文日期和时间规范化处理模块
"""

import re
from typing import Dict, List

from .constants import DIGIT_MAP, ORDINAL_MAP, TENS_MAP
from .num import verbalize_number

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
RE_TIME = re.compile(r'(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(am|pm|a\.m\.|p\.m\.|AM|PM|A\.M\.|P\.M\.))?', re.IGNORECASE)
RE_TIME_RANGE = re.compile(r'(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(am|pm|a\.m\.|p\.m\.|AM|PM|A\.M\.|P\.M\.))?\s*[-~]\s*(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(am|pm|a\.m\.|p\.m\.|AM|PM|A\.M\.|P\.M\.))?', re.IGNORECASE)

# 日期表达式 (MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD)
RE_DATE = re.compile(r'(\d{1,2})/(\d{1,2})/(\d{2,4})')
RE_DATE2 = re.compile(r'(\d{4})[-.\/](\d{1,2})[-.\/](\d{1,2})')

# 年份范围表达式
RE_YEAR_RANGE = re.compile(r'(\d{4})[-~到至](\d{4})')

# 添加日期范围的处理
RE_DATE_RANGE = re.compile(r'(\d{1,2})/(\d{1,2})/(\d{2,4})\s*[-~]\s*(\d{1,2})/(\d{1,2})/(\d{2,4})')
RE_DATE_RANGE2 = re.compile(r'(\d{4})[-.\/](\d{1,2})[-.\/](\d{1,2})\s*[-~]\s*(\d{4})[-.\/](\d{1,2})[-.\/](\d{1,2})')

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
    groups = match.groups()
    hour = groups[0]
    minute = groups[1]
    second = groups[2] if len(groups) > 2 else None
    ampm = groups[3] if len(groups) > 3 else None
    
    # 处理12小时制
    hour_num = int(hour)
    if ampm:
        if ampm.lower() in ['pm', 'p.m.'] and hour_num < 12:
            hour_num += 12
        elif ampm.lower() in ['am', 'a.m.'] and hour_num == 12:
            hour_num = 0
    
    hour_text = verbalize_number(str(hour_num))
    minute_text = verbalize_number(minute)
    
    # 统一使用 "hour:minute" 格式
    time_text = f"{hour_text} {minute_text}"
    
    # 添加上午/下午标记
    if ampm:
        if ampm.lower() in ['am', 'a.m.']:
            time_text += ' in the morning'
        else:
            if hour_num < 18:
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
    
    # 验证月份是否有效
    month_num = int(month.lstrip('0'))
    if month_num < 1 or month_num > 12:
        return match.group(0)  # 返回原始文本
    
    month_name = MONTH_NAMES[str(month_num)]
    day_num = int(day.lstrip('0'))
    
    # 验证日期是否有效
    if day_num < 1 or day_num > 31:
        return match.group(0)  # 返回原始文本
    
    # 处理年份
    if len(year) == 2:
        year_num = int(year)
        if year_num < 10:
            year_text = f"two thousand {DIGIT_MAP[year]}"
        else:
            # 使用"twenty"读法
            year_text = f"twenty {verbalize_number(year)}"
    else:
        # 4位年份
        if year.startswith('19'):
            # 1900-1999年按"nineteen XX"读
            year_text = f"nineteen {verbalize_number(year[2:])}"
        elif year.startswith('20'):
            # 2000-2099年支持两种读法
            if year[2:] == '00':
                year_text = "two thousand"
            elif year[2] == '0':
                year_text = f"two thousand {DIGIT_MAP[year[3]]}".strip()
            else:
                # 使用"twenty"读法
                year_text = f"twenty {verbalize_number(year[2:])}"
        else:
            # 其他年份按完整数字读
            year_text = verbalize_number(year)
    
    # 组合日期 (加上ordinal后缀)
    if day_num == 1:
        day_ordinal = "first"
    elif day_num == 2:
        day_ordinal = "second"
    elif day_num == 3:
        day_ordinal = "third"
    elif day_num == 21:
        day_ordinal = "twenty first"
    elif day_num == 22:
        day_ordinal = "twenty second" 
    elif day_num == 23:
        day_ordinal = "twenty third"
    elif day_num == 31:
        day_ordinal = "thirty first"
    else:
        day_ordinal = verbalize_ordinal(day_num)
    
    return f"{month_name} {day_ordinal}, {year_text}"


def replace_date2(match) -> str:
    """处理ISO日期格式 (YYYY-MM-DD)
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    year, month, day = match.groups()
    
    # 验证月份是否有效
    month_num = int(month.lstrip('0'))
    if month_num < 1 or month_num > 12:
        return match.group(0)  # 返回原始文本
    
    month_name = MONTH_NAMES[str(month_num)]
    day_num = int(day.lstrip('0'))
    
    # 处理年份
    if year.startswith('19'):
        # 1900-1999年按"nineteen XX"读
        year_text = f"nineteen {verbalize_number(year[2:])}"
    elif year.startswith('20'):
        # 2000-2099年支持两种读法
        if year[2:] == '00':
            year_text = "two thousand"
        elif year[2] == '0':
            year_text = f"two thousand {DIGIT_MAP[year[3]]}".strip()
        else:
            # 使用"twenty"读法
            year_text = f"twenty {verbalize_number(year[2:])}"
    else:
        # 其他年份按完整数字读
        year_text = verbalize_number(year)
    
    # 组合日期
    return f"{month_name} {verbalize_ordinal(day_num)}, {year_text}"


def replace_year_range(match) -> str:
    """处理年份范围
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    start_year, end_year = match.groups()
    
    # 判断是否为中文环境
    is_chinese = any(c in match.group(0) for c in ['年', '至', '到']) or match.group(0).count('-') == 1 and '年' in match.group(0).split('-')[1]
    
    if is_chinese:
        # 中文数字处理
        from ..zh.num import num2str
        start_text = num2str(start_year)
        end_text = num2str(end_year)
        return f"{start_text}年至{end_text}年"
    else:
        # 英文数字处理
        if start_year.startswith('19'):
            # 1900-1999年按"nineteen XX"读
            start_text = f"nineteen {verbalize_number(start_year[2:])}"
        elif start_year.startswith('20'):
            # 2000-2099年支持两种读法
            if start_year[2:] == '00':
                start_text = "two thousand"
            elif start_year[2] == '0':
                start_text = f"two thousand {DIGIT_MAP[start_year[3]]}".strip()
            else:
                # 使用"twenty"读法
                start_text = f"twenty {verbalize_number(start_year[2:])}"
        else:
            # 其他年份按完整数字读
            start_text = verbalize_number(start_year)
        
        # 处理结束年份
        if end_year.startswith('19'):
            # 1900-1999年按"nineteen XX"读
            end_text = f"nineteen {verbalize_number(end_year[2:])}"
        elif end_year.startswith('20'):
            # 2000-2099年支持两种读法
            if end_year[2:] == '00':
                end_text = "two thousand"
            elif end_year[2] == '0':
                end_text = f"two thousand {DIGIT_MAP[end_year[3]]}".strip()
            else:
                # 使用"twenty"读法
                end_text = f"twenty {verbalize_number(end_year[2:])}"
        else:
            # 其他年份按完整数字读
            end_text = verbalize_number(end_year)
        
        return f"from {start_text} to {end_text}"


def replace_date_range(match) -> str:
    """处理日期范围 (MM/DD/YYYY 美式)
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    month1, day1, year1, month2, day2, year2 = match.groups()
    
    # 处理第一个日期
    month_num1 = int(month1.lstrip('0'))
    if month_num1 < 1 or month_num1 > 12:
        return match.group(0)  # 返回原始文本
    
    month_name1 = MONTH_NAMES[str(month_num1)]
    day_num1 = int(day1.lstrip('0'))
    
    # 处理第二个日期
    month_num2 = int(month2.lstrip('0'))
    if month_num2 < 1 or month_num2 > 12:
        return match.group(0)  # 返回原始文本
    
    month_name2 = MONTH_NAMES[str(month_num2)]
    day_num2 = int(day2.lstrip('0'))
    
    # 处理年份 (复用已有的年份处理逻辑)
    year_match1 = re.match(r'(\d{2,4})', year1)
    year_match2 = re.match(r'(\d{2,4})', year2)
    
    year_text1 = replace_year_simple(year_match1)
    year_text2 = replace_year_simple(year_match2)
    
    # 组合日期范围
    date1 = f"{month_name1} {verbalize_ordinal(day_num1)}, {year_text1}"
    date2 = f"{month_name2} {verbalize_ordinal(day_num2)}, {year_text2}"
    
    return f"from {date1} to {date2}"


def replace_date_range2(match) -> str:
    """处理ISO日期范围格式 (YYYY-MM-DD - YYYY-MM-DD)
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    year1, month1, day1, year2, month2, day2 = match.groups()
    
    # 处理第一个日期
    month_name1 = MONTH_NAMES[month1.lstrip('0')]
    day_num1 = int(day1.lstrip('0'))
    
    # 处理第二个日期
    month_name2 = MONTH_NAMES[month2.lstrip('0')]
    day_num2 = int(day2.lstrip('0'))
    
    # 处理年份 (复用已有的年份处理逻辑)
    year_match1 = re.match(r'(\d{4})', year1)
    year_match2 = re.match(r'(\d{4})', year2)
    
    year_text1 = replace_year_simple(year_match1)
    year_text2 = replace_year_simple(year_match2)
    
    # 组合日期范围
    date1 = f"{month_name1} {verbalize_ordinal(day_num1)}, {year_text1}"
    date2 = f"{month_name2} {verbalize_ordinal(day_num2)}, {year_text2}"
    
    return f"from {date1} to {date2}"


def replace_year_simple(match) -> str:
    """简化的年份处理函数，用于日期范围处理
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的年份文本
    """
    year = match.group(1)
    
    if len(year) == 2:
        year_num = int(year)
        if year_num < 10:
            return f"two thousand {DIGIT_MAP[year]}"
        else:
            # 使用"twenty"读法
            return f"twenty {verbalize_number(year)}"
    else:
        # 4位年份
        if year.startswith('19'):
            # 1900-1999年按"nineteen XX"读
            return f"nineteen {verbalize_number(year[2:])}"
        elif year.startswith('20'):
            # 2000-2099年支持两种读法
            if year[2:] == '00':
                return "two thousand"
            elif year[2] == '0':
                return f"two thousand {DIGIT_MAP[year[3]]}".strip()
            else:
                # 使用"twenty"读法
                return f"twenty {verbalize_number(year[2:])}"
        else:
            # 其他年份按完整数字读
            return verbalize_number(year)