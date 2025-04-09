#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文数字规范化处理模块
"""

import re
from typing import List, Dict, Tuple

from .constants import DIGIT_MAP, TENS_MAP, MAGNITUDE_MAP, ORDINAL_MAP

# 数字表达式
RE_INTEGER = re.compile(r'(-?)(\d+)')
RE_DECIMAL_NUM = re.compile(r'(-?)(\d+)\.(\d+)')
RE_NUMBER = re.compile(r'(-?)(\d+)(?:\.(\d+))?')

# 分数表达式
RE_FRACTION = re.compile(r'(-??\d+)/(\d+)')

# 百分比表达式
RE_PERCENTAGE = re.compile(r'(-?)(\d+(?:\.\d+)?)%')

# 范围表达式
RE_RANGE = re.compile(r'(\d+(?:\.\d+)?)[-~](\d+(?:\.\d+)?)')


def verbalize_number(num_str: str) -> str:
    """将数字转换为英文单词
    
    Args:
        num_str: 数字字符串
        
    Returns:
        英文单词表示
    """
    # 处理零
    if num_str == '0':
        return DIGIT_MAP['0']
    
    # 移除前导零
    num_str = num_str.lstrip('0')
    if not num_str:
        return DIGIT_MAP['0']
    
    # 小于100的数字处理
    if len(num_str) <= 2:
        if num_str in DIGIT_MAP:
            return DIGIT_MAP[num_str]
        
        tens, ones = num_str[0], num_str[1]
        if ones == '0':
            return TENS_MAP[tens]
        else:
            return f"{TENS_MAP[tens]} {DIGIT_MAP[ones]}"
    
    # 处理三位数
    if len(num_str) == 3:
        hundreds = num_str[0]
        rest = num_str[1:]
        
        if rest == '00':
            return DIGIT_MAP[hundreds] + ' hundred'
        else:
            return DIGIT_MAP[hundreds] + ' hundred ' + verbalize_number(rest)
    
    # 处理更大的数字
    for mag_digits, mag_name in sorted(MAGNITUDE_MAP.items(), reverse=True):
        if len(num_str) > mag_digits:
            front = num_str[:-mag_digits]
            back = num_str[-mag_digits:]
            
            if all(d == '0' for d in back):
                return verbalize_number(front) + ' ' + mag_name
            else:
                return verbalize_number(front) + ' ' + mag_name + ' ' + verbalize_number(back)
    
    # 如果未匹配任何情况，按数字分组处理
    if len(num_str) > 3:
        thousands = num_str[:-3]
        rest = num_str[-3:]
        
        if rest == '000':
            return verbalize_number(thousands) + ' thousand'
        else:
            return verbalize_number(thousands) + ' thousand ' + verbalize_number(rest)
    
    return num_str  # 如果发生任何问题，返回原始字符串


def replace_integer(match) -> str:
    """处理整数
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    groups = match.groups()
    
    # 根据匹配组的数量不同，处理方式有所不同
    if len(groups) == 1:
        # 只有数字，没有符号
        return verbalize_number(groups[0])
    elif len(groups) >= 2:
        # 有符号和数字
        sign, number = groups[0], groups[1]
        if sign and sign == '-':
            return 'negative ' + verbalize_number(number)
        return verbalize_number(number)
    else:
        # 直接返回原始匹配结果
        return verbalize_number(match.group(0))


def replace_decimal(match) -> str:
    """处理小数
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    sign, integer_part, decimal_part = match.groups()
    
    integer_text = verbalize_number(integer_part)
    decimal_text = ' '.join(DIGIT_MAP[digit] for digit in decimal_part)
    
    if sign:
        return 'negative ' + integer_text + ' point ' + decimal_text
    return integer_text + ' point ' + decimal_text


def replace_number(match) -> str:
    """处理通用数字
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    sign = match.group(1)
    integer_part = match.group(2)
    decimal_part = match.group(3)
    
    if decimal_part:
        result = replace_decimal(match)
    else:
        result = replace_integer(match)
    
    return result


def replace_fraction(match) -> str:
    """处理分数
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    numerator, denominator = match.groups()
    
    # 使用负数标记
    is_negative = False
    if numerator.startswith('-'):
        is_negative = True
        numerator = numerator[1:]
    
    num_int = int(numerator)
    denom_int = int(denominator)
    
    # 处理特殊分母
    if denom_int == 2:
        if num_int == 1:
            return 'negative one half' if is_negative else 'one half'
        else:
            num_text = verbalize_number(numerator)
            return f"negative {num_text} halves" if is_negative else f"{num_text} halves"
    elif denom_int == 3:
        if num_int == 1:
            return 'negative one third' if is_negative else 'one third'
        else:
            num_text = verbalize_number(numerator)
            return f"negative {num_text} thirds" if is_negative else f"{num_text} thirds"
    elif denom_int == 4:
        if num_int == 1:
            return 'negative one quarter' if is_negative else 'one quarter'
        elif num_int == 3:
            return 'negative three quarters' if is_negative else 'three quarters'
        else:
            num_text = verbalize_number(numerator)
            return f"negative {num_text} quarters" if is_negative else f"{num_text} quarters"
    
    # 处理一般情况
    num_text = verbalize_number(numerator)
    denom_text = verbalize_ordinal(denom_int)
    
    if num_int == 1:
        return f"negative one {denom_text}" if is_negative else f"one {denom_text}"
    else:
        # 处理复数形式
        if denom_text.endswith('th'):
            plural_denom_text = f"{denom_text}s"
        elif denom_text.endswith('first'):
            plural_denom_text = denom_text.replace('first', 'firsts')
        elif denom_text.endswith('second'):
            plural_denom_text = denom_text.replace('second', 'seconds')
        elif denom_text.endswith('third'):
            plural_denom_text = denom_text.replace('third', 'thirds')
        else:
            plural_denom_text = f"{denom_text}s"
        
        return f"negative {num_text} {plural_denom_text}" if is_negative else f"{num_text} {plural_denom_text}"


def replace_percentage(match) -> str:
    """处理百分比
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    sign, number = match.groups()
    
    number_text = verbalize_number(number.split('.')[0])
    if '.' in number:
        decimal_part = number.split('.')[1]
        if decimal_part != '0':
            number_text += ' point ' + ' '.join(DIGIT_MAP[digit] for digit in decimal_part)
    
    if sign:
        return f"negative {number_text} percent"
    return f"{number_text} percent"


def replace_range(match) -> str:
    """处理范围
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    start, end = match.groups()
    
    start_text = RE_NUMBER.sub(replace_number, start)
    end_text = RE_NUMBER.sub(replace_number, end)
    
    return f"{start_text} to {end_text}"