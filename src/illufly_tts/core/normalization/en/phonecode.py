#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文电话号码规范化处理模块
"""

import re
from typing import List

from .constants import DIGIT_MAP

# 美国电话号码 (XXX) XXX-XXXX 或 XXX-XXX-XXXX
RE_PHONE = re.compile(r'(?<!\d)(\+?1[-\s]?)?(\(?\d{3}\)?[-\s]?)(\d{3})[-\s]?(\d{4})(?!\d)')

# 国际电话号码 +XX XXXX XXXX
RE_MOBILE = re.compile(r'(?<!\d)(\+\d{1,3})\s?(\d{1,4}[-\s]?\d{1,4}[-\s]?\d{1,9})(?!\d)')


def verbalize_phone_digits(digits: str) -> str:
    """将电话号码数字转换为单词
    
    Args:
        digits: 数字字符串
        
    Returns:
        处理后的文本
    """
    # 电话号码按单个数字读
    result = []
    for digit in digits:
        if digit.isdigit():
            result.append(DIGIT_MAP[digit])
        elif digit in ['-', ' ']:
            continue
        else:
            result.append(digit)
    
    return ' '.join(result)


def replace_phone(match) -> str:
    """处理美国电话号码
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    country_code = match.group(1)
    area_code = match.group(2)
    middle = match.group(3)
    last = match.group(4)
    
    # 清理格式字符
    area_code = ''.join(c for c in area_code if c.isdigit())
    
    result = []
    
    # 国家代码
    if country_code:
        country_digits = ''.join(c for c in country_code if c.isdigit())
        if country_digits:
            result.append(verbalize_phone_digits(country_digits))
    
    # 区号
    result.append(verbalize_phone_digits(area_code))
    
    # 中间部分
    result.append(verbalize_phone_digits(middle))
    
    # 最后部分
    result.append(verbalize_phone_digits(last))
    
    return ', '.join(result)


def replace_mobile(match) -> str:
    """处理国际电话号码
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    country_code = match.group(1)
    number = match.group(2)
    
    # 清理格式字符
    country_digits = ''.join(c for c in country_code if c.isdigit())
    number_digits = ''.join(c for c in number if c.isdigit())
    
    # 格式化读法
    country_part = verbalize_phone_digits(country_digits)
    number_part = verbalize_phone_digits(number_digits)
    
    return f"{country_part}, {number_part}"