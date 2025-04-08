#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文货币规范化处理模块
"""

import re
from typing import Dict, Tuple

from .constants import CURRENCY_SYMBOLS

# 货币表达式
RE_CURRENCY = re.compile(r'([$€£¥₩])(\d+(\.\d+)?)')


def replace_currency(match) -> str:
    """处理货币表达式
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    symbol, amount, decimal = match.groups()
    
    # 获取货币名称
    currency_name = CURRENCY_SYMBOLS.get(symbol, 'currency')
    
    # 拆分整数和小数部分
    if decimal:
        integer_part, decimal_part = amount.split('.')
    else:
        integer_part, decimal_part = amount, None
    
    # 特殊处理1
    if integer_part == '1' and not decimal_part:
        return f"one {currency_name}"
    
    # 处理整数部分
    from .num import verbalize_number
    integer_text = verbalize_number(integer_part)
    
    # 处理小数部分
    if decimal_part:
        # 处理美元的美分表示
        if symbol == '$' and len(decimal_part) <= 2:
            if decimal_part == '01':
                return f"{integer_text} {currency_name}s and one cent"
            else:
                cent_value = int(decimal_part)
                if len(decimal_part) == 1:
                    cent_value *= 10
                cent_text = verbalize_number(str(cent_value))
                
                if cent_value == 1:
                    return f"{integer_text} {currency_name}s and one cent"
                else:
                    return f"{integer_text} {currency_name}s and {cent_text} cents"
        else:
            # 其他货币小数读法
            decimal_text = ' '.join(DIGIT_MAP[digit] for digit in decimal_part)
            return f"{integer_text} point {decimal_text} {currency_name}s"
    else:
        # 只有整数部分
        if integer_part == '1':
            return f"one {currency_name}"
        else:
            return f"{integer_text} {currency_name}s"