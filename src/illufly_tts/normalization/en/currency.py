#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文货币规范化处理模块
"""

import re
from typing import Dict, Tuple

from .constants import CURRENCY_SYMBOLS

# 货币表达式
RE_CURRENCY = re.compile(r'([¥￥$€£₩])\s?(\d+(?:\.\d+)?)')


def replace_currency(match) -> str:
    """处理货币金额
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    symbol = match.group(1)
    amount = match.group(2)
    
    # 判断是否为中文环境
    is_chinese = symbol in ['¥', '￥', '元', 'RMB']
    
    # 判断是否被保护
    if '.' in amount and (amount.endswith('00') or len(amount.split('.')[1]) > 2):
        # 可能是保护的内容，直接返回原始文本
        return match.group(0)
        
    # 处理金额
    if '.' in amount:
        integer, decimal = amount.split('.')
        if is_chinese:
            # 中文数字处理
            from ..zh.num import num2str
            integer_text = num2str(integer)
            decimal_text = num2str(decimal)
            return f"{integer_text}点{decimal_text}元"
        else:
            # 英文数字处理
            from .num import verbalize_number
            integer_text = verbalize_number(integer)
            decimal_text = verbalize_number(decimal)
            
            # 处理美元特殊情况
            if symbol == '$':
                if integer == '0':
                    if decimal == '01':
                        return "one cent"
                    else:
                        decimal_num = int(decimal)
                        decimal_text = verbalize_number(str(decimal_num))
                        return f"{decimal_text} cents"
                elif integer == '1' and decimal == '00':
                    return "one dollar"
                elif decimal == '00':
                    return f"{integer_text} dollars"
                else:
                    return f"{integer_text} dollars and {decimal_text} cents"
            else:
                return f"{symbol}{integer_text} point {decimal_text}"
    else:
        if is_chinese:
            # 中文数字处理
            from ..zh.num import num2str
            return f"{num2str(amount)}元"
        else:
            # 英文数字处理
            from .num import verbalize_number
            amount_text = verbalize_number(amount)
            
            # 处理美元特殊情况
            if symbol == '$':
                if amount == '1':
                    return "one dollar"
                else:
                    return f"{amount_text} dollars"
            else:
                return f"{symbol}{amount_text}"