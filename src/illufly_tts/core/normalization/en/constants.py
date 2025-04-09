#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文文本规范化常量定义
"""

# 数字和单词的映射
DIGIT_MAP = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen', 
    '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
    '18': 'eighteen', '19': 'nineteen'
}

# 十位数
TENS_MAP = {
    '2': 'twenty', '3': 'thirty', '4': 'forty', '5': 'fifty',
    '6': 'sixty', '7': 'seventy', '8': 'eighty', '9': 'ninety'
}

# 数量级
MAGNITUDE_MAP = {
    2: 'hundred',
    3: 'thousand',
    6: 'million',
    9: 'billion',
    12: 'trillion',
    15: 'quadrillion'
}

# 序数词后缀
ORDINAL_SUFFIXES = {
    '1': 'st', '2': 'nd', '3': 'rd'
}

# 序数词映射
ORDINAL_MAP = {
    '1': 'first', '2': 'second', '3': 'third', '4': 'fourth', '5': 'fifth',
    '6': 'sixth', '7': 'seventh', '8': 'eighth', '9': 'ninth', '10': 'tenth',
    '11': 'eleventh', '12': 'twelfth', '13': 'thirteenth', '14': 'fourteenth',
    '15': 'fifteenth', '16': 'sixteenth', '17': 'seventeenth', '18': 'eighteenth',
    '19': 'nineteenth', '20': 'twentieth', '30': 'thirtieth', '40': 'fortieth',
    '50': 'fiftieth', '60': 'sixtieth', '70': 'seventieth', '80': 'eightieth',
    '90': 'ninetieth', '100': 'hundredth', '1000': 'thousandth'
}

# 货币符号映射
CURRENCY_SYMBOLS = {
    '$': 'dollar',
    '€': 'euro',
    '£': 'pound',
    '¥': 'yen',
    '₩': 'won'
}

# 单位映射
UNIT_MAP = {
    'km': 'kilometer',
    'km²': 'square kilometer',
    'km³': 'cubic kilometer',
    'm': 'meter',
    'm²': 'square meter',
    'm³': 'cubic meter',
    'cm': 'centimeter',
    'cm²': 'square centimeter',
    'cm³': 'cubic centimeter',
    'mm': 'millimeter',
    'kg': 'kilogram',
    'g': 'gram',
    'mg': 'milligram',
    'lb': 'pound',
    'oz': 'ounce',
    'l': 'liter',
    'ml': 'milliliter',
    's': 'second',
    'min': 'minute',
    'h': 'hour',
    '°C': 'degree celsius',
    '°F': 'degree fahrenheit'
}

# 特殊符号映射
SPECIAL_SYMBOLS = {
    '+': 'plus',
    '-': 'minus',
    '=': 'equals',
    '*': 'times',
    '/': 'divided by',
    '%': 'percent',
    '&': 'and',
    '@': 'at',
    '#': 'number',
    '^': 'caret',
    '~': 'tilde',
    '<': 'less than',
    '>': 'greater than',
    '≤': 'less than or equal to',
    '≥': 'greater than or equal to',
    '≠': 'not equal to',
    '≈': 'approximately equal to',
    '∞': 'infinity',
    'π': 'pi',
    'α': 'alpha',
    'β': 'beta',
    'γ': 'gamma',
    'δ': 'delta',
    'ε': 'epsilon',
    'θ': 'theta',
    'λ': 'lambda',
    'μ': 'mu',
    'σ': 'sigma',
    'τ': 'tau',
    'φ': 'phi',
    'ω': 'omega'
}