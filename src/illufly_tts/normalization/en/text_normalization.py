#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文文本规范化处理模块
"""

import re
from typing import List, Dict, Optional

from .chronology import RE_DATE, RE_DATE2, RE_TIME, RE_TIME_RANGE
from .chronology import replace_date, replace_date2, replace_time
from .num import RE_DECIMAL_NUM, RE_INTEGER, RE_NUMBER, RE_PERCENTAGE, RE_RANGE, RE_FRACTION
from .num import replace_number, replace_decimal, replace_integer, replace_percentage, replace_range, replace_fraction
from .phonecode import RE_PHONE, RE_MOBILE, replace_phone, replace_mobile
from .currency import RE_CURRENCY, replace_currency
from .constants import SPECIAL_SYMBOLS


class EnTextNormalizer:
    """英文文本规范化处理类"""
    
    def __init__(self):
        """初始化英文文本规范化器"""
        # 英文句子分割器
        self.SENTENCE_SPLITOR = re.compile(r'([.!?][ \n"])')
    
    def _split(self, text: str) -> List[str]:
        """将长文本分割为句子
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        text = self.SENTENCE_SPLITOR.sub(r'\1\n', text)
        text = text.strip()
        sentences = [sentence.strip() for sentence in re.split(r'\n+', text)]
        return sentences
    
    def _post_replace(self, sentence: str) -> str:
        """应用规则，将特殊符号转换为单词
        
        Args:
            sentence: 输入句子
            
        Returns:
            处理后的句子
        """
        # 处理特殊符号
        for symbol, replacement in SPECIAL_SYMBOLS.items():
            sentence = sentence.replace(symbol, replacement)
        
        # 过滤特殊字符
        sentence = re.sub(r'[<=>{}()\[\]#&@^_|…\\]', '', sentence)
        
        return sentence
    
    def normalize_sentence(self, sentence: str) -> str:
        """对单个句子进行规范化处理
        
        Args:
            sentence: 输入句子
            
        Returns:
            规范化后的句子
        """
        # 数字相关NSW规范化
        sentence = RE_DATE.sub(replace_date, sentence)
        sentence = RE_DATE2.sub(replace_date2, sentence)
        
        # 时间处理
        sentence = RE_TIME_RANGE.sub(replace_time, sentence)
        sentence = RE_TIME.sub(replace_time, sentence)
        
        # 电话号码处理
        sentence = RE_PHONE.sub(replace_phone, sentence)
        sentence = RE_MOBILE.sub(replace_mobile, sentence)
        
        # 数字处理
        sentence = RE_PERCENTAGE.sub(replace_percentage, sentence)
        sentence = RE_FRACTION.sub(replace_fraction, sentence)
        sentence = RE_RANGE.sub(replace_range, sentence)
        sentence = RE_INTEGER.sub(replace_integer, sentence)
        sentence = RE_DECIMAL_NUM.sub(replace_decimal, sentence)
        sentence = RE_NUMBER.sub(replace_number, sentence)
        
        # 货币处理
        sentence = RE_CURRENCY.sub(replace_currency, sentence)
        
        # 后处理
        sentence = self._post_replace(sentence)
        
        return sentence
    
    def normalize(self, text: str) -> str:
        """对英文文本进行规范化处理
        
        Args:
            text: 输入英文文本
            
        Returns:
            规范化后的文本
        """
        sentences = self._split(text)
        sentences = [self.normalize_sentence(sent) for sent in sentences]
        return ' '.join(sentences)