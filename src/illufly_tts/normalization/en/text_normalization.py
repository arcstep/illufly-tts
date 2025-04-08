#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文文本规范化处理模块
"""

import re
from typing import List, Dict, Optional

from .chronology import RE_DATE, RE_DATE2, RE_DATE_RANGE, RE_DATE_RANGE2, RE_TIME, RE_TIME_RANGE, RE_YEAR_RANGE
from .chronology import replace_date, replace_date2, replace_date_range, replace_date_range2, replace_time, replace_year_range
from .num import RE_DECIMAL_NUM, RE_INTEGER, RE_NUMBER, RE_PERCENTAGE, RE_RANGE, RE_FRACTION
from .num import replace_number, replace_decimal, replace_integer, replace_percentage, replace_range, replace_fraction
from .phone import RE_PHONE, RE_MOBILE, replace_phone, replace_mobile
from .currency import RE_CURRENCY, replace_currency
from .constants import SPECIAL_SYMBOLS, DIGIT_MAP
from .num import verbalize_number
from .chronology import verbalize_ordinal


class EnTextNormalizer:
    """英文文本规范化处理类"""
    
    def __init__(self):
        """初始化英文文本规范化器"""
        # 英文句子分割器
        self.SENTENCE_SPLITOR = re.compile(r'([.!?][ \n"])')
        
        # 保护特殊格式
        self.protected_patterns = [
            # 保护URL
            r'https?://\S+',
            # 保护文件路径
            r'[a-zA-Z]:\\[^\\]+(\\[^\\]+)*|/[^/]+(/[^/]+)*',
            # 保护版本号
            r'\d+\.\d+(\.\d+)*',
        ]
        
        # 编译保护模式
        self.protected_regex = re.compile('|'.join(self.protected_patterns))
        
        # 保护邮箱的正则表达式
        self.email_regex = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        
        # 存储被保护的内容
        self.protected_content = {}
        
        # 添加对带序数词后缀日期的正则表达式
        self.RE_ORDINAL_DATE = re.compile(r'([A-Za-z]+)\s+(\d{1,2})(st|nd|rd|th),?\s+(\d{4})', re.IGNORECASE)
    
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
        
        # 处理连续多个空格
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        return sentence
    
    def _protect_special_formats(self, text: str) -> str:
        """保护特殊格式文本，避免被规范化
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本
        """
        result = text
        
        # 定义正则表达式模式
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        url_pattern = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
        
        # 保护邮箱地址
        for match in re.finditer(email_pattern, result):
            placeholder = f'<PROTECTED_EMAIL_{len(self.protected_content)}>'
            self.protected_content[placeholder] = match.group()
            result = result.replace(match.group(), placeholder)
        
        # 保护URL
        for match in re.finditer(url_pattern, result):
            placeholder = f'<PROTECTED_URL_{len(self.protected_content)}>'
            self.protected_content[placeholder] = match.group()
            result = result.replace(match.group(), placeholder)
        
        return result
    
    def _restore_special_formats(self, text: str) -> str:
        """恢复被保护的特殊格式文本
        
        Args:
            text: 包含占位符的文本
            
        Returns:
            恢复原始特殊格式后的文本
        """
        result = text
        
        # 按占位符长度降序排序，确保长的占位符先被替换
        sorted_placeholders = sorted(self.protected_content.keys(), 
                                   key=len, 
                                   reverse=True)
        
        # 恢复所有被保护的内容
        for placeholder in sorted_placeholders:
            result = result.replace(placeholder, self.protected_content[placeholder])
            
        return result
    
    def _handle_ordinal_dates(self, text: str) -> str:
        """处理带有序数词后缀的日期，如 June 1st, 2023
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本
        """
        def replace_ordinal_date(match):
            month = match.group(1)
            day = match.group(2)
            suffix = match.group(3)
            year = match.group(4)
            
            # 转换日期
            day_num = int(day)
            
            # 处理序数词
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
            
            return f"{month} {day_ordinal}, {year_text}"
        
        return self.RE_ORDINAL_DATE.sub(replace_ordinal_date, text)
    
    def normalize_sentence(self, sentence: str) -> str:
        """对单个句子进行规范化处理
        
        Args:
            sentence: 输入句子
            
        Returns:
            规范化后的句子
        """
        # 保护特殊格式
        sentence = self._protect_special_formats(sentence)
        
        # 处理带序数词后缀的日期
        sentence = self._handle_ordinal_dates(sentence)
        
        # 数字相关NSW规范化
        sentence = RE_YEAR_RANGE.sub(replace_year_range, sentence)  # 先处理年份范围
        sentence = RE_DATE_RANGE.sub(replace_date_range, sentence)  # 处理日期范围
        sentence = RE_DATE_RANGE2.sub(replace_date_range2, sentence)  # 处理ISO日期范围
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
        
        # 恢复特殊格式
        sentence = self._restore_special_formats(sentence)
        
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