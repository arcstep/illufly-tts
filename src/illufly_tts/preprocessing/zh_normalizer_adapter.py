#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文标准化适配器 - 集成中文文本标准化功能
"""

import re
import logging
from typing import Dict, Optional, List, Tuple

from .normalizer import TextNormalizer

logger = logging.getLogger(__name__)

class ChineseNormalizerAdapter:
    """中文标准化适配器，集成中文文本标准化功能"""
    
    def __init__(self, misaki_module_path: Optional[str] = None, dictionary_path: Optional[str] = None):
        """
        初始化中文标准化适配器
        
        Args:
            misaki_module_path: 不再使用，保留参数兼容性
            dictionary_path: 自定义词典路径
        """
        self.base_normalizer = TextNormalizer(dictionary_path)
        
        # 编译正则表达式
        self.zh_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.en_pattern = re.compile(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?')
        self.punc_pattern = re.compile(r'([，,。.！!？?；;])\1+')
    
    def protect_english(self, text: str) -> Tuple[str, Dict[str, str]]:
        """保护英文部分不被标准化"""
        # 直接使用TextNormalizer的方法
        return self.base_normalizer.protect_english(text)
    
    def restore_protected(self, text: str, preserved: Dict[str, str]) -> str:
        """恢复被保护的内容"""
        # 直接使用TextNormalizer的方法
        return self.base_normalizer.restore_protected(text, preserved)
    
    def clean_punctuation(self, text: str) -> str:
        """清理重复的标点符号"""
        # 直接使用TextNormalizer的方法
        return self.base_normalizer.clean_punctuation(text)
    
    def normalize(self, text: str) -> str:
        """
        标准化中文文本
        
        Args:
            text: 输入中文文本
            
        Returns:
            标准化后的文本
        """
        # 检测是否包含中文
        has_chinese = bool(self.zh_pattern.search(text))
        
        if has_chinese:
            # 使用TextNormalizer处理中文文本
            return self.base_normalizer.normalize(text)
        else:
            # 非中文文本，简单处理
            return text
    
    def process_text(self, text: str) -> str:
        """
        处理文本，标准化中文
        
        This is for backward compatibility
        """
        return self.normalize(text) 