#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
G2P基类 - 文本到音素转换的基础接口
"""

import abc
from typing import List, Dict, Any, Optional, Union, Set
import re


class BaseG2P(abc.ABC):
    """
    文本到音素(G2P)转换的基类
    
    定义了所有G2P转换器需要实现的接口
    """
    
    @abc.abstractmethod
    def text_to_phonemes(self, text: str) -> str:
        """
        将文本转换为音素序列
        
        Args:
            text: 输入文本
            
        Returns:
            音素序列字符串，使用空格分隔
        """
        pass
    
    @abc.abstractmethod
    def get_phoneme_set(self) -> Set[str]:
        """
        获取当前G2P使用的所有音素集合
        
        Returns:
            音素列表
        """
        pass
    
    def sanitize_text(self, text: str) -> str:
        """
        文本清理，去除不必要的空白字符等
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 这里可以添加其他通用的文本清理规则
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理，子类可以覆盖此方法实现特定语言的预处理
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的文本
        """
        return self.sanitize_text(text)
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        处理文本，返回音素及元数据
        
        Args:
            text: 输入文本
            
        Returns:
            包含音素序列及元数据的字典
        """
        if not text:
            return {"phonemes": "", "text": "", "language": self.get_language()}
        
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # 转换为音素
        phonemes = self.text_to_phonemes(processed_text)
        
        # 返回结果
        return {
            "phonemes": phonemes,
            "text": processed_text,
            "language": self.get_language()
        }
    
    def get_language(self) -> str:
        """
        获取当前G2P支持的语言
        
        Returns:
            语言代码，例如'zh'或'en'
        """
        # 默认语言为空，子类需要覆盖此方法
        return "" 