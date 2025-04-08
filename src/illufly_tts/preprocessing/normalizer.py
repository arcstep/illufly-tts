#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本标准化模块 - 处理文本中的数字、符号等
"""

import re
import os
import json
import logging
from typing import List, Dict, Optional, Pattern, Match, Any

# 直接导入PaddleSpeech版本的文本正规化
from ..zh_normalization.text_normalization import TextNormalizer as PaddleTextNormalizer

logger = logging.getLogger(__name__)

class TextNormalizer:
    """基础文本标准化器，集成了中文文本标准化功能"""
    
    def __init__(self, dictionary_path: Optional[str] = None):
        """
        初始化文本标准化器
        
        Args:
            dictionary_path: 标准化词典路径，包含自定义替换规则
        """
        self.dictionary = {}
        
        if dictionary_path and os.path.exists(dictionary_path):
            try:
                with open(dictionary_path, 'r', encoding='utf-8') as f:
                    self.dictionary = json.load(f)
                logger.info(f"已加载标准化词典，包含 {len(self.dictionary)} 条规则")
            except Exception as e:
                logger.error(f"加载标准化词典失败: {e}")
        
        # 英文保护的正则表达式
        self.en_pattern = re.compile(r'[a-zA-Z]+(?:\'[a-zA-Z]+)?')
        
        # 数字标准化的正则表达式
        self.number_pattern = re.compile(r'([0-9]+(?:\.[0-9]+)?)')
        
        # 简单句子切分正则表达式
        self.sentence_pattern = re.compile(r'([.!?。！？;；])')
        
        # 标点符号清理
        self.punc_pattern = re.compile(r'([，,。.！!？?；;])\1+')
        
        # 中文字符检测
        self.zh_pattern = re.compile(r'[\u4e00-\u9fff]')
        
        # 初始化PaddleSpeech的中文文本正规化器
        self.zh_normalizer = PaddleTextNormalizer()
        
        # 编译词典中的正则表达式
        self._compile_dictionary()
    
    def _compile_dictionary(self):
        """编译词典中的正则表达式，提高性能"""
        self.compiled_patterns = []
        for pattern, replacement in self.dictionary.items():
            try:
                # 尝试添加词边界，但对于特殊规则可能会失败
                compiled = re.compile(r'\b' + pattern + r'\b')
                self.compiled_patterns.append((compiled, replacement, True))
            except re.error:
                # 如果添加边界导致错误，则使用原始模式
                try:
                    compiled = re.compile(pattern)
                    self.compiled_patterns.append((compiled, replacement, False))
                except re.error as e:
                    logger.error(f"正则表达式编译失败: {pattern} - {e}")
    
    def split_sentences(self, text: str) -> List[str]:
        """
        将长文本分割成句子
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        sentences = []
        splits = self.sentence_pattern.split(text)
        
        current = ""
        for i in range(0, len(splits), 2):
            if i < len(splits):
                current += splits[i]
            if i + 1 < len(splits):
                current += splits[i+1]
                sentences.append(current.strip())
                current = ""
        
        if current:
            sentences.append(current.strip())
            
        return [s for s in sentences if s]
    
    def protect_english(self, text: str) -> tuple:
        """
        保护英文部分不被标准化
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本，和被保护的内容与其占位符的映射
        """
        preserved = {}
        
        def replace(match):
            token = f"__EN_{len(preserved):03d}__"
            preserved[token] = match.group(0)
            return token
            
        protected_text = self.en_pattern.sub(replace, text)
        return protected_text, preserved
    
    def restore_protected(self, text: str, preserved: Dict[str, str]) -> str:
        """
        恢复被保护的内容
        
        Args:
            text: 处理过的文本
            preserved: 被保护的内容映射
            
        Returns:
            恢复后的文本
        """
        result = text
        for token, original in preserved.items():
            result = result.replace(token, original)
        return result
    
    def normalize_chinese(self, text: str) -> str:
        """
        标准化中文文本，使用PaddleSpeech的实现
        
        Args:
            text: 中文文本
            
        Returns:
            标准化后的文本
        """
        # 使用PaddleSpeech的文本正规化器
        normalized_sentences = self.zh_normalizer.normalize(text)
        return "，".join(normalized_sentences) if normalized_sentences else text
    
    def clean_punctuation(self, text: str) -> str:
        """
        清理重复的标点符号
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 1. 将中英文标点统一化
        text = text.replace(',', '，')
        text = text.replace('.', '。')
        text = text.replace('!', '！')
        text = text.replace('?', '？')
        text = text.replace(';', '；')
        
        # 2. 删除重复的标点
        while True:
            new_text = self.punc_pattern.sub(r'\1', text)
            if new_text == text:
                break
            text = new_text
        
        return text
    
    def apply_dictionary(self, text: str) -> str:
        """
        应用自定义词典进行替换
        
        Args:
            text: 输入文本
            
        Returns:
            替换后的文本
        """
        result = text
        
        # 处理特定的英文缩写
        if "TTS" in result:
            result = result.replace("TTS", "语音合成")
        if "AI" in result:
            result = result.replace("AI", "人工智能")
        
        # 应用预编译的正则表达式
        if hasattr(self, 'compiled_patterns'):
            for pattern, replacement, has_boundary in self.compiled_patterns:
                result = pattern.sub(replacement, result)
        else:
            # 如果字典还没有编译，使用旧的方法
            for pattern, replacement in self.dictionary.items():
                try:
                    regex = re.compile(r'\b' + pattern + r'\b')
                    result = regex.sub(replacement, result)
                except re.error:
                    try:
                        result = re.sub(pattern, replacement, result)
                    except re.error as e:
                        logger.error(f"正则表达式替换失败: {pattern} - {e}")
                        
        return result
    
    def normalize(self, text: str) -> str:
        """
        标准化文本，自动处理中文和英文
        
        Args:
            text: 输入文本
            
        Returns:
            标准化后的文本
        """
        logger.debug(f"标准化文本: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # 应用词典替换
        text = self.apply_dictionary(text)
        
        # 检测是否包含中文
        has_chinese = bool(self.zh_pattern.search(text))
        
        if has_chinese:
            # 处理混合文本
            # 1. 保护英文
            protected_text, preserved = self.protect_english(text)
            
            # 2. 中文文本标准化
            normalized = self.normalize_chinese(protected_text)
            
            # 3. 恢复被保护的内容
            result = self.restore_protected(normalized, preserved)
        else:
            # 纯英文文本，简单处理
            result = text
        
        # 清理标点符号
        result = self.clean_punctuation(result)
        
        logger.debug(f"标准化结果: '{result[:50]}{'...' if len(result) > 50 else ''}'")
        return result 