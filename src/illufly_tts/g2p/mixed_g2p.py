#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合语言G2P - 处理中英混合文本的音素转换
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from .base_g2p import BaseG2P
from .chinese_g2p import ChineseG2P
from .english_g2p import EnglishG2P
from ..preprocessing.segmenter import LanguageSegmenter

logger = logging.getLogger(__name__)

class MixedG2P(BaseG2P):
    """混合语言G2P转换器"""
    
    def __init__(self, nltk_data_path: Optional[str] = None):
        """初始化混合语言G2P转换器
        
        Args:
            nltk_data_path: NLTK数据目录路径
        """
        super().__init__()
        self.chinese_g2p = ChineseG2P()
        self.english_g2p = EnglishG2P(nltk_data_path=nltk_data_path)
        self.segmenter = LanguageSegmenter()
        
    def text_to_phonemes(self, text: str) -> str:
        """将文本转换为音素序列
        
        Args:
            text: 输入文本
            
        Returns:
            音素序列
        """
        logger.info(f"处理混合文本: {text}")
        
        # 分割中英文
        segments = self.segmenter.segment(text)
        logger.info(f"文本分段结果: {segments}")
        
        phonemes = []
        for segment in segments:
            text = segment["text"]
            lang = segment["lang"]
            
            if not text.strip():
                continue
                
            if lang == "en":
                logger.info(f"处理英文片段: {text}")
                eng_phonemes = self.english_g2p.text_to_phonemes(text)
                logger.info(f"英文音素结果: {eng_phonemes}")
                if eng_phonemes:
                    phonemes.append(eng_phonemes)
            else:
                logger.info(f"处理中文片段: {text}")
                cn_phonemes = self.chinese_g2p.text_to_phonemes(text)
                logger.info(f"中文音素结果: {cn_phonemes}")
                if cn_phonemes:
                    phonemes.append(cn_phonemes)
        
        result = " ".join(filter(None, phonemes))
        logger.info(f"最终音素序列: {result}")
        return result
    
    def preprocess_text(self, text: str) -> str:
        """
        混合文本预处理
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的文本
        """
        # 基础清理
        text = self.sanitize_text(text)
        
        # 可以在这里添加混合文本的特定预处理逻辑
        
        return text
    
    def _process_segment(self, segment: Dict[str, Any]) -> str:
        """
        处理单个语言段落
        
        Args:
            segment: 语言段落，包含文本和语言类型
            
        Returns:
            该段落的音素序列
        """
        text = segment["text"]
        lang = segment["lang"]
        
        if not text:
            return ""
            
        # 根据语言类型选择相应的G2P
        if lang == "zh":
            return self.chinese_g2p.text_to_phonemes(text)
        elif lang == "en":
            return self.english_g2p.text_to_phonemes(text)
        else:
            # 默认使用中文G2P处理未知语言
            logger.warning(f"未知语言类型: {lang}，使用中文G2P处理")
            return self.chinese_g2p.text_to_phonemes(text)
    
    def _convert_with_language_switches(self, segments: List[Dict[str, Any]]) -> str:
        """
        转换带有语言切换标记的混合文本
        
        Args:
            segments: 语言段落列表
            
        Returns:
            带语言切换标记的音素序列
        """
        result = []
        prev_lang = None
        
        for segment in segments:
            lang = segment["lang"]
            
            # 添加语言切换标记
            if prev_lang and prev_lang != lang:
                if prev_lang == "zh" and lang == "en":
                    result.append("<ZH2EN>")
                elif prev_lang == "en" and lang == "zh":
                    result.append("<EN2ZH>")
                    
            # 处理当前段落
            phonemes = self._process_segment(segment)
            if phonemes:
                result.append(phonemes)
                
            prev_lang = lang
            
        return " ".join(result)
    
    def _convert_segments_separately(self, segments: List[Dict[str, Any]]) -> str:
        """
        分别转换各语言段落，不添加语言切换标记
        
        Args:
            segments: 语言段落列表
            
        Returns:
            音素序列，各语言段落分开处理
        """
        result = []
        
        for segment in segments:
            phonemes = self._process_segment(segment)
            if phonemes:
                result.append(phonemes)
                
        return " ".join(result)
    
    def get_phoneme_set(self) -> List[str]:
        """
        获取混合语言音素集
        
        Returns:
            混合音素列表
        """
        # 合并中英文音素集
        phoneme_set = set()
        phoneme_set.update(self.chinese_g2p.get_phoneme_set())
        phoneme_set.update(self.english_g2p.get_phoneme_set())
        
        # 添加特殊标记
        phoneme_set.update(["<ZH2EN>", "<EN2ZH>", "<NUM>"])
        
        return sorted(list(phoneme_set))
    
    def get_language(self) -> str:
        """
        获取语言代码
        
        Returns:
            语言代码
        """
        return "mixed"
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        处理混合文本，返回音素及详细元数据
        
        Args:
            text: 输入混合文本
            
        Returns:
            包含音素序列及元数据的字典
        """
        if not text:
            return {"phonemes": "", "text": "", "language": self.get_language(), "segments": []}
        
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # 分割文本为不同语言段落
        segments = self.segmenter.process_text(processed_text)
        
        # 处理每个段落
        processed_segments = []
        for segment in segments:
            phonemes = self._process_segment(segment)
            processed_segments.append({
                "text": segment["text"],
                "language": segment["lang"],
                "phonemes": phonemes
            })
        
        # 转换为音素
        phonemes = self._convert_with_language_switches(segments)
        
        # 返回结果
        return {
            "phonemes": phonemes,
            "text": processed_text,
            "language": self.get_language(),
            "segments": processed_segments
        } 