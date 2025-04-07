#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文G2P - 将中文文本转换为拼音音素序列
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple, Union

try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    logging.warning("pypinyin模块不可用，中文G2P将不能正常工作")

from .base_g2p import BaseG2P

logger = logging.getLogger(__name__)

class ChineseG2P(BaseG2P):
    """中文文本到拼音音素转换器"""
    
    def __init__(self, 
                 use_dict: bool = True, 
                 dict_path: Optional[str] = None,
                 pinyin_style: str = "TONE3"):
        """
        初始化中文G2P
        
        Args:
            use_dict: 是否使用词典辅助转换
            dict_path: 词典路径，若为None则使用默认词典
            pinyin_style: 拼音风格，支持TONE3(数字音调)和NORMAL(无音调)
        """
        self.use_dict = use_dict
        self.dict_path = dict_path
        
        # 设置拼音风格
        if pinyin_style == "TONE3" and PYPINYIN_AVAILABLE:
            self.pinyin_style = Style.TONE3
        elif pinyin_style == "NORMAL" and PYPINYIN_AVAILABLE:
            self.pinyin_style = Style.NORMAL
        else:
            self.pinyin_style = None
            
        # 初始化声母韵母映射
        self._init_pinyin_maps()
        
        # 加载拼音词典
        self.load_pinyin_dictionary()
    
    def _init_pinyin_maps(self):
        """初始化拼音声母韵母映射"""
        # 声母列表
        self.initials = [
            'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
            'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w'
        ]
        
        # 韵母列表
        self.finals = [
            'a', 'o', 'e', 'i', 'u', 'v', 'ai', 'ei', 'ui', 'ao', 'ou', 'iu',
            'ie', 've', 'er', 'an', 'en', 'in', 'un', 'vn', 'ang', 'eng', 'ing', 'ong'
        ]
        
        # 完整拼音列表
        self.valid_syllables = set()
        
        # 特殊音节（不遵循声母+韵母规则的拼音）
        self.special_syllables = {
            'a', 'o', 'e', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng', 'er',
            'yi', 'wu', 'yu', 'yin', 'yun', 'ye'
        }
        
        # 生成有效拼音组合
        for i in self.initials:
            for f in self.finals:
                syllable = i + f
                self.valid_syllables.add(syllable)
                
        # 添加特殊音节
        self.valid_syllables.update(self.special_syllables)
        
        # 音调映射
        self.tone_map = {
            '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': ''
        }
    
    def load_pinyin_dictionary(self):
        """加载自定义拼音词典"""
        self.pinyin_dict = {}
        
        if not self.use_dict:
            return
            
        if self.dict_path and os.path.exists(self.dict_path):
            try:
                with open(self.dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().split()
                            if len(parts) > 1:
                                word = parts[0]
                                pinyin_text = ' '.join(parts[1:])
                                self.pinyin_dict[word] = pinyin_text
                logger.info(f"已加载拼音词典，包含 {len(self.pinyin_dict)} 个条目")
            except Exception as e:
                logger.error(f"加载拼音词典失败: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """
        中文文本预处理
        
        Args:
            text: 输入中文文本
            
        Returns:
            预处理后的文本
        """
        # 基础清理
        text = self.sanitize_text(text)
        
        # 处理常见标点
        text = text.replace('，', ' ')
        text = text.replace('。', ' ')
        text = text.replace('？', ' ')
        text = text.replace('！', ' ')
        text = text.replace('：', ' ')
        text = text.replace('；', ' ')
        text = text.replace('"', ' ')
        text = text.replace('"', ' ')
        text = text.replace('「', ' ')
        text = text.replace('」', ' ')
        text = text.replace('（', ' ')
        text = text.replace('）', ' ')
        text = text.replace('【', ' ')
        text = text.replace('】', ' ')
        
        # 合并多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _format_tone(self, pinyin_with_tone: str) -> str:
        """
        格式化带音调的拼音
        
        Args:
            pinyin_with_tone: 带数字音调的拼音（TONE3风格）
            
        Returns:
            格式化后的拼音
        """
        if not pinyin_with_tone[-1].isdigit():
            # 没有音调，添加轻声调
            return pinyin_with_tone + self.tone_map['5']
            
        tone_num = pinyin_with_tone[-1]
        pinyin_base = pinyin_with_tone[:-1]
        
        if tone_num in self.tone_map:
            return pinyin_base + self.tone_map[tone_num]
        else:
            return pinyin_with_tone
    
    def _clean_pinyin(self, pinyin_list: List[List[str]]) -> List[str]:
        """
        清理pypinyin返回的拼音列表
        
        Args:
            pinyin_list: pypinyin返回的嵌套列表
            
        Returns:
            清理后的拼音列表
        """
        result = []
        for item in pinyin_list:
            if item and item[0]:
                # 获取拼音并格式化
                py = item[0]
                # 可选：格式化拼音的音调
                # py = self._format_tone(py)
                result.append(py)
                
        return result
    
    def _syllable_to_phoneme(self, syllable: str) -> str:
        """
        将拼音音节转换为音素序列
        
        Args:
            syllable: 拼音音节，可能带有音调
            
        Returns:
            对应的音素序列
        """
        # 提取基本拼音（不含音调）
        base = re.sub(r'\d', '', syllable).lower()
        
        # 提取音调（如果有）
        tone = ''
        if syllable and syllable[-1].isdigit():
            tone = syllable[-1]
        
        # 尝试分解为声母和韵母
        initial = ''
        final = base
        
        # 查找最长的可能声母
        for i in sorted(self.initials, key=len, reverse=True):
            if base.startswith(i):
                initial = i
                final = base[len(i):]
                break
                
        # 验证韵母
        if final not in self.finals and base not in self.special_syllables:
            logger.warning(f"无效的拼音音节: {syllable}, 基本拼音: {base}")
            
        # 构建音素序列
        if initial and final:
            if tone:
                return f"{initial} {final}{tone}"
            else:
                return f"{initial} {final}"
        else:
            if tone:
                return f"{final}{tone}"
            else:
                return final
    
    def _word_to_phonemes(self, word: str) -> str:
        """
        将单个词转换为音素序列
        
        Args:
            word: 中文词
            
        Returns:
            音素序列
        """
        # 检查词典
        if self.use_dict and word in self.pinyin_dict:
            return self.pinyin_dict[word]
            
        # 使用pypinyin转换
        if PYPINYIN_AVAILABLE:
            try:
                py_list = pinyin(word, style=self.pinyin_style)
                py_cleaned = self._clean_pinyin(py_list)
                
                # 转换为音素
                phonemes = []
                for syll in py_cleaned:
                    ph = self._syllable_to_phoneme(syll)
                    phonemes.append(ph)
                    
                return ' '.join(phonemes)
            except Exception as e:
                logger.error(f"转换为拼音失败: {e}")
                
        # 简单备用方案
        return word
    
    def text_to_phonemes(self, text: str) -> str:
        """
        将中文文本转换为音素序列
        
        Args:
            text: 输入中文文本
            
        Returns:
            音素序列字符串，使用空格分隔
        """
        if not text:
            return ""
            
        # 检查整句是否在词典中
        if self.use_dict and text in self.pinyin_dict:
            return self.pinyin_dict[text]
            
        # 按词处理
        words = list(text)  # 简单实现，每个字符当作一个词
        phonemes = []
        
        for word in words:
            if word.strip():  # 跳过空白字符
                ph = self._word_to_phonemes(word)
                if ph:
                    phonemes.append(ph)
                    
        return ' '.join(phonemes)
    
    def get_phoneme_set(self) -> List[str]:
        """
        获取中文拼音音素集
        
        Returns:
            拼音音素列表
        """
        phonemes = set()
        
        # 添加所有声母
        phonemes.update(self.initials)
        
        # 添加所有韵母及其与音调的组合
        for final in self.finals:
            phonemes.add(final)
            # 添加带音调的版本
            for tone in range(1, 6):
                phonemes.add(f"{final}{tone}")
                
        return sorted(list(phonemes))
    
    def get_language(self) -> str:
        """
        获取语言代码
        
        Returns:
            语言代码
        """
        return "zh" 