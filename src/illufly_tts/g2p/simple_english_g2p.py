#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化的英文G2P - 不依赖NLTK的英文文本到音素转换
"""

import re
import logging
from typing import Dict, Set, List

from .base_g2p import BaseG2P

logger = logging.getLogger(__name__)

class SimpleEnglishG2P(BaseG2P):
    """简化的英文G2P转换器，不依赖NLTK"""
    
    def __init__(self):
        """初始化简化英文G2P"""
        super().__init__()
        
        # 基本英文音素映射
        self.word_to_phonemes = {
            # 常用词
            "hello": "h eh l ou",
            "world": "w er l d",
            "test": "t eh s t",
            "this": "dh ih s",
            "is": "ih z",
            "a": "ah",
            "an": "ae n",
            "and": "ae n d",
            "the": "dh ah",
            "of": "ah v",
            "to": "t uw",
            "in": "ih n",
            "for": "f ao r",
            "on": "aa n",
            "with": "w ih dh",
            "by": "b ay",
            "at": "ae t",
            "from": "f r ah m",
            "about": "ah b aw t",
            
            # 数字
            "zero": "z ih r ou",
            "one": "w ah n",
            "two": "t uw",
            "three": "th r iy",
            "four": "f ao r",
            "five": "f ay v",
            "six": "s ih k s",
            "seven": "s eh v ah n",
            "eight": "ey t",
            "nine": "n ay n",
            "ten": "t eh n",
            
            # 常用缩写
            "tts": "t iy t iy eh s",
            "ai": "ey ay",
            "ml": "eh m eh l",
            "ok": "ou k ey"
        }
        
        # 数字音素
        self.digit_phonemes = {
            '0': 'z ih r ou',
            '1': 'w ah n',
            '2': 't uw',
            '3': 'th r iy',
            '4': 'f ao r',
            '5': 'f ay v',
            '6': 's ih k s',
            '7': 's eh v ah n',
            '8': 'ey t',
            '9': 'n ay n'
        }
        
        # 字母音素
        self.letter_phonemes = {
            'a': 'ey',
            'b': 'b iy',
            'c': 's iy',
            'd': 'd iy',
            'e': 'iy',
            'f': 'eh f',
            'g': 'jh iy',
            'h': 'ey ch',
            'i': 'ay',
            'j': 'jh ey',
            'k': 'k ey',
            'l': 'eh l',
            'm': 'eh m',
            'n': 'eh n',
            'o': 'ou',
            'p': 'p iy',
            'q': 'k y uw',
            'r': 'aa r',
            's': 'eh s',
            't': 't iy',
            'u': 'y uw',
            'v': 'v iy',
            'w': 'd ah b ah l y uw',
            'x': 'eh k s',
            'y': 'w ay',
            'z': 'z iy'
        }
    
    def text_to_phonemes(self, text: str) -> str:
        """将英文文本转换为音素序列
        
        Args:
            text: 英文文本
            
        Returns:
            音素序列
        """
        if not text:
            return ""
        
        # 预处理文本
        text = self.preprocess_text(text)
        
        # 分词 - 简单的空格分割
        words = re.findall(r'\b[a-zA-Z0-9]+\b|[,.!?;:\'"\(\)\[\]\-]', text)
        
        phonemes = []
        for word in words:
            # 标点符号
            if re.match(r'[,.!?;:\'"\(\)\[\]\-]', word):
                phonemes.append(word)
                continue
                
            # 数字
            if re.match(r'^\d+$', word):
                # 按位读数字
                digit_phoneme = []
                for digit in word:
                    if digit in self.digit_phonemes:
                        digit_phoneme.append(self.digit_phonemes[digit])
                phonemes.append(" ".join(digit_phoneme))
                continue
            
            # 单词或其他字符
            word_lower = word.lower()
            if word_lower in self.word_to_phonemes:
                # 已知单词
                phonemes.append(self.word_to_phonemes[word_lower])
            else:
                # 未知单词 - 按字母朗读
                letter_phoneme = []
                for letter in word_lower:
                    if letter in self.letter_phonemes:
                        letter_phoneme.append(self.letter_phonemes[letter])
                    else:
                        letter_phoneme.append(letter)
                phonemes.append(" ".join(letter_phoneme))
        
        return " ".join(phonemes)
    
    def get_phoneme_set(self) -> Set[str]:
        """获取当前G2P使用的所有音素集合
        
        Returns:
            音素集合
        """
        phoneme_set = set()
        
        # 所有已知单词的音素
        for phoneme_str in self.word_to_phonemes.values():
            for phoneme in phoneme_str.split():
                phoneme_set.add(phoneme)
        
        # 字母音素
        for phoneme_str in self.letter_phonemes.values():
            for phoneme in phoneme_str.split():
                phoneme_set.add(phoneme)
        
        # 数字音素
        for phoneme_str in self.digit_phonemes.values():
            for phoneme in phoneme_str.split():
                phoneme_set.add(phoneme)
        
        # 标点符号
        for p in [',', '.', '?', '!', ';', ':', '"', "'", '(', ')', '[', ']', '-', ' ']:
            phoneme_set.add(p)
        
        return phoneme_set
    
    def get_language(self) -> str:
        """获取当前G2P支持的语言
        
        Returns:
            语言代码
        """
        return "en" 