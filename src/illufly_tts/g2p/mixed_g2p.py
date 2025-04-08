#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合语言G2P（Grapheme-to-Phoneme）模块
支持中英文混合文本处理
"""

import os
import re
import logging
from typing import Dict, List, Optional, Union, Any

# 尝试导入nltk用于英文处理
try:
    import nltk
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK不可用，将使用简单的英文处理")

from ..preprocessing.zh_frontend import ZHFrontend
from .base_g2p import BaseG2P
from .custom_zh_g2p import CustomZHG2P  # 导入我们的自定义中文G2P
from .simple_english_g2p import SimpleEnglishG2P  # 导入简化版英文G2P

logger = logging.getLogger(__name__)

class EnglishG2P:
    """英文G2P转换器"""
    
    def __init__(self, nltk_data_path: Optional[str] = None):
        """初始化英文G2P转换器
        
        Args:
            nltk_data_path: NLTK数据目录，如果为None则使用默认位置
        """
        self.nltk_data_path = nltk_data_path
        
        # 检查和下载必要的NLTK数据
        if NLTK_AVAILABLE:
            self._setup_nltk()
            
        # 编译正则表达式
        self.word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        self.number_pattern = re.compile(r'\b[0-9]+\b')
        
        # 常见词发音映射
        self.word_to_phonemes = {
            "hello": "h eh l ou",
            "world": "w er l d",
            "how": "h au",
            "are": "aa r",
            "you": "y uw",
            "the": "dh ah",
            "test": "t eh s t",
            "this": "dh ih s",
            "is": "ih z",
            "a": "ah",
            "an": "ae n",
            "and": "ae n d",
            "but": "b ah t",
            "or": "ao r",
            "not": "n aa t",
            "TTS": "t iy t iy eh s",
            "AI": "ey ay"
        }
    
    def _setup_nltk(self):
        """设置NLTK资源"""
        if self.nltk_data_path:
            nltk.data.path.append(self.nltk_data_path)
        
        try:
            # 下载分词器
            nltk.download('punkt', quiet=True)
            
            # 尝试下载CMU词典
            try:
                nltk.download('cmudict', quiet=True)
                from nltk.corpus import cmudict
                self.cmudict = cmudict.dict()
                logger.info("已加载CMU词典")
            except Exception as e:
                logger.warning(f"加载CMU词典失败: {e}")
                self.cmudict = {}
                
        except Exception as e:
            logger.error(f"NLTK资源下载失败: {e}")
    
    def text_to_phonemes(self, text: str) -> str:
        """将英文文本转换为音素序列
        
        Args:
            text: 英文文本
            
        Returns:
            英文音素序列
        """
        if not NLTK_AVAILABLE:
            # 如果NLTK不可用，使用简单的单词映射
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            phonemes = []
            
            for word in words:
                word_lower = word.lower()
                if word_lower in self.word_to_phonemes:
                    phonemes.append(self.word_to_phonemes[word_lower])
                else:
                    # 简单的字母朗读
                    phonemes.append(" ".join(word_lower))
            
            return " ".join(phonemes)
        
        try:
            # 使用NLTK分词
            tokens = word_tokenize(text)
            phonemes = []
            
            for token in tokens:
                # 检查是否是单词
                if self.word_pattern.match(token):
                    token_lower = token.lower()
                    
                    # 先从我们的映射中查找
                    if token_lower in self.word_to_phonemes:
                        phonemes.append(self.word_to_phonemes[token_lower])
                    # 然后从CMU词典中查找
                    elif hasattr(self, 'cmudict') and token_lower in self.cmudict:
                        # 获取第一个发音变体
                        pron = self.cmudict[token_lower][0]
                        # 移除数字（声调）
                        pron = [re.sub(r'\d+', '', p) for p in pron]
                        phonemes.append(" ".join(pron))
                    else:
                        # 回退到简单的字母朗读
                        phonemes.append(" ".join(token_lower))
                # 数字处理
                elif self.number_pattern.match(token):
                    phonemes.append(self._process_number(token))
                # 标点符号
                else:
                    phonemes.append(token)
            
            return " ".join(phonemes)
            
        except Exception as e:
            logger.error(f"英文文本转音素失败: {e}")
            # 回退到简单处理
            return text
    
    def _process_number(self, number: str) -> str:
        """处理数字，转换为读法
        
        Args:
            number: 数字字符串
            
        Returns:
            数字的读法音素
        """
        # 简单处理，逐位读数
        digit_map = {
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
        
        phonemes = []
        for digit in number:
            if digit in digit_map:
                phonemes.append(digit_map[digit])
            else:
                phonemes.append(digit)
        
        return " ".join(phonemes)

class MixedG2P(BaseG2P):
    """混合语言G2P转换器，支持中英文"""
    
    def __init__(self, dictionary_path: Optional[str] = None, nltk_data_path: Optional[str] = None, 
                 use_custom_zh_g2p: bool = True, use_zhuyin: bool = True, use_special_tones: bool = True):
        """初始化混合语言G2P转换器
        
        Args:
            dictionary_path: 音素字典路径
            nltk_data_path: NLTK数据目录
            use_custom_zh_g2p: 是否使用自定义中文G2P
            use_zhuyin: 是否使用注音符号（针对自定义中文G2P）
            use_special_tones: 是否使用特殊声调符号（针对自定义中文G2P）
        """
        super().__init__()
        
        # 新参数
        self.use_custom_zh_g2p = use_custom_zh_g2p
        
        # 设置词典路径
        if dictionary_path is None:
            # 尝试使用默认词典
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_dict_path = os.path.join(current_dir, '../resources/dictionaries/mixed_dict.txt')
            if os.path.exists(default_dict_path):
                dictionary_path = default_dict_path
                logger.info(f"使用默认混合语言词典: {dictionary_path}")
            else:
                # 尝试查找中文和英文词典
                zh_dict_path = os.path.join(current_dir, '../resources/dictionaries/chinese_dict.txt')
                en_dict_path = os.path.join(current_dir, '../resources/dictionaries/english_dict.txt')
                if os.path.exists(zh_dict_path):
                    logger.info(f"发现中文词典: {zh_dict_path}")
                if os.path.exists(en_dict_path):
                    logger.info(f"发现英文词典: {en_dict_path}")
        
        self.dictionary_path = dictionary_path
        self.nltk_data_path = nltk_data_path
        
        # 加载音素字典
        self.phoneme_dict = {}
        if dictionary_path and os.path.exists(dictionary_path):
            self._load_phoneme_dictionary(dictionary_path)
        
        # 初始化处理器
        if self.use_custom_zh_g2p:
            # 使用自定义的中文G2P
            self.zh_g2p = CustomZHG2P(use_zhuyin=use_zhuyin, use_special_tones=use_special_tones)
            logger.info("使用自定义中文G2P处理器")
        else:
            # 使用原有的处理器
            self.zh_frontend = ZHFrontend()
            logger.info("使用原始ZHFrontend处理器")
            
        # 使用简化版英文G2P，不依赖NLTK
        self.en_g2p = SimpleEnglishG2P()
        logger.info("使用简化版英文G2P处理器")
        
        # 语言检测正则表达式
        self.zh_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.en_pattern = re.compile(r'[a-zA-Z]')
    
    def _load_phoneme_dictionary(self, dictionary_path: str):
        """加载音素字典
        
        Args:
            dictionary_path: 音素字典路径
        """
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        phonemes = ' '.join(parts[1:])
                        self.phoneme_dict[word] = phonemes
            
            logger.info(f"已加载音素字典，包含 {len(self.phoneme_dict)} 个词条")
        except Exception as e:
            logger.error(f"加载音素字典失败: {e}")
    
    def detect_language(self, text: str) -> str:
        """检测文本语言类型
        
        Args:
            text: 输入文本
            
        Returns:
            语言类型："zh"中文，"en"英文，"mixed"混合
        """
        # 更严格地检测英文（至少需要2个连续英文字符才算英文）
        en_pattern = re.compile(r'[a-zA-Z]{2,}')
        zh_pattern = re.compile(r'[\u4e00-\u9fff]')
        
        has_en = bool(en_pattern.search(text))
        has_zh = bool(zh_pattern.search(text))
        
        logger.debug(f"检测语言 - 文本: '{text[:30]}{'...' if len(text) > 30 else ''}', 包含英文: {has_en}, 包含中文: {has_zh}")
        
        # 只要包含中文字符，优先返回中文
        if has_zh:
            return "zh"
        # 只有在纯英文情况下才返回英文
        elif has_en:
            return "en"
        # 其他情况默认当作混合文本
        return "mixed"
    
    def process_chinese(self, text: str) -> str:
        """处理中文文本
        
        Args:
            text: 中文文本
            
        Returns:
            中文音素序列
        """
        try:
            if self.use_custom_zh_g2p:
                # 使用自定义中文G2P处理
                return self.zh_g2p.text_to_phonemes(text)
            else:
                # 使用原有处理器
                return self.zh_frontend.text_to_phonemes(text)
        except Exception as e:
            logger.error(f"中文处理失败: {e}")
            return text
    
    def process_english(self, text: str) -> str:
        """处理英文文本，返回音素序列
        
        Args:
            text: 英文文本
            
        Returns:
            音素序列
        """
        return self.en_g2p.text_to_phonemes(text)
    
    def text_to_phonemes(self, text: str) -> str:
        """将文本转换为音素序列
        
        Args:
            text: 输入文本
            
        Returns:
            音素序列字符串
        """
        if not text:
            return ""
            
        # 预处理文本
        text = self.preprocess_text(text)
        
        # 检测语言
        lang = self.detect_language(text)
        
        if lang == "zh":
            # 中文处理
            return self.process_chinese(text)
        elif lang == "en":
            # 英文处理
            return self.process_english(text)
        else:
            # 混合文本处理
            # 分段处理，识别中英文片段
            phonemes = []
            
            # 使用正则表达式分离中英文
            segments = re.finditer(r'([\u4e00-\u9fff]+|[a-zA-Z0-9]+|[^\u4e00-\u9fffa-zA-Z0-9]+)', text)
            
            for segment in segments:
                seg_text = segment.group(0)
                
                # 检查段落类型
                if re.search(r'[\u4e00-\u9fff]', seg_text):
                    # 中文段落
                    phonemes.append(self.process_chinese(seg_text))
                elif re.search(r'[a-zA-Z]', seg_text):
                    # 英文段落
                    phonemes.append(self.process_english(seg_text))
                else:
                    # 标点符号等其他字符
                    phonemes.append(seg_text)
            
            return " ".join(phonemes)
    
    def process(self, text: str) -> Dict[str, Any]:
        """处理文本，返回详细结果
        
        Args:
            text: 输入文本
            
        Returns:
            包含处理结果的字典
        """
        phonemes = self.text_to_phonemes(text)
        
        return {
            "text": text,
            "phonemes": phonemes,
            "language": self.detect_language(text)
        }
    
    def get_phoneme_set(self) -> set:
        """获取支持的音素集合
        
        Returns:
            音素集合
        """
        phoneme_set = set()
        
        # 添加英文音素
        for phoneme in self.en_g2p.word_to_phonemes.values():
            for ph in phoneme.split():
                phoneme_set.add(ph)
        
        # 添加中文音素
        if self.use_custom_zh_g2p:
            phoneme_set.update(self.zh_g2p.get_phoneme_set())
        else:
            # 添加基本中文音素
            # 声母
            for initial in ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"]:
                phoneme_set.add(initial)
                
            # 韵母
            for final in ["a", "o", "e", "i", "u", "v", "ai", "ei", "ui", "ao", "ou", "iu", "ie", "ue", "ve", "an", "en", "in", "un", "vn", "ang", "eng", "ing", "ong", "er"]:
                phoneme_set.add(final)
                
            # 声调
            for tone in ["1", "2", "3", "4", "5"]:
                phoneme_set.add(tone)
        
        # 标点符号
        for p in [',', '.', '?', '!', ';', ':', '"', "'", '(', ')', '[', ']', '-', ' ']:
            phoneme_set.add(p)
        
        return phoneme_set 