#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文G2P (Grapheme-to-Phoneme) 模块
将英文文本转换为音素序列
"""

import re
import os
import logging
from typing import Dict, Any, List, Optional, Set

from .base_g2p import BaseG2P

logger = logging.getLogger(__name__)

class EnglishG2P(BaseG2P):
    """英文G2P转换器"""
    
    def __init__(self, nltk_data_path: Optional[str] = None, dictionary_path: Optional[str] = None):
        """初始化英文G2P转换器
        
        Args:
            nltk_data_path: NLTK数据目录路径
            dictionary_path: 可选的音素字典路径
        """
        super().__init__()
        
        # 设置词典路径
        if dictionary_path is None:
            # 尝试使用默认词典
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_dict_path = os.path.join(current_dir, '../resources/dictionaries/english_dict.txt')
            if os.path.exists(default_dict_path):
                dictionary_path = default_dict_path
                logger.info(f"使用默认英文词典: {dictionary_path}")
        
        self.nltk_data_path = nltk_data_path
        self.dictionary_path = dictionary_path
        
        # NLTK环境变量设置
        if nltk_data_path:
            os.environ['NLTK_DATA'] = nltk_data_path
            logger.info(f"设置NLTK_DATA环境变量: {nltk_data_path}")
        
        # 初始化NLTK和g2p_en（如果可用）
        self._init_nltk()
        self._init_g2p_en()
        
        # 加载音素字典
        self.phoneme_dict = {}
        if dictionary_path:
            self._load_dictionary(dictionary_path)
        
        # 常见词发音映射表，作为备用
        self.word_to_phonemes = {
            "hello": "hh eh l ow",
            "world": "w er l d",
            "how": "hh aw",
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
            "tts": "t iy t iy eh s",
            "ai": "ey ay"
        }
        
        # 编译正则表达式
        self.word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        self.number_pattern = re.compile(r'\b[0-9]+\b')
    
    def _init_nltk(self):
        """初始化NLTK资源"""
        try:
            import nltk
            
            # 设置NLTK数据路径
            if self.nltk_data_path:
                nltk.data.path.append(self.nltk_data_path)
                logger.info(f"添加NLTK数据路径: {self.nltk_data_path}")
            
            # 打印NLTK数据搜索路径
            logger.info(f"NLTK数据搜索路径: {nltk.data.path}")
            
            # 下载必要的资源
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('cmudict', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                logger.info("成功下载averaged_perceptron_tagger资源")
                
                # 检查tagger资源
                data_path = nltk.data.find('taggers/averaged_perceptron_tagger')
                logger.info(f"找到NLTK tagger资源: {data_path}")
                
                # 检查g2p_en需要的资源
                try:
                    taggers_dir = os.path.dirname(data_path)
                    logger.info(f"资源目录内容: {os.listdir(taggers_dir)}")
                    
                    # 为g2p_en准备tagger资源
                    g2p_dir = os.path.join(taggers_dir, 'averaged_perceptron_tagger_eng')
                    logger.info(f"g2p_en资源目录已存在: {g2p_dir}")
                    
                    if os.path.exists(g2p_dir):
                        logger.info(f"g2p_en资源目录内容: {os.listdir(g2p_dir)}")
                except Exception as e:
                    logger.warning(f"检查g2p_en资源时出错: {e}")
                
            except Exception as e:
                logger.error(f"下载NLTK资源失败: {e}")
            
            self.nltk_available = True
            self.cmu_dict = None
            
            # 尝试加载CMU词典
            try:
                from nltk.corpus import cmudict
                self.cmu_dict = cmudict.dict()
                logger.info("成功加载CMU词典")
            except Exception as e:
                logger.warning(f"加载CMU词典失败: {e}")
            
            return True
            
        except ImportError:
            logger.warning("NLTK不可用，将使用基本的单词映射")
            self.nltk_available = False
            return False
    
    def _init_g2p_en(self):
        """初始化g2p_en"""
        try:
            import g2p_en
            self.g2p_en = g2p_en.G2p()
            logger.info("成功导入g2p_en模块")
            self.g2p_en_available = True
            
            # 测试g2p_en
            logger.info("尝试初始化G2P转换器...")
            test_text = "Hello world"
            test_result = self.g2p_en(test_text)
            logger.info("G2P转换器初始化成功")
            
            return True
            
        except ImportError:
            logger.warning("g2p_en不可用，将使用备用方法")
            self.g2p_en_available = False
            return False
        except Exception as e:
            logger.error(f"初始化g2p_en失败: {e}")
            self.g2p_en_available = False
            return False
    
    def _load_dictionary(self, dictionary_path: str):
        """加载音素字典
        
        Args:
            dictionary_path: 字典文件路径
        """
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0].lower()
                        phonemes = ' '.join(parts[1:])
                        self.phoneme_dict[word] = phonemes
            
            logger.info(f"已加载英文音素字典，包含 {len(self.phoneme_dict)} 个词条")
        except Exception as e:
            logger.error(f"加载英文音素字典失败: {e}")
    
    def _process_number(self, number: str) -> str:
        """处理数字，转换为读法
        
        Args:
            number: 数字字符串
            
        Returns:
            数字的读法音素
        """
        # 简单处理，逐位读数
        digit_map = {
            '0': 'z ih r ow',
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
        
        return ' '.join(phonemes)
    
    def _process_with_g2p_en(self, text: str) -> str:
        """使用g2p_en处理文本
        
        Args:
            text: 英文文本
            
        Returns:
            音素序列
        """
        try:
            logger.info(f"使用g2p_en转换文本: {text}")
            phonemes = self.g2p_en(text)
            logger.info(f"g2p_en转换结果: {phonemes}")
            return ' '.join(phonemes)
        except Exception as e:
            logger.error(f"g2p_en转换失败: {e}")
            return self._process_with_fallback(text)
    
    def _process_with_cmu(self, text: str) -> str:
        """使用CMU词典处理文本
        
        Args:
            text: 英文文本
            
        Returns:
            音素序列
        """
        try:
            import nltk
            words = nltk.word_tokenize(text.lower())
            phonemes = []
            
            for word in words:
                # 首先在词典中查找
                if word in self.phoneme_dict:
                    phonemes.append(self.phoneme_dict[word])
                # 然后在CMU词典中查找
                elif self.cmu_dict and word in self.cmu_dict:
                    # 获取第一个发音
                    pron = self.cmu_dict[word][0]
                    # 移除数字 (重音标记)
                    pron = [re.sub(r'\d+', '', p) for p in pron]
                    phonemes.append(' '.join(pron))
                # 最后在备用映射中查找
                elif word in self.word_to_phonemes:
                    phonemes.append(self.word_to_phonemes[word])
                # 标点符号保持不变
                elif word in ",.!?;:\"'()[]{}":
                    phonemes.append(word)
                # 数字处理
                elif word.isdigit():
                    phonemes.append(self._process_number(word))
                # 回退到简单的字母朗读
                else:
                    # 如果是缩写，逐字母朗读
                    if word.isupper() and len(word) > 1:
                        letter_phonemes = []
                        for letter in word:
                            if letter in self.word_to_phonemes:
                                letter_phonemes.append(self.word_to_phonemes[letter])
                            else:
                                letter_phonemes.append(letter)
                        phonemes.append(' '.join(letter_phonemes))
                    else:
                        # 尝试推测发音
                        phonemes.append(' '.join(word))
            
            return ' '.join(phonemes)
            
        except Exception as e:
            logger.error(f"CMU词典处理失败: {e}")
            return self._process_with_fallback(text)
    
    def _process_with_fallback(self, text: str) -> str:
        """使用备用方法处理文本
        
        Args:
            text: 英文文本
            
        Returns:
            音素序列
        """
        words = re.findall(r'\b[a-zA-Z]+\b|\b\d+\b|[,.!?;:\'"()]', text.lower())
        phonemes = []
        
        for word in words:
            if word in self.phoneme_dict:
                phonemes.append(self.phoneme_dict[word])
            elif word in self.word_to_phonemes:
                phonemes.append(self.word_to_phonemes[word])
            elif word.isdigit():
                phonemes.append(self._process_number(word))
            elif word in ",.!?;:\"'()[]{}":
                phonemes.append(word)
            else:
                # 简单的字母朗读
                phonemes.append(' '.join(word))
        
        return ' '.join(phonemes)
    
    def text_to_phonemes(self, text: str) -> str:
        """将文本转换为音素序列
        
        Args:
            text: 输入文本
            
        Returns:
            音素序列
        """
        # 清理文本
        text = self.sanitize_text(text)
        
        # 如果g2p_en可用，优先使用
        if hasattr(self, 'g2p_en_available') and self.g2p_en_available:
            return self._process_with_g2p_en(text)
        
        # 如果NLTK和CMU词典可用，使用它们
        if hasattr(self, 'nltk_available') and self.nltk_available and self.cmu_dict:
            return self._process_with_cmu(text)
        
        # 回退到基本处理
        return self._process_with_fallback(text)
    
    def get_phoneme_set(self) -> Set[str]:
        """获取支持的音素集合
        
        Returns:
            音素集合
        """
        # 基础英文音素集 (CMU风格)
        phoneme_set = set([
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
            'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
            'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
            # 小写版本
            'aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey',
            'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy',
            'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh',
            # 重音标记
            '0', '1', '2'
        ])
        
        # 添加来自词典的音素
        for phonemes in self.phoneme_dict.values():
            for phoneme in phonemes.split():
                phoneme_set.add(phoneme)
        
        # 添加来自备用映射的音素
        for phonemes in self.word_to_phonemes.values():
            for phoneme in phonemes.split():
                phoneme_set.add(phoneme)
        
        return phoneme_set
    
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
            "language": "en"
        } 