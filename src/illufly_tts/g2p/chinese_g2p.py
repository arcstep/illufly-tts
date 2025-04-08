#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文G2P - 文本到音素转换模块
使用ZHFrontend实现与Misaki格式一致的输出
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple, Any

import cn2an
import jieba
from pypinyin import lazy_pinyin, Style

from .base_g2p import BaseG2P
from .token import MToken  # 确保MToken类已复制
from .zh_frontend import ZHFrontend  # 直接导入已复制的ZHFrontend
from .transcription import pinyin_to_ipa  # 如果仍需IPA转换

logger = logging.getLogger(__name__)

class ChineseG2P(BaseG2P):
    """中文G2P转换器 - 直接使用ZHFrontend"""
    
    def __init__(self, unk='❓', en_callable=None):
        """初始化中文G2P转换器
        
        Args:
            unk: 未知符号的表示
            en_callable: 处理英文的回调函数
        """
        super().__init__()
        self.unk = unk
        self.en_callable = en_callable
        
        # 直接初始化ZHFrontend
        self.frontend = ZHFrontend(unk=unk)
        
        # 继承ZHG2P的静态方法以支持legacy模式
        self.retone = self._retone
        self.py2ipa = self._py2ipa
        self.word2ipa = self._word2ipa
        self.map_punctuation = self._map_punctuation
    
    @staticmethod
    def _retone(p):
        """处理音调标记，与Misaki保持一致"""
        p = p.replace('˧˩˧', '↓')  # third tone
        p = p.replace('˧˥', '↗')   # second tone
        p = p.replace('˥˩', '↘')   # fourth tone
        p = p.replace('˥', '→')    # first tone
        p = p.replace(chr(635)+chr(809), 'ɨ').replace(chr(633)+chr(809), 'ɨ')
        assert chr(809) not in p, p
        return p
    
    @staticmethod
    def _py2ipa(py):
        """将单个拼音转换为IPA音素"""
        return ''.join(ChineseG2P._retone(p) for p in pinyin_to_ipa(py)[0])
    
    @staticmethod
    def _word2ipa(w):
        """将词转换为IPA音素"""
        pinyins = lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)
        return ''.join(ChineseG2P._py2ipa(py) for py in pinyins)
    
    @staticmethod
    def _map_punctuation(text):
        """标点符号处理"""
        text = text.replace('、', ', ').replace('，', ', ')
        text = text.replace('。', '. ').replace('．', '. ')
        text = text.replace('！', '! ')
        text = text.replace('：', ': ')
        text = text.replace('；', '; ')
        text = text.replace('？', '? ')
        text = text.replace('«', ' "').replace('»', '" ')
        text = text.replace('《', ' "').replace('》', '" ')
        text = text.replace('「', ' "').replace('」', '" ')
        text = text.replace('【', ' "').replace('】', '" ')
        text = text.replace('（', ' (').replace('）', ') ')
        return text.strip()
    
    def legacy_call(self, text):
        """仅为兼容性保留的legacy_call处理逻辑"""
        is_zh = re.match(r'[\u4E00-\u9FFF]', text[0]) is not None
        result = ''
        for segment in re.findall(r'[\u4E00-\u9FFF]+|[^\u4E00-\u9FFF]+', text):
            if is_zh:
                words = jieba.lcut(segment, cut_all=False)
                segment = ' '.join(self.word2ipa(w) for w in words)
            result += segment
            is_zh = not is_zh
        return result.replace(chr(815), '')
    
    def convert_to_ipa(self, phonemes: str) -> str:
        """将注音符号转换为IPA格式
        
        Args:
            phonemes: 注音符号音素序列
            
        Returns:
            IPA格式音素序列
        """
        # 使用legacy_call进行转换
        logger.info(f"转换注音为IPA: {phonemes[:30]}...")
        result = self.legacy_call(phonemes)
        logger.info(f"IPA结果: {result[:30]}...")
        return result

    
    def text_to_phonemes(self, text: str) -> str:
        """将文本转换为音素序列，直接使用ZHFrontend
        
        Args:
            text: 输入文本
            
        Returns:
            音素序列字符串
        """
        if not text.strip():
            return ""
        
        # 数字转中文
        text = cn2an.transform(text, 'an2cn')
        
        # 处理标点符号
        text = self.map_punctuation(text)
        
        # 使用frontend处理
        result, _ = self.frontend(text)
        
        return (result)
    
    def __call__(self, text, en_callable=None) -> Tuple[str, None]:
        """实现与Misaki的ZHG2P相同的调用接口
        
        Args:
            text: 输入文本
            en_callable: 英文处理函数（可选）
            
        Returns:
            音素序列和tokens
        """
        if not text.strip():
            return '', None
        
        # 预处理
        text = cn2an.transform(text, 'an2cn')
        text = self.map_punctuation(text)
        
        # 使用回调处理混合文本
        en_callable = self.en_callable if en_callable is None else en_callable
        segments = []
        
        for en, zh in re.findall(r'([A-Za-z \'-]*[A-Za-z][A-Za-z \'-]*)|([^A-Za-z]+)', text):
            en, zh = en.strip(), zh.strip()
            if zh:
                # 使用frontend处理中文
                result, _ = self.frontend(zh)
                segments.append(result)
            elif en_callable is None:
                segments.append(self.unk)
            else:
                segments.append(en_callable(en))
            
        return (' '.join(segments), None)

    def convert_to_IAP(self, phonemes: str) -> str:
        try:
            logger.debug(f"原始输入: {phonemes[:50]}...")
            
            # 1. 预处理：先将多音节分开处理
            segments = []
            for segment in phonemes.split('/'):
                # 对每个音节单独处理
                processed = self._process_segment(segment)
                segments.append(processed)
            
            # 2. 合并并规范化
            result = ' '.join(segments)
            result = re.sub(r'\s+', ' ', result).strip()
            
            logger.debug(f"转换结果: {result[:50]}...")
            return result
        except Exception as e:
            logger.error(f"转换失败: {e}")
            return phonemes
        
    def _process_segment(self, segment: str) -> str:
        """处理单个音节片段"""
        # 移除可能的前导/后缀空格
        segment = segment.strip()
        
        # 处理带数字的情况 "我3" -> "wo↓"
        match = re.match(r'^([\u4e00-\u9fff])([1-5])$', segment)
        if match:
            hanzi, tone = match.groups()
            # 使用word2ipa确保正确的IPA生成
            ipa = self.word2ipa(hanzi)
            # 替换可能不正确的声调
            tone_map = {'1': '→', '2': '↗', '3': '↓', '4': '↘', '5': ''}
            # 确保声调正确应用
            return re.sub(r'[→↗↓↘]', tone_map[tone], ipa)
        
        # 处理纯注音符号情况 "ㄉㄜ5" -> "tɤ"
        if re.match(r'^[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩ]+[1-5]?$', segment):
            # 分离注音和声调
            zhuyin_part = re.sub(r'[1-5]', '', segment)
            tone_match = re.search(r'[1-5]', segment)
            tone = tone_match.group(0) if tone_match else '5'  # 默认为轻声
            
            # 转换注音到拼音
            pinyin = self._zhuyin_to_pinyin(zhuyin_part)
            # 添加音调并转为IPA
            with_tone = pinyin + tone
            return self.py2ipa(with_tone)
        
        # 处理混合情况 - 逐字符处理保证完整性
        # ...更多逻辑
        
        # 对于无法识别的情况，保持原样
        return segment

    def _zhuyin_to_pinyin(self, zhuyin: str) -> str:
        """注音符号转拼音（简化示例）"""
        # 实际实现需要完整的注音-拼音映射表
        # 这里只是一个简单示例
        ZHUYIN_TO_PINYIN = {
            "ㄅ": "b", "ㄆ": "p", # ...更多映射
        }
        # 转换逻辑
        # ...

    def get_phoneme_set(self) -> Set[str]:
        """获取当前G2P使用的所有音素集合
        
        Returns:
            音素集合
        """
        # 直接使用ZHFrontend中定义的音素符号
        from .zh_frontend import ZH_MAP
        return set(ZH_MAP.values())
    
    def get_language(self) -> str:
        """获取当前G2P支持的语言
        
        Returns:
            语言代码
        """
        return "zh" 