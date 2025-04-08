#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立的中文前端处理模块
ADAPTED from https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/frontend/zh_frontend.py
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any

import jieba.posseg as psg
from pypinyin import lazy_pinyin
from pypinyin import load_phrases_dict
from pypinyin import load_single_dict
from pypinyin import Style
try:
    # 优先使用pypinyin_dict提供的更全面的拼音数据
    from pypinyin_dict.phrase_pinyin_data import large_pinyin
    HAS_LARGE_PINYIN = True
except ImportError:
    HAS_LARGE_PINYIN = False
    logging.warning("未安装pypinyin_dict，中文拼音处理可能不完整")

from .token import MToken
from .tone_sandhi import ToneSandhi

logger = logging.getLogger(__name__)

# 中文声母列表
INITIALS = [
    'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'zh', 'ch', 'sh',
    'r', 'z', 'c', 's', 'j', 'q', 'x'
]
INITIALS += ['y', 'w', ' ']

# 声调: 0为无, 5为轻声
TONES = ["0", "1", "2", "3", "4", "5"]

# 拼音到注音符号映射
ZH_MAP = {"b":"ㄅ","p":"ㄆ","m":"ㄇ","f":"ㄈ","d":"ㄉ","t":"ㄊ","n":"ㄋ","l":"ㄌ","g":"ㄍ","k":"ㄎ","h":"ㄏ","j":"ㄐ","q":"ㄑ","x":"ㄒ","zh":"ㄓ","ch":"ㄔ","sh":"ㄕ","r":"ㄖ","z":"ㄗ","c":"ㄘ","s":"ㄙ","a":"ㄚ","o":"ㄛ","e":"ㄜ","ie":"ㄝ","ai":"ㄞ","ei":"ㄟ","ao":"ㄠ","ou":"ㄡ","an":"ㄢ","en":"ㄣ","ang":"ㄤ","eng":"ㄥ","er":"ㄦ","i":"ㄧ","u":"ㄨ","v":"ㄩ","ii":"ㄭ","iii":"十","ve":"月","ia":"压","ian":"言","iang":"阳","iao":"要","in":"阴","ing":"应","iong":"用","iou":"又","ong":"中","ua":"穵","uai":"外","uan":"万","uang":"王","uei":"为","uen":"文","ueng":"瓮","uo":"我","van":"元","vn":"云"}
# 添加标点符号映射
for p in ';:,.!?/—…"()"" 12345R':
    ZH_MAP[p] = p

class ZHFrontend:
    """中文前端处理器，负责中文文本到音素的转换"""
    
    def __init__(self, unk: str = '❓'):
        """初始化中文前端处理器
        
        Args:
            unk: 未知字符的替代符号
        """
        self.unk = unk
        self.punc = frozenset(';:,.!?—…"()""')
        
        # 多音字和固定发音词典
        self.phrases_dict = {
            '开户行': [['ka1i'], ['hu4'], ['hang2']],
            '发卡行': [['fa4'], ['ka3'], ['hang2']],
            '放款行': [['fa4ng'], ['kua3n'], ['hang2']],
            '茧行': [['jia3n'], ['hang2']],
            '行号': [['hang2'], ['ha4o']],
            '各地': [['ge4'], ['di4']],
            '借还款': [['jie4'], ['hua2n'], ['kua3n']],
            '时间为': [['shi2'], ['jia1n'], ['we2i']],
            '为准': [['we2i'], ['zhu3n']],
            '色差': [['se4'], ['cha1']],
            '嗲': [['dia3']],
            '呗': [['bei5']],
            '不': [['bu4']],
            '咗': [['zuo5']],
            '嘞': [['lei5']],
            '掺和': [['chan1'], ['huo5']]
        }
        
        # 儿化音词典
        self.must_erhua = {
            "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"
        }
        self.not_erhua = {
            "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
            "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
            "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
            "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
            "狗儿", "少儿"
        }
        
        # 声调变化处理器
        self.tone_modifier = ToneSandhi()
        
        # 初始化拼音处理
        self._init_pypinyin()
    
    def _init_pypinyin(self):
        """初始化拼音处理模块"""
        # 加载大型拼音词典（如果有）
        if HAS_LARGE_PINYIN:
            large_pinyin.load()
            
        # 加载自定义词典
        load_phrases_dict(self.phrases_dict)
        
        # 调整特定字的拼音
        load_single_dict({ord(u'地'): u'de,di4'})
    
    def _get_initials_finals(self, word: str) -> Tuple[List[str], List[str]]:
        """获取词的声母和韵母
        
        Args:
            word: 输入词
            
        Returns:
            (声母列表, 韵母列表)
        """
        initials = []
        finals = []
        orig_initials = lazy_pinyin(
            word, neutral_tone_with_five=True, style=Style.INITIALS)
        orig_finals = lazy_pinyin(
            word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            
        # 特殊处理"嗯"
        en_index = [index for index, c in enumerate(word) if c == "嗯"]
        for i in en_index:
            orig_finals[i] = "n2"
        
        # 处理声母和韵母
        for c, v in zip(orig_initials, orig_finals):
            if re.match(r'i\d', v):
                if c in ['z', 'c', 's']:
                    # zi, ci, si
                    v = re.sub('i', 'ii', v)
                elif c in ['zh', 'ch', 'sh', 'r']:
                    # zhi, chi, shi
                    v = re.sub('i', 'iii', v)
            initials.append(c)
            finals.append(v)
        
        return initials, finals
    
    def _merge_erhua(self, initials: List[str], finals: List[str], word: str, pos: str) -> Tuple[List[str], List[str]]:
        """处理儿化音
        
        Args:
            initials: 声母列表
            finals: 韵母列表
            word: 词
            pos: 词性
            
        Returns:
            (处理后的声母列表, 处理后的韵母列表)
        """
        # 修复er1
        for i, phn in enumerate(finals):
            if i == len(finals) - 1 and word[i] == "儿" and phn == 'er1':
                finals[i] = 'er2'
        
        # 判断是否需要儿化
        if word not in self.must_erhua and (word in self.not_erhua or pos in {"a", "j", "nr"}):
            return initials, finals
        
        # 特殊情况直接返回
        if len(finals) != len(word):
            return initials, finals
        
        # 处理不发音的儿化
        new_initials = []
        new_finals = []
        for i, phn in enumerate(finals):
            if i == len(finals) - 1 and word[i] == "儿" and phn in {"er2", "er5"} and word[-2:] not in self.not_erhua and new_finals:
                # 将儿化添加到前一个音节
                new_finals[-1] = new_finals[-1][:-1] + "R" + new_finals[-1][-1]
            else:
                new_initials.append(initials[i])
                new_finals.append(phn)
        
        return new_initials, new_finals
    
    def __call__(self, text: str, with_erhua: bool = True) -> Tuple[str, List[MToken]]:
        """将文本转换为音素序列
        
        Args:
            text: 输入文本
            with_erhua: 是否处理儿化音
            
        Returns:
            (音素序列, 标记列表)
        """
        tokens = []
        
        # 分词
        seg_cut = psg.lcut(text)
        
        # 声调变化预处理
        seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)
        
        # 处理每个词
        for word, pos in seg_cut:
            # 处理特殊词性
            if pos == 'x' and '\u4E00' <= min(word) and max(word) <= '\u9FFF':
                pos = 'X'
            elif pos != 'x' and word in self.punc:
                pos = 'x'
            
            # 创建标记
            tk = MToken(text=word, tag=pos, whitespace='')
            
            # 处理标点和英文
            if pos in ('x', 'eng'):
                if not word.isspace():
                    if pos == 'x' and word in self.punc:
                        tk.phonemes = word
                    tokens.append(tk)
                elif tokens:
                    tokens[-1].whitespace += word
                continue
            elif tokens and tokens[-1].tag not in ('x', 'eng') and not tokens[-1].whitespace:
                tokens[-1].whitespace = '/'
            
            # 获取声母和韵母
            sub_initials, sub_finals = self._get_initials_finals(word)
            
            # 应用声调变化规则
            sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)
            
            # 处理儿化音
            if with_erhua:
                sub_initials, sub_finals = self._merge_erhua(sub_initials, sub_finals, word, pos)
            
            # 转换为音素序列
            phones = []
            for c, v in zip(sub_initials, sub_finals):
                if c:
                    phones.append(c)
                if v and (v not in self.punc or v != c):
                    phones.append(v)
            
            # 格式化音素序列
            phones = '_'.join(phones).replace('_eR', '_er').replace('R', '_R')
            phones = re.sub(r'(?=\d)', '_', phones).split('_')
            
            # 将音素转换为注音符号
            tk.phonemes = ''.join(ZH_MAP.get(p, self.unk) for p in phones)
            tokens.append(tk)
        
        # 生成完整音素序列
        result = ''.join((self.unk if tk.phonemes is None else tk.phonemes) + tk.whitespace for tk in tokens)
        
        return result, tokens
    
    def text_to_phonemes(self, text: str) -> str:
        """将文本转换为音素序列 (对外接口)
        
        Args:
            text: 输入文本
            
        Returns:
            音素序列
        """
        phonemes, _ = self.__call__(text)
        return phonemes
