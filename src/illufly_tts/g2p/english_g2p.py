#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文G2P (Grapheme-to-Phoneme) 模块
将英文文本转换为音素序列，融合Misaki的en.py实现
"""

import re
import os
import logging
import json
import importlib.resources
import unicodedata
from dataclasses import dataclass, replace
from typing import Dict, Any, List, Optional, Set, Tuple, Union

from .base_g2p import BaseG2P
from .token import MToken  # 确保MToken类已复制或导入

try:
    import spacy
    import numpy as np
    from num2words import num2words
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ====== 常量和辅助函数 ======
# 从en.py复制的常量和辅助函数
DIPHTHONGS = frozenset('AIOQWYʤʧ')
STRESSES = 'ˌˈ'
PRIMARY_STRESS = STRESSES[1]
SECONDARY_STRESS = STRESSES[0]
VOWELS = frozenset('AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ')
CONSONANTS = frozenset('bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ')
US_TAUS = frozenset('AIOWYiuæɑəɛɪɹʊʌ')
US_VOCAB = frozenset('AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔ')
GB_VOCAB = frozenset('AIQWYabdfhijklmnpstuvwzðŋɑɒɔəɛɜɡɪɹʃʊʌʒʤʧˈˌːθᵊ')

PUNCTS = frozenset(';:,.!?—…"""')
NON_QUOTE_PUNCTS = frozenset(p for p in PUNCTS if p not in '"""')
PUNCT_TAGS = frozenset([".",",","-LRB-","-RRB-","``",'""',"''",":","$","#",'NFP'])
PUNCT_TAG_PHONEMES = {'-LRB-':'(', '-RRB-':')', '``':chr(8220), '""':chr(8221), "''":chr(8221)}
LINK_REGEX = re.compile(r'\[([^\]]+)\]\(([^\)]*)\)')
CURRENCIES = {'$': ('dollar', 'cent'), '£': ('pound', 'pence'), '€': ('euro', 'cent')}
SYMBOLS = {'%':'percent', '&':'and', '+':'plus', '@':'at'}
ADD_SYMBOLS = {'.':'dot', '/':'slash'}
ORDINALS = frozenset(['st', 'nd', 'rd', 'th'])
LEXICON_ORDS = [39, 45, *range(65, 91), *range(97, 123)]
SUBTOKEN_JUNKS = frozenset("',-._''/")

@dataclass
class TokenContext:
    """令牌上下文，用于上下文相关处理"""
    future_vowel: Optional[bool] = None
    future_to: bool = False

def apply_stress(ps, stress):
    """应用重音标记到音素序列"""
    def restress(ps):
        ips = list(enumerate(ps))
        stresses = {i: next(j for j, v in ips[i:] if v in VOWELS) for i, p in ips if p in STRESSES}
        for i, j in stresses.items():
            _, s = ips[i]
            ips[i] = (j - 0.5, s)
        ps = ''.join([p for _, p in sorted(ips)])
        return ps
        
    if stress is None:
        return ps
    elif stress < -1:
        return ps.replace(PRIMARY_STRESS, '').replace(SECONDARY_STRESS, '')
    elif stress == -1 or (stress in (0, -0.5) and PRIMARY_STRESS in ps):
        return ps.replace(SECONDARY_STRESS, '').replace(PRIMARY_STRESS, SECONDARY_STRESS)
    elif stress in (0, 0.5, 1) and all(s not in ps for s in STRESSES):
        if all(v not in ps for v in VOWELS):
            return ps
        return restress(SECONDARY_STRESS + ps)
    elif stress >= 1 and PRIMARY_STRESS not in ps and SECONDARY_STRESS in ps:
        return ps.replace(SECONDARY_STRESS, PRIMARY_STRESS)
    elif stress > 1 and all(s not in ps for s in STRESSES):
        if all(v not in ps for v in VOWELS):
            return ps
        return restress(PRIMARY_STRESS + ps)
    return ps

def is_digit(text):
    """检查文本是否全为数字"""
    return bool(re.match(r'^[0-9]+$', text))

def merge_tokens(tokens: List[MToken], unk: Optional[str] = None) -> MToken:
    """合并多个令牌"""
    stress = {tk._.stress for tk in tokens if tk._.stress is not None}
    currency = {tk._.currency for tk in tokens if tk._.currency is not None}
    rating = {tk._.rating for tk in tokens}
    if unk is None:
        phonemes = None
    else:
        phonemes = ''
        for tk in tokens:
            if tk._.prespace and phonemes and not phonemes[-1].isspace() and tk.phonemes:
                phonemes += ' '
            phonemes += unk if tk.phonemes is None else tk.phonemes
    return MToken(
        text=''.join(tk.text + tk.whitespace for tk in tokens[:-1]) + tokens[-1].text,
        tag=max(tokens, key=lambda tk: sum(1 if c == c.lower() else 2 for c in tk.text)).tag,
        whitespace=tokens[-1].whitespace,
        phonemes=phonemes,
        start_ts=tokens[0].start_ts,
        end_ts=tokens[-1].end_ts,
        _=MToken.Underscore(
            is_head=tokens[0]._.is_head,
            alias=None,
            stress=list(stress)[0] if len(stress) == 1 else None,
            currency=max(currency) if currency else None,
            num_flags=''.join(sorted({c for tk in tokens for c in tk._.num_flags})),
            prespace=tokens[0]._.prespace,
            rating=None if None in rating else min(rating),
        )
    )

def stress_weight(ps):
    """计算音素序列的重音权重"""
    return sum(2 if c in DIPHTHONGS else 1 for c in ps) if ps else 0

class Lexicon:
    """单词到音素的映射词典"""
    
    @staticmethod
    def grow_dictionary(d):
        """扩展词典，添加大小写变体"""
        e = {}
        for k, v in d.items():
            if len(k) < 2:
                continue
            if k == k.lower():
                if k != k.capitalize():
                    e[k.capitalize()] = v
            elif k == k.lower().capitalize():
                e[k.lower()] = v
        return {**e, **d}

    def __init__(self, british=False, dict_dir=None):
        """初始化词典
        
        Args:
            british: 是否使用英式发音
            dict_dir: 词典目录路径
        """
        self.british = british
        self.cap_stresses = (0.5, 2)
        self.golds = {}
        self.silvers = {}
        
        from . import data
        
        try:
            # 优先使用importlib.resources从data目录加载
            gold_file = "gb_gold.json" if british else "us_gold.json"
            silver_file = "gb_silver.json" if british else "us_silver.json"
            
            # 与en.py相同的加载方法
            with importlib.resources.open_text(data, gold_file) as r:
                self.golds = Lexicon.grow_dictionary(json.load(r))
            
            with importlib.resources.open_text(data, silver_file) as r:
                self.silvers = Lexicon.grow_dictionary(json.load(r))
                
            logger.info(f"成功从内置data目录加载词典: {gold_file} 和 {silver_file}")
        except Exception as e:
            # 如果找不到内置资源，再尝试从外部路径加载
            logger.error(f"从内置data目录加载词典失败: {e}")
            
            if dict_dir:
                try:
                    with open(os.path.join(dict_dir, gold_file), 'r', encoding='utf-8') as r:
                        self.golds = Lexicon.grow_dictionary(json.load(r))
                    
                    if os.path.exists(os.path.join(dict_dir, silver_file)):
                        with open(os.path.join(dict_dir, silver_file), 'r', encoding='utf-8') as r:
                            self.silvers = Lexicon.grow_dictionary(json.load(r))
                    
                    logger.info(f"从外部路径加载词典成功: {dict_dir}")
                except Exception as e2:
                    logger.error(f"从外部路径加载词典失败: {e2}")
                    raise ValueError(f"无法加载词典，内置和外部路径均失败")
            else:
                raise ValueError(f"无法加载词典: {e}")
        
        assert all(isinstance(v, str) or isinstance(v, dict) for v in self.golds.values())
        vocab = GB_VOCAB if british else US_VOCAB
        
        # 验证音素是否在词汇表中
        for vs in self.golds.values():
            if isinstance(vs, str):
                assert all(c in vocab for c in vs), f"无效音素：{vs}"
            else:
                assert 'DEFAULT' in vs, f"缺少DEFAULT键：{vs}"
                for v in vs.values():
                    assert v is None or all(c in vocab for c in v), f"无效音素：{v}"
    
    def get_NNP(self, word):
        """处理专有名词"""
        ps = [self.golds.get(c.upper()) for c in word if c.isalpha()]
        if None in ps:
            return None, None
        ps = apply_stress(''.join(ps), 0)
        ps = ps.rsplit(SECONDARY_STRESS, 1)
        return PRIMARY_STRESS.join(ps), 3

    def get_special_case(self, word, tag, stress, ctx):
        """处理特殊情况词汇"""
        if tag == 'ADD' and word in ADD_SYMBOLS:
            return self.lookup(ADD_SYMBOLS[word], None, -0.5, ctx)
        elif word in SYMBOLS:
            return self.lookup(SYMBOLS[word], None, None, ctx)
        elif '.' in word.strip('.') and word.replace('.', '').isalpha() and len(max(word.split('.'), key=len)) < 3:
            return self.get_NNP(word)
        elif word in ('a', 'A'):
            return 'ɐ' if tag == 'DT' else 'ˈA', 4
        elif word in ('am', 'Am', 'AM'):
            if tag.startswith('NN'):
                return self.get_NNP(word)
            elif ctx.future_vowel is None or word != 'am' or stress and stress > 0:
                return self.golds.get('am', 'æm'), 4
            return 'ɐm', 4
        elif word in ('an', 'An', 'AN'):
            if word == 'AN' and tag.startswith('NN'):
                return self.get_NNP(word)
            return 'ɐn', 4
        elif word == 'I' and tag == 'PRP':
            return f'{SECONDARY_STRESS}aɪ', 4
        elif word in ('by', 'By', 'BY') and Lexicon.get_parent_tag(tag) == 'ADV':
            return 'bˈaɪ', 4
        elif word in ('to', 'To') or (word == 'TO' and tag in ('TO', 'IN')):
            return {None: self.golds.get('to', 'tu'), False: 'tə', True: 'tʊ'}[ctx.future_vowel], 4
        elif word in ('in', 'In') or (word == 'IN' and tag != 'NNP'):
            stress = PRIMARY_STRESS if ctx.future_vowel is None or tag != 'IN' else ''
            return stress + 'ɪn', 4
        elif word in ('the', 'The') or (word == 'THE' and tag == 'DT'):
            return 'ði' if ctx.future_vowel == True else 'ðə', 4
        elif tag == 'IN' and re.match(r'(?i)vs\.?$', word):
            return self.lookup('versus', None, None, ctx)
        elif word in ('used', 'Used', 'USED'):
            if tag in ('VBD', 'JJ') and ctx.future_to:
                return self.golds.get('used', {}).get('VBD', 'juzd'), 4
            return self.golds.get('used', {}).get('DEFAULT', 'just'), 4
        return None, None

    @staticmethod
    def get_parent_tag(tag):
        """获取父标签"""
        if tag is None:
            return tag
        elif tag.startswith('VB'):
            return 'VERB'
        elif tag.startswith('NN'):
            return 'NOUN'
        elif tag.startswith('ADV') or tag.startswith('RB'):
            return 'ADV'
        elif tag.startswith('ADJ') or tag.startswith('JJ'):
            return 'ADJ'
        return tag

    def is_known(self, word, tag):
        """检查单词是否已知"""
        if word in self.golds or word in SYMBOLS or word in self.silvers:
            return True
        elif not word.isalpha() or not all(ord(c) in LEXICON_ORDS for c in word):
            return False
        elif len(word) == 1:
            return True
        elif word == word.upper() and word.lower() in self.golds:
            return True
        return word[1:] == word[1:].upper()

    def lookup(self, word, tag, stress, ctx):
        """查找单词的音素"""
        is_NNP = None
        if word == word.upper() and word not in self.golds:
            word = word.lower()
            is_NNP = tag == 'NNP'
        ps, rating = self.golds.get(word), 4
        if ps is None and not is_NNP:
            ps, rating = self.silvers.get(word), 3
        if isinstance(ps, dict):
            if ctx and ctx.future_vowel is None and 'None' in ps:
                tag = 'None'
            elif tag not in ps:
                tag = Lexicon.get_parent_tag(tag)
            ps = ps.get(tag, ps.get('DEFAULT'))
        if ps is None or (is_NNP and PRIMARY_STRESS not in ps):
            ps, rating = self.get_NNP(word)
            if ps is not None:
                return ps, rating
        return apply_stress(ps, stress), rating

    def _s(self, stem):
        """处理词尾s"""
        if not stem:
            return None
        elif stem[-1] in 'ptkfθ':
            return stem + 's'
        elif stem[-1] in 'szʃʒʧʤ':
            return stem + ('ɪ' if self.british else 'ᵻ') + 'z'
        return stem + 'z'

    def stem_s(self, word, tag, stress, ctx):
        """处理词尾s的单词"""
        if len(word) < 3 or not word.endswith('s'):
            return None, None
        if not word.endswith('ss') and self.is_known(word[:-1], tag):
            stem = word[:-1]
        elif (word.endswith("'s") or (len(word) > 4 and word.endswith('es') and not word.endswith('ies'))) and self.is_known(word[:-2], tag):
            stem = word[:-2]
        elif len(word) > 4 and word.endswith('ies') and self.is_known(word[:-3]+'y', tag):
            stem = word[:-3] + 'y'
        else:
            return None, None
        stem, rating = self.lookup(stem, tag, stress, ctx)
        return self._s(stem), rating

    def _ed(self, stem):
        """处理词尾ed"""
        if not stem:
            return None
        elif stem[-1] in 'pkfθʃsʧ':
            return stem + 't'
        elif stem[-1] == 'd':
            return stem + ('ɪ' if self.british else 'ᵻ') + 'd'
        elif stem[-1] != 't':
            return stem + 'd'
        elif self.british or len(stem) < 2:
            return stem + 'ɪd'
        elif stem[-2] in US_TAUS:
            return stem[:-1] + 'ɾᵻd'
        return stem + 'ᵻd'

    def stem_ed(self, word, tag, stress, ctx):
        """处理词尾ed的单词"""
        if len(word) < 4 or not word.endswith('d'):
            return None, None
        if not word.endswith('dd') and self.is_known(word[:-1], tag):
            stem = word[:-1]
        elif len(word) > 4 and word.endswith('ed') and not word.endswith('eed') and self.is_known(word[:-2], tag):
            stem = word[:-2]
        else:
            return None, None
        stem, rating = self.lookup(stem, tag, stress, ctx)
        return self._ed(stem), rating

    def _ing(self, stem):
        """处理词尾ing"""
        if not stem:
            return None
        elif self.british:
            if stem[-1] in 'əː':
                return None
        elif len(stem) > 1 and stem[-1] == 't' and stem[-2] in US_TAUS:
            return stem[:-1] + 'ɾɪŋ'
        return stem + 'ɪŋ'

    def stem_ing(self, word, tag, stress, ctx):
        """处理词尾ing的单词"""
        if len(word) < 5 or not word.endswith('ing'):
            return None, None
        if len(word) > 5 and self.is_known(word[:-3], tag):
            stem = word[:-3]
        elif self.is_known(word[:-3]+'e', tag):
            stem = word[:-3] + 'e'
        elif len(word) > 5 and re.search(r'([bcdgklmnprstvxz])\1ing$|cking$', word) and self.is_known(word[:-4], tag):
            stem = word[:-4]
        else:
            return None, None
        stem, rating = self.lookup(stem, tag, stress, ctx)
        return self._ing(stem), rating

    def get_word(self, word, tag, stress, ctx):
        """获取单词的音素"""
        ps, rating = self.get_special_case(word, tag, stress, ctx)
        if ps is not None:
            return ps, rating
        wl = word.lower()
        if len(word) > 1 and word.replace("'", '').isalpha() and word != word.lower() and (
            tag != 'NNP' or len(word) > 7
        ) and word not in self.golds and word not in self.silvers and (
            word == word.upper() or word[1:] == word[1:].lower()
        ) and (
            wl in self.golds or wl in self.silvers or any(
                fn(wl, tag, stress, ctx)[0] for fn in (self.stem_s, self.stem_ed, self.stem_ing)
            )
        ):
            word = wl
        if self.is_known(word, tag):
            return self.lookup(word, tag, stress, ctx)
        elif word.endswith("s'") and self.is_known(word[:-2] + "'s", tag):
            return self.lookup(word[:-2] + "'s", tag, stress, ctx)
        elif word.endswith("'") and self.is_known(word[:-1], tag):
            return self.lookup(word[:-1], tag, stress, ctx)
        _s, rating = self.stem_s(word, tag, stress, ctx)
        if _s is not None:
            return _s, rating
        _ed, rating = self.stem_ed(word, tag, stress, ctx)
        if _ed is not None:
            return _ed, rating
        _ing, rating = self.stem_ing(word, tag, 0.5 if stress is None else stress, ctx)
        if _ing is not None:
            return _ing, rating
        return None, None

    @staticmethod
    def is_currency(word):
        """检查单词是否是货币表示"""
        if '.' not in word:
            return True
        elif word.count('.') > 1:
            return False
        cents = word.split('.')[1]
        return len(cents) < 3 or set(cents) == {0}

    def get_number(self, word, currency, is_head, num_flags):
        """处理数字"""
        suffix = re.search(r"[a-z']+$", word)
        suffix = suffix.group() if suffix else None
        word = word[:-len(suffix)] if suffix else word
        result = []
        if word.startswith('-'):
            result.append(self.lookup('minus', None, None, None))
            word = word[1:]
            
        def extend_num(num, first=True, escape=False):
            try:
                splits = re.split(r'[^a-z]+', num if escape else num2words(int(num)))
                for i, w in enumerate(splits):
                    if w != 'and' or '&' in num_flags:
                        if first and i == 0 and len(splits) > 1 and w == 'one' and 'a' in num_flags:
                            result.append(('ə', 4))
                        else:
                            result.append(self.lookup(w, None, -2 if w == 'point' else None, None))
                    elif w == 'and' and 'n' in num_flags and result:
                        result[-1] = (result[-1][0] + 'ən', result[-1][1])
            except:
                # 如果num2words出错，回退到简单处理
                for digit in num:
                    if digit.isdigit():
                        result.append(self.lookup(['zero', 'one', 'two', 'three', 'four', 
                                                'five', 'six', 'seven', 'eight', 'nine'][int(digit)], 
                                            None, None, None))
                    else:
                        result.append((digit, 4))
        
        try:
            if is_digit(word) and suffix in ORDINALS:
                extend_num(num2words(int(word), to='ordinal'), escape=True)
            elif not result and len(word) == 4 and currency not in CURRENCIES and is_digit(word):
                extend_num(num2words(int(word), to='year'), escape=True)
            elif not is_head and '.' not in word:
                num = word.replace(',', '')
                if num[0] == '0' or len(num) > 3:
                    [extend_num(n, first=False) for n in num]
                elif len(num) == 3 and not num.endswith('00'):
                    extend_num(num[0])
                    if num[1] == '0':
                        result.append(self.lookup('O', None, -2, None))
                        extend_num(num[2], first=False)
                    else:
                        extend_num(num[1:], first=False)
                else:
                    extend_num(num)
            elif word.count('.') > 1 or not is_head:
                first = True
                for num in word.replace(',', '').split('.'):
                    if not num:
                        pass
                    elif num[0] == '0' or (len(num) != 2 and any(n != '0' for n in num[1:])):
                        [extend_num(n, first=False) for n in num]
                    else:
                        extend_num(num, first=first)
                    first = False
            elif currency in CURRENCIES and Lexicon.is_currency(word):
                pairs = [(int(num) if num else 0, unit) for num, unit in zip(word.replace(',', '').split('.'), CURRENCIES[currency])]
                if len(pairs) > 1:
                    if pairs[1][0] == 0:
                        pairs = pairs[:1]
                    elif pairs[0][0] == 0:
                        pairs = pairs[1:]
                for i, (num, unit) in enumerate(pairs):
                    if i > 0:
                        result.append(self.lookup('and', None, None, None))
                    extend_num(num, first=i==0)
                    result.append(self.stem_s(unit+'s', None, None, None) if abs(num) != 1 and unit != 'pence' else self.lookup(unit, None, None, None))
            else:
                if is_digit(word):
                    word = num2words(int(word), to='cardinal')
                elif '.' not in word:
                    word = num2words(int(word.replace(',', '')), to='ordinal' if suffix in ORDINALS else 'cardinal')
                else:
                    word = word.replace(',', '')
                    if word[0] == '.':
                        word = 'point ' + ' '.join(num2words(int(n)) for n in word[1:])
                    else:
                        word = num2words(float(word))
                extend_num(word, escape=True)
        except Exception as e:
            logger.error(f"数字处理失败: {word}, {e}")
            # 出错时的应急处理 - 逐位处理数字
            for c in word:
                if c.isdigit():
                    result.append(self.lookup(['zero', 'one', 'two', 'three', 'four', 
                                            'five', 'six', 'seven', 'eight', 'nine'][int(c)], 
                                        None, None, None))
                else:
                    result.append((c, 4))
                
        if not result:
            logger.warning(f'处理数字失败: {word}')
            return None, None
            
        result, rating = ' '.join(p for p, _ in result), min(r for _, r in result)
        if suffix in ('s', "'s"):
            return self._s(result), rating
        elif suffix in ('ed', "'d"):
            return self._ed(result), rating
        elif suffix == 'ing':
            return self._ing(result), rating
        return result, rating

    def append_currency(self, ps, currency):
        """添加货币符号"""
        if not currency:
            return ps
        currency = CURRENCIES.get(currency)
        currency = self.stem_s(currency[0]+'s', None, None, None)[0] if currency else None
        return f'{ps} {currency}' if currency else ps

    @staticmethod
    def numeric_if_needed(c):
        """转换Unicode数字为ASCII数字"""
        if not c.isdigit():
            return c
        n = unicodedata.numeric(c)
        return str(int(n)) if n == int(n) else c

    @staticmethod
    def is_number(word, is_head):
        """检查是否是数字"""
        if all(not is_digit(c) for c in word):
            return False
        suffixes = ('ing', "'d", 'ed', "'s", *ORDINALS, 's')
        for s in suffixes:
            if word.endswith(s):
                word = word[:-len(s)]
                break
        return all(is_digit(c) or c in ',.' or (is_head and i == 0 and c == '-') for i, c in enumerate(word))

    def __call__(self, tk, ctx):
        """主处理函数"""
        word = (tk.text if tk._.alias is None else tk._.alias).replace(chr(8216), "'").replace(chr(8217), "'")
        word = unicodedata.normalize('NFKC', word)
        word = ''.join(Lexicon.numeric_if_needed(c) for c in word)
        stress = None if word == word.lower() else self.cap_stresses[int(word == word.upper())]
        ps, rating = self.get_word(word, tk.tag, stress, ctx)
        if ps is not None:
            return apply_stress(self.append_currency(ps, tk._.currency), tk._.stress), rating
        elif Lexicon.is_number(word, tk._.is_head):
            ps, rating = self.get_number(word, tk._.currency, tk._.is_head, tk._.num_flags)
            return apply_stress(ps, tk._.stress), rating
        elif not all(ord(c) in LEXICON_ORDS for c in word):
            return None, None
        return None, None

class EnglishG2P(BaseG2P):
    """英文G2P转换器，直接使用en.py的实现"""
    
    def __init__(
        self, 
        british: bool = False,
        unk: str = '❓'
    ):
        """初始化英文G2P转换器"""
        super().__init__()
        self.british = british
        self.unk = unk
        
        # 加载spaCy模型
        name = "en_core_web_sm"
        if not spacy.util.is_package(name):
            spacy.cli.download(name)
        components = ['tok2vec', 'tagger']
        self.nlp = spacy.load(name, enable=components)
        logger.info(f"已加载spaCy模型: {name}")
        
        # 【关键修改】直接使用Lexicon类，它会自动从data目录加载词典
        # 完全复用en.py中的词典加载逻辑，不再自己实现
        self.lexicon = Lexicon(british=self.british)
        
        # 其他属性保持不变
        self.arpa_to_ipa_map = {
            'AA': 'ɑ', 'AA0': 'ɑ', 'AA1': 'ˈɑ', 'AA2': 'ˌɑ',
            'AE': 'æ', 'AE0': 'æ', 'AE1': 'ˈæ', 'AE2': 'ˌæ',
            'AH': 'ʌ', 'AH0': 'ə', 'AH1': 'ˈʌ', 'AH2': 'ˌʌ',
            'AO': 'ɔ', 'AO0': 'ɔ', 'AO1': 'ˈɔ', 'AO2': 'ˌɔ',
            'AW': 'aʊ', 'AW0': 'aʊ', 'AW1': 'ˈaʊ', 'AW2': 'ˌaʊ',
            'AY': 'aɪ', 'AY0': 'aɪ', 'AY1': 'ˈaɪ', 'AY2': 'ˌaɪ',
            'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
            'EH': 'ɛ', 'EH0': 'ɛ', 'EH1': 'ˈɛ', 'EH2': 'ˌɛ',
            'ER': 'ɝ', 'ER0': 'ɚ', 'ER1': 'ˈɝ', 'ER2': 'ˌɝ',
            'EY': 'eɪ', 'EY0': 'eɪ', 'EY1': 'ˈeɪ', 'EY2': 'ˌeɪ',
            'F': 'f', 'G': 'ɡ', 'HH': 'h', 
            'IH': 'ɪ', 'IH0': 'ɪ', 'IH1': 'ˈɪ', 'IH2': 'ˌɪ',
            'IY': 'i', 'IY0': 'i', 'IY1': 'ˈi', 'IY2': 'ˌi',
            'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
            'OW': 'oʊ', 'OW0': 'oʊ', 'OW1': 'ˈoʊ', 'OW2': 'ˌoʊ',
            'OY': 'ɔɪ', 'OY0': 'ɔɪ', 'OY1': 'ˈɔɪ', 'OY2': 'ˌɔɪ',
            'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
            'T': 't', 'TH': 'θ', 
            'UH': 'ʊ', 'UH0': 'ʊ', 'UH1': 'ˈʊ', 'UH2': 'ˌʊ',
            'UW': 'u', 'UW0': 'u', 'UW1': 'ˈu', 'UW2': 'ˌu',
            'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
            # 小写版本也包含
            'aa': 'ɑ', 'ae': 'æ', 'ah': 'ʌ', 'ao': 'ɔ', 'aw': 'aʊ',
            'ay': 'aɪ', 'b': 'b', 'ch': 'tʃ', 'd': 'd', 'dh': 'ð',
            'eh': 'ɛ', 'er': 'ɝ', 'ey': 'eɪ', 'f': 'f', 'g': 'ɡ',
            'hh': 'h', 'ih': 'ɪ', 'iy': 'i', 'jh': 'dʒ', 'k': 'k',
            'l': 'l', 'm': 'm', 'n': 'n', 'ng': 'ŋ', 'ow': 'oʊ',
            'oy': 'ɔɪ', 'p': 'p', 'r': 'ɹ', 's': 's', 'sh': 'ʃ',
            't': 't', 'th': 'θ', 'uh': 'ʊ', 'uw': 'u', 'v': 'v',
            'w': 'w', 'y': 'j', 'z': 'z', 'zh': 'ʒ',
        }
    
    def _create_token_from_text(self, text: str, tag: str = 'NN') -> MToken:
        """从文本创建MToken对象"""
        return MToken(
            text=text,
            tag=tag,
            whitespace=' ',
            phonemes=None,
            start_ts=None,
            end_ts=None,
            _=MToken.Underscore(
                is_head=True,
                alias=None,
                stress=None,
                currency=None,
                num_flags='',
                prespace=False,
                rating=None,
            )
        )
    
    def preprocess(self, text: str) -> Tuple[str, List[str], Dict]:
        """预处理文本，返回处理后的文本、标记和特征"""
        result = ''
        tokens = []
        features = {}
        last_end = 0
        text = text.lstrip()
        
        # 处理特殊标记（如果有）
        for m in LINK_REGEX.finditer(text):
            result += text[last_end:m.start()]
            tokens.extend(text[last_end:m.start()].split())
            f = m.group(2)
            if is_digit(f[1 if f[:1] in ('-', '+') else 0:]):
                f = int(f)
            elif f in ('0.5', '+0.5'):
                f = 0.5
            elif f == '-0.5':
                f = -0.5
            elif len(f) > 1 and f[0] == '/' and f[-1] == '/':
                f = f[0] + f[1:].rstrip('/')
            elif len(f) > 1 and f[0] == '#' and f[-1] == '#':
                f = f[0] + f[1:].rstrip('#')
            else:
                f = None
            if f is not None:
                features[len(tokens)] = f
            result += m.group(1)
            tokens.append(m.group(1))
            last_end = m.end()
            
        if last_end < len(text):
            result += text[last_end:]
            tokens.extend(text[last_end:].split())
            
        return result, tokens, features
    
    def tokenize(self, text: str) -> List[MToken]:
        """将文本标记化为MToken对象列表"""
        processed_text, tokens, features = self.preprocess(text)
        
        # 使用spaCy进行标记化和词性标注
        doc = self.nlp(processed_text)
        mutable_tokens = [MToken(
            text=tk.text, tag=tk.tag_, whitespace=tk.whitespace_,
            phonemes=None, start_ts=None, end_ts=None,
            _=MToken.Underscore(is_head=True, alias=None, stress=None, 
                                currency=None, num_flags='', prespace=False,
                                rating=None)
        ) for tk in doc]
        
        # 应用特征（如果有）
        if features:
            align = spacy.training.Alignment.from_strings(tokens, [tk.text for tk in mutable_tokens])
            for k, v in features.items():
                for i, j in enumerate(np.where(align.y2x.data == k)[0]):
                    if j >= len(mutable_tokens):
                        continue
                    if not isinstance(v, str):
                        mutable_tokens[j]._.stress = v
        
        return mutable_tokens
    
    def text_to_phonemes(self, text: str) -> str:
        """将文本转换为音素序列"""
        # 清理文本
        text = self.sanitize_text(text)
        
        # 标记化
        tokens = self.tokenize(text)
        
        # 初始化上下文
        ctx = TokenContext()
        phonemes = []
        
        # 逆序处理标记，以便更新上下文
        for i, token in reversed(list(enumerate(tokens))):
            # 使用Lexicon处理
            ps, rating = self.lexicon(token, ctx)
            if ps is not None:
                token.phonemes = ps
                token._.rating = rating
            else:
                # 如果处理失败，使用未知符号
                token.phonemes = self.unk
                token._.rating = 1
            
            # 更新上下文
            vowel = ctx.future_vowel
            for c in token.phonemes or '':
                if c in VOWELS:
                    vowel = True
                    break
                if c in CONSONANTS:
                    vowel = False
                    break
            ctx.future_vowel = vowel
            ctx.future_to = token.text.lower() == 'to'
        
        # 正序收集音素
        for token in tokens:
            if token.phonemes:
                phonemes.append(token.phonemes)
            else:
                phonemes.append(self.unk)
        
        return ' '.join(phonemes)
    
    def arpa_to_ipa_string(self, arpa_phonemes: str) -> str:
        """将ARPAbet音素字符串转换为IPA格式"""
        # 处理ARPAbet音素
        ipa_result = []
        for phoneme in arpa_phonemes.split():
            if phoneme in self.arpa_to_ipa_map:
                ipa_result.append(self.arpa_to_ipa_map[phoneme])
            else:
                # 如果无法映射，保留原始音素
                ipa_result.append(phoneme)
        
        return ''.join(ipa_result)

    def text_to_ipa(self, text: str) -> str:
        """将文本直接转换为IPA音素"""
        # 获取音素
        phonemes = self.text_to_phonemes(text)
        
        # 检查是否已经是IPA格式
        is_already_ipa = any(char in 'æɑəɛɪɔʊʌɝɚθðʃʒŋɹˌˈ' for char in phonemes)
        if is_already_ipa:
            return phonemes
        
        # 转换为IPA
        return self.arpa_to_ipa_string(phonemes)
    
    def get_phoneme_set(self) -> Set[str]:
        """获取所有支持的音素集合"""
        # 基础IPA音素集
        phoneme_set = set()
        
        # 添加所有ARPAbet到IPA映射中的音素
        for ipa in self.arpa_to_ipa_map.values():
            for char in ipa:
                phoneme_set.add(char)
        
        # 添加常见元音和辅音音素
        phoneme_set.update('aeiouæɑɒɔəɛɜɪʊʌᵻ')  # 元音
        phoneme_set.update('bdfghjklmnpqrstvwxyzðŋɡɹɾʃʒʤʧθ')  # 辅音
        
        # 添加重音符号
        phoneme_set.update('ˌˈ')
        
        # 添加标点符号
        phoneme_set.update(',.!?;:')
        
        return phoneme_set

    def __call__(self, text: str) -> str:
        """实现回调接口，供ChineseG2P调用"""
        logger.info(f"处理英文文本: {text[:20]}...")
        result = self.text_to_ipa(text)
        logger.info(f"英文处理结果: {result[:20]}...")
        return result 