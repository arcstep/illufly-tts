#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
定制化中文G2P - 专为Kokoro模型设计的文本到音素转换
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple, Any

import jieba
import pypinyin
from pypinyin import lazy_pinyin, Style

try:
    from pypinyin_dict.phrase_pinyin_data import large_pinyin
    HAS_LARGE_PINYIN = True
except ImportError:
    HAS_LARGE_PINYIN = False
    logging.warning("未安装pypinyin_dict，中文拼音处理可能不完整")

from .base_g2p import BaseG2P

logger = logging.getLogger(__name__)

# 拼音到注音符号映射
PINYIN_TO_ZHUYIN = {
    "b":"ㄅ", "p":"ㄆ", "m":"ㄇ", "f":"ㄈ", "d":"ㄉ", "t":"ㄊ", "n":"ㄋ", "l":"ㄌ",
    "g":"ㄍ", "k":"ㄎ", "h":"ㄏ", "j":"ㄐ", "q":"ㄑ", "x":"ㄒ", "zh":"ㄓ", "ch":"ㄔ",
    "sh":"ㄕ", "r":"ㄖ", "z":"ㄗ", "c":"ㄘ", "s":"ㄙ", "a":"ㄚ", "o":"ㄛ", "e":"ㄜ",
    "i":"ㄧ", "u":"ㄨ", "v":"ㄩ", "ai":"ㄞ", "ei":"ㄟ", "ao":"ㄠ", "ou":"ㄡ", "an":"ㄢ", 
    "en":"ㄣ", "ang":"ㄤ", "eng":"ㄥ", "er":"ㄦ", "ia":"ㄧㄚ", "ie":"ㄧㄝ", "iao":"ㄧㄠ", 
    "iu":"ㄧㄡ", "ian":"ㄧㄢ", "in":"ㄧㄣ", "iang":"ㄧㄤ", "ing":"ㄧㄥ", "iong":"ㄩㄥ", 
    "ua":"ㄨㄚ", "uo":"ㄨㄛ", "uai":"ㄨㄞ", "ui":"ㄨㄟ", "uan":"ㄨㄢ", "un":"ㄨㄣ",
    "uang":"ㄨㄤ", "ong":"ㄨㄥ", "ve":"ㄩㄝ", "vn":"ㄩㄣ", "van":"ㄩㄢ",
    # 特殊音素
    "ii":"ㄭ", "iii":"十", "iou":"ㄧㄡ", "uei":"ㄨㄟ", "uen":"ㄨㄣ", "ueng":"ㄨㄥ",
    # 零声母处理
    "y":"ㄧ", "w":"ㄨ", "yu":"ㄩ",
    # 特殊单韵母
    "yi":"ㄧ", "wu":"ㄨ", "yu":"ㄩ"
}

# 声调符号转换 - Kokoro特有的声调表示
TONE_SYMBOLS = {
    "1": "→",  # 第一声用右箭头
    "2": "↗",  # 第二声用右上箭头
    "3": "↓",  # 第三声用下箭头
    "4": "↘",  # 第四声用右下箭头
    "5": "",   # 轻声不标记
    "0": ""    # 无声调不标记
}

# 标点符号映射
PUNCTUATIONS = {
    '，': ', ',
    '。': '. ',
    '？': '? ',
    '！': '! ',
    '；': '; ',
    '：': ': ',
    '"': ' " ',
    '"': ' " ',
    ''': ' \' ',
    ''': ' \' ',
    '（': ' (',
    '）': ') ',
    '【': ' [',
    '】': '] ',
    '《': ' "',
    '》': '" ',
    '—': ' - ',
    '…': '... ',
    '、': ', '
}

class CustomZHG2P(BaseG2P):
    """定制化中文G2P转换器，专为Kokoro模型设计"""
    
    def __init__(self, use_zhuyin: bool = True, use_special_tones: bool = True):
        """初始化定制化中文G2P转换器
        
        Args:
            use_zhuyin: 是否使用注音符号
            use_special_tones: 是否使用特殊声调符号
        """
        super().__init__()
        
        self.use_zhuyin = use_zhuyin
        self.use_special_tones = use_special_tones
        
        # 初始化拼音处理
        self._init_pypinyin()
        
        # 自定义词典
        self.custom_dict = {
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
            '不': [['bu4']]
        }
        
        # 加载自定义词典
        pypinyin.load_phrases_dict(self.custom_dict)
        
        # 儿化音词集
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
    
    def _init_pypinyin(self):
        """初始化拼音处理"""
        # 加载大词典(如果有)
        if HAS_LARGE_PINYIN:
            large_pinyin.load()
            
        # 调整特定字的拼音
        pypinyin.load_single_dict({ord(u'地'): u'de,di4'})
    
    def _convert_pinyin_to_zhuyin(self, pinyin: str) -> str:
        """将拼音转换为注音符号，改进处理非标准格式
        
        Args:
            pinyin: 带声调的拼音，如"zhong1"
            
        Returns:
            注音符号字符串
        """
        # 严格检查是否为标准拼音格式 (字母+数字)
        if not pinyin or not re.match(r'^[a-z]+[1-5]$', pinyin):
            logger.debug(f"非标准拼音格式，直接返回: '{pinyin}'")
            return pinyin
        
        # 提取声调
        tone = pinyin[-1]
        base_pinyin = pinyin[:-1]
        
        logger.debug(f"处理拼音: '{pinyin}', 基础拼音: '{base_pinyin}', 声调: '{tone}'")
        
        # 转换声母+韵母
        zhuyin = ""
        initial = ""
        final = base_pinyin
        
        # 处理声母 - 先检查双字母声母
        for init in ["zh", "ch", "sh"]:
            if base_pinyin.startswith(init):
                initial = init
                final = base_pinyin[len(init):]
                break
        
        # 如果不是双字母声母，检查单字母声母
        if not initial and base_pinyin:
            if base_pinyin[0] in "bpmfdtnlgkhjqxzcsryw":
                initial = base_pinyin[0]
                final = base_pinyin[1:]
        
        logger.debug(f"分解结果 - 声母: '{initial}', 韵母: '{final}'")
        
        # 特殊情况处理：零声母处理 - 'y'和'w'开头
        if initial in ['y', 'w']:
            # 特殊情况：单个y或w作为声母，直接映射
            if base_pinyin == 'yi':
                zhuyin = PINYIN_TO_ZHUYIN['yi']
                logger.debug(f"特殊处理yi: '{base_pinyin}' -> '{zhuyin}'")
            elif base_pinyin == 'wu':
                zhuyin = PINYIN_TO_ZHUYIN['wu']
                logger.debug(f"特殊处理wu: '{base_pinyin}' -> '{zhuyin}'")
            elif base_pinyin == 'yu':
                zhuyin = PINYIN_TO_ZHUYIN['yu']
                logger.debug(f"特殊处理yu: '{base_pinyin}' -> '{zhuyin}'")
            elif initial == 'y':
                # y作为声母，后面跟韵母，将y转换为ㄧ
                zhuyin = PINYIN_TO_ZHUYIN['y']
                
                # 处理y后面的韵母
                if final in PINYIN_TO_ZHUYIN:
                    # 如果是以'i'开头的韵母，去掉i（因为y已经转为ㄧ了）
                    if final.startswith('i') and len(final) > 1:
                        mod_final = final[1:]
                        if mod_final in PINYIN_TO_ZHUYIN:
                            zhuyin += PINYIN_TO_ZHUYIN[mod_final]
                            logger.debug(f"处理y声母后的i韵母: '{final}' -> '{mod_final}' -> '{PINYIN_TO_ZHUYIN[mod_final]}'")
                        else:
                            zhuyin += final  # 保留原样
                    else:
                        zhuyin += PINYIN_TO_ZHUYIN[final]
                        logger.debug(f"处理y声母后的韵母: '{final}' -> '{PINYIN_TO_ZHUYIN[final]}'")
                else:
                    zhuyin += final  # 保留原样
            elif initial == 'w':
                # w作为声母，后面跟韵母，将w转换为ㄨ
                zhuyin = PINYIN_TO_ZHUYIN['w']
                
                # 处理w后面的韵母
                if final in PINYIN_TO_ZHUYIN:
                    # 如果是以'u'开头的韵母，去掉u（因为w已经转为ㄨ了）
                    if final.startswith('u') and len(final) > 1:
                        mod_final = final[1:]
                        if mod_final in PINYIN_TO_ZHUYIN:
                            zhuyin += PINYIN_TO_ZHUYIN[mod_final]
                            logger.debug(f"处理w声母后的u韵母: '{final}' -> '{mod_final}' -> '{PINYIN_TO_ZHUYIN[mod_final]}'")
                        else:
                            zhuyin += final  # 保留原样
                    else:
                        zhuyin += PINYIN_TO_ZHUYIN[final]
                        logger.debug(f"处理w声母后的韵母: '{final}' -> '{PINYIN_TO_ZHUYIN[final]}'")
                else:
                    zhuyin += final  # 保留原样
        else:
            # 转换声母
            if initial:
                if initial in PINYIN_TO_ZHUYIN:
                    zhuyin_initial = PINYIN_TO_ZHUYIN[initial]
                    zhuyin += zhuyin_initial
                    logger.debug(f"声母转换: '{initial}' -> '{zhuyin_initial}'")
                else:
                    # 未知声母，保留原样
                    logger.warning(f"未知声母: '{initial}'，保留原样")
                    zhuyin += initial
            
            # 转换韵母
            if final in PINYIN_TO_ZHUYIN:
                zhuyin_final = PINYIN_TO_ZHUYIN[final]
                zhuyin += zhuyin_final
                logger.debug(f"韵母转换: '{final}' -> '{zhuyin_final}'")
            elif final:
                # 对于复杂韵母，尝试匹配最长可能的前缀
                matched = False
                for length in range(len(final), 0, -1):
                    if final[:length] in PINYIN_TO_ZHUYIN:
                        zhuyin_final = PINYIN_TO_ZHUYIN[final[:length]]
                        zhuyin += zhuyin_final
                        logger.debug(f"复杂韵母部分转换: '{final[:length]}' -> '{zhuyin_final}'")
                        
                        rest = final[length:]
                        if rest:
                            logger.warning(f"韵母剩余部分无法转换: '{rest}'")
                            zhuyin += rest
                        
                        matched = True
                        break
                
                if not matched:
                    # 无法匹配，直接使用原韵母
                    logger.warning(f"无法转换韵母: '{final}'，保持原样")
                    zhuyin += final
        
        # 添加声调
        if self.use_special_tones and tone in TONE_SYMBOLS:
            tone_symbol = TONE_SYMBOLS[tone]
            if tone_symbol:  # 仅在有声调符号时添加
                zhuyin += tone_symbol
                logger.debug(f"添加特殊声调符号: '{tone}' -> '{tone_symbol}'")
        elif tone != '0' and tone != '5':  # 保留原始声调数字
            zhuyin += tone
            logger.debug(f"添加数字声调: '{tone}'")
        
        logger.debug(f"最终转换结果: '{pinyin}' -> '{zhuyin}'")
        return zhuyin
    
    def _convert_number_to_chinese(self, text: str) -> str:
        """将数字转换为中文读法
        
        Args:
            text: 输入文本
            
        Returns:
            转换后的文本
        """
        # 查找所有数字
        number_pattern = re.compile(r'\d+(?:\.\d+)?')
        
        # 单个数字到中文的映射
        digit_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
            '.': '点'
        }
        
        # 数字到中文的转换函数
        def _num_to_chinese(match):
            num_str = match.group(0)
            
            # 简单数字转换 - 按位读
            chinese = ''
            for char in num_str:
                chinese += digit_map.get(char, char)
            
            return chinese
        
        # 替换所有数字
        return number_pattern.sub(_num_to_chinese, text)
    
    def _process_erhua(self, pinyins: List[List[str]], words: List[str]) -> List[List[str]]:
        """处理儿化音
        
        Args:
            pinyins: 拼音列表
            words: 词列表
            
        Returns:
            处理后的拼音列表
        """
        result = []
        
        for i, (py_list, word) in enumerate(zip(pinyins, words)):
            # 处理词尾儿化
            if word.endswith('儿') and len(word) > 1 and word not in self.not_erhua:
                if len(py_list) > 1 and py_list[-1][0:2] == 'er':
                    # 合并儿化音到前一个音节
                    new_py_list = py_list[:-1]  # 去掉"儿"
                    if py_list[-2][-1] in '12345':
                        # 有声调的情况
                        tone = py_list[-2][-1]
                        new_py = py_list[-2][:-1] + 'r' + tone
                        new_py_list[-1] = new_py
                    else:
                        # 无声调的情况
                        new_py_list[-1] = py_list[-2] + 'r'
                    result.append(new_py_list)
                else:
                    result.append(py_list)
            else:
                result.append(py_list)
        
        return result
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的文本
        """
        # 基础清理
        text = self.sanitize_text(text)
        
        # 标点符号转换
        for zh_punc, en_punc in PUNCTUATIONS.items():
            text = text.replace(zh_punc, en_punc)
        
        # 数字转中文
        text = self._convert_number_to_chinese(text)
        
        return text
    
    def _format_final_phonemes(self, phonemes: List[str]) -> str:
        """格式化最终音素序列
        
        Args:
            phonemes: 音素列表
            
        Returns:
            格式化后的音素字符串
        """
        # 对于Kokoro模型，需要确保音素之间有空格分隔
        formatted = []
        
        for ph in phonemes:
            # 跳过空音素
            if not ph.strip():
                continue
                
            # 标点符号不需要特殊处理
            if ph in [',', '.', '?', '!', ';', ':', "'", '"', '(', ')', '[', ']', '-']:
                formatted.append(ph)
            else:
                # 非标点符号的音素添加到结果
                formatted.append(ph)
        
        # 合并结果，确保用空格分隔
        return " ".join(formatted)
    
    def text_to_phonemes(self, text: str) -> str:
        """将文本转换为音素序列
        
        Args:
            text: 输入文本
            
        Returns:
            音素序列
        """
        if not text:
            return ""
        
        logger.debug(f"原始文本: '{text}'")
        
        # 预处理文本
        text = self.preprocess_text(text)
        logger.debug(f"预处理后文本: '{text}'")
        
        # 分词
        words = jieba.lcut(text)
        logger.debug(f"分词结果: {words}")
        
        # 获取拼音
        pinyin_results = []
        for word in words:
            # 检查是否是汉字
            if re.match(r'[\u4e00-\u9fff]', word):
                py = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
                pinyin_results.append(py)
            else:
                # 非汉字直接作为单个元素添加
                # 如果是标点，直接添加；如果是其他字符，按需处理
                if re.match(r'[,.!?;:\'"\(\)\[\]\-]', word):
                    pinyin_results.append([word])
                else:
                    # 确保非汉字不被误识别为拼音
                    pinyin_results.append([f"#{word}#"])  # 用特殊标记包围
        
        logger.debug(f"拼音结果: {pinyin_results}")
        
        # 处理儿化音
        pinyin_results = self._process_erhua(pinyin_results, words)
        logger.debug(f"儿化音处理后: {pinyin_results}")
        
        # 转换为最终音素
        phonemes = []
        
        for py_list in pinyin_results:
            for py in py_list:
                # 1. 检查是否是标记的非拼音字符
                if py.startswith('#') and py.endswith('#'):
                    # 移除标记并添加原始字符
                    original = py[1:-1]
                    phonemes.append(original)
                    logger.debug(f"保留原始字符: '{original}'")
                    continue
                    
                # 2. 检查是否为标准拼音格式 (字母+数字)
                if re.match(r'^[a-z]+[1-5]$', py):
                    # 中文拼音
                    if self.use_zhuyin:
                        # 转换为注音符号
                        phoneme = self._convert_pinyin_to_zhuyin(py)
                    else:
                        # 使用原始拼音，但可能需要特殊声调
                        if self.use_special_tones and py[-1] in TONE_SYMBOLS:
                            base_py = py[:-1]
                            tone_symbol = TONE_SYMBOLS[py[-1]]
                            phoneme = base_py + tone_symbol
                        else:
                            phoneme = py
                    
                    phonemes.append(phoneme)
                    logger.debug(f"拼音到音素转换: '{py}' -> '{phoneme}'")
                else:
                    # 3. 其他情况直接添加（标点等）
                    if py.strip():
                        phonemes.append(py)
                        logger.debug(f"保留非拼音字符: '{py}'")
        
        # 格式化最终结果
        result = self._format_final_phonemes(phonemes)
        logger.debug(f"最终音素序列 ({len(result.split())} 个音素): '{result}'")
        return result
    
    def get_phoneme_set(self) -> Set[str]:
        """获取音素集合
        
        Returns:
            音素集合
        """
        phoneme_set = set()
        
        if self.use_zhuyin:
            # 添加所有注音符号
            for zhuyin in PINYIN_TO_ZHUYIN.values():
                phoneme_set.add(zhuyin)
            
            # 添加声调符号
            if self.use_special_tones:
                for tone in TONE_SYMBOLS.values():
                    if tone:
                        phoneme_set.add(tone)
        else:
            # 使用拼音时的音素集
            # 声母
            for initial in ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"]:
                phoneme_set.add(initial)
                
            # 韵母
            for final in ["a", "o", "e", "i", "u", "v", "ai", "ei", "ui", "ao", "ou", "iu", "ie", "ue", "ve", "an", "en", "in", "un", "vn", "ang", "eng", "ing", "ong", "er", "ii", "iii"]:
                phoneme_set.add(final)
                
            # 声调
            if self.use_special_tones:
                for tone in TONE_SYMBOLS.values():
                    if tone:
                        phoneme_set.add(tone)
            else:
                for tone in ["1", "2", "3", "4", "5"]:
                    phoneme_set.add(tone)
        
        # 标点符号
        for p in [',', '.', '?', '!', ';', ':', '"', "'", '(', ')', '[', ']', '-', ' ']:
            phoneme_set.add(p)
        
        return phoneme_set
    
    def get_language(self) -> str:
        """获取语言代码
        
        Returns:
            语言代码
        """
        return "zh"
    
    def process_english(self, text: str) -> str:
        """处理英文文本
        
        Args:
            text: 英文文本
            
        Returns:
            处理后的英文文本（简单拼写形式）
        """
        # 简单的英文处理，将单词拆分为字母
        if not text:
            return ""
            
        # 清理文本
        text = self.sanitize_text(text)
        
        # 简单的单词分割
        words = text.split()
        
        # 将每个单词拆分为字母
        result = []
        for word in words:
            # 将单词中的字母用空格分开
            letters = " ".join(word)
            result.append(letters)
        
        # 用空格连接单词
        return " ".join(result) 