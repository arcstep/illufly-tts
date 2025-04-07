#!/usr/bin/env python
"""
英文G2P (石墨音素)转换模块

这个模块提供了将英文文本转换为音素的功能，使用g2p_en库或回退到基于字典的方法。
支持离线模式，即使在网络受限环境也能工作。
"""

import logging
import re
import sys
import os
import importlib.util
from pathlib import Path
import socket
import time

logger = logging.getLogger(__name__)

# 模块级别配置
_OFFLINE_MODE = False  # 默认在线模式
_NLTK_DATA_DIR = os.environ.get("NLTK_DATA") or Path.home() / "nltk_data"
_G2P_AVAILABLE = False  # 是否有g2p_en库
_NETWORK_AVAILABLE = True  # 默认认为网络可用

# 预定义的常见英文词汇及其音素映射 (扩展版)
COMMON_WORDS = {
    # 基础词汇
    "hello": "HH AH L OW",
    "world": "W ER L D",
    "this": "DH IH S",
    "is": "IH Z",
    "a": "AH",
    "test": "T EH S T",
    "of": "AH V",
    "english": "IH NG G L IH SH",
    "text": "T EH K S T",
    "to": "T UW",
    "speech": "S P IY CH",
    "synthesis": "S IH N TH AH S IH S",
    "and": "AE N D",
    "chinese": "CH AY N IY Z",
    "mixed": "M IH K S T",
    "for": "F AO R",
    "in": "IH N",
    "on": "AA N",
    "with": "W IH DH",
    "please": "P L IY Z",
    "thank": "TH AE NG K",
    "you": "Y UW",
    
    # 扩展词汇
    "good": "G UH D",
    "morning": "M AO R N IH NG",
    "afternoon": "AE F T ER N UW N",
    "evening": "IY V N IH NG",
    "night": "N AY T",
    "welcome": "W EH L K AH M",
    "example": "IH G Z AE M P AH L",
    "my": "M AY",
    "name": "N EY M",
    "the": "DH AH",
    "i": "AY",
    "me": "M IY",
    "we": "W IY",
    "they": "DH EY",
    "have": "HH AE V",
    "has": "HH AE Z",
    "had": "HH AE D",
    "do": "D UW",
    "does": "D AH Z",
    "did": "D IH D",
    "what": "W AH T",
    "who": "HH UW",
    "how": "HH AW",
    "why": "W AY",
    "where": "W EH R",
    "when": "W EH N",
    "which": "W IH CH",
    "project": "P R AA JH EH K T",
    "computer": "K AH M P Y UW T ER",
    "language": "L AE NG G W AH JH",
    "time": "T AY M",
    "day": "D EY",
    "today": "T AH D EY",
    "tomorrow": "T AH M AA R OW",
    "yesterday": "Y EH S T ER D EY",
    "now": "N AW",
    "later": "L EY T ER",
    "never": "N EH V ER",
    "always": "AO L W EY Z",
    "sometimes": "S AH M T AY M Z",
    "yes": "Y EH S",
    "no": "N OW",
    "maybe": "M EY B IY",
    "high": "HH AY", 
    "low": "L OW",
    "quality": "K W AA L AH T IY",
    "first": "F ER S T",
    "last": "L AE S T",
    "can": "K AE N",
    "could": "K UH D",
    "would": "W UH D",
    "should": "SH UH D",
    "will": "W IH L",
    "song": "S AO NG",
    "music": "M Y UW Z IH K",
    "voice": "V OY S",
    "sound": "S AW N D",
    "speak": "S P IY K",
    "speaker": "S P IY K ER",
    "listen": "L IH S AH N",
    "hear": "HH IY R",
    "make": "M EY K",
    "create": "K R IY EY T",
    "use": "Y UW Z",
    "work": "W ER K",
    "great": "G R EY T",
    "beautiful": "B Y UW T AH F AH L",
    "nice": "N AY S",
    "okay": "OW K EY",
    "fine": "F AY N",
    "sorry": "S AA R IY",
    "excuse": "IH K S K Y UW S",
    "help": "HH EH L P",
    "thanks": "TH AE NG K S",
    "hi": "HH AY",
    "bye": "B AY",
    "goodbye": "G UH D B AY",
    "here": "HH IY R",
    "there": "DH EH R",
    "one": "W AH N",
    "two": "T UW",
    "three": "TH R IY",
    "four": "F AO R",
    "five": "F AY V",
    "six": "S IH K S",
    "seven": "S EH V AH N",
    "eight": "EY T",
    "nine": "N AY N",
    "ten": "T EH N"
}

# 英文单词规则的替换表
SYLLABLE_PATTERNS = {
    r"ing\b": "IH NG",
    r"ed\b": "D",
    r"es\b": "Z",
    r"s\b": "S",
    r"er\b": "ER",
    r"or\b": "ER",
    r"tion\b": "SH AH N",
    r"sion\b": "ZH AH N",
    r"ly\b": "L IY",
    r"ment\b": "M AH N T",
    r"ness\b": "N AH S",
    r"ful\b": "F AH L",
    r"able\b": "AH B AH L",
    r"al\b": "AH L",
    r"ic\b": "IH K",
    r"ize\b": "AY Z",
    r"ise\b": "AY Z",
    r"ity\b": "IH T IY",
    r"age\b": "IH JH",
    r"ive\b": "IH V",
    r"ist\b": "IH S T",
}

# 辅音音素对应
CONSONANT_MAP = {
    'b': 'B',
    'c': 'K',
    'd': 'D',
    'f': 'F',
    'g': 'G',
    'h': 'HH',
    'j': 'JH',
    'k': 'K',
    'l': 'L',
    'm': 'M',
    'n': 'N',
    'p': 'P',
    'q': 'K W',
    'r': 'R',
    's': 'S',
    't': 'T',
    'v': 'V',
    'w': 'W',
    'x': 'K S',
    'y': 'Y',
    'z': 'Z',
    'ch': 'CH',
    'sh': 'SH',
    'th': 'TH',
    'ph': 'F',
    'wh': 'W',
    'gh': 'G',
}

# 元音音素对应 (简化版)
VOWEL_MAP = {
    'a': 'AE',
    'e': 'EH',
    'i': 'IH',
    'o': 'AA',
    'u': 'AH',
    'ai': 'EY',
    'ay': 'EY',
    'ee': 'IY',
    'ea': 'IY',
    'ie': 'IY',
    'oo': 'UW',
    'ou': 'AW',
    'ow': 'AW',
    'oi': 'OY',
    'oy': 'OY',
    'au': 'AO',
    'aw': 'AO',
    'ue': 'UW',
    'ui': 'UW',
}

def check_network():
    """检查网络连接状态"""
    global _NETWORK_AVAILABLE
    
    # 尝试连接常用服务器
    test_servers = [
        ('www.baidu.com', 443),
        ('mirrors.aliyun.com', 443),
        ('pypi.org', 443)
    ]
    
    for server, port in test_servers:
        try:
            socket.create_connection((server, port), timeout=1)
            _NETWORK_AVAILABLE = True
            return True
        except (socket.timeout, socket.error, OSError):
            continue
    
    _NETWORK_AVAILABLE = False
    return False

def check_g2p_available():
    """检查g2p_en库是否可用"""
    global _G2P_AVAILABLE
    try:
        spec = importlib.util.find_spec('g2p_en')
        _G2P_AVAILABLE = spec is not None
        return _G2P_AVAILABLE
    except ImportError:
        _G2P_AVAILABLE = False
        return False

def nltk_data_exists(resource):
    """检查NLTK资源是否已下载"""
    try:
        import nltk
        try:
            nltk.data.find(resource)
            return True
        except LookupError:
            return False
    except ImportError:
        return False

def set_offline_mode(offline=True):
    """设置离线模式"""
    global _OFFLINE_MODE
    _OFFLINE_MODE = offline
    if offline:
        logger.info("已启用离线模式，将使用内置词典进行英文处理")
    else:
        logger.info("已禁用离线模式，将优先使用g2p_en库")

class EnglishG2P:
    """英文G2P转换类"""
    
    def __init__(self, offline_mode=None):
        """
        初始化英文G2P处理器
        
        Args:
            offline_mode: 是否使用离线模式，None表示自动检测
        """
        global _OFFLINE_MODE, _G2P_AVAILABLE, _NETWORK_AVAILABLE
        
        # 设置模式
        self.offline_mode = _OFFLINE_MODE if offline_mode is None else offline_mode
        self.g2p = None
        
        # 自动检测
        if not self.offline_mode:
            # 检查g2p_en是否可用
            if not _G2P_AVAILABLE:
                _G2P_AVAILABLE = check_g2p_available()
                
            if not _G2P_AVAILABLE:
                logger.warning("未找到g2p_en库，将使用内置词典处理英文")
                self.offline_mode = True
            else:
                # 检查网络连接
                if not _NETWORK_AVAILABLE and not check_network():
                    logger.warning("网络连接不可用，切换到离线模式")
                    self.offline_mode = True
        
        # 初始化G2P处理器
        if not self.offline_mode:
            try:
                # 尝试导入g2p_en模块
                import g2p_en
                
                # 尝试预加载NLTK资源
                try:
                    import nltk
                    # 设置NLTK数据目录
                    if os.path.exists(str(_NLTK_DATA_DIR)):
                        nltk.data.path.append(str(_NLTK_DATA_DIR))
                        
                    # 相关NLTK资源
                    resources = [
                        'taggers/averaged_perceptron_tagger',
                        'corpora/cmudict'
                    ]
                    
                    missing_resources = []
                    for res in resources:
                        if not nltk_data_exists(res):
                            missing_resources.append(res.split('/')[-1])
                    
                    if missing_resources:
                        if not _NETWORK_AVAILABLE and not check_network():
                            logger.warning(f"网络不可用，无法下载NLTK资源: {', '.join(missing_resources)}")
                            logger.warning("切换到离线模式")
                            self.offline_mode = True
                            return
                        
                        # 尝试下载缺失资源
                        for res_name in missing_resources:
                            logger.info(f"正在下载NLTK资源: {res_name}")
                            try:
                                nltk.download(res_name, quiet=True, download_dir=str(_NLTK_DATA_DIR))
                                logger.info(f"NLTK资源下载成功: {res_name}")
                            except Exception as e:
                                logger.warning(f"NLTK资源下载失败: {res_name}, 错误: {e}")
                                if res_name == 'averaged_perceptron_tagger':
                                    logger.warning("标注器下载失败，切换到离线模式")
                                    self.offline_mode = True
                                    return
                
                except Exception as e:
                    logger.warning(f"NLTK资源初始化失败: {e}")
                
                # 初始化G2P
                self.g2p = g2p_en.G2p()
                logger.info("成功加载g2p_en库")
                
            except Exception as e:
                logger.warning(f"g2p_en初始化失败: {e}")
                logger.warning("将使用备用拼音转换方法")
                self.offline_mode = True
    
    def convert(self, text):
        """将英文文本转换为音素"""
        if not text:
            return ""
        
        # 转换为小写以便字典查找
        text = text.lower()
        
        # 尝试使用g2p_en库进行转换（如果可用且非离线模式）
        if self.g2p and not self.offline_mode:
            try:
                start_time = time.time()
                phonemes = self.g2p(text)
                phoneme_str = " ".join(phonemes)
                end_time = time.time()
                
                logger.info(f"使用g2p_en处理文本: {text}")
                logger.info(f"g2p_en转换结果: {phoneme_str}")
                logger.debug(f"g2p_en处理耗时: {end_time - start_time:.3f}秒")
                
                return phoneme_str
            except Exception as e:
                logger.warning(f"g2p_en处理失败: {e}，使用备用方法")
        
        # 使用备用方法：基于词典和规则的转换
        return self._enhanced_fallback_conversion(text)
    
    def _enhanced_fallback_conversion(self, text):
        """增强的备用转换方法，基于词典和语言规则"""
        logger.info(f"使用备用方法处理: {text}")
        
        # 单词切分
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 转换每个单词
        result_phonemes = []
        for word in words:
            # 1. 首先尝试字典查找
            if word in COMMON_WORDS:
                result_phonemes.append(COMMON_WORDS[word])
                continue
                
            # 2. 尝试应用规则
            phonemes = self._apply_word_rules(word)
            if phonemes:
                result_phonemes.append(phonemes)
                continue
                
            # 3. 回退到字符级别的转换
            result_phonemes.append(self._character_to_phoneme(word))
        
        # 合并结果
        result = " ".join(result_phonemes)
        logger.info(f"备用方法转换结果: {result}")
        return result
    
    def _apply_word_rules(self, word):
        """应用单词规则来生成音素"""
        # 检查单词是否含有常见后缀，并应用规则
        original_word = word
        phonemes_parts = []
        
        # 检查后缀
        for pattern, phoneme in SYLLABLE_PATTERNS.items():
            match = re.search(pattern, word)
            if match:
                # 移除后缀，添加音素
                suffix = match.group(0)
                word = word.replace(suffix, "")
                phonemes_parts.append(phoneme)
                break
        
        # 如果已处理后缀，并且剩余部分在字典中，则使用字典
        if phonemes_parts and word in COMMON_WORDS:
            return f"{COMMON_WORDS[word]} {' '.join(phonemes_parts)}"
        
        # 如果没有处理任何后缀或未找到剩余部分，返回None
        if word == original_word:
            return None
            
        # 处理剩余部分
        word_phonemes = self._character_to_phoneme(word)
        return f"{word_phonemes} {' '.join(phonemes_parts)}"
    
    def _character_to_phoneme(self, word):
        """将单词按字符转换为音素"""
        phonemes = []
        i = 0
        
        while i < len(word):
            # 检查双字符辅音
            if i < len(word) - 1 and word[i:i+2] in CONSONANT_MAP:
                phonemes.append(CONSONANT_MAP[word[i:i+2]])
                i += 2
                continue
                
            # 检查双字符元音
            if i < len(word) - 1 and word[i:i+2] in VOWEL_MAP:
                phonemes.append(VOWEL_MAP[word[i:i+2]])
                i += 2
                continue
                
            # 检查单字符
            if word[i] in CONSONANT_MAP:
                phonemes.append(CONSONANT_MAP[word[i]])
            elif word[i] in VOWEL_MAP:
                phonemes.append(VOWEL_MAP[word[i]])
            else:
                phonemes.append(word[i])
            
            i += 1
            
        return " ".join(phonemes)

# 创建单例实例
_g2p_instance = None

def english_g2p(text, offline_mode=None):
    """转换英文文本为音素的便捷函数"""
    global _g2p_instance
    
    # 如果实例不存在或离线模式变更，重新创建实例
    if _g2p_instance is None or (_g2p_instance.offline_mode != _OFFLINE_MODE and offline_mode is None):
        _g2p_instance = EnglishG2P(offline_mode)
        
    return _g2p_instance.convert(text)

# 初始函数，自动检测环境
def initialize():
    """初始化函数，自动检测环境和设置模式"""
    global _OFFLINE_MODE, _G2P_AVAILABLE, _NETWORK_AVAILABLE
    
    # 检查环境变量
    if os.environ.get("KOKORO_OFFLINE") == "1":
        _OFFLINE_MODE = True
        logger.info("根据环境变量设置为离线模式")
        return
        
    # 检查g2p_en是否可用
    _G2P_AVAILABLE = check_g2p_available()
    if not _G2P_AVAILABLE:
        logger.info("g2p_en库不可用，设置为离线模式")
        _OFFLINE_MODE = True
        return
        
    # 检查网络连接
    _NETWORK_AVAILABLE = check_network()
    if not _NETWORK_AVAILABLE:
        logger.info("网络连接不可用，设置为离线模式")
        _OFFLINE_MODE = True
        return
        
    logger.info("英文G2P模块初始化完成，使用在线模式")

# 自动初始化
initialize()

# 简单测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_text = "Hello world, this is a test of English G2P conversion."
    print(f"Input: {test_text}")
    print(f"Output (Online mode): {english_g2p(test_text, offline_mode=False)}")
    print(f"Output (Offline mode): {english_g2p(test_text, offline_mode=True)}")
    # 使用一些不在字典中的单词测试规则
    test_text2 = "running quickly wonderful happiness computerization"
    print(f"\nTesting word rules: {test_text2}")
    print(f"Output: {english_g2p(test_text2, offline_mode=True)}") 