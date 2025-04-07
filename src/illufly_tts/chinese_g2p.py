#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文G2P（Grapheme-to-Phoneme）模块
将中文文本转换为音素序列
"""

import re
import logging
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# 拼音风格选择
# NORMAL: 不带声调的拼音，如 zhong wen（优点：简洁；缺点：丢失声调信息，可能导致多音字混淆）
# TONE: 声调符号在韵母上的拼音，如 zhōng wén（优点：符合人类阅读习惯；缺点：难以用ASCII表示）
# TONE2: 声调数字在韵母后的拼音，如 zho1ng we2n（优点：保留声调信息；缺点：可能打断音素连贯性）
# TONE3: 声调数字在末尾的拼音，如 zhong1 wen2（优点：保留声调信息，且格式规整；缺点：无）
# INITIALS: 声母，如 zh w（优点：极度简化；缺点：信息损失严重）
# FIRST_LETTER: 首字母，如 z w（优点：更简化；缺点：信息损失最严重）

# 默认推荐TONE3风格，因为它将声调数字放在末尾，不会干扰音素本身的拼写，对模型更友好
PINYIN_STYLE = "TONE3"  

# 尝试导入pypinyin库用于中文转拼音
try:
    from pypinyin import pinyin, Style, lazy_pinyin
    PYPINYIN_AVAILABLE = True
    
    # 创建风格映射
    STYLE_MAP = {
        "NORMAL": Style.NORMAL,          # 不带声调
        "TONE": Style.TONE,              # 声调符号在韵母上，如zhōng
        "TONE2": Style.TONE2,            # 声调数字在韵母后，如zho1ng
        "TONE3": Style.TONE3,            # 声调数字在末尾，如zhong1
        "INITIALS": Style.INITIALS,      # 声母
        "FIRST_LETTER": Style.FIRST_LETTER  # 首字母
    }
    
    # 根据配置获取对应的拼音风格
    STYLE = STYLE_MAP.get(PINYIN_STYLE, Style.TONE3)
    logger.info(f"成功导入pypinyin库，使用拼音风格: {PINYIN_STYLE}")
except ImportError:
    PYPINYIN_AVAILABLE = False
    logger.warning("未安装pypinyin库，将使用原始中文文本作为音素")

def preprocess_text(text: str) -> str:
    """预处理文本，标准化标点符号等"""
    # 处理标点符号，保留基本的标点，去除不必要的符号
    text = re.sub(r'[""''「」『』〈〉《》【】\(\)]', '', text)
    # 替换常见标点为空格或基本标点
    text = re.sub(r'[,，、]', ',', text)
    text = re.sub(r'[.。]', '.', text)
    text = re.sub(r'[?？]', '?', text)
    text = re.sub(r'[!！]', '!', text)
    text = re.sub(r'[;；]', ';', text)
    text = re.sub(r'[：:]', ':', text)
    
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def handle_multi_pronunciations(word: str) -> str:
    """处理多音字词"""
    # 这里可以添加特殊的多音字词处理规则
    # 例如常见的多音字词映射
    multi_readings = {
        "重要": ["zhong4", "yao4"],
        "银行": ["yin2", "hang2"],
        "都是": ["dou1", "shi4"],
        "行业": ["hang2", "ye4"],
        # 可以添加更多特殊词汇
    }
    
    if word in multi_readings:
        return " ".join(multi_readings[word])
    return None  # 没有特殊规则，使用默认处理

def chinese_g2p(text: str, style: str = None) -> str:
    """
    中文G2P函数，将中文文本转换为拼音音素序列
    
    Args:
        text: 中文文本
        style: 可选的拼音风格，覆盖默认设置
        
    Returns:
        处理后的音素序列，如果安装了pypinyin则返回拼音，否则返回原文本
    """
    logger.info(f"中文G2P输入文本: '{text}'")
    
    # 预处理文本
    orig_text = text
    text = preprocess_text(text)
    
    # 重要：检查结果是否为空或只有空格
    if not text or text.isspace():
        logger.warning(f"警告：中文G2P处理结果为空，原文本：'{orig_text}'")
        return orig_text  # 返回原始文本而不是空字符串
    
    # 使用pypinyin将中文转换为拼音
    if PYPINYIN_AVAILABLE:
        try:
            # 使用指定风格或默认风格
            current_style = STYLE_MAP.get(style, STYLE) if style else STYLE
            
            # 方法1: 词级处理 - 尝试处理整个词语
            if True:  # 启用词级处理
                # 检查是否有特殊处理规则
                special_reading = handle_multi_pronunciations(text)
                if special_reading:
                    logger.info(f"使用特殊规则处理: '{text}' -> '{special_reading}'")
                    return special_reading
            
            # 方法2: 使用整句拼音转换
            py_list = pinyin(text, style=current_style, errors='default', heteronym=False)
            
            # 展平结果
            result = []
            for item in py_list:
                if item and item[0]:
                    # 处理拼音结果
                    result.append(item[0])
            
            # 获取最终拼音字符串
            phoneme_text = ' '.join(result)
            
            # 额外的后处理（可选）
            # 例如替换特定音素或调整格式
            # phoneme_text = phoneme_text.replace(..., ...)
            
            logger.info(f"中文G2P输出音素(拼音): '{phoneme_text}'")
            return phoneme_text
            
        except Exception as e:
            logger.error(f"拼音转换失败: {e}")
            logger.warning("使用原始文本作为音素")
    
    # 如果pypinyin不可用或转换失败，返回原始文本
    logger.info(f"中文G2P输出音素(原文): '{text}'")
    return text

def process_text_with_all_styles(text: str) -> Dict[str, str]:
    """使用所有可用拼音风格处理文本，用于比较"""
    results = {}
    if not PYPINYIN_AVAILABLE:
        return {"ORIGINAL": text}
        
    for style_name, style_value in STYLE_MAP.items():
        try:
            result = chinese_g2p(text, style_name)
            results[style_name] = result
        except Exception as e:
            results[style_name] = f"错误: {str(e)}"
    
    return results

class ChineseG2P:
    """
    中文G2P处理类，提供中文音素处理功能
    """
    
    def __init__(self, style: str = None):
        """初始化中文G2P处理器
        
        Args:
            style: 可选的拼音风格，覆盖默认设置
        """
        self.style = style
        logger.info("初始化中文G2P处理器")
        
        # 提示用户安装pypinyin
        if not PYPINYIN_AVAILABLE:
            logger.warning("推荐安装pypinyin库以获得更好的中文处理效果:")
            logger.warning("pip install pypinyin")
    
    def __call__(self, text: str) -> str:
        """
        将中文文本转换为音素序列
        
        Args:
            text: 中文文本
            
        Returns:
            处理后的"音素"序列
        """
        return chinese_g2p(text, self.style)
    
    @staticmethod
    def get_all_styles(text: str) -> Dict[str, str]:
        """使用所有风格处理文本，返回结果字典"""
        return process_text_with_all_styles(text) 