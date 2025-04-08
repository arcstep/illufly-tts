#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token类，用于在文本处理过程中跟踪单词或字符标记信息
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class MToken:
    """标记类，用于跟踪文本处理中的标记信息"""
    
    text: str  # 原始文本
    tag: str = ""  # 词性标记
    phonemes: Optional[str] = None  # 音素序列
    whitespace: str = ""  # 标记后的空白字符
    start_ts: Optional[float] = None  # 开始时间戳
    end_ts: Optional[float] = None  # 结束时间戳
    
    def __repr__(self) -> str:
        """
        字符串表示
        """
        return f"MToken(text='{self.text}', tag='{self.tag}', phonemes='{self.phonemes}', whitespace='{self.whitespace}')" 