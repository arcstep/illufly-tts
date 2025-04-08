#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音合成系统的G2P（Grapheme-to-Phoneme）模块
提供文本到音素的转换功能
"""

from .base_g2p import BaseG2P
from .chinese_g2p import ChineseG2P
from .english_g2p import EnglishG2P

__all__ = ['BaseG2P', 'ChineseG2P', 'EnglishG2P'] 