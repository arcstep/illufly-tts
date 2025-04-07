#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
illufly-tts: 高质量多语言TTS系统
"""

import logging
from .pipeline import TTSPipeline, MixedLanguagePipeline
from .preprocessing.normalizer import TextNormalizer
from .preprocessing.segmenter import LanguageSegmenter
from .g2p.mixed_g2p import MixedG2P
from .vocoders.kokoro_adapter import KokoroAdapter, KokoroVoice

__version__ = "0.1.0"

# 设置基本日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = [
    'TTSPipeline',
    'MixedLanguagePipeline',
    'TextNormalizer',
    'LanguageSegmenter',
    'MixedG2P',
    'KokoroAdapter',
    'KokoroVoice',
]
