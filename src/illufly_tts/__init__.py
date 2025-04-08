#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
illufly-tts: 高质量多语言TTS系统
"""

import logging
from .pipeline import TTSPipeline
from .utils.logging_config import configure_logging

__version__ = "0.1.0"

# 设置基本日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = [
    'TTSPipeline', 
]
