#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
illufly-tts: 高质量多语言TTS系统
"""

import logging
from .core import TTSPipeline, TTSServiceManager
from .api import mount_tts_service

__version__ = "0.2.1"

# 设置基本日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__all__ = [
    'TTSPipeline', 
    'mount_tts_service',
]
