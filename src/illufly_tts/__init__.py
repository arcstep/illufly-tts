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

# 下载NLTK资源
try:
    import nltk
    
    def download_nltk_resources():
        """下载NLTK所需资源"""
        resources = ['punkt', 'punkt_tab']
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
                print(f"已下载NLTK资源: {resource}")
            except Exception as e:
                print(f"下载NLTK资源 {resource} 失败: {e}")
    
    # 执行下载
    download_nltk_resources()
except ImportError:
    print("NLTK未安装，跳过资源下载")
except Exception as e:
    print(f"下载NLTK资源时出错: {e}")

# 设置默认日志级别
configure_logging(debug_modules=[
    'illufly_tts.g2p.custom_zh_g2p',
    'illufly_tts.vocoders.kokoro_adapter'
])

__all__ = [
    'TTSPipeline',  
    'TextNormalizer',
    'LanguageSegmenter',
    'KokoroAdapter',
    'KokoroVoice',
]
