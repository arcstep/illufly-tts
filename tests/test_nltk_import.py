#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试NLTK与g2p_en的资源加载问题修复
"""

import sys
import os
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nltk-test")

def test_nltk_resource_loading():
    """测试NLTK资源加载"""
    logger.info("开始测试NLTK资源加载...")
    
    # 导入EnglishG2P
    from illufly_tts.g2p.english_g2p import EnglishG2P, load_g2p_en, G2P_EN_AVAILABLE
    
    # 检查g2p_en是否可用
    logger.info(f"G2P_EN_AVAILABLE: {G2P_EN_AVAILABLE}")
    
    # 尝试加载g2p_en
    if not G2P_EN_AVAILABLE:
        success = load_g2p_en()
        logger.info(f"加载g2p_en: {'成功' if success else '失败'}")
    
    # 创建EnglishG2P实例
    eng_g2p = EnglishG2P()
    
    # 测试文本转换
    test_text = "Hello world!"
    logger.info(f"测试文本: {test_text}")
    
    phonemes = eng_g2p.text_to_phonemes(test_text)
    logger.info(f"转换结果: {phonemes}")
    
    return phonemes

if __name__ == "__main__":
    test_nltk_resource_loading() 