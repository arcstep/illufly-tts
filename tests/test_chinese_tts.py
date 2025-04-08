#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试中文文本到语音的转换，对比官方pipeline和自定义实现的效果
"""

import os
import logging
import pytest
from illufly_tts.vocoders.kokoro_adapter import KokoroAdapter
from illufly_tts.g2p.mixed_g2p import MixedG2P

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chinese_generation():
    """测试中文语音生成，对比官方Pipeline和自定义处理流程"""
    
    # 配置路径
    repo_id = "hexgrad/Kokoro-82M" 
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化组件
    g2p = MixedG2P()
    logger.info(f"使用 repo_id: {repo_id}")
    adapter = KokoroAdapter(
        repo_id=repo_id,
        g2p=g2p
    )
    
    # 测试文本
    test_cases = [
        {
            "text": "你好，这是一个测试。",
            "description": "基础中文句子"
        },
        {
            "text": "今天天气真不错！",
            "description": "带有感叹号的句子"
        },
        {
            "text": "请问，这个功能可以用吗？",
            "description": "带有问号的句子"
        }
    ]
    
    # 获取可用的语音列表
    voices = adapter.list_voices()
    assert voices, "未能从 Adapter 获取可用语音列表"
    logger.info(f"可用语音: {voices}")
    
    # 选择第一个中文语音 (假设命名符合 zf_xxx)
    voice_id = next((v for v in voices if v.startswith("zf")), None)
    assert voice_id is not None, "在可用语音中未找到中文语音 (zf 开头)"
    logger.info(f"选择语音进行测试: {voice_id}")
    
    # 测试每个案例
    for i, case in enumerate(test_cases):
        text = case["text"]
        desc = case["description"]
        logger.info(f"\n测试案例 {i+1}: {desc}")
        logger.info(f"输入文本: {text}")
        
        # 使用官方Pipeline生成
        official_output = os.path.join(output_dir, f"chinese_test_{i+1}_official.wav")
        logger.info("\n使用官方Pipeline生成...")
        success = adapter.process_text(
            text=text,
            voice_id=voice_id,
            output_path=official_output,
            use_pipeline=True
        )
        assert success, f"案例 {i+1} 官方Pipeline生成失败"
        assert os.path.exists(official_output), f"音频文件 {official_output} 未生成"
        logger.info(f"官方Pipeline音频已保存到: {official_output}")
        
        # 使用自定义处理流程生成
        custom_output = os.path.join(output_dir, f"chinese_test_{i+1}_custom.wav")
        logger.info("\n使用自定义处理流程生成...")
        success = adapter.process_text(
            text=text,
            voice_id=voice_id,
            output_path=custom_output,
            use_pipeline=False
        )
        assert success, f"案例 {i+1} 自定义处理流程生成失败"
        assert os.path.exists(custom_output), f"音频文件 {custom_output} 未生成"
        logger.info(f"自定义处理流程音频已保存到: {custom_output}")
        
        logger.info("\n对比信息：")
        logger.info(f"- 官方Pipeline输出: {official_output}")
        logger.info(f"- 自定义处理流程输出: {custom_output}")
        logger.info("-" * 50)

if __name__ == "__main__":
    test_chinese_generation() 