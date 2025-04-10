#!/usr/bin/env python
"""
逐层测试TTS系统各个组件（简化版本）
1. 首先测试Pipeline层 - 直接调用CachedTTSPipeline
2. 然后测试FastAPI层 - 使用HTTP请求测试API接口
"""
import asyncio
import os
import logging
import torch
import torchaudio
import json
import sys
import aiohttp
import base64
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("debug_tts")

# 添加目录到 PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

# 测试的文本
TEST_TEXT = "这是一个测试文本，用于验证TTS系统各个层级的功能。"
TEST_VOICE = "zf_001"
OUTPUT_DIR = Path("./output/debug_test")

# API测试服务器地址
API_URL = "http://localhost:31572"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def test_pipeline_layer():
    """测试Pipeline层 - 直接调用CachedTTSPipeline"""
    logger.info("==== 测试Pipeline层 ====")
    
    try:
        from src.illufly_tts.core.pipeline import CachedTTSPipeline
        
        logger.info("创建CachedTTSPipeline实例...")
        
        # 使用实际的repo_id和voices_dir
        pipeline = CachedTTSPipeline(
            repo_id="hexgrad/Kokoro-82M-v1.1-zh",
            device=None  # 自动选择设备
        )
        
        logger.info("开始生成音频...")
        # 调用pipeline生成音频
        audio = pipeline.process(
            text=TEST_TEXT,
            voice_id=TEST_VOICE,
            output_path=str(OUTPUT_DIR / "pipeline_output.wav")
        )
        
        logger.info(f"生成成功! 音频形状: {audio.shape}, 保存到: {OUTPUT_DIR/'pipeline_output.wav'}")
        return True
    except Exception as e:
        logger.error(f"Pipeline层测试失败: {e}", exc_info=True)
        return False

async def test_fastapi_layer():
    """测试FastAPI层 - 直接调用API接口"""
    logger.info("==== 测试FastAPI层 ====")
    
    try:
        # 确保API服务器在运行
        logger.info(f"将连接到API服务器: {API_URL}")
        logger.info("请确保TTS服务已在运行（python -m illufly_tts serve）")
        
        # 请求API生成语音
        async with aiohttp.ClientSession() as session:
            # 创建请求数据
            request_data = {
                "text": TEST_TEXT,
                "voice": TEST_VOICE
            }
            
            # 发送请求
            logger.info(f"发送TTS请求到 {API_URL}/api/tts")
            async with session.post(
                f"{API_URL}/api/tts",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                # 检查响应
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API请求失败，状态码: {response.status}，错误: {error_text}")
                    return False
                
                # 解析响应
                result = await response.json()
                logger.info(f"API响应: {json.dumps(result, ensure_ascii=False)[:100]}...")
                
                # 检查结果
                if result.get("status") != "success" or not result.get("audio_base64"):
                    logger.error(f"API响应错误: {result}")
                    return False
                
                # 解码base64音频
                audio_bytes = base64.b64decode(result["audio_base64"])
                
                # 验证WAV文件头
                if not audio_bytes.startswith(b'RIFF') or b'WAVE' not in audio_bytes[:12]:
                    logger.warning(f"解码的音频字节可能不是有效的WAV文件，前12字节: {audio_bytes[:12]}")
                
                # 保存音频文件
                output_path = OUTPUT_DIR / "fastapi_output.wav"
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                
                logger.info(f"音频已保存到: {output_path}，大小: {len(audio_bytes)} 字节")
                
                # 如果成功生成并保存音频，则返回成功
                return True
    except Exception as e:
        logger.error(f"FastAPI层测试失败: {e}", exc_info=True)
        return False

async def main():
    """执行分层测试"""
    logger.info("开始测试各层级...")
    
    # 测试Pipeline层
    pipeline_success = await test_pipeline_layer()
    logger.info(f"Pipeline层测试结果: {'成功' if pipeline_success else '失败'}")
    
    # 测试FastAPI层
    if pipeline_success:
        fastapi_success = await test_fastapi_layer()
        logger.info(f"FastAPI层测试结果: {'成功' if fastapi_success else '失败'}")
    
    logger.info("测试完成!")

if __name__ == "__main__":
    asyncio.run(main()) 