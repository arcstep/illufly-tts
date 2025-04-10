#!/usr/bin/env python
"""
逐层测试TTS系统各个组件（增强版）
1. 首先测试Pipeline层 - 直接调用CachedTTSPipeline
2. 然后测试FastAPI层 - 使用HTTP请求测试API接口
3. 模拟多用户并发请求 - 测试请求排队和处理
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
import uuid
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("debug_tts")

# 添加目录到 PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

# 测试的文本列表
TEST_TEXTS = [
    "这是第一个测试文本，用于验证TTS系统的基本功能。",
    "这是第二个测试文本，包含一些不同的内容和长度变化。",
    "这是第三个较长的测试文本，它的目的是测试系统对于较长文本的处理能力和效率，以及音频生成质量。",
    "这是最后一个测试文本，我们希望验证批处理能够正确处理多个用户的并发请求。"
]
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
            text=TEST_TEXTS[0],
            voice_id=TEST_VOICE,
            output_path=str(OUTPUT_DIR / "pipeline_output.wav")
        )
        
        logger.info(f"生成成功! 音频形状: {audio.shape}, 保存到: {OUTPUT_DIR/'pipeline_output.wav'}")
        return True
    except Exception as e:
        logger.error(f"Pipeline层测试失败: {e}", exc_info=True)
        return False

async def send_tts_request(session, text, voice, user_id, request_index):
    """发送单个TTS请求"""
    try:
        # 创建请求数据
        request_data = {
            "text": text,
            "voice": voice
        }
        
        # 添加唯一标识符到请求中，模拟不同的用户
        headers = {
            "Content-Type": "application/json",
            "X-User-ID": user_id  # 假设API使用这个头来识别用户
        }
        
        # 发送请求
        logger.info(f"用户 {user_id} 发送第 {request_index} 个TTS请求: {text[:20]}...")
        async with session.post(
            f"{API_URL}/api/tts",
            json=request_data,
            headers=headers
        ) as response:
            # 检查响应
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"用户 {user_id} 请求失败，状态码: {response.status}，错误: {error_text}")
                return False, None
            
            # 解析响应
            result = await response.json()
            
            # 检查结果
            if result.get("status") != "success" or not result.get("audio_base64"):
                logger.error(f"用户 {user_id} 请求响应错误: {result}")
                return False, None
            
            # 解码base64音频
            audio_bytes = base64.b64decode(result["audio_base64"])
            
            # 保存音频文件
            output_path = OUTPUT_DIR / f"user_{user_id}_request_{request_index}.wav"
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            
            logger.info(f"用户 {user_id} 请求 {request_index} 音频已保存: {output_path}，大小: {len(audio_bytes)} 字节")
            
            return True, result
    except Exception as e:
        logger.error(f"用户 {user_id} 请求 {request_index} 失败: {e}", exc_info=True)
        return False, None

async def test_multi_user_requests(num_users=3, requests_per_user=2):
    """测试多用户并发请求"""
    logger.info(f"==== 测试多用户并发请求 ({num_users}用户，每用户{requests_per_user}请求) ====")
    
    try:
        # 创建会话
        async with aiohttp.ClientSession() as session:
            # 创建请求任务
            tasks = []
            
            # 为每个用户创建多个请求
            for user_idx in range(num_users):
                user_id = f"test_user_{user_idx}"
                
                for req_idx in range(requests_per_user):
                    # 随机选择一个测试文本
                    text = TEST_TEXTS[random.randint(0, len(TEST_TEXTS) - 1)]
                    
                    # 创建任务
                    task = send_tts_request(
                        session=session,
                        text=text,
                        voice=TEST_VOICE,
                        user_id=user_id,
                        request_index=req_idx
                    )
                    tasks.append(task)
            
            # 同时发送所有请求
            logger.info(f"同时发送 {len(tasks)} 个请求...")
            results = await asyncio.gather(*tasks)
            
            # 计算成功率
            success_count = sum(1 for success, _ in results if success)
            logger.info(f"请求完成: 成功 {success_count}/{len(tasks)}")
            
            return success_count == len(tasks)
    except Exception as e:
        logger.error(f"多用户测试失败: {e}", exc_info=True)
        return False

async def main():
    """执行分层测试"""
    logger.info("开始测试各层级...")
    
    # 测试Pipeline层
    pipeline_success = await test_pipeline_layer()
    logger.info(f"Pipeline层测试结果: {'成功' if pipeline_success else '失败'}")
    
    # 测试多用户并发请求
    if pipeline_success:
        multi_user_success = await test_multi_user_requests(num_users=3, requests_per_user=2)
        logger.info(f"多用户并发请求测试结果: {'成功' if multi_user_success else '失败'}")
    
    logger.info("测试完成!")

if __name__ == "__main__":
    asyncio.run(main()) 