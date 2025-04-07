#!/usr/bin/env python
"""
基于MCP规范的文本转语音服务器 - 使用FastMCP实现
"""
import anyio
import click
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from mcp.server.fastmcp import FastMCP

from .tts import TTSService

logger = logging.getLogger(__name__)

def create_mcp_server(
    repo_id: str = 'hexgrad/Kokoro-82M-v1.1-zh',
    sample_rate: int = 24000,
    voice: str = 'zf_001',
    cache_size: int = 100,
    device: Optional[str] = None,
    model_path: Optional[str] = None
):
    """创建MCP服务器实例
    
    Args:
        repo_id: 模型仓库ID
        sample_rate: 音频采样率
        voice: 默认语音
        cache_size: 缓存大小
        device: 计算设备（None表示自动选择）
        model_path: 本地模型路径，若指定则使用本地模型而非下载
    Returns:
        MCP服务器实例
    """
    # 创建TTS服务
    tts_service = TTSService(
        repo_id=repo_id,
        sample_rate=sample_rate,
        voice=voice,
        cache_size=cache_size,
        device=device,
        model_path=model_path
    ).start()
    
    # 创建FastMCP服务器
    mcp = FastMCP("illufly-tts-service")
    
    # 单文本转语音工具
    @mcp.tool()
    async def text_to_speech(text: str, voice: str = None) -> str:
        """将单条文本转换为语音
        
        Args:
            text: 要转换的文本
            voice: 语音ID，可选，默认使用配置的语音
            
        Returns:
            JSON格式的语音数据（包含音频base64编码）
        """
        result = tts_service.text_to_speech_sync(text, voice)
        return json.dumps(result, ensure_ascii=False)
    
    # 批量文本转语音工具
    @mcp.tool()
    async def batch_text_to_speech(texts: List[str], voice: str = None) -> str:
        """将多条文本转换为语音（批处理）
        
        Args:
            texts: 要转换的文本列表
            voice: 语音ID，可选，默认使用配置的语音
            
        Returns:
            JSON格式的语音数据列表
        """
        results = []
        async for result in tts_service.text_to_speech(texts, voice):
            results.append(result)
        
        return json.dumps(results, ensure_ascii=False)
    
    # 保存语音到文件工具
    @mcp.tool()
    async def save_speech_to_file(text: str, output_path: str, voice: str = None) -> str:
        """将文本转换为语音并保存到文件
        
        Args:
            text: 要转换的文本
            output_path: 输出文件路径
            voice: 语音ID，可选，默认使用配置的语音
            
        Returns:
            JSON格式的处理结果
        """
        result = await tts_service.save_speech_to_file(text, output_path, voice)
        return json.dumps(result, ensure_ascii=False)
    
    # 获取可用语音列表
    @mcp.tool()
    async def get_available_voices() -> str:
        """获取可用的语音列表
        
        Returns:
            JSON格式的语音列表
        """
        # 当前只有一个语音可用
        voices = [
            {"id": "zf_001", "name": "普通话女声", "description": "标准普通话女声"}
        ]
        
        return json.dumps(voices, ensure_ascii=False)
    
    # 获取服务信息
    @mcp.tool()
    async def get_service_info() -> str:
        """获取服务信息
        
        Returns:
            JSON格式的服务信息
        """
        info = {
            "service": "illufly-tts-service",
            "version": "0.1.0",
            "model": tts_service.repo_id,
            "sample_rate": tts_service.sample_rate,
            "default_voice": tts_service.voice,
            "device": tts_service.device
        }
        
        return json.dumps(info, ensure_ascii=False)
    
    # 在应用关闭时注册清理任务
    async def cleanup():
        logger.info("正在关闭TTS服务...")
        await tts_service.stop()
    
    mcp.on_shutdown(cleanup)
    
    return mcp


@click.command()
@click.option("--repo-id", default="hexgrad/Kokoro-82M-v1.1-zh", help="模型仓库ID")
@click.option("--sample-rate", default=24000, help="音频采样率")
@click.option("--voice", default="zf_001", help="默认语音")
@click.option("--cache-size", default=100, help="缓存大小")
@click.option("--device", default=None, help="计算设备 (None表示自动选择)")
@click.option("--port", default=31572, help="SSE传输的端口号")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="传输类型",
)
@click.option(
    "--model-path", 
    default=None, 
    help="本地模型路径，若指定则使用本地模型而非下载"
)
def main(
    repo_id: str, 
    sample_rate: int, 
    voice: str, 
    cache_size: int, 
    device: Optional[str],
    port: int, 
    transport: str,
    model_path: Optional[str] = None
) -> int:
    """启动MCP文本转语音服务
    
    Args:
        repo_id: 模型仓库ID
        sample_rate: 音频采样率
        voice: 默认语音
        cache_size: 缓存大小
        device: 计算设备
        port: SSE传输的端口号
        transport: 传输类型（stdio或sse）
        model_path: 本地模型路径
    
    Returns:
        状态码
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # 创建MCP服务器
    mcp = create_mcp_server(
        repo_id=repo_id,
        sample_rate=sample_rate,
        voice=voice,
        cache_size=cache_size,
        device=device,
        model_path=model_path
    )
    
    logger.info(f"启动TTS服务")
    
    # 启动服务器
    if transport == "sse":
        logger.info(f"使用SSE传输 - 监听 0.0.0.0:{port}")
        import uvicorn
        from fastapi import FastAPI
        
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        logger.info("使用STDIO传输")
        import asyncio
        asyncio.run(mcp.run_stdio_async())
    
    return 0


if __name__ == "__main__":
    main() 