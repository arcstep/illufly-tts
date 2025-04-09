#!/usr/bin/env python
"""
基于MCP规范的文本转语音服务器 - 使用FastMCP实现
"""
import anyio
import click
import json
import logging
import os
import base64
import uuid
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncGenerator

from mcp.server.fastmcp import FastMCP
import torch
import torchaudio

from ..core.service import TTSServiceManager

logger = logging.getLogger(__name__)

async def create_mcp_server(
    repo_id: str = 'hexgrad/Kokoro-82M-v1.1-zh',
    voices_dir: str = 'voices',
    device: Optional[str] = None,
    batch_size: int = 4,
    max_wait_time: float = 0.2,
    chunk_size: int = 200,
    output_dir: Optional[str] = None
):
    """创建MCP服务器实例
    
    Args:
        repo_id: 模型仓库ID
        voices_dir: 语音目录
        device: 计算设备（None表示自动选择）
        batch_size: 批处理大小
        max_wait_time: 最大等待时间
        chunk_size: 文本分块大小
        output_dir: 音频输出目录
    Returns:
        MCP服务器实例
    """
    # 创建TTS服务管理器
    service_manager = TTSServiceManager(
        repo_id=repo_id,
        voices_dir=voices_dir,
        device=device,
        batch_size=batch_size,
        max_wait_time=max_wait_time,
        chunk_size=chunk_size,
        output_dir=output_dir
    )
    
    # 异步启动服务
    await service_manager.start()
    
    # 创建FastMCP服务器
    mcp = FastMCP("illufly-tts-service")
    
    # 注册关闭处理
    if hasattr(mcp, 'sse_app'):
        @mcp.sse_app().on_event("shutdown")
        async def on_shutdown():
            logger.info("正在关闭TTS服务...")
            await service_manager.shutdown()
    
    # 辅助函数：将音频张量转为base64编码
    async def _tensor_to_base64(audio_tensor, sample_rate=24000):
        # 确保张量在CPU上
        audio_tensor = audio_tensor.cpu()
        
        # 确保是2D格式 [channels, time]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # 保存为临时WAV文件
        temp_path = f"/tmp/tts_temp_{uuid.uuid4()}.wav"
        try:
            torchaudio.save(temp_path, audio_tensor, sample_rate)
            
            # 读取文件并转为base64
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()
            
            # 编码为base64
            return base64.b64encode(audio_bytes).decode("utf-8")
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # 单文本转语音工具
    @mcp.tool()
    async def text_to_speech(text: str, voice: str = None) -> str:
        """将单条文本转换为语音
        
        Args:
            text: 要转换的文本
            voice: 语音ID，可选
            
        Returns:
            JSON格式的语音数据（包含音频base64编码）
        """
        # 提交任务
        task_id = await service_manager.submit_task(text, voice or "zf_001")
        
        # 等待任务完成
        while True:
            status = await service_manager.get_task_status(task_id)
            if status["status"] in ["completed", "failed", "canceled"]:
                break
            await anyio.sleep(0.1)
        
        # 检查任务是否成功
        if status["status"] == "failed":
            return json.dumps({"status": "error", "error": status.get("error", "处理失败")}, ensure_ascii=False)
        
        if status["status"] == "canceled":
            return json.dumps({"status": "error", "error": "任务被取消"}, ensure_ascii=False)
        
        # 获取结果并转为base64
        audio_chunks = []
        async for chunk in service_manager.stream_result(task_id):
            audio_base64 = await _tensor_to_base64(chunk)
            audio_chunks.append(audio_base64)
        
        # 返回结果
        result = {
            "status": "success",
            "task_id": task_id,
            "audio_base64": audio_chunks[0] if audio_chunks else "",
            "sample_rate": 24000,
            "created_at": status["created_at"],
            "completed_at": status["completed_at"]
        }
        
        return json.dumps(result, ensure_ascii=False)
    
    # 批量文本转语音工具
    @mcp.tool()
    async def batch_text_to_speech(texts: List[str], voice: str = None) -> str:
        """将多条文本转换为语音（批处理）
        
        Args:
            texts: 要转换的文本列表
            voice: 语音ID，可选
            
        Returns:
            JSON格式的语音数据列表
        """
        voice_id = voice or "zf_001"
        
        # 提交所有任务
        task_ids = []
        for text in texts:
            task_id = await service_manager.submit_task(text, voice_id)
            task_ids.append(task_id)
        
        # 等待所有任务完成
        results = []
        for task_id in task_ids:
            # 等待单个任务完成
            while True:
                status = await service_manager.get_task_status(task_id)
                if status["status"] in ["completed", "failed", "canceled"]:
                    break
                await anyio.sleep(0.1)
            
            # 处理结果
            if status["status"] == "completed":
                # 获取所有音频块
                audio_chunks = []
                async for chunk in service_manager.stream_result(task_id):
                    audio_base64 = await _tensor_to_base64(chunk)
                    audio_chunks.append(audio_base64)
                
                results.append({
                    "status": "success",
                    "task_id": task_id,
                    "audio_base64": audio_chunks[0] if audio_chunks else "",
                    "sample_rate": 24000,
                    "created_at": status["created_at"],
                    "completed_at": status["completed_at"]
                })
            else:
                # 失败或取消
                results.append({
                    "status": "error",
                    "task_id": task_id,
                    "error": status.get("error", f"任务{status['status']}")
                })
        
        return json.dumps(results, ensure_ascii=False)
    
    # 保存语音到文件工具
    @mcp.tool()
    async def save_speech_to_file(text: str, output_path: str, voice: str = None) -> str:
        """将文本转换为语音并保存到文件
        
        Args:
            text: 要转换的文本
            output_path: 输出文件路径
            voice: 语音ID，可选
            
        Returns:
            JSON格式的处理结果
        """
        # 提交任务
        task_id = await service_manager.submit_task(text, voice or "zf_001")
        
        # 等待任务完成
        while True:
            status = await service_manager.get_task_status(task_id)
            if status["status"] in ["completed", "failed", "canceled"]:
                break
            await anyio.sleep(0.1)
        
        # 检查任务是否成功
        if status["status"] != "completed":
            return json.dumps({
                "status": "error",
                "error": status.get("error", f"任务{status['status']}")
            }, ensure_ascii=False)
        
        # 获取结果并保存到文件
        all_audio = []
        async for chunk in service_manager.stream_result(task_id):
            all_audio.append(chunk)
        
        # 如果有多个块，需要合并
        if len(all_audio) > 1:
            # 简单拼接所有音频块
            combined_audio = torch.cat(all_audio, dim=1)
        else:
            combined_audio = all_audio[0] if all_audio else torch.zeros((1, 1))
        
        # 保存到文件
        try:
            output_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存音频
            await service_manager.save_audio_chunk(combined_audio, output_path, 24000)
            
            return json.dumps({
                "status": "success",
                "task_id": task_id,
                "output_path": output_path,
                "sample_rate": 24000,
                "duration": combined_audio.shape[1] / 24000
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"保存文件失败: {str(e)}"
            }, ensure_ascii=False)
    
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
            "version": "0.2.0",
            "model": repo_id,
            "device": device or "auto",
            "batch_size": batch_size,
            "max_wait_time": max_wait_time,
            "chunk_size": chunk_size
        }
        
        return json.dumps(info, ensure_ascii=False)
    
    return mcp


@click.command()
@click.option("--repo-id", default="hexgrad/Kokoro-82M-v1.1-zh", help="模型仓库ID")
@click.option("--voices-dir", default="voices", help="语音目录")
@click.option("--device", default=None, help="计算设备 (None表示自动选择)")
@click.option("--batch-size", default=4, help="批处理大小")
@click.option("--max-wait-time", default=0.2, help="最大等待时间")
@click.option("--chunk-size", default=200, help="文本分块大小")
@click.option("--output-dir", default=None, help="音频输出目录")
@click.option("--port", default=31572, help="SSE传输的端口号")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="传输类型",
)
def main(
    repo_id: str, 
    voices_dir: str, 
    device: Optional[str],
    batch_size: int,
    max_wait_time: float,
    chunk_size: int,
    output_dir: Optional[str],
    port: int, 
    transport: str
) -> int:
    """启动MCP文本转语音服务
    
    Args:
        repo_id: 模型仓库ID
        voices_dir: 语音目录
        device: 计算设备
        batch_size: 批处理大小
        max_wait_time: 最大等待时间
        chunk_size: 文本分块大小
        output_dir: 音频输出目录
        port: SSE传输的端口号
        transport: 传输类型（stdio或sse）
    
    Returns:
        状态码
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # 创建并运行服务器
    if transport == "sse":
        logger.info(f"使用SSE传输 - 监听 0.0.0.0:{port}")
        import uvicorn
        from fastapi import FastAPI
        
        async def run_server():
            mcp = await create_mcp_server(
                repo_id=repo_id,
                voices_dir=voices_dir,
                device=device,
                batch_size=batch_size,
                max_wait_time=max_wait_time,
                chunk_size=chunk_size,
                output_dir=output_dir
            )
            app = mcp.sse_app()
            await uvicorn.run(app, host="0.0.0.0", port=port)
        
        asyncio.run(run_server())
    else:
        logger.info("使用STDIO传输")
        async def run_server():
            mcp = await create_mcp_server(
                repo_id=repo_id,
                voices_dir=voices_dir,
                device=device,
                batch_size=batch_size,
                max_wait_time=max_wait_time,
                chunk_size=chunk_size,
                output_dir=output_dir
            )
            await mcp.run_stdio_async()
        
        try:
            asyncio.run(run_server())
        except KeyboardInterrupt:
            logger.info("服务被用户中断")
        except Exception as e:
            logger.error(f"服务异常退出: {e}", exc_info=True)
            return 1
        return 0


if __name__ == "__main__":
    main() 