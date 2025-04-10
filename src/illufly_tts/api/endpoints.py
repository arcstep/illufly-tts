"""
文本转语音服务的FastAPI接口 - 直接使用TTSServiceManager和Pipeline的简化版本
"""
import os
import logging
import asyncio
import base64
import io
import time
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse

# 直接导入ServiceManager和Pipeline
from ..core.service import TTSServiceManager
from ..core.pipeline import CachedTTSPipeline

# 导入JWT验证相关
from .auth import require_user, JWT_ACCESS_TOKEN_EXPIRE_MINUTES

# 导入开发模式相关
from .dev_mode import is_dev_mode
from .dev_endpoints import create_dev_router

logger = logging.getLogger(__name__)

# 定义请求模型
class TextToSpeechRequest(BaseModel):
    """文本转语音请求"""
    text: str
    voice_id: str = "zf_001"
    speed: float = 1.0
    sequence_id: Optional[int] = None
    cancel_pending: bool = False  # 是否取消用户的待处理请求

# 定义用户类型
UserDict = Dict[str, Any]

def mount_tts_service(
    app: FastAPI,
    repo_id: str = "hexgrad/Kokoro-82M-v1.1-zh",
    voices_dir: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 4,
    max_wait_time: float = 0.2,
    chunk_size: int = 200,
    output_dir: Optional[str] = None,
    prefix: str = "/api"
) -> None:
    """挂载TTS服务到FastAPI应用"""
    # 创建路由
    router = APIRouter()
    
    # 为ServiceManager指定的输出目录（如果未指定）
    if not output_dir:
        output_dir = os.path.join(tempfile.gettempdir(), "illufly_tts_output")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"未指定输出目录，将使用临时目录: {output_dir}")
    
    logger.info(f"创建TTSServiceManager: repo_id={repo_id}, voices_dir={voices_dir or '使用HF缓存'}, output_dir={output_dir}")
    
    @app.on_event("startup")
    async def startup_service_manager():
        # 创建服务管理器实例，指定output_dir确保保存音频文件
        app.state.service_manager = TTSServiceManager(
            repo_id=repo_id,
            voices_dir=voices_dir,
            device=device,
            batch_size=batch_size,
            max_wait_time=max_wait_time,
            chunk_size=chunk_size,
            output_dir=output_dir  # 确保指定output_dir
        )
        
        # 启动服务
        await app.state.service_manager.start()
        logger.info("TTS服务已启动")
    
    # 获取服务管理器的依赖
    async def get_service_manager():
        return app.state.service_manager
    
    # 获取Pipeline的依赖
    async def get_pipeline():
        return app.state.pipeline

    # 首先创建共用的处理函数
    async def _process_tts_request(
        text: str,
        voice_id: str,
        user_id: Optional[str],
        sequence_id: Optional[float],
        service_manager: TTSServiceManager
    ) -> Dict[str, Any]:
        """处理单个TTS请求的内部函数"""
        # 提交任务
        task_id = await service_manager.submit_task(
            text=text,
            voice_id=voice_id,
            user_id=user_id,
            sequence_id=sequence_id
        )
        
        # 等待任务完成
        while True:
            status = await service_manager.get_task_status(task_id)
            if status["status"] in ["completed", "failed", "canceled"]:
                break
            await asyncio.sleep(0.1)
        
        # 检查任务状态
        if status["status"] != "completed":
            error_message = status.get("error", "处理失败")
            logger.error(f"TTS任务失败: {error_message}")
            return {
                "status": "error",
                "task_id": task_id,
                "error": error_message
            }
        
        # 音频文件路径
        output_file_path = os.path.join(service_manager.output_dir, f"{task_id}.wav")
        
        # 添加重试机制，以防文件系统延迟
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            if os.path.exists(output_file_path):
                break
            logger.warning(f"文件尚未就绪，等待重试({retry_count+1}/{max_retries}): {output_file_path}")
            await asyncio.sleep(0.2)  # 等待200ms
            retry_count += 1
        
        # 检查文件是否存在
        if not os.path.exists(output_file_path):
            logger.error(f"音频文件不存在(已重试{max_retries}次): {output_file_path}")
            return {
                "status": "error",
                "task_id": task_id,
                "error": "音频文件未生成"
            }
        
        # 读取文件并转换为base64
        with open(output_file_path, "rb") as f:
            file_bytes = f.read()
            audio_base64 = base64.b64encode(file_bytes).decode("utf-8")
        
        # 返回结果
        return {
            "status": "success",
            "task_id": task_id,
            "audio_base64": audio_base64,
            "sample_rate": 24000,
            "created_at": status["created_at"],
            "completed_at": status["completed_at"]
        }

    @router.post("/tts", response_class=JSONResponse)
    async def text_to_speech(
        request: TextToSpeechRequest, 
        user: UserDict = Depends(require_user()),
        service_manager: TTSServiceManager = Depends(get_service_manager)
    ):
        """
        将文本转换为语音
        """
        # 使用JWT中的用户ID而不是请求参数中的ID
        user_id = user.get("user_id")
        
        # 日志记录用户信息，便于调试
        logger.info(f"处理TTS请求: 用户ID={user_id}, 用户信息={user}")
        
        # 如果设置了取消选项，则取消该用户的所有待处理任务
        if request.cancel_pending and user_id:
            canceled_count = await service_manager.cancel_user_pending_tasks(user_id)
            logger.info(f"已取消用户 {user_id} 的 {canceled_count} 个待处理任务")
        
        try:
            # 设置输出目录，如果管理器没有设置的话
            if not service_manager.output_dir:
                temp_dir = os.path.join(tempfile.gettempdir(), "illufly_tts_output")
                os.makedirs(temp_dir, exist_ok=True)
                service_manager.output_dir = temp_dir
                logger.info(f"为ServiceManager设置临时输出目录: {temp_dir}")
            
            # 提交任务
            logger.info(f"提交TTS任务: 文本长度={len(request.text)}, 语音={request.voice_id}, 序列ID={request.sequence_id}")
            
            # 调用共用处理函数
            result = await _process_tts_request(
                text=request.text,
                voice_id=request.voice_id,
                user_id=user_id,  # 使用JWT中的用户ID
                sequence_id=request.sequence_id,
                service_manager=service_manager
            )
            
            # 检查是否出错
            if result["status"] == "error":
                raise HTTPException(status_code=400, detail=result["error"])
            
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"TTS处理失败: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
        
    @router.get("/tts/voices")
    async def get_voices(
        user: UserDict = Depends(require_user())  # 使用新的验证依赖
    ):
        """获取可用语音列表"""
        # 目前只有一个语音
        voices = [
            {"id": "zf_001", "name": "普通话女声", "description": "标准普通话女声"}
        ]
        return {"voices": voices}
    
    @router.get("/tts/info")
    async def get_service_info(
        user: UserDict = Depends(require_user()),  # 使用新的验证依赖
        service_manager: TTSServiceManager = Depends(get_service_manager)
    ):
        """获取服务信息"""
        return {
            "service": "illufly-tts-service",
            "version": "0.3.0",
            "model": repo_id,
            "device": device or "auto",
            "batch_size": batch_size,
            "max_wait_time": max_wait_time,
            "chunk_size": chunk_size
        }
    
    # 注册路由
    app.include_router(router, prefix=prefix)
    
    # 如果开发模式已启用，添加开发模式路由
    if is_dev_mode():
        logger.info("开发模式已启用，添加开发模式API端点")
        dev_router = create_dev_router()
        app.include_router(dev_router, prefix=prefix)
    
    # 应用关闭时关闭服务
    @app.on_event("shutdown")
    async def shutdown_service_manager():
        if hasattr(app.state, "service_manager"):
            logger.info("关闭TTS服务...")
            await app.state.service_manager.shutdown()