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
from fastapi.responses import JSONResponse

# 直接导入ServiceManager和Pipeline
from ..core.service import TTSServiceManager
from ..core.pipeline import CachedTTSPipeline

logger = logging.getLogger(__name__)

# 定义请求模型
class TextToSpeechRequest(BaseModel):
    """文本转语音请求"""
    text: str
    voice: Optional[str] = None

class BatchTextToSpeechRequest(BaseModel):
    """批量文本转语音请求"""
    texts: List[str]
    voice: Optional[str] = None

# 定义用户类型
UserDict = Dict[str, Any]

def mount_tts_service(
    app: FastAPI,
    require_user: Callable[[], Awaitable[UserDict]],
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
    
    @router.post("/tts")
    async def text_to_speech(
        request: TextToSpeechRequest, 
        user: UserDict = Depends(require_user),
        service_manager: TTSServiceManager = Depends(get_service_manager)
    ):
        """将文本转换为语音"""
        try:
            # 设置输出目录，如果管理器没有设置的话
            if not service_manager.output_dir:
                temp_dir = os.path.join(tempfile.gettempdir(), "illufly_tts_output")
                os.makedirs(temp_dir, exist_ok=True)
                service_manager.output_dir = temp_dir
                logger.info(f"为ServiceManager设置临时输出目录: {temp_dir}")
            
            # 提交任务
            logger.info(f"提交TTS任务: 文本长度={len(request.text)}, 语音={request.voice}")
            voice_id = request.voice or "zf_001"
            task_id = await service_manager.submit_task(
                text=request.text,
                voice_id=voice_id,
                user_id=user.get("user_id")
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
                raise HTTPException(status_code=400, detail=error_message)
            
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
                raise HTTPException(status_code=500, detail="音频文件未生成")
            
            # 读取文件并转换为base64
            with open(output_file_path, "rb") as f:
                file_bytes = f.read()
                audio_base64 = base64.b64encode(file_bytes).decode("utf-8")
            
            # 返回结果
            result = {
                "task_id": task_id,
                "status": "success",
                "audio_base64": audio_base64,
                "sample_rate": 24000,
                "created_at": status["created_at"],
                "completed_at": status["completed_at"]
            }
            
            # 清理文件（可选）
            # try:
            #     os.remove(output_file_path)
            #     logger.debug(f"已清理音频文件: {output_file_path}")
            # except Exception as e:
            #     logger.warning(f"清理文件失败: {e}")
            
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"TTS处理失败: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/tts/batch")
    async def batch_text_to_speech(
        request: BatchTextToSpeechRequest,
        user: UserDict = Depends(require_user),
        service_manager: TTSServiceManager = Depends(get_service_manager),
        pipeline: CachedTTSPipeline = Depends(get_pipeline)
    ):
        """批量将文本转换为语音"""
        try:
            results = []
            for i, text in enumerate(request.texts):
                # 提交任务跟踪状态
                voice_id = request.voice or "zf_001"
                task_id = await service_manager.submit_task(
                    text=text,
                    voice_id=voice_id,
                    user_id=user.get("user_id")
                )
                
                # 等待任务完成
                while True:
                    status = await service_manager.get_task_status(task_id)
                    if status["status"] in ["completed", "failed", "canceled"]:
                        break
                    await asyncio.sleep(0.1)
                
                # 处理结果
                if status["status"] == "completed":
                    # 使用临时文件和pipeline直接生成
                    temp_file_path = os.path.join(tempfile.gettempdir(), f"batch_{task_id}.wav")
                    
                    try:
                        # 使用Pipeline直接生成音频
                        await asyncio.to_thread(
                            pipeline.process,
                            text=text, 
                            voice_id=voice_id,
                            output_path=temp_file_path
                        )
                        
                        # 读取文件并转换为base64
                        with open(temp_file_path, "rb") as f:
                            file_bytes = f.read()
                            audio_base64 = base64.b64encode(file_bytes).decode("utf-8")
                        
                        # 添加到结果
                        results.append({
                            "status": "success",
                            "task_id": task_id,
                            "audio_base64": audio_base64,
                            "sample_rate": 24000,
                            "created_at": status["created_at"],
                            "completed_at": status["completed_at"]
                        })
                    except Exception as e:
                        logger.error(f"生成第{i+1}个音频失败: {e}")
                        results.append({
                            "status": "error",
                            "task_id": task_id,
                            "error": f"生成音频失败: {str(e)}"
                        })
                    finally:
                        # 清理临时文件
                        try:
                            if os.path.exists(temp_file_path):
                                os.remove(temp_file_path)
                        except:
                            pass
                else:
                    # 失败或取消
                    results.append({
                        "status": "error",
                        "task_id": task_id,
                        "error": status.get("error", f"任务{status['status']}")
                    })
            
            return {"results": results}
        except Exception as e:
            logger.error(f"批量TTS处理失败: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/tts/voices")
    async def get_voices(
        user: UserDict = Depends(require_user)
    ):
        """获取可用语音列表"""
        # 目前只有一个语音
        voices = [
            {"id": "zf_001", "name": "普通话女声", "description": "标准普通话女声"}
        ]
        return {"voices": voices}
    
    @router.get("/tts/info")
    async def get_service_info(
        user: UserDict = Depends(require_user),
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
    
    # 应用关闭时关闭服务
    @app.on_event("shutdown")
    async def shutdown_service_manager():
        if hasattr(app.state, "service_manager"):
            logger.info("关闭TTS服务...")
            await app.state.service_manager.shutdown()
