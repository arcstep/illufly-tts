"""
文本转语音服务的FastAPI接口 - 轻量级版本
"""
import os
import logging
import asyncio
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel

# 使用相对导入，客户端可以选择自己复制mcp_client.py
from ..client.mcp_client import TTSMcpClient

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
    host: str = "localhost",
    port: int = 8000,
    prefix: str = "/api",
    use_stdio: bool = True,
    process_command: Optional[str] = None,
    process_args: Optional[List[str]] = None
) -> TTSMcpClient:
    """挂载TTS服务到FastAPI应用
    
    支持通过网络或子进程方式连接到MCP服务器
    
    Args:
        app: FastAPI应用
        require_user: 获取当前用户的函数
        host: MCP服务器主机 (当use_stdio=False时使用)
        port: MCP服务器端口 (当use_stdio=False时使用)
        prefix: API前缀
        use_stdio: 是否使用stdio传输 (子进程方式)
        process_command: 子进程命令 (当use_stdio=True时使用)
        process_args: 子进程参数 (当use_stdio=True时使用)
        
    Returns:
        TTSMcpClient: MCP客户端
    """
    # 创建MCP客户端
    if use_stdio:
        if not process_command:
            raise ValueError("使用stdio传输时必须提供process_command参数")
        
        client = TTSMcpClient(
            process_command=process_command,
            process_args=process_args or [],
            use_stdio=True
        )
        logger.info(f"使用stdio传输连接到TTS服务: {process_command}")
    else:
        client = TTSMcpClient(
            host=host,
            port=port,
            use_stdio=False
        )
        logger.info(f"使用网络连接到TTS服务: {host}:{port}")
    
    # 全局共享客户端实例（单例模式）
    app.state.tts_client = client
    
    # 创建API路由
    router = APIRouter()
    
    # 用于获取客户端的依赖
    async def get_tts_client():
        if not hasattr(app.state, "tts_client_initialized"):
            # 首次使用时初始化连接
            try:
                await app.state.tts_client._ensure_connected()
                app.state.tts_client_initialized = True
                logger.info("TTS客户端连接初始化成功")
            except Exception as e:
                logger.error(f"TTS客户端连接初始化失败: {e}")
                # 不在这里抛出异常，让实际API调用时处理错误
                pass
        return app.state.tts_client
    
    @router.post("/tts")
    async def text_to_speech(
        request: TextToSpeechRequest,
        user: UserDict = Depends(require_user),
        client: TTSMcpClient = Depends(get_tts_client)
    ):
        """将文本转换为语音"""
        try:
            result = await client.text_to_speech(request.text, request.voice)
            if isinstance(result, dict) and result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error", "转换失败"))
            return result
        except Exception as e:
            logger.error(f"TTS调用失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/tts/batch")
    async def batch_text_to_speech(
        request: BatchTextToSpeechRequest,
        user: UserDict = Depends(require_user),
        client: TTSMcpClient = Depends(get_tts_client)
    ):
        """批量将文本转换为语音"""
        try:
            results = await client.batch_text_to_speech(request.texts, request.voice)
            return {"results": results}
        except Exception as e:
            logger.error(f"批量TTS调用失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/tts/voices")
    async def get_voices(
        user: UserDict = Depends(require_user),
        client: TTSMcpClient = Depends(get_tts_client)
    ):
        """获取可用语音列表"""
        try:
            voices = await client.get_available_voices()
            return {"voices": voices}
        except Exception as e:
            logger.error(f"获取语音列表失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/tts/info")
    async def get_service_info(
        user: UserDict = Depends(require_user),
        client: TTSMcpClient = Depends(get_tts_client)
    ):
        """获取服务信息"""
        try:
            info = await client.get_service_info()
            return info
        except Exception as e:
            logger.error(f"获取服务信息失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # 注册路由
    app.include_router(router, prefix=prefix)
    
    # 在应用关闭时关闭客户端
    @app.on_event("shutdown")
    async def close_tts_client():
        logger.info("关闭MCP客户端连接...")
        await app.state.tts_client.close()
    
    return client
