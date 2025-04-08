"""
文本转语音服务的FastAPI接口
"""
import os
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .mcp_client import TTSMcpClient

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

def setup_tts_service(
    app: FastAPI,
    require_user: Callable[[], Awaitable[UserDict]],
    host: str = "localhost",
    port: int = 8000,
    prefix: str = "/api"
) -> TTSMcpClient:
    """设置TTS服务 - 连接到现有的MCP服务器 (SSE方式)
    
    Args:
        app: FastAPI应用
        require_user: 获取当前用户的函数
        host: MCP服务器主机
        port: MCP服务器端口
        prefix: API前缀
        
    Returns:
        TTSMcpClient: MCP客户端
    """
    # 创建MCP客户端
    client = TTSMcpClient(
        host=host,
        port=port,
        use_stdio=False
    )
    
    router = APIRouter()
    
    @router.post("/tts")
    async def text_to_speech(
        request: TextToSpeechRequest,
        user: UserDict = Depends(require_user)
    ):
        """将文本转换为语音"""
        try:
            result = await client.text_to_speech(request.text, request.voice)
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=result.get("error", "转换失败"))
            return result
        except Exception as e:
            logger.error(f"TTS调用失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/tts/batch")
    async def batch_text_to_speech(
        request: BatchTextToSpeechRequest,
        user: UserDict = Depends(require_user)
    ):
        """批量将文本转换为语音"""
        try:
            results = await client.batch_text_to_speech(request.texts, request.voice)
            return {"results": results}
        except Exception as e:
            logger.error(f"批量TTS调用失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/tts/voices")
    async def get_voices(user: UserDict = Depends(require_user)):
        """获取可用语音列表"""
        try:
            voices = await client.get_available_voices()
            return {"voices": voices}
        except Exception as e:
            logger.error(f"获取语音列表失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/tts/info")
    async def get_service_info(user: UserDict = Depends(require_user)):
        """获取服务信息"""
        try:
            info = await client.get_service_info()
            return info
        except Exception as e:
            logger.error(f"获取服务信息失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # 注册路由
    app.include_router(router, prefix=prefix)
    
    return client


def mount_tts_service_stdio(
    app: FastAPI,
    require_user: Callable[[], Awaitable[UserDict]],
    process_command: str,
    process_args: List[str] = None,
    prefix: str = "/api"
) -> TTSMcpClient:
    """
    挂载TTS服务到FastAPI应用 - 使用子进程STDIO方式
    
    通过子进程和stdio方式与MCP服务通信，推荐在主应用中使用此方法
    
    Args:
        app: FastAPI应用
        require_user: 获取当前用户的函数
        process_command: 子进程命令 (如 python 或 /usr/bin/python)
        process_args: 子进程参数列表 (如 ["-m", "illufly_tts", "--transport", "stdio"])
        prefix: API前缀
        
    Returns:
        TTSMcpClient: MCP客户端
    """
    # 创建MCP客户端 - 使用子进程方式
    client = TTSMcpClient(
        process_command=process_command,
        process_args=process_args or [],
        use_stdio=True
    )
    
    # 全局共享客户端实例（单例模式）
    app.state.tts_client = client
    
    router = APIRouter()
    
    # 用于获取客户端的依赖
    async def get_tts_client():
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
            if result.get("status") == "error":
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


def mount_tts_service(
    app: FastAPI,
    require_user: Callable[[], Awaitable[UserDict]],
    host: str = "localhost",
    port: int = 8000,
    prefix: str = "/api",
) -> TTSMcpClient:
    """
    挂载TTS服务到FastAPI应用 - 使用SSE方式连接
    
    此方法保留用于向后兼容，推荐使用 mount_tts_service_stdio 代替
    """
    # 直接使用setup_tts_service连接到MCP服务器
    return setup_tts_service(
        app=app,
        require_user=require_user,
        host=host,
        port=port,
        prefix=prefix
    )
