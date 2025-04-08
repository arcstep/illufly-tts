"""
基于MCP规范的文本转语音客户端
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from mcp import ClientSession, StdioServerParameters 
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# 设置更详细的日志输出
DEBUG = os.environ.get("MCP_DEBUG", "").lower() in ("1", "true", "yes")
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)


class TTSMcpClient:
    """基于MCP的文本转语音客户端"""
    
    def __init__(
        self,
        process_command: Optional[str] = None,
        process_args: Optional[List[str]] = None,
        host: str = "localhost",
        port: int = 8000,
        use_stdio: bool = True
    ):
        """初始化MCP客户端
        
        Args:
            process_command: 子进程命令（用于stdio传输）
            process_args: 子进程参数（用于stdio传输）
            host: 服务器主机地址（用于SSE传输）
            port: 服务器端口（用于SSE传输）
            use_stdio: 是否使用stdio传输
        """
        self._session = None
        self._exit_stack = None
        
        # 保存传输相关参数
        self._host = host
        self._port = port
        self._process_command = process_command
        self._process_args = process_args or []
        self._use_stdio = use_stdio
        
        # 记录初始化信息
        logger.info(f"初始化TTS MCP客户端: use_stdio={use_stdio}")
        if use_stdio:
            logger.info(f"准备使用子进程: {process_command} {' '.join(process_args)}")
        else:
            logger.info(f"准备连接到服务器: {host}:{port}")
    
    async def _ensure_connected(self):
        """确保连接到MCP服务器"""
        if self._session is None:
            from contextlib import AsyncExitStack
            
            logger.debug("创建MCP连接...")
            self._exit_stack = AsyncExitStack()
            
            # 创建传输层
            if self._use_stdio:
                if not self._process_command:
                    raise ValueError("使用stdio传输时必须提供process_command")
                
                logger.debug(f"创建stdio连接: {self._process_command} {' '.join(self._process_args)}")
                server_params = StdioServerParameters(
                    command=self._process_command,
                    args=self._process_args
                )
                
                try:
                    stdio_transport = await self._exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    read, write = stdio_transport
                    
                    logger.debug("创建ClientSession...")
                    self._session = await self._exit_stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    logger.debug("初始化ClientSession...")
                    await self._session.initialize()
                    logger.info("MCP客户端连接成功")
                except Exception as e:
                    logger.error(f"连接MCP服务器失败: {str(e)}")
                    if self._exit_stack:
                        await self._exit_stack.aclose()
                        self._exit_stack = None
                    raise
            else:
                # SSE传输暂不实现
                logger.error("SSE传输暂不支持")
                raise NotImplementedError("SSE传输暂不支持")
    
    async def _call_tool_safe(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """安全地调用MCP工具并处理结果
        
        Args:
            tool_name: 工具名称
            params: 参数
            
        Returns:
            工具返回的结果（已解析为Python对象）
            
        Raises:
            ValueError: 如果调用失败
        """
        await self._ensure_connected()
        
        try:
            logger.debug(f"调用工具: {tool_name}, 参数: {json.dumps(params)}")
            result = await self._session.call_tool(tool_name, params)
            
            # 记录结果类型信息，便于调试
            logger.debug(f"工具返回类型: {type(result)}")
            
            # 获取结果中的实际内容
            if hasattr(result, 'content') and result.content and hasattr(result.content[0], 'text'):
                json_str = result.content[0].text
                logger.debug(f"提取到结果内容（前100字符）: {json_str[:100]}...")
            else:
                # 如果不是预期的结构，尝试直接使用result
                logger.debug(f"无法从结果中提取内容，尝试直接使用结果: {result!r}")
                json_str = result
            
            # 检查是否为错误消息（非JSON格式）
            if json_str and isinstance(json_str, str) and json_str.startswith('Error executing tool'):
                logger.warning(f"服务器返回错误消息: {json_str}")
                # 提取错误消息，通常格式为"Error executing tool xxx: 具体错误信息"
                error_parts = json_str.split(':', 1)
                error_msg = error_parts[1].strip() if len(error_parts) > 1 else json_str
                raise ValueError(error_msg)
            
            # 解析JSON结果
            try:
                parsed_result = json.loads(json_str)
                return parsed_result
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"解析结果失败: {str(e)}, 原始结果: {json_str!r}")
                # 如果解析失败，把原始消息当作错误信息返回
                if isinstance(json_str, str):
                    raise ValueError(json_str)
                else:
                    raise ValueError(f"无法解析工具返回的结果: {str(e)}")
        except Exception as e:
            logger.error(f"调用工具失败: {tool_name}, 错误: {str(e)}", exc_info=DEBUG)
            raise ValueError(f"调用工具失败: {str(e)}")
    
    async def text_to_speech(self, text: str, voice: Optional[str] = None) -> Dict[str, Any]:
        """将文本转换为语音
        
        Args:
            text: 要转换的文本
            voice: 语音ID，默认使用服务器配置的语音
            
        Returns:
            语音数据（包含base64编码的音频）
            
        Raises:
            ValueError: 如果转换失败
        """
        params = {"text": text}
        if voice:
            params["voice"] = voice
            
        return await self._call_tool_safe("text_to_speech", params)
    
    async def batch_text_to_speech(self, texts: List[str], voice: Optional[str] = None) -> List[Dict[str, Any]]:
        """批量将文本转换为语音
        
        Args:
            texts: 要转换的文本列表
            voice: 语音ID，默认使用服务器配置的语音
            
        Returns:
            语音数据列表
            
        Raises:
            ValueError: 如果转换失败
        """
        params = {"texts": texts}
        if voice:
            params["voice"] = voice
            
        return await self._call_tool_safe("batch_text_to_speech", params)
    
    async def save_speech_to_file(self, text: str, output_path: str, voice: Optional[str] = None) -> Dict[str, Any]:
        """将文本转换为语音并保存到文件
        
        Args:
            text: 要转换的文本
            output_path: 输出文件路径
            voice: 语音ID，默认使用服务器配置的语音
            
        Returns:
            处理结果
            
        Raises:
            ValueError: 如果转换或保存失败
        """
        params = {
            "text": text,
            "output_path": output_path
        }
        if voice:
            params["voice"] = voice
            
        return await self._call_tool_safe("save_speech_to_file", params)
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """获取可用的语音列表
        
        Returns:
            语音列表
            
        Raises:
            ValueError: 如果获取失败
        """
        return await self._call_tool_safe("get_available_voices", {})
    
    async def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息
        
        Returns:
            服务信息
            
        Raises:
            ValueError: 如果获取失败
        """
        return await self._call_tool_safe("get_service_info", {})
    
    async def close(self):
        """关闭客户端连接"""
        if self._session and self._exit_stack:
            logger.info("关闭MCP客户端连接")
            try:
                await self._exit_stack.aclose()
                logger.debug("MCP客户端连接已关闭")
            except Exception as e:
                logger.error(f"关闭连接时出错: {str(e)}")
            finally:
                self._session = None
                self._exit_stack = None


# 同步包装器，便于非异步代码调用
class SyncTTSMcpClient:
    """同步TTS MCP客户端"""
    
    def __init__(
        self,
        process_command: Optional[str] = None,
        process_args: Optional[List[str]] = None,
        host: str = "localhost",
        port: int = 8000,
        use_stdio: bool = True
    ):
        """初始化同步TTS MCP客户端
        
        Args:
            process_command: 子进程命令（用于stdio传输）
            process_args: 子进程参数（用于stdio传输）
            host: 服务器主机地址（用于SSE传输）
            port: 服务器端口（用于SSE传输）
            use_stdio: 是否使用stdio传输
        """
        self._client = TTSMcpClient(
            process_command=process_command,
            process_args=process_args,
            host=host,
            port=port,
            use_stdio=use_stdio
        )
        self._loop = asyncio.get_event_loop()
    
    def _run_async(self, coro):
        """运行异步方法并返回结果"""
        return self._loop.run_until_complete(coro)
    
    def text_to_speech(self, text: str, voice: Optional[str] = None) -> Dict[str, Any]:
        """将文本转换为语音（同步版本）"""
        return self._run_async(self._client.text_to_speech(text, voice))
    
    def batch_text_to_speech(self, texts: List[str], voice: Optional[str] = None) -> List[Dict[str, Any]]:
        """批量将文本转换为语音（同步版本）"""
        return self._run_async(self._client.batch_text_to_speech(texts, voice))
    
    def save_speech_to_file(self, text: str, output_path: str, voice: Optional[str] = None) -> Dict[str, Any]:
        """将文本转换为语音并保存到文件（同步版本）"""
        return self._run_async(self._client.save_speech_to_file(text, output_path, voice))
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """获取可用的语音列表（同步版本）"""
        return self._run_async(self._client.get_available_voices())
    
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息（同步版本）"""
        return self._run_async(self._client.get_service_info())
    
    def close(self):
        """关闭客户端"""
        self._run_async(self._client.close())
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 