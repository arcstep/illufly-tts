#!/usr/bin/env python
"""
TTS API服务命令行入口 - 轻量级REST API服务
"""
import click
import logging
import uvicorn
from fastapi import FastAPI
from typing import Optional, List

logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", default="0.0.0.0", help="API服务地址")
@click.option("--port", default=8001, help="API服务端口")
@click.option("--server-host", default="localhost", help="MCP服务器地址")
@click.option("--server-port", default=31572, help="MCP服务器端口")
@click.option("--use-stdio", is_flag=True, help="使用子进程方式连接")
@click.option("--process-command", help="子进程命令 (use-stdio时必须)")
@click.option("--process-args", help="子进程参数 (逗号分隔)")
@click.option("--prefix", default="/api", help="API前缀")
def main(
    host: str, 
    port: int, 
    server_host: str, 
    server_port: int,
    use_stdio: bool,
    process_command: Optional[str],
    process_args: Optional[str],
    prefix: str
):
    """启动TTS API服务"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # 创建FastAPI应用
    app = FastAPI(
        title="Illufly TTS API",
        description="轻量级文本转语音API服务",
        version="0.2.0"
    )
    
    # 处理参数
    process_args_list = None
    if process_args:
        process_args_list = process_args.split(",")
    
    # 简单的用户认证依赖 - 实际项目中请替换为真正的认证逻辑
    async def get_current_user():
        return {"user_id": "api_user"}
    
    # 导入并挂载TTS服务
    from illufly_tts.api.endpoints import mount_tts_service
    
    # 挂载服务
    mount_tts_service(
        app=app,
        require_user=get_current_user,
        host=server_host,
        port=server_port,
        prefix=prefix,
        use_stdio=use_stdio,
        process_command=process_command,
        process_args=process_args_list
    )
    
    # 启动服务
    logger.info(f"启动TTS API服务: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main() 