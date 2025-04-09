#!/usr/bin/env python
"""
TTS服务的主入口点 - 支持多种运行模式
"""
import sys
import logging
import click
from typing import Optional
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("illufly_tts")

@click.group()
def cli():
    """Illufly TTS 命令行工具 - 支持多种运行模式
    
    用法示例:
      完整服务:    python -m illufly_tts serve
      仅服务端:    python -m illufly_tts server
      API服务:     python -m illufly_tts api
      客户端:      python -m illufly_tts client
    """
    pass

@cli.command()
@click.option("--host", default="0.0.0.0", help="服务监听地址")
@click.option("--port", default=8000, help="服务监听端口")
@click.option("--repo-id", default="hexgrad/Kokoro-82M-v1.1-zh", help="模型仓库ID")
@click.option("--voices-dir", default="voices", help="语音目录路径")
@click.option("--device", default=None, help="使用的设备 (cpu或cuda)")
@click.option("--batch-size", default=4, help="批处理大小")
@click.option("--max-wait-time", default=0.2, help="最大等待时间")
@click.option("--chunk-size", default=200, help="文本分块大小")
@click.option("--output-dir", default=None, help="音频输出目录")
def serve(host, port, repo_id, voices_dir, device, batch_size, max_wait_time, chunk_size, output_dir):
    """启动完整的TTS服务 (API + MCP子进程)"""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    # 创建FastAPI应用
    app = FastAPI(
        title="Illufly TTS服务",
        description="高质量中文语音合成服务",
        version="0.2.0"
    )
    
    # 添加CORS支持
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 简单的用户鉴权函数
    async def get_current_user():
        """开发环境下的简单用户认证，总是返回测试用户"""
        return {"user_id": "test_user", "username": "测试用户"}
    
    # 根路径处理
    @app.get("/")
    async def root():
        """根路径响应"""
        return {
            "service": "Illufly TTS服务",
            "version": "0.2.0",
            "status": "运行中",
            "docs": f"http://{host}:{port}/docs"
        }
    
    # 准备MCP子进程启动命令
    process_command = sys.executable
    process_args = [
        "-m", "illufly_tts.server",  # 使用新的服务端模块
        "--repo-id", repo_id,
        "--voices-dir", voices_dir,
        f"--batch-size={batch_size}",
        f"--max-wait-time={max_wait_time}",
        f"--chunk-size={chunk_size}",
        "--transport", "stdio"
    ]
    
    if device:
        process_args.append(f"--device={device}")
    
    if output_dir:
        process_args.append(f"--output-dir={output_dir}")
    
    # 挂载TTS服务到FastAPI应用
    from .api.endpoints import mount_tts_service
    
    logger.info(f"启动TTS服务 - MCP子进程: {process_command} {' '.join(process_args)}")
    mount_tts_service(
        app=app,
        require_user=get_current_user,
        use_stdio=True,
        process_command=process_command,
        process_args=process_args,
        prefix="/api"
    )
    
    # 启动uvicorn服务
    logger.info(f"启动FastAPI服务 - 监听: {host}:{port}")
    uvicorn.run(app, host=host, port=port)

@cli.command()
def server():
    """启动仅TTS服务器组件"""
    from .server.__main__ import main as server_main
    sys.argv[0] = "illufly_tts_server"
    server_main()

@cli.command()
def api():
    """启动仅API服务组件"""
    from .api.__main__ import main as api_main
    sys.argv[0] = "illufly_tts_api"
    api_main()

@cli.command()
def client():
    """启动客户端命令行工具"""
    from .client.__main__ import cli as client_cli
    sys.argv[0] = "illufly_tts_client"
    client_cli(obj={})

# 默认没有子命令时，使用完整服务
def main():
    if len(sys.argv) == 1:
        # 如果没有参数，默认使用serve模式
        sys.argv.append("serve")
    cli()

if __name__ == "__main__":
    main()
