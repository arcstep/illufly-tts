#!/usr/bin/env python
"""
TTS服务器的命令行启动入口
"""
import asyncio
import logging
import click
from typing import Optional

logger = logging.getLogger(__name__)

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
    """启动TTS服务器 (仅后端服务，不包含FastAPI)"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    from .mcp_server import create_mcp_server
    
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