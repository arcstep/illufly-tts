#!/usr/bin/env python
"""
TTS服务的主入口点
"""
import sys
import logging
import click
from typing import Optional
from pathlib import Path
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("illufly_tts")

@click.group()
def cli():
    """Illufly TTS 命令行工具"""
    pass

@cli.command()
@click.option("--host", default="0.0.0.0", help="服务监听地址")
@click.option("--port", default=31572, help="服务监听端口")
@click.option("--repo-id", default="hexgrad/Kokoro-82M-v1.1-zh", help="模型仓库ID")
@click.option("--voices-dir", default=None, help="语音目录路径，不指定则使用模型缓存中的voices目录")
@click.option("--device", default=None, help="使用的设备 (cpu或cuda)")
@click.option("--batch-size", default=4, help="批处理大小")
@click.option("--max-wait-time", default=0.2, help="最大等待时间")
@click.option("--chunk-size", default=200, help="文本分块大小")
@click.option("--output-dir", default=None, help="音频输出目录")
@click.option("--debug-output", help="调试输出目录，不设置则不输出调试文件")
def serve(host, port, repo_id, voices_dir, device, batch_size, max_wait_time, chunk_size, output_dir, debug_output):
    """启动简化版TTS服务 (直接FastAPI接口)"""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    # 设置调试环境变量
    if debug_output:
        debug_output_abs = os.path.abspath(debug_output)
        os.environ["TTS_DEBUG_OUTPUT"] = debug_output_abs
        print(f"已启用调试输出，文件将保存到绝对路径: {debug_output_abs}")
        
        # 创建目录并验证
        try:
            os.makedirs(debug_output_abs, exist_ok=True)
            print(f"调试目录已创建/确认: {debug_output_abs}")
        except Exception as e:
            print(f"警告: 创建调试目录失败: {str(e)}")
    
    # 创建FastAPI应用
    app = FastAPI(
        title="Illufly TTS服务",
        description="高质量中文语音合成服务",
        version="0.3.0"
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
            "version": "0.3.0",
            "status": "运行中",
            "docs": f"http://{host}:{port}/docs"
        }
    
    # 挂载TTS服务到FastAPI应用
    from .api.endpoints import mount_tts_service
    
    # 处理voices_dir参数
    voices_dir_param = None
    if voices_dir:
        voices_dir_param = os.path.abspath(voices_dir)
        logger.info(f"使用指定的voices_dir: {voices_dir_param}")
    else:
        logger.info(f"未指定voices_dir，将使用模型缓存中的voices目录")
    
    # 处理output_dir参数
    output_dir_param = output_dir
    if output_dir:
        output_dir_param = os.path.abspath(output_dir)
    
    logger.info(f"启动TTS服务: repo_id={repo_id}")
    mount_tts_service(
        app=app,
        require_user=get_current_user,
        repo_id=repo_id,
        voices_dir=voices_dir_param,  # 可能为None，这是正常的
        device=device,
        batch_size=batch_size,
        max_wait_time=max_wait_time,
        chunk_size=chunk_size,
        output_dir=output_dir_param,
        prefix="/api"
    )
    
    # 添加测试页面 (可以保留原有的测试页面代码)
    
    # 启动uvicorn服务
    logger.info(f"启动FastAPI服务 - 监听: {host}:{port}")
    uvicorn.run(app, host=host, port=port)

# 默认没有子命令时，使用serve命令
def main():
    if len(sys.argv) == 1:
        # 如果没有参数，默认使用serve模式
        sys.argv.append("serve")
    cli()

if __name__ == "__main__":
    main()
