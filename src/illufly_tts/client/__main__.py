#!/usr/bin/env python
"""
TTS客户端命令行工具
"""
import click
import json
import logging
import os
import sys
from typing import Optional, List

logger = logging.getLogger(__name__)

@click.group()
@click.option("--debug", is_flag=True, help="启用调试输出")
@click.option("--host", default="localhost", help="服务器地址")
@click.option("--port", default=31572, help="服务器端口")
@click.option("--process-command", help="服务器子进程命令")
@click.option("--process-args", help="服务器子进程参数，逗号分隔")
@click.option("--use-stdio", is_flag=True, default=True, help="是否使用stdio传输")
@click.pass_context
def cli(ctx, debug, host, port, process_command, process_args, use_stdio):
    """TTS客户端命令行工具
    
    用于与TTS服务器通信的轻量级命令行工具
    """
    # 配置日志级别
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # 处理进程参数
    if process_args:
        process_args = process_args.split(",")
    else:
        process_args = []
    
    # 将配置保存到上下文中
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["host"] = host
    ctx.obj["port"] = port
    ctx.obj["process_command"] = process_command
    ctx.obj["process_args"] = process_args
    ctx.obj["use_stdio"] = use_stdio

@cli.command()
@click.option("--voice", help="语音ID")
@click.option("--output", "-o", help="输出音频文件路径")
@click.argument("text")
@click.pass_context
def speak(ctx, voice, output, text):
    """将文本转换为语音
    
    如果指定了output，将音频保存到文件，否则返回JSON格式的结果
    """
    from .mcp_client import SyncTTSMcpClient
    import base64
    
    # 创建客户端
    client = SyncTTSMcpClient(
        process_command=ctx.obj["process_command"],
        process_args=ctx.obj["process_args"],
        host=ctx.obj["host"],
        port=ctx.obj["port"],
        use_stdio=ctx.obj["use_stdio"]
    )
    
    try:
        # 调用文本转语音
        result = client.text_to_speech(text, voice)
        
        # 如果指定了输出文件，保存音频
        if output:
            # 解码base64
            audio_base64 = result.get("audio_base64", "")
            if not audio_base64:
                print(f"错误：未收到音频数据: {result}", file=sys.stderr)
                sys.exit(1)
            
            audio_data = base64.b64decode(audio_base64)
            
            # 保存到文件
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            with open(output, "wb") as f:
                f.write(audio_data)
                
            print(f"音频已保存到: {output}")
        else:
            # 打印JSON结果
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        client.close()

@cli.command()
@click.option("--voice", help="语音ID")
@click.argument("texts", nargs=-1)
@click.pass_context
def batch(ctx, voice, texts):
    """批量转换多条文本为语音"""
    from .mcp_client import SyncTTSMcpClient
    
    # 文本列表必须非空
    if not texts:
        print("错误：必须提供至少一条文本", file=sys.stderr)
        sys.exit(1)
    
    # 创建客户端
    client = SyncTTSMcpClient(
        process_command=ctx.obj["process_command"],
        process_args=ctx.obj["process_args"],
        host=ctx.obj["host"],
        port=ctx.obj["port"],
        use_stdio=ctx.obj["use_stdio"]
    )
    
    try:
        # 调用批量文本转语音
        results = client.batch_text_to_speech(list(texts), voice)
        
        # 打印结果
        print(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        client.close()

@cli.command()
@click.pass_context
def voices(ctx):
    """列出所有可用的语音"""
    from .mcp_client import SyncTTSMcpClient
    
    # 创建客户端
    client = SyncTTSMcpClient(
        process_command=ctx.obj["process_command"],
        process_args=ctx.obj["process_args"],
        host=ctx.obj["host"],
        port=ctx.obj["port"],
        use_stdio=ctx.obj["use_stdio"]
    )
    
    try:
        # 获取语音列表
        voices = client.get_available_voices()
        
        # 打印结果
        print(json.dumps(voices, ensure_ascii=False, indent=2))
    finally:
        client.close()

@cli.command()
@click.pass_context
def info(ctx):
    """获取服务信息"""
    from .mcp_client import SyncTTSMcpClient
    
    # 创建客户端
    client = SyncTTSMcpClient(
        process_command=ctx.obj["process_command"],
        process_args=ctx.obj["process_args"],
        host=ctx.obj["host"],
        port=ctx.obj["port"],
        use_stdio=ctx.obj["use_stdio"]
    )
    
    try:
        # 获取服务信息
        info = client.get_service_info()
        
        # 打印结果
        print(json.dumps(info, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    cli(obj={}) 