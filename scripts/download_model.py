#!/usr/bin/env python
"""
下载TTS模型并保存到本地
"""
import os
import sys
import argparse
import time
from pathlib import Path
import logging
import json
from huggingface_hub import hf_hub_download

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("download_model")

def main():
    parser = argparse.ArgumentParser(description="下载TTS模型并保存到本地")
    parser.add_argument(
        "--repo-id", 
        default="hexgrad/Kokoro-82M-v1.1-zh", 
        help="模型仓库ID"
    )
    parser.add_argument(
        "--output-dir", 
        default="./models", 
        help="输出目录"
    )
    parser.add_argument(
        "--device", 
        default=None, 
        help="设备 (cpu, cuda)"
    )
    args = parser.parse_args()
    
    # 开始时间
    start_time = time.time()
    
    # 设置环境变量以显示进度条
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / args.repo_id.split("/")[-1]
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 设置进度条显示
        print(f"准备下载模型: {args.repo_id}")
        print(f"模型将保存到: {model_dir}")
        print("下载可能需要几分钟时间，请耐心等待...")
        
        # 尝试导入kokoro
        try:
            from kokoro import KModel
        except ImportError:
            print("错误: 请先安装Kokoro包")
            print("pip install kokoro")
            return 1
        
        # 导入torch
        try:
            import torch
            device = args.device
            if not device:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"使用设备: {device}")
        except ImportError:
            print("错误: 请先安装PyTorch")
            print("pip install torch")
            return 1
        
        # 首先下载配置文件
        print("正在下载配置文件...")
        config_path = hf_hub_download(repo_id=args.repo_id, filename='config.json')
        print(f"配置文件下载完成: {config_path}")
        
        # 读取配置文件
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 复制配置文件到输出目录
        local_config_path = model_dir / 'config.json'
        with open(local_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"配置文件已保存到: {local_config_path}")
        
        # 下载模型文件
        print("正在下载模型文件...")
        model_filename = 'kokoro-v1_1-zh.pth'  # 使用正确的文件名
        model_file_path = hf_hub_download(repo_id=args.repo_id, filename=model_filename)
        print(f"模型文件下载完成: {model_file_path}")
        
        # 复制模型文件到输出目录
        local_model_path = model_dir / model_filename
        with open(model_file_path, 'rb') as f_src, open(local_model_path, 'wb') as f_dst:
            f_dst.write(f_src.read())
        print(f"模型文件已保存到: {local_model_path}")
        
        # 下载默认语音文件
        try:
            print("正在下载默认语音文件...")
            voice_filename = 'voices/zf_001.pt'
            voice_file_path = hf_hub_download(repo_id=args.repo_id, filename=voice_filename)
            print(f"语音文件下载完成: {voice_file_path}")
            
            # 创建语音目录
            voices_dir = model_dir / 'voices'
            voices_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制语音文件
            local_voice_path = voices_dir / 'zf_001.pt'
            with open(voice_file_path, 'rb') as f_src, open(local_voice_path, 'wb') as f_dst:
                f_dst.write(f_src.read())
            print(f"语音文件已保存到: {local_voice_path}")
        except Exception as e:
            print(f"警告: 下载语音文件时出错，但这不影响模型使用: {str(e)}")
        
        # 计算耗时
        elapsed = time.time() - start_time
        print(f"模型下载并保存完成! 耗时: {elapsed:.2f}秒")
        print(f"您可以使用以下命令启动服务:")
        print(f"python -m illufly_tts --model-path {model_dir} --transport fastapi")
        
        return 0
    
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 