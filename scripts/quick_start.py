#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速启动示例脚本
演示如何使用illufly-tts进行文本转语音
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
import re
from scipy.io.wavfile import write

# 添加项目根目录到导入路径
script_dir = Path(os.path.abspath(__file__)).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,  # 恢复为INFO级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# 在导入任何其他库之前应用G2P补丁
from src.illufly_tts.g2p_patch import apply_g2p_patches
print("正在应用G2P补丁...")
apply_g2p_patches()

# 导入自定义 Pipeline 替代 KPipeline
from src.illufly_tts import CustomPipeline

def check_dependencies():
    """检查并提示安装依赖"""
    missing_packages = []
    
    try:
        import g2p_en
    except ImportError:
        missing_packages.append("g2p_en")
        
    try:
        import langid
    except ImportError:
        missing_packages.append("langid")
    
    try:
        import jieba
    except ImportError:
        missing_packages.append("jieba")
    
    if missing_packages:
        logger.warning(f"缺少必要的依赖包: {', '.join(missing_packages)}")
        logger.info("您可以运行以下命令安装依赖:")
        logger.info(f"python {project_root}/tools/setup_g2p.py")
        
        response = input("是否现在安装这些依赖? [y/N]: ")
        if response.lower() in ('y', 'yes'):
            try:
                import subprocess
                subprocess.run([sys.executable, f"{project_root}/tools/setup_g2p.py"], check=True)
                logger.info("依赖安装完成!")
            except Exception as e:
                logger.error(f"安装依赖失败: {e}")
                logger.info("请手动安装必要的依赖包")
                sys.exit(1)
        else:
            logger.warning("继续运行，但可能会出现问题")

async def initialize_tts_service(args):
    """初始化TTS服务"""
    try:
        from kokoro import KModel
        
        # 设置离线模式环境变量
        if args.offline:
            os.environ["KOKORO_OFFLINE"] = "1"
            logger.info("启用离线模式")
        
        # 初始化模型
        model_path = args.model_path or "hexgrad/Kokoro-82M-v1.1-zh"
        logger.info(f"初始化TTS服务，模型路径: {model_path}")
        
        # 设置设备
        device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")
        
        # 使用正确的构造函数参数
        model = KModel(repo_id=model_path)
        if device == "cuda":
            model = model.cuda()
        
        # 创建自定义Pipeline
        pipeline = CustomPipeline(
            model=model,
            repo_id=model_path,
            device=device
        )
        
        logger.info("TTS服务初始化完成")
        return pipeline
        
    except ImportError as e:
        logger.error(f"导入Kokoro库失败: {e}")
        logger.error("请确保已安装Kokoro库")
        sys.exit(1)
    except Exception as e:
        logger.error(f"初始化TTS服务失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

async def generate_speech(pipeline, args):
    """生成语音"""
    if not args.text:
        logger.error("未提供文本，无法生成语音")
        return
    
    try:
        voice_id = args.voice_id
        logger.info(f"使用语音ID: {voice_id}")
        
        # 在这里使用自定义Pipeline
        logger.info(f"文本: {args.text}")
        
        # 使用正则表达式判断是否包含中文
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', args.text))
        logger.info(f"语言判断: {'中文' if has_chinese else '英文'}")
        
        # 保存原始输入文本
        with open(f"{args.output}.input.txt", "w", encoding="utf-8") as f:
            f.write(args.text)
            logger.info(f"原始输入文本已保存至: {args.output}.input.txt")
        
        # 生成语音
        logger.info(f"开始生成语音，语速: {args.speed}")
        results = pipeline(
            text=args.text,
            voice=voice_id,
            speed=args.speed
        )
        
        # 记录音素序列
        phonemes_list = []
        
        # 处理生成结果
        output_audio = None
        for i, result in enumerate(results):
            # 收集音素序列
            if result.phonemes:
                phonemes_list.append(result.phonemes)
                logger.info(f"段落 {i+1} 音素序列: '{result.phonemes[:50]}{'...' if len(result.phonemes) > 50 else ''}'")
                
                # 保存音素序列到文件
                with open(f"{args.output}.phonemes.{i+1}.txt", "w", encoding="utf-8") as f:
                    f.write(result.phonemes)
                    logger.info(f"音素序列已保存至: {args.output}.phonemes.{i+1}.txt")
            
            if result.audio is not None:
                logger.info(f"生成语音段落 {i+1}")
                
                # 保存每个段落的音频
                if args.save_segments:
                    segment_path = f"{args.output}.segment.{i+1}.wav"
                    segment_audio = result.audio.cpu().numpy()
                    segment_audio = segment_audio / max(0.01, np.max(np.abs(segment_audio)))
                    write(segment_path, 24000, segment_audio)
                    logger.info(f"语音段落 {i+1} 已保存至: {segment_path}")
                
                if output_audio is None:
                    output_audio = result.audio
                else:
                    # 添加短暂的停顿
                    pause = torch.zeros(int(24000 * 0.5))  # 0.5秒的停顿
                    output_audio = torch.cat([output_audio, pause, result.audio])
        
        # 保存所有音素序列
        if phonemes_list:
            with open(f"{args.output}.phonemes.all.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(phonemes_list))
                logger.info(f"所有音素序列已保存至: {args.output}.phonemes.all.txt")
        
        if output_audio is not None:
            # 保存语音文件
            # 确保输出目录存在
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 规范化音频
            output_audio = output_audio / torch.max(torch.abs(output_audio))
            output_audio = output_audio.cpu().numpy()
            
            # 保存为WAV文件
            write(args.output, 24000, output_audio)
            logger.info(f"语音已保存至: {args.output}")
            
            # 尝试播放音频
            if not args.no_play:
                try:
                    play_audio(args.output)
                except Exception as e:
                    logger.warning(f"播放音频失败: {e}")
        else:
            logger.error("生成语音失败")
            
    except Exception as e:
        logger.error(f"生成语音过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def play_audio(file_path):
    """根据操作系统播放音频"""
    try:
        import platform
        system = platform.system()
        
        import subprocess
        
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", file_path], check=True)
        elif system == "Windows":
            subprocess.run(["start", file_path], shell=True, check=True)
        else:  # Linux
            subprocess.run(["aplay", file_path], check=True)
            
        logger.info("正在播放音频...")
        
    except Exception as e:
        logger.warning(f"播放音频失败: {e}")
        logger.info(f"请手动播放文件: {file_path}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='illufly-tts 快速启动示例')
    parser.add_argument('--text', type=str, default='你好，欢迎使用illufly中文语音合成系统。', help='要转换的文本，默认为中文示例')
    parser.add_argument('--output', type=str, default='./output/tts_output.wav', help='输出音频文件的路径')
    parser.add_argument('--voice_id', type=str, default='zf_001', help='语音ID')
    parser.add_argument('--model_path', type=str, help='模型路径，默认使用huggingface上的模型')
    parser.add_argument('--device', type=str, default='auto', help='设备(cpu, cuda, auto)')
    parser.add_argument('--speed', type=float, default=1.0, help='语速，1.0为正常速度')
    parser.add_argument('--offline', action='store_true', help='启用离线模式')
    parser.add_argument('--save_segments', action='store_true', help='保存各段落音频')
    parser.add_argument('--no_play', action='store_true', help='不自动播放音频')
    parser.add_argument('--debug', action='store_true', help='启用详细调试日志')
    
    return parser.parse_args()

async def main():
    """主函数"""
    import torch
    
    args = parse_args()
    
    # 如果指定了debug参数，设置更详细的日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('src.illufly_tts').setLevel(logging.DEBUG)
        logger.info("启用详细调试日志")
    
    # 检查依赖
    check_dependencies()
    
    # 初始化TTS服务
    pipeline = await initialize_tts_service(args)
    
    # 如果没有提供文本，提示用户输入
    if not args.text:
        args.text = input("请输入要转换为语音的文本（推荐中文）: ")
    
    # 生成语音
    await generate_speech(pipeline, args)

if __name__ == "__main__":
    asyncio.run(main()) 