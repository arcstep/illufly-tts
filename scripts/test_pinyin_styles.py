#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
拼音风格测试脚本
测试不同的拼音风格对中文语音合成的影响
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from scipy.io.wavfile import write

# 添加项目根目录到导入路径
script_dir = Path(os.path.abspath(__file__)).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# 在导入任何其他库之前应用G2P补丁
from src.illufly_tts.g2p_patch import apply_g2p_patches
print("正在应用G2P补丁...")
apply_g2p_patches()

# 导入相关模块
from src.illufly_tts import CustomPipeline
from src.illufly_tts.chinese_g2p import ChineseG2P, STYLE_MAP

# 测试文本列表
TEST_TEXTS = [
    "这是一个中文测试文本，看看是否能正常生成语音。",
    "我叫张三，很高兴认识你。今天天气真好，我们一起去公园吧。",
    "中文文本转语音系统现在能够处理各种中文文本了。",
    "重要的事情说三遍：测试，测试，测试！",
    "行业内的行人都在银行行长的办公室里。", # 包含多音字"行"
    "你会弹钢琴吗？我小时候学过一段时间，但都忘记了。", # 复杂语句
]

async def initialize_tts_service(args):
    """初始化TTS服务"""
    try:
        from kokoro import KModel
        
        # A设置离线模式环境变量
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
        return pipeline, model
        
    except ImportError as e:
        logger.error(f"导入Kokoro库失败: {e}")
        logger.error("请确保已安装Kokoro库")
        sys.exit(1)
    except Exception as e:
        logger.error(f"初始化TTS服务失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

async def generate_speech_with_style(pipeline, text, output_base, voice_id, style=None):
    """使用特定拼音风格生成语音"""
    try:
        # 创建自定义ChineseG2P处理器
        chinese_g2p = ChineseG2P(style=style)
        
        # 临时替换pipeline的zh_callable
        original_zh_callable = pipeline.zh_callable
        pipeline.zh_callable = chinese_g2p
        
        # 生成output_path
        style_suffix = f"_{style.lower()}" if style else ""
        output_path = f"{output_base}{style_suffix}.wav"
        
        logger.info(f"生成使用{style or '默认'}风格的语音: {output_path}")
        
        # 生成语音
        results = list(pipeline(
            text=text,
            voice=voice_id,
            speed=1.0
        ))
        
        # 处理生成结果
        output_audio = None
        for i, result in enumerate(results):
            if result.audio is not None:
                logger.info(f"生成语音段落 {i+1}")
                
                if output_audio is None:
                    output_audio = result.audio
                else:
                    # 添加短暂的停顿
                    pause = torch.zeros(int(24000 * 0.5))  # 0.5秒的停顿
                    output_audio = torch.cat([output_audio, pause, result.audio])
        
        # 保存语音文件
        if output_audio is not None:
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 规范化音频
            output_audio = output_audio / torch.max(torch.abs(output_audio))
            output_audio = output_audio.cpu().numpy()
            
            # 保存为WAV文件
            write(output_path, 24000, output_audio)
            logger.info(f"语音已保存至: {output_path}")
            
            # 保存拼音结果
            pinyin_path = f"{output_base}{style_suffix}.pinyin.txt"
            with open(pinyin_path, "w", encoding="utf-8") as f:
                f.write(f"原文: {text}\n")
                f.write(f"拼音风格: {style or '默认'}\n")
                phonemes = [r.phonemes for r in results if r.phonemes]
                f.write(f"拼音: {' '.join(phonemes)}")
            logger.info(f"拼音已保存至: {pinyin_path}")
        else:
            logger.error("生成语音失败")
        
        # 恢复原始zh_callable
        pipeline.zh_callable = original_zh_callable
        
    except Exception as e:
        logger.error(f"生成语音过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

async def test_all_styles(pipeline, args):
    """测试所有拼音风格"""
    # 选择要测试的文本
    text_index = min(max(0, args.text_index), len(TEST_TEXTS) - 1)
    text = args.text or TEST_TEXTS[text_index]
    
    logger.info(f"测试文本: {text}")
    
    # 输出基本路径
    output_base = args.output or f"./output/pinyin_test_{text_index}"
    
    # 保存原始文本
    with open(f"{output_base}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    # 测试每种拼音风格
    styles = list(STYLE_MAP.keys())
    if args.style:
        if args.style.upper() in styles:
            styles = [args.style.upper()]
        else:
            logger.warning(f"未知风格: {args.style}，将测试所有风格")
    
    logger.info(f"将测试以下拼音风格: {', '.join(styles)}")
    
    for style in styles:
        await generate_speech_with_style(
            pipeline=pipeline,
            text=text,
            output_base=output_base,
            voice_id=args.voice_id,
            style=style
        )
    
    logger.info(f"所有拼音风格测试完成，输出文件前缀: {output_base}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='拼音风格测试工具')
    parser.add_argument('--text', type=str, help='要测试的文本，不指定则使用内置测试文本')
    parser.add_argument('--text_index', type=int, default=0, help='要使用的内置测试文本索引 (0-5)')
    parser.add_argument('--output', type=str, help='输出文件基本路径，会添加风格后缀')
    parser.add_argument('--voice_id', type=str, default='zf_001', help='语音ID')
    parser.add_argument('--model_path', type=str, help='模型路径，默认使用huggingface上的模型')
    parser.add_argument('--device', type=str, default='auto', help='设备(cpu, cuda, auto)')
    parser.add_argument('--style', type=str, help='仅测试指定的拼音风格')
    parser.add_argument('--offline', action='store_true', help='启用离线模式')
    
    return parser.parse_args()

async def main():
    """主函数"""
    args = parse_args()
    
    # 初始化TTS服务
    pipeline, model = await initialize_tts_service(args)
    
    # 测试所有拼音风格
    await test_all_styles(pipeline, args)
    
    # 输出风格说明
    print("\n拼音风格说明:")
    print("  NORMAL: 不带声调的拼音，如 zhong wen")
    print("  TONE: 声调符号在韵母上的拼音，如 zhōng wén")
    print("  TONE2: 声调数字在韵母后的拼音，如 zho1ng we2n")
    print("  TONE3: 声调数字在末尾的拼音，如 zhong1 wen2")
    print("  INITIALS: 声母，如 zh w")
    print("  FIRST_LETTER: 首字母，如 z w")

if __name__ == "__main__":
    asyncio.run(main()) 