#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
illufly-tts 命令行入口
"""

import os
import sys
import argparse
import logging
from typing import Optional, List

from .pipeline import TTSPipeline, MixedLanguagePipeline

logger = logging.getLogger(__name__)

def main(args: Optional[List[str]] = None) -> int:
    """
    命令行入口主函数
    
    Args:
        args: 命令行参数，默认使用sys.argv
        
    Returns:
        退出码
    """
    parser = argparse.ArgumentParser(
        description="illufly-tts: 高质量多语言TTS系统"
    )
    
    # 基本参数
    parser.add_argument(
        "-t", "--text", 
        type=str, 
        help="要转换的文本内容"
    )
    parser.add_argument(
        "-f", "--file", 
        type=str, 
        help="包含要转换文本的文件路径"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="output.wav", 
        help="输出音频文件路径"
    )
    
    # 模型和语音资源参数
    parser.add_argument(
        "-m", "--model-path", 
        type=str, 
        required=True, 
        help="Kokoro模型路径"
    )
    parser.add_argument(
        "-v", "--voices-dir", 
        type=str, 
        required=True, 
        help="语音目录路径"
    )
    parser.add_argument(
        "--voice-id", 
        type=str, 
        default="z001", 
        help="语音ID"
    )
    
    # 性能和功能选项
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda"], 
        help="使用的设备"
    )
    parser.add_argument(
        "--speed", 
        type=float, 
        default=1.0, 
        help="语速，默认1.0"
    )
    parser.add_argument(
        "--use-official", 
        action="store_true", 
        help="使用官方Pipeline"
    )
    parser.add_argument(
        "--mixed-language", 
        action="store_true", 
        help="使用混合语言Pipeline"
    )
    
    # 工具选项
    parser.add_argument(
        "--list-voices", 
        action="store_true", 
        help="列出所有可用的语音"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="显示详细日志"
    )
    
    # 解析参数
    args = parser.parse_args(args)
    
    # 设置日志级别
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # 检查必要的参数
    if not args.list_voices and not args.text and not args.file:
        parser.error("必须提供--text或--file参数，或使用--list-voices选项")
    
    try:
        # 创建Pipeline
        pipeline_class = MixedLanguagePipeline if args.mixed_language else TTSPipeline
        pipeline = pipeline_class(
            model_path=args.model_path,
            voices_dir=args.voices_dir,
            device=args.device
        )
        
        # 列出语音列表
        if args.list_voices:
            voices = pipeline.list_voices()
            print("可用的语音列表:")
            for voice in voices:
                print(f"  - {voice}")
            return 0
        
        # 获取输入文本
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                logger.error(f"读取文件失败: {e}")
                return 1
        else:
            text = args.text
        
        # 检查语音ID是否可用
        if args.voice_id not in pipeline.list_voices():
            available_voices = pipeline.list_voices()
            logger.error(f"语音 {args.voice_id} 不可用")
            logger.info(f"可用的语音: {', '.join(available_voices)}")
            return 1
            
        logger.info(f"处理文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # 生成语音
        audio = pipeline.text_to_speech(
            text=text,
            voice_id=args.voice_id,
            output_path=args.output,
            speed=args.speed,
            use_official_pipeline=args.use_official
        )
        
        if audio is None:
            logger.error("生成语音失败")
            return 1
            
        logger.info(f"语音生成成功，已保存至: {args.output}")
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断，退出")
        return 1
    except Exception as e:
        logger.error(f"发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 