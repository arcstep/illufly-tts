#!/usr/bin/env python
"""
Illufly TTS 启动器脚本

这个脚本是一个简单的包装器，使用我们的G2P解决方案运行TTS服务。
它会自动处理依赖检查、导入必要的补丁，并支持中英文混合输入。
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import soundfile as sf

# 将项目根目录添加到Python路径
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Illufly TTS 命令行工具")
    
    # 模型和配置参数
    parser.add_argument("--model-path", type=str, default="./models/Kokoro-82M-v1.1-zh", help="模型路径")
    parser.add_argument("--voice-dir", type=str, default="./voices", help="语音目录")
    parser.add_argument("--output", type=str, default="./output/tts_output.wav", help="输出音频文件路径")
    parser.add_argument("--device", type=str, default="自动", help="设备类型: 'cpu', 'cuda', 'mps'或'自动'")
    parser.add_argument("--voice", type=str, default="zf_001", help="语音ID")
    
    # 文本参数
    parser.add_argument("--text", type=str, required=True, help="要转换的文本")
    parser.add_argument("--repeat", type=int, default=0, help="如果文本过短，重复次数")
    parser.add_argument("--file", type=str, help="从文件读取文本")
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小")
    
    # 调试和控制参数
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--force-custom", action="store_true", help="强制使用自定义Pipeline")
    parser.add_argument("--force-official", action="store_true", help="强制使用官方KPipeline")
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 设置输出路径
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 确定设备类型
    device = None
    if args.device == "自动":
        # 自动选择设备
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    # 打印配置信息
    print("\n=== Illufly TTS 配置 ===")
    print(f"模型路径: {args.model_path}")
    print(f"语音目录: {args.voice_dir}")
    print(f"输出文件: {output_path.absolute()}")
    print(f"设备: {args.device}")
    print(f"语音ID: {args.voice}")
    print(f"文本: {args.text}")
    print("=====================\n")
    
    # 检查自定义Pipeline选项
    use_custom_pipeline = False
    if args.force_custom and args.force_official:
        print("错误: --force-custom和--force-official不能同时使用")
        return 1
    elif args.force_custom:
        use_custom_pipeline = True
        print("使用自定义Pipeline (无espeak依赖)")
    elif args.force_official:
        use_custom_pipeline = False
        print("使用官方KPipeline")
        
    try:
        # 初始化TTS服务
        print("初始化TTS服务...")
        from src.illufly_tts import TTSService
        service = TTSService(
            model_path=args.model_path,
            voice_dir=args.voice_dir,
            device=device,
            use_custom_pipeline=use_custom_pipeline
        )
        
        # 预处理文本
        text = args.text
        
        # 检查文本长度
        min_length = 5
        if len(text) < min_length and args.repeat > 0:
            # 对短文本进行重复以获得更好的生成效果
            repeat_count = args.repeat
            print(f"文本太短，重复{repeat_count}次以确保生成质量")
            text = text + "。" * repeat_count
        
        # 生成语音
        print("生成语音...")
        try:
            audio = service.convert_text(
                text=text,
                voice_id=args.voice
            )
            
            # 保存音频文件
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            sf.write(args.output, audio, 24000)
            print(f"语音已保存至: {args.output}")
            return 0
            
        except Exception as e:
            print(f"语音生成失败: {str(e)}")
            return 1
            
    except Exception as e:
        print(f"初始化TTS服务失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main() 