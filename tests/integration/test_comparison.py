#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS系统对比测试脚本 - 用于比较官方和自定义实现的差异
"""

import os
import sys
import logging
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

import torch
import numpy as np

# 添加src目录到PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("tts-comparison")

# 导入自定义模块
from illufly_tts.preprocessing.normalizer import TextNormalizer, ChineseNormalizerAdapter
from illufly_tts.g2p.mixed_g2p import MixedG2P
from illufly_tts.pipeline import TTSPipeline, MixedLanguagePipeline

# 尝试导入官方模块
try:
    from kokoro.pipeline import KPipeline
    from kokoro.model import KModel
    try:
        # 尝试导入misaki库
        import misaki
        from misaki import en
        # 检查是否有G2P类
        misaki_g2p_available = hasattr(misaki.en, 'G2P')
        if misaki_g2p_available:
            MISAKI_AVAILABLE = True
            logger.info("成功导入misaki库和G2P功能")
        else:
            MISAKI_AVAILABLE = False
            logger.warning("misaki库中没有发现G2P功能")
    except ImportError:
        MISAKI_AVAILABLE = False
        logging.warning("无法导入misaki，官方文本规范化对比将被跳过")
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    logging.warning("无法导入kokoro，官方实现对比将被跳过")

class ComparisonTest:
    """TTS实现对比测试类"""
    
    def __init__(self, output_dir: str, verbose: bool = False):
        """初始化测试
        
        Args:
            output_dir: 输出目录
            verbose: 是否详细输出
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志级别
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info(f"测试结果将保存到: {self.output_dir}")
        
        # 记录支持的测试
        self.can_test_normalizer = MISAKI_AVAILABLE
        self.can_test_pipeline = KOKORO_AVAILABLE
        
        logger.info(f"misaki可用: {MISAKI_AVAILABLE}")
        logger.info(f"kokoro可用: {KOKORO_AVAILABLE}")
    
    def save_text_file(self, name: str, content: str):
        """保存文本文件
        
        Args:
            name: 文件名
            content: 文件内容
        """
        file_path = self.output_dir / f"{name}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"已保存文本文件: {file_path}")
        return file_path
    
    def save_audio(self, name: str, audio: torch.Tensor, sample_rate: int = 24000):
        """保存音频文件
        
        Args:
            name: 文件名
            audio: 音频张量
            sample_rate: 采样率
        """
        file_path = self.output_dir / f"{name}.wav"
        
        try:
            # 转换为NumPy数组
            audio_np = audio.cpu().numpy()
            
            # 保存音频
            try:
                from scipy.io import wavfile
                wavfile.write(file_path, sample_rate, audio_np)
            except ImportError:
                logger.warning("未找到scipy，尝试使用soundfile")
                try:
                    import soundfile as sf
                    sf.write(file_path, audio_np, sample_rate)
                except ImportError:
                    logger.error("无法保存音频，请安装scipy或soundfile")
                    return None
                
            logger.info(f"已保存音频文件: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            return None
    
    def play_audio(self, file_path_1: Path, file_path_2: Optional[Path] = None):
        """播放音频文件进行对比
        
        Args:
            file_path_1: 第一个音频文件路径
            file_path_2: 第二个音频文件路径
        """
        try:
            import platform
            system = platform.system()
            
            logger.info(f"播放音频1: {file_path_1}")
            
            if system == "Darwin":  # macOS
                os.system(f"afplay {file_path_1}")
            elif system == "Linux":
                os.system(f"aplay {file_path_1}")
            elif system == "Windows":
                os.system(f"start {file_path_1}")
            else:
                logger.warning(f"未知系统类型: {system}，无法自动播放音频")
            
            if file_path_2:
                # 等待一秒再播放第二个
                time.sleep(1)
                logger.info(f"播放音频2: {file_path_2}")
                
                if system == "Darwin":  # macOS
                    os.system(f"afplay {file_path_2}")
                elif system == "Linux":
                    os.system(f"aplay {file_path_2}")
                elif system == "Windows":
                    os.system(f"start {file_path_2}")
                
        except Exception as e:
            logger.error(f"播放音频失败: {e}")
    
    def compare_text_normalization(self, input_text: str) -> Dict[str, Any]:
        """比较文本规范化实现
        
        Args:
            input_text: 输入文本
            
        Returns:
            比较结果
        """
        logger.info("=== 比较文本规范化 ===")
        
        result = {
            "input_text": input_text,
            "timestamp": time.time()
        }
        
        # 测试自定义规范化器
        try:
            custom_normalizer = TextNormalizer()
            start_time = time.time()
            custom_result = custom_normalizer.normalize(input_text)
            custom_time = time.time() - start_time
            
            result["custom"] = {
                "output": custom_result,
                "time": custom_time
            }
            
            logger.info(f"自定义规范化 (耗时: {custom_time:.4f}秒):\n{custom_result}")
            
            # 保存到文本文件
            self.save_text_file("custom_normalized", custom_result)
        except Exception as e:
            logger.error(f"自定义规范化失败: {e}")
            result["custom"] = {
                "error": str(e)
            }
        
        # 测试中文规范化适配器
        try:
            adapter_normalizer = ChineseNormalizerAdapter(use_misaki=MISAKI_AVAILABLE)  # 使用misaki如果可用
            start_time = time.time()
            adapter_result = adapter_normalizer.normalize(input_text)
            adapter_time = time.time() - start_time
            
            result["adapter"] = {
                "output": adapter_result,
                "time": adapter_time
            }
            
            logger.info(f"适配器规范化 (耗时: {adapter_time:.4f}秒):\n{adapter_result}")
            
            # 保存到文本文件
            self.save_text_file("adapter_normalized", adapter_result)
        except Exception as e:
            logger.error(f"适配器规范化失败: {e}")
            result["adapter"] = {
                "error": str(e)
            }
        
        # 如果misaki可用，进行官方规范化对比
        if MISAKI_AVAILABLE:
            try:
                # 使用misaki的G2P
                g2p = en.G2P(trf=False, british=False, fallback=None)
                
                start_time = time.time()
                # 只进行G2P转换
                phonemes, tokens = g2p(input_text)
                misaki_time = time.time() - start_time
                
                misaki_result = {
                    "phonemes": phonemes,
                    "tokens": [t.text for t in tokens if hasattr(t, 'text')]
                }
                
                result["misaki"] = {
                    "output": " ".join(misaki_result["tokens"]),  # 只是简单合并token文本
                    "phonemes": phonemes,
                    "tokens_count": len(tokens),
                    "time": misaki_time
                }
                
                logger.info(f"Misaki G2P (耗时: {misaki_time:.4f}秒):")
                logger.info(f"音素结果: {phonemes}")
                
                # 保存到文本文件
                self.save_text_file("misaki_result", json.dumps(misaki_result, ensure_ascii=False, indent=2))
            except Exception as e:
                logger.error(f"Misaki G2P失败: {e}")
                result["misaki"] = {
                    "error": str(e)
                }
        else:
            logger.warning("Misaki不可用，跳过官方规范化对比")
            result["misaki"] = {
                "error": "Misaki库不可用"
            }
        
        return result
    
    def compare_pipeline(
        self, 
        input_text: str, 
        model_path: str, 
        voices_dir: str, 
        voice_id: str
    ) -> Dict[str, Any]:
        """比较流水线实现
        
        Args:
            input_text: 输入文本
            model_path: 模型路径
            voices_dir: 语音目录
            voice_id: 语音ID
            
        Returns:
            比较结果
        """
        logger.info("=== 比较完整流水线 ===")
        
        result = {
            "input_text": input_text,
            "timestamp": time.time()
        }
        
        if not self.can_test_pipeline:
            logger.error("kokoro不可用，无法比较流水线实现")
            result["error"] = "kokoro不可用"
            return result
        
        # 获取可用语音列表
        try:
            # 初始化模型
            model = KModel(repo_id="hexgrad/Kokoro-82M-v1.1-zh", model_path=model_path)
            
            # 检查语音文件
            if not os.path.exists(voices_dir):
                logger.error(f"语音目录不存在: {voices_dir}")
                result["error"] = f"语音目录不存在: {voices_dir}"
                return result
            
            voice_files = [f for f in os.listdir(voices_dir) if f.endswith('.pt') or f.endswith('.npy')]
            if not voice_files:
                logger.error(f"语音目录中没有有效的语音文件: {voices_dir}")
                result["error"] = f"语音目录中没有有效的语音文件: {voices_dir}"
                return result
            
            available_voices = [os.path.splitext(f)[0] for f in voice_files]
            logger.info(f"可用语音: {', '.join(available_voices)}")
            
            if voice_id and voice_id not in available_voices:
                logger.warning(f"指定的语音ID {voice_id} 不可用，使用可用语音: {available_voices[0]}")
                voice_id = available_voices[0]
            elif not voice_id:
                voice_id = available_voices[0]
            
            result["voice_id"] = voice_id
            result["available_voices"] = available_voices
            
        except Exception as e:
            logger.error(f"检查语音失败: {e}")
            result["error"] = f"检查语音失败: {e}"
            return result
        
        # 测试官方Pipeline
        try:
            # 创建官方Pipeline
            kpipeline = KPipeline(
                lang_code="z", 
                model=model, 
                repo_id="hexgrad/Kokoro-82M-v1.1-zh"
            )
            
            # 记录起始时间
            start_time = time.time()
            
            # 生成音频
            official_results = list(kpipeline(input_text, voice_id))
            
            # 合并所有音频
            official_audio_parts = [r.audio for r in official_results if r.audio is not None]
            if not official_audio_parts:
                raise ValueError("官方Pipeline未生成任何音频")
            
            official_audio = torch.cat(official_audio_parts, dim=0)
            official_time = time.time() - start_time
            
            # 保存结果
            official_path = self.save_audio("official_pipeline", official_audio)
            
            result["official"] = {
                "audio_path": str(official_path),
                "time": official_time,
                "segment_count": len(official_results)
            }
            
            logger.info(f"官方Pipeline生成音频完成 (耗时: {official_time:.4f}秒)")
            
        except Exception as e:
            logger.error(f"官方Pipeline生成失败: {e}")
            result["official"] = {
                "error": str(e)
            }
        
        # 测试自定义Pipeline
        try:
            # 创建自定义Pipeline
            custom_pipeline = MixedLanguagePipeline(
                model_path=model_path,
                voices_dir=voices_dir
            )
            
            # 记录起始时间
            start_time = time.time()
            
            # 生成音频 - 使用官方的调用方式
            official_custom_audio = custom_pipeline.text_to_speech(
                text=input_text,
                voice_id=voice_id,
                use_official_pipeline=True
            )
            official_custom_time = time.time() - start_time
            
            # 保存结果
            official_custom_path = self.save_audio("custom_with_official", official_custom_audio)
            
            result["custom_official"] = {
                "audio_path": str(official_custom_path),
                "time": official_custom_time
            }
            
            logger.info(f"自定义Pipeline(官方模式)生成音频完成 (耗时: {official_custom_time:.4f}秒)")
            
            # 测试完全自定义Pipeline
            start_time = time.time()
            
            # 生成音频 - 使用自定义的调用方式
            full_custom_audio = custom_pipeline.text_to_speech(
                text=input_text,
                voice_id=voice_id,
                use_official_pipeline=False
            )
            full_custom_time = time.time() - start_time
            
            # 保存结果
            full_custom_path = self.save_audio("full_custom", full_custom_audio)
            
            result["full_custom"] = {
                "audio_path": str(full_custom_path),
                "time": full_custom_time
            }
            
            logger.info(f"完全自定义Pipeline生成音频完成 (耗时: {full_custom_time:.4f}秒)")
            
        except Exception as e:
            logger.error(f"自定义Pipeline生成失败: {e}")
            if "custom_official" not in result:
                result["custom_official"] = {
                    "error": str(e)
                }
            if "full_custom" not in result:
                result["full_custom"] = {
                    "error": str(e)
                }
        
        # 保存比较结果
        comparison_text = f"输入文本: {input_text}\n\n"
        comparison_text += f"语音ID: {voice_id}\n"
        comparison_text += f"可用语音: {', '.join(available_voices)}\n\n"
        
        if "official" in result:
            comparison_text += "=== 官方Pipeline ===\n"
            if "error" in result["official"]:
                comparison_text += f"错误: {result['official']['error']}\n"
            else:
                comparison_text += f"音频: {result['official']['audio_path']}\n"
                comparison_text += f"耗时: {result['official']['time']:.4f}秒\n"
                comparison_text += f"分段数: {result['official']['segment_count']}\n"
            comparison_text += "\n"
        
        if "custom_official" in result:
            comparison_text += "=== 自定义Pipeline(官方模式) ===\n"
            if "error" in result["custom_official"]:
                comparison_text += f"错误: {result['custom_official']['error']}\n"
            else:
                comparison_text += f"音频: {result['custom_official']['audio_path']}\n"
                comparison_text += f"耗时: {result['custom_official']['time']:.4f}秒\n"
            comparison_text += "\n"
        
        if "full_custom" in result:
            comparison_text += "=== 完全自定义Pipeline ===\n"
            if "error" in result["full_custom"]:
                comparison_text += f"错误: {result['full_custom']['error']}\n"
            else:
                comparison_text += f"音频: {result['full_custom']['audio_path']}\n"
                comparison_text += f"耗时: {result['full_custom']['time']:.4f}秒\n"
            comparison_text += "\n"
        
        # 比较性能
        if all(k in result and "time" in result[k] for k in ["official", "custom_official", "full_custom"]):
            comparison_text += "=== 性能比较 ===\n"
            official_time = result["official"]["time"]
            custom_official_time = result["custom_official"]["time"]
            full_custom_time = result["full_custom"]["time"]
            
            comparison_text += f"官方Pipeline:          {official_time:.4f}秒\n"
            comparison_text += f"自定义Pipeline(官方模式): {custom_official_time:.4f}秒 ({custom_official_time/official_time:.2f}x)\n"
            comparison_text += f"完全自定义Pipeline:     {full_custom_time:.4f}秒 ({full_custom_time/official_time:.2f}x)\n"
        
        self.save_text_file("pipeline_comparison", comparison_text)
        
        # 播放音频对比
        logger.info("播放音频进行对比...")
        try:
            if "official" in result and "audio_path" in result["official"]:
                logger.info("播放官方Pipeline生成的音频")
                self.play_audio(Path(result["official"]["audio_path"]))
                time.sleep(0.5)
            
            if "custom_official" in result and "audio_path" in result["custom_official"]:
                logger.info("播放自定义Pipeline(官方模式)生成的音频")
                self.play_audio(Path(result["custom_official"]["audio_path"]))
                time.sleep(0.5)
            
            if "full_custom" in result and "audio_path" in result["full_custom"]:
                logger.info("播放完全自定义Pipeline生成的音频")
                self.play_audio(Path(result["full_custom"]["audio_path"]))
        except Exception as e:
            logger.error(f"播放音频失败: {e}")
        
        return result
    
    def run_comparison(
        self, 
        text: str, 
        model_path: str, 
        voices_dir: str, 
        voice_id: str, 
        mode: str
    ):
        """运行对比测试
        
        Args:
            text: 测试文本
            model_path: 模型路径
            voices_dir: 语音目录
            voice_id: 语音ID
            mode: 测试模式
        """
        logger.info(f"开始比较测试，模式: {mode}, 文本: {text}")
        
        if mode == "normalization" or mode == "all":
            # 即使misaki不可用，也执行我们自己的文本规范化
            try:
                result = self.compare_text_normalization(text)
                output_path = self.output_dir / "normalization_result.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"文本规范化对比结果已保存到: {output_path}")
            except Exception as e:
                logger.error(f"文本规范化比较失败: {e}")
        
        if mode == "pipeline" or mode == "all":
            if self.can_test_pipeline:
                self.compare_pipeline(text, model_path, voices_dir, voice_id)
            else:
                logger.error("无法进行流水线比较，kokoro不可用")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TTS实现对比测试")
    
    # 基本参数
    parser.add_argument(
        "-t", "--text", 
        type=str, 
        default="你好，这是语音合成测试。Hello, this is a TTS test.", 
        help="测试文本"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        type=str, 
        default="tests/output/comparison", 
        help="输出目录"
    )
    
    # 模型和语音资源参数
    parser.add_argument(
        "-m", "--model-path", 
        type=str, 
        default="./models/kokoro_model",
        help="Kokoro模型路径"
    )
    parser.add_argument(
        "-v", "--voices-dir", 
        type=str, 
        default="./models/voices",
        help="语音目录路径"
    )
    parser.add_argument(
        "--voice-id", 
        type=str, 
        default=None, 
        help="语音ID"
    )
    
    # 测试控制参数
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["normalization", "pipeline", "all"], 
        default="normalization", 
        help="比较模式"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="详细输出"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查模型路径和语音目录是否存在，自动调整模式
    if args.mode in ["pipeline", "all"] and (not os.path.exists(args.model_path) or not os.path.exists(args.voices_dir)):
        logger.warning(f"模型路径或语音目录不存在，将仅运行文本规范化测试")
        args.mode = "normalization"
    
    try:
        # 创建测试对象
        tester = ComparisonTest(
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # 运行测试
        tester.run_comparison(
            text=args.text,
            model_path=args.model_path,
            voices_dir=args.voices_dir,
            voice_id=args.voice_id,
            mode=args.mode
        )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断，退出")
        return 1
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 