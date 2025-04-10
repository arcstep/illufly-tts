#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS系统分步测试脚本 - 简化版，仅测试G2P和Pipeline
"""

import os
import sys
import logging
import argparse
import time
from typing import Dict, List, Optional, Union, Any
import json
from pathlib import Path

import torch
import torchaudio
import numpy as np

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加src目录到PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from illufly_tts.core.g2p.chinese_g2p import ChineseG2P
from illufly_tts.core.pipeline import TTSPipeline
from misaki.zh import ZHG2P  # 用于直接比较

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("tts-test")

class TestResult:
    """测试结果类"""
    
    def __init__(self, name: str, success: bool, data: Any = None, error: Optional[str] = None):
        """初始化测试结果
        
        Args:
            name: 测试名称
            success: 是否成功
            data: 测试数据
            error: 错误信息
        """
        self.name = name
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """转换为字典形式
        
        Returns:
            结果字典
        """
        return {
            "name": self.name,
            "success": self.success,
            "data": str(self.data) if self.data is not None else None,
            "error": self.error,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        """字符串表示
        
        Returns:
            结果描述
        """
        status = "成功" if self.success else "失败"
        result = f"测试 '{self.name}': {status}"
        if self.error:
            result += f", 错误: {self.error}"
        return result

class TTSComponentTest:
    """TTS组件测试类 - 简化版"""
    
    def __init__(self, output_dir: str, verbose: bool = False):
        """初始化测试
        
        Args:
            output_dir: 输出目录
            verbose: 是否详细输出
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.results = []
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志级别
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info(f"测试结果将保存到: {self.output_dir}")
    
    def log_result(self, result: TestResult):
        """记录测试结果
        
        Args:
            result: 测试结果
        """
        self.results.append(result)
        logger.info(str(result))
        
        # 保存测试结果
        result_file = self.output_dir / f"{result.name}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    
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
        """保存音频文件"""
        file_path = str(self.output_dir / f"{name}.wav")  # 确保是字符串
        
        try:
            # 验证音频数据
            logger.info(f"保存音频 - 类型: {type(audio)}, 形状: {audio.shape if hasattr(audio, 'shape') else 'unknown'}")
            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio)
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
            
            # 使用torchaudio保存音频
            torchaudio.save(file_path, audio, sample_rate)
            logger.info(f"已保存音频文件: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"保存音频失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def play_audio(self, file_path: Union[str, Path]):
        """播放音频文件
        
        Args:
            file_path: 音频文件路径
        """
        try:
            import platform
            system = platform.system()
            
            logger.info(f"尝试播放音频: {file_path}")
            
            if system == "Darwin":  # macOS
                os.system(f"afplay {file_path}")
            elif system == "Linux":
                os.system(f"aplay {file_path}")
            elif system == "Windows":
                os.system(f"start {file_path}")
            else:
                logger.warning(f"未知系统类型: {system}，无法自动播放音频")
                
        except Exception as e:
            logger.error(f"播放音频失败: {e}")
    
    def test_g2p(self, text: str):
        """测试G2P转换
        
        Args:
            text: 输入文本
        """
        logger.info("=== 测试G2P转换 ===")
        
        # 测试结果汇总
        result_text = f"输入文本: {text}\n\n"

        # 测试ChineseG2P
        try:
            chinese_g2p = ChineseG2P()
            
            start_time = time.time()
            # 1. 首先获取注音格式
            zhuyin_phonemes = chinese_g2p.text_to_phonemes(text)
            zhuyin_duration = time.time() - start_time
            
            # 2. 获取转换为IPA的结果
            start_time = time.time()
            ipa_phonemes = chinese_g2p.convert_to_ipa(zhuyin_phonemes)
            ipa_duration = time.time() - start_time
            
            # 3. 比较用官方ZHG2P的结果
            try:
                start_time = time.time()
                misaki_g2p = ZHG2P()
                misaki_phonemes, _ = misaki_g2p(text)
                misaki_duration = time.time() - start_time
                
                result_text += f"官方Misaki G2P结果:\n"
                result_text += f"  输入: {text}\n"
                result_text += f"  音素: {misaki_phonemes}\n"
                result_text += f"  时间: {misaki_duration:.4f}秒\n\n"
            except ImportError:
                logger.warning("Misaki未安装，跳过官方G2P测试")
                misaki_phonemes = "N/A"
                
            result_text += f"ChineseG2P结果:\n"
            result_text += f"  输入: {text}\n"
            result_text += f"  注音格式: {zhuyin_phonemes}\n"
            result_text += f"  注音耗时: {zhuyin_duration:.4f}秒\n"
            result_text += f"  IPA格式: {ipa_phonemes}\n"
            result_text += f"  IPA转换耗时: {ipa_duration:.4f}秒\n\n"
            result_text += f"格式比较:\n"
            result_text += f"  官方IPA: {misaki_phonemes}\n"
            result_text += f"  本地IPA: {ipa_phonemes}\n"
            
            # 记录测试结果
            self.log_result(TestResult(
                name="g2p_test", 
                success=True, 
                data={
                    "zhuyin": zhuyin_phonemes,
                    "ipa": ipa_phonemes,
                    "misaki_ipa": misaki_phonemes
                }
            ))
            
            # 保存所有G2P结果
            self.save_text_file("g2p_result", result_text)
            
            return {
                "zhuyin": zhuyin_phonemes,
                "ipa": ipa_phonemes,
                "misaki_ipa": misaki_phonemes
            }
            
        except Exception as e:
            logger.error(f"测试G2P失败: {e}")
            result_text += f"G2P测试失败: {e}\n"
            self.save_text_file("g2p_result", result_text)
            
            self.log_result(TestResult(
                name="g2p_test", 
                success=False, 
                error=str(e)
            ))
            return None
    
    def test_pipeline(self, text: str, repo_id: str, voices_dir: str, voice_id: str):
        """测试完整流水线，对比官方KPipeline和自定义TTSPipeline
        
        Args:   
            text: 输入文本
            repo_id: 模型ID
            voices_dir: 语音目录
            voice_id: 语音ID
        """
        logger.info("=== 测试完整流水线 ===")
        
        try:
            # 1. 首先用官方KPipeline生成音频
            logger.info("使用官方KPipeline生成音频...")
            k_start_time = time.time()
            try:
                from kokoro import KPipeline
                # 加载本地声音文件
                voice_path = os.path.join(voices_dir, f"{voice_id}.pt")
                voice_tensor = torch.load(voice_path, map_location="cpu", weights_only=True)
                
                # 使用本地模型路径和本地声音文件
                k_pipeline = KPipeline(repo_id=repo_id, lang_code='z')
                generator = k_pipeline(text, voice=voice_tensor)  # 直接传入声音张量
                
                # 记录处理信息
                k_audio = None
                k_phonemes = None
                
                # 生成音频
                for i, (gs, ps, audio_data) in enumerate(generator):
                    logger.info(f"KPipeline返回 - 段落 {i+1}:")
                    logger.info(f"  文本: {gs}")
                    logger.info(f"  音素: {ps}")
                    logger.info(f"  音频: {type(audio_data)}, shape: {audio_data.shape if hasattr(audio_data, 'shape') else 'unknown'}")
                    
                    # 只保存第一段
                    if i == 0:
                        k_audio = audio_data
                        k_phonemes = ps
                    
                    # 不再继续处理后续段落
                    break
                    
                k_duration = time.time() - k_start_time
                
                # 保存KPipeline生成的音频
                if k_audio is not None:
                    k_audio_path = self.save_audio("kpipeline_reference", k_audio)
                    self.log_result(TestResult(
                        name="kpipeline_reference", 
                        success=True, 
                        data={"duration": k_duration, "audio_path": str(k_audio_path)}
                    ))
                    logger.info(f"KPipeline生成音频已保存: {k_audio_path}")
                else:
                    logger.warning("KPipeline没有生成音频")
                    
            except Exception as e:
                logger.error(f"KPipeline直接调用失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                k_audio = None
                k_phonemes = None
                k_duration = 0

            # 2. 使用我们的TTSPipeline
            logger.info("\n使用自定义TTSPipeline处理文本...")
            pipeline = TTSPipeline(
                repo_id=repo_id,
                voices_dir=voices_dir
            )
            
            # 检查语音是否可用
            available_voices = pipeline.list_voices()
            if not available_voices:
                raise ValueError(f"没有可用的语音（目录: {voices_dir}）")
                
            logger.info(f"可用语音: {', '.join(available_voices)}")
            
            if voice_id not in available_voices:
                logger.warning(f"指定的语音ID {voice_id} 不可用，使用可用语音: {available_voices[0]}")
                voice_id = available_voices[0]
            
            # 处理文本生成音频
            start_time = time.time()
            audio = pipeline.process(
                text=text,
                voice_id=voice_id
            )
            process_duration = time.time() - start_time
            
            logger.info(f"TTSPipeline返回: {type(audio)}, shape: {audio.shape if hasattr(audio, 'shape') else 'unknown'}")
            
            # 保存TTSPipeline生成的音频
            if audio is not None:
                audio_path = self.save_audio("pipeline_process", audio)
                self.log_result(TestResult(
                    name="pipeline_process", 
                    success=True, 
                    data={"duration": process_duration, "audio_path": str(audio_path)}
                ))
                
                # 记录步骤执行详情
                steps_text = (
                    f"Pipeline处理过程详情:\n"
                    f"1. 处理文本: {text}\n"
                    f"2. 注音格式: {pipeline.text_to_phonemes(text)}\n"
                    f"3. IPA格式: {pipeline.phonemes_to_ipa(pipeline.text_to_phonemes(text))}\n"
                    f"4. 总处理时间: {process_duration:.4f}秒\n"
                )
                self.save_text_file("pipeline_steps", steps_text)
                
                # 3. 对比两个结果
                if k_audio is not None:
                    # 计算差异 - 如果数组长度不同，选择较短的长度
                    try:
                        k_tensor = torch.tensor(k_audio)
                        if k_tensor.shape != audio.shape:
                            min_len = min(k_tensor.shape[0], audio.shape[0])
                            k_tensor = k_tensor[:min_len]
                            audio_short = audio[:min_len]
                            
                            # 计算均方误差
                            mse = torch.mean((k_tensor - audio_short) ** 2).item()
                            logger.info(f"音频差异 (MSE): {mse:.6f}")
                            
                            # 保存对比结果
                            comparison_text = (
                                f"音频对比结果:\n"
                                f"官方KPipeline音频: {k_audio_path}\n"
                                f"自定义TTSPipeline音频: {audio_path}\n"
                                f"音频形状 - 官方: {k_tensor.shape}, 自定义: {audio.shape}\n"
                                f"计算MSE使用长度: {min_len}\n"
                                f"均方误差 (MSE): {mse:.10f}\n"
                                f"官方耗时: {k_duration:.4f}秒\n"
                                f"自定义耗时: {process_duration:.4f}秒\n"
                            )
                            
                            if k_phonemes is not None:
                                our_phonemes = pipeline.text_to_phonemes(text)
                                comparison_text += f"\n音素对比:\n"
                                comparison_text += f"官方: {k_phonemes}\n"
                                comparison_text += f"自定义: {our_phonemes}\n"
                            
                            self.save_text_file("audio_comparison", comparison_text)
                            
                        else:
                            logger.info("两个音频长度一致，直接计算MSE")
                            mse = torch.mean((k_tensor - audio) ** 2).item()
                            logger.info(f"音频差异 (MSE): {mse:.6f}")
                            
                    except Exception as e:
                        logger.error(f"计算音频差异失败: {e}")
                
                # 播放两个音频供听觉对比
                logger.info("\n=== 播放音频对比 ===")
                if k_audio is not None:
                    logger.info("播放官方KPipeline生成的音频:")
                    self.play_audio(k_audio_path)
                    
                logger.info("\n播放自定义TTSPipeline生成的音频:")
                self.play_audio(audio_path)
                
            else:
                logger.error("TTSPipeline生成音频失败")
                self.log_result(TestResult(
                    name="pipeline_process", 
                    success=False, 
                    error="生成的音频为空"
                ))
            
            return True
            
        except Exception as e:
            logger.error(f"测试完整流水线失败: {e}")
            self.log_result(TestResult(
                name="pipeline", 
                success=False, 
                error=str(e)
            ))
            return False
    
    def run_all_tests(self, text: str, repo_id: str, voices_dir: str, voice_id: str, stage: Optional[str] = None):
        """运行所有测试
        
        Args:
            text: 输入文本
            repo_id: 模型ID
            voices_dir: 语音目录
            voice_id: 语音ID
            stage: 测试阶段
        """
        logger.info(f"开始测试流程，文本: {text}")
        
        # 根据指定的阶段运行测试
        if stage == "g2p" or stage is None:
            self.test_g2p(text)
            
        if stage == "pipeline" or stage is None:
            self.test_pipeline(text, repo_id, voices_dir, voice_id)
        
        # 总结测试结果
        success_count = sum(1 for result in self.results if result.success)
        total_count = len(self.results)
        success_rate = success_count / total_count * 100 if total_count > 0 else 0
        
        summary = f"测试完成: {success_count}/{total_count} 通过 ({success_rate:.1f}%)\n\n"
        
        for result in self.results:
            status = "✓" if result.success else "✗"
            summary += f"{status} {result.name}\n"
            if not result.success and result.error:
                summary += f"  错误: {result.error}\n"
        
        self.save_text_file("test_summary", summary)
        logger.info(f"测试摘要:\n{summary}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TTS处理步骤测试")
    
    # 基本参数
    parser.add_argument(
        "-t", "--text", 
        type=str, 
        default="你好，这是一个测试。Hello, this is a test.", 
        help="测试文本"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        type=str, 
        default="tests/output/processing", 
        help="输出目录"
    )
    
    # 模型和语音资源参数
    parser.add_argument(
        "-m", "--repo-id", 
        type=str, 
        default="hexgrad/Kokoro-82M-v1.1-zh",
        help="Kokoro模型路径"
    )
    parser.add_argument(
        "-v", "--voices-dir", 
        type=str, 
        default="./models/Kokoro-82M-v1.1-zh/voices",
        help="语音目录路径"
    )
    parser.add_argument(
        "--voice-id", 
        type=str, 
        default="zf_001", 
        help="语音ID"
    )
    
    # 测试控制参数
    parser.add_argument(
        "--stage", 
        type=str, 
        choices=["g2p", "pipeline"], 
        help="仅测试指定阶段"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="详细输出"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查模型路径和语音目录是否存在
    if (args.stage == "pipeline" or args.stage is None) and \
       (not os.path.exists(args.repo_id)):
        logger.warning(f"模型ID不存在，将跳过需要这些资源的测试阶段")
        # 设置默认的stage
        if args.stage is None:
            args.stage = "g2p"  # 默认执行到g2p阶段
    
    try:
        # 创建测试对象
        tester = TTSComponentTest(
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # 运行测试
        tester.run_all_tests(
            text=args.text,
            repo_id=args.repo_id,
            voices_dir=args.voices_dir,
            voice_id=args.voice_id,
            stage=args.stage
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