#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS系统分步测试脚本 - 用于逐步验证每个组件的效果
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
import numpy as np

# 添加src目录到PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from illufly_tts.preprocessing.segmenter import LanguageSegmenter
from illufly_tts.preprocessing.normalizer import TextNormalizer
from illufly_tts.preprocessing.zh_normalizer_adapter import ChineseNormalizerAdapter
from illufly_tts.g2p.mixed_g2p import MixedG2P
from illufly_tts.g2p.english_g2p import EnglishG2P
from illufly_tts.g2p.chinese_g2p import ChineseG2P
from illufly_tts.vocoders.kokoro_adapter import KokoroAdapter, KokoroVoice
from illufly_tts.pipeline import TTSPipeline, MixedLanguagePipeline

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
    """TTS组件测试类"""
    
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
    
    def test_segmenter(self, text: str):
        """测试语言分段器
        
        Args:
            text: 输入文本
        """
        logger.info("=== 测试语言分段器 ===")
        
        try:
            # 初始化分段器
            segmenter = LanguageSegmenter()
            
            # 分段处理
            start_time = time.time()
            segments = segmenter.segment(text)
            duration = time.time() - start_time
            
            # 保存分段结果
            result_text = f"输入文本: {text}\n\n分段结果: {len(segments)}个段落\n"
            for i, segment in enumerate(segments):
                result_text += f"\n段落 {i+1}:\n"
                result_text += f"  语言: {segment['lang']}\n"
                result_text += f"  文本: {segment['text']}\n"
            
            result_text += f"\n处理时间: {duration:.4f}秒"
            self.save_text_file("segmenter_result", result_text)
            
            # 记录测试结果
            self.log_result(TestResult(
                name="language_segmenter", 
                success=True, 
                data={"segments": len(segments), "time": duration}
            ))
            
            return segments
            
        except Exception as e:
            logger.error(f"测试语言分段器失败: {e}")
            self.log_result(TestResult(
                name="language_segmenter", 
                success=False, 
                error=str(e)
            ))
            return None
    
    def test_normalizer(self, text: str):
        """测试文本规范化器
        
        Args:
            text: 输入文本
        """
        logger.info("=== 测试文本规范化器 ===")
        
        try:
            # 初始化规范化器
            normalizer = TextNormalizer()
            
            # 文本规范化
            start_time = time.time()
            normalized_text = normalizer.normalize(text)
            duration = time.time() - start_time
            
            # 保存规范化结果
            result_text = f"输入文本: {text}\n\n规范化结果: \n{normalized_text}\n"
            result_text += f"\n处理时间: {duration:.4f}秒"
            self.save_text_file("normalizer_result", result_text)
            
            # 记录测试结果
            self.log_result(TestResult(
                name="text_normalizer", 
                success=True, 
                data={"normalized_text": normalized_text, "time": duration}
            ))
            
            # 测试中文规范化适配器
            try:
                chinese_normalizer = ChineseNormalizerAdapter()
                
                start_time = time.time()
                cn_normalized_text = chinese_normalizer.normalize(text)
                cn_duration = time.time() - start_time
                
                # 保存规范化结果
                result_text = f"输入文本: {text}\n\n中文规范化结果: \n{cn_normalized_text}\n"
                result_text += f"\n处理时间: {cn_duration:.4f}秒"
                self.save_text_file("chinese_normalizer_result", result_text)
                
                # 记录测试结果
                self.log_result(TestResult(
                    name="chinese_normalizer", 
                    success=True, 
                    data={"normalized_text": cn_normalized_text, "time": cn_duration}
                ))
                
            except Exception as e:
                logger.error(f"测试中文规范化适配器失败: {e}")
                self.log_result(TestResult(
                    name="chinese_normalizer", 
                    success=False, 
                    error=str(e)
                ))
            
            return normalized_text
            
        except Exception as e:
            logger.error(f"测试文本规范化器失败: {e}")
            self.log_result(TestResult(
                name="text_normalizer", 
                success=False, 
                error=str(e)
            ))
            return None
    
    def test_g2p(self, text: str):
        """测试G2P转换
        
        Args:
            text: 输入文本
        """
        logger.info("=== 测试G2P转换 ===")
        
        # 测试结果汇总
        result_text = f"输入文本: {text}\n\n"
        
        # 测试英文G2P
        try:
            english_g2p = EnglishG2P(nltk_data_path=os.path.expanduser("~/.nltk_data"))
            
            # 仅处理英文部分
            english_text = "Hello world, this is a test."
            
            start_time = time.time()
            english_phonemes = english_g2p.text_to_phonemes(english_text)
            duration = time.time() - start_time
            
            result_text += f"英文G2P结果:\n"
            result_text += f"  输入: {english_text}\n"
            result_text += f"  音素: {english_phonemes}\n"
            result_text += f"  时间: {duration:.4f}秒\n\n"
            
            # 记录测试结果
            self.log_result(TestResult(
                name="english_g2p", 
                success=True, 
                data={"phonemes": english_phonemes, "time": duration}
            ))
            
        except Exception as e:
            logger.error(f"测试英文G2P失败: {e}")
            result_text += f"英文G2P失败: {e}\n\n"
            self.log_result(TestResult(
                name="english_g2p", 
                success=False, 
                error=str(e)
            ))
        
        # 测试中文G2P
        try:
            chinese_g2p = ChineseG2P()
            
            # 仅处理中文部分
            chinese_text = "你好，世界。"
            
            start_time = time.time()
            chinese_phonemes = chinese_g2p.text_to_phonemes(chinese_text)
            duration = time.time() - start_time
            
            result_text += f"中文G2P结果:\n"
            result_text += f"  输入: {chinese_text}\n"
            result_text += f"  音素: {chinese_phonemes}\n"
            result_text += f"  时间: {duration:.4f}秒\n\n"
            
            # 记录测试结果
            self.log_result(TestResult(
                name="chinese_g2p", 
                success=True, 
                data={"phonemes": chinese_phonemes, "time": duration}
            ))
            
        except Exception as e:
            logger.error(f"测试中文G2P失败: {e}")
            result_text += f"中文G2P失败: {e}\n\n"
            self.log_result(TestResult(
                name="chinese_g2p", 
                success=False, 
                error=str(e)
            ))
        
        # 测试混合G2P
        try:
            mixed_g2p = MixedG2P(nltk_data_path=os.path.expanduser("~/.nltk_data"))
            
            start_time = time.time()
            mixed_phonemes = mixed_g2p.text_to_phonemes(text)
            duration = time.time() - start_time
            
            result_text += f"混合G2P结果:\n"
            result_text += f"  输入: {text}\n"
            result_text += f"  音素: {mixed_phonemes}\n"
            result_text += f"  时间: {duration:.4f}秒\n"
            
            # 记录测试结果
            self.log_result(TestResult(
                name="mixed_g2p", 
                success=True, 
                data={"phonemes": mixed_phonemes, "time": duration}
            ))
            
            # 保存所有G2P结果
            self.save_text_file("g2p_result", result_text)
            
            return mixed_phonemes
            
        except Exception as e:
            logger.error(f"测试混合G2P失败: {e}")
            result_text += f"混合G2P失败: {e}\n"
            self.save_text_file("g2p_result", result_text)
            
            self.log_result(TestResult(
                name="mixed_g2p", 
                success=False, 
                error=str(e)
            ))
            return None
    
    def test_vocoder(self, text: str, model_path: str, voices_dir: str, voice_id: str):
        """测试语音合成器
        
        Args:
            text: 输入文本
            model_path: 模型路径
            voices_dir: 语音目录
            voice_id: 语音ID
        """
        logger.info("=== 测试语音合成器 ===")
        
        try:
            # 先预处理文本
            normalizer = TextNormalizer()
            normalized_text = normalizer.normalize(text)
            
            # 初始化G2P
            g2p = MixedG2P()
            
            # 初始化语音合成器
            vocoder = KokoroAdapter(
                model_path=model_path,
                voices_dir=voices_dir,
                g2p=g2p
            )
            
            # 检查语音是否可用
            available_voices = vocoder.list_voices()
            if not available_voices:
                raise ValueError(f"没有可用的语音（目录: {voices_dir}）")
                
            logger.info(f"可用语音: {', '.join(available_voices)}")
            
            if voice_id not in available_voices:
                logger.warning(f"指定的语音ID {voice_id} 不可用，使用可用语音: {available_voices[0]}")
                voice_id = available_voices[0]
            
            # 测试生成音频 - 使用官方Pipeline
            logger.info("使用官方Pipeline生成音频...")
            start_time = time.time()
            official_audio = vocoder.generate_audio(
                text=normalized_text,
                voice_id=voice_id,
                use_pipeline=True
            )
            official_duration = time.time() - start_time
            
            # 保存官方Pipeline生成的音频
            if official_audio is not None:
                official_path = self.save_audio("official_audio", official_audio)
                self.log_result(TestResult(
                    name="official_vocoder", 
                    success=True, 
                    data={"duration": official_duration, "audio_path": str(official_path)}
                ))
                
                # 播放音频
                self.play_audio(official_path)
            else:
                logger.error("官方Pipeline生成音频失败")
                self.log_result(TestResult(
                    name="official_vocoder", 
                    success=False, 
                    error="生成的音频为空"
                ))
            
            # 测试生成音频 - 使用自定义处理
            logger.info("使用自定义处理生成音频...")
            start_time = time.time()
            custom_audio = vocoder.generate_audio(
                text=normalized_text,
                voice_id=voice_id,
                use_pipeline=False
            )
            custom_duration = time.time() - start_time
            
            # 保存自定义处理生成的音频
            if custom_audio is not None:
                custom_path = self.save_audio("custom_audio", custom_audio)
                self.log_result(TestResult(
                    name="custom_vocoder", 
                    success=True, 
                    data={"duration": custom_duration, "audio_path": str(custom_path)}
                ))
                
                # 播放音频
                self.play_audio(custom_path)
            else:
                logger.error("自定义处理生成音频失败")
                self.log_result(TestResult(
                    name="custom_vocoder", 
                    success=False, 
                    error="生成的音频为空"
                ))
            
            return True
            
        except Exception as e:
            logger.error(f"测试语音合成器失败: {e}")
            self.log_result(TestResult(
                name="vocoder", 
                success=False, 
                error=str(e)
            ))
            return False
    
    def test_pipeline(self, text: str, model_path: str, voices_dir: str, voice_id: str, use_mixed: bool = True):
        """测试完整流水线
        
        Args:
            text: 输入文本
            model_path: 模型路径
            voices_dir: 语音目录
            voice_id: 语音ID
            use_mixed: 是否使用混合语言流水线
        """
        logger.info("=== 测试完整流水线 ===")
        
        try:
            # 选择流水线类型
            pipeline_class = MixedLanguagePipeline if use_mixed else TTSPipeline
            pipeline_name = "混合语言流水线" if use_mixed else "标准流水线"
            
            logger.info(f"使用{pipeline_name}")
            
            # 初始化流水线
            pipeline = pipeline_class(
                model_path=model_path,
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
            
            # 测试官方Pipeline
            logger.info("使用官方Pipeline...")
            start_time = time.time()
            official_audio = pipeline.text_to_speech(
                text=text,
                voice_id=voice_id,
                use_official_pipeline=True
            )
            official_duration = time.time() - start_time
            
            # 保存官方Pipeline生成的音频
            if official_audio is not None:
                official_path = self.save_audio("official_pipeline", official_audio)
                self.log_result(TestResult(
                    name="official_pipeline", 
                    success=True, 
                    data={"duration": official_duration, "audio_path": str(official_path)}
                ))
                
                # 播放音频
                self.play_audio(official_path)
            else:
                logger.error("官方Pipeline生成音频失败")
                self.log_result(TestResult(
                    name="official_pipeline", 
                    success=False, 
                    error="生成的音频为空"
                ))
            
            # 测试自定义Pipeline
            logger.info("使用自定义Pipeline...")
            start_time = time.time()
            custom_audio = pipeline.text_to_speech(
                text=text,
                voice_id=voice_id,
                use_official_pipeline=False
            )
            custom_duration = time.time() - start_time
            
            # 保存自定义Pipeline生成的音频
            if custom_audio is not None:
                custom_path = self.save_audio("custom_pipeline", custom_audio)
                self.log_result(TestResult(
                    name="custom_pipeline", 
                    success=True, 
                    data={"duration": custom_duration, "audio_path": str(custom_path)}
                ))
                
                # 播放音频
                self.play_audio(custom_path)
            else:
                logger.error("自定义Pipeline生成音频失败")
                self.log_result(TestResult(
                    name="custom_pipeline", 
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
    
    def run_all_tests(self, text: str, model_path: str, voices_dir: str, voice_id: str, stage: Optional[str] = None):
        """运行所有测试
        
        Args:
            text: 输入文本
            model_path: 模型路径
            voices_dir: 语音目录
            voice_id: 语音ID
            stage: 测试阶段
        """
        logger.info(f"开始测试流程，文本: {text}")
        
        # 根据指定的阶段运行测试
        if stage == "segmenter" or stage is None:
            self.test_segmenter(text)
            
        if stage == "normalizer" or stage is None:
            self.test_normalizer(text)
            
        if stage == "g2p" or stage is None:
            self.test_g2p(text)
            
        if stage == "vocoder" or stage is None:
            self.test_vocoder(text, model_path, voices_dir, voice_id)
            
        if stage == "pipeline" or stage is None:
            self.test_pipeline(text, model_path, voices_dir, voice_id)
        
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
        "--stage", 
        type=str, 
        choices=["segmenter", "normalizer", "g2p", "vocoder", "pipeline"], 
        help="仅测试指定阶段"
    )
    parser.add_argument(
        "--nltk-data", 
        type=str, 
        default="~/.nltk_data", 
        help="NLTK数据目录路径"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="详细输出"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查模型路径和语音目录是否存在
    if (args.stage == "vocoder" or args.stage == "pipeline" or args.stage is None) and \
       (not os.path.exists(args.model_path) or not os.path.exists(args.voices_dir)):
        logger.warning(f"模型路径或语音目录不存在，将跳过需要这些资源的测试阶段")
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
            model_path=args.model_path,
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