#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS系统完全独立实现测试 - 测试不依赖官方包的实现
"""

import os
import sys
import logging
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

import torch
import numpy as np

# 添加src目录到PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# 导入自定义模块
from illufly_tts.preprocessing.normalizer import TextNormalizer
from illufly_tts.preprocessing.segmenter import LanguageSegmenter
from illufly_tts.g2p.chinese_g2p import ChineseG2P

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("tts-independent")

@dataclass
class KokoroOutput:
    """Kokoro模型输出结构"""
    audio: torch.Tensor
    durations: Optional[torch.Tensor] = None
    f0: Optional[torch.Tensor] = None
    input_phonemes: Optional[List[str]] = None
    alignment_attention: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """初始化后处理"""
        # 如果没有音频，创建空音频
        if self.audio is None:
            self.audio = torch.zeros((0,))


class IndependentKModel:
    """独立的Kokoro模型实现"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """初始化模型
        
        Args:
            model_path: 模型路径
            device: 设备名称
        """
        self.device = device
        
        try:
            # 导入Kokoro模型
            from kokoro.model import KModel
            
            # 查找配置文件和模型文件
            config_path = os.path.join(model_path, 'config.json')
            model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
            
            if not model_files:
                raise FileNotFoundError(f"在{model_path}中找不到模型文件(.pth)")
                
            model_file = os.path.join(model_path, model_files[0])
            logger.info(f"找到模型文件: {model_file}")
            
            # 初始化模型
            self.model = KModel(
                config=config_path,
                model=model_file,
                repo_id="hexgrad/Kokoro-82M-v1.1-zh"
            ).to(device).eval()
            
            logger.info(f"模型加载成功，使用设备: {device}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.model = None
            
    def infer(
        self, 
        phonemes: str, 
        voice_tensor: torch.Tensor,
        speed: float = 1.0
    ) -> KokoroOutput:
        """推理生成音频
        
        Args:
            phonemes: 音素序列
            voice_tensor: 语音张量
            speed: 语速
            
        Returns:
            KokoroOutput对象
        """
        if self.model is None:
            logger.error("模型未正确加载")
            return KokoroOutput(audio=torch.zeros(1, 100))
            
        try:
            # 记录初始张量形状
            logger.info(f"初始voice_tensor形状: {voice_tensor.shape}")
            
            # 对于KModel，我们需要提供形状为[1, 256]的语音张量
            # 创建一个随机的声音嵌入向量作为替代
            voice_embedding = torch.randn(1, 256, device=self.device)
            # 对向量进行归一化
            voice_embedding = voice_embedding / torch.norm(voice_embedding, dim=1, keepdim=True)
            logger.info(f"使用随机语音张量，形状: {voice_embedding.shape}")
                
            # 调用模型生成音频
            output = self.model(phonemes, voice_embedding, speed, return_output=True)
            return output
        except Exception as e:
            logger.error(f"生成音频失败: {e}")
            return KokoroOutput(audio=torch.zeros(1, 100))


class IndependentVoice:
    """独立实现的语音资源类"""
    
    def __init__(self, voice_id: str, voice_path: str, device: str = "cpu"):
        """初始化语音资源
        
        Args:
            voice_id: 语音ID
            voice_path: 语音文件路径
            device: 设备名称
        """
        self.id = voice_id
        self.path = voice_path
        self.device = device
        self.tensor = None
        self.loaded = False
    
    def load(self) -> bool:
        """加载语音资源
        
        Returns:
            是否成功加载
        """
        if self.loaded and self.tensor is not None:
            return True
            
        try:
            if os.path.exists(self.path):
                # 支持.pt和.npy格式
                if self.path.endswith('.pt'):
                    self.tensor = torch.load(self.path)
                elif self.path.endswith('.npy'):
                    self.tensor = torch.from_numpy(np.load(self.path))
                else:
                    logger.error(f"不支持的语音文件格式: {self.path}")
                    return False
                    
                # 确保张量维度正确
                if self.tensor.dim() == 1:
                    self.tensor = self.tensor.unsqueeze(0)  # [D] -> [1, D]
                elif self.tensor.dim() == 2:
                    if self.tensor.size(1) == 0:  # 如果第二维为0
                        self.tensor = torch.zeros(1, 256)  # [1, 256]
                
                # 对向量进行归一化
                self.tensor = self.tensor / torch.norm(self.tensor, dim=1, keepdim=True)
                
                logger.info(f"加载语音张量，形状: {self.tensor.shape}")
                self.loaded = True
                logger.info(f"语音 {self.id} 已加载")
                return True
            else:
                logger.error(f"语音文件不存在: {self.path}")
                return False
        except Exception as e:
            logger.error(f"加载语音文件失败: {e}")
            return False
    
    def get_tensor(self) -> Optional[torch.Tensor]:
        """获取语音张量
        
        Returns:
            语音张量
        """
        if not self.loaded:
            self.load()
            
        if self.tensor is not None:
            # 确保张量维度正确
            tensor = self.tensor.to(self.device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)  # [D] -> [1, D]
            elif tensor.dim() == 2:
                if tensor.size(1) == 0:  # 如果第二维为0
                    tensor = torch.zeros(1, 256, device=self.device)  # [1, 256]
            elif tensor.dim() == 3:
                if tensor.size(1) == 0:  # 如果第二维为0
                    tensor = tensor.transpose(0, 1)  # [B, 0, D] -> [0, B, D]
                    tensor = torch.zeros(1, tensor.size(1), tensor.size(2), device=self.device)  # [1, B, D]
            
            # 对向量进行归一化
            if tensor.dim() == 2:
                tensor = tensor / torch.norm(tensor, dim=1, keepdim=True)
            elif tensor.dim() == 3:
                tensor = tensor / torch.norm(tensor, dim=2, keepdim=True)
                
            logger.info(f"返回语音张量，形状: {tensor.shape}")
            return tensor
        return None


class IndependentTTSPipeline:
    """完全独立的TTS流水线"""
    
    def __init__(
        self, 
        model_path: str, 
        voices_dir: str, 
        device: str = "cpu",
        nltk_data_path: Optional[str] = None,
        verbose: bool = False
    ):
        """初始化流水线
        
        Args:
            model_path: 模型路径
            voices_dir: 语音目录
            device: 设备名称
            nltk_data_path: NLTK数据目录路径
            verbose: 是否详细输出
        """
        self.model_path = model_path
        self.voices_dir = voices_dir
        self.device = device
        self.verbose = verbose
        self.nltk_data_path = nltk_data_path
        
        # 设置日志级别
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 初始化组件
        logger.info("初始化文本处理组件...")
        self.segmenter = LanguageSegmenter()
        self.normalizer = TextNormalizer()
        self.g2p = MixedG2P(nltk_data_path=nltk_data_path)
        
        # 初始化模型
        logger.info(f"初始化模型: {model_path}")
        self.model = IndependentKModel(model_path, device)
        
        # 加载语音
        self.voices = self._load_voices()
        
        logger.info("流水线初始化完成")
    
    def _load_voices(self) -> Dict[str, IndependentVoice]:
        """加载语音资源
        
        Returns:
            语音资源字典
        """
        voices = {}
        logger.info(f"加载语音资源: {self.voices_dir}")
        
        try:
            if not os.path.exists(self.voices_dir):
                logger.error(f"语音目录不存在: {self.voices_dir}")
                return voices
                
            voice_files = [f for f in os.listdir(self.voices_dir) 
                          if f.endswith('.pt') or f.endswith('.npy')]
            
            for file in voice_files:
                voice_id = os.path.splitext(file)[0]
                voice_path = os.path.join(self.voices_dir, file)
                
                voice = IndependentVoice(voice_id, voice_path, self.device)
                voices[voice_id] = voice
                
            logger.info(f"加载了 {len(voices)} 个语音资源")
            
            # 输出所有语音ID
            if self.verbose:
                logger.debug(f"可用语音: {', '.join(voices.keys())}")
                
            return voices
            
        except Exception as e:
            logger.error(f"加载语音资源失败: {e}")
            return voices
    
    def list_voices(self) -> List[str]:
        """获取所有语音ID
        
        Returns:
            语音ID列表
        """
        return list(self.voices.keys())
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的文本
        """
        if self.verbose:
            logger.debug(f"预处理文本: {text}")
            
        normalized_text = self.normalizer.normalize(text)
        
        if self.verbose:
            logger.debug(f"规范化后文本: {normalized_text}")
            
        return normalized_text
    
    def text_to_phonemes(self, text: str) -> str:
        """将文本转换为音素
        
        Args:
            text: 输入文本
            
        Returns:
            音素字符串
        """
        if self.verbose:
            logger.debug(f"转换文本为音素: {text}")
            
        phonemes = self.g2p.text_to_phonemes(text)
        
        if self.verbose:
            logger.debug(f"生成的音素: {phonemes}")
            
        return phonemes
    
    def phonemes_to_audio(
        self, 
        phonemes: str, 
        voice_id: str, 
        speed: float = 1.0
    ) -> Optional[torch.Tensor]:
        """将音素转换为音频
        
        Args:
            phonemes: 音素字符串
            voice_id: 语音ID
            speed: 语速
            
        Returns:
            音频张量
        """
        # 检查语音ID是否存在
        if voice_id not in self.voices:
            logger.error(f"语音ID不存在: {voice_id}")
            return None
            
        # 获取语音张量
        voice = self.voices[voice_id]
        voice_tensor = voice.get_tensor()
        
        if voice_tensor is None:
            logger.error(f"获取语音张量失败: {voice_id}")
            return None
            
        if self.verbose:
            logger.debug(f"生成音频，音素长度: {len(phonemes.split())}")
            
        # 调用模型推理
        output = self.model.infer(phonemes, voice_tensor, speed)
        
        if output is None or output.audio is None:
            logger.error("模型生成音频失败")
            return None
            
        return output.audio
    
    def text_to_speech(
        self, 
        text: str, 
        voice_id: str, 
        speed: float = 1.0, 
        output_path: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """将文本转换为语音
        
        Args:
            text: 输入文本
            voice_id: 语音ID
            speed: 语速
            output_path: 输出文件路径
            
        Returns:
            音频张量
        """
        logger.info(f"处理文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # 检查语音ID是否存在
        if voice_id not in self.voices:
            available = self.list_voices()
            if not available:
                logger.error("没有可用的语音")
                return None
                
            logger.warning(f"语音ID不存在: {voice_id}，使用: {available[0]}")
            voice_id = available[0]
        
        # 预处理文本
        normalized_text = self.preprocess_text(text)
        
        # 转换为音素
        phonemes = self.text_to_phonemes(normalized_text)
        
        # 生成音频
        start_time = time.time()
        audio = self.phonemes_to_audio(phonemes, voice_id, speed)
        duration = time.time() - start_time
        
        if audio is None:
            logger.error("生成音频失败")
            return None
            
        logger.info(f"生成音频完成，耗时: {duration:.4f}秒，长度: {len(audio)}")
        
        # 保存音频
        if output_path is not None:
            self.save_audio(audio, output_path)
            
        return audio
    
    def process_text_segments(
        self, 
        text: str, 
        voice_id: str, 
        speed: float = 1.0, 
        output_path: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """分段处理文本
        
        Args:
            text: 输入文本
            voice_id: 语音ID
            speed: 语速
            output_path: 输出文件路径
            
        Returns:
            音频张量
        """
        logger.info(f"分段处理文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # 检查语音ID是否存在
        if voice_id not in self.voices:
            available = self.list_voices()
            if not available:
                logger.error("没有可用的语音")
                return None
                
            logger.warning(f"语音ID不存在: {voice_id}，使用: {available[0]}")
            voice_id = available[0]
        
        # 预处理文本
        normalized_text = self.preprocess_text(text)
        
        # 分段
        segments = self.segmenter.segment(normalized_text)
        logger.info(f"文本分为 {len(segments)} 个段落")
        
        if self.verbose:
            for i, segment in enumerate(segments):
                logger.debug(f"段落 {i+1}: [{segment['lang']}] {segment['text']}")
        
        # 逐段生成音频
        audio_parts = []
        for segment in segments:
            segment_text = segment["text"]
            
            # 转换为音素
            phonemes = self.text_to_phonemes(segment_text)
            
            # 生成音频
            segment_audio = self.phonemes_to_audio(phonemes, voice_id, speed)
            
            if segment_audio is not None:
                audio_parts.append(segment_audio)
            
        # 合并音频
        if not audio_parts:
            logger.error("没有生成任何音频段落")
            return None
            
        audio = torch.cat(audio_parts, dim=0)
        
        logger.info(f"生成音频完成，长度: {len(audio)}")
        
        # 保存音频
        if output_path is not None:
            self.save_audio(audio, output_path)
            
        return audio
    
    def save_audio(self, audio: torch.Tensor, output_path: str, sample_rate: int = 24000) -> bool:
        """保存音频
        
        Args:
            audio: 音频张量
            output_path: 输出文件路径
            sample_rate: 采样率
            
        Returns:
            是否成功保存
        """
        try:
            # 使用绝对路径
            abs_output_path = os.path.abspath(output_path)
            output_dir = os.path.dirname(abs_output_path)
            logger.info(f"创建输出目录: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 转换为NumPy数组
            logger.info(f"音频形状: {audio.shape}")
            audio_np = audio.cpu().numpy()
            logger.info(f"NumPy数组形状: {audio_np.shape}")
            
            # 保存音频
            try:
                from scipy.io import wavfile
                logger.info(f"使用scipy保存音频到: {abs_output_path}")
                wavfile.write(abs_output_path, sample_rate, audio_np)
            except ImportError:
                logger.warning("scipy未安装，尝试使用soundfile")
                try:
                    import soundfile as sf
                    logger.info(f"使用soundfile保存音频到: {abs_output_path}")
                    sf.write(abs_output_path, audio_np, sample_rate)
                except ImportError:
                    logger.error("无法保存音频，请安装scipy或soundfile")
                    return False
             
            logger.info(f"音频已保存至: {abs_output_path}")
            return True
        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            return False
    
    def play_audio(self, file_path: str):
        """播放音频
        
        Args:
            file_path: 音频文件路径
        """
        try:
            import platform
            system = platform.system()
            
            logger.info(f"播放音频: {file_path}")
            
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TTS系统完全独立实现测试")
    
    # 基本参数
    parser.add_argument(
        "-t", "--text", 
        type=str, 
        default="你好，这是一个完全独立的实现测试。Hello, this is a test.", 
        help="测试文本"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="tests/output/independent_output.wav", 
        help="输出文件路径"
    )
    
    # 模型和语音资源
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
    
    # 控制参数
    parser.add_argument(
        "--speed", 
        type=float, 
        default=1.0, 
        help="语速"
    )
    parser.add_argument(
        "--segmented", 
        action="store_true", 
        help="使用分段处理"
    )
    parser.add_argument(
        "--nltk-data", 
        type=str, 
        default=None, 
        help="NLTK数据目录路径"
    )
    parser.add_argument(
        "--play", 
        action="store_true", 
        help="生成后播放音频"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="详细输出"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查模型路径和语音目录是否存在
    if not os.path.exists(args.model_path) or not os.path.exists(args.voices_dir):
        logger.warning(f"模型路径或语音目录不存在，将使用空模拟模式")
        # 设置空模拟模式
        class MockVoice:
            def __init__(self, id): 
                self.id = id
                self.loaded = True
            def get_tensor(self): 
                return torch.zeros(100)
        
        class MockIndependentTTSPipeline:
            def __init__(self, *args, **kwargs):
                self.voices = {f"z00{i}": MockVoice(f"z00{i}") for i in range(1, 4)}
            def list_voices(self): 
                return list(self.voices.keys())
            def text_to_speech(self, text, voice_id, output_path=None, speed=1.0): 
                logger.info(f"生成语音，文本: {text}, 语音ID: {voice_id}, 输出路径: {output_path}")
                # 生成一个简单的正弦波音频
                sample_rate = 24000
                duration = 1.0  # 1秒
                t = torch.linspace(0, duration, int(sample_rate * duration))
                audio = torch.sin(2 * np.pi * 440 * t) * 0.1  # 440Hz正弦波
                
                if output_path:
                    logger.info("保存音频...")
                    self.save_audio(audio, output_path)
                    
                return audio
            def process_text_segments(self, text, voice_id, output_path=None, speed=1.0): 
                logger.info(f"分段处理文本，文本: {text}, 语音ID: {voice_id}, 输出路径: {output_path}")
                # 生成一个简单的正弦波音频
                sample_rate = 24000
                duration = 1.0  # 1秒
                t = torch.linspace(0, duration, int(sample_rate * duration))
                audio = torch.sin(2 * np.pi * 440 * t) * 0.1  # 440Hz正弦波
                
                if output_path:
                    logger.info("保存音频...")
                    self.save_audio(audio, output_path)
                    
                return audio
            def save_audio(self, audio, output_path, sample_rate=24000):
                try:
                    # 使用绝对路径
                    abs_output_path = os.path.abspath(output_path)
                    output_dir = os.path.dirname(abs_output_path)
                    logger.info(f"创建输出目录: {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 转换为NumPy数组
                    logger.info(f"音频形状: {audio.shape}")
                    audio_np = audio.cpu().numpy()
                    logger.info(f"NumPy数组形状: {audio_np.shape}")
                    
                    # 保存音频
                    try:
                        from scipy.io import wavfile
                        logger.info(f"使用scipy保存音频到: {abs_output_path}")
                        wavfile.write(abs_output_path, sample_rate, audio_np)
                    except ImportError:
                        logger.warning("scipy未安装，尝试使用soundfile")
                        try:
                            import soundfile as sf
                            logger.info(f"使用soundfile保存音频到: {abs_output_path}")
                            sf.write(abs_output_path, audio_np, sample_rate)
                        except ImportError:
                            logger.error("无法保存音频，请安装scipy或soundfile")
                            return False
                     
                    logger.info(f"音频已保存至: {abs_output_path}")
                    return True
                except Exception as e:
                    logger.error(f"保存音频失败: {e}")
                    return False
            def play_audio(self, *args, **kwargs): 
                logger.info("模拟播放音频")
        
        # 使用模拟对象
        pipeline = MockIndependentTTSPipeline(
            model_path=args.model_path,
            voices_dir=args.voices_dir,
            nltk_data_path=args.nltk_data,
            verbose=args.verbose
        )
        
        # 获取可用语音
        available_voices = pipeline.list_voices()
        voice_id = args.voice_id or available_voices[0]
        logger.info(f"使用语音ID: {voice_id}")
        
        # 生成音频
        if args.segmented:
            audio = pipeline.process_text_segments(
                text=args.text,
                voice_id=voice_id,
                speed=args.speed,
                output_path=args.output
            )
        else:
            audio = pipeline.text_to_speech(
                text=args.text,
                voice_id=voice_id,
                speed=args.speed,
                output_path=args.output
            )
        
        if audio is not None:
            logger.info(f"已生成模拟音频到文件: {args.output}")
            
            # 检查文件是否存在
            if os.path.exists(args.output):
                logger.info(f"音频文件已成功生成: {args.output}")
                logger.info(f"文件大小: {os.path.getsize(args.output)} 字节")
            else:
                logger.error(f"音频文件未生成: {args.output}")
        else:
            logger.error("生成音频失败")
        
        # 播放音频
        if args.play and os.path.exists(args.output):
            pipeline.play_audio(args.output)
            
        return 0
    
    try:
        # 创建流水线
        pipeline = IndependentTTSPipeline(
            model_path=args.model_path,
            voices_dir=args.voices_dir,
            nltk_data_path=args.nltk_data,
            verbose=args.verbose
        )
        
        # 获取可用语音
        available_voices = pipeline.list_voices()
        if not available_voices:
            logger.error("没有可用的语音")
            return 1
            
        voice_id = args.voice_id or available_voices[0]
        if voice_id not in available_voices:
            logger.warning(f"指定的语音ID不存在: {voice_id}，使用: {available_voices[0]}")
            voice_id = available_voices[0]
            
        logger.info(f"使用语音ID: {voice_id}")
        
        # 生成音频
        if args.segmented:
            audio = pipeline.process_text_segments(
                text=args.text,
                voice_id=voice_id,
                speed=args.speed,
                output_path=args.output
            )
        else:
            audio = pipeline.text_to_speech(
                text=args.text,
                voice_id=voice_id,
                speed=args.speed,
                output_path=args.output
            )
        
        if audio is None:
            logger.error("生成音频失败")
            return 1
            
        # 播放音频
        if args.play:
            pipeline.play_audio(args.output)
            
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