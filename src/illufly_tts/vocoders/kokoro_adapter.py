#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kokoro模型适配器 - 连接G2P和语音生成
"""

import os
import logging
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple, Callable, Any

try:
    from kokoro.model import KModel
    from kokoro.pipeline import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    logging.warning("Kokoro模块不可用，请安装kokoro包")

from ..g2p.base_g2p import BaseG2P
from ..g2p.mixed_g2p import MixedG2P

logger = logging.getLogger(__name__)

@dataclass
class KokoroVoice:
    """Kokoro语音配置"""
    
    id: str  # 语音ID
    path: str  # 语音文件路径
    tensor: Optional[torch.Tensor] = None  # 加载的语音张量
    language_code: str = "zh"  # 语言代码
    
    def __post_init__(self):
        self.loaded = False
        
    def load(self) -> bool:
        """加载语音向量
        
        Returns:
            是否成功加载
        """
        if self.loaded and self.tensor is not None:
            return True
            
        try:
            if os.path.exists(self.path):
                # 支持.pt和.npy格式
                if self.path.endswith('.pt'):
                    self.tensor = torch.load(self.path, weights_only=True)
                elif self.path.endswith('.npy'):
                    self.tensor = torch.from_numpy(np.load(self.path))
                else:
                    logger.error(f"不支持的语音文件格式: {self.path}")
                    return False
                    
                self.loaded = True
                logger.info(f"语音 {self.id} 已加载")
                return True
            else:
                logger.error(f"语音文件不存在: {self.path}")
                return False
        except Exception as e:
            logger.error(f"加载语音文件失败: {e}")
            return False
    
    def get_tensor(self, device: str = "cpu") -> Optional[torch.Tensor]:
        """获取语音张量
        
        Args:
            device: 设备名称
            
        Returns:
            语音张量，若未加载则返回None
        """
        if not self.loaded:
            self.load()
            
        if self.tensor is not None:
            return self.tensor.to(device)
        return None

class KokoroAdapter:
    """Kokoro模型适配器"""
    
    # 添加英文音素映射表
    ENGLISH_PHONEME_MAP = {
        'aa': 'a',
        'ae': 'ae',
        'ah': 'a',
        'ao': 'ao',
        'aw': 'au',
        'ay': 'ai',
        'eh': 'e',
        'er': 'er',
        'ey': 'ei',
        'ih': 'i',
        'iy': 'i',
        'ow': 'ou',
        'oy': 'oi',
        'uh': 'u',
        'uw': 'u',
        'hh': 'h',
        'r': 'r',
        'y': 'y',
        'w': 'w',
        'b': 'b',
        'd': 'd',
        'g': 'g',
        'p': 'p',
        't': 't',
        'k': 'k',
        'm': 'm',
        'n': 'n',
        'ng': 'ng',
        'f': 'f',
        'v': 'v',
        'th': 'th',
        'dh': 'dh',
        's': 's',
        'z': 'z',
        'sh': 'sh',
        'zh': 'zh',
        'ch': 'ch',
        'jh': 'j',
        'l': 'l'
    }

    def __init__(
        self,
        model_path: str,
        voices_dir: str,
        device: Optional[str] = None,
        g2p: Optional[BaseG2P] = None,
        repo_id: str = "hexgrad/Kokoro-82M-v1.1-zh"
    ):
        """初始化Kokoro适配器
        
        Args:
            model_path: 模型路径
            voices_dir: 语音目录
            device: 设备名称（默认自动选择）
            g2p: G2P转换器，默认使用MixedG2P
            repo_id: HuggingFace模型ID
        """
        self.model_path = model_path
        self.voices_dir = voices_dir
        self.repo_id = repo_id
        
        # 检查Kokoro是否可用
        if not KOKORO_AVAILABLE:
            logger.error("Kokoro模块不可用，适配器将无法工作")
            self.available = False
            return
            
        # 确定设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 加载G2P
        self.g2p = g2p or MixedG2P()
        
        # 设置英文处理回调
        def en_callable(text: str) -> str:
            logger.info(f"处理英文文本: {text}")
            try:
                # 获取原始音素
                phonemes = self.g2p.english_g2p.text_to_phonemes(text)
                logger.info(f"原始英文音素: {phonemes}")
                
                # 映射音素
                mapped_phonemes = []
                for phoneme in phonemes.split():
                    # 移除音调标记（数字）
                    base_phoneme = ''.join([c for c in phoneme if not c.isdigit()])
                    # 查找映射后的音素
                    mapped = self.ENGLISH_PHONEME_MAP.get(base_phoneme.lower(), base_phoneme)
                    mapped_phonemes.append(mapped)
                
                result = ' '.join(mapped_phonemes)
                logger.info(f"映射后的英文音素: {result}")
                return result
            except Exception as e:
                logger.error(f"英文处理失败: {e}")
                return text
        
        self.en_callable = en_callable
        
        # 初始化模型
        self.model = None
        self.pipeline = None
        self.voices = {}
        self.available = True
        
        # 加载模型
        self.load_model()
        
        # 加载语音列表
        self.load_voices()
    
    def load_model(self) -> bool:
        """加载Kokoro模型
        
        Returns:
            是否成功加载
        """
        if not self.available:
            return False
            
        try:
            logger.info(f"加载Kokoro模型: {self.model_path}")
            
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA不可用，将使用CPU")
                self.device = "cpu"
            
            # 修改KModel初始化方式，不传入model_path参数    
            self.model = KModel(repo_id=self.repo_id).to(self.device).eval()
            logger.info(f"Kokoro模型加载成功，使用设备: {self.device}")
            
            # 初始化中文Pipeline，添加英文处理回调
            self.pipeline = KPipeline(
                lang_code="z",
                model=self.model,
                repo_id=self.repo_id,
                en_callable=self.en_callable
            )
            
            return True
        except Exception as e:
            logger.error(f"加载Kokoro模型失败: {e}")
            self.available = False
            return False
    
    def load_voices(self) -> int:
        """加载语音列表
        
        Returns:
            加载的语音数量
        """
        if not self.available:
            return 0
            
        try:
            count = 0
            logger.info(f"从目录加载语音: {self.voices_dir}")
            
            # 确保目录存在
            if not os.path.exists(self.voices_dir):
                logger.error(f"语音目录不存在: {self.voices_dir}")
                return 0
                
            # 加载所有.pt和.npy文件
            for file in os.listdir(self.voices_dir):
                if file.endswith('.pt') or file.endswith('.npy'):
                    voice_id = file.split('.')[0]
                    voice_path = os.path.join(self.voices_dir, file)
                    
                    # 确定语言代码
                    lang_code = voice_id[0] if len(voice_id) > 0 else "z"
                    
                    # 创建语音对象
                    voice = KokoroVoice(id=voice_id, path=voice_path, language_code=lang_code)
                    self.voices[voice_id] = voice
                    count += 1
                    
            logger.info(f"加载了 {count} 个语音")
            return count
        except Exception as e:
            logger.error(f"加载语音列表失败: {e}")
            return 0
    
    def get_voice(self, voice_id: str) -> Optional[KokoroVoice]:
        """获取指定ID的语音
        
        Args:
            voice_id: 语音ID
            
        Returns:
            语音对象，若不存在则返回None
        """
        if voice_id in self.voices:
            return self.voices[voice_id]
        logger.warning(f"语音 {voice_id} 不存在")
        return None
    
    def list_voices(self) -> List[str]:
        """获取所有可用的语音ID列表
        
        Returns:
            语音ID列表
        """
        return list(self.voices.keys())
    
    def generate_audio(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        use_pipeline: bool = True
    ) -> Optional[torch.Tensor]:
        """生成语音音频
        
        Args:
            text: 输入文本
            voice_id: 语音ID
            speed: 语速
            use_pipeline: 是否使用官方Pipeline
            
        Returns:
            音频张量，若生成失败则返回None
        """
        if not self.available:
            logger.error("Kokoro适配器不可用")
            return None
            
        # 获取语音
        voice = self.get_voice(voice_id)
        if voice is None:
            return None
            
        # 确保语音已加载
        voice_tensor = voice.get_tensor(self.device)
        if voice_tensor is None:
            return None
            
        try:
            # 两种模式：使用官方Pipeline或自定义G2P
            if use_pipeline:
                logger.info(f"使用官方Pipeline处理文本: {text[:50]}{'...' if len(text) > 50 else ''}")
                
                # 使用官方Pipeline
                results = list(self.pipeline(text, voice_id, speed=speed))
                
                # 合并音频
                audio_parts = [result.audio for result in results if result.audio is not None]
                if not audio_parts:
                    logger.error("生成的音频为空")
                    return None
                    
                # 连接所有音频片段
                audio = torch.cat(audio_parts, dim=0)
                return audio
                
            else:
                logger.info(f"使用自定义G2P处理文本: {text[:50]}{'...' if len(text) > 50 else ''}")
                
                # 使用自定义G2P
                phonemes = self.g2p.text_to_phonemes(text)
                
                if not phonemes:
                    logger.error("生成的音素为空")
                    return None
                    
                # 截断音素序列，确保不超过模型限制
                if len(phonemes) > 510:
                    logger.warning(f"音素序列过长 ({len(phonemes)}), 截断至510个字符")
                    phonemes = phonemes[:510]
                    
                # 调用模型生成音频
                voice_vector = voice_tensor
                
                # 使用官方的infer方法
                output = KPipeline.infer(self.model, phonemes, voice_vector, speed)
                
                if output is None or output.audio is None:
                    logger.error("生成的音频为空")
                    return None
                    
                return output.audio
                
        except Exception as e:
            logger.error(f"生成音频失败: {e}")
            return None
    
    def save_audio(
        self,
        audio: torch.Tensor,
        output_path: str,
        sample_rate: int = 24000
    ) -> bool:
        """保存音频到文件
        
        Args:
            audio: 音频张量
            output_path: 输出文件路径
            sample_rate: 采样率
            
        Returns:
            是否成功保存
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 转换为NumPy数组
            audio_np = audio.cpu().numpy()
            
            # 使用scipy保存音频
            try:
                from scipy.io import wavfile
                wavfile.write(output_path, sample_rate, audio_np)
            except ImportError:
                logger.warning("scipy未安装，使用替代方法保存音频")
                try:
                    import soundfile as sf
                    sf.write(output_path, audio_np, sample_rate)
                except ImportError:
                    logger.error("无法保存音频，请安装scipy或soundfile")
                    return False
                    
            logger.info(f"音频已保存至: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            return False
    
    def process_text(
        self,
        text: str,
        voice_id: str,
        output_path: str,
        speed: float = 1.0,
        use_pipeline: bool = True
    ) -> bool:
        """处理文本并保存音频
        
        Args:
            text: 输入文本
            voice_id: 语音ID
            output_path: 输出文件路径
            speed: 语速
            use_pipeline: 是否使用官方Pipeline
            
        Returns:
            是否成功处理并保存
        """
        if not self.available:
            logger.error("Kokoro适配器不可用")
            return False
            
        # 生成音频
        audio = self.generate_audio(text, voice_id, speed, use_pipeline)
        if audio is None:
            return False
            
        # 保存音频
        return self.save_audio(audio, output_path)
    
    def is_available(self) -> bool:
        """检查适配器是否可用
        
        Returns:
            是否可用
        """
        return self.available and self.model is not None 