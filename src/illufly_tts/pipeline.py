#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS系统的完整流水线
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import torch

from .preprocessing.segmenter import LanguageSegmenter
from .preprocessing.normalizer import TextNormalizer
from .g2p.mixed_g2p import MixedG2P
from .vocoders.kokoro_adapter import KokoroAdapter

logger = logging.getLogger(__name__)

class TTSPipeline:
    """TTS系统的基础流水线"""
    
    def __init__(
        self,
        model_path: str,
        voices_dir: str,
        device: str = "cpu",
        nltk_data_path: Optional[str] = None,
        segmenter_class=LanguageSegmenter,
        normalizer_class=TextNormalizer,
        g2p_class=MixedG2P,
        adapter_class=KokoroAdapter
    ):
        """初始化TTS流水线
        
        Args:
            model_path: 模型路径
            voices_dir: 语音目录
            device: 设备名称
            nltk_data_path: NLTK数据目录路径
            segmenter_class: 分割器类 (用于测试)
            normalizer_class: 规范化器类 (用于测试)
            g2p_class: G2P类 (用于测试)
            adapter_class: 适配器类 (用于测试)
        """
        self.model_path = model_path
        self.voices_dir = voices_dir
        self.device = device
        self.nltk_data_path = nltk_data_path
        self.sample_rate = 24000  # 添加采样率属性
        
        # 保存类引用用于初始化
        self._segmenter_class = segmenter_class
        self._normalizer_class = normalizer_class
        self._g2p_class = g2p_class
        self._adapter_class = adapter_class
        
        # 初始化组件
        self.segmenter = None
        self.normalizer = None
        self.g2p = None
        self.vocoder = None
        
        # 初始化语音列表
        self.voices = []
        
        # 初始化所有组件
        self._init_components()
    
    def _init_components(self):
        """初始化所有组件"""
        # 使用传入的类引用创建对象
        self.segmenter = self._segmenter_class()
        self.normalizer = self._normalizer_class()
        
        # G2P转换器
        self.g2p = self._g2p_class(
            nltk_data_path=self.nltk_data_path
        )
        
        # 语音合成器
        self.vocoder = self._adapter_class(
            model_path=self.model_path,
            voices_dir=self.voices_dir,
            g2p=self.g2p,
            device=self.device
        )
        
        # 从vocoder获取语音列表
        self.voices = self.vocoder.list_voices()
        
        logger.info("TTS流水线初始化完成")
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的文本
        """
        # 检查组件是否初始化
        if self.normalizer is None:
            logger.error("文本规范化器未初始化")
            return text
        
        # 规范化文本 - 确保调用normalize方法
        normalized_text = self.normalizer.normalize(text)
        logger.info(f"文本规范化完成: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        return normalized_text
    
    def text_to_phonemes(self, text: str) -> str:
        """将文本转换为音素序列
        
        Args:
            text: 输入文本
            
        Returns:
            音素序列
        """
        # 检查组件是否初始化
        if self.g2p is None:
            logger.error("G2P转换器未初始化")
            return ""
        
        # 转换为音素 - 确保调用text_to_phonemes方法
        phonemes = self.g2p.text_to_phonemes(text)
        logger.info(f"文本转换为音素完成: {phonemes[:50]}{'...' if len(phonemes) > 50 else ''}")
        
        return phonemes
    
    def phonemes_to_speech(
        self,
        phonemes: str,
        voice_id: str,
        speed: float = 1.0,
        use_official_pipeline: bool = True
    ) -> Optional[torch.Tensor]:
        """将音素序列转换为语音
        
        Args:
            phonemes: 音素序列
            voice_id: 语音ID
            speed: 语速
            use_official_pipeline: 是否使用官方Pipeline
            
        Returns:
            语音音频张量
        """
        # 检查组件是否初始化
        if self.vocoder is None:
            logger.error("语音合成器未初始化")
            return None
        
        # 检查语音是否可用
        if voice_id not in self.list_voices():
            logger.error(f"语音 {voice_id} 不可用")
            return None
        
        # 生成语音
        return self.vocoder.generate_audio(phonemes, voice_id, speed, use_pipeline=use_official_pipeline)
    
    def text_to_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Optional[str] = None,
        speed: float = 1.0,
        use_official_pipeline: bool = True
    ) -> Optional[torch.Tensor]:
        """将文本转换为语音
        
        Args:
            text: 输入文本
            voice_id: 语音ID
            output_path: 输出文件路径
            speed: 语速
            use_official_pipeline: 是否使用官方Pipeline
            
        Returns:
            语音音频张量
        """
        # 检查组件是否初始化
        if not all([self.normalizer, self.g2p, self.vocoder]):
            logger.error("流水线组件未完全初始化")
            return None
        
        # 检查语音是否可用
        if voice_id not in self.list_voices():
            logger.error(f"语音 {voice_id} 不可用")
            return None
        
        try:
            # 方式1：使用官方Pipeline
            if use_official_pipeline:
                logger.info(f"使用官方Pipeline处理文本: {text[:50]}{'...' if len(text) > 50 else ''}")
                # 直接调用vocoder的generate_audio方法
                audio = self.vocoder.generate_audio(text, voice_id, speed, use_pipeline=True)
                
                # 保存音频
                if output_path is not None and audio is not None:
                    self.vocoder.save_audio(audio, output_path, self.sample_rate)
                
                return audio
            
            # 方式2：使用自定义Pipeline
            logger.info(f"使用自定义Pipeline处理文本: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # 预处理文本
            normalized_text = self.preprocess_text(text)
            
            # 转换为音素
            phonemes = self.text_to_phonemes(normalized_text)
            
            # 生成语音
            audio = self.vocoder.generate_audio(text, voice_id, speed, use_pipeline=False)
            
            # 保存音频
            if output_path is not None and audio is not None:
                self.vocoder.save_audio(audio, output_path, self.sample_rate)
            
            return audio
            
        except Exception as e:
            logger.error(f"文本转语音失败: {e}")
            return None
    
    def list_voices(self) -> List[str]:
        """获取所有可用的语音ID列表
        
        Returns:
            语音ID列表
        """
        if self.vocoder is None:
            logger.error("语音合成器未初始化")
            return []
        
        # 从vocoder获取语音列表并更新self.voices
        self.voices = self.vocoder.list_voices()
        return self.voices


class MixedLanguagePipeline(TTSPipeline):
    """混合语言TTS流水线"""
    
    def __init__(
        self,
        model_path: str,
        voices_dir: str,
        device: str = "cpu",
        nltk_data_path: Optional[str] = None,
        language_threshold: float = 0.5,
        segmenter_class=LanguageSegmenter,
        normalizer_class=TextNormalizer,
        g2p_class=MixedG2P,
        adapter_class=KokoroAdapter
    ):
        """初始化混合语言TTS流水线
        
        Args:
            model_path: 模型路径
            voices_dir: 语音目录
            device: 设备名称
            nltk_data_path: NLTK数据目录路径
            language_threshold: 语言检测阈值
            segmenter_class: 分割器类 (用于测试)
            normalizer_class: 规范化器类 (用于测试)
            g2p_class: G2P类 (用于测试)
            adapter_class: 适配器类 (用于测试)
        """
        self.language_threshold = language_threshold
        super().__init__(
            model_path, 
            voices_dir, 
            device, 
            nltk_data_path,
            segmenter_class,
            normalizer_class,
            g2p_class,
            adapter_class
        )
    
    def text_to_speech(
        self,
        text: str,
        voice_id: str,
        output_path: Optional[str] = None,
        speed: float = 1.0,
        use_official_pipeline: bool = True
    ) -> Optional[torch.Tensor]:
        """将混合语言文本转换为语音
        
        Args:
            text: 输入文本
            voice_id: 语音ID
            output_path: 输出文件路径
            speed: 语速
            use_official_pipeline: 是否使用官方Pipeline
            
        Returns:
            语音音频张量
        """
        # 检查组件是否初始化
        if not all([self.segmenter, self.normalizer, self.g2p, self.vocoder]):
            logger.error("流水线组件未完全初始化")
            return None
        
        # 检查语音是否可用 - 更新检查方式
        available_voices = self.list_voices()
        if voice_id not in available_voices:
            logger.error(f"语音 {voice_id} 不可用")
            return None
        
        try:
            # 使用官方Pipeline
            if use_official_pipeline:
                return super().text_to_speech(
                    text, voice_id, output_path, speed, use_official_pipeline=True
                )
                
            # 使用自定义Pipeline，增加语言分段处理
            logger.info(f"使用混合语言Pipeline处理文本: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # 预处理文本
            normalized_text = self.preprocess_text(text)
            
            # 分段处理
            segments = self.segmenter.segment(normalized_text)
            logger.info(f"文本分段完成，共 {len(segments)} 个段落")
            
            # 逐段生成音频
            audio_parts = []
            for segment in segments:
                segment_text = segment["text"]
                segment_lang = segment["lang"]
                
                # 转换为音素
                segment_phonemes = self.g2p.text_to_phonemes(segment_text)
                
                # 选择与语言匹配的声音
                current_voice_id = voice_id
                if segment_lang == "en" and not voice_id.startswith("e"):
                    # 如果是英文段落但使用的不是英文声音，尝试找一个英文声音
                    english_voices = [v for v in self.list_voices() if v.startswith("e")]
                    if english_voices:
                        current_voice_id = english_voices[0]
                        logger.info(f"英文段落使用英文声音: {current_voice_id}")
                
                # 生成语音
                segment_audio = self.vocoder.generate_audio(
                    segment_text,
                    current_voice_id,
                    speed,
                    use_pipeline=False
                )
                
                if segment_audio is not None:
                    audio_parts.append(segment_audio)
            
            # 合并所有音频
            if not audio_parts:
                logger.error("没有生成任何音频段落")
                return None
            
            full_audio = torch.cat(audio_parts, dim=0)
            
            # 保存音频
            if output_path is not None:
                self.vocoder.save_audio(full_audio, output_path, self.sample_rate)
            
            return full_audio
                
        except Exception as e:
            logger.error(f"混合语言文本转语音失败: {e}")
            return None 