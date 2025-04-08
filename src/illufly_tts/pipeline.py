#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS系统的完整流水线 - 直接使用KModel实现
"""

import os
import re
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Generator
import torch

from .preprocessing.segmenter import LanguageSegmenter
from .preprocessing.normalizer import TextNormalizer
from .g2p.chinese_g2p import ChineseG2P
from kokoro.model import KModel  # 仅依赖KModel，不使用KPipeline

logger = logging.getLogger(__name__)

class TTSPipeline:
    """直接使用KModel的TTS流水线"""
    
    def __init__(
        self,
        repo_id: str,
        voices_dir: str,
        device: str = "cpu",
        nltk_data_path: Optional[str] = None,
        segmenter_class=LanguageSegmenter,
        normalizer_class=TextNormalizer,
        g2p_class=ChineseG2P,
        **kwargs
    ):
        """初始化TTS流水线
        
        Args:
            repo_id: 模型ID或路径
            voices_dir: 语音目录
            device: 设备名称
            nltk_data_path: NLTK数据目录路径
            segmenter_class: 分割器类
            normalizer_class: 规范化器类
            g2p_class: G2P类
        """
        self.repo_id = repo_id
        self.voices_dir = voices_dir
        self.device = device
        self.nltk_data_path = nltk_data_path
        self.sample_rate = 24000  # 采样率
        
        # 初始化组件
        self.segmenter = segmenter_class()
        self.normalizer = normalizer_class()
        self.g2p = g2p_class()
        
        # 直接加载KModel
        logger.info(f"正在加载KModel (repo_id={repo_id})")
        self.model = KModel(repo_id=repo_id).to(device).eval()
        
        # 语音包字典
        self.voices = {}
        
        # 日志记录模型加载成功
        logger.info("TTSPipeline初始化完成")
    
    def load_voice(self, voice_id: str) -> torch.FloatTensor:
        """加载语音包
        
        Args:
            voice_id: 语音ID
            
        Returns:
            语音张量
        """
        if voice_id in self.voices:
            return self.voices[voice_id]
            
        # 构建语音包路径
        voice_path = os.path.join(self.voices_dir, f"{voice_id}.pt")
        if not os.path.exists(voice_path):
            voice_path = os.path.join(self.voices_dir, f"{voice_id}.pth")
            
        # 加载语音包
        if os.path.exists(voice_path):
            logger.info(f"从{voice_path}加载语音: {voice_id}")
            pack = torch.load(voice_path, map_location=self.device, weights_only=True)
            self.voices[voice_id] = pack
            return pack
        else:
            raise ValueError(f"找不到语音文件: {voice_path}")
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            预处理后的文本
        """
        # 规范化文本
        normalized_text = self.normalizer.normalize(text)
        logger.info(f"文本规范化完成: {text[:50]}{'...' if len(text) > 50 else ''}")
        return normalized_text
    
    def segment_text(self, text: str, max_len: int = 400) -> List[str]:
        """分割文本为多个段落
        
        Args:
            text: 输入文本
            max_len: 每段最大长度
            
        Returns:
            段落列表
        """
        # 首先尝试按句子分割
        sentences = re.split(r'([。！？.!?]+)', text)
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            # 添加标点符号（如果存在）
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
                
            if len(current_chunk) + len(sentence) <= max_len:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # 如果没有找到句子边界，则按字符分割
        if not chunks:
            chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
        
        return chunks
    
    def text_to_phonemes(self, text: str) -> str:
        """将文本转换为音素序列
        
        Args:
            text: 输入文本
            
        Returns:
            音素序列
        """
        phonemes = self.g2p.text_to_phonemes(text)
        logger.info(f"音素生成完成: {phonemes[:50]}{'...' if len(phonemes) > 50 else ''}")
        return phonemes
    
    def phonemes_to_ipa(self, phonemes: str) -> str:
        """将注音符号转换为IPA格式
        
        Args:
            phonemes: 注音符号音素序列
            
        Returns:
            IPA格式音素序列
        """
        ipa = self.g2p.convert_to_ipa(phonemes)
        logger.info(f"IPA转换完成: {ipa[:50]}{'...' if len(ipa) > 50 else ''}")
        return ipa
    
    def generate_from_phonemes(
        self,
        phonemes: str,
        voice_id: str,
        speed: float = 1.0
    ) -> torch.Tensor:
        """从音素直接生成音频 - 核心生成函数
        
        Args:
            phonemes: 音素序列（IPA格式）
            voice_id: 语音ID
            speed: 语速
            
        Returns:
            生成的音频张量
        """
        # 检查音素长度限制
        if len(phonemes) > 510:
            logger.warning(f"音素序列过长 ({len(phonemes)}), 截断至510字符")
            phonemes = phonemes[:510]
        
        # 加载语音包
        voice_pack = self.load_voice(voice_id)
        
        # 获取匹配长度的语音嵌入
        voice_embedding = voice_pack[len(phonemes)-1]
        
        # 直接调用KModel生成音频
        logger.info(f"使用KModel生成音频: {phonemes[:30]}...")
        with torch.no_grad():
            output = self.model(phonemes, voice_embedding, speed, return_output=True)
        
        return output.audio

    def process(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        output_path: Optional[str] = None,
        segment_text: bool = False
    ) -> torch.Tensor:
        """处理文本生成语音
        
        Args:
            text: 输入文本
            voice_id: 语音ID
            speed: 语速
            output_path: 输出文件路径
            segment_text: 是否分割文本
            
        Returns:
            生成的音频张量
        """
        # 1. 预处理文本
        normalized_text = self.preprocess_text(text)
        
        # 是否分割文本
        if segment_text:
            segments = self.segment_text(normalized_text)
            logger.info(f"文本已分割为{len(segments)}个段落")
            
            # 处理每个段落
            all_audio = []
            for i, segment in enumerate(segments):
                logger.info(f"处理段落 {i+1}/{len(segments)}")
                
                # 2. 转换为音素序列（注音格式）
                phonemes = self.text_to_phonemes(segment)
                
                # 3. 转换为IPA格式
                ipa_phonemes = self.phonemes_to_ipa(phonemes)
                
                # 4. 生成音频
                audio = self.generate_from_phonemes(ipa_phonemes, voice_id, speed)
                all_audio.append(audio)
            
            # 合并所有音频
            combined_audio = torch.cat(all_audio, dim=0)
            
            # 5. 保存音频
            if output_path:
                import torchaudio
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torchaudio.save(
                    str(output_path),  # 确保是字符串路径
                    combined_audio.unsqueeze(0), 
                    self.sample_rate
                )
                logger.info(f"音频已保存至: {output_path}")
            
            return combined_audio
        else:
            # 不分割，作为一个整体处理
            # 2. 转换为音素序列（注音格式）
            phonemes = self.text_to_phonemes(normalized_text)
            
            # 3. 转换为IPA格式
            ipa_phonemes = self.phonemes_to_ipa(phonemes)
            
            # 4. 生成音频
            audio = self.generate_from_phonemes(ipa_phonemes, voice_id, speed)
            
            # 5. 保存音频
            if output_path:
                import torchaudio
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torchaudio.save(
                    str(output_path),  # 确保是字符串路径
                    audio.unsqueeze(0), 
                    self.sample_rate
                )
                logger.info(f"音频已保存至: {output_path}")
            
            return audio
    
    def list_voices(self) -> List[str]:
        """列出可用的语音ID
        
        Returns:
            语音ID列表
        """
        if not os.path.exists(self.voices_dir):
            return []
            
        voice_files = []
        for file in os.listdir(self.voices_dir):
            if file.endswith('.pt') or file.endswith('.pth'):
                voice_id = os.path.splitext(file)[0]
                voice_files.append(voice_id)
                
        return voice_files
    
    def batch_process(
        self,
        texts: List[str],
        voice_id: str,
        speed: float = 1.0,
        output_dir: Optional[str] = None,
        output_prefix: str = "tts_output"
    ) -> List[torch.Tensor]:
        """批量处理多个文本
        
        Args:
            texts: 文本列表
            voice_id: 语音ID
            speed: 语速
            output_dir: 输出目录
            output_prefix: 输出文件前缀
            
        Returns:
            音频张量列表
        """
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"处理文本 {i+1}/{len(texts)}")
            
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"{output_prefix}_{i+1}.wav")
            
            audio = self.process(
                text=text,
                voice_id=voice_id,
                speed=speed,
                output_path=output_path
            )
            
            results.append(audio)
        
        return results
