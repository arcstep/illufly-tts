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

from .g2p.chinese_g2p import ChineseG2P
from .g2p.english_g2p import EnglishG2P
from .normalization import ZhTextNormalizer, EnTextNormalizer
from kokoro.model import KModel  # 仅依赖KModel，不使用KPipeline

logger = logging.getLogger(__name__)

class TTSPipeline:
    """直接使用KModel的TTS流水线"""
    
    def __init__(
        self,
        repo_id: str,
        voices_dir: str,
        device: str = "cpu"
    ):
        """初始化TTS流水线
        
        Args:
            repo_id: 模型ID或路径
            voices_dir: 语音目录
            device: 设备名称
        """
        self.repo_id = repo_id
        self.voices_dir = voices_dir
        self.device = device
        self.sample_rate = 24000  # 采样率
        
        # 初始化英文G2P
        self.en_g2p = EnglishG2P()
        
        # 明确地创建回调函数引用
        self.en_callback = self.en_g2p.text_to_ipa
        
        # 将本地回调传递给中文G2P
        self.g2p = ChineseG2P(en_callable=self.en_callback)
        
        # 初始化文本规范化器
        self.zh_normalizer = ZhTextNormalizer()
        self.en_normalizer = EnTextNormalizer()
        
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

    def preprocess_text(self, text: str) -> str:
        """预处理文本，根据内容分别进行中英文规范化
        
        Args:
            text: 输入文本
            
        Returns:
            规范化后的文本
        """
        logger.info(f"开始文本预处理: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        segments = []
        chunks = []
        last_end = 0
        
        # 使用更健壮的方式分割文本
        pattern = re.compile(
            r'([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+)|'  # 中文字符
            r'([a-zA-Z]+(?:[\s\-\'"][a-zA-Z]+)*)|'  # 英文单词
            r'((?:气温)?-?\d+(?:\.\d+)?(?:°C|℃|度|摄氏度)?)|'  # 数字（包括温度相关文本）
            r'([\u2000-\u206F\u2E00-\u2E7F\'!"#$%&\(\)*+,\-.\/:;<=>?@\[\]^_`{|}~]+)'  # 标点符号
        )
        
        for match in pattern.finditer(text):
            # 处理未匹配文本
            if match.start() > last_end:
                unmatched = text[last_end:match.start()]
                if unmatched.strip():
                    chunks.append((None, unmatched))
            
            # 判断匹配类型
            if match.group(1):  # 中文
                chunks.append(('zh', match.group(1)))
            elif match.group(2):  # 英文
                chunks.append(('en', match.group(2)))
            elif match.group(3):  # 数字（可能包含温度单位）
                # 检查数字的上下文和是否包含温度单位
                number_text = match.group(3)
                has_temp_unit = any(unit in number_text for unit in ['°C', '℃', '度', '摄氏度', '气温'])
                
                prev_type = chunks[-1][0] if chunks else None
                next_char = text[match.end():match.end()+1]
                
                # 如果数字包含温度单位，或者前后有中文，按中文处理
                if has_temp_unit or (next_char and '\u4e00' <= next_char <= '\u9fff') or prev_type == 'zh':
                    chunks.append(('zh', number_text))
                else:
                    chunks.append(('en', number_text))
            else:  # 标点符号
                # 根据前后文判断标点符号归属
                prev_type = chunks[-1][0] if chunks else None
                chunks.append((prev_type or 'zh', match.group(4)))
            
            last_end = match.end()
        
        # 处理剩余文本
        if last_end < len(text):
            unmatched = text[last_end:]
            if unmatched.strip():
                chunks.append((None, unmatched))
        
        # 合并相邻的同类型块
        merged_chunks = []
        current_type = None
        current_text = ""
        
        for chunk_type, chunk_text in chunks:
            if chunk_type == current_type:
                current_text += chunk_text
            else:
                if current_text:
                    merged_chunks.append((current_type, current_text))
                current_type = chunk_type
                current_text = chunk_text
        
        if current_text:
            merged_chunks.append((current_type, current_text))
        
        # 应用规范化处理
        for chunk_type, chunk_text in merged_chunks:
            if chunk_type == 'zh':
                normalized = ''.join(self.zh_normalizer.normalize(chunk_text))
            else:
                normalized = self.en_normalizer.normalize(chunk_text)
            segments.append(normalized)
        
        result = ''.join(segments)
        logger.info(f"文本预处理完成: {result[:50]}{'...' if len(result) > 50 else ''}")
        return result

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
        # 1. 预处理文本（分别应用中英文规范化）
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

    def arpa_to_ipa(self, arpa_phonemes: str) -> str:
        """将ARPAbet音素转换为IPA格式
        
        Args:
            arpa_phonemes: ARPAbet格式的音素序列
            
        Returns:
            IPA格式的音素序列
        """
        # ARPAbet到IPA的映射
        arpa_to_ipa_map = {
            'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ',
            'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
            'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ',
            'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
            'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
            'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
            'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
            'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
            # 小写版本同样映射
            'aa': 'ɑ', 'ae': 'æ', 'ah': 'ʌ', 'ao': 'ɔ', 'aw': 'aʊ',
            # ... 其余映射 ...
        }
        
        # 处理ARPAbet音素
        words = arpa_phonemes.split()
        ipa_result = []
        
        for word in words:
            if word in arpa_to_ipa_map:
                ipa_result.append(arpa_to_ipa_map[word])
            else:
                # 保留原始标记
                ipa_result.append(word)
        
        return ''.join(ipa_result)
