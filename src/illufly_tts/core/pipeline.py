#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS系统的完整流水线 - 直接使用KModel实现
"""

import os
import re
import logging
import functools
from typing import Dict, List, Optional, Union, Tuple, Any, Generator
import torch
import time

from .g2p.chinese_g2p import ChineseG2P
from .g2p.english_g2p import EnglishG2P
from .normalization import ZhTextNormalizer, EnTextNormalizer
from .kmodel import BatchKModel  # 仅依赖KModel，不使用KPipeline

logger = logging.getLogger(__name__)

class TTSPipeline:
    """直接使用KModel的TTS流水线"""
    
    def __init__(
        self,
        repo_id: str,
        voices_dir: str = None,
        device: str = None,
        default_language: str = "zh"  # 新增默认语言参数
    ):
        """初始化TTS流水线
        
        Args:
            repo_id: 模型ID或路径
            voices_dir: 语音目录
            device: 设备名称
            default_language: 默认语言，用于处理无明确语言上下文的文本
        """
        self.repo_id = repo_id
        self.voices_dir = voices_dir
        self.device = device
        self.sample_rate = 24000  # 采样率
        self.default_language = default_language  # 存储默认语言设置
        
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
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"  # M系列芯片
            else:
                device = "cpu"
        self.device = device
        logger.info(f"正在加载KModel (repo_id={repo_id})")
        self.model = BatchKModel(repo_id=repo_id).to(device).eval()
        
        # 语音包字典
        self.voices = {}
        
        # 日志记录模型加载成功
        logger.info("TTSPipeline初始化完成")

    def load_voice(self, voice_id: str) -> torch.FloatTensor:
        """加载语音包"""
        if voice_id in self.voices:
            return self.voices[voice_id]
        
        # 搜索路径顺序
        search_paths = []
        
        # 1. 首先检查指定路径
        if self.voices_dir:
            search_paths.append((self.voices_dir, ".pt"))
        
        # 2. 检查HF缓存路径
        from huggingface_hub import snapshot_download, hf_hub_download
        try:
            # 尝试直接从HF下载/获取路径
            model_dir = snapshot_download(self.repo_id)
            search_paths.append((os.path.join(model_dir, "voices"), ".pt"))
        except:
            pass
        
        # 实际查找文件
        for base_path, ext in search_paths:
            voice_path = os.path.join(base_path, f"{voice_id}{ext}")
            if os.path.exists(voice_path):
                logger.info(f"从{voice_path}加载语音: {voice_id}")
                pack = torch.load(voice_path, map_location=self.device, weights_only=True)
                self.voices[voice_id] = pack
                return pack
        
        # 提供更有用的错误信息        
        raise ValueError(f"找不到语音文件: 已尝试路径 {[os.path.join(p, f'{voice_id}{e}') for p, e in search_paths]}")
        
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
        voice_id: str = "zf_001",
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
                number_text = match.group(3)
                has_temp_unit = any(unit in number_text for unit in ['°C', '℃', '度', '摄氏度', '气温'])
                
                prev_type = chunks[-1][0] if chunks else None
                prev_char = text[match.start()-1:match.start()] if match.start() > 0 else ""
                next_char = text[match.end():match.end()+1] if match.end() < len(text) else ""
                
                # 判断数字前面是否有货币符号
                is_after_currency = prev_char in ['￥', '¥', '$', '€', '£', '₽', '₹']
                
                # 检查上下文来判断数字应该使用哪种语言处理
                # 初始化语言变量为None
                language_context = None
                
                # 中文上下文线索
                if (has_temp_unit or 
                    (next_char and '\u4e00' <= next_char <= '\u9fff') or 
                    prev_type == 'zh' or
                    is_after_currency and prev_type == 'zh' or
                    (prev_char and '\u4e00' <= prev_char <= '\u9fff')):
                    language_context = 'zh'
                # 英文上下文线索
                elif prev_type == 'en' or (next_char.isalpha() and not '\u4e00' <= next_char <= '\u9fff'):
                    language_context = 'en'
                
                # 如果没有明确上下文，使用默认语言
                if language_context is None:
                    language_context = self.default_language
                    logger.debug(f"纯数字文本 '{number_text}' 无明确语言上下文，使用默认语言: {self.default_language}")
                
                chunks.append((language_context, number_text))
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
                # 确保英文片段以空格开始和结束
                normalized = self.en_normalizer.normalize(chunk_text)
                # 确保"for"和"ten"之间有空格
                normalized = re.sub(r'(\w+)(\d+|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)', r'\1 \2', normalized)
            
            # 检查是否需要添加空格
            if segments and chunk_type == 'en' and not normalized.startswith(' ') and not segments[-1].endswith(' '):
                segments.append(' ')
            
            segments.append(normalized)
        
        result = ''.join(segments)
        
        # 额外处理中文环境中的货币金额
        zh_currency_pattern = re.compile(r'([\u4e00-\u9fff])?([￥¥$€£₽₹])?\s*(\d+(?:\.\d+)?)([\u4e00-\u9fff])?')
        def normalize_zh_currency(match):
            prev_cn = match.group(1)
            currency = match.group(2)
            amount = match.group(3)
            next_cn = match.group(4)
            
            # 如果前后有中文字符或货币符号是中文，则使用中文规范化
            if (prev_cn or next_cn or currency in ['￥', '¥']) and amount:
                # 对金额应用中文规范化
                amount_zh = self.zh_normalizer.normalize(amount)
                if currency:
                    return f"{prev_cn or ''}{currency}{amount_zh}{next_cn or ''}"
                return f"{prev_cn or ''}{amount_zh}{next_cn or ''}"
            return match.group(0)
        
        result = zh_currency_pattern.sub(normalize_zh_currency, result)
        
        # 手动处理英文序数词日期
        month_pattern = re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(st|nd|rd|th)', re.IGNORECASE)
        
        def replace_ordinal_date(match):
            month = match.group(1)
            day = match.group(2)
            suffix = match.group(3)
            
            day_num = int(day)
            if day_num == 1:
                day_text = "first" 
            elif day_num == 2:
                day_text = "second"
            elif day_num == 3:
                day_text = "third" 
            elif day_num == 21:
                day_text = "twenty first"
            elif day_num == 22:
                day_text = "twenty second"
            elif day_num == 23:
                day_text = "twenty third"
            elif day_num == 31:
                day_text = "thirty first"
            else:
                from .normalization.en.chronology import verbalize_ordinal
                day_text = verbalize_ordinal(day_num)
            
            return f"{month} {day_text}"
        
        result = month_pattern.sub(replace_ordinal_date, result)
        
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

    async def async_batch_process_texts(self, texts, voice_ids, speeds=None):
        """异步包装器"""
        return self.batch_process_texts(texts, voice_ids, speeds)

    def batch_process_texts(
        self,
        texts: List[str],
        voice_ids: List[str],
        speeds: Optional[List[float]] = None,
    ) -> List[torch.Tensor]:
        """优化缓存的批量处理多条文本"""
        if speeds is None:
            speeds = [1.0] * len(texts)
        
        # 添加缓存状态日志
        logger.info(f"批处理开始: {len(texts)}个文本, 当前缓存命中率: {self.get_cache_stats().get('text_hit_rate', 0):.2f}")
        
        # 批处理重复文本检测
        unique_texts = set(texts)
        if len(unique_texts) < len(texts):
            logger.info(f"批处理中包含{len(texts)-len(unique_texts)}个重复文本，可充分利用缓存")
        
        # 去重文本和语音ID，减少重复处理
        unique_texts = {}
        unique_voice_ids = set()
        
        for i, text in enumerate(texts):
            unique_texts[text] = unique_texts.get(text, []) + [i]
            unique_voice_ids.add(voice_ids[i])
        
        # 预热缓存 - 批量加载语音包
        for voice_id in unique_voice_ids:
            self.load_voice(voice_id)
        
        # 批量预处理文本（利用缓存）
        normalized_texts = [self.preprocess_text(text) for text in texts]
        
        # 批量转换为音素
        phonemes_list = [self.text_to_phonemes(text) for text in normalized_texts]
        
        # 批量转换为IPA
        ipa_phonemes_list = [self.phonemes_to_ipa(phonemes) for phonemes in phonemes_list]
        
        # 批量准备语音嵌入（利用缓存）
        ref_embeddings = []
        for i, voice_id in enumerate(voice_ids):
            voice_pack = self.load_voice(voice_id)
            phonemes_len = len(ipa_phonemes_list[i])
            voice_embedding = voice_pack[min(phonemes_len-1, len(voice_pack)-1)]
            ref_embeddings.append(voice_embedding)
        
        ref_s_batch = torch.stack(ref_embeddings)
        
        # 使用BatchKModel批量生成音频
        with torch.no_grad():
            audio_outputs = self.model.forward_batch(
                ipa_phonemes_list,
                ref_s_batch,
                speeds
            )
        
        logger.info(f"批处理完成，新缓存命中率: {self.get_cache_stats().get('text_hit_rate', 0):.2f}")
        return audio_outputs
        
    def stream_batch_process(
        self,
        long_texts: List[str],
        voice_ids: List[str],
        speeds: Optional[List[float]] = None,
        chunk_size: int = 200
    ) -> Generator[List[torch.Tensor], None, None]:
        """流式批量处理长文本
        
        Args:
            long_texts: 长文本列表
            voice_ids: 对应的语音ID列表
            speeds: 对应的语速列表
            chunk_size: 文本分块大小
            
        Yields:
            每批次生成的音频列表
        """
        if speeds is None:
            speeds = [1.0] * len(long_texts)
            
        # 将长文本分割成多个块
        text_chunks_list = [self.segment_text(text, chunk_size) for text in long_texts]
        
        # 找出最大块数
        max_chunks = max(len(chunks) for chunks in text_chunks_list)
        
        # 按块处理，确保每个文本都有对应的块
        for i in range(max_chunks):
            current_chunks = []
            current_voice_ids = []
            current_speeds = []
            
            for text_idx, chunks in enumerate(text_chunks_list):
                if i < len(chunks):
                    current_chunks.append(chunks[i])
                    current_voice_ids.append(voice_ids[text_idx])
                    current_speeds.append(speeds[text_idx])
            
            if current_chunks:
                # 生成当前批次的音频
                batch_audios = self.batch_process_texts(
                    current_chunks,
                    current_voice_ids,
                    current_speeds
                )
                
                yield batch_audios

class CachedTTSPipeline(TTSPipeline):
    """优化的TTS流水线，使用缓存提高性能"""
    
    def __init__(
        self,
        repo_id: str,
        voices_dir: str = None,
        device: str = "cpu",
        default_language: str = "zh",  # 添加默认语言参数
        voice_cache_size: int = 32,    # 语音包缓存大小
        text_cache_size: int = 1024,   # 文本处理缓存大小
        phoneme_cache_size: int = 1024 # 音素处理缓存大小
    ):
        """初始化带缓存的TTS流水线
        
        Args:
            repo_id: 模型ID或路径
            voices_dir: 语音目录
            device: 设备名称
            default_language: 默认语言，用于处理无明确语言上下文的文本
            voice_cache_size: 语音包缓存大小
            text_cache_size: 文本处理缓存大小
            phoneme_cache_size: 音素处理缓存大小
        """
        super().__init__(repo_id, voices_dir, device, default_language)
        
        self._cache_dict = {}  # 使用字典实现自定义缓存
        self._audio_cache = {}  # 添加音频结果缓存
        
        # 缓存统计
        self.cache_stats = {
            "voice_hits": 0,
            "voice_misses": 0,
            "text_hits": 0,
            "text_misses": 0,
            "phoneme_hits": 0,
            "phoneme_misses": 0,
            "ipa_hits": 0,
            "ipa_misses": 0
        }
    
    def preprocess_text(self, text: str) -> str:
        """改进的文本预处理方法，使用透明缓存"""
        cache_key = f"text:{hash(text)}"
        if cache_key in self._cache_dict:
            logger.debug(f"缓存命中! preprocess_text: '{text[:20]}...'")
            self.cache_stats["text_hits"] += 1
            return self._cache_dict[cache_key]
        
        # 缓存未命中，调用父类方法
        start_time = time.time()
        result = super().preprocess_text(text)
        process_time = time.time() - start_time
        
        # 保存到缓存
        self._cache_dict[cache_key] = result
        self.cache_stats["text_misses"] += 1
        logger.debug(f"缓存未命中: preprocess_text: '{text[:20]}...', 处理耗时: {process_time:.3f}秒")
        
        return result
    
    def text_to_phonemes(self, text: str) -> str:
        """智能缓存版本的文本到音素转换"""
        # 使用动态字典缓存而不是lru_cache，更便于控制和观察
        cache_key = f"text_to_phonemes_{hash(text)}"
        if cache_key in self._cache_dict:
            logger.debug(f"缓存命中! text_to_phonemes({text[:20]}...)")
            self.cache_stats["phoneme_hits"] += 1
            return self._cache_dict[cache_key]
        else:
            result = super().text_to_phonemes(text)
            self._cache_dict[cache_key] = result
            self.cache_stats["phoneme_misses"] += 1
            logger.debug(f"缓存未命中: text_to_phonemes({text[:20]}...), 已保存")
            return result
    
    def phonemes_to_ipa(self, phonemes: str) -> str:
        """智能缓存版本的音素到IPA转换"""
        # 使用动态字典缓存而不是lru_cache，更便于控制和观察
        cache_key = f"phonemes_to_ipa_{hash(phonemes)}"
        if cache_key in self._cache_dict:
            logger.debug(f"缓存命中! phonemes_to_ipa({phonemes[:20]}...)")
            self.cache_stats["ipa_hits"] += 1
            return self._cache_dict[cache_key]
        else:
            result = super().phonemes_to_ipa(phonemes)
            self._cache_dict[cache_key] = result
            self.cache_stats["ipa_misses"] += 1
            logger.debug(f"缓存未命中: phonemes_to_ipa({phonemes[:20]}...), 已保存")
            return result
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        # 计算命中率
        for cache_type in ["voice", "text", "phoneme", "ipa"]:
            hits = self.cache_stats[f"{cache_type}_hits"]
            misses = self.cache_stats[f"{cache_type}_misses"]
            total = hits + misses
            if total > 0:
                self.cache_stats[f"{cache_type}_hit_rate"] = hits / total
            else:
                self.cache_stats[f"{cache_type}_hit_rate"] = 0
                
        return self.cache_stats
    
    def clear_caches(self):
        """清除所有缓存"""
        self._cache_dict.clear()

    def is_voice_loaded(self, voice_id):
        """检查语音是否已加载，不抛出异常"""
        try:
            # 检查缓存
            if voice_id in self.voices:
                return True
            
            # 如果未缓存，检查文件是否存在
            voice_path = self.get_voice_path(voice_id)
            return os.path.exists(voice_path)
        except:
            return False

    def batch_process_texts(self, texts, voice_ids, speeds=None):
        """重写批处理方法，添加音频结果缓存"""
        if speeds is None:
            speeds = [1.0] * len(texts)
            
        # 检查结果缓存
        results = []
        uncached_indices = []
        uncached_texts = []
        uncached_voice_ids = []
        uncached_speeds = []
        
        # 先检查哪些已缓存
        for i, (text, voice_id, speed) in enumerate(zip(texts, voice_ids, speeds)):
            cache_key = f"audio:{voice_id}:{speed}:{hash(text)}"
            if cache_key in self._audio_cache:
                logger.debug(f"音频缓存命中! text={text[:20]}...")
                results.append(self._audio_cache[cache_key])
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                uncached_voice_ids.append(voice_id)
                uncached_speeds.append(speed)
                results.append(None)  # 占位
                
        # 如果全部命中缓存，直接返回
        if not uncached_texts:
            logger.info(f"全部{len(texts)}个文本命中音频缓存，跳过模型推理")
            return results
            
        # 对未缓存的进行处理
        logger.info(f"处理{len(uncached_texts)}/{len(texts)}个未缓存文本")
        uncached_results = super().batch_process_texts(
            uncached_texts, uncached_voice_ids, uncached_speeds)
            
        # 填充结果并更新缓存
        for i, (text, voice_id, speed, audio) in enumerate(zip(
                uncached_texts, uncached_voice_ids, uncached_speeds, uncached_results)):
            orig_idx = uncached_indices[i]
            results[orig_idx] = audio
            
            # 更新缓存
            cache_key = f"audio:{voice_id}:{speed}:{hash(text)}"
            self._audio_cache[cache_key] = audio
            
        return results
