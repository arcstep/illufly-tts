#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自定义pipeline模块，不依赖espeak，专门用于中英文G2P处理
但保持与官方KPipeline大部分逻辑一致
"""

from dataclasses import dataclass
from typing import Callable, Generator, List, Optional, Tuple, Union, Dict, Any
import re
import torch
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# 检查是否在离线模式
OFFLINE_MODE = os.environ.get("KOKORO_OFFLINE", "0").lower() in ("1", "true", "yes", "y")

# 导入我们的中英文G2P模块
from .english_g2p import english_g2p, EnglishG2P
from .chinese_g2p import chinese_g2p, ChineseG2P

@dataclass
class MToken:
    """模拟misaki.en.MToken, 用于跟踪标记信息"""
    text: str
    phonemes: Optional[str] = None
    whitespace: str = ""
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None

@dataclass
class PipelineResult:
    """管道处理结果，与KPipeline.Result保持兼容"""
    graphemes: str
    phonemes: str
    tokens: Optional[List[MToken]] = None
    output: Optional[Any] = None
    text_index: Optional[int] = None

    @property
    def audio(self) -> Optional[torch.FloatTensor]:
        return None if self.output is None else self.output.audio

    @property
    def pred_dur(self) -> Optional[torch.LongTensor]:
        return None if self.output is None else self.output.pred_dur

    # 向后兼容
    def __iter__(self):
        yield self.graphemes
        yield self.phonemes
        yield self.audio

    def __getitem__(self, index):
        return [self.graphemes, self.phonemes, self.audio][index]

    def __len__(self):
        return 3

class CustomPipeline:
    """
    自定义Pipeline，完全不依赖espeak，但保持与KPipeline大部分逻辑一致
    
    主要功能：
    1. 处理中英文混合文本
    2. 将文本转换为音素序列
    3. 使用模型生成音频
    """
    def __init__(
        self,
        model = None,
        repo_id: Optional[str] = None,
        device: Optional[str] = None,
        en_callable: Optional[Callable] = None,
        zh_callable: Optional[Callable] = None
    ):
        """初始化自定义Pipeline
        
        Args:
            model: 语音合成模型实例
            repo_id: 模型仓库ID
            device: 计算设备（'cpu'或'cuda'）
            en_callable: 英文G2P处理函数
            zh_callable: 中文G2P处理函数
        """
        self.model = model
        self.repo_id = repo_id
        self.device = device
        self.voices = {}
        
        # 注册英文处理函数
        self.en_callable = en_callable or english_g2p
        
        # 初始化专用英文G2P处理器
        self.en_g2p = EnglishG2P()
        
        # 尝试使用misaki中文处理模块（官方实现）
        try:
            from misaki import zh
            logger.info("成功导入misaki.zh模块，使用官方中文G2P处理")
            # 使用misaki的ZHG2P，传入我们自己的英文处理函数以避免使用espeak
            # 注意：添加version参数与Kokoro官方实现保持一致
            self.zh_g2p = zh.ZHG2P(
                version='1.1',  # 对应Kokoro-82M-v1.1-zh模型
                en_callable=self.en_callable
            )
            # 不再使用zh_callable，直接在process_chinese中使用zh_g2p
            self.zh_callable = None
        except ImportError:
            # 如果无法导入misaki，回退到我们自己的实现
            logger.warning("未能导入misaki.zh模块，回退到自定义中文G2P处理")
            self.zh_g2p = ChineseG2P()
            self.zh_callable = zh_callable or chinese_g2p
        
        logger.info(f"初始化自定义Pipeline: repo_id={repo_id}, device={device}")
    
    def load_single_voice(self, voice: str):
        """加载单个语音ID对应的向量
        
        与KPipeline.load_single_voice保持一致
        """
        if voice in self.voices:
            return self.voices[voice]
            
        try:
            from huggingface_hub import hf_hub_download
            
            if voice.endswith('.pt'):
                f = voice
            else:
                f = hf_hub_download(repo_id=self.repo_id, filename=f'voices/{voice}.pt')
            
            pack = torch.load(f, weights_only=True)
            self.voices[voice] = pack
            return pack
            
        except Exception as e:
            logger.error(f"加载语音向量失败: {e}")
            
            # 创建一个随机向量作为备用，这部分与官方KPipeline不同，但必要时能提供替代方案
            logger.warning(f"使用随机向量替代{voice}")
            voice_tensor = torch.randn(1, 256)
            voice_tensor = voice_tensor / torch.norm(voice_tensor, dim=1, keepdim=True)
            if self.device:
                voice_tensor = voice_tensor.to(self.device)
            self.voices[voice] = voice_tensor
            return voice_tensor
    
    def load_voice(self, voice: Union[str, torch.FloatTensor], delimiter: str = ",") -> torch.FloatTensor:
        """加载语音参考向量
        
        与KPipeline.load_voice保持一致
        """
        if isinstance(voice, torch.FloatTensor):
            return voice
            
        if voice in self.voices:
            return self.voices[voice]
            
        # 完全复制官方KPipeline.load_voice的实现
        logger.debug(f"加载语音向量: {voice}")
        packs = [self.load_single_voice(v) for v in voice.split(delimiter)]
        
        if len(packs) == 1:
            return packs[0]
            
        # 平均多个语音向量
        self.voices[voice] = torch.mean(torch.stack(packs), dim=0)
        return self.voices[voice]
    
    def detect_language(self, text: str) -> str:
        """检测文本语言
        
        Args:
            text: 输入文本
            
        Returns:
            语言代码：'zh'（中文）或'en'（英文）
        """
        # 如果包含中文字符，判定为中文
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
            
        # 使用langid库进行更精确的语言检测
        try:
            import langid
            lang, _ = langid.classify(text)
            if lang == 'zh':
                return 'zh'
            else:
                return 'en'
        except ImportError:
            logger.warning("langid库未安装，使用简单规则判断语言")
            
        # 回退到简单规则：纯ASCII字符视为英文
        if all(ord(c) < 128 for c in text):
            return 'en'
            
        return 'zh'
    
    def process_english(self, text: str) -> Tuple[str, List[MToken]]:
        """处理英文文本，返回音素和标记列表
        
        Args:
            text: 英文文本
            
        Returns:
            (音素序列, 标记列表)
        """
        try:
            # 使用G2P处理器将文本拆分为标记
            tokens = []
            words = text.split()
            
            for i, word in enumerate(words):
                # 寻找标点符号和单词分界
                match = re.match(r'([a-zA-Z\'-]+)([^a-zA-Z\'-]*)', word)
                if match:
                    word_part, punct_part = match.groups()
                    # 处理单词部分
                    if word_part:
                        phonemes = self.en_callable(word_part)
                        whitespace = " " if i < len(words) - 1 or punct_part else ""
                        tokens.append(MToken(text=word_part, phonemes=phonemes, whitespace=whitespace))
                    # 处理标点部分
                    if punct_part:
                        whitespace = " " if i < len(words) - 1 else ""
                        tokens.append(MToken(text=punct_part, phonemes=punct_part, whitespace=whitespace))
                else:
                    # 如果没有匹配到模式，直接添加整个单词
                    whitespace = " " if i < len(words) - 1 else ""
                    tokens.append(MToken(text=word, phonemes=word, whitespace=whitespace))
            
            # 转换为音素序列
            phonemes = self.tokens_to_ps(tokens)
            return phonemes, tokens
            
        except Exception as e:
            logger.error(f"英文处理失败: {e}")
            # 创建一个基本的token
            token = MToken(text=text, phonemes=text)
            return text, [token]
    
    def process_chinese(self, text: str) -> Tuple[str, List[MToken]]:
        """处理中文文本，返回音素和标记列表
        
        Args:
            text: 中文文本
            
        Returns:
            (音素序列, 标记列表)
        """
        try:
            # 首先检查文本中是否含有英文单词
            # 如果有，将英文单词识别出来，并替换为特殊标记，避免被misaki处理
            english_placeholders = {}
            english_pattern = r'\b[a-zA-Z]+\b'
            
            # 查找所有英文单词并替换
            english_words = re.findall(english_pattern, text)
            if english_words:
                logger.info(f"混合文本中发现英文单词: {english_words}")
                # 为每个英文单词生成唯一标记
                for i, word in enumerate(english_words):
                    placeholder = f"__EN_WORD_{i}__"
                    english_placeholders[placeholder] = word
                    # 替换文本中的英文单词
                    text = re.sub(r'\b' + re.escape(word) + r'\b', placeholder, text, 1)
                logger.info(f"替换英文单词后的文本: {text}")
            
            # 添加文本标准化预处理步骤
            try:
                from misaki.zh_normalization.text_normalization import TextNormalizer
                normalizer = TextNormalizer()
                # 对文本进行标准化处理
                normalized_sentences = normalizer.normalize(text)
                normalized_text = "，".join(normalized_sentences)
                logger.info(f"文本标准化处理结果: '{normalized_text[:50]}{'...' if len(normalized_text) > 50 else ''}'")
                # 使用标准化后的文本
                text_to_process = normalized_text
            except ImportError:
                logger.warning("未能导入TextNormalizer，使用原始文本")
                text_to_process = text
            
            # 直接使用misaki处理整个文本
            logger.info(f"开始处理中文文本: '{text_to_process[:50]}{'...' if len(text_to_process) > 50 else ''}'")
            result = self.zh_g2p(text_to_process)
            logger.info(f"misaki.zh.ZHG2P返回类型: {type(result)}")
            
            # 输出结果内容用于调试
            if isinstance(result, tuple):
                logger.info(f"元组长度: {len(result)}")
                for i, item in enumerate(result):
                    logger.info(f"元组元素[{i}]类型: {type(item)}")
            
            # 根据返回类型处理
            if isinstance(result, str):
                # 如果只返回音素字符串
                phonemes = result
                # 创建一个简单的token
                tokens = [MToken(text=text_to_process, phonemes=phonemes)]
            elif isinstance(result, tuple) and len(result) > 0:
                # 取第一个元素作为音素
                phonemes = result[0]
                
                # 检查第二个元素是否存在且不为None
                if len(result) > 1 and result[1] is not None:
                    tokens_info = result[1]
                    # 从misaki的结果创建MToken对象
                    tokens = []
                    for token_info in tokens_info:
                        token = MToken(
                            text=token_info.text,
                            phonemes=token_info.phonemes,
                            whitespace=token_info.whitespace if hasattr(token_info, 'whitespace') else ""
                        )
                        tokens.append(token)
                else:
                    # 如果第二个元素不存在或为None，创建一个简单的token
                    tokens = [MToken(text=text_to_process, phonemes=phonemes)]
            else:
                # 未知返回格式，使用结果作为音素
                logger.warning(f"未知的misaki返回格式: {result}")
                phonemes = str(result)
                tokens = [MToken(text=text_to_process, phonemes=phonemes)]
            
            # 替换回英文单词对应的音素
            if english_placeholders:
                logger.info("替换占位符为英文音素")
                for placeholder, word in english_placeholders.items():
                    # 使用英文G2P处理单词
                    english_phoneme = self.en_callable(word)
                    logger.info(f"英文单词 '{word}' 音素: '{english_phoneme}'")
                    
                    # 查找占位符在音素序列中的位置
                    if placeholder in phonemes:
                        phonemes = phonemes.replace(placeholder, english_phoneme)
                    else:
                        # 如果找不到精确的占位符，可能是因为它被misaki拆分处理了
                        # 尝试查找和替换类似的部分
                        placeholder_pattern = re.compile(r'__EN_WORD_\d+__')
                        ph_matches = placeholder_pattern.findall(phonemes)
                        if ph_matches:
                            for ph in ph_matches:
                                phonemes = phonemes.replace(ph, english_phoneme)
            
            # 发音后处理步骤
            phonemes = self._post_process_phonemes(phonemes)
            
            logger.info(f"中文处理完成, 最终音素序列: '{phonemes[:50]}{'...' if len(phonemes) > 50 else ''}'")
            return phonemes, tokens
            
        except Exception as e:
            logger.error(f"中文处理失败: {e}")
            # 创建一个基本的token
            token = MToken(text=text, phonemes=text)
            return text, [token]
    
    def _post_process_phonemes(self, phonemes: str) -> str:
        """对音素序列进行后处理，确保格式正确
        
        Args:
            phonemes: 原始音素序列
            
        Returns:
            处理后的音素序列
        """
        # 检查并修复可能的格式问题
        # 1. 移除可能被错误处理的TONE标记
        phonemes = re.sub(r'TONE\d', '', phonemes)
        
        # 2. 确保空格正确
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        
        # 3. 修复可能的其他格式问题
        # ...根据需要添加其他修复步骤
        
        return phonemes
    
    @staticmethod
    def tokens_to_ps(tokens: List[MToken]) -> str:
        """从标记列表生成音素序列
        
        与KPipeline.tokens_to_ps保持一致
        """
        return ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens).strip()
    
    @staticmethod
    def tokens_to_text(tokens: List[MToken]) -> str:
        """从标记列表生成原始文本
        
        与KPipeline.tokens_to_text保持一致
        """
        return ''.join(t.text + t.whitespace for t in tokens).strip()
    
    @staticmethod
    def waterfall_last(
        tokens: List[MToken],
        next_count: int,
        waterfall: List[str] = ['!.?…', ':;', ',—'],
        bumps: List[str] = [')', '"']
    ) -> int:
        """根据标点符号查找最佳分割点
        
        与KPipeline.waterfall_last保持一致
        """
        for w in waterfall:
            z = next((i for i, t in reversed(list(enumerate(tokens))) if t.phonemes in set(w)), None)
            if z is None:
                continue
            z += 1
            if z < len(tokens) and tokens[z].phonemes in bumps:
                z += 1
            if next_count - len(CustomPipeline.tokens_to_ps(tokens[:z])) <= 510:
                return z
        return len(tokens)
    
    def en_tokenize(
        self,
        tokens: List[MToken]
    ) -> Generator[Tuple[str, str, List[MToken]], None, None]:
        """将标记列表分割为最大长度为510的块
        
        与KPipeline.en_tokenize保持一致
        """
        tks = []
        pcount = 0
        for t in tokens:
            t.phonemes = '' if t.phonemes is None else t.phonemes
            next_ps = t.phonemes + (' ' if t.whitespace else '')
            next_pcount = pcount + len(next_ps.rstrip())
            if next_pcount > 510:
                z = CustomPipeline.waterfall_last(tks, next_pcount)
                text = CustomPipeline.tokens_to_text(tks[:z])
                logger.debug(f"Chunking text at {z}: '{text[:30]}{'...' if len(text) > 30 else ''}'")
                ps = CustomPipeline.tokens_to_ps(tks[:z])
                yield text, ps, tks[:z]
                tks = tks[z:]
                pcount = len(CustomPipeline.tokens_to_ps(tks))
                if not tks:
                    next_ps = next_ps.lstrip()
            tks.append(t)
            pcount += len(next_ps)
        if tks:
            text = CustomPipeline.tokens_to_text(tks)
            ps = CustomPipeline.tokens_to_ps(tks)
            yield ''.join(text).strip(), ''.join(ps).strip(), tks
    
    @staticmethod
    def infer(
        model: Any,
        ps: str,
        pack: torch.FloatTensor,
        speed: Union[float, Callable[[int], float]] = 1
    ) -> Any:
        """使用模型生成音频
        
        与KPipeline.infer保持一致，但增加安全检查
        """
        # 计算速度
        if callable(speed):
            speed = speed(len(ps))
            
        # 打印调试信息
        logger.info(f"infer: ps长度={len(ps)}, pack形状={pack.shape}, pack维度={pack.dim()}")
        
        # 极度简化处理：我们只使用pack作为一个整体
        ref_s = pack
        
        # 如果是2维，取第一个元素
        if ref_s.dim() > 1 and ref_s.shape[0] > 0:
            logger.info(f"取第一个元素: {ref_s.shape[0]}")
            ref_s = ref_s[0]
        
        logger.info(f"最终ref_s形状={ref_s.shape}, 维度={ref_s.dim()}")
        
        # 使用KModel处理
        return model(ps, ref_s, speed, return_output=True)
    
    @staticmethod
    def join_timestamps(tokens: List[MToken], pred_dur: torch.LongTensor):
        """将时间戳添加到标记中
        
        与KPipeline.join_timestamps保持一致
        """
        # Multiply by 600 to go from pred_dur frames to sample_rate 24000
        # Equivalent to dividing pred_dur frames by 40 to get timestamp in seconds
        # We will count nice round half-frames, so the divisor is 80
        MAGIC_DIVISOR = 80
        if not tokens or len(pred_dur) < 3:
            # We expect at least 3: <bos>, token, <eos>
            return
        # We track 2 counts, measured in half-frames: (left, right)
        # This way we can cut space characters in half
        # TODO: Is -3 an appropriate offset?
        left = right = 2 * max(0, pred_dur[0].item() - 3)
        # Updates:
        # left = right + (2 * token_dur) + space_dur
        # right = left + space_dur
        i = 1
        for t in tokens:
            if i >= len(pred_dur)-1:
                break
            if not t.phonemes:
                if t.whitespace:
                    i += 1
                    left = right + pred_dur[i].item()
                    right = left + pred_dur[i].item()
                    i += 1
                continue
            j = i + len(t.phonemes)
            if j >= len(pred_dur):
                break
            t.start_ts = left / MAGIC_DIVISOR
            token_dur = pred_dur[i: j].sum().item()
            space_dur = pred_dur[j].item() if t.whitespace else 0
            left = right + (2 * token_dur) + space_dur
            t.end_ts = left / MAGIC_DIVISOR
            right = left + space_dur
            i = j + (1 if t.whitespace else 0)
    
    def generate_from_tokens(
        self,
        tokens: Union[str, List[MToken]],
        voice: str,
        speed: float = 1,
        model: Optional[Any] = None
    ) -> Generator[PipelineResult, None, None]:
        """从标记列表生成音频
        
        与KPipeline.generate_from_tokens保持一致
        """
        model = model or self.model
        if model and voice is None:
            raise ValueError('必须指定语音ID: pipeline.generate_from_tokens(..., voice="zf_001")')
        
        pack = self.load_voice(voice).to(model.device) if model else None

        # 处理原始音素字符串
        if isinstance(tokens, str):
            logger.debug("处理原始音素字符串")
            if len(tokens) > 510:
                raise ValueError(f'音素字符串过长: {len(tokens)} > 510')
            output = CustomPipeline.infer(model, tokens, pack, speed) if model else None
            yield PipelineResult(graphemes='', phonemes=tokens, output=output)
            return
        
        logger.debug("处理标记列表")
        # 处理标记列表
        for gs, ps, tks in self.en_tokenize(tokens):
            if not ps:
                continue
            elif len(ps) > 510:
                logger.warning(f"音素序列过长: {len(ps)} > 510，截断为510字符")
                ps = ps[:510]
            output = CustomPipeline.infer(model, ps, pack, speed) if model else None
            if output is not None and output.pred_dur is not None:
                CustomPipeline.join_timestamps(tks, output.pred_dur)
            yield PipelineResult(graphemes=gs, phonemes=ps, tokens=tks, output=output)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        voice: Optional[str] = None,
        speed: Union[float, Callable[[int], float]] = 1,
        split_pattern: Optional[str] = r'\n+',
        model: Optional[Any] = None
    ) -> Generator[PipelineResult, None, None]:
        """处理文本并生成语音
        
        与KPipeline.__call__保持一致
        """
        model = model or self.model
        if model and voice is None:
            raise ValueError('必须指定语音ID: pipeline(text="你好世界", voice="zf_001")')
        pack = self.load_voice(voice).to(model.device) if model else None
        
        # 将输入转换为段落列表
        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]
            
        # 处理每个段落
        for graphemes_index, graphemes in enumerate(text):
            if not graphemes.strip():  # 跳过空段落
                continue
            
            # 检测是否为纯英文文本
            is_pure_english = not re.search(r'[\u4e00-\u9fff]', graphemes)
            
            # 纯英文处理
            if is_pure_english:
                logger.debug(f"处理纯英文文本: {graphemes[:50]}{'...' if len(graphemes) > 50 else ''}")
                ps, tokens = self.process_english(graphemes)
                for gs, ps, tks in self.en_tokenize(tokens):
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logger.warning(f"音素序列过长: {len(ps)} > 510，截断为510字符")
                        ps = ps[:510]
                    output = CustomPipeline.infer(model, ps, pack, speed) if model else None
                    if output is not None and output.pred_dur is not None:
                        CustomPipeline.join_timestamps(tks, output.pred_dur)
                    yield PipelineResult(graphemes=gs, phonemes=ps, tokens=tks, output=output, text_index=graphemes_index)
            
            # 混合文本处理
            else:
                # 检查是否需要拆分句子（按标点符号）
                sentences = re.split(r'([.!?。！？]+)', graphemes)
                chunks = []
                
                # 将句子重组并保留标点
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    # 添加标点符号（如果存在）
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]
                    
                    if sentence.strip():
                        chunks.append(sentence.strip())
                
                # 如果没有分出任何句子，则将整个文本作为一个块
                if not chunks:
                    chunks = [graphemes]
                
                # 处理每个句子块
                for chunk_index, chunk in enumerate(chunks):
                    # 检测这个句子块是否为纯英文
                    is_chunk_english = not re.search(r'[\u4e00-\u9fff]', chunk)
                    
                    logger.debug(f"处理文本块 {chunk_index+1}/{len(chunks)}: "
                               f"{chunk[:30]}{'...' if len(chunk) > 30 else ''} "
                               f"[{'英文' if is_chunk_english else '中文/混合'}]")
                    
                    if is_chunk_english:
                        # 处理纯英文句子块
                        ps, tokens = self.process_english(chunk)
                    else:
                        # 处理中文或混合文本句子块
                        ps, tokens = self.process_chinese(chunk)
                    
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logger.warning(f'音素序列过长: {len(ps)} > 510，截断为510字符')
                        ps = ps[:510]
                    
                    output = CustomPipeline.infer(model, ps, pack, speed) if model else None
                    yield PipelineResult(graphemes=chunk, phonemes=ps, tokens=tokens, output=output, text_index=graphemes_index) 
    
    def predict(
        self,
        text: str,
        voice_samples: torch.FloatTensor, 
        speed: Union[float, Callable[[int], float]] = 1
    ) -> torch.FloatTensor:
        """预测音频，兼容KPipeline.predict接口
        
        Args:
            text: 输入文本
            voice_samples: 语音参考向量
            speed: 语速（默认为1）
            
        Returns:
            生成的音频张量
        """
        # 使用__call__方法生成音频
        results = list(self(text, voice=voice_samples, speed=speed))
        
        # 连接所有音频块
        if not results:
            return torch.zeros((1, 100), device=self.device)
            
        # 合并所有音频块
        audio_chunks = []
        for result in results:
            if result.audio is not None:
                audio_chunks.append(result.audio)
                
        if not audio_chunks:
            return torch.zeros((1, 100), device=self.device)
            
        # 连接所有音频块
        return torch.cat(audio_chunks, dim=1) 