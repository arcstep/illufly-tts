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
import torchaudio
import json
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

try:
    from kokoro.model import KModel
    from kokoro.pipeline import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    logging.warning("Kokoro模块不可用，请安装kokoro包")

from ..g2p.base_g2p import BaseG2P
from ..g2p.mixed_g2p import MixedG2P
from ..g2p.custom_zh_g2p import CustomZHG2P

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
    """Kokoro TTS适配器"""
    
    def __init__(self,
                 repo_id: str = "hexgrad/Kokoro-82M", # 默认Hub ID
                 voices_dir: Optional[str] = None, # 可选，语音文件目录
                 g2p: Optional[BaseG2P] = None,
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None, # 添加cache_dir作为repo_id的别名
                 use_zhuyin: bool = True,         # 是否使用注音符号
                 use_special_tones: bool = True): # 是否使用特殊声调符号
        """初始化 Kokoro 适配器

        Args:
            repo_id: Hugging Face Hub 模型 ID 或本地缓存格式路径。
            voices_dir: 可选，语音文件目录。如果为 None，则假定在模型目录下。
            g2p: G2P 处理器实例
            device: 设备 ('cuda' or 'cpu')
            cache_dir: HuggingFace 缓存目录，可以指向本地模型文件夹
            use_zhuyin: 是否使用注音符号
            use_special_tones: 是否使用特殊声调符号
        """
        super().__init__()

        # 如果提供了cache_dir，优先使用它
        if cache_dir is not None:
            repo_id = cache_dir

        # 设置离线模式可能会减少网络问题
        try:
            if os.environ.get("HF_HUB_OFFLINE") != "1":
                logger.info("设置HF_HUB_OFFLINE=1来减少网络问题")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # 禁用HF特殊传输
        except:
            pass

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        logger.info(f"KokoroAdapter 使用设备: {self.device}")

        self.repo_id = repo_id
        self.voices_dir = voices_dir
        self.use_zhuyin = use_zhuyin
        self.use_special_tones = use_special_tones
        
        # 初始化G2P处理器
        if g2p is None:
            # 默认使用支持注音符号和特殊声调的CustomZHG2P
            self.g2p = MixedG2P(
                use_custom_zh_g2p=True, 
                use_zhuyin=use_zhuyin, 
                use_special_tones=use_special_tones
            )
            logger.info(f"使用默认的MixedG2P处理器 (use_zhuyin={use_zhuyin}, use_special_tones={use_special_tones})")
        else:
            self.g2p = g2p
            
        self._pipelines: Dict[str, KPipeline] = {}

        logger.info(f"准备加载 KModel (repo_id={self.repo_id})")
        try:
            # 判断是否是本地路径
            if os.path.exists(self.repo_id):
                # 加载本地模型
                logger.info(f"检测到本地模型路径: {self.repo_id}")
                
                # 查找本地模型文件
                model_file = None
                config_file = None
                
                # 查找可能的模型文件
                for file in os.listdir(self.repo_id):
                    if file.endswith('.pth') or file.endswith('.pt'):
                        model_file = os.path.join(self.repo_id, file)
                    elif file == 'config.json':
                        config_file = os.path.join(self.repo_id, file)
                
                # 确保找到了必要的文件
                if not model_file or not config_file:
                    raise ValueError(f"在目录 {self.repo_id} 中未找到模型文件(.pth/.pt)或配置文件(config.json)")
                
                # 加载本地配置文件
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # 尝试实例化模型 - 以最直接的方式创建模型
                try:
                    # 关闭HuggingFace Hub离线模式
                    original_offline = os.environ.get('HF_HUB_OFFLINE')
                    if original_offline:
                        logger.info(f"暂时关闭HF_HUB_OFFLINE: {original_offline}")
                        os.environ.pop('HF_HUB_OFFLINE')
                    
                    # 导入必要的模块
                    from kokoro.modules import Encoder, Decoder, Postnet, VarianceAdaptor
                    
                    # 使用配置创建模型组件
                    model = KModel(repo_id=self.repo_id)
                    
                    # 加载模型权重
                    logger.info(f"从文件加载模型权重: {model_file}")
                    state_dict = torch.load(model_file, map_location=self.device)
                    model.load_state_dict(state_dict)
                    self.model = model.to(self.device).eval()
                    
                    # 恢复原来的环境变量
                    if original_offline:
                        os.environ['HF_HUB_OFFLINE'] = original_offline
                    
                    logger.info("模型加载成功")
                    
                except Exception as e:
                    if original_offline:
                        os.environ['HF_HUB_OFFLINE'] = original_offline
                    logger.error(f"加载模型失败: {e}")
                    raise ValueError(f"无法加载模型: {e}")
                    
                # 设置语音目录
                if not self.voices_dir:
                    voices_dir_candidate = os.path.join(self.repo_id, 'voices')
                    if os.path.exists(voices_dir_candidate) and os.path.isdir(voices_dir_candidate):
                        self.voices_dir = voices_dir_candidate
                        logger.info(f"使用模型目录中的语音文件夹: {self.voices_dir}")
                
                logger.info(f"已从本地加载模型: {model_file}")
            else:
                # 使用repo_id从HuggingFace Hub加载
                self.model = KModel(repo_id=self.repo_id).to(self.device).eval()
                logger.info("已从HuggingFace Hub加载模型")
        except Exception as e:
            logger.error(f"无法加载 KModel (repo_id={self.repo_id}): {e}", exc_info=True)
            logger.error("请确保模型路径正确或HF_HUB_CACHE设置正确，且本地缓存结构符合预期。")
            raise ValueError(f"无法初始化 KModel (repo_id={self.repo_id})") from e
    
    def _get_pipeline(self, voice_id: str) -> Optional[KPipeline]:
        """获取或创建Pipeline
        
        Args:
            voice_id: 语音ID
            
        Returns:
            KPipeline实例
        """
        try:
            if voice_id not in self._pipelines:
                # 获取语言代码前缀
                lang_prefix = voice_id.split('_')[0] if '_' in voice_id else voice_id[:1]
                
                # 直接根据前缀判断支持的语言代码
                if lang_prefix == 'zf' or lang_prefix == 'z':
                    lang_code = 'z'
                elif lang_prefix == 'a':
                    lang_code = 'a'
                elif lang_prefix == 'e':
                    lang_code = 'e'
                else:
                    # 对于不支持的前缀，抛出错误
                    raise ValueError(f"不支持的 voice_id 前缀 '{lang_prefix}' (来自 '{voice_id}'). 只支持 z/zf, a, e.")

                logger.info(f"为语音 '{voice_id}' (lang={lang_code}) 创建 KPipeline")
                
                # 创建Pipeline
                try:
                    if os.path.exists(self.repo_id):
                        # 本地模型路径
                        self._pipelines[voice_id] = KPipeline(
                            lang_code=lang_code,
                            model=self.model,  # 传递预加载的模型
                            en_callable=self._get_en_callable()
                        )
                    else:
                        # HuggingFace Hub模型ID
                        self._pipelines[voice_id] = KPipeline(
                            lang_code=lang_code,
                            repo_id=self.repo_id,  # 传递repo_id用于加载模型
                            model=self.model,  # 传递预加载的模型
                            en_callable=self._get_en_callable()
                        )
                except Exception as e:
                    logger.error(f"创建Pipeline失败: {e}")
                    return None
            
            return self._pipelines[voice_id]
            
        except Exception as e:
            logger.error(f"获取Pipeline时出错: {e}")
            return None
    
    def _get_en_callable(self):
        """获取英文处理回调函数
        
        Returns:
            英文处理回调函数
        """
        def en_callable(text: str) -> str:
            try:
                # 直接使用我们的G2P处理英文
                if hasattr(self.g2p, 'process_english'):
                    return self.g2p.process_english(text)
                else:
                    # 否则，尝试标准的text_to_phonemes方法
                    return self.g2p.text_to_phonemes(text)
            except Exception as e:
                logger.error(f"英文处理失败: {e}")
                return text
        return en_callable
        
    def _generate_with_official_pipeline(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0
    ) -> Optional[torch.Tensor]:
        """使用官方KPipeline生成语音
        
        Args:
            text: 输入文本
            voice_id: 语音ID
            speed: 语速
            
        Returns:
            生成的音频张量
        """
        try:
            logger.info(f"使用官方Pipeline生成音频，文本: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # 获取或创建Pipeline
            pipeline = self._get_pipeline(voice_id)
            if pipeline is None:
                logger.error(f"无法获取Pipeline，voice_id={voice_id}")
                return None
                
            # 获取语音对象
            voice = self.get_voice(voice_id)
            if voice is None:
                logger.error(f"未找到语音: {voice_id}")
                return None
            
            # 确保语音张量已加载
            if voice.tensor is None:
                logger.error(f"语音张量未加载: {voice_id}")
                return None
                
            # 使用Pipeline生成音频 - 修复：KPipeline是一个生成器，需要迭代处理
            logger.info("开始生成...")
            
            # 收集所有音频片段
            audio_parts = []
            for result in pipeline(text, voice=voice.tensor, speed=speed):
                # KPipeline.Result对象有audio属性
                if result.audio is not None:
                    audio_parts.append(result.audio)
            
            if not audio_parts:
                logger.error("Pipeline未返回有效音频")
                return None
                
            # 合并所有音频片段
            audio = torch.cat(audio_parts, dim=0)
            
            return audio
        except Exception as e:
            logger.error(f"官方Pipeline生成失败: {e}")
            return None
    
    def _generate_with_custom_pipeline(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        is_phonemes: bool = False
    ) -> Optional[torch.Tensor]:
        """使用自定义流程生成音频

        Args:
            text: 输入文本或音素序列
            voice_id: 语音ID
            speed: 语速 (1.0为正常速度)
            is_phonemes: 输入是否已经是音素序列

        Returns:
            生成的音频张量，失败时返回None
        """
        try:
            logger.info(f"使用自定义处理流程生成音频，{'音素' if is_phonemes else '文本'}: {text[:30]}{'...' if len(text) > 30 else ''}")
            
            # 获取语音
            voice = self.get_voice(voice_id)
            if not voice:
                logger.error(f"找不到语音: {voice_id}")
                return None
            
            # 加载语音张量
            voice_tensor = voice.get_tensor(self.device)
            if voice_tensor is None:
                logger.error(f"加载语音张量失败: {voice_id}")
                return None
            
            logger.info(f"已加载语音: {voice_id} 从 {voice.path}")
            logger.debug(f"语音张量形状: {voice_tensor.shape}")
            
            # 如果输入是文本（非音素），则需要转换为音素
            phonemes = text
            if not is_phonemes:
                # 将文本转换为音素
                logger.info("将文本转换为音素...")
                phonemes = self.g2p.text_to_phonemes(text)
                logger.info(f"转换后的音素: {phonemes[:50]}{'...' if len(phonemes) > 50 else ''}")
            else:
                logger.info(f"直接使用输入的音素序列: {phonemes[:50]}{'...' if len(phonemes) > 50 else ''}")
            
            # 检查音素是否为空
            if not phonemes or not phonemes.strip():
                logger.error("音素序列为空，无法生成音频")
                return None
                        
            # 尝试使用官方管道方式
            logger.info("尝试使用官方KPipeline生成音频")
            
            # 获取对应语言的管道
            pipeline = self._get_pipeline(voice_id)
            if pipeline is None:
                logger.error(f"无法为语音 {voice_id} 创建管道")
                return None
            
            try:
                # 使用官方管道处理
                logger.debug(f"使用管道处理音素: '{phonemes}'")
                
                audio_segments = []
                results = list(pipeline(phonemes, voice=voice_id, speed=speed))
                
                if not results:
                    logger.error("管道未生成任何结果")
                    return None
                    
                logger.debug(f"管道生成了 {len(results)} 个结果段")
                
                for i, result in enumerate(results):
                    if hasattr(result, 'audio') and result.audio is not None:
                        audio_segments.append(result.audio)
                        logger.debug(f"段落 {i+1} 生成成功，音频形状: {result.audio.shape}")
                    else:
                        logger.warning(f"段落 {i+1} 没有生成音频")
                
                # 合并所有段落的音频
                if not audio_segments:
                    logger.error("未生成任何音频")
                    return None
                    
                # 合并音频段落
                logger.info(f"合并 {len(audio_segments)} 个音频段落")
                final_audio = torch.cat(audio_segments, dim=0)
                logger.debug(f"最终音频形状: {final_audio.shape}")
                
                return final_audio
                
            except Exception as e:
                logger.error(f"使用官方管道生成失败: {str(e)}", exc_info=True)
                logger.info("尝试使用直接模型调用方式...")
                
                # 如果官方管道失败，尝试直接调用模型
                if not hasattr(self, 'model') or self.model is None:
                    self._initialize_model()
                    
                if self.model is None:
                    logger.error("模型初始化失败")
                    return None
                
                # 分段处理
                segments = phonemes.split(".")
                segments = [s.strip() for s in segments if s.strip()]
                
                # 如果没有分段，则整体作为一段
                if not segments:
                    segments = [phonemes.strip()]
                
                logger.info(f"分段后的音素: {len(segments)} 段")
                
                # 生成每个段落的音频
                audio_segments = []
                
                for i, segment in enumerate(segments):
                    try:
                        logger.info(f"处理第 {i+1}/{len(segments)} 段...")
                        
                        # 确保音素不为空
                        if not segment.strip():
                            continue
                        
                        # 检查音素长度是否超过模型限制
                        tokens = segment.split()
                        if len(tokens) > 510:  # Kokoro模型限制
                            segment = " ".join(tokens[:510])
                        
                        # 尝试使用不同的语音包索引策略
                        try:
                            # 策略1: 使用简单索引
                            audio = self._generate_segment_with_simple_index(segment, voice_tensor, speed)
                            if audio is not None:
                                audio_segments.append(audio)
                                continue
                                
                            # 策略2: 使用语音索引对应音素长度
                            audio = self._generate_segment_with_length_match(segment, voice_tensor, speed)
                            if audio is not None:
                                audio_segments.append(audio)
                                continue
                                
                            # 策略3: 不使用索引，直接传递整个语音包
                            audio = self._generate_segment_with_full_pack(segment, voice_tensor, speed)
                            if audio is not None:
                                audio_segments.append(audio)
                                continue
                                
                            logger.error(f"段落 {i+1} 所有生成策略均失败")
                            
                        except Exception as seg_error:
                            logger.error(f"段落 {i+1} 生成失败: {str(seg_error)}", exc_info=True)
                    except Exception as e:
                        logger.error(f"段落 {i+1} 处理异常: {str(e)}", exc_info=True)
                
                # 合并所有段落的音频
                if not audio_segments:
                    logger.error("未生成任何音频")
                    return None
                    
                # 合并音频段落
                final_audio = torch.cat(audio_segments, dim=0)
                return final_audio
        
        except Exception as e:
            logger.error(f"生成音频失败: {str(e)}", exc_info=True)
            return None
    
    def _generate_segment_with_simple_index(self, segment: str, voice_tensor: torch.Tensor, speed: float) -> Optional[torch.Tensor]:
        """使用简单索引策略生成音频段落
        
        Args:
            segment: 音素段落
            voice_tensor: 语音张量
            speed: 语速
            
        Returns:
            音频段落，失败时返回None
        """
        try:
            logger.debug("尝试使用简单索引策略生成音频")
            
            # 使用固定索引0
            voice_pack = voice_tensor[0]
            logger.debug(f"使用语音包索引 0, 形状: {voice_pack.shape}")
            
            output = self.model(segment, voice_pack, speed, return_output=True)
            
            if output is not None and output.audio is not None:
                logger.debug(f"简单索引策略生成成功，音频形状: {output.audio.shape}")
                return output.audio
            return None
        except Exception as e:
            logger.debug(f"简单索引策略失败: {str(e)}")
            return None

    def _generate_segment_with_length_match(self, segment: str, voice_tensor: torch.Tensor, speed: float) -> Optional[torch.Tensor]:
        """使用长度匹配策略生成音频段落
        
        Args:
            segment: 音素段落
            voice_tensor: 语音张量
            speed: 语速
            
        Returns:
            音频段落，失败时返回None
        """
        try:
            logger.debug("尝试使用长度匹配策略生成音频")
            
            # 使用音素长度作为索引（但不超过语音张量的最大索引）
            seg_len = len(segment.split())
            index = min(seg_len, voice_tensor.shape[0] - 1)
            
            logger.debug(f"音素长度: {seg_len}, 使用语音包索引: {index}")
            voice_pack = voice_tensor[index]
            
            output = self.model(segment, voice_pack, speed, return_output=True)
            
            if output is not None and output.audio is not None:
                logger.debug(f"长度匹配策略生成成功，音频形状: {output.audio.shape}")
                return output.audio
            return None
        except Exception as e:
            logger.debug(f"长度匹配策略失败: {str(e)}")
            return None

    def _generate_segment_with_full_pack(self, segment: str, voice_tensor: torch.Tensor, speed: float) -> Optional[torch.Tensor]:
        """使用整个语音包生成音频段落
        
        Args:
            segment: 音素段落
            voice_tensor: 语音张量
            speed: 语速
            
        Returns:
            音频段落，失败时返回None
        """
        try:
            logger.debug("尝试使用整个语音包生成音频")
            
            # 使用最后一个索引，这通常是语音特征的汇总
            index = voice_tensor.shape[0] - 1
            voice_pack = voice_tensor[index]
            logger.debug(f"使用最后一个语音包索引: {index}")
            
            output = self.model(segment, voice_pack, speed, return_output=True)
            
            if output is not None and output.audio is not None:
                logger.debug(f"整个语音包策略生成成功，音频形状: {output.audio.shape}")
                return output.audio
            return None
        except Exception as e:
            logger.debug(f"整个语音包策略失败: {str(e)}")
            return None
    
    def generate_audio(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        use_pipeline: bool = True,
        is_phonemes: bool = False
    ) -> Optional[torch.Tensor]:
        """生成语音
        
        Args:
            text: 输入文本或音素序列
            voice_id: 语音ID
            speed: 语速
            use_pipeline: 是否使用官方Pipeline
            is_phonemes: 输入是否已经是音素序列
            
        Returns:
            语音音频张量
        """
        try:
            if use_pipeline:
                return self._generate_with_official_pipeline(text, voice_id, speed)
            else:
                return self._generate_with_custom_pipeline(text, voice_id, speed, is_phonemes=is_phonemes)
                
        except Exception as e:
            logger.error(f"生成语音失败: {e}")
            return None
    
    def list_voices(self) -> List[str]:
        """列出可用的语音ID
        
        Returns:
            可用语音ID列表
        """
        # 检查模型和语音目录
        if not hasattr(self, 'model') or self.model is None:
            logger.error("模型未初始化，无法列出语音")
            return []
        
        # 如果有语音目录，查找语音文件
        if self.voices_dir and os.path.exists(self.voices_dir):
            voice_files = []
            
            # 查找语音文件
            for file in os.listdir(self.voices_dir):
                if file.endswith('.pt') or file.endswith('.pth'):
                    # 去掉扩展名作为语音ID
                    voice_id = os.path.splitext(file)[0]
                    voice_files.append(voice_id)
            
            if voice_files:
                logger.info(f"从目录 {self.voices_dir} 中找到 {len(voice_files)} 个语音: {', '.join(voice_files)}")
                return voice_files
        
        # 如果没有找到语音文件，返回默认语音
        logger.warning(f"未找到语音文件，使用默认语音ID")
        return ["zf_001"]
    
    def save_audio(
        self,
        audio: torch.Tensor,
        output_path: str,
        sample_rate: int = 24000
    ) -> bool:
        """保存音频文件
        
        Args:
            audio: 音频张量
            output_path: 输出文件路径
            sample_rate: 采样率
            
        Returns:
            是否保存成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存音频
            torchaudio.save(output_path, audio.unsqueeze(0), sample_rate)
            logger.info(f"音频已保存到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            return False
    
    def process_text(
        self,
        text: str,
        voice_id: str,
        output_path: Optional[str] = None,
        speed: float = 1.0,
        use_pipeline: bool = True
    ) -> bool:
        """处理文本并生成语音
        
        Args:
            text: 输入文本
            voice_id: 语音ID
            output_path: 输出文件路径
            speed: 语速
            use_pipeline: 是否使用官方Pipeline
            
        Returns:
            是否成功处理并保存
        """
        # 生成音频
        audio = self.generate_audio(text, voice_id, speed, use_pipeline)
        if audio is None:
            return False
            
        # 保存音频
        if output_path:
            return self.save_audio(audio, output_path)
            
        return True
    
    def is_available(self) -> bool:
        """检查适配器是否可用
        
        Returns:
            是否可用
        """
        return self.model is not None 

    def get_voice(self, voice_id: str) -> Optional[KokoroVoice]:
        """获取语音对象
        
        Args:
            voice_id: 语音ID
            
        Returns:
            语音对象，如果不存在返回None
        """
        # 检查模型和语音目录
        if not hasattr(self, 'model') or self.model is None:
            logger.error("模型未初始化，无法获取语音")
            return None
        
        # 如果语音目录存在，尝试加载
        if self.voices_dir and os.path.exists(self.voices_dir):
            # 构建可能的语音文件路径
            voice_path = os.path.join(self.voices_dir, f"{voice_id}.pt")
            if not os.path.exists(voice_path):
                voice_path = os.path.join(self.voices_dir, f"{voice_id}.pth")
            
            # 如果找到语音文件，加载它
            if os.path.exists(voice_path):
                try:
                    # 创建语音对象
                    voice = KokoroVoice(
                        id=voice_id,
                        path=voice_path,
                        language_code="zh" if voice_id.startswith("z") else "en"
                    )
                    
                    # 加载语音张量
                    voice.tensor = torch.load(voice_path, map_location=self.device)
                    logger.info(f"已加载语音: {voice_id} 从 {voice_path}")
                    return voice
                except Exception as e:
                    logger.error(f"加载语音文件失败: {e}")
                    return None 