#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TTS核心服务"""

import logging
import os
import re
import sys
import time
import asyncio
import json
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, AsyncGenerator

import numpy as np
import torch
import soundfile as sf
import base64
import langid
import warnings

# 导入我们的自定义英文G2P模块和自定义Pipeline
from .english_g2p import english_g2p
from .custom_pipeline import CustomPipeline, OFFLINE_MODE

# 设置日志记录器
logger = logging.getLogger(__name__)

# 检查是否在离线模式
OFFLINE_MODE = os.environ.get("KOKORO_OFFLINE", "0").lower() in ("1", "true", "yes", "y")

class TTSService:
    """语音合成服务
    
    同时支持官方KPipeline和自定义Pipeline，可根据需要选择
    """
    def __init__(
        self, 
        model_path: Optional[str] = None,
        voice_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_custom_pipeline: bool = False  # 默认使用官方KPipeline
    ):
        """初始化TTS服务
        
        Args:
            model_path: 模型路径
            voice_dir: 语音目录
            device: 计算设备
            use_custom_pipeline: 是否使用自定义Pipeline (默认False，使用官方KPipeline)
        """
        self.model_path = model_path
        self.voice_dir = voice_dir or './voices'
        self.device = device
        self.use_custom_pipeline = use_custom_pipeline
        
        logger.info(f"初始化TTS服务: 设备={device}, 模型路径={model_path}")
        
        # 检查模型是否存在
        if model_path:
            logger.info(f"加载模型: {model_path}")
            try:
                from kokoro import KPipeline
                logger.info("导入KPipeline成功")
            except ImportError:
                logger.warning("无法导入kokoro.KPipeline")
                
            # 如果提供了模型路径，加载该模型
            logger.info(f"初始化TTS服务: 设备={device}")
            logger.info(f"使用本地模型: {model_path}")
            
            # 查找.pth文件
            model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
            if model_files:
                model_file = os.path.join(model_path, model_files[0])
                logger.info(f"找到模型文件: {model_file}")
            else:
                raise FileNotFoundError(f"在{model_path}中找不到模型文件(.pth)")
            
            # 扩展语言代码检测
            if 'zh' in model_path or '-z-' in model_path or '-z.' in model_path:
                self.lang_code = 'z'  # 中文
            elif 'ja' in model_path or '-j-' in model_path or '-j.' in model_path:
                self.lang_code = 'j'  # 日文
            else:
                self.lang_code = 'a'  # 默认为英文（美式）
                
            # 加载预训练模型
            from kokoro import KModel
            config_path = os.path.join(model_path, 'config.json')
            self.model = KModel(
                config=config_path, 
                model=model_file,
                repo_id="hexgrad/Kokoro-82M-v1.1-zh"  # 使用固定的repo_id
            ).to(device or 'cpu').eval()
                
            # 初始化管道
            if use_custom_pipeline:
                # 使用自定义Pipeline（完全不依赖espeak）
                from .custom_pipeline import CustomPipeline
                logger.info("使用自定义Pipeline（无espeak依赖）")
                self.pipeline = CustomPipeline(
                    model=self.model,
                    repo_id="hexgrad/Kokoro-82M-v1.1-zh",  # 使用固定的repo_id
                    device=device
                )
            else:
                # 使用官方KPipeline
                logger.info("使用官方KPipeline")
                self.pipeline = KPipeline(
                    lang_code=self.lang_code,
                    model=self.model,
                    repo_id="hexgrad/Kokoro-82M-v1.1-zh",  # 使用固定的repo_id
                    device=device
                )
                
            logger.info("模型加载完成")
        else:
            logger.warning("未指定模型路径，仅初始化TTS服务框架")
            self.model = None
            self.pipeline = None
            
        logger.info("TTS服务初始化完成")
    
    def _en_callable(self, text):
        """处理英文文本（无需phonemizer）
        
        Args:
            text: 英文文本
            
        Returns:
            处理后的音素序列
        """
        try:
            logger.info(f"处理英文文本: {text}")
            # 使用我们的英文G2P模块处理
            phonemes = english_g2p(text)
            logger.info(f"英文转换为音素: {text} -> {phonemes}")
            return phonemes
        except Exception as e:
            logger.error(f"英文处理失败: {e}")
            # 如果处理失败，返回原始文本
            return text
    
    def _load_voice_tensor(self, voice_id: str) -> torch.FloatTensor:
        """加载语音张量
        
        Args:
            voice_id: 语音ID
            
        Returns:
            语音张量
        """
        # 检查本地语音文件
        npy_path = os.path.join(self.voice_dir, f"{voice_id}.npy")
        pt_path = os.path.join(self.voice_dir, f"{voice_id}.pt")
        
        if os.path.exists(npy_path):
            logger.info(f"加载本地语音文件: {npy_path}")
            # 加载.npy文件
            voice_np = np.load(npy_path)
            voice_tensor = torch.from_numpy(voice_np).float()
            
            # 确保形状正确（应该是[N, 256]或[256]）
            if voice_tensor.dim() == 1 and voice_tensor.shape[0] != 256:
                raise ValueError(f"语音向量维度错误: {voice_tensor.shape}, 应为[256]")
            elif voice_tensor.dim() == 2 and voice_tensor.shape[1] != 256:
                raise ValueError(f"语音向量维度错误: {voice_tensor.shape}, 应为[N, 256]")
                
            # 如果是1维，扩展为2维 [256] -> [1, 256]
            if voice_tensor.dim() == 1:
                voice_tensor = voice_tensor.unsqueeze(0)
                
            # 归一化
            voice_tensor = voice_tensor / torch.norm(voice_tensor, dim=1, keepdim=True)
            
            # 将张量移动到正确的设备上
            if self.device:
                voice_tensor = voice_tensor.to(self.device)
                
            return voice_tensor
            
        elif os.path.exists(pt_path):
            logger.info(f"加载本地语音文件: {pt_path}")
            # 加载.pt文件
            voice_tensor = torch.load(pt_path, map_location=self.device)
            return voice_tensor
            
        else:
            # 创建随机向量（仅用于测试）
            logger.warning(f"找不到语音文件: {voice_id}，使用随机向量")
            voice_tensor = torch.randn(1, 256)
            voice_tensor = voice_tensor / torch.norm(voice_tensor, dim=1, keepdim=True)
            if self.device:
                voice_tensor = voice_tensor.to(self.device)
            return voice_tensor
    
    def _generate_random_voice_vector(self) -> np.ndarray:
        """生成随机语音向量"""
        # 默认256维，与Kokoro模型兼容
        vector_size = 256
        
        try:
            # 尝试获取正确的向量维度
            if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'hidden_size'):
                vector_size = self.model.cfg.hidden_size
                logger.info(f"使用模型配置的向量维度: {vector_size}")
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                vector_size = self.model.config.hidden_size
                logger.info(f"使用模型配置的向量维度: {vector_size}")
        except Exception as e:
            logger.warning(f"获取模型向量维度失败，使用默认值256: {e}")
        
        # 生成随机向量
        voice_np = np.random.randn(1, vector_size).astype(np.float32)
        # 归一化向量
        voice_np = voice_np / np.linalg.norm(voice_np, axis=1, keepdims=True)
        return voice_np
    
    def _get_voice_tensor(self, voice_id: str = None) -> torch.Tensor:
        """获取声音张量，如果指定ID则使用指定ID，否则使用默认"""
        voice_to_use = voice_id or self.voice
        
        # 检查缓存中是否已有此voice的张量
        if voice_to_use in self.voice_tensor_cache:
            return self.voice_tensor_cache[voice_to_use]
        
        # 加载声音张量
        voice_tensor = self._load_voice_tensor(voice_to_use)
        
        # 缓存语音张量以提高性能
        self.voice_tensor_cache[voice_to_use] = voice_tensor
        
        return voice_tensor
    
    def _is_english_text(self, text: str) -> bool:
        """判断文本是否含有英文"""
        # 检查文本是否包含英文字符（a-zA-Z）
        return bool(re.search(r'[a-zA-Z]', text))
    
    def _is_chinese_text(self, text: str) -> bool:
        """判断文本是否含有中文"""
        # 检查文本是否包含中文字符（\u4e00-\u9fff）
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def start(self):
        """启动处理任务"""
        if self.processing_task is None or self.processing_task.done():
            logger.info("启动TTS处理任务")
            self.processing_task = asyncio.create_task(self._process_queue())
        return self
    
    async def stop(self):
        """停止处理任务"""
        if self.processing_task and not self.processing_task.done():
            logger.info("停止TTS处理任务")
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None
    
    def _speed_callable(self, len_ps: int) -> float:
        """根据文本长度动态调整语速
        
        非常短的文本使用正常语速，长文本稍微减慢
        """
        speed = 0.8
        if len_ps <= 83:
            speed = 1
        elif len_ps < 183:
            speed = 1 - (len_ps - 83) / 500
        return speed * 1.1
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单判断：如果包含中文字符，则认为是中文
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        
        # 使用langid库进行更精确的语言检测
        try:
            lang, _ = langid.classify(text)
            if lang == 'zh':
                return 'zh'
            else:
                return 'en'
        except:
            # 回退到简单规则：假设纯ASCII文本是英文
            if all(ord(c) < 128 for c in text):
                return 'en'
            return 'zh'
    
    def _generate_speech_impl(self, text: str, voice_id: str = None) -> tuple:
        """
        生成语音的内部实现(带缓存)
        """
        start_time = time.time()
        logger.debug(f"开始TTS处理，文本：{text}")
        
        try:
            # 1. 处理语音ID
            voice_tensor = self._get_voice_tensor(voice_id)
            
            # 2. 检测语言
            is_english = self._is_english_text(text)
            is_chinese = self._is_chinese_text(text)
            is_mixed = is_english and is_chinese
            
            logger.debug(f"语言检测: 英文={is_english}, 中文={is_chinese}, 混合={is_mixed}")
            
            # 3. 根据文本选择合适的处理管道
            audio_array = None
            
            # 3.1 使用Pipeline处理
            if self.use_pipeline:
                if is_english and self.has_en_pipeline and not is_mixed:
                    # 纯英文文本且有英文管道
                    logger.debug("使用英文管道处理纯英文文本")
                    audio_array = self.en_pipeline.predict(
                        text,
                        voice_samples=voice_tensor
                    )
                else:
                    # 中文或混合文本，使用标准管道处理
                    logger.debug("使用标准管道处理文本")
                    audio_array = self.pipeline.predict(
                        text,
                        voice_samples=voice_tensor
                    )
            # 3.2 不使用Pipeline，直接使用模型
            else:
                # 根据语言类型选择处理方法
                if is_english and not is_chinese:
                    phones = self.en_callable(text)
                    logger.debug(f"英文转换音素: {phones}")
                elif is_chinese and not is_english:
                    phones = self.zh_callable(text)
                    logger.debug(f"中文转换音素: {phones}")
                else:
                    # 混合文本先尝试用中文处理
                    phones = self.zh_callable(text)
                    logger.debug(f"混合文本转换音素: {phones}")
                
                # 直接使用模型生成
                if phones:
                    attention_text = None # add_params.get("attention_text")
                    audio_array = self.inference_func(
                        phones, 
                        voice_tensor,
                        attention_text=attention_text
                    ).squeeze().detach().cpu().numpy() 
                else:
                    raise ValueError("音素转换失败")
            
            # 4. 检查结果
            if audio_array is None:
                raise ValueError("语音生成失败，返回了空的音频数据")

            # 5. 计算音频持续时间和采样率
            duration = len(audio_array) / self.sample_rate
            logger.debug(f"语音生成完成，持续时间：{duration:.2f}秒")
            
            # 6. 统计处理用时
            end_time = time.time()
            processing_time = end_time - start_time
            logger.debug(f"TTS处理完成，耗时：{processing_time:.2f}秒")
            
            return audio_array, self.sample_rate
            
        except Exception as e:
            logger.error(f"生成语音时出错: {str(e)}")
            # 在出错时返回日志信息
            raise e
    
    async def _process_queue(self):
        """处理队列中的文本"""
        while True:
            try:
                # 获取一批文本
                batch = []
                for _ in range(self.batch_size):
                    try:
                        item = await asyncio.wait_for(self.processing_queue.get(), timeout=0.1)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if not batch:
                    await asyncio.sleep(0.1)
                    continue
                
                # 处理这批文本
                for index, text, voice in batch:
                    result = self._generate_speech(text, voice)
                    result["index"] = index
                    await self.result_queue.put(result)
                
            except Exception as e:
                logger.error(f"处理队列失败: {str(e)}")
                await asyncio.sleep(0.1)
    
    async def text_to_speech(
        self, 
        texts: List[str], 
        voice: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """将文本转换为语音，并生成音频数据
        
        Args:
            texts: 要转换的文本列表
            voice: 语音ID，默认使用实例的voice
            
        Yields:
            包含音频数据的字典，格式为：
            {
                "text": "原始文本",
                "audio": "base64编码的音频数据",
                "index": 当前文本的索引,
                "status": "success"或"error"
            }
        """
        # 确保处理任务正在运行
        self.start()
        
        # 将文本添加到处理队列
        for i, text in enumerate(texts):
            await self.processing_queue.put((i, text, voice))
        
        # 从结果队列获取结果
        for _ in range(len(texts)):
            result = await self.result_queue.get()
            yield result
    
    def text_to_speech_sync(
        self,
        text: str,
        voice: Optional[str] = None
    ) -> Dict[str, Any]:
        """同步将文本转换为语音（同步版本）
        
        Args:
            text: 要转换的文本
            voice: 语音ID（可选）
            
        Returns:
            结果字典，包含音频数据
        """
        result = self.generate_speech(text, voice, return_base64=False)
        return result
    
    async def save_speech_to_file(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None
    ) -> Dict[str, Any]:
        """将文本转换为语音并保存到文件
        
        Args:
            text: 要转换的文本
            output_path: 输出文件路径
            voice: 语音ID（可选）
            
        Returns:
            结果字典
        """
        try:
            # 生成语音
            result = self.text_to_speech_sync(text, voice)
            
            if result["status"] == "success":
                # 获取音频数据
                audio_array = result.pop("audio_array", None)
                if audio_array is None:
                    # 如果没有audio_array字段，尝试从base64解码
                    if "audio" in result:
                        import base64
                        import io
                        audio_data = base64.b64decode(result["audio"])
                        result.pop("audio")  # 移除base64数据以减少日志大小
                        
                        # 使用soundfile直接写入
                        import soundfile as sf
                        with open(output_path, "wb") as f:
                            f.write(audio_data)
                        logger.info(f"音频已保存到: {output_path}")
                else:
                    # 直接使用numpy数组保存
                    import soundfile as sf
                    sf.write(output_path, audio_array, self.sample_rate)
                    logger.info(f"音频已保存到: {output_path}")
                
                # 添加文件路径到结果
                result["file_path"] = output_path
            
            return result
            
        except Exception as e:
            logger.error(f"保存音频到文件失败: {str(e)}")
            return {
                "status": "error",
                "text": text,
                "error": str(e)
            }
    
    def _init_custom_pipeline(self):
        """初始化自定义Pipeline"""
        logger.info("使用自定义Pipeline（无espeak依赖）")
        
        # 导入自定义管道
        from illufly_tts.custom_pipeline import CustomPipeline
        
        # 检查我们是否已有模型实例
        if self.model is None:
            # 导入并加载模型
            try:
                # 尝试从官方管道加载
                try:
                    from kokoro.pipeline import KPipeline
                    # 不要使用pipeline，只为了获取模型
                    temp_pipeline = KPipeline(
                        lang_code='z',
                        repo_id=self.repo_id,
                        device=self.device
                    )
                    self.model = temp_pipeline.model
                    logger.info("从KPipeline获取模型成功")
                except ImportError:
                    # 否则直接加载模型
                    from kokoro.model import KModel
                    self.model = KModel(self.repo_id).to(self.device)
                    logger.info("直接加载KModel成功")
            except Exception as e:
                logger.error(f"加载KModel失败: {e}")
                raise
        
        # 初始化自定义管道
        self.pipeline = CustomPipeline(
            model=self.model,
            repo_id=self.repo_id,
            device=self.device,
            en_callable=self.en_callable
        )
        logger.info("自定义Pipeline初始化完成")
        self.use_pipeline = True
        self.has_en_pipeline = True  # 自定义Pipeline自带英文支持
    
    def _init_official_pipeline(self):
        """初始化官方KPipeline"""
        logger.info("使用官方KPipeline")
        
        # 确保device参数兼容Kokoro
        # MPS设备需要转换为CPU，因为Kokoro不直接支持MPS
        kokoro_device = 'cpu' if self.device == 'mps' else self.device
        
        # 导入KPipeline
        from kokoro.pipeline import KPipeline
        
        # 初始化中文管道 (z 表示中文)
        try:
            self.pipeline = KPipeline(
                lang_code='z',  # 中文
                repo_id=self.repo_id,
                device=kokoro_device,  # 使用兼容的设备
                en_callable=self.en_callable  # 添加英文处理函数
            )
            logger.info("中文语音管道初始化完成")
            self.use_pipeline = True
            
            # 尝试获取模型，如果关联了模型
            if hasattr(self.pipeline, 'model'):
                self.model = self.pipeline.model
                
            # 初始化英文管道 (a 表示英文)
            try:
                self.en_pipeline = KPipeline(
                    lang_code='a',  # 英文
                    repo_id=self.repo_id,
                    model=self.pipeline.model,  # 重用同一个模型
                    device=kokoro_device  # 使用兼容的设备
                )
                logger.info("英文语音管道初始化完成")
                self.has_en_pipeline = True
            except Exception as e:
                logger.warning(f"英文语音管道初始化失败: {str(e)}")
                logger.info("将使用中文管道处理所有文本")
                self.has_en_pipeline = False
                
            # 测试管道是否正常工作
            try:
                # 使用简单文本测试管道
                test_result = None
                for result in self.pipeline("测试"):
                    test_result = result
                    break  # 只需要检测第一个结果
            except Exception as e:
                error_msg = str(e).lower()
                if ("espeak" in error_msg or 
                    "espeakng" in error_msg or 
                    "phontab" in error_msg or 
                    "no such file or directory" in error_msg):
                    logger.warning(f"检测到espeak加载错误: {str(e)}")
                    raise ValueError("espeak加载失败，需要回退到自定义pipeline")
                else:
                    # 其他错误直接抛出
                    raise
                
        except Exception as e:
            logger.error(f"中文语音管道初始化失败: {str(e)}")
            raise

    def generate_speech(
        self, 
        text: str, 
        voice: Optional[str] = None,
        return_base64: bool = True
    ) -> Dict[str, Any]:
        """生成语音
        
        Args:
            text: 要转换的文本
            voice: 语音ID，默认使用实例的voice
            return_base64: 是否返回base64编码的音频
            
        Returns:
            包含音频数据的字典
        """
        try:
            # 使用指定语音或默认语音
            voice_id = voice or self.voice
            
            # 检查文本是否为空
            if not text or text.strip() == "":
                logger.warning("收到空文本，使用默认文本")
                text = "测试文本"
                
            # 如果文本太短，可能会生成异常短的音频，这里确保文本长度
            if len(text) < 10:
                # 重复文本以确保有足够长度
                repeat_count = max(1, 10 // len(text))
                logger.info(f"文本太短，重复{repeat_count}次以确保生成质量")
                text = (text + "。") * repeat_count
            
            # 检测文本语言
            lang = self._detect_language(text)
            logger.info(f"生成语音: {text} (语言: {lang})")
            
            # 调用内部实现生成语音
            audio_np, sample_rate = self._generate_speech(text, voice_id)
            
            # 验证音频数据长度是否合理
            audio_duration = len(audio_np) / sample_rate
            logger.info(f"生成的音频长度: {audio_duration:.2f}秒 ({len(audio_np)}样本)")
            
            response = {
                "text": text,
                "status": "success",
                "sample_rate": sample_rate,
                "voice": voice_id,
                "duration": audio_duration
            }
            
            # 根据需要返回base64编码的音频或原始数组
            if return_base64:
                # 将音频数据转换为base64
                buffer = io.BytesIO()
                sf.write(buffer, audio_np, sample_rate, format='WAV')
                audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                response["audio"] = audio_base64
            else:
                response["audio_array"] = audio_np.tolist()
            
            return response
            
        except Exception as e:
            logger.error(f"转换文本失败: {text}, 错误: {str(e)}")
            return {
                "text": text,
                "error": str(e),
                "status": "error"
            }

    def convert_text(self, text: str, voice_id: Optional[str] = None) -> np.ndarray:
        """将文本转换为语音
        
        Args:
            text: 输入文本
            voice_id: 语音ID
        
        Returns:
            音频numpy数组
            
        Raises:
            ValueError: 如果语音生成失败
        """
        if not self.model or not self.pipeline:
            raise ValueError("TTS服务未正确初始化")
            
        voice_id = voice_id or "zf_001"
        
        try:
            # 检测并记录语言
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
            has_english = bool(re.search(r'[a-zA-Z]', text))
            is_mixed = has_chinese and has_english
            
            lang = "zh" if has_chinese else "en"
            logger.info(f"生成语音: {text} (语言: {lang})")
            
            # 调试模式下输出更多信息
            logger.debug(f"开始TTS处理，文本：{text}")
            logger.debug(f"语言检测: 英文={has_english}, 中文={has_chinese}, 混合={is_mixed}")
            
            # 使用处理管道处理文本
            logger.debug("使用标准管道处理文本")
            
            # 使用pipeline处理文本
            results = list(self.pipeline(text, voice=voice_id))
            
            if not results:
                raise ValueError("语音生成失败: 未生成任何结果")
                
            # 合并所有音频块
            audio_chunks = []
            for result in results:
                if result.audio is not None:
                    audio_chunks.append(result.audio.numpy())
                    
            if not audio_chunks:
                raise ValueError("语音生成失败: 未生成任何音频")
                
            # 连接所有音频块
            audio = np.concatenate(audio_chunks)
            return audio
            
        except Exception as e:
            logger.error(f"转换文本失败: {text}, 错误: {str(e)}")
            raise ValueError(f"语音生成失败: {str(e)}") 