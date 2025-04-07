#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TTS流水线测试模块
"""

import os
import unittest
import tempfile
from unittest.mock import MagicMock, patch
import torch

# 确保目录结构正确
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from illufly_tts.pipeline import TTSPipeline, MixedLanguagePipeline
from illufly_tts.preprocessing.segmenter import LanguageSegmenter
from illufly_tts.preprocessing.normalizer import TextNormalizer
from illufly_tts.g2p.mixed_g2p import MixedG2P
from illufly_tts.vocoders.kokoro_adapter import KokoroAdapter, KOKORO_AVAILABLE


@patch('illufly_tts.vocoders.kokoro_adapter.KokoroAdapter')
@patch('illufly_tts.g2p.mixed_g2p.MixedG2P')
@patch('illufly_tts.preprocessing.normalizer.TextNormalizer')
@patch('illufly_tts.preprocessing.segmenter.LanguageSegmenter')
class TestTTSPipeline(unittest.TestCase):
    """测试基础TTS流水线"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model")
        self.voices_dir = os.path.join(self.temp_dir, "voices")
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.voices_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        if os.path.exists(self.voices_dir):
            os.rmdir(self.voices_dir)
        if os.path.exists(self.model_path):
            os.rmdir(self.model_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_init_pipeline(self, mock_segmenter_class, mock_normalizer_class, 
                           mock_g2p_class, mock_adapter_class):
        """测试初始化流水线"""
        # 设置模拟对象
        mock_segmenter = MagicMock()
        mock_segmenter_class.return_value = mock_segmenter
        
        mock_normalizer = MagicMock()
        mock_normalizer_class.return_value = mock_normalizer
        
        mock_g2p = MagicMock()
        mock_g2p_class.return_value = mock_g2p
        
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter
        
        # 创建流水线 - 显式传递模拟类
        pipeline = TTSPipeline(
            model_path=self.model_path,
            voices_dir=self.voices_dir,
            device="cpu",
            segmenter_class=mock_segmenter_class,
            normalizer_class=mock_normalizer_class,
            g2p_class=mock_g2p_class,
            adapter_class=mock_adapter_class
        )
        
        # 检查初始化
        self.assertEqual(pipeline.model_path, self.model_path)
        self.assertEqual(pipeline.voices_dir, self.voices_dir)
        self.assertEqual(pipeline.device, "cpu")
        self.assertEqual(pipeline.sample_rate, 24000)
        
        # 检查组件初始化
        mock_segmenter_class.assert_called_once()
        mock_normalizer_class.assert_called_once()
        mock_g2p_class.assert_called_once()
        mock_adapter_class.assert_called_once()
        
        # 检查组件引用
        self.assertEqual(pipeline.segmenter, mock_segmenter)
        self.assertEqual(pipeline.normalizer, mock_normalizer)
        self.assertEqual(pipeline.g2p, mock_g2p)
        self.assertEqual(pipeline.vocoder, mock_adapter)
    
    def test_preprocess_text(self, mock_segmenter_class, mock_normalizer_class, 
                             mock_g2p_class, mock_adapter_class):
        """测试文本预处理"""
        # 设置模拟对象
        mock_normalizer = MagicMock()
        mock_normalizer.normalize.return_value = "规范化的文本"
        mock_normalizer_class.return_value = mock_normalizer
        
        # 创建流水线
        pipeline = TTSPipeline(
            model_path=self.model_path,
            voices_dir=self.voices_dir,
            segmenter_class=mock_segmenter_class,
            normalizer_class=mock_normalizer_class,
            g2p_class=mock_g2p_class,
            adapter_class=mock_adapter_class
        )
        
        # 测试预处理
        result = pipeline.preprocess_text("测试文本")
        self.assertEqual(result, "规范化的文本")
        mock_normalizer.normalize.assert_called_once_with("测试文本")
    
    def test_text_to_phonemes(self, mock_segmenter_class, mock_normalizer_class, 
                              mock_g2p_class, mock_adapter_class):
        """测试文本转音素"""
        # 设置模拟对象
        mock_g2p = MagicMock()
        mock_g2p.text_to_phonemes.return_value = "t e4 s t"
        mock_g2p_class.return_value = mock_g2p
        
        # 创建流水线
        pipeline = TTSPipeline(
            model_path=self.model_path,
            voices_dir=self.voices_dir,
            segmenter_class=mock_segmenter_class,
            normalizer_class=mock_normalizer_class,
            g2p_class=mock_g2p_class,
            adapter_class=mock_adapter_class
        )
        
        # 测试文本转音素
        result = pipeline.text_to_phonemes("测试文本")
        self.assertEqual(result, "t e4 s t")
        mock_g2p.text_to_phonemes.assert_called_once_with("测试文本")
    
    def test_text_to_speech_official_pipeline(self, mock_segmenter_class, mock_normalizer_class, 
                                              mock_g2p_class, mock_adapter_class):
        """测试使用官方Pipeline生成语音"""
        # 设置模拟对象
        mock_adapter = MagicMock()
        mock_adapter.list_voices.return_value = ["z001"]
        mock_adapter.generate_audio.return_value = torch.ones((1000,))
        mock_adapter_class.return_value = mock_adapter
        
        # 创建流水线
        pipeline = TTSPipeline(
            model_path=self.model_path,
            voices_dir=self.voices_dir,
            segmenter_class=mock_segmenter_class,
            normalizer_class=mock_normalizer_class,
            g2p_class=mock_g2p_class,
            adapter_class=mock_adapter_class
        )
        
        # 测试生成语音
        result = pipeline.text_to_speech("测试文本", "z001", use_official_pipeline=True)
        
        # 检查结果
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 1000)
        
        # 检查调用
        mock_adapter.generate_audio.assert_called_once_with(
            "测试文本", "z001", 1.0, use_pipeline=True
        )
    
    def test_text_to_speech_custom_pipeline(self, mock_segmenter_class, mock_normalizer_class, 
                                           mock_g2p_class, mock_adapter_class):
        """测试使用自定义Pipeline生成语音"""
        # 设置模拟对象
        mock_normalizer = MagicMock()
        mock_normalizer.normalize.return_value = "规范化的文本"
        mock_normalizer_class.return_value = mock_normalizer
        
        mock_g2p = MagicMock()
        mock_g2p.text_to_phonemes.return_value = "t e4 s t"
        mock_g2p_class.return_value = mock_g2p
        
        mock_adapter = MagicMock()
        mock_adapter.list_voices.return_value = ["z001"]
        mock_adapter.generate_audio.return_value = torch.ones((1000,))
        mock_adapter_class.return_value = mock_adapter
        
        # 创建流水线
        pipeline = TTSPipeline(
            model_path=self.model_path,
            voices_dir=self.voices_dir,
            segmenter_class=mock_segmenter_class,
            normalizer_class=mock_normalizer_class,
            g2p_class=mock_g2p_class,
            adapter_class=mock_adapter_class
        )
        
        # 测试生成语音
        result = pipeline.text_to_speech("测试文本", "z001", use_official_pipeline=False)
        
        # 检查结果
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 1000)
        
        # 检查调用顺序
        mock_normalizer.normalize.assert_called_once_with("测试文本")
        mock_g2p.text_to_phonemes.assert_called_once_with("规范化的文本")
        mock_adapter.generate_audio.assert_called_once_with(
            "测试文本", "z001", 1.0, use_pipeline=False
        )
    
    def test_list_voices(self, mock_segmenter_class, mock_normalizer_class, 
                         mock_g2p_class, mock_adapter_class):
        """测试获取语音列表"""
        # 设置模拟对象
        mock_adapter = MagicMock()
        mock_adapter.list_voices.return_value = ["z001", "z002", "e001"]
        mock_adapter_class.return_value = mock_adapter
        
        # 创建流水线
        pipeline = TTSPipeline(
            model_path=self.model_path,
            voices_dir=self.voices_dir,
            segmenter_class=mock_segmenter_class,
            normalizer_class=mock_normalizer_class,
            g2p_class=mock_g2p_class,
            adapter_class=mock_adapter_class
        )
        
        # 重置mock，清除_init_components中调用的记录
        mock_adapter.list_voices.reset_mock()
        
        # 测试获取语音列表
        voices = pipeline.list_voices()
        self.assertEqual(voices, ["z001", "z002", "e001"])
        mock_adapter.list_voices.assert_called_once()


@patch('illufly_tts.vocoders.kokoro_adapter.KokoroAdapter')
@patch('illufly_tts.g2p.mixed_g2p.MixedG2P')
@patch('illufly_tts.preprocessing.normalizer.TextNormalizer')
@patch('illufly_tts.preprocessing.segmenter.LanguageSegmenter')
class TestMixedLanguagePipeline(unittest.TestCase):
    """测试混合语言TTS流水线"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model")
        self.voices_dir = os.path.join(self.temp_dir, "voices")
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.voices_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        if os.path.exists(self.voices_dir):
            os.rmdir(self.voices_dir)
        if os.path.exists(self.model_path):
            os.rmdir(self.model_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_text_to_speech_mixed_language(self, mock_segmenter_class, mock_normalizer_class, 
                                          mock_g2p_class, mock_adapter_class):
        """测试混合语言生成"""
        # 设置模拟对象
        mock_normalizer = MagicMock()
        mock_normalizer.normalize.return_value = "你好，Hello World！"
        mock_normalizer_class.return_value = mock_normalizer
        
        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = [
            {"text": "你好，", "lang": "zh"},
            {"text": "Hello World", "lang": "en"},
            {"text": "！", "lang": "zh"}
        ]
        mock_segmenter_class.return_value = mock_segmenter
        
        mock_g2p = MagicMock()
        mock_g2p.text_to_phonemes.side_effect = lambda text: "n i3 h ao3" if text == "你好，" else "hh eh l l ow"
        mock_g2p_class.return_value = mock_g2p
        
        mock_adapter = MagicMock()
        mock_adapter.list_voices.return_value = ["z001", "e001"]
        mock_adapter.generate_audio.side_effect = lambda text, voice_id, speed, use_pipeline: torch.ones((1000,)) if voice_id == "z001" else torch.ones((800,))
        mock_adapter_class.return_value = mock_adapter
        
        # 创建流水线 - 添加模拟类参数
        pipeline = MixedLanguagePipeline(
            model_path=self.model_path,
            voices_dir=self.voices_dir,
            segmenter_class=mock_segmenter_class,
            normalizer_class=mock_normalizer_class,
            g2p_class=mock_g2p_class,
            adapter_class=mock_adapter_class
        )
        
        # 测试生成语音
        result = pipeline.text_to_speech(
            "你好，Hello World！",
            "z001",
            use_official_pipeline=False
        )
        
        # 检查结果
        self.assertIsNotNone(result)
        
    def test_text_to_speech_official_pipeline(self, mock_segmenter_class, mock_normalizer_class, 
                                             mock_g2p_class, mock_adapter_class):
        """测试混合语言流水线使用官方Pipeline"""
        # 设置模拟对象
        mock_adapter = MagicMock()
        mock_adapter.list_voices.return_value = ["z001"]
        mock_adapter.generate_audio.return_value = torch.ones((1000,))
        mock_adapter_class.return_value = mock_adapter
        
        # 创建流水线 - 添加模拟类参数
        pipeline = MixedLanguagePipeline(
            model_path=self.model_path,
            voices_dir=self.voices_dir,
            segmenter_class=mock_segmenter_class,
            normalizer_class=mock_normalizer_class,
            g2p_class=mock_g2p_class,
            adapter_class=mock_adapter_class
        )
        
        # 测试生成语音
        result = pipeline.text_to_speech(
            "你好，Hello World！",
            "z001",
            use_official_pipeline=True
        )
        
        # 检查结果
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main() 