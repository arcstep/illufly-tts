#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kokoro适配器测试模块
"""

import os
import unittest
import tempfile
from unittest.mock import MagicMock, patch
import torch
import numpy as np

# 确保目录结构正确
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from illufly_tts.vocoders.kokoro_adapter import KokoroAdapter, KokoroVoice, KOKORO_AVAILABLE


class TestKokoroVoice(unittest.TestCase):
    """测试KokoroVoice类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试用的语音文件
        self.pt_voice_path = os.path.join(self.temp_dir, "test_voice.pt")
        self.npy_voice_path = os.path.join(self.temp_dir, "test_voice.npy")
        
        # 创建伪语音张量
        self.test_tensor = torch.ones((128,))
        torch.save(self.test_tensor, self.pt_voice_path)
        np.save(self.npy_voice_path, self.test_tensor.numpy())
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时文件
        if os.path.exists(self.pt_voice_path):
            os.remove(self.pt_voice_path)
        if os.path.exists(self.npy_voice_path):
            os.remove(self.npy_voice_path)
        
        # 删除临时目录
        os.rmdir(self.temp_dir)
    
    def test_load_pt_voice(self):
        """测试加载.pt格式语音"""
        voice = KokoroVoice(id="test", path=self.pt_voice_path)
        self.assertFalse(voice.loaded)
        
        # 测试加载
        result = voice.load()
        self.assertTrue(result)
        self.assertTrue(voice.loaded)
        self.assertIsNotNone(voice.tensor)
        
        # 检查张量值
        self.assertTrue(torch.all(voice.tensor == self.test_tensor))
    
    def test_load_npy_voice(self):
        """测试加载.npy格式语音"""
        voice = KokoroVoice(id="test", path=self.npy_voice_path)
        self.assertFalse(voice.loaded)
        
        # 测试加载
        result = voice.load()
        self.assertTrue(result)
        self.assertTrue(voice.loaded)
        self.assertIsNotNone(voice.tensor)
        
        # 检查张量值
        self.assertTrue(torch.all(voice.tensor == self.test_tensor))
    
    def test_load_nonexistent_voice(self):
        """测试加载不存在的语音文件"""
        voice = KokoroVoice(id="test", path="/nonexistent/path.pt")
        result = voice.load()
        self.assertFalse(result)
        self.assertFalse(voice.loaded)
        self.assertIsNone(voice.tensor)
    
    def test_get_tensor(self):
        """测试获取语音张量"""
        voice = KokoroVoice(id="test", path=self.pt_voice_path)
        
        # 获取张量应自动加载
        tensor = voice.get_tensor()
        self.assertIsNotNone(tensor)
        self.assertTrue(voice.loaded)
        self.assertTrue(torch.all(tensor == self.test_tensor))
        
        # 测试设备指定
        cpu_tensor = voice.get_tensor("cpu")
        self.assertEqual(cpu_tensor.device.type, "cpu")


@unittest.skipIf(not KOKORO_AVAILABLE, "Kokoro模块不可用，跳过适配器测试")
class TestKokoroAdapter(unittest.TestCase):
    """测试KokoroAdapter类"""
    
    def setUp(self):
        """测试前准备"""
        # 模拟Kokoro模块
        self.model_mock = MagicMock()
        self.pipeline_mock = MagicMock()
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.voices_dir = os.path.join(self.temp_dir, "voices")
        os.makedirs(self.voices_dir, exist_ok=True)
        
        # 创建测试用的语音文件
        self.voice_path = os.path.join(self.voices_dir, "z001.pt")
        self.test_tensor = torch.ones((128,))
        torch.save(self.test_tensor, self.voice_path)
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时文件
        if os.path.exists(self.voice_path):
            os.remove(self.voice_path)
        
        # 删除临时目录
        if os.path.exists(self.voices_dir):
            os.rmdir(self.voices_dir)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    @patch('illufly_tts.vocoders.kokoro_adapter.KModel')
    @patch('illufly_tts.vocoders.kokoro_adapter.KPipeline')
    def test_init_adapter(self, mock_pipeline_class, mock_model_class):
        """测试初始化适配器"""
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # 创建适配器
        adapter = KokoroAdapter(
            model_path="/path/to/model",
            voices_dir=self.voices_dir,
            device="cpu"
        )
        
        # 检查初始化
        self.assertEqual(adapter.model_path, "/path/to/model")
        self.assertEqual(adapter.voices_dir, self.voices_dir)
        self.assertEqual(adapter.device, "cpu")
        self.assertTrue(adapter.available)
        
        # 检查模型加载
        mock_model_class.assert_called_once()
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()
        
        # 检查pipeline初始化
        mock_pipeline_class.assert_called_once()
        
        # A4检查语音加载
        self.assertIn("z001", adapter.voices)
        self.assertEqual(len(adapter.voices), 1)
    
    @patch('illufly_tts.vocoders.kokoro_adapter.KModel')
    @patch('illufly_tts.vocoders.kokoro_adapter.KPipeline')
    def test_list_voices(self, mock_pipeline_class, mock_model_class):
        """测试获取语音列表"""
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # 创建适配器
        adapter = KokoroAdapter(
            model_path="/path/to/model",
            voices_dir=self.voices_dir,
            device="cpu"
        )
        
        # 测试获取语音列表
        voices = adapter.list_voices()
        self.assertIn("z001", voices)
        self.assertEqual(len(voices), 1)
    
    @patch('illufly_tts.vocoders.kokoro_adapter.KModel')
    @patch('illufly_tts.vocoders.kokoro_adapter.KPipeline')
    def test_get_voice(self, mock_pipeline_class, mock_model_class):
        """测试获取语音"""
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # 创建适配器
        adapter = KokoroAdapter(
            model_path="/path/to/model",
            voices_dir=self.voices_dir,
            device="cpu"
        )
        
        # 测试获取存在的语音
        voice = adapter.get_voice("z001")
        self.assertIsNotNone(voice)
        self.assertEqual(voice.id, "z001")
        self.assertEqual(voice.path, self.voice_path)
        
        # 测试获取不存在的语音
        voice = adapter.get_voice("nonexistent")
        self.assertIsNone(voice)
    
    @patch('illufly_tts.vocoders.kokoro_adapter.KModel')
    @patch('illufly_tts.vocoders.kokoro_adapter.KPipeline')
    def test_generate_audio_with_pipeline(self, mock_pipeline_class, mock_model_class):
        """测试使用官方Pipeline生成音频"""
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # 设置pipeline输出
        result1 = MagicMock()
        result1.audio = torch.ones((1000,))
        result2 = MagicMock()
        result2.audio = torch.ones((1000,))
        mock_pipeline.return_value = [result1, result2]
        
        # 创建适配器
        adapter = KokoroAdapter(
            model_path="/path/to/model",
            voices_dir=self.voices_dir,
            device="cpu"
        )
        
        # 测试生成音频
        audio = adapter.generate_audio(
            text="测试文本",
            voice_id="z001",
            use_pipeline=True
        )
        
        # 检查结果
        self.assertIsNotNone(audio)
        self.assertEqual(audio.shape[0], 2000)  # 两个1000长度的音频拼接
        
        # 检查pipeline调用
        mock_pipeline.assert_called_once_with("测试文本", "z001", speed=1.0)
    
    @patch('illufly_tts.vocoders.kokoro_adapter.KModel')
    @patch('illufly_tts.vocoders.kokoro_adapter.KPipeline')
    def test_generate_audio_with_custom_g2p(self, mock_pipeline_class, mock_model_class):
        """测试使用自定义G2P生成音频"""
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # 设置infer输出
        result = MagicMock()
        result.audio = torch.ones((2000,))
        mock_pipeline_class.infer.return_value = result
        
        # 创建适配器
        adapter = KokoroAdapter(
            model_path="/path/to/model",
            voices_dir=self.voices_dir,
            device="cpu"
        )
        
        # 模拟G2P转换器
        mock_g2p = MagicMock()
        mock_g2p.text_to_phonemes.return_value = "n i3 h ao3"
        adapter.g2p = mock_g2p
        
        # 测试生成音频
        audio = adapter.generate_audio(
            text="测试文本",
            voice_id="z001",
            use_pipeline=False
        )
        
        # 检查结果
        self.assertIsNotNone(audio)
        self.assertEqual(audio.shape[0], 2000)
        
        # 检查G2P调用
        mock_g2p.text_to_phonemes.assert_called_once_with("测试文本")
        
        # 检查infer调用
        mock_pipeline_class.infer.assert_called_once()
    
    @patch('illufly_tts.vocoders.kokoro_adapter.KModel')
    @patch('illufly_tts.vocoders.kokoro_adapter.KPipeline')
    @patch('scipy.io.wavfile.write')
    def test_save_audio(self, mock_wavfile_write, mock_pipeline_class, mock_model_class):
        """测试保存音频"""
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        
        # 创建适配器
        adapter = KokoroAdapter(
            model_path="/path/to/model",
            voices_dir=self.voices_dir,
            device="cpu"
        )
        
        # 创建临时输出路径
        output_path = os.path.join(self.temp_dir, "output.wav")
        
        # 测试保存音频
        audio = torch.ones((2000,))
        result = adapter.save_audio(audio, output_path)
        
        # 检查结果
        self.assertTrue(result)
        
        # 检查wavfile.write调用
        mock_wavfile_write.assert_called_once()
        args, _ = mock_wavfile_write.call_args
        self.assertEqual(args[0], output_path)
        self.assertEqual(args[1], 24000)  # 默认采样率


if __name__ == "__main__":
    unittest.main() 