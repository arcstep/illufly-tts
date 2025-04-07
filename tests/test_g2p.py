#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
G2P模块测试
"""

import os
import sys
import unittest
import logging

# 添加项目根目录到路径，确保可以导入illufly_tts模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.illufly_tts.g2p import BaseG2P, EnglishG2P, ChineseG2P, MixedG2P

logging.basicConfig(level=logging.INFO)

class TestEnglishG2P(unittest.TestCase):
    """测试英文G2P功能"""
    
    def setUp(self):
        self.g2p = EnglishG2P()
        
    def test_english_phonemes(self):
        """测试英文音素转换"""
        # 测试基本英文
        text = "hello world"
        phonemes = self.g2p.text_to_phonemes(text)
        
        # 检查输出不为空
        self.assertTrue(phonemes)
        
        # 检查音素分隔
        self.assertIn(" ", phonemes)
        
        # 打印结果以便调试
        print(f"英文G2P测试 - 输入: '{text}' -> 输出: '{phonemes}'")
        
    def test_preprocess(self):
        """测试英文预处理"""
        text = "I'm going to school."
        processed = self.g2p.preprocess_text(text)
        
        # 检查缩写处理
        self.assertIn("am", processed)
        self.assertNotIn("I'm", processed)
        
    def test_get_phoneme_set(self):
        """测试获取音素集"""
        phoneme_set = self.g2p.get_phoneme_set()
        
        # 检查音素集不为空
        self.assertTrue(phoneme_set)
        
        # 检查常见英文音素存在
        common_phonemes = ['aa', 'ae', 'b', 'k', 't']
        for phoneme in common_phonemes:
            self.assertIn(phoneme, phoneme_set)
        
class TestChineseG2P(unittest.TestCase):
    """测试中文G2P功能"""
    
    def setUp(self):
        self.g2p = ChineseG2P()
        
    def test_chinese_phonemes(self):
        """测试中文音素转换"""
        # 测试基本中文
        text = "你好世界"
        phonemes = self.g2p.text_to_phonemes(text)
        
        # 检查输出不为空
        self.assertTrue(phonemes)
        
        # 打印结果以便调试
        print(f"中文G2P测试 - 输入: '{text}' -> 输出: '{phonemes}'")
        
    def test_chinese_with_tones(self):
        """测试带声调的中文拼音"""
        # 测试一个具有明确拼音的词
        text = "中国"
        phonemes = self.g2p.text_to_phonemes(text)
        
        # 检查输出不为空
        self.assertTrue(phonemes)
        
        # 打印结果以便调试
        print(f"中文G2P音调测试 - 输入: '{text}' -> 输出: '{phonemes}'")
        
    def test_get_phoneme_set(self):
        """测试获取拼音音素集"""
        phoneme_set = self.g2p.get_phoneme_set()
        
        # 检查音素集不为空
        self.assertTrue(phoneme_set)
        
        # 检查常见声母存在
        common_initials = ['b', 'd', 'g', 'zh', 'ch', 'sh']
        for initial in common_initials:
            self.assertIn(initial, phoneme_set)
            
        # 检查常见韵母存在
        common_finals = ['a', 'o', 'e', 'ai', 'ei', 'ao']
        for final in common_finals:
            self.assertIn(final, phoneme_set)
        
class TestMixedG2P(unittest.TestCase):
    """测试混合语言G2P功能"""
    
    def setUp(self):
        self.g2p = MixedG2P()
        
    def test_mixed_text(self):
        """测试混合语言文本"""
        # 测试中英混合文本
        text = "你好，Hello World！这是一个测试。"
        phonemes = self.g2p.text_to_phonemes(text)
        
        # 检查输出不为空
        self.assertTrue(phonemes)
        
        # 打印结果以便调试
        print(f"混合G2P测试 - 输入: '{text}' -> 输出: '{phonemes}'")
        
    def test_language_switch(self):
        """测试语言切换模式"""
        # 测试带切换标记的处理
        text = "这是中文Chinese混合文本"
        
        # 默认模式（分开处理）
        phonemes1 = self.g2p.text_to_phonemes(text)
        
        # 切换到混合模式
        self.g2p.align_mode = 'mixed'
        phonemes2 = self.g2p.text_to_phonemes(text)
        
        # 检查两种模式的输出
        self.assertTrue(phonemes1)
        self.assertTrue(phonemes2)
        
        # 在混合模式下应该有语言切换标记
        if phonemes2.find("<ZH2EN>") >= 0 or phonemes2.find("<EN2ZH>") >= 0:
            has_switch_tokens = True
        else:
            has_switch_tokens = False
            
        # 打印结果以便调试
        print(f"语言切换测试 - 分开模式: '{phonemes1}'")
        print(f"语言切换测试 - 混合模式: '{phonemes2}'")
        print(f"包含切换标记: {has_switch_tokens}")
        
    def test_process_metadata(self):
        """测试处理元数据"""
        # 测试详细处理结果
        text = "中文和English混合"
        result = self.g2p.process(text)
        
        # 检查结果包含必要字段
        self.assertIn("phonemes", result)
        self.assertIn("text", result)
        self.assertIn("language", result)
        self.assertIn("segments", result)
        
        # 检查段落信息
        self.assertTrue(len(result["segments"]) > 0)
        
        # 打印结果以便调试
        print(f"处理元数据测试 - 输入: '{text}'")
        print(f"段落数: {len(result['segments'])}")
        for i, seg in enumerate(result["segments"]):
            print(f"段落[{i+1}]: 文本='{seg['text']}', 语言={seg['language']}, 音素='{seg['phonemes']}'")
        
if __name__ == '__main__':
    unittest.main() 