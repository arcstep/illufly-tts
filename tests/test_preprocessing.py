#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预处理模块测试
"""

import os
import sys
import unittest
import logging

# 添加项目根目录到路径，确保可以导入illufly_tts模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from illufly_tts.preprocessing import LanguageSegmenter, TextNormalizer, ChineseNormalizerAdapter

logging.basicConfig(level=logging.INFO)

class TestLanguageSegmenter(unittest.TestCase):
    """测试语言分段器功能"""
    
    def setUp(self):
        self.segmenter = LanguageSegmenter()
        
    def test_segment_simple(self):
        """测试简单文本分段"""
        text = "这是中文，This is English."
        segments = self.segmenter.process_text(text)
        
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0]["lang"], "zh")
        self.assertEqual(segments[0]["text"], "这是中文，")
        self.assertEqual(segments[1]["lang"], "en")
        self.assertEqual(segments[1]["text"], "This is English")
        self.assertEqual(segments[2]["lang"], "zh")  # 句号跟随上下文语言
        
    def test_segment_mixed(self):
        """测试混合文本分段"""
        text = "今天的温度是25℃，weather is good。"
        segments = self.segmenter.process_text(text)
        
        self.assertGreaterEqual(len(segments), 3)
        self.assertEqual(segments[0]["lang"], "zh")
        
        # 检查温度数字是否被识别为英文
        found_number = False
        for segment in segments:
            if "25" in segment["text"]:
                found_number = True
                self.assertEqual(segment["lang"], "en")
                break
        self.assertTrue(found_number)
        
    def test_empty_text(self):
        """测试空文本"""
        segments = self.segmenter.process_text("")
        self.assertEqual(len(segments), 0)
        
    def test_merge_small_segments(self):
        """测试小片段合并功能"""
        # 包含很多短小标点的文本
        text = "测试，test，123，好的"
        segments = self.segmenter.process_text(text)
        
        # 期望标点被合并入相邻段落
        self.assertLess(len(segments), 7)  # 如果不合并会有7个或更多段落
        
class TestTextNormalizer(unittest.TestCase):
    """测试文本标准化功能"""
    
    def setUp(self):
        # 尝试使用测试用词典
        dict_path = os.path.join(os.path.dirname(__file__), '../illufly_tts/resources/dictionaries/replacements.json')
        self.normalizer = TextNormalizer(dict_path if os.path.exists(dict_path) else None)
        
    def test_normalize_numbers(self):
        """测试数字标准化"""
        # 测试整数
        text = "这里有123个苹果"
        normalized = self.normalizer.normalize(text)
        self.assertIn("一二三", normalized)
        
        # 测试小数
        text = "温度是36.5度"
        normalized = self.normalizer.normalize(text)
        self.assertIn("三六", normalized)
        self.assertIn("点", normalized)
        self.assertIn("五", normalized)
        
    def test_english_protection(self):
        """测试英文保护"""
        text = "I'm using TTS technology"
        protected, preserved = self.normalizer.protect_english(text)
        
        # 检查英文是否被替换为占位符
        self.assertNotIn("I'm", protected)
        self.assertNotIn("using", protected)
        self.assertNotIn("TTS", protected)
        self.assertNotIn("technology", protected)
        
        # 恢复保护的内容
        restored = self.normalizer.restore_protected(protected, preserved)
        self.assertEqual(text, restored)
        
    def test_dictionary_application(self):
        """测试词典应用"""
        # 如果词典可用，测试替换功能
        if os.path.exists(os.path.join(os.path.dirname(__file__), '../illufly_tts/resources/dictionaries/replacements.json')):
            text = "TTS是一种AI技术"
            normalized = self.normalizer.normalize(text)
            self.assertIn("语音合成", normalized)
            self.assertIn("人工智能", normalized)
        else:
            self.skipTest("替换词典不可用")
            
    def test_split_sentences(self):
        """测试句子分割"""
        text = "第一句。第二句！第三句？最后一句。"
        sentences = self.normalizer.split_sentences(text)
        self.assertEqual(len(sentences), 4)
        self.assertEqual(sentences[0], "第一句。")
        self.assertEqual(sentences[1], "第二句！")
        self.assertEqual(sentences[2], "第三句？")
        self.assertEqual(sentences[3], "最后一句。")
        
class TestChineseNormalizerAdapter(unittest.TestCase):
    """测试中文标准化适配器"""
    
    def setUp(self):
        dict_path = os.path.join(os.path.dirname(__file__), '../illufly_tts/resources/dictionaries/replacements.json')
        self.adapter = ChineseNormalizerAdapter(dictionary_path=dict_path if os.path.exists(dict_path) else None)
        
    def test_basic_normalization(self):
        """测试基本标准化功能"""
        text = "测试AI和123"
        normalized = self.adapter.process_text(text)
        
        # 基础规则应该生效
        if hasattr(self.adapter, 'misaki_available') and self.adapter.misaki_available:
            self.assertNotEqual(text, normalized)
        else:
            # 如果没有misaki，应该使用基础标准化
            self.assertIn("一二三", normalized)
            
    def test_english_preservation(self):
        """测试英文保护"""
        text = "混合English和中文的句子"
        normalized = self.adapter.process_text(text)
        
        # 英文应该被保留
        self.assertIn("English", normalized)
        
if __name__ == '__main__':
    unittest.main() 