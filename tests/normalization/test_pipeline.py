#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试pipeline.py中的文本预处理功能
"""

import re
import pytest
from unittest.mock import Mock, patch, MagicMock

from illufly_tts.pipeline import TTSPipeline
from illufly_tts.normalization.zh.text_normalization import ZhTextNormalizer
from illufly_tts.normalization.en.text_normalization import EnTextNormalizer


class TestPipelinePreprocessing:
    """测试pipeline预处理功能"""

    @pytest.fixture
    def mock_pipeline(self):
        """创建带有mock依赖的TTSPipeline实例"""
        # 创建实际的文本规范化器
        zh_normalizer = ZhTextNormalizer()
        en_normalizer = EnTextNormalizer()
        
        # 创建pipeline实例
        with patch('illufly_tts.pipeline.KModel') as mock_kmodel:
            # 配置mock模型
            mock_model = MagicMock()
            mock_kmodel.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = mock_model
            
            # 创建pipeline
            pipeline = TTSPipeline(
                repo_id="mock-repo",
                voices_dir="/tmp/voices",
                device="cpu"
            )
            
            # 替换规范化器为实际实例
            pipeline.zh_normalizer = zh_normalizer
            pipeline.en_normalizer = en_normalizer
            
            # 使用mock g2p（我们只关注预处理阶段）
            pipeline.g2p = Mock()
            
            yield pipeline

    def test_pure_chinese_text(self, mock_pipeline):
        """测试纯中文文本处理"""
        text = "今天是2023年5月10日，天气真好。"
        result = mock_pipeline.preprocess_text(text)
        
        # 验证结果包含规范化后的日期
        assert "二零二三年五月十日" in result
        
    def test_pure_english_text(self, mock_pipeline):
        """测试纯英文文本处理"""
        text = "Today is May 10th, 2023. The weather is nice."
        result = mock_pipeline.preprocess_text(text)
        
        # 验证包含规范化后的日期
        assert "May" in result
        assert "tenth" in result or "10th" in result
        assert "twenty twenty three" in result or "two thousand twenty three" in result
        
    def test_mixed_chinese_english_text(self, mock_pipeline):
        """测试中英混合文本处理"""
        text = "今天是May 10th，气温25°C，真是a beautiful day！"
        result = mock_pipeline.preprocess_text(text)
        
        # 验证中文部分规范化
        assert "今天是" in result
        assert "气温二十五度" in result or "气温二十五摄氏度" in result
        
        # 验证英文部分规范化
        assert "May" in result
        assert "beautiful day" in result
        
    def test_chinese_with_special_characters(self, mock_pipeline):
        """测试包含特殊符号的中文文本"""
        text = "价格是￥1234.56元，占比75%，电话是13812345678。"
        result = mock_pipeline.preprocess_text(text)
        
        # 验证规范化结果
        assert "价格是" in result
        assert "一千二百三十四点五六元" in result
        assert "百分之七十五" in result
        assert "一三八一二三四五六七八" in result
        
    def test_english_with_special_characters(self, mock_pipeline):
        """测试包含特殊符号的英文文本"""
        text = "The price is $1234.56, which is 75% of the total. Call +1-234-567-8900."
        result = mock_pipeline.preprocess_text(text)
        
        # 验证规范化结果
        assert "price" in result
        assert "dollars" in result or "one thousand" in result
        assert "percent" in result or "seventy five percent" in result
        assert "Call" in result
        
    def test_mixed_text_with_special_characters(self, mock_pipeline):
        """测试包含特殊符号的中英混合文本"""
        text = "购买iPhone 13 Pro的价格是$999.99，折合人民币约￥6400元。"
        result = mock_pipeline.preprocess_text(text)
        
        # 验证规范化结果
        assert "购买" in result
        assert "iPhone" in result
        assert "价格是" in result
        assert "折合人民币约" in result
        assert "六千四百元" in result
        
    def test_number_processing(self, mock_pipeline):
        """测试数字处理（中英文环境）"""
        # 中文环境数字
        cn_text = "这个班有42名学生，其中女生占比约为2/3。"
        cn_result = mock_pipeline.preprocess_text(cn_text)
        assert "四十二名" in cn_result
        assert "三分之二" in cn_result
        
        # 英文环境数字
        en_text = "There are 42 students in this class, about 2/3 of them are girls."
        en_result = mock_pipeline.preprocess_text(en_text)
        assert "forty" in en_result.lower() and "two" in en_result
        assert "two" in en_result and "thirds" in en_result or "third" in en_result
        
    def test_date_time_processing(self, mock_pipeline):
        """测试日期时间处理（中英文环境）"""
        # 中文环境日期时间
        cn_text = "会议安排在2023年6月1日上午10:30开始。"
        cn_result = mock_pipeline.preprocess_text(cn_text)
        assert "二零二三年六月一日" in cn_result
        assert "十点三十分" in cn_result or "十点半" in cn_result
        
        # 英文环境日期时间
        en_text = "The meeting is scheduled for 10:30 AM on June 1st, 2023."
        en_result = mock_pipeline.preprocess_text(en_text)
        assert "ten" in en_result.lower() and "thirty" in en_result
        assert "June" in en_result
        assert "first" in en_result.lower() or "1st" in en_result
        assert "twenty twenty three" in en_result.lower() or "two thousand twenty three" in en_result.lower()
        
    def test_currency_processing(self, mock_pipeline):
        """测试货币处理（中英文环境）"""
        # 中文环境货币
        cn_text = "这件商品原价￥1299.99，现在降价到￥999元。"
        cn_result = mock_pipeline.preprocess_text(cn_text)
        assert "一千二百九十九点九九" in cn_result
        assert "九百九十九元" in cn_result
        
        # 英文环境货币
        en_text = "This product was originally $1299.99, now reduced to $999."
        en_result = mock_pipeline.preprocess_text(en_text)
        assert "dollar" in en_result.lower() or "dollars" in en_result.lower()
        assert "one thousand" in en_result.lower() or "twelve hundred" in en_result.lower()
        assert "ninety nine" in en_result.lower() or "nine hundred" in en_result.lower()
        
    def test_percentage_processing(self, mock_pipeline):
        """测试百分比处理（中英文环境）"""
        # 中文环境百分比
        cn_text = "此次考试及格率为85.5%，比去年提高了3.2%。"
        cn_result = mock_pipeline.preprocess_text(cn_text)
        assert "百分之八十五点五" in cn_result
        assert "百分之三点二" in cn_result
        
        # 英文环境百分比
        en_text = "The pass rate for this exam is 85.5%, which is 3.2% higher than last year."
        en_result = mock_pipeline.preprocess_text(en_text)
        assert "eighty" in en_result.lower() and "five" in en_result
        assert "percent" in en_result.lower()
        assert "three" in en_result and "two" in en_result
        
    def test_phone_number_processing(self, mock_pipeline):
        """测试电话号码处理（中英文环境）"""
        # 中文环境电话号码
        cn_text = "请拨打客服电话400-123-4567或者13812345678。"
        cn_result = mock_pipeline.preprocess_text(cn_text)
        assert "四零零一二三四五六七" in cn_result.replace(" ", "") or "四零零，一二三，四五六七" in cn_result.replace(" ", "")
        assert "一三八一二三四五六七八" in cn_result.replace(" ", "")
        
        # 英文环境电话号码
        en_text = "Please call our customer service at +1-800-123-4567 or (123) 456-7890."
        en_result = mock_pipeline.preprocess_text(en_text)
        assert "eight hundred" in en_result.lower() or "eight zero zero" in en_result.lower()
        assert "one two three" in en_result.lower() or "one twenty three" in en_result.lower()
        
    def test_complex_mixed_text(self, mock_pipeline):
        """测试复杂混合文本"""
        text = "欢迎来到Apple Store，iPhone 13 Pro (128GB) 售价为¥7999元，折扣价为原价的85%，约$1199.99。详情请致电400-666-8800。The event starts at 10:30 AM on 2023/06/15, 请准时参加！"
        result = mock_pipeline.preprocess_text(text)
        
        # 验证中文部分规范化
        assert "欢迎来到" in result
        assert "售价为" in result
        assert "七千九百九十九元" in result
        assert "折扣价为原价的百分之八十五" in result
        assert "详情请致电" in result
        assert "请准时参加" in result
        
        # 验证英文部分规范化
        assert "Apple Store" in result
        assert "iPhone" in result
        assert "event starts" in result
        assert "ten thirty" in result.lower() or "half past ten" in result.lower()
        assert "June" in result or "jun" in result.lower()
        assert "fifteen" in result.lower() or "fifteenth" in result.lower()
        assert "twenty twenty three" in result.lower() or "two thousand twenty three" in result.lower()
        
    def test_edge_cases(self, mock_pipeline):
        """测试边缘情况"""
        # 空字符串
        assert mock_pipeline.preprocess_text("") == ""
        
        # 单个字符
        assert mock_pipeline.preprocess_text("a") == "a"
        assert mock_pipeline.preprocess_text("啊") == "啊"
        assert mock_pipeline.preprocess_text("1") in ["one", "1", "一"]
        
        # 只有特殊符号
        special_chars = "@#$%^&*()_+-=[]{}|;:,./<>?"
        result = mock_pipeline.preprocess_text(special_chars)
        assert len(result) > 0  # 应当有处理结果
        
        # 极长文本的片段处理
        long_cn = "中文" * 50
        long_en = "English " * 50
        mixed_long = long_cn + long_en
        result = mock_pipeline.preprocess_text(mixed_long)
        assert len(result) > 0
        assert "中文" in result
        assert "English" in result

    def test_special_symbol_boundary_cases(self, mock_pipeline):
        """测试特殊符号边界情况"""
        # 英文中包含特殊符号但不应被分割
        text1 = "Please visit www.example.com or contact info@example.com."
        result1 = mock_pipeline.preprocess_text(text1)
        assert "www.example.com" in result1
        assert "info@example.com" in result1
        
        # 中文中包含特殊符号但不应被分割
        text2 = "请访问www.example.com或发邮件至info@example.com。"
        result2 = mock_pipeline.preprocess_text(text2)
        assert "www.example.com" in result2
        assert "info@example.com" in result2
        
        # 带特殊符号的英文缩写
        text3 = "项目已完成50%，距离目标还有9.5km，请于A.S.A.P.完成。"
        result3 = mock_pipeline.preprocess_text(text3)
        assert "百分之五十" in result3
        assert "九点五千米" in result3 or "九点五公里" in result3
        assert "A.S.A.P" in result3 or "ASAP" in result3
        
        # 中英文混合的产品型号
        text4 = "iPhone-13Pro和Galaxy S22-Ultra都是高端手机。"
        result4 = mock_pipeline.preprocess_text(text4)
        assert "iPhone" in result4
        assert "13" in result4 or "thirteen" in result4.lower()
        assert "Pro" in result4
        assert "Galaxy" in result4
        assert "S22" in result4 or "S twenty two" in result4.lower()
        assert "Ultra" in result4
        assert "高端手机" in result4

    def test_sentence_boundary_detection(self, mock_pipeline):
        """测试句子边界检测"""
        # 中文句子边界
        cn_text = "今天天气真好。明天可能会下雨！后天将会放晴？我们拭目以待。"
        cn_result = mock_pipeline.preprocess_text(cn_text)
        assert "今天天气真好" in cn_result
        assert "明天可能会下雨" in cn_result
        assert "后天将会放晴" in cn_result
        assert "我们拭目以待" in cn_result
        
        # 英文句子边界
        en_text = "The weather is nice today. It might rain tomorrow! It will be sunny the day after? We shall see."
        en_result = mock_pipeline.preprocess_text(en_text)
        assert "weather is nice today" in en_result
        assert "might rain tomorrow" in en_result
        assert "will be sunny" in en_result
        assert "shall see" in en_result
        
        # 混合句子边界
        mixed_text = "今天是fine day。Tomorrow可能会下雨！Let's wait and see。"
        mixed_result = mock_pipeline.preprocess_text(mixed_text)
        assert "今天是" in mixed_result
        assert "fine day" in mixed_result
        assert "Tomorrow" in mixed_result
        assert "可能会下雨" in mixed_result
        assert "Let's wait and see" in mixed_result or "Let us wait and see" in mixed_result