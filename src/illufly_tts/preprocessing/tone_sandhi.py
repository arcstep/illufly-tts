#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文声调变化处理模块
ADAPTED from https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/frontend/zh_frontend.py
"""

import re
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class ToneSandhi:
    """中文声调变化处理类"""
    
    def __init__(self):
        """初始化声调变化处理类"""
        # 定义需要和后一个词语合并的复姓词语
        self.must_not_neural_tone_words = {
            "多", "麼", "嘛", "呢", "吧", "啦", "啊", "呀", "吖", "噢", "呐", "哈",
            "哒", "么", "嘞", "啰", "嘛", "耶", "哩", "啰", "噢"
        }
        self.must_neural_tone_words = {
            "们", "了", "过", "的", "地", "得", "着", "不", "别"
        }
        self.neural_tone_words = {
            "们", "么", "了", "过", "的", "地", "得", "着", "上", "去", "起", "不", "别"
        }
    
    def pre_merge_for_modify(self, seg_cut: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """合并复姓词语
        
        Args:
            seg_cut: 分词结果，每个元素是(词语, 词性)元组
            
        Returns:
            处理后的分词结果
        """
        # 这里暂时不进行任何合并，直接返回原始切分
        return seg_cut
    
    def _neural_sandhi(self, finals: List[str]) -> List[str]:
        """处理轻声音节
        
        Args:
            finals: 韵母列表
            
        Returns:
            处理后的韵母列表
        """
        # 一个简单的轻声处理示例
        # 将末尾"5"声调标记的音节改为轻声
        for i in range(len(finals)):
            if finals[i].endswith('5'):
                # 确保格式正确，去掉原声调标记，加上轻声标记
                finals[i] = re.sub(r'\d', '', finals[i]) + '5'
        return finals
    
    def _bu_sandhi(self, word: str, pos: str, finals: List[str]) -> List[str]:
        """处理"不"的变调
        
        'bu4' -> 'bu2' when followed by a 4th tone character
        
        Args:
            word: 词语
            pos: 词性
            finals: 韵母列表
            
        Returns:
            处理后的韵母列表
        """
        # 检查是否需要处理
        if word != "不" or pos != "d" or len(finals) != 1:
            return finals
            
        # 如果下一个字是四声，"不"变成二声
        if len(finals) > 1 and finals[1][-1] == '4':
            finals[0] = finals[0][:-1] + '2'
        
        return finals
    
    def _yi_sandhi(self, word: str, pos: str, finals: List[str]) -> List[str]:
        """处理"一"的变调
        
        'yi1' -> 'yi4' when followed by a 1st/2nd/3rd tone character
        'yi1' -> 'yi2' when followed by a 4th tone character
        
        Args:
            word: 词语
            pos: 词性
            finals: 韵母列表
            
        Returns:
            处理后的韵母列表
        """
        # 检查是否需要处理
        if word != "一" or pos != "m" or len(finals) != 1:
            return finals
            
        # 根据后一个字的声调调整"一"的声调
        if len(finals) > 1:
            next_tone = finals[1][-1]
            if next_tone in '123':
                finals[0] = finals[0][:-1] + '4'
            elif next_tone == '4':
                finals[0] = finals[0][:-1] + '2'
        
        return finals
    
    def _third_tone_sandhi(self, finals: List[str]) -> List[str]:
        """处理上声连读变调
        
        3rd tone + 3rd tone -> 2nd tone + 3rd tone
        
        Args:
            finals: 韵母列表
            
        Returns:
            处理后的韵母列表
        """
        # 检查三声连读，将相邻的三声改为二声+三声
        for i in range(len(finals) - 1):
            if finals[i][-1] == '3' and finals[i+1][-1] == '3':
                finals[i] = finals[i][:-1] + '2'
                
        return finals
    
    def modified_tone(self, word: str, pos: str, finals: List[str]) -> List[str]:
        """声调变化主函数
        
        Args:
            word: 词语
            pos: 词性
            finals: 韵母列表
            
        Returns:
            处理后的韵母列表
        """
        if not finals:
            return finals
            
        # 按顺序应用各种声调规则
        # 处理轻声
        finals = self._neural_sandhi(finals)
        
        # 处理"不"的变调
        finals = self._bu_sandhi(word, pos, finals)
        
        # 处理"一"的变调
        finals = self._yi_sandhi(word, pos, finals)
        
        # 处理三声连读
        finals = self._third_tone_sandhi(finals)
        
        return finals 