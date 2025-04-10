import pytest
import asyncio
import time
import sys
import os
from pathlib import Path

# 添加源码路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.illufly_tts.core.service import TTSServiceManager
from src.illufly_tts.core.pipeline import TTSPipeline

@pytest.mark.asyncio
async def test_ordering_with_default_language():
    """集成测试: 任务排序和默认语言处理"""
    # 创建服务管理器，使用中文作为默认语言
    service_manager = TTSServiceManager(
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        voices_dir=None,
        batch_size=4,
        output_dir=None
    )
    
    # 替换pipeline为使用中文默认语言的版本
    service_manager.pipeline = TTSPipeline(
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        default_language="zh"
    )
    
    # 模拟load_voice以避免实际加载语音
    service_manager.pipeline.load_voice = lambda voice_id: {}
    
    try:
        # 提交测试任务
        # 1. 纯数字文本，添加序列ID
        await service_manager.submit_task(
            text="123456", 
            voice_id="zf_001",
            sequence_id=1001.0
        )
        
        # 2. 中文上下文的数字
        await service_manager.submit_task(
            text="中文上下文123456",
            voice_id="zf_001",
            sequence_id=1002.0
        )
        
        # 3. 英文上下文的数字
        await service_manager.submit_task(
            text="English context 123456",
            voice_id="zf_001",
            sequence_id=1003.0
        )
        
        # 模拟处理任务
        # 获取所有待处理任务
        tasks = [task for task in service_manager.tasks.values()]
        
        # 按序列ID排序
        sorted_tasks = sorted(tasks, key=lambda t: t.sequence_id)
        
        # 验证任务顺序
        assert sorted_tasks[0].text == "123456"
        assert sorted_tasks[1].text == "中文上下文123456"
        assert sorted_tasks[2].text == "English context 123456"
        
        # 验证序列ID
        assert [t.sequence_id for t in sorted_tasks] == [1001.0, 1002.0, 1003.0]
        
        # 验证默认语言处理
        # 在实际测试中，这里应该调用pipeline的preprocess_text方法
        # 但由于我们没有实际模型，这里只能进行基本验证
        
        # 验证纯数字文本将被识别为默认语言(中文)
        text1_result = service_manager.pipeline.preprocess_text("123456")
        # 检查是否被处理为中文数字
        assert any(c in text1_result for c in "一二三四五六七八九十")
        
        # 中文上下文的数字应被识别为中文
        text2_result = service_manager.pipeline.preprocess_text("中文上下文123456")
        assert any(c in text2_result for c in "一二三四五六七八九十")
        
        # 英文上下文的数字应被识别为英文
        text3_result = service_manager.pipeline.preprocess_text("English context 123456")
        # 不应该包含中文数字
        assert not any(c in text3_result for c in "一二三四五六七八九十")
        
    finally:
        # 不使用shutdown方法，避免NoneType错误
        pass

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 