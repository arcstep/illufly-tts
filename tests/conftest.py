import pytest
import asyncio
import torch
import os
import logging
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("test_utils")

# 模拟 Pipeline 类
class MockPipeline:
    def __init__(self, *args, **kwargs):
        logger.info("创建模拟Pipeline")
        self.sample_rate = 24000
        self.segment_text = MagicMock(return_value=["测试文本片段"])
        
        async def mock_process(*args, **kwargs):
            logger.debug("模拟处理开始")
            await asyncio.sleep(0.2)  # 模拟处理时间
            logger.debug("模拟处理完成")
            # 返回一个简单的音频张量模拟
            return [torch.tensor([0.1, 0.2, 0.3, 0.4] * 1000).float()]
        
        self.batch_process_texts = AsyncMock(side_effect=mock_process)

@pytest.fixture
def mock_pipeline():
    """返回一个模拟的TTS Pipeline实例"""
    logger.debug("创建 mock_pipeline")
    return MockPipeline()

@pytest.fixture
def temp_dir(tmpdir):
    """创建一个临时目录用于测试输出"""
    logger.debug(f"创建临时目录: {tmpdir}")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

@pytest.fixture
def event_loop():
    """创建一个新的事件循环，用于测试"""
    logger.debug("创建新的事件循环")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    logger.debug("关闭事件循环")
    loop.close() 