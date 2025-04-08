import pytest
import pytest_asyncio
import asyncio
import time
import os
import logging
import sys
from pathlib import Path
from unittest.mock import patch
import traceback

from illufly_tts.service import TTSServiceManager, TaskStatus

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("test_integration")

# 创建全局变量存储patcher
_patcher = None

@pytest_asyncio.fixture
async def patched_service_manager(mock_pipeline, temp_dir, event_loop):
    """创建一个打了补丁的服务管理器实例，用于集成测试"""
    global _patcher
    logger.debug("创建服务管理器开始")
    
    # 使用模拟Pipeline
    _patcher = patch('illufly_tts.service.CachedTTSPipeline', return_value=mock_pipeline)
    _patcher.start()
    logger.debug("应用Mock替换完成")
    
    # 创建服务管理器
    manager = TTSServiceManager(
        repo_id="mock_repo",
        voices_dir="mock_voices",
        device="cpu",
        batch_size=1,
        max_wait_time=0.1,
        chunk_size=10,
        output_dir=temp_dir
    )
    
    logger.debug("服务管理器创建完成")
    
    # 使用标准的yield模式
    yield manager
    
    # 清理代码会在yield后执行
    logger.debug("开始清理服务管理器")
    if hasattr(manager, 'processing_task') and not manager.processing_task.done():
        logger.debug("取消处理任务")
        manager.processing_task.cancel()
        try:
            await asyncio.wait_for(manager.processing_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.debug("处理任务取消超时或已被取消")
        except Exception as e:
            logger.error(f"取消处理任务时出错: {e}")
    
    await manager.shutdown()
    logger.debug("服务管理器已关闭")
    
    # 停止patcher
    if _patcher:
        logger.debug("停止Mock替换")
        _patcher.stop()
        _patcher = None
    logger.debug("服务管理器清理完成")

@pytest.mark.asyncio
async def test_full_task_lifecycle(patched_service_manager):
    """测试任务的完整生命周期：提交、处理、完成"""
    service_manager = patched_service_manager  # 重命名变量使更清晰
    logger.info("开始集成测试: 任务完整生命周期")
    
    # 提交任务
    logger.info("提交语音合成任务...")
    task_id = await service_manager.submit_task("这是一个集成测试文本", "test_voice")
    logger.info(f"任务已提交，ID: {task_id}")
    
    # 获取并打印初始状态
    status = await service_manager.get_task_status(task_id)
    logger.info(f"任务初始状态: {status['status']}")
    
    # 等待任务完成
    logger.info("等待任务完成...")
    max_wait = 10  # 最多等待10秒
    start_time = time.time()
    completed = False
    
    while time.time() - start_time < max_wait:
        await asyncio.sleep(0.2)
        status = await service_manager.get_task_status(task_id)
        logger.info(f"任务当前状态: {status['status']}")
        
        if status["status"] in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]:
            completed = True
            logger.info(f"任务已完成，最终状态: {status['status']}")
            break
    
    if not completed:
        logger.error(f"任务未在{max_wait}秒内完成，当前状态: {status['status']}")
        # 检查任务队列和处理任务
        logger.debug(f"任务队列大小: {service_manager.task_queue.qsize()}")
        if hasattr(service_manager, 'processing_task'):
            process_status = '已完成' if service_manager.processing_task.done() else '运行中'
            logger.debug(f"处理任务状态: {process_status}")
            if service_manager.processing_task.done():
                try:
                    exception = service_manager.processing_task.exception()
                    if exception:
                        logger.error(f"处理任务异常: {exception}")
                except Exception as e:
                    logger.error(f"获取处理任务异常时出错: {e}")
    
    # 断言任务已完成
    assert completed, f"任务未在{max_wait}秒内完成"
    assert status["status"] == TaskStatus.COMPLETED.value, f"任务状态不是已完成: {status['status']}"

@pytest.mark.asyncio
async def test_multiple_tasks(patched_service_manager):
    """测试同时提交多个任务"""
    service_manager = patched_service_manager  # 重命名变量使更清晰
    logger.info("开始集成测试: 多任务测试")
    
    # 提交多个任务
    task_ids = []
    for i in range(3):
        task_id = await service_manager.submit_task(f"这是测试文本 {i+1}", "test_voice")
        task_ids.append(task_id)
        logger.info(f"提交任务 {i+1}，ID: {task_id}")
    
    # 等待所有任务完成
    logger.info("等待所有任务完成...")
    max_wait = 15  # 最多等待15秒
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        await asyncio.sleep(0.5)
        
        # 检查所有任务状态
        statuses = [await service_manager.get_task_status(task_id) for task_id in task_ids]
        status_values = [s["status"] for s in statuses]
        logger.info(f"当前任务状态: {status_values}")
        
        # 如果所有任务都已完成或失败
        if all(s in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value] for s in status_values):
            logger.info("所有任务已完成")
            break
    
    # 断言所有任务都已完成
    final_statuses = [await service_manager.get_task_status(task_id) for task_id in task_ids]
    for i, status in enumerate(final_statuses):
        assert status["status"] == TaskStatus.COMPLETED.value, f"任务 {i+1} 未完成，状态: {status['status']}" 