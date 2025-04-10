import pytest
import asyncio
import time
import sys
import os
from pathlib import Path
import torch

# 添加源码路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.illufly_tts.core.service import TTSServiceManager, TTSTask, TaskStatus

@pytest.mark.asyncio
async def test_task_sequence_ordering():
    """测试任务按序列ID顺序处理"""
    # 模拟服务管理器
    service_manager = TTSServiceManager(
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        voices_dir=None,
        batch_size=4
    )
    
    # 保存原始方法以便稍后恢复
    original_load_voice = service_manager.pipeline.load_voice
    
    # 模拟load_voice方法避免实际加载
    async def mock_load_voice(self, *args, **kwargs):
        return {}
    
    # 替换方法
    service_manager.pipeline.load_voice = lambda voice_id: {}
    
    try:
        # 提交任务，故意使用相反顺序的序列ID
        task3_id = await service_manager.submit_task(
            text="第三个任务", 
            voice_id="zf_001", 
            sequence_id=1003.0
        )
        
        task2_id = await service_manager.submit_task(
            text="第二个任务", 
            voice_id="zf_001", 
            sequence_id=1002.0
        )
        
        task1_id = await service_manager.submit_task(
            text="第一个任务", 
            voice_id="zf_001", 
            sequence_id=1001.0
        )
        
        # 验证任务是否按提交顺序保存
        tasks = [
            service_manager.tasks[task1_id],
            service_manager.tasks[task2_id],
            service_manager.tasks[task3_id]
        ]
        
        # 按用户ID分组，模拟批处理循环中的逻辑
        tasks_by_user = {}
        for task in tasks:
            user_id = task.user_id or "anonymous"
            if user_id not in tasks_by_user:
                tasks_by_user[user_id] = []
            tasks_by_user[user_id].append(task)
        
        # 对每个用户的任务按序列ID排序
        for user_id, user_tasks in tasks_by_user.items():
            user_tasks.sort(key=lambda t: t.sequence_id)
            # 验证排序后的顺序是否正确
            assert [t.text for t in user_tasks] == ["第一个任务", "第二个任务", "第三个任务"]
            assert [t.sequence_id for t in user_tasks] == [1001.0, 1002.0, 1003.0]
    
    finally:
        # 恢复原始方法
        service_manager.pipeline.load_voice = original_load_voice
        # 不使用shutdown方法，直接让测试结束

@pytest.mark.asyncio
async def test_task_sequence_multiple_users():
    """测试多用户场景下的任务序列排序"""
    # 模拟服务管理器
    service_manager = TTSServiceManager(
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        voices_dir=None,
        batch_size=4
    )
    
    # 模拟load_voice方法
    service_manager.pipeline.load_voice = lambda voice_id: {}
    
    try:
        # 用户1的任务
        await service_manager.submit_task(
            text="用户1的第二个任务", 
            voice_id="zf_001",
            user_id="user1",
            sequence_id=2002.0
        )
        
        await service_manager.submit_task(
            text="用户1的第一个任务", 
            voice_id="zf_001",
            user_id="user1",
            sequence_id=2001.0
        )
        
        # 用户2的任务
        await service_manager.submit_task(
            text="用户2的第二个任务", 
            voice_id="zf_001",
            user_id="user2",
            sequence_id=3002.0
        )
        
        await service_manager.submit_task(
            text="用户2的第一个任务", 
            voice_id="zf_001",
            user_id="user2",
            sequence_id=3001.0
        )
        
        # 收集待处理的任务
        pending_tasks = [task for task in service_manager.tasks.values() 
                        if task.status == TaskStatus.PENDING]
        
        # 按用户ID分组
        tasks_by_user = {}
        for task in pending_tasks:
            user_id = task.user_id or "anonymous"
            if user_id not in tasks_by_user:
                tasks_by_user[user_id] = []
            tasks_by_user[user_id].append(task)
        
        # 从每个用户选择一个任务
        batch_tasks = []
        for user_id, user_tasks in tasks_by_user.items():
            # 按序列ID排序
            user_tasks.sort(key=lambda t: t.sequence_id)
            # 应该选择序列ID最小的任务
            selected_task = user_tasks[0]
            batch_tasks.append(selected_task)
            # 验证选择的是第一个任务
            assert "第一个任务" in selected_task.text
    
    finally:
        pass  # 不再调用shutdown方法

@pytest.mark.asyncio
async def test_continuous_task_submission():
    """测试用户持续追加任务的场景"""
    # 模拟服务管理器
    service_manager = TTSServiceManager(
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        voices_dir=None,
        batch_size=2  # 设置较小的批次大小，便于测试
    )
    
    # 模拟load_voice方法
    service_manager.pipeline.load_voice = lambda voice_id: {}
    
    # 创建一个列表存储处理的任务
    processed_tasks = []
    
    # 创建一个模拟的批处理函数
    async def mock_batch_processing():
        # 收集待处理的任务
        pending_tasks = [task for task in service_manager.tasks.values() 
                         if task.status == TaskStatus.PENDING]
        
        if not pending_tasks:
            return
            
        # 按用户ID分组
        tasks_by_user = {}
        for task in pending_tasks:
            user_id = task.user_id or "anonymous"
            if user_id not in tasks_by_user:
                tasks_by_user[user_id] = []
            tasks_by_user[user_id].append(task)
        
        # 从每个用户选择一个任务
        batch_tasks = []
        for user_id, user_tasks in tasks_by_user.items():
            # 按序列ID排序
            user_tasks.sort(key=lambda t: t.sequence_id)
            # 应该选择序列ID最小的任务
            batch_tasks.append(user_tasks[0])
            
            # 批次已满则停止
            if len(batch_tasks) >= service_manager.batch_size:
                break
        
        # 记录被处理的任务文本
        for task in batch_tasks:
            processed_tasks.append(task.text)
            # 标记任务为已完成
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
    
    try:
        # 模拟第一波提交任务（如同用户收到大模型的第一段回复）
        await service_manager.submit_task(
            text="第一段第1句", voice_id="zf_001", user_id="user1", sequence_id=1001.0
        )
        await service_manager.submit_task(
            text="第一段第2句", voice_id="zf_001", user_id="user1", sequence_id=1002.0
        )
        
        # 模拟处理第一波任务
        await mock_batch_processing()
        await mock_batch_processing()  # 需要处理两次，每次最多处理2个任务
        
        # 模拟第二波提交任务（如同用户收到大模型的第二段回复）
        await service_manager.submit_task(
            text="第二段第1句", voice_id="zf_001", user_id="user1", sequence_id=1003.0
        )
        await service_manager.submit_task(
            text="第二段第2句", voice_id="zf_001", user_id="user1", sequence_id=1004.0
        )
        
        # 模拟处理第二波任务
        await mock_batch_processing()
        await mock_batch_processing()
        
        # 模拟第三波提交任务（如同用户收到大模型的第三段回复）
        await service_manager.submit_task(
            text="第三段第1句", voice_id="zf_001", user_id="user1", sequence_id=1005.0
        )
        
        # 处理第三波任务
        await mock_batch_processing()
        
        # 验证所有任务是否按序列ID顺序处理
        expected_order = [
            "第一段第1句", "第一段第2句", 
            "第二段第1句", "第二段第2句", 
            "第三段第1句"
        ]
        
        # 打印处理结果，方便调试
        print(f"处理的任务: {processed_tasks}")
        print(f"预期的任务: {expected_order}")
        
        assert len(processed_tasks) == len(expected_order), "处理的任务数量不符"
        assert processed_tasks == expected_order, "任务处理顺序不符合预期"
            
    finally:
        pass

@pytest.mark.asyncio
async def test_cancel_user_pending_tasks():
    """测试取消用户待处理任务的功能"""
    # 模拟服务管理器
    service_manager = TTSServiceManager(
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        voices_dir=None,
        batch_size=4
    )
    
    # 模拟load_voice方法
    service_manager.pipeline.load_voice = lambda voice_id: {}
    
    try:
        # 提交多个用户的多个任务
        # 用户1的任务
        task1_1_id = await service_manager.submit_task(
            text="用户1的第一个任务", 
            voice_id="zf_001",
            user_id="user1",
            sequence_id=1001.0
        )
        
        task1_2_id = await service_manager.submit_task(
            text="用户1的第二个任务", 
            voice_id="zf_001",
            user_id="user1",
            sequence_id=1002.0
        )
        
        # 用户2的任务
        task2_1_id = await service_manager.submit_task(
            text="用户2的第一个任务", 
            voice_id="zf_001",
            user_id="user2",
            sequence_id=2001.0
        )
        
        task2_2_id = await service_manager.submit_task(
            text="用户2的第二个任务", 
            voice_id="zf_001",
            user_id="user2",
            sequence_id=2002.0
        )
        
        # 验证所有任务都处于PENDING状态
        for task_id in [task1_1_id, task1_2_id, task2_1_id, task2_2_id]:
            task = service_manager.tasks[task_id]
            assert task.status == TaskStatus.PENDING
        
        # 取消用户1的所有待处理任务
        canceled_count = await service_manager.cancel_user_pending_tasks("user1")
        
        # 验证返回值
        assert canceled_count == 2, f"应取消2个任务，但实际取消了{canceled_count}个"
        
        # 验证用户1的所有任务都被取消，用户2的任务不受影响
        for task_id in [task1_1_id, task1_2_id]:
            task = service_manager.tasks[task_id]
            assert task.status == TaskStatus.CANCELED, f"任务{task_id}应该被取消，但状态为{task.status}"
            
        for task_id in [task2_1_id, task2_2_id]:
            task = service_manager.tasks[task_id]
            assert task.status == TaskStatus.PENDING, f"任务{task_id}不应该被取消，但状态为{task.status}"
        
        # 测试取消一个不存在的用户，应返回0
        no_user_canceled = await service_manager.cancel_user_pending_tasks("no_such_user")
        assert no_user_canceled == 0, f"不存在的用户应返回0，但返回了{no_user_canceled}"
        
        # 测试取消空用户ID，应返回0
        empty_user_canceled = await service_manager.cancel_user_pending_tasks("")
        assert empty_user_canceled == 0, f"空用户ID应返回0，但返回了{empty_user_canceled}"
        
        # 测试取消None用户ID，应返回0
        none_user_canceled = await service_manager.cancel_user_pending_tasks(None)
        assert none_user_canceled == 0, f"None用户ID应返回0，但返回了{none_user_canceled}"
    
    finally:
        pass  # 不调用shutdown方法，直接让测试结束

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 