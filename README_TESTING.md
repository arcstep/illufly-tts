# illufly-tts 测试指南

## 代码结构

此项目的测试代码已经重构为更合理的结构：

1. `tests/conftest.py` - 包含共享的测试工具和fixture，如`MockPipeline`和事件循环管理
2. `tests/test_integration.py` - 包含集成测试，测试TTSServiceManager的完整功能
3. `scripts/run_tts_test.py` - 独立的测试脚本，可以从命令行运行
4. `run_tts_test.py` - 根目录下的入口脚本，是对`scripts/run_tts_test.py`的简单包装

## 运行测试

### 使用pytest运行测试

```bash
# 运行所有测试
poetry run pytest

# 运行特定的测试文件
poetry run pytest tests/test_integration.py

# 运行特定的测试用例（verbose模式）
poetry run pytest tests/test_integration.py::test_full_task_lifecycle -v
```

### 使用独立脚本运行测试

```bash
# 从根目录运行
python run_tts_test.py

# 或者直接运行scripts中的脚本
python scripts/run_tts_test.py
```

## 测试框架说明

### MockPipeline

`MockPipeline`类模拟了真实的TTS Pipeline，它提供：

- 一个`segment_text`方法，返回预定义的文本片段
- 一个异步的`batch_process_texts`方法，模拟处理时间并返回模拟的音频数据

### 事件循环管理

由于`TTSServiceManager`使用asyncio进行异步操作，测试代码需要正确设置事件循环：

1. pytest测试通过`event_loop` fixture提供事件循环
2. 独立脚本通过`asyncio.run()`创建和管理事件循环

### 集成测试

集成测试验证TTSServiceManager的完整功能：

1. 任务提交
2. 状态查询
3. 任务处理完成
4. 多任务并发处理

## 故障排除

如果测试失败或卡住：

1. 检查日志输出，查找异常或错误信息
2. 确认`CachedTTSPipeline`是否被正确替换为`MockPipeline`
3. 检查事件循环是否正确创建和管理
4. 查看timeout设置是否足够长，特别是对多任务测试

## 注意事项

- 这些测试使用MockPipeline，不需要实际的模型或声音文件
- 请不要在源代码包内（如`src/illufly_tts/`）放置测试代码或主脚本
- 测试完成后，服务管理器会自动关闭和清理资源

## 问题概述

```python
@pytest.fixture
def event_loop():
    """创建一个新的事件循环，用于测试"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
```

2. 将`service_manager`fixture改为异步的，并依赖于`event_loop`：

```python
@pytest.fixture
async def service_manager(mock_pipeline, temp_dir, event_loop):
    # ...fixture实现...
```

3. 现在可以这样运行测试：

```bash
# 运行特定测试
poetry run pytest tests/test_minimal.py -v

# 运行所有测试
poetry run pytest
```

## 调试提示

如果测试仍然失败或卡住：

1. 检查日志输出，查找任何异常或错误信息
2. 使用`asyncio.all_tasks()`查看当前正在运行的所有任务
3. 增加日志级别和详细程度来更好地了解程序执行流程
4. 考虑使用超时机制避免测试永久挂起

## 资源

- [asyncio文档](https://docs.python.org/3/library/asyncio.html)
 