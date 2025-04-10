import pytest
import asyncio
import time
import sys
import os
from pathlib import Path
import json
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

# 添加源码路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.illufly_tts.core.service import TTSServiceManager, TTSTask, TaskStatus
from src.illufly_tts.api.endpoints import TextToSpeechRequest, mount_tts_service

# 创建一个模拟的require_user函数，用于API测试
async def mock_require_user():
    """模拟用户认证函数"""
    return {"user_id": "test_user", "roles": ["user"]}

# 创建应用和测试客户端
app = FastAPI()
mount_tts_service(app, mock_require_user, repo_id="test_repo", voices_dir=None, output_dir="./test_output")
client = TestClient(app)

@pytest.mark.asyncio
async def test_api_cancel_pending():
    """测试通过API使用cancel_pending参数取消用户待处理任务"""
    # 模拟服务管理器
    mock_service_manager = MagicMock(spec=TTSServiceManager)
    
    # 模拟取消用户待处理任务的方法
    mock_service_manager.cancel_user_pending_tasks = MagicMock(return_value=3)
    
    # 模拟submit_task方法
    async def mock_submit_task(*args, **kwargs):
        return "test_task_id"
    mock_service_manager.submit_task = mock_submit_task
    
    # 模拟get_task_status方法
    async def mock_get_task_status(*args, **kwargs):
        return {
            "task_id": "test_task_id",
            "status": "completed",
            "created_at": time.time(),
            "completed_at": time.time(),
            "text": "test text",
            "voice_id": "zf_001",
            "user_id": "test_user",
            "chunks_completed": 1
        }
    mock_service_manager.get_task_status = mock_get_task_status
    
    # 模拟服务管理器的output_dir属性
    mock_service_manager.output_dir = "./test_output"
    
    # 使用临时目录作为output_dir
    os.makedirs("./test_output", exist_ok=True)
    
    # 创建一个空的WAV文件用于测试
    with open("./test_output/test_task_id.wav", "wb") as f:
        f.write(b"\x00" * 44)  # 写入一个空的WAV头
    
    # 使用patch替换get_service_manager
    with patch("src.illufly_tts.api.endpoints.get_service_manager", return_value=mock_service_manager):
        # 发送带有cancel_pending=True的请求
        response = client.post(
            "/api/tts",
            json={
                "text": "test text",
                "voice_id": "zf_001",
                "cancel_pending": True
            }
        )
        
        # 验证响应
        assert response.status_code == 200
        
        # 验证cancel_user_pending_tasks方法被调用
        mock_service_manager.cancel_user_pending_tasks.assert_called_once_with("test_user")
        
        # 发送不带cancel_pending的请求
        response = client.post(
            "/api/tts",
            json={
                "text": "test text",
                "voice_id": "zf_001"
            }
        )
        
        # 验证响应
        assert response.status_code == 200
        
        # 验证cancel_user_pending_tasks方法没有被再次调用
        assert mock_service_manager.cancel_user_pending_tasks.call_count == 1
    
    # 清理
    os.remove("./test_output/test_task_id.wav")
    os.rmdir("./test_output")

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 