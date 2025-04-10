import os
import sys
import logging
import jwt
from fastapi import FastAPI, Depends, Request
from fastapi.testclient import TestClient

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 设置环境变量
os.environ["FASTAPI_SECRET_KEY"] = "testkey"
os.environ["FASTAPI_ALGORITHM"] = "HS256"

# 添加源码路径
sys.path.append(".")

# 导入我们需要测试的模块
from src.illufly_tts.api.auth import require_user, TokenVerifier
from src.illufly_tts.api.endpoints import mount_tts_service

# 创建一个测试JWT令牌
def create_test_token():
    payload = {
        "user_id": "test_user_id",
        "username": "test_user",
        "roles": ["user", "admin"],
        "iat": 1617000000,
        "exp": 4617000000  # 非常长的过期时间
    }
    return jwt.encode(
        payload=payload,
        key=os.environ["FASTAPI_SECRET_KEY"],
        algorithm=os.environ["FASTAPI_ALGORITHM"]
    )

# 创建一个FastAPI应用
app = FastAPI()

# 定义一个简单的模拟用户认证函数
async def mock_require_user():
    return {"user_id": "test_user_id", "username": "test_user"}

# 将TTS服务挂载到应用上
mount_tts_service(
    app=app, 
    require_user=None,  # 使用我们的JWT验证，而不是外部传入的验证函数
    repo_id="test_repo",
    output_dir="./test_output"
)

# 创建一个测试客户端
client = TestClient(app)

# 测试一个需要JWT验证的端点
def test_jwt_auth():
    token = create_test_token()
    
    # 使用Authorization头测试
    response = client.get(
        "/api/tts/voices",
        headers={"Authorization": f"Bearer {token}"}
    )
    print(f"Authorization头测试: {response.status_code}")
    print(response.json())
    
    # 使用Cookie测试
    response = client.get(
        "/api/tts/voices",
        cookies={"access_token": token}
    )
    print(f"Cookie测试: {response.status_code}")
    print(response.json())
    
    # 无令牌测试
    response = client.get("/api/tts/voices")
    print(f"无令牌测试: {response.status_code}")
    print(response.text)

if __name__ == "__main__":
    print("开始测试JWT验证...")
    test_jwt_auth()
    print("测试完成！") 