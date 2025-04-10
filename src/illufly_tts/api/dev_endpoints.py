"""
开发模式下的API端点
这个模块包含所有开发测试相关的API端点
"""
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from .dev_mode import generate_dev_token, is_dev_mode, DEV_SECRET_KEY

logger = logging.getLogger(__name__)

# 开发模式下使用的请求模型
class DevTokenRequest(BaseModel):
    """开发测试令牌请求"""
    user_id: str = "dev_tester"
    secret_key: Optional[str] = None

def create_dev_router() -> APIRouter:
    """创建开发模式路由器"""
    if not is_dev_mode():
        return APIRouter()
        
    router = APIRouter(prefix="/dev", tags=["development"])
    
    @router.post("/token")
    async def get_dev_token(request: DevTokenRequest) -> Dict[str, Any]:
        """获取开发测试令牌（仅开发模式可用）"""
        if not is_dev_mode():
            raise HTTPException(status_code=404, detail="此端点仅在开发模式下可用")
        
        # 从auth模块导入这个变量
        from .auth import JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        
        secret = request.secret_key or DEV_SECRET_KEY
        token = generate_dev_token(request.user_id, secret)
        
        return {
            "token": token,
            "token_type": "bearer",
            "user_id": request.user_id,
            "expires_in": 60 * JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
            "dev_mode": True
        }
    
    @router.get("/status")
    async def dev_status() -> Dict[str, Any]:
        """获取开发模式状态信息"""
        import os
        
        # 从auth模块导入这个变量
        from .auth import JWT_COOKIE_NAME
        
        return {
            "dev_mode": True,
            "status": "active",
            "env_vars": {
                "TTS_DEV_MODE": os.environ.get("TTS_DEV_MODE", "<未设置>"),
                "TTS_DEV_SECRET_KEY": "******" if DEV_SECRET_KEY != "development_secret_key" else "<默认值>",
                "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"),
                "JWT_COOKIE_NAME": JWT_COOKIE_NAME
            }
        }
    
    return router 