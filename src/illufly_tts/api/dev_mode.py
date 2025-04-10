"""
开发模式相关功能模块
这个模块包含所有与开发测试相关的代码，将其与正常业务逻辑分离
"""
import jwt
import logging
import os
from typing import Dict, Any, Optional
from fastapi import HTTPException, Request, status
from datetime import datetime

# 获取日志记录器
logger = logging.getLogger(__name__)

# 从环境变量获取开发模式配置
DEV_MODE = os.environ.get("TTS_DEV_MODE", "").lower() in ["true", "1", "yes"]
DEV_SECRET_KEY = os.environ.get("TTS_DEV_SECRET_KEY", "development_secret_key")

# 从auth模块导入JWT配置（避免重复定义）
from .auth import JWT_ALGORITHM, JWT_ACCESS_TOKEN_EXPIRE_MINUTES

def is_dev_mode() -> bool:
    """检查当前是否为开发模式"""
    # 直接从环境变量读取，避免模块加载时的值可能不准确
    return os.environ.get("TTS_DEV_MODE", "").lower() in ["true", "1", "yes"]

def generate_dev_token(user_id: str, secret_key: str = DEV_SECRET_KEY) -> str:
    """生成开发模式的测试令牌
    
    Args:
        user_id: 用户ID
        secret_key: 用于签名的密钥（默认使用开发密钥）
        
    Returns:
        生成的JWT令牌
    """
    if not is_dev_mode():
        logger.warning("非开发模式下不应该调用此方法")
        return ""
        
    payload = {
        "user_id": user_id,
        "username": f"开发用户_{user_id}",
        "roles": ["user", "admin"],
        "iat": int(datetime.now().timestamp()),
        "exp": int((datetime.now().timestamp() + 60 * JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
    }
    
    token = jwt.encode(payload, secret_key, algorithm=JWT_ALGORITHM)
    logger.warning(f"开发模式: 生成测试令牌，用户ID={user_id}")
    return token

def verify_token_dev_mode(token: str) -> Dict[str, Any]:
    """开发模式下的令牌验证
    
    Args:
        token: JWT令牌
        
    Returns:
        解码后的令牌数据
    """
    logger.warning("开发模式: 使用开发模式验证逻辑")
    
    # 检查是否是特殊的开发测试令牌
    if token == "dev_token":
        logger.warning("开发模式: 使用预设的开发测试令牌")
        return {
            "user_id": "test_user",
            "username": "测试用户",
            "roles": ["user", "admin"],
            "iat": int(datetime.now().timestamp()),
            "exp": int((datetime.now().timestamp() + 3600))
        }
    
    try:
        # 1. 先尝试使用开发密钥解码
        try:
            dev_verified = jwt.decode(
                token,
                key=DEV_SECRET_KEY,
                algorithms=[JWT_ALGORITHM],
                options={
                    'verify_signature': True,
                    'verify_exp': True
                }
            )
            logger.warning(f"开发模式: 使用开发密钥验证令牌成功: {dev_verified.get('user_id')}")
            return dev_verified
        except (jwt.InvalidSignatureError, jwt.ExpiredSignatureError):
            # 如果使用开发密钥验证失败，继续尝试其他方法
            pass
        
        # 2. 尝试解码令牌但不验证签名（适用于任何令牌）
        unverified = jwt.decode(
            token,
            key=None,
            options={'verify_signature': False, 'verify_exp': False}
        )
        
        # 3. 在开发模式下，接受任何格式正确的令牌
        if "user_id" in unverified:
            logger.warning(f"开发模式: 接受未验证的令牌，用户ID: {unverified.get('user_id')}")
            return unverified
        else:
            logger.warning("开发模式: 令牌格式正确但缺少user_id字段，使用默认开发用户")
            return {
                "user_id": "dev_user",
                "username": "开发默认用户",
                "roles": ["user", "admin"],
                "iat": int(datetime.now().timestamp()),
                "exp": int((datetime.now().timestamp() + 3600))
            }
    except Exception as e:
        # 4. 如果连解码都失败，返回默认用户
        logger.warning(f"开发模式: 令牌解析错误，使用默认开发用户: {e}")
        return {
            "user_id": "dev_fallback",
            "username": "开发后备用户",
            "roles": ["user", "admin"],
            "iat": int(datetime.now().timestamp()),
            "exp": int((datetime.now().timestamp() + 3600))
        }

def handle_dev_auth(request: Request) -> Optional[Dict[str, Any]]:
    """处理开发模式下的认证逻辑
    
    Args:
        request: FastAPI请求对象
        
    Returns:
        如果是开发模式并且能认证，返回用户信息；否则返回None
    """
    if not is_dev_mode():
        return None
        
    # 处理Swagger UI请求
    referer = request.headers.get("referer", "")
    if "/docs" in referer or request.url.path.startswith("/docs") or request.url.path == "/openapi.json":
        logger.warning("开发模式: Swagger UI请求自动通过认证")
        return {
            "user_id": "swagger_dev_user", 
            "username": "Swagger测试用户",
            "roles": ["user", "admin"],
        }
    
    # 如果请求头中有X-Dev-Secret-Key头，则使用该值生成JWT令牌
    dev_key = request.headers.get("X-Dev-Secret-Key")
    if dev_key:
        dev_user = request.headers.get("X-Dev-User", "custom_dev_user")
        logger.warning(f"开发模式: 使用用户提供的密钥生成令牌: 用户={dev_user}")
        # 不实际生成令牌，而是直接返回用户信息
        return {
            "user_id": dev_user,
            "username": f"开发用户_{dev_user}",
            "roles": ["user", "admin"],
            "iat": int(datetime.now().timestamp()),
            "exp": int((datetime.now().timestamp() + 3600))
        }
        
    # 如果请求头中有X-Dev-User头，则使用该值作为用户ID
    dev_user = request.headers.get("X-Dev-User")
    if dev_user:
        logger.warning(f"开发模式: 使用X-Dev-User头: {dev_user}")
        return {
            "user_id": dev_user,
            "username": f"开发用户_{dev_user}",
            "roles": ["user", "admin"],
        }
        
    # 如果请求中有dev_token参数，则使用测试用户身份
    if request.query_params.get("dev_token") == "true":
        logger.warning("开发模式: 使用dev_token参数")
        return {
            "user_id": "dev_admin",
            "username": "开发管理员",
            "roles": ["user", "admin"],
        }

    # 在开发模式下，尝试从标准认证头获取令牌
    auth_header = request.headers.get("Authorization")
    if auth_header:
        try:
            parts = auth_header.split()
            if parts[0].lower() == "bearer" and len(parts) > 1:
                access_token = parts[1]
                # 尝试解码令牌但不验证签名
                try:
                    unverified = jwt.decode(
                        access_token,
                        key=None,
                        options={'verify_signature': False, 'verify_exp': False}
                    )
                    if "user_id" in unverified:
                        logger.warning(f"开发模式: 使用令牌中的用户ID（不验证签名）: {unverified['user_id']}")
                        return unverified
                except Exception as e:
                    logger.warning(f"开发模式: 令牌解析出错，但将使用默认开发用户: {e}")
        except Exception:
            pass
                    
    # 如果没有任何认证信息，或者认证失败，在开发模式下使用默认用户
    logger.warning("开发模式: 没有有效的认证信息，使用默认开发用户")
    return {
        "user_id": "dev_default",
        "username": "默认开发用户",
        "roles": ["user", "admin"],
        "iat": int(datetime.now().timestamp()),
        "exp": int((datetime.now().timestamp() + 3600))
    } 