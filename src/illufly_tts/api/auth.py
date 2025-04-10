import jwt
import logging
from typing import Dict, Any, List, Optional, Callable
from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel
import os
from datetime import datetime

# 从环境变量获取JWT配置
JWT_SECRET_KEY = os.environ.get("FASTAPI_SECRET_KEY", "MY-SECRET-KEY")
JWT_ALGORITHM = os.environ.get("FASTAPI_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
JWT_COOKIE_NAME = os.environ.get("JWT_COOKIE_NAME", "access_token")

# 延迟初始化，确保即使环境变量后加载也能正确获取
def get_jwt_secret_key():
    """延迟获取JWT密钥，确保环境变量已加载"""
    key = JWT_SECRET_KEY
    if not key:
        # 再次尝试从环境变量获取
        key = os.environ.get("FASTAPI_SECRET_KEY", "")
        # 检查并移除可能的引号
        if key and key.startswith('"') and key.endswith('"'):
            key = key.strip('"')
    return key

# 引入开发模式功能
from .dev_mode import is_dev_mode, verify_token_dev_mode, handle_dev_auth

logger = logging.getLogger(__name__)

class TokenVerifier:
    """JWT令牌验证器"""
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """验证JWT令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            解码后的令牌数据
            
        Raises:
            HTTPException: 如果令牌无效或已过期
        """
        # 使用延迟初始化获取JWT密钥
        jwt_secret_key = get_jwt_secret_key()
        
        # 调试信息 - 打印环境变量和密钥信息（注意不打印完整密钥内容）
        key_info = "空" if not jwt_secret_key else f"长度为{len(jwt_secret_key)}的字符串"
        logger.warning(f"[调试] JWT验证详情 - 算法: {JWT_ALGORITHM}, 密钥: {key_info}, Cookie名: {JWT_COOKIE_NAME}")
        
        # 检查是否处于开发模式
        if is_dev_mode():
            # 使用开发模式的令牌验证逻辑
            return verify_token_dev_mode(token)
        
        # 生产模式处理 - 正常的JWT验证
        try:
            # 先解码令牌不验证签名，获取基本信息
            unverified = jwt.decode(
                token,
                key=None,
                options={'verify_signature': False, 'verify_exp': False}
            )
            logger.debug(f"未验证的令牌: {unverified}")
            
            # 详细记录令牌结构
            logger.warning(f"[调试] 令牌结构: token_type={unverified.get('token_type')}, user_id={unverified.get('user_id')}, device_id={unverified.get('device_id')}")
            logger.warning(f"[调试] 令牌字段: {', '.join(unverified.keys())}")
            logger.warning(f"[调试] 令牌体完整内容: {unverified}")
            
            # 正常验证令牌
            try:
                # 详细记录验证过程
                logger.warning(f"[调试] 尝试使用密钥验证令牌签名...")
                valid_data = jwt.decode(
                    token,
                    key=jwt_secret_key,
                    algorithms=[JWT_ALGORITHM],
                    options={
                        'verify_signature': True,
                        'verify_exp': True,
                        'require': ['exp', 'iat'],
                    }
                )
                logger.warning(f"[调试] 令牌验证成功!")
                logger.info(f"令牌验证成功: {valid_data.get('username')}")
                return valid_data
                
            except jwt.ExpiredSignatureError:
                logger.warning(f"[调试] 令牌已过期: {unverified.get('username')}")
                logger.warning(f"令牌已过期: {unverified.get('username')}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="令牌已过期，请重新登录"
                )
                
            except jwt.InvalidSignatureError:
                logger.warning(f"[调试] 令牌签名无效: JWT_SECRET_KEY前几个字符: {jwt_secret_key[:5] if jwt_secret_key else '无'}")
                logger.error(f"令牌签名无效")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"令牌签名无效"
                )
                
            except Exception as e:
                logger.warning(f"[调试] 令牌验证错误: {str(e)}, 类型: {type(e).__name__}")
                logger.error(f"令牌验证错误: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"令牌验证错误: {str(e)}"
                )
                
        except Exception as e:
            logger.warning(f"[调试] 令牌解析错误: {str(e)}, 类型: {type(e).__name__}")
            logger.error(f"令牌解析错误: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"无效的令牌格式: {str(e)}"
            )

def require_user(require_roles: Optional[List[str]] = None) -> Callable:
    """验证用户权限的依赖函数"""
    def verify_auth(request: Request) -> Dict[str, Any]:
        # 先检查是否是开发模式，尝试使用开发模式的认证逻辑
        dev_user = handle_dev_auth(request)
        if dev_user is not None:
            # 检查角色权限
            if require_roles:
                user_roles = dev_user.get("roles", [])
                if not all(role in user_roles for role in require_roles):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="权限不足，需要特定角色"
                    )
            return dev_user
                
        # 不是开发模式或开发模式认证失败，使用标准认证流程
        # 从Cookie中获取令牌
        access_token = request.cookies.get(JWT_COOKIE_NAME)
        if access_token:
            logger.debug(f"从Cookie中获取到令牌，长度: {len(access_token)}")
        else:
            # 如果没有获取到有效令牌，则认证失败
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="未提供认证令牌"
            )
        
        # 验证令牌
        user_data = TokenVerifier.verify_token(access_token)
        
        # 检查角色权限
        if require_roles:
            user_roles = user_data.get("roles", [])
            if not all(role in user_roles for role in require_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="权限不足，需要特定角色"
                )
        
        return user_data
    
    return verify_auth 