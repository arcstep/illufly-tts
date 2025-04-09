# Illufly TTS - 模块化语音合成服务

高质量的模块化中文语音合成系统，支持多种部署方式。

## 特点

- **模块化设计**：可以按需安装客户端、服务端或完整包
- **低资源需求**：客户端几乎不需要额外依赖
- **易于集成**：可以直接复制API端点到其他应用
- **多种部署**：支持命令行、MCP协议和REST API接口

## 安装方式

### 1. 完整安装

包含所有组件，适合单机部署或完整服务：

```bash
pip install "illufly-tts[full]"
```

### 2. 仅客户端

轻量级安装，只包含客户端组件：

```bash
pip install "illufly-tts[client]"
```

### 3. 仅服务端

包含TTS核心服务：

```bash
pip install "illufly-tts[server]"
```

## 使用方式

### 完整服务（单机模式）

```bash
# 启动完整服务（包含API和TTS引擎）
python -m illufly_tts serve --voices-dir=./voices --device=cpu
```

### 拆分部署

```bash
# 在服务器上启动MCP服务
python -m illufly_tts server --voices-dir=./voices --device=cuda --transport=sse --port=31572

# 在客户端启动API服务
python -m illufly_tts api --server-host=tts-server-ip --server-port=31572
```

### 命令行客户端

```bash
# 通过命令行使用
python -m illufly_tts client speak --process-command="/usr/bin/python" --process-args="-m,illufly_tts,server" "你好世界"

# 获取可用语音
python -m illufly_tts client voices
```

### 集成到其他应用

```python
from fastapi import FastAPI
from illufly_tts.api.endpoints import mount_tts_service

app = FastAPI()

# 添加自定义认证逻辑
async def get_current_user():
    # 实现你的认证逻辑
    return {"user_id": "my_user"}

# 挂载TTS服务
mount_tts_service(
    app=app,
    require_user=get_current_user,
    host="tts-server-host",  # 或使用子进程方式
    port=31572,
    prefix="/api/tts"
)
```

## 直接复制集成

如果你希望完全避免依赖，可以直接复制以下关键文件：

1. `src/illufly_tts/client/mcp_client.py` - MCP客户端
2. `src/illufly_tts/api/endpoints.py` - FastAPI端点

然后按照上面的集成示例整合到你的应用中。

## 更多信息

详细文档请参阅项目Wiki或API文档。
