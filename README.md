# Illufly TTS

高质量多语言文本转语音(TTS)系统

## 特性

- 支持中文、英文和中英混合文本
- 基于Kokoro模型的高质量语音合成
- 模块化设计，易于扩展
- 支持多种语音和音色
- 提供完整的命令行工具

## 架构

系统由以下模块组成:

1. **文本预处理模块**
   - 语言分段器: 检测和分离混合文本中的不同语言
   - 文本规范化器: 处理数字、标点符号等

2. **G2P模块**
   - 英文G2P: 将英文文本转换为音素
   - 中文G2P: 将中文文本转换为拼音和音素
   - 混合G2P: 处理混合语言文本

3. **音频生成模块**
   - Kokoro适配器: 连接自定义处理与Kokoro模型
   - 语音资源管理: 管理多种语音和音色

4. **集成流水线**
   - 标准流水线: 基本的TTS流程
   - 混合语言流水线: 专为混合语言文本优化

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA (推荐，但非必须)

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/illufly/illufly-tts.git
cd illufly-tts

# 安装依赖
pip install -e .

# 开发环境安装
pip install -e ".[dev]"
```

## 使用方法

### 命令行工具

```bash
# 单句合成
illufly-tts --text "今天天气真好！" --model-path /path/to/model --voices-dir /path/to/voices --voice-id z001 --output output.wav

# 从文件读取
illufly-tts --file input.txt --model-path /path/to/model --voices-dir /path/to/voices --output output.wav

# 列出可用语音
illufly-tts --model-path /path/to/model --voices-dir /path/to/voices --list-voices

# 使用混合语言模式
illufly-tts --text "你好，Hello World！" --model-path /path/to/model --voices-dir /path/to/voices --mixed-language
```

### 编程接口

```python
from illufly_tts import MixedLanguagePipeline

# 创建流水线
tts = MixedLanguagePipeline(
    model_path="/path/to/model",
    voices_dir="/path/to/voices"
)

# 生成语音
audio = tts.text_to_speech(
    text="今天天气真好！Hello World!",
    voice_id="z001",
    output_path="output.wav"
)
```

## 测试

### 单元测试

```bash
# 运行单元测试
pytest

# 运行覆盖率测试
pytest --cov=illufly_tts
```

### 集成测试

我们提供了一系列集成测试脚本，用于分步验证TTS系统的各个组件功能：

1. **分步测试** - 逐步验证各模块效果:

```bash
# 运行所有测试
python tests/integration/test_processing_steps.py -m /path/to/model -v /path/to/voices

# 测试特定阶段
python tests/integration/test_processing_steps.py --stage segmenter -t "你好，测试文本。Hello, test text."
python tests/integration/test_processing_steps.py --stage normalizer -t "数字123转换测试"
python tests/integration/test_processing_steps.py --stage g2p -t "G2P测试文本"
python tests/integration/test_processing_steps.py --stage vocoder -m /path/to/model -v /path/to/voices
python tests/integration/test_processing_steps.py --stage pipeline -m /path/to/model -v /path/to/voices

# 详细日志
python tests/integration/test_processing_steps.py -m /path/to/model -v /path/to/voices --verbose
```

2. **对比测试** - 比较官方与自定义实现:

```bash
# 运行所有对比测试
python tests/integration/test_comparison.py -m /path/to/model -v /path/to/voices

# 只比较文本规范化
python tests/integration/test_comparison.py --mode normalization -t "规范化对比测试123"

# 只比较流水线
python tests/integration/test_comparison.py --mode pipeline -m /path/to/model -v /path/to/voices
```

3. **独立实现测试** - 测试完全独立的实现:

```bash
# 基本测试
python tests/integration/test_independent_pipeline.py -m /path/to/model -v /path/to/voices

# 分段处理
python tests/integration/test_independent_pipeline.py -m /path/to/model -v /path/to/voices --segmented

# 生成后播放
python tests/integration/test_independent_pipeline.py -m /path/to/model -v /path/to/voices --play

# 指定语音ID和语速
python tests/integration/test_independent_pipeline.py -m /path/to/model -v /path/to/voices --voice-id z001 --speed 1.2
```

所有测试脚本都会生成详细的日志和结果文件，方便排查问题和优化实现。

## 许可证

MIT

## 致谢

- [Kokoro TTS](https://github.com/hexgrad/kokoro) - 基础语音合成模型
- [g2p_en](https://github.com/Kyubyong/g2p) - 英文G2P转换
- [pypinyin](https://github.com/mozillazg/python-pinyin) - 中文拼音转换
