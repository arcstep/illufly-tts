# test_kokoro_direct.py
import os
import torch
import torchaudio
from kokoro import KPipeline

# 设置日志
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kokoro_test")

# 直接测试
text = "你好，这是一个测试"
pipeline = KPipeline(lang_code='z')

try:
    generator = pipeline(text, voice="zf_001")
    for i, (gs, ps, audio) in enumerate(generator):
        logger.info(f"生成成功: {type(audio)}, shape: {audio.shape}")
        
        # 保存音频
        torchaudio.save(f"test_direct_{i}.wav", 
                      torch.tensor(audio).unsqueeze(0), 
                      24000)
        logger.info(f"已保存音频: test_direct_{i}.wav")
except Exception as e:
    logger.error(f"直接调用失败: {e}")
    import traceback
    logger.error(traceback.format_exc())