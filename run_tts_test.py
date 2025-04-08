#!/usr/bin/env python
"""
illufly-tts 测试运行脚本
使用方法: python run_tts_test.py

此脚本是转发至scripts/run_tts_test.py的简单包装
"""

import os
import sys

# 直接执行scripts目录下的测试脚本
script_path = os.path.join(os.path.dirname(__file__), "scripts", "run_tts_test.py")

if __name__ == "__main__":
    print("启动 illufly-tts 测试...")
    # 导入并执行脚本
    exec(open(script_path).read())
    print("测试完成") 