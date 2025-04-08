#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志配置模块 - 配置项目日志
"""

import os
import sys
import logging
from typing import Optional

def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    debug_modules: Optional[list] = None
):
    """配置日志系统
    
    Args:
        level: 基础日志级别
        log_file: 可选的日志文件路径
        log_to_console: 是否输出到控制台
        debug_modules: 需要debug级别日志的模块列表
    """
    # 创建根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的处理程序
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理程序
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
    
    # 文件处理程序
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # 为特定模块设置更详细的日志级别
    if debug_modules:
        for module_name in debug_modules:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.DEBUG)
    
    # 设置基础库的日志级别为WARNING，减少噪音
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # 设置项目基本模块的日志级别
    logging.getLogger('illufly_tts').setLevel(level)
    
    # 返回根日志器，以便进一步配置
    return root_logger 