#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文G2P - 将英文文本转换为音素序列
"""

import re
import os
import logging
import time
import traceback
from typing import List, Dict, Any, Optional, Set, Tuple
import importlib.util
import json
import pickle

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置NLTK数据路径环境变量
nltk_data_paths = [
    os.path.expanduser('~/.nltk_data'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources/nltk_data')
]
for path in nltk_data_paths:
    if os.path.exists(path):
        os.environ['NLTK_DATA'] = path
        logger.info(f"设置NLTK_DATA环境变量: {path}")
        break

# 检查nltk是否可用，并预先准备资源
try:
    import nltk
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    
    # 确保NLTK资源目录存在
    for path in nltk_data_paths:
        if os.path.exists(path):
            nltk.data.path.insert(0, path)
            logger.info(f"添加NLTK数据路径: {path}")
            
    # 尝试修复averaged_perceptron_tagger资源
    try:
        # 下载必要资源
        try:
            nltk.download('averaged_perceptron_tagger')
            logger.info("成功下载averaged_perceptron_tagger资源")
        except Exception as e:
            logger.warning(f"无法下载NLTK资源: {e}")
        
        # 检查tagger资源
        for data_path in nltk.data.path:
            tagger_dir = os.path.join(data_path, 'taggers')
            if os.path.exists(tagger_dir):
                orig_tagger = os.path.join(tagger_dir, 'averaged_perceptron_tagger')
                eng_tagger = os.path.join(tagger_dir, 'averaged_perceptron_tagger_eng')
                
                logger.info(f"检查tagger目录: {tagger_dir}")
                logger.info(f"原始tagger路径: {orig_tagger}")
                logger.info(f"g2p_en tagger路径: {eng_tagger}")
                
                # 确保目录存在
                if os.path.exists(orig_tagger) and not os.path.exists(eng_tagger):
                    os.makedirs(eng_tagger, exist_ok=True)
                    logger.info(f"创建g2p_en tagger目录: {eng_tagger}")
                    
                    # 尝试创建基本JSON文件以便g2p_en能够工作
                    weights_json = os.path.join(eng_tagger, 'averaged_perceptron_tagger_eng.weights.json')
                    classes_json = os.path.join(eng_tagger, 'averaged_perceptron_tagger_eng.classes.json')
                    tagdict_json = os.path.join(eng_tagger, 'averaged_perceptron_tagger_eng.tagdict.json')
                    
                    # 创建基本的空JSON文件
                    if not os.path.exists(weights_json):
                        with open(weights_json, 'w') as f:
                            json.dump({}, f)
                        logger.info(f"创建weights.json文件: {weights_json}")
                    if not os.path.exists(classes_json):
                        with open(classes_json, 'w') as f:
                            json.dump(["NN", "NNP", "JJ", "VB"], f)
                        logger.info(f"创建classes.json文件: {classes_json}")
                    if not os.path.exists(tagdict_json):
                        with open(tagdict_json, 'w') as f:
                            json.dump({}, f)
                        logger.info(f"创建tagdict.json文件: {tagdict_json}")
                elif os.path.exists(eng_tagger):
                    logger.info(f"g2p_en tagger目录已存在: {eng_tagger}")
                elif not os.path.exists(orig_tagger):
                    logger.warning(f"找不到原始tagger目录: {orig_tagger}")
    except Exception as e:
        logger.warning(f"在模块加载时修复NLTK资源失败: {e}")
        traceback.print_exc()
            
except ImportError:
    nltk = None
    word_tokenize = None
    NLTK_AVAILABLE = False
    logger.warning("nltk模块不可用，将使用简化英文G2P")

# 延迟导入g2p_en，直到资源准备就绪
G2P_EN_AVAILABLE = False
g2p_en = None

def load_g2p_en():
    """延迟加载g2p_en模块"""
    global g2p_en, G2P_EN_AVAILABLE
    try:
        if g2p_en is None:
            import g2p_en as g2p_en_module
            g2p_en = g2p_en_module
            G2P_EN_AVAILABLE = True
            logger.info("成功导入g2p_en模块")
        return True
    except ImportError as e:
        logger.warning(f"g2p_en模块不可用: {e}")
        G2P_EN_AVAILABLE = False
        return False
    except Exception as e:
        logger.error(f"导入g2p_en时发生错误: {e}")
        traceback.print_exc()
        G2P_EN_AVAILABLE = False
        return False

from .base_g2p import BaseG2P

class EnglishG2P(BaseG2P):
    """英文G2P转换器"""
    
    def __init__(self, dict_path: Optional[str] = None, nltk_data_path: Optional[str] = None):
        """初始化英文G2P
        
        Args:
            dict_path: 自定义词典路径
            nltk_data_path: NLTK数据目录路径
        """
        super().__init__()
        
        # 初始化音素字典
        self.phoneme_dict = {}
        
        # 初始化音素映射
        self.phoneme_mapping = {
            # 基本元音
            'AA': 'aa', 'AE': 'ae', 'AH': 'ah', 'AO': 'ao', 'AW': 'aw',
            'AY': 'ay', 'EH': 'eh', 'ER': 'er', 'EY': 'ey', 'IH': 'ih',
            'IY': 'iy', 'OW': 'ow', 'OY': 'oy', 'UH': 'uh', 'UW': 'uw',
            # 辅音
            'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'dh', 'F': 'f',
            'G': 'g', 'HH': 'hh', 'JH': 'jh', 'K': 'k', 'L': 'l',
            'M': 'm', 'N': 'n', 'NG': 'ng', 'P': 'p', 'R': 'r',
            'S': 's', 'SH': 'sh', 'T': 't', 'TH': 'th', 'V': 'v',
            'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'zh'
        }
        
        # 初始化G2P转换器
        self.g2p = None
        
        # 设置NLTK数据路径
        self._setup_nltk_data(nltk_data_path)
        
        # 尝试解决g2p_en使用的pickle文件格式问题
        self._fix_g2p_en_resources()
        
        # 尝试初始化G2P转换器
        self._init_g2p()
        
        # 加载自定义词典（如果有）
        if dict_path:
            self.load_dict(dict_path)
        else:
            # 尝试加载默认词典
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_dict_path = os.path.join(current_dir, '../resources/dictionaries/english_dict.txt')
            if os.path.exists(default_dict_path):
                logger.info(f"加载默认英文词典: {default_dict_path}")
                self.load_dict(default_dict_path)
    
    def _setup_nltk_data(self, nltk_data_path: Optional[str] = None):
        """设置NLTK数据路径
        
        Args:
            nltk_data_path: 指定的NLTK数据路径
        """
        if not NLTK_AVAILABLE:
            return
            
        try:
            # 如果指定了数据路径，则使用
            if nltk_data_path and os.path.exists(nltk_data_path):
                os.environ['NLTK_DATA'] = nltk_data_path
                if nltk:
                    nltk.data.path.insert(0, nltk_data_path)
                logger.info(f"使用指定的NLTK数据目录: {nltk_data_path}")
            else:
                # 项目内置资源目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_nltk_data = os.path.abspath(os.path.join(current_dir, '../resources/nltk_data'))
                
                # 用户主目录
                user_nltk_data = os.path.join(os.path.expanduser('~'), 'nltk_data')
                
                # 添加多个可能的路径
                if os.path.exists(project_nltk_data):
                    os.environ['NLTK_DATA'] = project_nltk_data
                    if nltk:
                        nltk.data.path.insert(0, project_nltk_data)
                    logger.info(f"使用项目NLTK数据目录: {project_nltk_data}")
                    
                if os.path.exists(user_nltk_data):
                    os.environ['NLTK_DATA'] = user_nltk_data
                    if nltk:
                        nltk.data.path.append(user_nltk_data)
                
            # 确保必要的NLTK资源可用
            if nltk:
                try:
                    # 尝试找到并可能下载必要的资源
                    try:
                        nltk.data.find('taggers/averaged_perceptron_tagger.zip')
                    except LookupError:
                        logger.info("下载 averaged_perceptron_tagger 资源")
                        nltk.download('averaged_perceptron_tagger')
                    
                    try:
                        nltk.data.find('corpora/cmudict.zip')
                    except LookupError:
                        logger.info("下载 cmudict 资源")
                        nltk.download('cmudict')
                    
                except Exception as e:
                    logger.error(f"NLTK资源准备失败: {str(e)}")
            
            # 打印所有NLTK搜索路径
            logger.info(f"NLTK数据搜索路径: {nltk.data.path}")
            
        except Exception as e:
            logger.error(f"设置NLTK数据路径失败: {e}")
    
    def _fix_g2p_en_resources(self):
        """解决g2p_en使用的pickle文件格式问题
        
        g2p_en需要访问一些特定格式的资源文件，这个函数尝试解决这些问题
        """
        if not (NLTK_AVAILABLE and G2P_EN_AVAILABLE):
            return
        
        try:
            logger.info("开始检查和修复g2p_en资源文件")
            
            # 检查g2p_en模块的位置
            g2p_module_path = os.path.dirname(g2p_en.__file__)
            logger.info(f"g2p_en模块路径: {g2p_module_path}")
            
            # g2p_en可能查找的目录
            nltk_data_dir = os.environ.get('NLTK_DATA', os.path.expanduser('~/.nltk_data'))
            logger.info(f"NLTK数据主目录: {nltk_data_dir}")
            
            # 查找可能的tagger资源目录
            tagger_dirs = []
            for data_path in nltk.data.path:
                tagger_dir = os.path.join(data_path, 'taggers')
                if os.path.exists(tagger_dir):
                    tagger_dirs.append(tagger_dir)
                    logger.info(f"发现tagger目录: {tagger_dir}")
            
            # 检查特定的tagger资源
            original_tagger_dirs = []
            for tagger_dir in tagger_dirs:
                original_tagger = os.path.join(tagger_dir, 'averaged_perceptron_tagger')
                if os.path.exists(original_tagger):
                    original_tagger_dirs.append(original_tagger)
                    logger.info(f"发现tagger资源: {original_tagger}")
            
            if not original_tagger_dirs:
                logger.warning("未找到任何averaged_perceptron_tagger资源")
                # 尝试重新下载
                nltk.download('averaged_perceptron_tagger')
                return
            
            # 检查g2p_en需要的资源目录
            for tagger_dir in tagger_dirs:
                eng_tagger = os.path.join(tagger_dir, 'averaged_perceptron_tagger_eng')
                if not os.path.exists(eng_tagger):
                    logger.info(f"创建g2p_en需要的资源目录: {eng_tagger}")
                    os.makedirs(eng_tagger, exist_ok=True)
                else:
                    logger.info(f"g2p_en资源目录已存在: {eng_tagger}")
            
            # 重点：创建g2p_en需要的特定名称文件
            for original_tagger in original_tagger_dirs:
                # 获取原始的pickle文件
                pickle_file = os.path.join(original_tagger, 'averaged_perceptron_tagger.pickle')
                
                if os.path.exists(pickle_file):
                    logger.info(f"找到tagger pickle文件: {pickle_file}")
                    
                    try:
                        # 读取pickle文件
                        with open(pickle_file, 'rb') as f:
                            tagger_data = pickle.load(f)
                        
                        # 检查数据结构 - NLTK的perceptron tagger通常是一个元组 (weights, tagdict, classes)
                        # 输出调试信息
                        logger.info(f"Tagger数据类型: {type(tagger_data)}")
                        
                        tagger_weights = {}
                        tagger_classes = []
                        tagger_tagdict = {}
                        
                        # 处理tuple类型的数据
                        if isinstance(tagger_data, tuple) and len(tagger_data) == 3:
                            logger.info("检测到tuple格式的tagger数据")
                            tagger_weights = tagger_data[0]  # 第一个元素是weights
                            tagger_tagdict = tagger_data[1]  # 第二个元素是tagdict
                            tagger_classes = tagger_data[2]  # 第三个元素是classes
                            
                            logger.info(f"从tuple中提取数据: weights类型={type(tagger_weights)}, tagdict类型={type(tagger_tagdict)}, classes类型={type(tagger_classes)}")
                        # 处理对象类型的数据
                        elif hasattr(tagger_data, 'weights') and hasattr(tagger_data, 'classes') and hasattr(tagger_data, 'tagdict'):
                            logger.info("检测到对象格式的tagger数据")
                            tagger_weights = tagger_data.weights
                            tagger_tagdict = tagger_data.tagdict
                            tagger_classes = tagger_data.classes
                        # 处理字典类型的数据
                        elif isinstance(tagger_data, dict) and 'weights' in tagger_data and 'classes' in tagger_data and 'tagdict' in tagger_data:
                            logger.info("检测到字典格式的tagger数据")
                            tagger_weights = tagger_data['weights']
                            tagger_tagdict = tagger_data['tagdict']
                            tagger_classes = tagger_data['classes']
                        else:
                            logger.error(f"无法识别的tagger数据格式: {type(tagger_data)}")
                            # 尝试创建一个基本的数据结构
                            logger.info("使用基本tagger数据结构")
                            default_classes = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
                            tagger_classes = set(default_classes)
                        
                        # 处理集合类型，转换为JSON可序列化的类型
                        def convert_for_json(obj):
                            if isinstance(obj, set):
                                return list(obj)
                            elif isinstance(obj, dict):
                                return {k: convert_for_json(v) for k, v in obj.items()}
                            elif isinstance(obj, list) or isinstance(obj, tuple):
                                return [convert_for_json(x) for x in obj]
                            else:
                                return obj
                        
                        # 转换数据
                        temp_weights = convert_for_json(tagger_weights)
                        temp_tagdict = convert_for_json(tagger_tagdict)
                        temp_classes = convert_for_json(tagger_classes)
                        
                        # 确保classes不为空
                        if not temp_classes:
                            logger.warning("Classes为空，使用默认值")
                            default_classes = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
                            temp_classes = default_classes
                        
                        # 为每个tagger目录创建所需的json文件
                        for tagger_dir in tagger_dirs:
                            eng_tagger = os.path.join(tagger_dir, 'averaged_perceptron_tagger_eng')
                            
                            # 创建g2p_en需要的文件名格式
                            weights_json = os.path.join(eng_tagger, 'averaged_perceptron_tagger_eng.weights.json')
                            classes_json = os.path.join(eng_tagger, 'averaged_perceptron_tagger_eng.classes.json')
                            tagdict_json = os.path.join(eng_tagger, 'averaged_perceptron_tagger_eng.tagdict.json')
                            
                            # 保存weights.json
                            if not os.path.exists(weights_json) or os.path.getsize(weights_json) < 10:
                                logger.info(f"创建weights.json文件: {weights_json}")
                                try:
                                    with open(weights_json, 'w', encoding='utf-8') as f:
                                        json.dump(temp_weights, f, ensure_ascii=False)
                                except Exception as e:
                                    logger.error(f"创建weights.json失败: {e}")
                                    traceback.print_exc()
                            else:
                                logger.info(f"weights.json文件已存在: {weights_json}")
                            
                            # 保存classes.json
                            if not os.path.exists(classes_json) or os.path.getsize(classes_json) < 10:
                                logger.info(f"创建classes.json文件: {classes_json}")
                                try:
                                    with open(classes_json, 'w', encoding='utf-8') as f:
                                        json.dump(temp_classes, f, ensure_ascii=False)
                                except Exception as e:
                                    logger.error(f"创建classes.json失败: {e}")
                                    traceback.print_exc()
                            else:
                                logger.info(f"classes.json文件已存在: {classes_json}")
                            
                            # 保存tagdict.json
                            if not os.path.exists(tagdict_json) or os.path.getsize(tagdict_json) < 10:
                                logger.info(f"创建tagdict.json文件: {tagdict_json}")
                                try:
                                    with open(tagdict_json, 'w', encoding='utf-8') as f:
                                        json.dump(temp_tagdict, f, ensure_ascii=False)
                                except Exception as e:
                                    logger.error(f"创建tagdict.json失败: {e}")
                                    traceback.print_exc()
                            else:
                                logger.info(f"tagdict.json文件已存在: {tagdict_json}")
                            
                            # 同时也创建原始文件名格式的json，以便兼容
                            orig_json = os.path.join(eng_tagger, 'averaged_perceptron_tagger.json')
                            if not os.path.exists(orig_json) or os.path.getsize(orig_json) < 10:
                                logger.info(f"创建兼容格式json文件: {orig_json}")
                                try:
                                    # 合并所有数据
                                    combined_data = {
                                        'weights': temp_weights,
                                        'classes': temp_classes,
                                        'tagdict': temp_tagdict
                                    }
                                    with open(orig_json, 'w', encoding='utf-8') as f:
                                        json.dump(combined_data, f, ensure_ascii=False)
                                except Exception as e:
                                    logger.error(f"创建兼容格式json失败: {e}")
                                    traceback.print_exc()
                    except Exception as e:
                        logger.error(f"处理pickle文件失败: {e}")
                        traceback.print_exc()
                else:
                    logger.warning(f"未找到tagger pickle文件: {pickle_file}")
            
            logger.info("g2p_en资源文件检查和修复完成")
            
        except Exception as e:
            logger.error(f"修复g2p_en资源失败: {e}")
            traceback.print_exc()
    
    def _init_g2p(self):
        """初始化G2P转换器"""
        if not load_g2p_en():
            logger.warning("g2p_en不可用，将使用简化G2P")
            return
            
        try:
            # 修复g2p_en对NLTK资源的查找问题
            # 检查并创建必要的符号链接，让g2p_en能找到需要的资源
            if NLTK_AVAILABLE and nltk:
                nltk_data_dir = os.environ.get('NLTK_DATA', os.path.expanduser('~/.nltk_data'))
                tagger_dir = os.path.join(nltk_data_dir, 'taggers')
                
                # 检查原始tagger资源存在
                original_tagger = os.path.join(tagger_dir, 'averaged_perceptron_tagger')
                if os.path.exists(original_tagger):
                    logger.info(f"找到NLTK tagger资源: {original_tagger}")
                    
                    # 列出原始目录内容
                    files = os.listdir(original_tagger)
                    logger.info(f"资源目录内容: {files}")
                    
                    # 检查g2p_en可能需要的资源名称
                    eng_tagger = os.path.join(tagger_dir, 'averaged_perceptron_tagger_eng')
                    if not os.path.exists(eng_tagger):
                        # 如果不存在，创建目录和文件
                        logger.info(f"创建g2p_en需要的资源目录: {eng_tagger}")
                        try:
                            os.makedirs(eng_tagger, exist_ok=True)
                            
                            # 复制或链接文件
                            for filename in os.listdir(original_tagger):
                                src = os.path.join(original_tagger, filename)
                                dst = os.path.join(eng_tagger, filename)
                                if not os.path.exists(dst):
                                    if hasattr(os, 'symlink'):
                                        logger.info(f"创建符号链接: {src} -> {dst}")
                                        os.symlink(src, dst)
                                    else:
                                        # Windows或不支持符号链接的系统
                                        import shutil
                                        logger.info(f"复制文件: {src} -> {dst}")
                                        shutil.copy2(src, dst)
                                        
                                    # 特殊处理pickle文件，转换为json
                                    if filename.endswith('.pickle'):
                                        json_file = dst.replace('.pickle', '.json')
                                        logger.info(f"转换pickle到json: {dst} -> {json_file}")
                                        try:
                                            with open(src, 'rb') as f:
                                                data = pickle.load(f)
                                            with open(json_file, 'w', encoding='utf-8') as f:
                                                json.dump(data, f, ensure_ascii=False, indent=2)
                                            logger.info(f"成功创建json文件: {json_file}")
                                        except Exception as e:
                                            logger.error(f"转换pickle到json失败: {e}")
                        except Exception as e:
                            logger.error(f"创建资源目录或文件失败: {e}")
                    else:
                        logger.info(f"g2p_en资源目录已存在: {eng_tagger}")
                        
                        # 列出eng_tagger目录内容
                        files = os.listdir(eng_tagger)
                        logger.info(f"g2p_en资源目录内容: {files}")
                else:
                    logger.error(f"未找到NLTK tagger资源: {original_tagger}")
            
            # 尝试初始化G2P转换器
            logger.info("尝试初始化G2P转换器...")
            self.g2p = g2p_en.G2p()
            logger.info("G2P转换器初始化成功")
        except Exception as e:
            logger.error(f"G2P转换器初始化失败: {e}")
            traceback.print_exc()
            self.g2p = None
    
    def load_dict(self, dict_path: str):
        """加载音素词典
        
        Args:
            dict_path: 词典文件路径
        """
        try:
            if os.path.exists(dict_path):
                count = 0
                with open(dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().split()
                            if len(parts) > 1:
                                word = parts[0].lower()
                                phonemes = ' '.join(parts[1:])
                                self.phoneme_dict[word] = phonemes
                                count += 1
                logger.info(f"已加载英文音素词典，包含 {count} 个条目")
            else:
                logger.warning(f"词典文件不存在: {dict_path}")
        except Exception as e:
            logger.error(f"加载音素词典失败: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """
        英文文本预处理
        
        Args:
            text: 输入英文文本
            
        Returns:
            预处理后的文本
        """
        # 基础清理
        text = self.sanitize_text(text)
        
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        
        # 处理缩写
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'d", " would", text)
        
        # 合并多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_phonemes(self, raw_phonemes: List[str]) -> List[str]:
        """
        标准化音素序列
        
        Args:
            raw_phonemes: 原始音素列表
            
        Returns:
            标准化后的音素列表
        """
        result = []
        for ph in raw_phonemes:
            # 去掉音素中的数字(重音标记)
            ph_clean = re.sub(r'\d+', '', ph.upper())
            
            # 映射到标准音素
            if ph_clean in self.phoneme_mapping:
                result.append(self.phoneme_mapping[ph_clean])
            else:
                # 未知音素保持原样
                result.append(ph_clean.lower())
                
        return result
    
    def text_to_phonemes(self, text: str) -> str:
        """
        将英文文本转换为音素序列
        
        Args:
            text: 输入英文文本
            
        Returns:
            音素序列字符串，使用空格分隔
        """
        if not text:
            return ""
            
        # 检查词典
        if text.lower() in self.phoneme_dict:
            return self.phoneme_dict[text.lower()]
            
        # 使用g2p_en转换
        if self.g2p:
            try:
                # g2p_en处理
                logger.info(f"使用g2p_en转换文本: {text}")
                phonemes = self.g2p(text)
                logger.info(f"g2p_en转换结果: {phonemes}")
                
                # 移除可能的单词边界标记和非音素标记
                phonemes = [p for p in phonemes if not p in [' ', '', '{', '}', '.', ',', ':', '(', ')']]
                # 标准化
                normalized = self.normalize_phonemes(phonemes)
                return ' '.join(normalized)
            except Exception as e:
                logger.error(f"g2p_en转换失败: {e}")
                traceback.print_exc()
                # 出错时降级到简单转换，不直接返回
        
        # 备用简单转换
        return self._simple_g2p(text)
    
    def _simple_g2p(self, text: str) -> str:
        """
        简单的备用G2P，在g2p_en不可用时使用
        
        Args:
            text: 输入英文文本
            
        Returns:
            简单转换的音素序列
        """
        # 这里只是一个非常简化的映射，实际应用需要更复杂的规则
        simple_map = {
            'a': 'ae', 'e': 'eh', 'i': 'ih', 'o': 'aa', 'u': 'ah',
            'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g',
            'h': 'hh', 'j': 'jh', 'k': 'k', 'l': 'l', 'm': 'm',
            'n': 'n', 'p': 'p', 'q': 'k w', 'r': 'r', 's': 's',
            't': 't', 'v': 'v', 'w': 'w', 'x': 'k s', 'y': 'y', 'z': 'z'
        }
        
        # 先进行预处理
        text = self.preprocess_text(text)
        
        result = []
        for char in text.lower():
            if char in simple_map:
                result.append(simple_map[char])
            elif char == ' ':
                pass  # 忽略空格
            else:
                result.append(char)  # 保留未知字符
                
        return ' '.join(result)
    
    def get_phoneme_set(self) -> List[str]:
        """
        获取英文音素集
        
        Returns:
            英文音素列表
        """
        return list(set(self.phoneme_mapping.values()))
    
    def get_language(self) -> str:
        """
        获取语言代码
        
        Returns:
            语言代码
        """
        return "en" 