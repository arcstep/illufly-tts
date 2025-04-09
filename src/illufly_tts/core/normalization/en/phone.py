import re

# 电话号码正则表达式
RE_PHONE = re.compile(r'(\+?(?:86|1)[-\s]?\d{2,3}[-\s]?\d{3,4}[-\s]?\d{4})')
RE_MOBILE = re.compile(r'(1\d{2}[-\s]?\d{4}[-\s]?\d{4})')

def replace_phone(match) -> str:
    """处理电话号码
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    number = match.group(1)
    
    # 判断是否为中文环境
    is_chinese = any(c in number for c in ['-', '—', '–']) or number.startswith('+86') or number.startswith('86')
    
    # 移除所有分隔符
    clean_number = re.sub(r'[-—–\s+]', '', number)
    
    if is_chinese:
        # 中文数字处理
        from ..zh.num import num2str
        # 400电话特殊处理
        if clean_number.startswith('400'):
            parts = [clean_number[:3], clean_number[3:6], clean_number[6:]]
            return '，'.join(num2str(part) for part in parts)
        # 手机号特殊处理
        elif clean_number.startswith('1') and len(clean_number) >= 11:
            parts = [clean_number[:3], clean_number[3:7], clean_number[7:11]]
            return '，'.join(num2str(part) for part in parts)
        else:
            # 分组处理
            result = []
            for i in range(0, len(clean_number), 3):
                part = clean_number[i:i+3]
                if part:
                    result.append(num2str(part))
            return '，'.join(result)
    else:
        # 英文数字处理
        from .num import verbalize_number
        # 分组处理
        result = []
        for i in range(0, len(clean_number), 3):
            part = clean_number[i:i+3]
            if part:
                result.append(verbalize_number(part))
        return ', '.join(result)

def replace_mobile(match) -> str:
    """处理手机号
    
    Args:
        match: 正则匹配对象
        
    Returns:
        处理后的文本
    """
    # 复用电话号码处理逻辑
    return replace_phone(match) 