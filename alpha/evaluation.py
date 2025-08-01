"""Alpha评估模块"""
import pandas as pd
import numpy as np
import logging
import re
from mcts.formula_generator import safe_divide

logger = logging.getLogger(__name__)


def sanitize_formula(formula):
    """
    清理和验证公式，防止不安全的操作
    
    Parameters:
    - formula: 要验证的公式字符串
    
    Returns:
    - sanitized_formula: 清理后的公式，如果不安全则返回None
    """
    # 检查是否包含不安全的操作
    unsafe_patterns = [
        r'__\w+__',  # 防止访问特殊方法
        r'import\s+',  # 防止导入模块
        r'exec\s*\(',  # 防止执行代码
        r'eval\s*\(',  # 防止嵌套eval
        r'open\s*\(',  # 防止文件操作
        r'file\s*\(',  # 防止文件操作
        r'input\s*\(',  # 防止输入操作
        r'raw_input\s*\(',  # 防止输入操作
    ]
    
    for pattern in unsafe_patterns:
        if re.search(pattern, formula, re.IGNORECASE):
            logger.warning(f"Unsafe formula detected: {formula}")
            return None
    
    # 验证公式只包含允许的字符
    allowed_chars = re.compile(r'^[a-zA-Z0-9_\s\+\-\*\/\(\)\.\,]+$')
    if not allowed_chars.match(formula):
        logger.warning(f"Formula contains invalid characters: {formula}")
        return None
    
    return formula


def evaluate_formula(formula, data):
    """
    评估公式在数据集上的结果

    Parameters:
    - formula: 要评估的公式字符串
    - data: 特征数据集

    Returns:
    - result: 评估后的特征
    """
    try:
        # 清理和验证公式
        sanitized_formula = sanitize_formula(formula)
        if sanitized_formula is None:
            logger.error(f"Formula failed security validation: '{formula}'")
            return pd.Series(np.nan, index=data.index)
        
        # 创建安全的评估环境
        safe_dict = data.copy()
        safe_dict['safe_divide'] = safe_divide
        
        # 限制可用的函数和变量
        allowed_functions = {
            'abs': abs,
            'max': max,
            'min': min,
            'sum': sum,
            'len': len,
            'safe_divide': safe_divide,
            'np': np,  # 允许numpy函数
        }
        
        # 验证公式中的所有变量都存在
        formula_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
        for var in formula_vars:
            if var not in safe_dict and var not in allowed_functions:
                logger.warning(f"Unknown variable '{var}' in formula: {formula}")
                return pd.Series(np.nan, index=data.index)
        
        # 安全评估
        safe_dict.update(allowed_functions)
        try:
            result = pd.eval(sanitized_formula, local_dict=safe_dict)
        except ValueError as e:
            if "If using all scalar values, you must pass an index" in str(e):
                # 直接计算标量操作，然后转换为Series
                result = eval(sanitized_formula, {"__builtins__": {}}, safe_dict)
                result = pd.Series([result], index=[data.index[0]] if len(data.index) == 1 else data.index[:1])
            else:
                raise e
        
        # 处理其他类型的结果
        if not isinstance(result, (pd.Series, pd.DataFrame)):
            result = pd.Series([result], index=[data.index[0]] if len(data.index) == 1 else data.index[:1])
        elif isinstance(result, pd.Series):
            # 替换无限值为NaN
            result = result.replace([np.inf, -np.inf], np.nan)
        
        return result
        
    except Exception as e:
        logger.error(f"Error evaluating formula '{formula}': {e}")
        return pd.Series(np.nan, index=data.index)