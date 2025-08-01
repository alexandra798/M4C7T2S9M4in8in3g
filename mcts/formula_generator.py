"""公式生成模块"""
import numpy as np


def safe_divide(x, y, default_value=0):
    """
    安全除法函数，避免除零错误
    
    Parameters:
    - x: 被除数
    - y: 除数
    - default_value: 除零时的默认值
    
    Returns:
    - 除法结果或默认值
    """
    return np.where(np.abs(y) < 1e-8, default_value, x / y)


def generate_formula(all_features):
    """
    生成随机公式

    Parameters:
    - all_features: 可用特征列表

    Returns:
    - formula: 生成的公式字符串
    """
    operators = ['+', '-', '*', '/']
    feature1 = np.random.choice(all_features)
    feature2 = np.random.choice(all_features)
    operator = np.random.choice(operators)
    
    # 对于除法操作，使用安全除法包装
    if operator == '/':
        formula = f"safe_divide({feature1}, {feature2})"
    else:
        formula = f"{feature1} {operator} {feature2}"
    
    return formula