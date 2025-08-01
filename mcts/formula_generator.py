"""公式生成模块"""
import numpy as np


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
    formula = f"{feature1} {operator} {feature2}"
    return formula