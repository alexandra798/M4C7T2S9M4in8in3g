"""Alpha评估模块"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
        return pd.eval(formula, local_dict=data)
    except Exception as e:
        logger.error(f"Error evaluating formula '{formula}': {e}")
        return pd.Series(np.nan, index=data.index)