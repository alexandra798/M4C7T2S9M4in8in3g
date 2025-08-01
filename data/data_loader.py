"""数据加载和预处理模块"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_user_dataset(file_path, target_column):
    """
    加载用户数据集，设置目标列，并准备特征。

    Parameters:
    - file_path: 数据集文件路径 (CSV)
    - target_column: 数据集中的目标列名称

    Returns:
    - X (特征), y (目标), all_features (特征名称列表)
    """
    logger.info(f"Loading dataset from {file_path}")
    user_dataset = pd.read_csv(file_path)

    # 转换日期列为datetime格式
    if 'date' in user_dataset.columns:
        user_dataset['date'] = pd.to_datetime(user_dataset['date'], errors='coerce')
        user_dataset.dropna(subset=['date'], inplace=True)
        # 如果存在ticker和date，设置多级索引
        if 'ticker' in user_dataset.columns:
            user_dataset.set_index(['ticker', 'date'], inplace=True)

    # 确保目标列存在
    if target_column not in user_dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # 分离特征和目标
    X = user_dataset.drop(columns=[target_column])
    y = user_dataset[target_column]

    # 获取特征名称列表
    all_features = X.columns.tolist()

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y, all_features


def check_missing_values(dataset, dataset_name):
    """
    检查数据集中的缺失值

    Parameters:
    - dataset: 要检查缺失值的DataFrame
    - dataset_name: 数据集名称（用于打印）
    """
    missing_values = dataset.isnull().sum()
    missing_columns = missing_values[missing_values > 0]

    if not missing_columns.empty:
        logger.warning(f'Missing values in {dataset_name} dataset:')
        logger.warning(missing_columns)
    else:
        logger.info(f'No missing values in {dataset_name} dataset.')


def apply_alphas_and_return_transformed(X, alpha_formulas, evaluate_formula_func):
    """
    应用顶级alpha公式到数据集，返回包含原始特征和新alpha特征的转换数据集

    Parameters:
    - X: 原始特征数据集
    - alpha_formulas: 要应用的alpha公式列表
    - evaluate_formula_func: 评估公式的函数

    Returns:
    - transformed_X: 包含原始特征和新alpha特征的数据集
    """
    transformed_X = X.copy()

    for formula in alpha_formulas:
        transformed_X[formula] = evaluate_formula_func(formula, X)

    return transformed_X