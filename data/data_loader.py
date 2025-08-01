"""数据加载和预处理模块 - 支持批量CSV文件处理"""
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csi300_tickers(csi300_file_path: str) -> List[str]:
    """
    从csi300.txt文件中读取股票代码列表

    Parameters:
    - csi300_file_path: csi300.txt文件路径

    Returns:
    - tickers: 股票代码列表
    """
    logger.info(f"Loading CSI300 tickers from {csi300_file_path}")

    with open(csi300_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tickers = []
    for line in lines:
        line = line.strip()
        if line:  # 跳过空行
            # 提取股票代码（第一列）
            ticker = line.split('\t')[0]
            tickers.append(ticker)

    logger.info(f"Loaded {len(tickers)} tickers from CSI300 list")
    return tickers


def load_single_ticker_data(file_path: str, ticker: str, target_column: str, 
                           future_days: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
    """
    加载单个股票的CSV数据

    Parameters:
    - file_path: CSV文件路径
    - ticker: 股票代码
    - target_column: 目标列名称
    - future_days: 计算未来收益的天数

    Returns:
    - X: 特征数据
    - y: 目标数据
    """
    try:
        df = pd.read_csv(file_path)

        # 如果CSV中已有code列，使用它；否则添加ticker列
        if 'code' in df.columns:
            df['ticker'] = df['code']
        else:
            df['ticker'] = ticker

        # 处理日期列
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            # 按日期排序
            df = df.sort_values('date')

        # 如果目标列是 label_shifted，自动生成
        if target_column == 'label_shifted' and target_column not in df.columns:
            # 计算未来N天的收益率
            df[target_column] = df['close'].pct_change(future_days).shift(-future_days)
            logger.debug(f"Generated {target_column} for {ticker} with {future_days} days forward return")

        # 确保目标列存在
        if target_column not in df.columns:
            error_msg = f"Target column '{target_column}' not found in {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 移除最后future_days行（因为没有未来数据）
        if target_column == 'label_shifted':
            df = df[:-future_days]

        # 分离特征和目标
        # 删除不需要的列
        columns_to_drop = [target_column, 'ts_code_x', 'ts_code_y', 'preclose', 'code']
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        X = df.drop(columns=columns_to_drop)
        y = df[target_column]

        return X, y

    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise RuntimeError(f"Failed to load data from {file_path}: {e}") from e


def load_batch_datasets(data_directory: str, csi300_file_path: str, target_column: str,
                        file_pattern: str = "{ticker}.csv", future_days: int = 20) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    批量加载CSI300股票数据

    Parameters:
    - data_directory: CSV文件所在目录
    - csi300_file_path: csi300.txt文件路径
    - target_column: 目标列名称
    - file_pattern: 文件名模式，{ticker}会被替换为实际股票代码
    - future_days: 计算未来收益的天数

    Returns:
    - X: 合并后的特征数据（包含ticker和date的多级索引）
    - y: 合并后的目标数据
    - all_features: 特征名称列表
    """
    logger.info("Starting batch dataset loading...")

    # 加载CSI300股票代码列表
    tickers = load_csi300_tickers(csi300_file_path)

    # 存储所有数据
    all_X_data = []
    all_y_data = []
    successful_loads = 0
    failed_loads = 0

    for ticker in tickers:
        # 构建文件路径
        file_name = file_pattern.format(ticker=ticker)
        file_path = os.path.join(data_directory, file_name)

        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            failed_loads += 1
            continue

        # 加载数据
        try:
            X_ticker, y_ticker = load_single_ticker_data(file_path, ticker, target_column, future_days)
            all_X_data.append(X_ticker)
            all_y_data.append(y_ticker)
            successful_loads += 1
            logger.debug(f"Successfully loaded {ticker}: {X_ticker.shape[0]} records")
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Failed to load {ticker}: {e}")
            failed_loads += 1
        except Exception as e:
            logger.error(f"Unexpected error loading {ticker}: {e}")
            failed_loads += 1

    logger.info(f"Batch loading completed: {successful_loads} successful, {failed_loads} failed")

    if not all_X_data:
        raise ValueError("No data was successfully loaded!")

    # 合并所有数据
    logger.info("Merging all datasets...")
    X_combined = pd.concat(all_X_data, ignore_index=True)
    y_combined = pd.concat(all_y_data, ignore_index=True)

    # 设置多级索引（ticker, date）
    if 'ticker' in X_combined.columns and 'date' in X_combined.columns:
        # 验证和标准化数据类型
        try:
            # 确保ticker列是字符串类型
            X_combined['ticker'] = X_combined['ticker'].astype(str)
            # 确保date列是datetime类型
            X_combined['date'] = pd.to_datetime(X_combined['date'], errors='coerce')
            
            # 检查是否有无效的日期转换
            invalid_dates = X_combined['date'].isna()
            if invalid_dates.any():
                logger.warning(f"Found {invalid_dates.sum()} invalid date entries, removing them")
                X_combined = X_combined[~invalid_dates]
                y_combined = y_combined.iloc[X_combined.index]
            
            # 设置多级索引
            X_combined.set_index(['ticker', 'date'], inplace=True)
            y_combined.index = X_combined.index
            
        except Exception as e:
            logger.error(f"Error setting multi-level index: {e}")
            raise ValueError(f"Failed to create multi-level index: {e}") from e

    # 获取特征列表（排除ticker列）
    all_features = [col for col in X_combined.columns if col != 'ticker']

    logger.info(f"Combined dataset shape: {X_combined.shape}")
    logger.info(f"Number of features: {len(all_features)}")
    logger.info(f"Number of tickers: {X_combined.index.get_level_values('ticker').nunique()}")

    return X_combined, y_combined, all_features


def load_user_dataset(file_path, target_column, future_days=20):
    """
    加载单个用户数据集（保持向后兼容）
    """
    logger.info(f"Loading single dataset from {file_path}")
    user_dataset = pd.read_csv(file_path)

    # 转换日期列为datetime格式
    if 'date' in user_dataset.columns:
        user_dataset['date'] = pd.to_datetime(user_dataset['date'], errors='coerce')
        user_dataset.dropna(subset=['date'], inplace=True)
        # 按日期排序
        user_dataset = user_dataset.sort_values('date')
        
        # 如果存在ticker和date，设置多级索引
        if 'ticker' in user_dataset.columns:
            try:
                # 确保ticker列是字符串类型
                user_dataset['ticker'] = user_dataset['ticker'].astype(str)
                # 确保date列是datetime类型（已经转换过）
                user_dataset.set_index(['ticker', 'date'], inplace=True)
            except Exception as e:
                logger.error(f"Error setting multi-level index in single dataset: {e}")
                raise ValueError(f"Failed to create multi-level index: {e}") from e

    # 如果目标列是 label_shifted，自动生成
    if target_column == 'label_shifted' and target_column not in user_dataset.columns:
        # 计算未来N天的收益率
        user_dataset[target_column] = user_dataset['close'].pct_change(future_days).shift(-future_days)
        logger.info(f"Generated {target_column} with {future_days} days forward return")
        # 移除最后future_days行
        user_dataset = user_dataset[:-future_days]

    # 确保目标列存在
    if target_column not in user_dataset.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # 分离特征和目标
    # 删除不需要的列
    columns_to_drop = [target_column, 'ts_code_x', 'ts_code_y', 'preclose', 'code']
    columns_to_drop = [col for col in columns_to_drop if col in user_dataset.columns]
    X = user_dataset.drop(columns=columns_to_drop)
    y = user_dataset[target_column]

    # 获取特征名称列表
    all_features = X.columns.tolist()

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y, all_features


def check_missing_values(dataset, dataset_name):
    """
    检查数据集中的缺失值
    """
    missing_values = dataset.isnull().sum()
    missing_columns = missing_values[missing_values > 0]

    if not missing_columns.empty:
        logger.warning(f'Missing values in {dataset_name} dataset:')
        logger.warning(missing_columns.head(10))  # 只显示前10个
    else:
        logger.info(f'No missing values in {dataset_name} dataset.')


def apply_alphas_and_return_transformed(X, alpha_formulas, evaluate_formula_func):
    """
    应用顶级alpha公式到数据集，返回包含原始特征和新alpha特征的转换数据集
    """
    transformed_X = X.copy()

    for formula in alpha_formulas:
        try:
            transformed_X[formula] = evaluate_formula_func(formula, X)
            logger.debug(f"Applied formula: {formula}")
        except Exception as e:
            logger.error(f"Failed to apply formula '{formula}': {e}")

    return transformed_X


def get_data_statistics(X, y):
    """
    获取数据集统计信息

    Parameters:
    - X: 特征数据
    - y: 目标数据

    Returns:
    - stats: 统计信息字典
    """
    stats = {
        'total_records': len(X),
        'num_features': X.shape[1],
        'date_range': None,
        'tickers_count': 0,
        'missing_ratio': X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
    }

    if hasattr(X.index, 'get_level_values'):
        try:
            # 多级索引情况
            dates = X.index.get_level_values('date')
            tickers = X.index.get_level_values('ticker')
            stats['date_range'] = (dates.min(), dates.max())
            stats['tickers_count'] = tickers.nunique()
        except:
            pass

    return stats