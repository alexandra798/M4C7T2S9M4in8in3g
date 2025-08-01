"""主程序入口 - 支持批量CSV文件处理"""
import argparse
import logging
import os
from sklearn.model_selection import train_test_split

from config.config import *
from data.data_loader import (
    load_user_dataset,
    load_batch_datasets,
    check_missing_values,
    apply_alphas_and_return_transformed,
    get_data_statistics
)
from mcts.node import MCTSNode
from mcts.search import run_mcts_with_quantile
from alpha.pool import AlphaPool
from alpha.evaluation import evaluate_formula
from validation.cross_validation import cross_validate_formulas
from validation.backtest import backtest_formulas

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """主函数"""
    logger.info("Starting RiskMiner Algorithm with Batch Processing")

    # 第1部分：数据准备和探索
    logger.info("=== Part 1: Data Preparation & Exploration ===")

    if args.batch_mode:
        # 批量模式：从csi300.txt和数据目录加载
        logger.info("Running in batch mode...")
        if not args.data_directory:
            raise ValueError("--data_directory is required in batch mode")
        if not args.csi300_file:
            raise ValueError("--csi300_file is required in batch mode")

        X, y, all_features = load_batch_datasets(
            data_directory=args.data_directory,
            csi300_file_path=args.csi300_file,
            target_column=args.target_column,
            file_pattern=args.file_pattern,
            future_days=args.future_days  # 新增参数
        )
    else:
        # 单文件模式：保持原有逻辑
        logger.info("Running in single file mode...")
        X, y, all_features = load_user_dataset(
            args.data_path,
            args.target_column,
            future_days=args.future_days  # 新增参数
        )

    # 显示数据统计信息
    stats = get_data_statistics(X, y)
    logger.info("=== Dataset Statistics ===")
    logger.info(f"Total records: {stats['total_records']:,}")
    logger.info(f"Number of features: {stats['num_features']}")
    logger.info(f"Number of tickers: {stats['tickers_count']}")
    if stats['date_range']:
        logger.info(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    logger.info(f"Missing data ratio: {stats['missing_ratio']:.4f}")

    check_missing_values(X, 'dataset')

    # 划分训练集和测试集
    if args.batch_mode:
        # 对于多ticker数据，使用时间序列划分
        logger.info("Using time-based split for multi-ticker data...")

        # 获取所有日期并排序
        dates = X.index.get_level_values('date').unique().sort_values()
        dates_len = len(dates)
        if dates_len == 0:
            raise ValueError("No dates available for splitting dataset")
        split_index = int(dates_len * 0.8)
        if split_index >= dates_len:
            split_index = dates_len - 1
        split_date = dates[split_index]  # 80%作为训练集

        train_mask = X.index.get_level_values('date') <= split_date
        test_mask = X.index.get_level_values('date') > split_date

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        logger.info(f"Split date: {split_date}")
    else:
        # 单文件模式使用原有划分方式
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 第2-4部分：MCTS和Alpha池管理
    logger.info("=== Parts 2-4: MCTS & Alpha Pool Management ===")

    # 初始化根节点和alpha池
    root_node = MCTSNode(formula='')
    alpha_pool = AlphaPool(
        pool_size=ALPHA_POOL_CONFIG['pool_size'],
        lambda_param=ALPHA_POOL_CONFIG['lambda_param']
    )

    # 运行带分位数优化的MCTS
    if args.batch_mode:
        # 批量模式：确保MCTS能处理多级索引数据
        logger.info("Running MCTS with multi-ticker data...")
        best_formulas_quantile = run_mcts_with_quantile(
            root_node,
            X_train,
            y_train,
            all_features,
            MCTS_CONFIG['num_iterations'],
            evaluate_formula,
            MCTS_CONFIG['quantile_threshold']
        )
    else:
        # 单文件模式
        best_formulas_quantile = run_mcts_with_quantile(
            root_node,
            X_train,
            y_train,
            all_features,
            MCTS_CONFIG['num_iterations'],
            evaluate_formula,
            MCTS_CONFIG['quantile_threshold']
        )

    # 将最佳公式添加到alpha池
    for formula, score in best_formulas_quantile:
        alpha_pool.add_to_pool({'formula': formula, 'score': score})

    # 更新alpha池
    alpha_pool.update_pool(X_train, y_train, evaluate_formula)

    # 获取前5个公式
    top_formulas = alpha_pool.get_top_formulas(5)
    logger.info(f"Top formulas from alpha pool: {top_formulas}")

    # 第5部分：应用公式转换数据集
    if args.transform_data:
        logger.info("=== Part 5: Apply Formulas to Transform Dataset ===")
        transformed_X = apply_alphas_and_return_transformed(X, top_formulas, evaluate_formula)
        logger.info(f"Transformed dataset shape: {transformed_X.shape}")

        # 可选：保存转换后的数据
        if args.save_transformed:
            output_path = args.output_path or "transformed_data.csv"
            logger.info(f"Saving transformed data to {output_path}")
            transformed_X.to_csv(output_path)

    # 第6部分：交叉验证
    if args.cross_validate:
        logger.info("=== Part 6: Cross-Validation ===")
        cv_results = cross_validate_formulas(
            top_formulas,
            X,
            y,
            CV_CONFIG['n_splits'],
            evaluate_formula
        )

        logger.info("\nCross-validation results:")
        for formula, results in cv_results.items():
            logger.info(f"\nFormula: {formula}")
            logger.info(f"Mean IC: {results['Mean IC']:.4f}")
            logger.info(f"IC Std Dev: {results['IC Std Dev']:.4f}")

    # 第7部分：回测
    if args.backtest:
        logger.info("=== Part 7: Backtest ===")
        backtest_results = backtest_formulas(top_formulas, X_test, y_test)

        # 按IC值排序结果
        sorted_results = sorted(backtest_results.items(), key=lambda x: x[1], reverse=True)

        logger.info("\nSorted backtest results (by IC):")
        for formula, ic in sorted_results:
            logger.info(f"Formula: {formula}")
            logger.info(f"Information Coefficient (IC): {ic:.4f}\n")

    # 保存结果
    if args.save_results:
        results_path = args.results_path or "alpha_results.txt"
        logger.info(f"Saving results to {results_path}")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("=== Top Alpha Formulas ===\n")
            for i, formula in enumerate(top_formulas, 1):
                f.write(f"{i}. {formula}\n")

            if args.backtest:
                f.write("\n=== Backtest Results ===\n")
                for formula, ic in sorted_results:
                    f.write(f"Formula: {formula}, IC: {ic:.4f}\n")

    logger.info("RiskMiner Algorithm completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RiskMiner Algorithm with Batch Processing")

    # 基本参数
    parser.add_argument(
        "--target_column",
        type=str,
        default="label_shifted",
        help="Name of the target column"
    )

    # 新增参数：未来天数
    parser.add_argument(
        "--future_days",
        type=int,
        default=20,
        help="Number of days to calculate future returns (default: 20)"
    )

    # 批量处理模式参数
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Enable batch processing mode for multiple CSV files"
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        help="Directory containing CSV files (required in batch mode)"
    )
    parser.add_argument(
        "--csi300_file",
        type=str,
        help="Path to csi300.txt file (required in batch mode)"
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="{ticker}.csv",
        help="File naming pattern, {ticker} will be replaced with actual ticker"
    )

    # 单文件模式参数（保持向后兼容）
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the CSV data file (single file mode)"
    )

    # 功能开关
    parser.add_argument(
        "--transform_data",
        action="store_true",
        help="Apply formulas to transform the dataset"
    )
    parser.add_argument(
        "--cross_validate",
        action="store_true",
        help="Perform cross-validation"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Perform backtesting"
    )

    # 输出参数
    parser.add_argument(
        "--save_transformed",
        action="store_true",
        help="Save transformed dataset to file"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save results to file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save transformed data"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Path to save results"
    )

    args = parser.parse_args()

    # 验证参数
    if args.batch_mode:
        if not args.data_directory or not args.csi300_file:
            parser.error("--data_directory and --csi300_file are required in batch mode")
    else:
        if not args.data_path:
            parser.error("--data_path is required in single file mode")

    main(args)