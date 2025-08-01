"""主程序入口"""
import argparse
import logging
from sklearn.model_selection import train_test_split

from config.config import *
from data.data_loader import (
    load_user_dataset,
    check_missing_values,
    apply_alphas_and_return_transformed
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
    logger.info("Starting RiskMiner Algorithm")

    # 第1部分：数据准备和探索
    logger.info("=== Part 1: Data Preparation & Exploration ===")
    X, y, all_features = load_user_dataset(args.data_path, args.target_column)
    check_missing_values(X, 'user_dataset')

    # 划分训练集和测试集
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

    logger.info("RiskMiner Algorithm completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RiskMiner Algorithm")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV data file"
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="label_shifted",
        help="Name of the target column"
    )
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

    args = parser.parse_args()
    main(args)