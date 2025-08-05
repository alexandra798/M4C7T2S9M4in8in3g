"""MCTS搜索算法实现"""
import numpy as np
import logging
import warnings
from scipy.stats import spearmanr, ConstantInputWarning
from .node import MCTSNode
from .formula_generator import generate_formula

# 忽略常量输入警告
warnings.filterwarnings('ignore', category=ConstantInputWarning)

logger = logging.getLogger(__name__)


def ucb1(node, exploration_param=1.41):
    """UCB1算法计算节点的UCB值"""
    if node.visits == 0:
        return np.inf
    parent_visits = node.parent.visits if node.parent is not None else 1
    # 防御性编程：确保除法安全
    visits = max(node.visits, 1)
    parent_visits = max(parent_visits, 1)
    
    exploitation = node.value / visits
    exploration = exploration_param * np.sqrt(np.log(parent_visits) / visits)
    return exploitation + exploration


def select_best_node(node, exploration_param=1.41):
    """选择最佳节点进行扩展"""
    current_node = node
    while not current_node.is_fully_expanded():
        if not current_node.children:
            logger.debug(f"Node {current_node.formula} has no children to select.")
            break
        ucb_values = [ucb1(child, exploration_param) for child in current_node.children]
        current_node = current_node.children[np.argmax(ucb_values)]
    return current_node


def expand(node, all_features):
    """扩展节点，添加新公式的子节点"""
    new_formula = generate_formula(all_features)
    child_node = MCTSNode(formula=new_formula, parent=node)
    node.add_child(child_node)
    logger.debug(f"Expanded Node: {node.formula}, New Child Formula: {child_node.formula}")
    return child_node


def simulate_alpha_performance(node, X, y, ticker=None, evaluate_formula_func=None):
    """
    模拟alpha公式的性能并计算IC
    """
    formula = node.formula

    # 如果指定了ticker，则对数据进行子集化
    if ticker:
        X = X.loc[ticker]
        y = y.loc[ticker]

    # 评估alpha公式
    alpha_feature = evaluate_formula_func(formula, X)

    # 删除NaN值
    alpha_feature_nonan = alpha_feature.dropna()
    y_aligned = y.loc[alpha_feature_nonan.index]

    # 计算信息系数（IC）
    ic, _ = spearmanr(alpha_feature_nonan, y_aligned)
    if np.isnan(ic):
        ic = 0

    return ic


def simulate_alpha_performance_quantile(node, X_train, y_train, ticker=None,
                                        evaluate_formula_func=None, quantile_threshold=0.9):
    """
    使用分位数奖励计算模拟alpha公式的性能
    """
    formula = node.formula

    if ticker:
        X_train = X_train.loc[ticker]
        y_train = y_train.loc[ticker]

    alpha_feature = evaluate_formula_func(formula, X_train)

    alpha_feature_nonan = alpha_feature.dropna()
    y_train_aligned = y_train.loc[alpha_feature_nonan.index]

    # 计算高分位数的收益
    high_quantile_alpha = alpha_feature_nonan[
        alpha_feature_nonan >= alpha_feature_nonan.quantile(quantile_threshold)
        ]
    high_quantile_y = y_train_aligned.loc[high_quantile_alpha.index]

    # 检查是否有足够的数据点和变异性
    if len(high_quantile_alpha) < 2:
        return 0
    
    # 检查是否为常数数组
    if high_quantile_alpha.nunique() <= 1 or high_quantile_y.nunique() <= 1:
        return 0
    
    # 计算高分位数数据的IC
    try:
        ic, _ = spearmanr(high_quantile_alpha, high_quantile_y)
        if np.isnan(ic):
            ic = 0
    except:
        ic = 0

    return ic


def backpropagate(node, reward):
    """反向传播奖励"""
    while node is not None:
        node.update(reward)
        node = node.parent


def run_mcts(root, X, y, all_features, num_iterations, evaluate_formula_func):
    """
    运行MCTS算法
    """
    for i in range(num_iterations):
        logger.info(f"Iteration {i + 1}/{num_iterations}")
        node_to_expand = select_best_node(root)
        expanded_node = expand(node_to_expand, all_features)

        # 评估每个ticker的alpha公式
        if 'ticker' in X.index.names:
            tickers = X.index.get_level_values('ticker').unique()
            for ticker in tickers:
                reward = simulate_alpha_performance(
                    expanded_node, X, y, ticker, evaluate_formula_func
                )
                backpropagate(expanded_node, reward)
        else:
            reward = simulate_alpha_performance(
                expanded_node, X, y, evaluate_formula_func=evaluate_formula_func
            )
            backpropagate(expanded_node, reward)

    # 获取前5个公式
    top_5_nodes = sorted(
        root.children,
        key=lambda n: n.value / n.visits if n.visits > 0 else 0,
        reverse=True
    )[:5]
    top_5_formulas = [
        (node.formula, node.value / node.visits if node.visits > 0 else 0)
        for node in top_5_nodes
    ]

    logger.info("Top 5 formulas discovered by MCTS:")
    for i, (formula, score) in enumerate(top_5_formulas):
        logger.info(f"{i + 1}. Formula: {formula}, Score: {score:.4f}")

    return top_5_formulas


def run_mcts_with_quantile(root, X_train, y_train, all_features, num_iterations,
                           evaluate_formula_func, quantile_threshold=0.9):
    """
    运行带有分位数优化的MCTS
    """
    for i in range(num_iterations):
        logger.info(f"Iteration {i + 1}/{num_iterations}")
        node_to_expand = select_best_node(root, exploration_param=2.0)
        expanded_node = expand(node_to_expand, all_features)

        # 使用分位数奖励评估
        for ticker in X_train.index.get_level_values('ticker').unique():
            reward = simulate_alpha_performance_quantile(
                expanded_node, X_train, y_train, ticker,
                evaluate_formula_func, quantile_threshold
            )
            backpropagate(expanded_node, reward)

    # 收集所有访问过的节点
    all_nodes = []
    nodes_to_explore = [root]

    while nodes_to_explore:
        current_node = nodes_to_explore.pop(0)
        if current_node.visits > 0:
            all_nodes.append(current_node)
        nodes_to_explore.extend(current_node.children)

    all_nodes.sort(key=lambda n: n.value / n.visits if n.visits > 0 else 0, reverse=True)

    # 选择前5个唯一公式
    top_5_formulas = []
    seen_formulas = set()

    for node in all_nodes:
        formula = node.formula
        if formula not in seen_formulas:
            score = node.value / node.visits if node.visits > 0 else 0
            top_5_formulas.append((formula, score))
            seen_formulas.add(formula)
            if len(top_5_formulas) == 5:
                break

    logger.info("Top 5 formulas discovered by MCTS with quantile optimization:")
    for i, (formula, score) in enumerate(top_5_formulas):
        logger.info(f"{i + 1}. Formula: {formula}, Score: {score:.4f}")

    return top_5_formulas