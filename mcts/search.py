"""MCTS搜索算法实现 - 支持Token系统和传统方式"""
import numpy as np
import logging
import warnings
from scipy.stats import spearmanr, ConstantInputWarning

# 新系统导入
from .trainer import RiskMinerTrainer
from .mdp_environment import AlphaMiningMDP, MDPState
from .mcts_searcher import MCTSSearcher
from .node import MCTSNode as NewMCTSNode

# 保留旧系统的部分功能用于兼容
from alpha.pool import AlphaPool
from .formula_generator import generate_formula

warnings.filterwarnings('ignore', category=ConstantInputWarning)
logger = logging.getLogger(__name__)


def run_mcts_with_token_system(X_train, y_train, num_iterations=200,
                               use_policy_network=True, num_simulations=50, device=None):
    """
    使用新的Token系统运行MCTS

    Args:
        X_train: 训练数据特征
        y_train: 训练数据标签
        num_iterations: 训练迭代次数
        use_policy_network: 是否使用策略网络
        num_simulations: 每次迭代的模拟次数
        device: torch设备(cuda或cpu)

    Returns:
        top_formulas: 最佳公式列表
    """
    logger.info("Starting MCTS with Token System")

    # 创建训练器
    trainer = RiskMinerTrainer(X_train, y_train, use_policy_network=use_policy_network, device=device)

    # 训练
    trainer.train(
        num_iterations=num_iterations,
        num_simulations_per_iteration=num_simulations
    )

    # 获取最佳公式
    top_formulas = trainer.get_top_formulas(n=5)

    # 转换为兼容格式（formula, score）
    result = []
    for formula in top_formulas:
        # 计算IC作为分数
        if trainer.alpha_pool:
            matching_alpha = next((a for a in trainer.alpha_pool if a['formula'] == formula), None)
            if matching_alpha:
                result.append((formula, matching_alpha['ic']))
            else:
                result.append((formula, 0.0))
        else:
            result.append((formula, 0.0))

    return result


# ========== 以下为兼容旧系统的函数 ==========

class LegacyMCTSNode:
    """旧版MCTS节点（用于兼容）"""

    def __init__(self, formula='', parent=None):
        self.formula = formula
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, reward):
        self.visits += 1
        self.value += reward


def ucb1(node, exploration_param=1.41):
    """UCB1算法计算节点的UCB值"""
    if node.visits == 0:
        return np.inf
    parent_visits = node.parent.visits if node.parent is not None else 1
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
            break
        ucb_values = [ucb1(child, exploration_param) for child in current_node.children]
        current_node = current_node.children[np.argmax(ucb_values)]
    return current_node


def expand(node, all_features):
    """扩展节点（旧版）"""
    new_formula = generate_formula(all_features)
    child_node = LegacyMCTSNode(formula=new_formula, parent=node)
    node.add_child(child_node)
    return child_node


def simulate_alpha_performance(node, X, y, ticker=None, evaluate_formula_func=None):
    """模拟alpha公式的性能并计算IC（旧版）"""
    formula = node.formula
    if ticker:
        X = X.loc[ticker]
        y = y.loc[ticker]

    alpha_feature = evaluate_formula_func(formula, X)
    alpha_feature_nonan = alpha_feature.dropna()
    y_aligned = y.loc[alpha_feature_nonan.index]

    ic, _ = spearmanr(alpha_feature_nonan, y_aligned)
    if np.isnan(ic):
        ic = 0
    return ic


def backpropagate(node, reward):
    """反向传播奖励（旧版）"""
    while node is not None:
        node.update(reward)
        node = node.parent


def run_mcts(root, X, y, all_features, num_iterations, evaluate_formula_func):
    """运行MCTS算法（旧版）"""
    for i in range(num_iterations):
        logger.info(f"Legacy MCTS Iteration {i + 1}/{num_iterations}")
        node_to_expand = select_best_node(root)
        expanded_node = expand(node_to_expand, all_features)

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

    top_5_nodes = sorted(
        root.children,
        key=lambda n: n.value / n.visits if n.visits > 0 else 0,
        reverse=True
    )[:5]

    top_5_formulas = [
        (node.formula, node.value / node.visits if node.visits > 0 else 0)
        for node in top_5_nodes
    ]

    return top_5_formulas


def run_mcts_with_quantile(root, X_train, y_train, all_features, num_iterations,
                           evaluate_formula_func, quantile_threshold=0.9):
    """运行带有分位数优化的MCTS（旧版兼容）"""
    # 如果root是旧版节点，使用旧版方法
    if isinstance(root, LegacyMCTSNode):
        logger.info("Using legacy MCTS with quantile")
        # 使用旧版实现...
        return run_mcts(root, X_train, y_train, all_features, num_iterations, evaluate_formula_func)
    else:
        # 使用新的Token系统
        logger.info("Redirecting to Token-based MCTS")
        return run_mcts_with_token_system(X_train, y_train, num_iterations, use_policy_network=False)


def run_mcts_with_risk_seeking(root, X_train, y_train, all_features,
                               num_iterations, evaluate_formula_func,
                               quantile_threshold=0.85, device=None):
    """运行带风险寻求策略的MCTS"""
    logger.info("Using Token-based MCTS with Risk Seeking")
    return run_mcts_with_token_system(
        X_train, y_train,
        num_iterations=num_iterations,
        use_policy_network=True,  # 使用策略网络
        num_simulations=50,
        device=device
    )