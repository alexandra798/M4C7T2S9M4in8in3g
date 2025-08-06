"""MCTS搜索算法实现"""
import numpy as np
import logging
import warnings
from scipy.stats import spearmanr, ConstantInputWarning

from alpha.pool import AlphaPool
from .node import MCTSNode
from .formula_generator import generate_formula
from .policy_network import PolicyNetwork, RiskSeekingPolicyOptimizer
from .mdp import RewardDenseMDP

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


def puct_selection(node, policy_network=None, c_puct=1.0):
    """
    PUCT选择（论文公式6）
    a_t = argmax_a [Q(s,a) + P(s,a) * sqrt(Σ_b N(s,b)) / (1 + N(s,a))]
    """
    if not node.children:
        return None

    total_visits = sum(child.visits for child in node.children)

    best_value = -float('inf')
    best_child = None

    for child in node.children:
        # Q值：平均奖励
        q_value = child.value / child.visits if child.visits > 0 else 0

        # 先验概率P(s,a)
        if policy_network and hasattr(child, 'prior_prob'):
            prior_prob = child.prior_prob
        else:
            # 如果没有策略网络，使用均匀分布
            prior_prob = 1.0 / len(node.children)

        # PUCT值
        exploration_term = c_puct * prior_prob * np.sqrt(total_visits) / (1 + child.visits)
        puct_value = q_value + exploration_term

        if puct_value > best_value:
            best_value = puct_value
            best_child = child

    return best_child


def expand_with_policy(node, all_features, policy_network=None):
    """使用策略网络指导的扩展"""
    if policy_network:
        # 将当前状态编码为特征向量
        state_features = encode_state(node, all_features)

        # 获取动作概率分布
        action_probs = policy_network.get_action_probabilities(state_features)

        # 根据概率分布选择要生成的公式类型
        # 这里简化处理，实际应该根据action_probs选择具体的操作
        new_formula = generate_formula_with_guidance(all_features, action_probs)
    else:
        new_formula = generate_formula(all_features)

    child_node = MCTSNode(formula=new_formula, parent=node)

    # 设置先验概率
    if policy_network:
        child_node.prior_prob = action_probs[0]  # 简化：使用第一个概率
    else:
        child_node.prior_prob = 1.0 / 100  # 默认均匀分布

    node.add_child(child_node)
    return child_node


def run_mcts_with_risk_seeking(root, X_train, y_train, all_features,
                               num_iterations, evaluate_formula_func,
                               quantile_threshold=0.85):
    """
    运行带风险寻求策略的MCTS（完整实现）
    """
    # 初始化组件
    input_dim = len(all_features)
    policy_network = PolicyNetwork(input_dim=input_dim)
    policy_optimizer = RiskSeekingPolicyOptimizer(
        policy_network,
        quantile_alpha=quantile_threshold
    )

    alpha_pool = AlphaPool()
    mdp = RewardDenseMDP(alpha_pool)

    # 存储轨迹用于训练
    episode_buffer = []

    for iteration in range(num_iterations):
        logger.info(f"MCTS Iteration {iteration + 1}/{num_iterations}")

        # 执行MCTS搜索循环
        trajectory = []
        current_node = root

        # Selection阶段：使用PUCT选择
        path = [current_node]
        while current_node.children and not is_terminal(current_node):
            current_node = puct_selection(current_node, policy_network)
            path.append(current_node)

        # Expansion阶段：使用策略网络指导扩展
        if not is_terminal(current_node):
            new_node = expand_with_policy(current_node, all_features, policy_network)
            path.append(new_node)
            current_node = new_node

        # Simulation阶段：计算奖励
        rewards = []
        for node in path[1:]:  # 跳过根节点
            if node.formula:
                # 计算中间奖励
                intermediate_reward = mdp.calculate_intermediate_reward(
                    node.formula, X_train, y_train, evaluate_formula_func
                )
                rewards.append(intermediate_reward)

        # 如果到达终止状态，计算终止奖励
        if is_terminal(current_node):
            terminal_reward = mdp.calculate_terminal_reward(
                current_node.formula, X_train, y_train, evaluate_formula_func
            )
            rewards.append(terminal_reward)

        # Backpropagation阶段
        cumulative_reward = sum(rewards)
        for node in reversed(path):
            node.update(cumulative_reward)

        # 存储轨迹
        states = [encode_state(node, all_features) for node in path]
        actions = [0] * len(path)  # 简化：实际应该记录具体动作
        episode_buffer.append((states, actions, rewards))

        # 定期训练策略网络
        if len(episode_buffer) >= 32:  # 批量大小32
            policy_optimizer.train_on_batch(episode_buffer)
            episode_buffer = []

    # 返回最佳公式
    return get_best_formulas_from_tree(root, n=5)


def encode_state(node, all_features):
    """将节点状态编码为特征向量"""
    # 简化实现：使用one-hot编码或其他方式
    # 实际应该编码当前公式的结构信息
    state_vector = np.zeros(len(all_features))

    if node.formula:
        # 根据公式中包含的特征设置对应位置为1
        for i, feature in enumerate(all_features):
            if feature in node.formula:
                state_vector[i] = 1.0

    return state_vector


def is_terminal(node):
    """判断是否为终止状态"""
    # 简化：基于公式长度或复杂度判断
    if not node.formula:
        return False

    # 如果公式达到最大长度，视为终止
    max_length = 100  # 可配置
    return len(node.formula) > max_length


def generate_formula_with_guidance(all_features, action_probs):
    """根据策略网络的指导生成公式"""
    # 根据action_probs选择操作类型
    # 这里需要建立动作空间到公式生成的映射
    # 简化实现：仍使用随机生成，但可以根据概率调整
    return generate_formula(all_features)


def get_best_formulas_from_tree(root, n=5):
    """从搜索树中获取最佳公式"""
    all_nodes = []

    def traverse(node):
        if node.formula and node.visits > 0:
            all_nodes.append(node)
        for child in node.children:
            traverse(child)

    traverse(root)

    # 按平均奖励排序
    all_nodes.sort(
        key=lambda n: n.value / n.visits if n.visits > 0 else 0,
        reverse=True
    )

    # 返回前n个唯一公式
    formulas = []
    seen = set()
    for node in all_nodes:
        if node.formula not in seen:
            score = node.value / node.visits if node.visits > 0 else 0
            formulas.append((node.formula, score))
            seen.add(node.formula)
            if len(formulas) >= n:
                break

    return formulas

