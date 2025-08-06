"""RiskMiner完整训练器"""
import logging
import numpy as np
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .node import MCTSNode
from .mcts_searcher import MCTSSearcher
from .mdp_environment import AlphaMiningMDP, MDPState, RewardCalculator
from .token_system import TOKEN_DEFINITIONS
from .rpn_evaluator import RPNEvaluator
from policy.alpha_policy_network import AlphaMiningPolicyNetwork
from policy.risk_seeking import RiskSeekingOptimizer

logger = logging.getLogger(__name__)


class RiskMinerTrainer:
    """完整的RiskMiner训练器"""

    def __init__(self, X_data, y_data, use_policy_network=True, device=None):
        self.X_data = X_data
        self.y_data = y_data
        self.use_policy_network = use_policy_network
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 初始化组件
        self.mdp_env = AlphaMiningMDP()

        # 初始化策略网络（如果启用）
        if use_policy_network:
            self.policy_network = AlphaMiningPolicyNetwork().to(self.device)
            self.optimizer = RiskSeekingOptimizer(self.policy_network, device=self.device)
            logger.info(f"Policy network moved to {self.device}")
        else:
            self.policy_network = None
            self.optimizer = None

        self.mcts_searcher = MCTSSearcher(self.policy_network, c_puct=1.0, device=self.device)
        self.alpha_pool = []
        self.reward_calculator = RewardCalculator(self.alpha_pool)

        # 更新reward_calculator的评估函数
        self.reward_calculator.evaluate_partial_formula = self.evaluate_formula_wrapper
        self.reward_calculator.evaluate_complete_formula = self.evaluate_formula_wrapper

    def evaluate_formula_wrapper(self, state, X_data):
        """评估公式的包装函数"""
        try:
            # 将数据转换为字典格式
            if hasattr(X_data, 'to_dict'):
                data_dict = X_data.to_dict('series')
            else:
                data_dict = X_data

            # 使用RPN求值器评估
            result = RPNEvaluator.evaluate(state.token_sequence, data_dict)

            # 确保返回numpy数组
            if result is not None:
                if hasattr(result, 'values'):
                    return result.values
                else:
                    return np.array(result)
            return None
        except Exception as e:
            logger.error(f"Error evaluating formula: {e}")
            return None

    def train(self, num_iterations=1000, num_simulations_per_iteration=50):
        """主训练循环"""

        for iteration in range(num_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

            # 阶段1：MCTS搜索收集轨迹
            trajectories = self.collect_trajectories_with_mcts(
                num_episodes=10,
                num_simulations_per_episode=num_simulations_per_iteration
            )

            # 阶段2：使用收集的轨迹训练策略网络（如果启用）
            if self.use_policy_network and self.optimizer:
                avg_loss = self.train_policy_network(trajectories)
                logger.info(f"Policy network loss: {avg_loss:.4f}")

            # 阶段3：评估和更新Alpha池
            self.update_alpha_pool(trajectories, iteration)

            # 打印统计信息
            if (iteration + 1) % 10 == 0:
                self.print_statistics()

    def collect_trajectories_with_mcts(self, num_episodes, num_simulations_per_episode):
        """使用MCTS收集训练轨迹"""
        all_trajectories = []

        for episode in range(num_episodes):
            # 重置环境
            initial_state = self.mdp_env.reset()

            # 创建根节点
            root = MCTSNode(state=initial_state)

            # 执行MCTS搜索
            episode_trajectory = []
            for _ in range(num_simulations_per_episode):
                trajectory = self.mcts_searcher.search_one_iteration(
                    root, self.mdp_env, self.reward_calculator,
                    self.X_data, self.y_data
                )
                if trajectory:
                    episode_trajectory.extend(trajectory)

            # 选择最佳动作序列作为最终轨迹
            final_trajectory = self.extract_best_trajectory(root)
            if final_trajectory:
                all_trajectories.append(final_trajectory)

            # 记录生成的公式
            if root.children:
                best_child = root.get_best_child(c_puct=0)  # 贪婪选择
                if best_child and best_child.state:
                    formula = RPNEvaluator.tokens_to_infix(best_child.state.token_sequence)
                    logger.debug(f"Episode {episode + 1}: {formula}")

        return all_trajectories

    def train_policy_network(self, trajectories):
        """训练策略网络"""
        if not self.optimizer:
            return 0.0

        total_loss = 0
        num_updates = 0

        for trajectory in trajectories:
            if trajectory:  # 确保轨迹非空
                loss = self.optimizer.train_on_episode(trajectory)
                if loss > 0:  # 只有超过分位数的轨迹才会产生损失
                    total_loss += loss
                    num_updates += 1

        avg_loss = total_loss / max(num_updates, 1)

        if num_updates > 0:
            logger.info(f"Policy network updated: {num_updates}/{len(trajectories)} episodes")
            logger.info(f"Current quantile: {self.optimizer.quantile_estimate:.4f}")

        return avg_loss

    def extract_best_trajectory(self, root):
        """从MCTS树中提取最佳轨迹"""
        trajectory = []
        current = root
        max_depth = 30
        depth = 0

        while current.children and not current.is_terminal() and depth < max_depth:
            # 选择访问次数最多的子节点
            best_child = current.get_best_child(c_puct=0)  # 贪婪选择

            if not best_child:
                break

            # 计算这一步的奖励
            if best_child.state.token_sequence[-1].name == 'END':
                reward = self.reward_calculator.calculate_terminal_reward(
                    best_child.state, self.X_data, self.y_data,
                    self.evaluate_formula_wrapper
                )
            else:
                reward = self.reward_calculator.calculate_intermediate_reward(
                    best_child.state, self.X_data, self.y_data,
                    self.evaluate_formula_wrapper
                )

            trajectory.append((current.state, best_child.action, reward))
            current = best_child
            depth += 1

        return trajectory

    def update_alpha_pool(self, trajectories, iteration):
        """更新Alpha池"""
        new_formulas = []

        for trajectory in trajectories:
            if not trajectory:
                continue

            # 检查轨迹是否生成了完整公式
            last_state, last_action, _ = trajectory[-1]

            if last_action == 'END':
                # 构建完整状态
                final_state = last_state.copy()
                final_state.add_token('END')

                # 转换为可读公式
                formula_str = RPNEvaluator.tokens_to_infix(final_state.token_sequence)

                # 评估公式
                alpha_values = self.evaluate_formula_wrapper(final_state, self.X_data)
                if alpha_values is not None:
                    ic = self.reward_calculator.calculate_ic(alpha_values, self.y_data)

                    # 添加到池中
                    new_formulas.append({
                        'formula': formula_str,
                        'ic': ic,
                        'values': alpha_values,
                        'iteration': iteration
                    })

        # 添加新公式到池中
        for formula_info in new_formulas:
            self.alpha_pool.append(formula_info)
            logger.info(f"New formula added: {formula_info['formula'][:50]}... IC={formula_info['ic']:.4f}")

        # 保持池大小
        if len(self.alpha_pool) > 100:
            self.alpha_pool.sort(key=lambda x: x['ic'], reverse=True)
            self.alpha_pool = self.alpha_pool[:100]

    def print_statistics(self):
        """打印训练统计信息"""
        logger.info("\n=== Training Statistics ===")
        logger.info(f"Alpha pool size: {len(self.alpha_pool)}")

        if self.alpha_pool:
            top_5 = sorted(self.alpha_pool, key=lambda x: x['ic'], reverse=True)[:5]
            logger.info("\nTop 5 Alphas:")
            for i, alpha in enumerate(top_5, 1):
                formula = alpha['formula']
                if len(formula) > 80:
                    formula = formula[:77] + "..."
                logger.info(f"{i}. Formula: {formula}")
                logger.info(f"   IC: {alpha['ic']:.4f}")

        if self.optimizer:
            logger.info(f"\nCurrent quantile estimate: {self.optimizer.quantile_estimate:.4f}")

    def get_top_formulas(self, n=5):
        """获取最佳的n个公式"""
        if not self.alpha_pool:
            return []

        sorted_pool = sorted(self.alpha_pool, key=lambda x: x['ic'], reverse=True)
        return [alpha['formula'] for alpha in sorted_pool[:n]]