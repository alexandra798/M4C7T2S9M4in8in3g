"""MDP环境和状态表示"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import logging
from .token_system import (
    TOKEN_DEFINITIONS, TOKEN_TO_INDEX, INDEX_TO_TOKEN,
    TOTAL_TOKENS, TokenType, RPNValidator
)

logger = logging.getLogger(__name__)


class MDPState:
    """MDP环境的状态"""

    def __init__(self):
        self.token_sequence = [TOKEN_DEFINITIONS['BEG']]
        self.step_count = 0
        self.stack_size = 0

    def add_token(self, token_name):
        """添加一个Token到序列"""
        token = TOKEN_DEFINITIONS[token_name]
        self.token_sequence.append(token)
        self.step_count += 1

        # 更新栈大小
        if token.type == TokenType.OPERAND:
            self.stack_size += 1
        elif token.type == TokenType.OPERATOR:
            self.stack_size = self.stack_size - token.arity + 1

    def encode_for_network(self):
        """编码状态用于神经网络输入"""
        # 创建固定大小的编码（最大30个token）
        max_length = 30
        encoding = np.zeros((max_length, TOTAL_TOKENS + 3))  # +3 for position and stack info

        for i, token in enumerate(self.token_sequence[:max_length]):
            if i >= max_length:
                break

            # One-hot编码当前token
            token_idx = TOKEN_TO_INDEX[token.name]
            encoding[i, token_idx] = 1

            # 位置编码
            encoding[i, TOTAL_TOKENS] = i / max_length

            # 栈大小编码
            encoding[i, TOTAL_TOKENS + 1] = self.stack_size / 10.0

            # 步数编码
            encoding[i, TOTAL_TOKENS + 2] = self.step_count / max_length

        return encoding

    def to_formula_string(self):
        """将Token序列转换为可读的公式字符串"""
        # 这里需要实现RPN到中缀表达式的转换
        # 简化版本：直接返回token名称序列
        return ' '.join([t.name for t in self.token_sequence[1:]])  # 跳过BEG

    def copy(self):
        """深拷贝状态"""
        new_state = MDPState()
        new_state.token_sequence = self.token_sequence.copy()
        new_state.step_count = self.step_count
        new_state.stack_size = self.stack_size
        return new_state


class AlphaMiningMDP:
    """完整的马尔可夫决策过程环境"""

    def __init__(self):
        self.max_episode_length = 30  # 最多30个Token
        self.current_state = None
        self.alpha_pool = []  # 已发现的好公式池

    def reset(self):
        """开始新的episode"""
        self.current_state = MDPState()
        self.current_state.token_sequence = [TOKEN_DEFINITIONS['BEG']]
        self.current_state.stack_count = 0
        self.current_state.step_count = 0
        return self.current_state

    def step(self, action_token):
        """执行一个动作（选择一个Token）"""
        # 1. 验证动作合法性
        if not self.is_valid_action(action_token):
            return self.current_state, -1.0, True  # 非法动作，负奖励，结束

        # 2. 更新状态
        self.current_state.add_token(action_token)

        # 3. 计算奖励
        if action_token == 'END':
            # 终止奖励：基于完整公式的质量
            reward = 0.0  # 将在外部计算
            done = True
        else:
            # 中间奖励：基于部分公式的潜力
            reward = 0.0  # 将在外部计算
            done = False

        # 4. 检查是否达到最大长度
        if self.current_state.step_count >= self.max_episode_length:
            done = True

        return self.current_state, reward, done

    def is_valid_action(self, action_token):
        """检查动作是否合法"""
        valid_actions = RPNValidator.get_valid_next_tokens(self.current_state.token_sequence)
        return action_token in valid_actions

    def get_valid_actions(self):
        """获取当前状态的合法动作"""
        return RPNValidator.get_valid_next_tokens(self.current_state.token_sequence)


class RewardCalculator:
    """计算MDP的奖励"""

    def __init__(self, alpha_pool, lambda_param=0.1):
        self.alpha_pool = alpha_pool
        self.lambda_param = lambda_param

    def calculate_intermediate_reward(self, state, X_data, y_data, evaluate_func):
        """
        计算中间奖励（部分公式的奖励）

        公式：Reward_inter = IC - λ * (1/k) * Σ mutIC_i
        其中：
        - IC: 当前部分公式与目标的相关性
        - mutIC_i: 与池中第i个alpha的相互相关性
        - λ: 平衡参数（0.1）
        """
        # 检查是否为合法的部分表达式
        if not RPNValidator.is_valid_partial_expression(state.token_sequence):
            return 0.0

        # 评估部分公式
        try:
            alpha_values = self.evaluate_partial_formula(state, X_data)
            if alpha_values is None:
                return 0.0

            # 计算IC（信息系数）
            ic = self.calculate_ic(alpha_values, y_data)

            # 如果池为空，直接返回IC
            if len(self.alpha_pool) == 0:
                return ic

            # 计算与池中其他alpha的平均相互IC
            mutual_ic_sum = 0
            for existing_alpha in self.alpha_pool:
                mutual_ic = self.calculate_ic(alpha_values, existing_alpha['values'])
                mutual_ic_sum += abs(mutual_ic)

            avg_mutual_ic = mutual_ic_sum / len(self.alpha_pool)

            # 最终奖励
            reward = ic - self.lambda_param * avg_mutual_ic

            return reward

        except Exception as e:
            logger.error(f"Error calculating intermediate reward: {e}")
            return 0.0

    def calculate_terminal_reward(self, state, X_data, y_data, evaluate_func):
        """
        计算终止奖励（完整公式的奖励）

        步骤：
        1. 将新alpha添加到池中
        2. 训练线性组合模型
        3. 返回组合alpha的IC
        """
        # 验证是否为完整公式
        if state.token_sequence[-1].name != 'END':
            return -1.0  # 未正确终止的惩罚

        # 评估完整公式
        try:
            alpha_values = self.evaluate_complete_formula(state, X_data)
            if alpha_values is None:
                return -0.5

            # 计算单独的IC
            individual_ic = self.calculate_ic(alpha_values, y_data)

            # 添加到池中
            self.alpha_pool.append({
                'formula': state.to_formula_string(),
                'values': alpha_values,
                'ic': individual_ic
            })

            # 维护池大小（最多100个）
            if len(self.alpha_pool) > 100:
                # 删除IC最低的
                self.alpha_pool.sort(key=lambda x: x['ic'], reverse=True)
                self.alpha_pool = self.alpha_pool[:100]

            # 训练线性组合模型并计算组合IC
            composite_ic = self.train_and_evaluate_composite_alpha(X_data, y_data)

            return composite_ic

        except Exception as e:
            logger.error(f"Error calculating terminal reward: {e}")
            return -0.5

    def calculate_ic(self, predictions, targets):
        """计算信息系数（Pearson相关系数）"""
        # 处理NaN值
        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        if valid_mask.sum() < 2:
            return 0.0

        corr, _ = pearsonr(predictions[valid_mask], targets[valid_mask])
        return corr if not np.isnan(corr) else 0.0

    def evaluate_partial_formula(self, state, X_data):
        """评估部分RPN公式"""
        # 这里需要实现RPN求值器
        # 暂时返回None，将在rpn_evaluator.py中实现
        return None

    def evaluate_complete_formula(self, state, X_data):
        """评估完整RPN公式"""
        # 这里需要实现完整的RPN求值器
        # 暂时返回None，将在rpn_evaluator.py中实现
        return None

    def train_and_evaluate_composite_alpha(self, X_data, y_data):
        """训练线性组合模型并评估"""
        from sklearn.linear_model import LinearRegression

        if len(self.alpha_pool) == 0:
            return 0.0

        # 构建特征矩阵（每个alpha的值）
        features = np.column_stack([alpha['values'] for alpha in self.alpha_pool])

        # 训练线性模型
        model = LinearRegression()
        model.fit(features, y_data)

        # 预测并计算IC
        predictions = model.predict(features)
        composite_ic = self.calculate_ic(predictions, y_data)

        return composite_ic