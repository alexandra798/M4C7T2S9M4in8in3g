"""MCTS节点类定义"""
import numpy as np


class MCTSNode:
    """蒙特卡洛树搜索节点"""

    def __init__(self, formula='', parent=None):
        self.formula = formula
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        """检查节点是否已完全展开"""
        return len(self.children) > 0

    def add_child(self, child_node):
        """添加子节点"""
        self.children.append(child_node)

    def update(self, reward):
        """更新节点的访问次数和值"""
        self.visits += 1
        self.value += reward