"""配置文件"""

# MCTS参数
MCTS_CONFIG = {
    "num_iterations": 1000,
    "exploration_param": 1.41,
    "risk_seeking_exploration": 2.0,
    "quantile_threshold": 0.9
}

# Alpha池参数
ALPHA_POOL_CONFIG = {
    "pool_size": 100,
    "lambda_param": 0.5
}

# 交叉验证参数
CV_CONFIG = {
    "n_splits": 8
}

# 数据路径
DATA_CONFIG = {
    "default_data_path": "/path/to/data.csv",
    "target_column": "label_shifted"
}