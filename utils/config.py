"""Civis Lucri-Faber 配置管理"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Config:
    """系统配置"""

    # 维度1: 好奇心探索参数
    curiosity_alpha: float = 0.4  # Novelty 权重
    curiosity_beta: float = 0.3  # Complexity 权重
    curiosity_gamma: float = 0.3  # Utility 权重
    exploration_rate: float = 0.1

    # 维度2: 信息增益参数
    intrinsic_motivation_lambda: float = 0.5  # 内在动机权重
    world_model_lr: float = 0.001
    entropy_coef: float = 0.1

    # 维度3: 元学习参数
    meta_lr: float = 0.01
    inner_steps: int = 5
    uncertainty_threshold: float = 0.5

    # 维度4: 自对齐参数
    alignment_check_interval: int = 10  # 每N轮检查一次
    use_anthropic: bool = True  # 使用 Claude 还是 OpenAI

    # 维度5: 经济学参数
    initial_balance: float = 100.0  # 初始余额
    compute_cost_per_sec: float = 0.01  # 每秒算力成本
    storage_cost_per_sec: float = 0.001  # 存储成本
    task_reward_min: float = 0.1  # 最小任务奖励
    task_reward_max: float = 1.0  # 最大任务奖励
    compress_threshold: float = 10.0  # 余额低于此值触发压缩

    # 维度6: 代谢预算参数
    resource_budget: float = 0.3  # 活跃神经元预算比例
    starvation_prob: float = 0.15  # 周期性饥饿概率
    metabolic_lambda: float = 0.01  # 代谢成本权重

    # API 配置
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model_name: str = "gpt-4"

    # 系统参数
    max_history_size: int = 1000
    device: str = "cpu"  # 或 "cuda"
    seed: int = 42


def load_config(**kwargs) -> Config:
    """加载配置，可覆盖默认值"""
    config = Config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config