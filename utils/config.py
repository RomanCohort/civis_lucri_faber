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
    # LLM 多通道
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_model: str = "deepseek-chat"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    llm_provider: str = "auto"  # "openai" / "anthropic" / "deepseek" / "ollama" / "auto"
    # 对话参数
    chat_temperature: float = 0.7
    chat_max_tokens: int = 2048
    tool_max_rounds: int = 5

    # 维度8: 神经修剪参数
    prune_threshold: float = 0.15       # 权重低于此值触发硬剪枝
    prune_decay_rate: float = 0.002     # 基础权重衰减率
    growth_factor_baseline: float = 0.5 # 生长因子基线浓度
    neurogenesis_enabled: bool = True   # 是否启用神经发生

    # 维度9: 神经自调节参数
    # ANS (自主神经)
    ans_sympathetic_reactivity: float = 1.0   # 交感神经反应性
    ans_baseline_vagal_tone: float = 0.5      # 基础迷走神经张力
    ans_baroreceptor_setpoint: float = 0.5     # 压力感受器设定点
    # HPA轴
    hpa_stress_reactivity: float = 1.0        # 应激反应性
    hpa_cortisol_half_life_steps: int = 60     # 皮质醇半衰期 (步数)
    hpa_feedback_strength: float = 0.6         # 负反馈强度
    hpa_load_accumulation_rate: float = 0.002  # 稳态负荷累积率
    # 胶质系统
    glial_pruning_rate: float = 0.05           # 基础突触修剪率
    # 稳态调节
    allostatic_overload_threshold: float = 0.8  # 稳态超载阈值
    allostatic_load_recovery_rate: float = 0.005  # 稳态恢复率
    # 预测编码
    predictive_coding_layers: int = 3           # 预测编码层级数
    predictive_coding_lr: float = 0.01          # 生成模型学习率

    # 事件驱动参数
    event_log_enabled: bool = False   # 事件日志开关
    event_bus_debug: bool = False     # 调试模式

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