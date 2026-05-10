"""
神经调质系统 (Neuromodulation)

对应生物学的神经调质（多巴胺、血清素）：
- 多巴胺 = 预测误差信号
- 血清素 = 不确定性/风险感知

功能：
1. 全局信心门控 (Global Confidence Gating)
2. 价值方差预测
3. 不确定性检测 → 温度调整
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ModulationSignal:
    """神经调质信号"""
    dopamine: float = 0.5      # 0-1, 预测误差/奖励信号
    serotonin: float = 0.5    # 0-1, 不确定性/风险信号
    acetylcholine: float = 0.5   # 0-1, 注意力聚焦度


class UncertaintyDetector(nn.Module):
    """
    不确定性检测器

    检测当前输出的不确定性程度
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor) -> float:
        """返回不确定性 (0-1)"""
        # 对hidden states做pooling
        pooled = hidden_states.mean(dim=1) if hidden_states.dim() == 3 else hidden_states
        uncertainty = self.net(pooled).item()
        return uncertainty


class RewardPredictionError(nn.Module):
    """
    奖励预测误差 (Reward Prediction Error)

    对应多巴胺的核心功能：
    - 预测价值 vs 实际价值的差异
    - 正向误差 → 增强学习
    - 负向误差 → 抑制学习
    """

    def __init__(self, value_dim: int = 64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(value_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # -1 到 1
        )
        self.baseline = 0.0

    def forward(self, state: torch.Tensor) -> float:
        """预测价值"""
        return self.predictor(state).item()

    def compute_prediction_error(
        self,
        predicted: float,
        actual: float
    ) -> float:
        """计算预测误差"""
        return actual - predicted

    def update_baseline(self, actual_reward: float, alpha: float = 0.01):
        """更新baseline (Exponential Moving Average)"""
        self.baseline = self.baseline * (1 - alpha) + actual_reward * alpha


class DopamineGate(nn.Module):
    """
    多巴胺门控 (Dopamine Head)

    旁路网络，预测当前输出的"价值方差"
    不预测下一个词，而是预测"我对自己输出的信心"
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 768,
        n_heads: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 价值预测头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 0-1, "我有多大把握"
        )

        # 方差预测头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[float, float]:
        """
        返回:
        - confidence: 信心 (0-1)
        - uncertainty: 不确定性 (0-1)
        """
        # 处理不同形状
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.mean(dim=1)
        elif hidden_states.dim() == 2:
            pass
        else:
            hidden_states = hidden_states.unsqueeze(0)

        confidence = self.value_head(hidden_states).item()
        uncertainty = self.uncertainty_head(hidden_states).item()

        return confidence, uncertainty


class SerotoninGate(nn.Module):
    """
    血清素门控

    负责风险感知和不确定性调节
    高血清素 → 高风险规避 → 保守输出
    """

    def __init__(self, baseline: float = 0.5):
        super().__init__()
        self.baseline = baseline
        self.serotonin_level = baseline
        self.history = []

    def compute_risk_adjustment(
        self,
        uncertainty: float,
        task_type: str = "general"
    ) -> float:
        """
        根据不确定性计算风险调整

        返回温度调整因子 (0.5-2.0)
        - 高不确定性 → 升温 (更保守)
        - 低不确定性 → 降温 (更自信)
        """
        # 任务类型影响
        if task_type == "safety" or task_type == "moral":
            # 道德/安全相关 → 极其保守
            risk_multiplier = 2.0
        elif task_type == "creative":
            # 创意任务 → 可以冒险
            risk_multiplier = 0.5
        else:
            risk_multiplier = 1.0

        # 基础调整
        base_adjustment = 1.0 + (uncertainty - 0.5) * risk_multiplier

        # 血清素水平调节
        self.serotonin_level = self.baseline * 0.7 + uncertainty * 0.3

        return np.clip(base_adjustment, 0.3, 3.0)


class TemperatureController(nn.Module):
    """
    温度控制器

    全局调节Softmax温度
    模拟神经调质对神经网络增益的调节
    """

    def __init__(
        self,
        base_temperature: float = 1.0,
        min_temp: float = 0.3,
        max_temp: float = 2.0
    ):
        super().__init__()
        self.base_temp = base_temperature
        self.min_temp = min_temp
        self.max_temp = max_temp

        # 当前温度
        self.current_temp = base_temperature

        # 历史
        self.temp_history = []

    def compute(
        self,
        dopamine: float,
        serotonin: float,
        task_type: str = "general"
    ) -> float:
        """
        计算最终温度

        多巴胺 ↑ → 温度 ↓ (自信)
        血清素 ↑ → 温度 ↑ (保守)
        """
        # 多巴胺效应：奖励预测误差为正时，增加信心（降低温度）
        dopamine_effect = self.base_temp * (1.2 - dopamine)

        # 血清素效应：高不确定性时，增加保守（提高温度）
        serotonin_effect = self.base_temp * (0.8 + serotonin * 0.8)

        # 任务调节
        if task_type == "moral":
            final_temp = max(dopamine_effect, serotonin_effect, self.max_temp)
        elif task_type == "creative":
            final_temp = min(dopamine_effect, serotonin_effect, self.min_temp)
        else:
            final_temp = (dopamine_effect + serotonin_effect) / 2

        # 限制范围
        final_temp = np.clip(final_temp, self.min_temp, self.max_temp)

        self.current_temp = final_temp
        self.temp_history.append(final_temp)

        return final_temp

    def apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """应用温度到logits"""
        return logits / self.current_temp


class NeuromodulationSystem(nn.Module):
    """
    完整的神经调质系统

    整合：
    1. 多巴胺门控 (价��预测)
    2. 血清素门控 (风险感知)
    3. 温度控制器 (全局调节)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        vocab_size: int = 32000,
    ):
        super().__init__()

        # 多巴胺系统
        self.dopamine = DopamineGate(vocab_size, hidden_dim)

        # 血清素系统
        self.serotonin = SerotoninGate()

        # 温度控制
        self.temperature = TemperatureController()

        # 调质信号历史
        self.signal_history = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        task_type: str = "general",
        predicted_value: float = None,
        actual_value: float = None,
    ) -> Dict:
        """
        处理神经调质

        Returns:
        - temperature: 调整后的温度
        - confidence: 信心水平
        - uncertainty: 不确定性
        - modulation: 调质信号
        """
        # 1. 多巴胺：价值/信心预测
        confidence, uncertainty = self.dopamine(hidden_states)

        # 2. 奖励预测误差
        if predicted_value is not None and actual_value is not None:
            dopamine_signal = (actual_value - predicted_value + 1) / 2  # 归一化到0-1
        else:
            dopamine_signal = 0.5

        # 3. 血清素：风险感知
        serotonin_signal = uncertainty

        # 4. 计算温度调整
        temperature = self.temperature.compute(
            dopamine_signal,
            serotonin_signal,
            task_type
        )

        # 记录
        signal = ModulationSignal(
            dopamine=dopamine_signal,
            serotonin=serotonin_signal,
            acetylcholine=confidence
        )
        self.signal_history.append(signal)

        return {
            'temperature': temperature,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'dopamine': dopamine_signal,
            'serotonin': serotonin_signal,
            'modulation': signal,
        }

    def apply_to_logits(self, logits: torch.Tensor, task_type: str = "general") -> torch.Tensor:
        """将调质应用到logits"""
        # 对于已存在的logits（不调用forward），直接用当前温度
        temperature = self.temperature.current_temp
        return logits / temperature

    def get_summary(self) -> Dict:
        """获取摘要"""
        if not self.signal_history:
            return {'dopamine': 0.5, 'serotonin': 0.5, 'temperature': 1.0}

        recent = self.signal_history[-10:]
        return {
            'dopamine': np.mean([s.dopamine for s in recent]),
            'serotonin': np.mean([s.serotonin for s in recent]),
            'acetylcholine': np.mean([s.acetylcholine for s in recent]),
            'temperature': self.temperature.current_temp,
        }


# ============ 便捷函数 ============

def create_neuromodulation(
    hidden_dim: int = 768,
    vocab_size: int = 32000,
) -> NeuromodulationSystem:
    """创建神经调质系统"""
    return NeuromodulationSystem(hidden_dim, vocab_size)


__all__ = [
    "ModulationSignal",
    "UncertaintyDetector",
    "RewardPredictionError",
    "DopamineGate",
    "SerotoninGate",
    "TemperatureController",
    "NeuromodulationSystem",
    "create_neuromodulation",
]