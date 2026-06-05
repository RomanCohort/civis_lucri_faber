"""
神经调制信号系统 (Neuromodulation Integration)

整合奖赏信号与时间discount，复用现有的Neuromodulation模块：
1. 多巴胺系统 - 预测误差 + TD learning
2. 血清素系统 - 时间discount + 风险感知
3. 乙酰胆碱 - 注意力聚焦
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TemporalDiscountSignal:
    """时间折扣信号"""
    gamma: float = 0.99        # 时间折扣因子
    dopamine: float = 0.5     # 多巴胺信号 (奖励预测误差)
    baseline: float = 0.0      # 移动平均奖励baseline
    td_error: float = 0.0        # TD误差


@dataclass
class NeuromodulationState:
    """神经调制状态"""
    dopamine: float = 0.5       # 奖励/预测误差
    serotonin: float = 0.5     # 时间discount/风险
    acetylcholine: float = 0.5  # 注意力聚焦
    value_estimate: float = 0.0  # 价值估计


class RewardModulation(nn.Module):
    """
    奖赏调制模块

    实现TD学习的多巴胺信号：
    - 正向误差 → "惊喜" → 增强学习
    - 负向误差 → "失望" → 抑制学习
    """

    def __init__(
        self,
        gamma: float = 0.99,          # 时间折扣
        baseline_alpha: float = 0.01, # baseline更新率
        eligibility_trace: float = 0.9,  # 资格迹decay
    ):
        super().__init__()

        self.gamma = gamma
        self.baseline_alpha = baseline_alpha
        self.eligibility_trace = eligibility_trace

        # 价值估计器 (简化)
        self.value_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # 状态
        self.baseline = 0.0  # 平均奖励
        self.value_history = deque(maxlen=100)

    def forward(self, state: torch.Tensor) -> float:
        """预测状态价值"""
        value = self.value_net(state).item()
        return value

    def compute_td_error(
        self,
        reward: float,
        state: torch.Tensor,
        next_state: torch.Tensor | None = None,
    ) -> float:
        """
        计算TD误差

        TD error = r + γV(s') - V(s)
        这就是多巴胺信号！
        """
        # 当前价值
        current_value = self.forward(state)

        # 下一状态价值 (如果没有，给0)
        if next_state is not None:
            next_value = self.forward(next_state)
        else:
            next_value = 0.0

        # TD误差
        td_error = reward + self.gamma * next_value - current_value

        # 更新baseline (Exponential Moving Average)
        self.baseline = self.baseline * (1 - self.baseline_alpha) + reward * self.baseline_alpha

        # 记录
        self.value_history.append(current_value)

        return td_error

    def get_dopamine_signal(self, td_error: float) -> float:
        """
        将TD误差转换为多巴胺信号 (0-1)

        - 正向误差 → dopamine → 1
        - 负向误差 → dopamine → 0
        """
        # 归一化到0-1：tanh压缩后偏移
        dopamine = (np.tanh(td_error) + 1) / 2
        return dopamine

    def get_value_estimate(self, state: torch.Tensor) -> float:
        """获取价值估计"""
        return self.forward(state)


class TemporalDiscount:
    """
    时间折扣模块 (血清素系统)

    对应血清素的核心功能：
    - 未来奖���的折扣
    - 长程规划 vs 短视
    - 风险感知
    """

    def __init__(
        self,
        base_gamma: float = 0.99,      # 基础折扣
        min_gamma: float = 0.5,        # 最小折扣（高血清素）
        uncertainty_boost: float = 0.1, # 不确定性时的boost
        reward_volatility: float = 0.0,  # 奖励波动性
    ):
        self.base_gamma = base_gamma
        self.min_gamma = min_gamma
        self.uncertainty_boost = uncertainty_boost
        self.reward_volatility = reward_volatility

        # 波动性追踪
        self.reward_history = deque(maxlen=50)
        self.gamma_history = deque(maxlen=100)

    def compute_gamma(
        self,
        uncertainty: float = 0.0,
        risk_level: float = 0.0,
    ) -> float:
        """
        动态计算时间折扣

        - 高不确定性/风险 → 低gamma (短视)
        - 低不确定性/风险 → 高gamma (长程)
        """
        # 基础调整
        gamma = self.base_gamma

        # 不确定性boost（不确定性高时，更重视当下）
        uncertainty_effect = uncertainty * self.uncertainty_boost

        # 风险调整
        risk_effect = risk_level * (self.base_gamma - self.min_gamma)

        gamma = gamma - uncertainty_effect - risk_effect
        gamma = np.clip(gamma, self.min_gamma, self.base_gamma)

        # 更新历史
        self.gamma_history.append(gamma)

        return gamma

    def compute_serotonin_signal(
        self,
        reward_std: float = 0.0,
        time_horizon: int = 0,
    ) -> float:
        """
        计算血清素信号 (0-1)

        基于：
        - 奖励波动性
        - 时间视野
        """
        # 波动性高 → 血清素高 → 保守
        volatility_effect = np.tanh(self.reward_volatility * 2)

        # 时间视野短 → 血清素高
        horizon_effect = 1.0 / (1 + time_horizon * 0.1)

        serotonin = (volatility_effect * 0.5 + horizon_effect * 0.5)
        serotonin = np.clip(serotonin, 0, 1)

        return serotonin

    def update_volatility(self, reward: float):
        """更新奖励波动性估计"""
        self.reward_history.append(reward)
        if len(self.reward_history) >= 10:
            self.reward_volatility = np.std(list(self.reward_history))

    def get_discounted_return(
        self,
        rewards: list,
        gamma: float = None,
    ) -> float:
        """
        计算折扣返回

        G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
        """
        gamma = gamma or self.base_gamma
        discounted_return = 0.0

        for i, r in enumerate(rewards):
            discounted_return += (gamma ** i) * r

        return discounted_return


class AttentionModulator(nn.Module):
    """
    注意力调制 (乙酰胆碱系统)

    对应乙酰胆碱的核心功能：
    - 注意力聚焦
    - 工作记忆增强
    - 新奇检测
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        focus_baseline: float = 0.5,
    ):
        super().__init__()

        self.focus_baseline = focus_baseline

        # 新奇检测器
        self.novelty_detector = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 注意力强度
        self.focus_strength = focus_baseline
        self.focus_history = deque(maxlen=100)

    def compute_focus(
        self,
        hidden_states: torch.Tensor,
        novelty: float = 0.0,
    ) -> float:
        """
        计算注意力聚焦

        新奇度高 → 聚焦强
        """
        if hidden_states.dim() > 2:
            hidden_states = hidden_states.mean(dim=1)

        # 新奇检测
        novelty_score = self.novelty_detector(hidden_states).item()

        # 聚焦强度
        self.focus_strength = (
            self.focus_baseline * 0.5 +
            novelty * 0.3 +
            novelty_score * 0.2
        )
        self.focus_strength = np.clip(self.focus_strength, 0.1, 1.0)

        self.focus_history.append(self.focus_strength)

        return self.focus_strength

    def apply_gating(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用注意力门控

        高聚焦 → 保留更多细节
        低聚焦 → 稀疏化
        """
        if self.focus_strength < 0.5:
            # 稀疏化
            k = int(hidden_states.numel() * self.focus_strength)
            values, _ = hidden_states.abs().view(-1).topk(k)
            threshold = values.min() if k > 0 else 0
            mask = (hidden_states.abs() >= threshold).float()
            return hidden_states * mask
        else:
            return hidden_states


class NeuromodulationIntegration(nn.Module):
    """
    完整的神经调制整合系统

    复用并扩展现有的Neuromodulation模块：
    1. 多巴胺 - TD learning + 奖赏预测误差
    2. 血清素 - 时间discount + 风险
    3. 乙酰胆碱 - 注意力聚焦
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        baseline_alpha: float = 0.01,
    ):
        super().__init__()

        # 多巴胺系统
        self.dopamine = RewardModulation(
            gamma=gamma,
            baseline_alpha=baseline_alpha,
        )

        # 血清素系统
        self.serotonin = TemporalDiscount(base_gamma=gamma)

        # 乙酰胆碱系统
        self.acetylcholine = AttentionModulator(hidden_dim)

        # 状态
        self.current_state = NeuromodulationState()

    def step(
        self,
        state: np.ndarray,
        action: str,
        reward: float,
        next_state: np.ndarray = None,
        uncertainty: float = 0.0,
        novelty: float = 0.0,
    ) -> dict:
        """
        单步更新

        Args:
            state: 当前状态 (numpy)
            action: 动作
            reward: 即时奖励
            next_state: 下一状态
            uncertainty: 不确定性 (0-1)
            novelty: 新奇度 (0-1)

        Returns:
            modulation: 调制信号
        """
        # 转换为tensor
        state_t = torch.tensor(state, dtype=torch.float32)
        next_state_t = torch.tensor(next_state, dtype=torch.float32) if next_state is not None else None

        # 1. 多巴胺信号 (TD误差)
        td_error = self.dopamine.compute_td_error(reward, state_t, next_state_t)
        dopamine = self.dopamine.get_dopamine_signal(td_error)

        # 2. 血清素信号 (时间discount)
        gamma = self.serotonin.compute_gamma(uncertainty=uncertainty)
        serotonin = self.serotonin.compute_serotonin_signal(
            reward_std=self.serotonin.reward_volatility,
            time_horizon=0,
        )

        # 3. 乙酰胆碱信号 (注意力聚焦)
        hidden = state_t.unsqueeze(0)
        acetylcholine = self.acetylcholine.compute_focus(hidden, novelty)

        # 更新波动性
        self.serotonin.update_volatility(reward)

        # 保存状态
        self.current_state = NeuromodulationState(
            dopamine=dopamine,
            serotonin=serotonin,
            acetylcholine=acetylcholine,
            value_estimate=self.dopamine.get_value_estimate(state_t),
        )

        return {
            'dopamine': dopamine,
            'serotonin': serotonin,
            'acetylcholine': acetylcholine,
            'gamma': gamma,
            'td_error': td_error,
            'value': self.current_state.value_estimate,
        }

    def get_increased_reward(
        self,
        base_reward: float,
        consolidation_bonus: float = 0.0,
    ) -> float:
        """
        计算增强后的奖励

        多巴胺信号增强学习信号：
        - 高dopamine → 高有效奖励 → 强化学习
        - 低dopamine → 低有效奖励 → 抑制学习
        """
        dopamine = self.current_state.dopamine
        enhanced = base_reward * (1 + dopamine * 0.5)

        # 加入记忆巩固bonus (来自睡眠系统)
        if consolidation_bonus > 0:
            enhanced += consolidation_bonus * 0.3

        return enhanced

    def get_learning_rate_adjustment(
        self,
        base_lr: float = 0.001,
    ) -> float:
        """
        获取调整后的学习率

        - 高dopamine → 高学习率 (兴奋/学习)
        - 高serotonin → 低学习率 (保守)
        """
        dopamine = self.current_state.dopamine
        serotonin = self.current_state.serotonin

        # 高多巴胺 + 低血清素 → 高学习率
        lr_adjustment = dopamine * 2 - serotonin * 0.5
        adjusted_lr = base_lr * (1 + lr_adjustment * 0.5)

        return max(0.0001, min(base_lr * 2, adjusted_lr))

    def get_summary(self) -> dict:
        """获取摘要"""
        return {
            'dopamine': self.current_state.dopamine,
            'serotonin': self.current_state.serotonin,
            'acetylcholine': self.current_state.acetylcholine,
            'value_estimate': self.current_state.value_estimate,
            'gamma': self.serotonin.base_gamma,
            'avg_focus': np.mean(list(self.acetylcholine.focus_history)) if self.acetylcholine.focus_history else 0,
        }


# ============ 便捷函数 ============

def create_neuromodulation_integration(
    hidden_dim: int = 128,
    gamma: float = 0.99,
) -> NeuromodulationIntegration:
    """创建神经调制整合系统"""
    return NeuromodulationIntegration(hidden_dim, gamma)


__all__ = [
    'TemporalDiscountSignal',
    'NeuromodulationState',
    'RewardModulation',
    'TemporalDiscount',
    'AttentionModulator',
    'NeuromodulationIntegration',
    'create_neuromodulation_integration',
]
