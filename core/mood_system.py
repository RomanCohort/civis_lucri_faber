# =============================================================================
# Mood System - 心境系统
# =============================================================================
# 长时（小时-天）情绪动力学
#
# 情绪（Emotion）: 秒-分钟级别
# 心境（Mood）: 小时-天级别
#
# 核心机制：
# 1. Ornstein-Uhlenbeck过程：均值回归动力学
# 2. 昼夜节律：皮质醇/褪黑素周期
# 3. 心境障碍建模：抑郁/躁郁
# 4. 事件积分：长期事件对心境的影响
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import math


# =============================================================================
# 心境状态
# =============================================================================

@dataclass
class MoodState:
    """心境状态"""
    valence: float      # 效价 [-1, 1]: 积极-消极
    arousal: float    # 唤醒度 [0, 1]: 平静-激动
    dominance: float  # 支配感 [0, 1]: 无力-掌控

    # 延伸维度
    activation: float  # 激活度 [0, 1]: 抑郁-躁狂
    pleasantness: float  # 愉悦度 [0, 1]: 不快-愉快


class MoodParameters:
    """心境参数"""
    def __init__(
        self,
        mean_valence: float = 0.0,      # 目标效价（基线）
        mean_arousal: float = 0.3,    # 目标唤醒度
        reversion_speed: float = 0.1,   # 均值回归速度
        volatility: float = 0.2,      # 波动性
    ):
        self.mean_valence = mean_valence
        self.mean_arousal = mean_arousal
        self.reversion_speed = reversion_speed
        self.volatility = volatility


# =============================================================================
# Ornstein-Uhlenbeck 动力学
# =============================================================================

class OrnsteinUhlenbeck(nn.Module):
    """
    OU过程：均值回归随机过程

    dX = θ(μ - X)dt + σdW

    其中：
    - μ: 均值（目标心境地）
    - θ: 回归速度
    - σ: 波动性

    对应神经机制：
    - 中缝核血清素系统 → 心境稳定性
    - 前额叶皮层 → 目标导向
    """

    def __init__(
        self,
        dim: int = 5,  # 5个心境维度
        theta: float = 0.1,  # 回归速度
        sigma: float = 0.2,  # 波动性
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.sigma = sigma

        # 均值（可学习）
        self.mean = nn.Parameter(torch.zeros(dim))

        # 维度特定的回归速度
        self.theta_dim = nn.Parameter(torch.ones(dim) * theta)

        # 维度特定的波动性
        self.sigma_dim = nn.Parameter(torch.ones(dim) * sigma)

    def forward(
        self,
        current: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        OU过程前向

        Args:
            current: [B, dim] 当前心境
            dt: 时间步

        Returns:
            next: [B, dim] 下一状态
        """
        # 均值回归项: θ(μ - X)
        reversion = self.theta_dim * (self.mean - current)

        # 随机项: σ * N(0, dt)
        noise = torch.randn_like(current) * self.sigma_dim * math.sqrt(dt)

        # 更新
        next_state = current + reversion * dt + noise

        return next_state

    def sample(
        self,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """从平稳分布采样"""
        # 平稳分布: N(μ, σ²/2θ)
        var = self.sigma_dim ** 2 / (2 * self.theta_dim + 1e-8)
        std = torch.sqrt(var)
        return self.mean + torch.randn(batch_size, self.dim) * std


# =============================================================================
# 昼夜节律系统
# =============================================================================

class CircadianRhythm(nn.Module):
    """
    昼夜节律系统

    对应神经机制：
    - 视交叉上核 (SCN): 生物钟
    - 皮质醇: 觉醒激素（晨间高峰）
    - 褪黑素: 睡眠激素（夜间高峰）
    - 甲状腺激素: 代谢率

    功能：
    - 24小时周期的心境调节
    - 晨间效应（皮质醇峰值 → 警觉性↑）
    - 夜间效应（褪黑素峰值 → 困倦）
    """

    def __init__(
        self,
        period_hours: float = 24.0,
    ):
        super().__init__()
        self.period_hours = period_hours
        self.hours_per_cycle = 2 * math.pi / period_hours

        # 皮质醇节律参数
        self.cortisol_peak_hour = nn.Parameter(torch.tensor(8.0))  # 早8点峰值
        self.cortisol_amplitude = nn.Parameter(torch.tensor(0.3))

        # 褪黑素节律参数
        self.melatonin_peak_hour = nn.Parameter(torch.tensor(2.0))  # 凌晨2点峰值
        self.melatonin_amplitude = nn.Parameter(torch.tensor(0.4))

        # 对心境的影响
        self.cortisol_effect_on_arousal = 0.5
        self.cortisol_effect_on_valence = 0.2
        self.melatonin_effect_on_arousal = -0.4

    def get_cortisol(
        self,
        hour: float,
    ) -> float:
        """获取皮质醇水平"""
        phase = (hour - self.cortisol_peak_hour.item()) * self.hours_per_cycle
        return self.cortisol_amplitude.item() * (math.cos(phase) + 1) / 2

    def get_melatonin(
        self,
        hour: float,
    ) -> float:
        """获取褪黑素水平"""
        phase = (hour - self.melatonin_peak_hour.item()) * self.hours_per_cycle
        return self.melatonin_amplitude.item() * (math.cos(phase) + 1) / 2

    def forward(
        self,
        hour: float,
        external_cortisol: Optional[float] = None,
    ) -> Dict:
        """
        昼夜节律影响

        Args:
            hour: 小时 [0, 24]
            external_cortisol: 外部真实皮质醇水平 (优先于自带昼夜节律)

        Returns:
            rhythm_effects: 节律影响
        """
        # 如果有外部皮质醇输入 (来自HPA轴真实值), 优先使用
        if external_cortisol is not None:
            cortisol = external_cortisol
        else:
            cortisol = self.get_cortisol(hour)
        melatonin = self.get_melatonin(hour)

        # 心境影响
        arousal_effect = (
            cortisol * self.cortisol_effect_on_arousal +
            melatonin * self.melatonin_effect_on_arousal
        )
        valence_effect = cortisol * self.cortisol_effect_on_valence

        return {
            'cortisol': cortisol,
            'melatonin': melatonin,
            'arousal_modulation': arousal_effect,
            'valence_modulation': valence_effect,
            'hour': hour,
        }


# =============================================================================
# 心境障碍系统
# =============================================================================

class MoodDisorderSystem(nn.Module):
    """
    心境障碍系统

    建模：
    1. 重度抑郁障碍 (MDD)
    2. 双相障碍 (BD)
    3. 环性心境障碍 (Cyclothymia)

    特征：
    - 重抑: 低均值 + 低方差
    - 躁狂: 高激活 + 高效价
    - 郁/躁循环: 周期切换
    """

    def __init__(
        self,
        dim: int = 5,
    ):
        super().__init__()
        self.dim = dim

        # 障碍类型参数
        self.mdd_shift = nn.Parameter(torch.tensor([
            -0.5,  # valence down
            -0.3,  # arousal down
            -0.4,  # dominance down
            -0.6,  # activation down
            -0.5,  # pleasantness down
        ]))

        self.mania_shift = nn.Parameter(torch.tensor([
            0.4,   # valence up
            0.8,   # arousal high
            0.6,   # dominance up
            0.9,   # activation high
            0.5,   # pleasantness up
        ]))

        # 切换概率
        self.switch_prob = nn.Parameter(torch.tensor(0.01))

        # 当前障碍状态
        self.register_buffer('current_disorder', None)
        self.register_buffer('episode_duration', torch.tensor(0))

    def get_disorder_effect(
        self,
        disorder_type: str = 'none',
    ) -> torch.Tensor:
        """获取障碍影响"""
        if disorder_type == 'depression':
            return self.mdd_shift
        elif disorder_type == 'mania':
            return self.mania_shift
        else:
            return torch.zeros(self.dim)

    def can_switch(
        self,
        current_disorder: str,
    ) -> bool:
        """判断是否可以切换"""
        if self.episode_duration.item() > 24:  # 至少24小时
            return torch.rand(1).item() < self.switch_prob.item()
        return False


# =============================================================================
# 事件积分系统
# =============================================================================

class EventIntegrator(nn.Module):
    """
    事件积分系统

    将长期事件积分到心境：
    - 正性事件积累 → 心境偏正向
    - 负性事件积累 → 心境偏负向
    - 事件权重：近期事件权重高，遥远事件权重低
    """

    def __init__(
        self,
        event_dim: int = 64,
        mood_dim: int = 5,
        decay_rate: float = 0.95,
    ):
        super().__init__()
        self.event_dim = event_dim
        self.mood_dim = mood_dim
        self.decay_rate = decay_rate

        # 事件编码器
        self.event_encoder = nn.Sequential(
            nn.Linear(event_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 事件→心情影响
        self.event_to_mood = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, mood_dim),
        )

        # 事件历史
        self.event_history = deque(maxlen=100)

    def process_event(
        self,
        event: torch.Tensor,
    ) -> Dict:
        """
        处理事件

        Args:
            event: [B, event_dim] 事件特征

        Returns:
            mood_impact: 心情影响
        """
        # 编码
        event_repr = self.event_encoder(event)

        # 转换为心情影响
        mood_impact = self.event_to_mood(event_repr)

        # 记录
        self.event_history.append({
            'event': event_repr.detach(),
            'impact': mood_impact.detach(),
        })

        return {
            'mood_impact': mood_impact,
            'event_repr': event_repr,
        }

    def get_accumulated_impact(
        self,
    ) -> torch.Tensor:
        """获取累积影响"""
        if not self.event_history:
            return torch.zeros(1, self.mood_dim)

        # 加权求和（近期事件权重大）
        total_impact = 0
        total_weight = 0

        for i, e in enumerate(reversed(self.event_history)):
            weight = self.decay_rate ** i
            total_impact += e['impact'] * weight
            total_weight += weight

        return total_impact / (total_weight + 1e-8)


# =============================================================================
# 完整心境系统
# =============================================================================

class MoodSystem(nn.Module):
    """
    完整心���系���

    整合：
    1. OU动力学（均值回归）
    2. 昼夜节律
    3. 心境障碍
    4. 事件积分
    """

    def __init__(
        self,
        mood_dim: int = 5,
        event_dim: int = 64,
    ):
        super().__init__()
        self.mood_dim = mood_dim

        # OU过程
        self.ou = OrnsteinUhlenbeck(mood_dim, theta=0.1, sigma=0.2)

        # 昼夜节律
        self.circadian = CircadianRhythm()

        # 心境障碍
        self.disorder = MoodDisorderSystem(mood_dim)

        # 事件积分
        self.event_integrator = EventIntegrator(event_dim, mood_dim)

        # 心境状态缓冲区
        self.mood_buffer = deque(maxlen=1000)

        # 当前时间
        self.current_hour = 12.0  # 默认中午
        self.time_elapsed = 0.0

    def forward(
        self,
        emotional_input: Optional[torch.Tensor] = None,
        event: Optional[torch.Tensor] = None,
        hour: Optional[float] = None,
    ) -> Dict:
        """
        心境更新

        Args:
            emotional_input: [B, mood_dim] 情绪输入（从情绪系统来）
            event: [B, event_dim] 事件（可选）
            hour: 小时 [0, 24]

        Returns:
            mood: 当前心境
            mood_state: 心境��态
        """
        # 更新时间
        if hour is not None:
            self.current_hour = hour
        self.time_elapsed += 1.0

        # 1. OU动力学更新
        if len(self.mood_buffer) > 0:
            current_mood = self.mood_buffer[-1]
        else:
            current_mood = torch.zeros(1, self.mood_dim, device=next(self.parameters()).device)

        ou更新 = self.ou(current_mood)

        # 2. 情绪输入
        if emotional_input is not None:
            # 情绪输入通过衰减积分
            emotion_contribution = emotional_input * 0.1
        else:
            emotion_contribution = 0

        # 3. 事件影响
        event_impact = 0
        if event is not None:
            event_out = self.event_integrator.process_event(event)
            event_impact = event_out['mood_impact']

        # 4. 昼夜节律
        circadian_out = self.circadian(self.current_hour, external_cortisol=None)
        circadian_mod = torch.tensor([
            circadian_out['valence_modulation'],
            circadian_out['arousal_modulation'],
            0,
            circadian_out['arousal_modulation'] * 0.5,
            circadian_out['valence_modulation'],
        ], device=ou更新.device)

        # 综合更新
        base_mood = ou更新 + emotion_contribution + event_impact + circadian_mod

        # 限制范围
        base_mood = torch.clamp(base_mood, -1, 1)

        # 记录
        self.mood_buffer.append(base_mood.detach())

        # 转换为VAD状态
        mood_state = MoodState(
            valence=base_mood[0, 0].item(),
            arousal=base_mood[0, 1].item(),
            dominance=base_mood[0, 2].item(),
            activation=base_mood[0, 3].item(),
            pleasantness=base_mood[0, 4].item(),
        )

        return {
            'mood': base_mood,
            'mood_state': mood_state,
            'circadian': circadian_out,
            'event_impact': event_impact,
        }

    def get_mood_summary(self) -> Dict:
        """获取心境摘要"""
        if len(self.mood_buffer) < 10:
            return {'status': 'warming_up'}

        recent = torch.stack(list(self.mood_buffer)[-100:])
        return {
            'mean_valence': recent[:, 0].mean().item(),
            'mean_arousal': recent[:, 1].mean().item(),
            'mean_activation': recent[:, 3].mean().item(),
            'hour': self.current_hour,
            'time_elapsed': self.time_elapsed,
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_mood_system(
    mood_dim: int = 5,
    event_dim: int = 64,
) -> MoodSystem:
    """创建心境系统"""
    return MoodSystem(mood_dim, event_dim)


__all__ = [
    'MoodState',
    'MoodParameters',
    'OrnsteinUhlenbeck',
    'CircadianRhythm',
    'MoodDisorderSystem',
    'EventIntegrator',
    'MoodSystem',
    'create_mood_system',
]


# =============================================================================
# 测试
# =============================================================================

def test_mood_system():
    """测试心境系统"""
    print("=" * 60)
    print("Testing Mood System")
    print("=" * 60)

    # 创建模型
    model = MoodSystem()

    print("\n[1] Testing OU dynamics...")
    for i in range(5):
        out = model()
        print(f"  Step {i}: valence={out['mood'][0,0]:.3f}, arousal={out['mood'][0,1]:.3f}")

    print("\n[2] Testing circadian rhythm at different hours...")
    for h in [6, 12, 18, 2]:
        out = model(hour=h)
        print(f"  Hour {h}: cortisol={out['circadian']['cortisol']:.3f}, melatonin={out['circadian']['melatonin']:.3f}")

    print("\n[3] Testing event integration...")
    event = torch.randn(1, 64)
    out = model(event=event)
    print(f"  Event impact: {out['event_impact'][0]}")

    print("\n[4] Mood summary...")
    summary = model.get_mood_summary()
    print(f"  Mean valence: {summary.get('mean_valence', 0):.3f}")
    print(f"  Mean arousal: {summary.get('mean_arousal', 0):.3f}")

    print("\n" + "=" * 60)
    print("✓ Mood system working!")
    print("=" * 60)


if __name__ == "__main__":
    test_mood_system()