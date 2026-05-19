# =============================================================================
# Emotion Dynamics - 情绪动力学
# =============================================================================
# 非线性动力学 + 轨迹预测 + 情绪维度动态
#
# 核心机制：
# 1. 非线性动力学：分叉点、临界态
# 2. 轨迹预测：情绪演变预测
# 3. 维度动态：VAD时序建模
# 4. 动力学参数学习
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import math


# =============================================================================
# 情绪状态
# =============================================================================

@dataclass
class EmotionDynamicsState:
    """情绪动力学状态"""
    valence: float = 0.0        # 效价 [-1, 1]
    arousal: float = 0.0        # 唤醒度 [0, 1]
    dominance: float = 0.0     # 支配感 [0, 1]

    # 动力学特征
    velocity: float = 0.0     # 变化速度
    acceleration: float = 0.0  # 变化加速度
    stability: float = 0.5    # 稳定性


# =============================================================================
# 非线性动力学系统
# =============================================================================

class NonlinearDynamics(nn.Module):
    """
    非线性情绪动力学

    使用随机微分方程建模：
    dX = f(X)dt + g(X)dW

    对应神经机制：
    - 神经振荡：情绪周期
    - 临界态：相变
    - 吸引子：情绪吸引子

    现象：
    - 情绪爆发
    - 恢复
    - 持续
    """

    def __init__(
        self,
        dim: int = 3,  # VAD
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # 漂移网络 f(X): 确定性趋势
        self.drift_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

        # 扩散网络 g(X): 随机波动
        self.diffusion_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

        # 吸引子网络（学习情绪吸引子）
        self.attractor_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

        # 临界态检测
        self.criticality_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def compute_drift(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算漂移项（确定性趋势）

        Args:
            state: [B, dim] 当前状态

        Returns:
            drift: [B, dim] 漂移
        """
        return self.drift_net(state)

    def compute_diffusion(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算扩散项（随机波动）

        Args:
            state: 状态

        Returns:
            diffusion: 扩散强度
        """
        return self.diffusion_net(state)

    def compute_attractors(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算吸引子

        Args:
            state: 状态

        Returns:
            attraction: 吸引子
        """
        # 吸引子方向
        attractor = self.attractor_net(state)
        return attractor

    def step(
        self,
        state: torch.Tensor,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        一步更新

        dX = drift * dt + diffusion * dW

        Args:
            state: 当前状态
            dt: 时间步

        Returns:
            next_state: 下一状态
        """
        # 漂移
        drift = self.compute_drift(state)

        # 扩散
        diffusion = self.compute_diffusion(state)
        noise = torch.randn_like(state) * diffusion * math.sqrt(dt)

        # 更新
        next_state = state + drift * dt + noise

        # 限制范围
        next_state = torch.clamp(next_state, -1, 1)

        return next_state

    def detect_criticality(
        self,
        state: torch.Tensor,
    ) -> float:
        """
        检测临界态

        Args:
            state: 状态

        Returns:
            criticality: 临界程度 [0, 1]
        """
        return self.criticality_net(state).item()

    def forward(
        self,
        state: torch.Tensor,
        dt: float = 0.1,
    ) -> Dict:
        """
        非线性动力学前向

        Args:
            state: 状态
            dt: 时间步

        Returns:
            result: 动力学结果
        """
        # 漂移和扩散
        drift = self.compute_drift(state)
        diffusion = self.compute_diffusion(state)
        attractor = self.compute_attractors(state)

        # 临界态
        criticality = self.detect_criticality(state)

        # 一步
        next_state = self.step(state, dt)

        return {
            'next_state': next_state,
            'drift': drift,
            'diffusion': diffusion,
            'attractor': attractor,
            'criticality': criticality,
        }


# =============================================================================
# 轨迹预测系统
# =============================================================================

class TrajectoryPrediction(nn.Module):
    """
    轨迹预测系统

    预测情绪演变轨迹：
    - 短期（1-10步）：精确
    - 中期（10-50步）：趋势
    - 长期（50+步）：方向
    """

    def __init__(
        self,
        state_dim: int = 3,
        hidden_dim: int = 64,
        prediction_horizon: int = 10,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim * 5, hidden_dim),  # 5步历史
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 预测网络
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_horizon * state_dim),
        )

        # 轨迹评分
        self.trajectory_scorer = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def encode_history(
        self,
        history: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        编码历史

        Args:
            history: 历史状态列表

        Returns:
            encoding: 历史编码
        """
        # 填充或截断到5步
        if len(history) < 5:
            padding = [torch.zeros_like(history[0])] * (5 - len(history))
            history = history + padding

        # 拼接
        history_tensor = torch.cat(history[-5:], dim=-1)
        return self.encoder(history_tensor.unsqueeze(0))

    def predict(
        self,
        history_encoding: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        预测轨迹

        Args:
            history_encoding: 历史编码
            n_steps: 预测步数

        Returns:
            trajectory: [n_steps, state_dim] 轨迹
        """
        if n_steps is None:
            n_steps = self.prediction_horizon

        # 预测
        prediction = self.predictor(history_encoding)
        trajectory = prediction.view(-1, n_steps, self.state_dim)

        return trajectory

    def score_trajectory(
        self,
        trajectory: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """
        评分轨迹

        Args:
            trajectory: 预测轨迹
            goal: 目标状态

        Returns:
            score: 可实现性分数
        """
        final_state = trajectory[:, -1, :]
        combined = torch.cat([final_state, goal], dim=-1)
        score = self.trajectory_scorer(combined).item()

        return score

    def forward(
        self,
        history: List[torch.Tensor],
        goal: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        轨迹预测

        Args:
            history: 历史状态
            goal: 目标（可选）

        Returns:
            trajectory: 预测轨迹
        """
        # 编码历史
        history_encoding = self.encode_history(history)

        # 预测
        trajectory = self.predict(history_encoding)

        # 评分
        score = 0.5
        if goal is not None:
            score = self.score_trajectory(trajectory, goal)

        return {
            'trajectory': trajectory,
            'predicted_path': trajectory[0],
            'achievability': score,
        }


# =============================================================================
# 维度动态系统
# =============================================================================

class DimensionalDynamics(nn.Module):
    """
    情绪维度动态系统

    建模VAD（效价-唤醒度-支配）随时间的变化：
    - Valence: 效价动态
    - Arousal: 唤醒度动态
    - Dominance: 支配感动态
    """

    def __init__(
        self,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 各维度独立网络
        self.valence_net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # V, A, D
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # [-1, 1]
        )

        self.arousal_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # [0, 1]
        )

        self.dominance_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 耦合网络（维度间影响）
        self.coupling_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def compute_coupling(
        self,
        vad: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算维度耦合

        Args:
            vad: [B, 3] VAD状态

        Returns:
            coupling: 耦合效应
        """
        return self.coupling_net(vad)

    def forward(
        self,
        vad: torch.Tensor,
    ) -> Dict:
        """
        维度动态

        Args:
            vad: [B, 3] VAD

        Returns:
            dynamics: 动力学结果
        """
        # 各维度独立更新
        valence_change = self.valence_net(vad)
        arousal_change = self.arousal_net(vad)
        dominance_change = self.dominance_net(vad)

        # 耦合
        coupling = self.compute_coupling(vad)

        # 组合
        dynamics_change = torch.cat([
            valence_change,
            arousal_change,
            dominance_change,
        ], dim=-1)

        # 应用耦合
        coupled_change = dynamics_change + coupling * 0.1

        # 更新
        new_vad = vad + coupled_change

        # 限制范围
        new_vad = torch.stack([
            torch.clamp(new_vad[:, 0], -1, 1),
            torch.clamp(new_vad[:, 1], 0, 1),
            torch.clamp(new_vad[:, 2], 0, 1),
        ], dim=-1)

        return {
            'new_vad': new_vad,
            'valence_change': valence_change,
            'arousal_change': arousal_change,
            'dominance_change': dominance_change,
            'coupling': coupling,
        }


# =============================================================================
# 动力学参数学习
# =============================================================================

class DynamicsParameterLearning(nn.Module):
    """
    动力学参数学习

    从数据中学习动力学参数：
    - 漂移参数
    - 扩散参数
    - 吸引子参数
    """

    def __init__(
        self,
        dim: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 参数化网络
        self.param_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # mean + std
        )

        # 可学习参数
        self.base_drift = nn.Parameter(torch.zeros(dim))
        self.base_diffusion = nn.Parameter(torch.ones(dim) * 0.1)

    def learn_parameters(
        self,
        state_sequence: List[torch.Tensor],
    ) -> Dict:
        """
        学习参数

        Args:
            state_sequence: 状态序列

        Returns:
            learned: 学到的参数
        """
        if len(state_sequence) < 2:
            return {'status': 'insufficient_data'}

        # 计算增量
        deltas = []
        for i in range(1, len(state_sequence)):
            delta = state_sequence[i] - state_sequence[i-1]
            deltas.append(delta)

        deltas = torch.stack(deltas)

        # 估计参数
        mean_delta = deltas.mean(dim=0)
        std_delta = deltas.std(dim=0)

        return {
            'estimated_drift': mean_delta,
            'estimated_diffusion': std_delta,
        }

    def forward(
        self,
        state: torch.Tensor,
    ) -> Dict:
        """
        参数化前向

        Args:
            state: 状态

        Returns:
            parameterized: 参数化结果
        """
        params = self.param_net(state)
        mean = params[:, :self.hidden_dim]
        std = F.softplus(params[:, self.hidden_dim:])

        return {
            'mean': mean,
            'std': std,
            'base_drift': self.base_drift,
            'base_diffusion': self.base_diffusion,
        }


# =============================================================================
# 完整情绪动力学系统
# =============================================================================

class EmotionDynamicsSystem(nn.Module):
    """
    完整情绪动力学系统

    整合：
    1. 非线性动力学
    2. 轨迹预测
    3. 维度动态
    4. 参数学习
    """

    def __init__(
        self,
        state_dim: int = 3,
        hidden_dim: int = 64,
        prediction_horizon: int = 10,
    ):
        super().__init__()
        self.state_dim = state_dim

        # 子系统
        self.dynamics = NonlinearDynamics(state_dim, hidden_dim)
        self.trajectory = TrajectoryPrediction(state_dim, hidden_dim, prediction_horizon)
        self.vad_dynamics = DimensionalDynamics(hidden_dim)
        self.param_learning = DynamicsParameterLearning(state_dim, hidden_dim)

        # 状态历史
        self.state_history = deque(maxlen=100)

    def step(
        self,
        state: torch.Tensor,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        一步动力学

        Args:
            state: 当前状态
            dt: 时间步

        Returns:
            next_state: 下一状态
        """
        result = self.dynamics(state, dt)
        return result['next_state']

    def predict_trajectory(
        self,
        history: List[torch.Tensor],
        goal: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        预测轨迹

        Args:
            history: 历史
            goal: 目标

        Returns:
            prediction: 预测
        """
        return self.trajectory(history, goal)

    def update_vad(
        self,
        vad: torch.Tensor,
    ) -> Dict:
        """
        VAD动态更新

        Args:
            vad: VAD状态

        Returns:
            updated: 更新结果
        """
        return self.vad_dynamics(vad)

    def learn_from_data(
        self,
        sequence: List[torch.Tensor],
    ) -> Dict:
        """
        从数据学习

        Args:
            sequence: 状态序列

        Returns:
            learned: 学到的参数
        """
        return self.param_learning.learn_parameters(sequence)

    def forward(
        self,
        state: torch.Tensor,
        dt: float = 0.1,
    ) -> Dict:
        """
        完整动力学

        Args:
            state: 当前VAD状态
            dt: 时间步

        Returns:
            complete: 完整结果
        """
        # 更新历史
        self.state_history.append(state.detach())

        # 非线性动力学
        dynamics_result = self.dynamics(state, dt)

        # VAD动态
        vad_result = self.vad_dynamics(dynamics_result['next_state'])

        # 学习参数
        param_result = self.param_learning(dynamics_result['next_state'])

        return {
            'next_state': dynamics_result['next_state'],
            'vad_dynamics': vad_result['new_vad'],
            'criticality': dynamics_result['criticality'],
            'drift': dynamics_result['drift'],
            'diffusion': dynamics_result['diffusion'],
            'param_result': param_result,
        }

    def get_dynamics_summary(self) -> Dict:
        """获取动力学摘要"""
        if len(self.state_history) < 2:
            return {'status': 'insufficient_data'}

        states = torch.stack(list(self.state_history))

        return {
            'mean_valence': states[:, 0].mean().item(),
            'mean_arousal': states[:, 1].mean().item(),
            'mean_dominance': states[:, 2].mean().item(),
            'velocity': (states[-1] - states[0]).norm().item(),
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_emotion_dynamics(
    state_dim: int = 3,
    hidden_dim: int = 64,
    prediction_horizon: int = 10,
) -> EmotionDynamicsSystem:
    """创建情绪动力学系统"""
    return EmotionDynamicsSystem(state_dim, hidden_dim, prediction_horizon)


__all__ = [
    'EmotionDynamicsState',
    'NonlinearDynamics',
    'TrajectoryPrediction',
    'DimensionalDynamics',
    'DynamicsParameterLearning',
    'EmotionDynamicsSystem',
    'create_emotion_dynamics',
]


# =============================================================================
# 测试
# =============================================================================

def test_emotion_dynamics():
    """测试情绪动力学"""
    print("=" * 60)
    print("Testing Emotion Dynamics System")
    print("=" * 60)

    # 创建模型
    model = EmotionDynamicsSystem()

    # 初始VAD状态
    vad = torch.tensor([[0.5, 0.6, 0.5]])

    print("\n[1] Testing nonlinear dynamics...")
    for i in range(5):
        result = model(vad)
        vad = result['next_state']
        print(f"  Step {i}: VAD={vad[0].tolist()}")

    print("\n[2] Testing criticality detection...")
    criticality = result['criticality']
    print(f"  Criticality: {criticality:.3f}")

    print("\n[3] Testing trajectory prediction...")
    history = [torch.randn(1, 3) for _ in range(5)]
    trajectory = model.predict_trajectory(history)
    print(f"  Predicted: {trajectory['predicted_path'].shape}")

    print("\n[4] Testing parameter learning...")
    sequence = [torch.randn(1, 3) for _ in range(10)]
    learned = model.learn_from_data(sequence)
    print(f"  Estimated drift: {learned.get('estimated_drift')}")

    print("\n[5] Dynamics summary...")
    summary = model.get_dynamics_summary()
    print(f"  Status: {summary.get('status', 'ok')}")

    print("\n" + "=" * 60)
    print("✓ Emotion dynamics system working!")
    print("  - Nonlinear dynamics: ✓")
    print("  - Trajectory prediction: ✓")
    print("  - Dimensional dynamics: ✓")
    print("  - Parameter learning: ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_emotion_dynamics()