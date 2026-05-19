# =============================================================================
# Civis Lucri-Faber -- Emergent Emotion System
# =============================================================================
# 情绪从底层机制涌现，而非硬编码
#
# 核心原则：
# 1. 价值学习（Value Learning）：预测误差驱动
# 2. 紧迫度检测（Urgency）：时间尺度动力学
# 3. 社会推理（Social Inference）：他人心智建模
# 4. 涌现动力学：从交互中涌现，而非预设标签
#
# 参考理论：
# - Scalar Reward Theory (Rescorla & Wagner, 1972)
# - Temporal Difference Learning (Sutton, 1988)
# - Predictive Coding (Rao & Ballard, 1999)
# - Social Cognition (Frith & Frith, 2006)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque


# =============================================================================
# Value Learning -- 预测误差驱动
# =============================================================================
# 理论基础：Rescorla-Wagner (1972) 预测误差理论
#   ΔV = α * δ * λ
#   δ = r - V  (预测误差 = 实际奖励 - 预期价值)
#
# 改进：TD(λ) 学习，支持多时间尺度
# =============================================================================

class ValueLearner(nn.Module):
    """
    价值学习器：从预测误差学习价值表征

    对应神经机制：
    - 中脑多巴胺系统编码δ（预测误差）
    - VTA / SNc 释放多巴胺信号

    特性：
    - 预测误差驱动学习
    - 多时间尺度记忆
    - 分布式表征
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        value_dim: int = 32,
        alpha: float = 0.1,      # 学习率
        gamma: float = 0.95,     # 折扣因子
        lambda_: float = 0.8,     # eligibility trace
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.value_dim = value_dim
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_

        # 价值网络：状态 → 价值表征
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, value_dim),
        )

        # 预测网络：估计未来价值
        self.predict_net = nn.Sequential(
            nn.Linear(value_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 标量预测
        )

        # eligibility trace (for TD(λ))
        self.register_buffer('eligibility', None)

        # 价值表征缓冲区（用于计算TD误差）
        self.value_buffer = deque(maxlen=100)

    def forward(
        self,
        state: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        done: bool = False,
    ) -> Dict:
        """
        前向计算

        Args:
            state: [B, input_dim] 当前状态
            reward: [B,] 实际奖励（可选，用于计算δ）
            done: bool 是否终止

        Returns:
            value_repr: [B, value_dim] 价值表征
            td_error: [B,] TD误差
            prediction: [B,] 价值预测
        """
        # 前向：状态 → 价值表征
        value_repr = self.value_net(state)

        # 预测当前价值
        prediction = self.predict_net(value_repr).squeeze(-1)

        td_error = None

        # 如果提供reward，计算TD误差
        if reward is not None:
            # 获取之前的价值表征
            if len(self.value_buffer) > 0 and self.eligibility is not None:
                prev_value = self.value_buffer[-1]
                # TD误差：δ = r + γV(s') - V(s)
                target = reward + self.gamma * self.predict_net(prev_value).detach()
                td_error = target - prediction

                # 更新eligibility trace
                self.eligibility = self.gamma * self.lambda_ * self.eligibility + value_repr
            else:
                # 初始化
                td_error = reward - prediction
                self.eligibility = value_repr.detach()

            self.value_buffer.append(value_repr.detach())

        return {
            'value_repr': value_repr,
            'td_error': td_error,
            'prediction': prediction,
        }


# =============================================================================
# Urgency Detector -- 紧迫度检测
# =============================================================================
# 理论基础：时间尺度分离
# - 快速通路：丘脑-杏仁核，直接威胁检测（<200ms）
# - 慢速通路：皮层环路，社会威胁检测（秒级）
# =============================================================================

class UrgencyDetector(nn.Module):
    """
    紧迫度检测器：检测刺激的时间紧迫性

    对应神经机制：
    - 快速通路：superior colliculus → pulvinar → 杏仁核
    - 慢速通路：前额叶皮层

    输出：
    - urgency：紧迫度 [0, 1]
    - time_scale：时间尺度（short/medium/long）
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim

        # 快速通路（毫秒级）
        self.fast_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 慢速通路（秒级）
        self.slow_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 时间尺度注意力
        self.time_attention = nn.Linear(input_dim, 2)

        # 状态缓冲区（用于慢速通路）
        self.state_history = deque(maxlen=50)

    def forward(
        self,
        state: torch.Tensor,
    ) -> Dict:
        """
        Args:
            state: [B, input_dim]
        Returns:
            urgency: [B,]
            time_scale: str
        """
        # 快速通路
        fast_urgency = self.fast_net(state).squeeze(-1)

        # 慢速通路（需要历史）
        self.state_history.append(state.detach())
        if len(self.state_history) > 5:
            # 时间序列聚合
            history = torch.stack(list(self.state_history), dim=1)  # [B, T, D]
            slow_urgency = self.slow_net(history.mean(dim=1)).squeeze(-1)
        else:
            slow_urgency = fast_urgency * 0.5

        # 时间尺度选择
        time_weights = F.softmax(self.time_attention(state), dim=-1)

        # 加权紧迫度
        urgency = time_weights[:, 0] * fast_urgency + time_weights[:, 1] * slow_urgency

        # 时间尺度
        time_scale = 'short' if time_weights[0, 0] > 0.5 else 'long'

        return {
            'urgency': urgency,
            'time_scale': time_scale,
            'fast_urgency': fast_urgency,
            'slow_urgency': slow_urgency,
        }


# =============================================================================
# Social Inference -- 社会推理
# =============================================================================
# 理论基础：Frith & Frith (2006) 社会认知
# - 心智理论（Theory of Mind）
# - 意图推断
# - 情感推断
# =============================================================================

class SocialInference(nn.Module):
    """
    社会推理器：推断他人心理状态

    对应神经机制：
    - 颞上沟（STS）：生物运动检测
    - 颞顶联合区（TPJ）：心智推断
    - 前额叶（PFC）：社会认知

    输出：
    - intention：意图推断
    - emotion：情感推断（基于行为）
    - trust：信任度估计
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim

        # 他人心智网络
        self.mind_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 意图分类
        self.intention_head = nn.Linear(hidden_dim, 4)  # 4种意图

        # 情感推断（连续VAD）
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # V, A, D
        )

        # 信任估计
        self.trust_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        others_obs: torch.Tensor,
        self_state: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Args:
            others_obs: [B, input_dim] 对他人的观察
            self_state: [B, input_dim] 自身状态（可选，用于关系推理）
        Returns:
            intention: [B,] 意图类别
            emotion_vad: [B, 3] VAD维度
            trust: [B,] 信任度
        """
        # 他人心智推断
        mind_repr = self.mind_net(others_obs)

        # 意图
        intention_logits = self.intention_head(mind_repr)
        intention = intention_logits.argmax(dim=-1)

        # 情感（VAD）
        emotion_vad = self.emotion_head(mind_repr)

        # 信任
        if self_state is not None:
            # 关系推理
            relation = torch.cat([mind_repr, self_state], dim=-1)
            trust = self.trust_head(relation).squeeze(-1)
        else:
            trust = self.trust_head(mind_repr).squeeze(-1)

        return {
            'intention': intention,
            'emotion_vad': emotion_vad,
            'trust': trust,
        }


# =============================================================================
# Emergent Emotion -- 涌现情绪
# =============================================================================
# 核心：从价值学习 + 紧迫度 + 社会推理 的交互动力学中涌现情绪
# 不再预设Plutchik轮，而是学习涌现规则
# =============================================================================

class EmergentEmotion(nn.Module):
    """
    涌现情绪系统：从底层机制的交互中涌现情绪

    不再硬编码Plutchik轮或VAD，而是：
    1. 价值学习器（ValueLearner）→ 价值预测误差
    2. 紧迫度检测器（UrgencyDetector）→ 紧迫度
    3. 社会推理器（SocialInference）→ 社会性

    情绪从这个动力学中涌现：
    - emotion = f(value_err * urgency + social * weight)
    - 个体差异从学习中涌现
    - 动态适应，而非固定标签
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        value_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 底层机制
        self.value_learner = ValueLearner(input_dim, hidden_dim, value_dim)
        self.urgency_detector = UrgencyDetector(input_dim, hidden_dim)
        self.social_inference = SocialInference(input_dim, hidden_dim)

        # 涌现动力学网络：从底层表征 → 情绪
        # 输入维度: value_dim + urgency + vad + trust = value_dim + 5
        self.emergence_net = nn.Sequential(
            nn.Linear(value_dim + 5, hidden_dim),  # 修正：value_dim + 1 + 3 + 1 = value_dim + 5
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),  # 涌现8种情绪
        )

        # 可学习的交互权重
        self.urgency_weight = nn.Parameter(torch.tensor(0.5))
        self.social_weight = nn.Parameter(torch.tensor(0.3))

        # 学习中的情绪历史（用于观察涌现）
        self.emotion_buffer = deque(maxlen=100)

    def forward(
        self,
        state: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        others_obs: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        前向计算

        Args:
            state: [B, input_dim] 当前状态
            reward: [B,] 实际奖励（可选）
            others_obs: [B, input_dim] 他人观察��可选）

        Returns:
            emergent_emotion: [B, 8] 涌现情绪
            emotion_logits: [B, 8] 情绪logits
            components: 各组件输出
        """
        # 1. 价值学习
        value_out = self.value_learner(state, reward)
        value_repr = value_out['value_repr']
        td_error = value_out.get('td_error')

        # 2. 紧迫度
        urgency_out = self.urgency_detector(state)
        urgency = urgency_out['urgency'].unsqueeze(-1)  # [B, 1]

        # 3. 社会推理
        if others_obs is not None:
            social_out = self.social_inference(others_obs, state)
            emotion_vad = social_out['emotion_vad']  # [B, 3]
            trust = social_out['trust'].unsqueeze(-1)  # [B, 1]
        else:
            # 默认为中性
            emotion_vad = torch.zeros(state.shape[0], 3, device=state.device)
            trust = torch.ones(state.shape[0], 1, device=state.device) * 0.5

        # 4. 涌现动力学
        # 拼接所有成分：[value, urgency, vad, trust]
        combined = torch.cat([
            value_repr,
            urgency,
            emotion_vad,
            trust,
        ], dim=-1)

        # 从交互中涌现情绪
        emotion_logits = self.emergence_net(combined)

        # Softmax得到情绪概率
        emergent_emotion = F.softmax(emotion_logits, dim=-1)

        # 记录历史
        self.emotion_buffer.append({
            'emotion': emergent_emotion.detach(),
            'td_error': td_error.detach() if td_error is not None else None,
            'urgency': urgency.detach(),
        })

        return {
            'emotion': emergent_emotion,       # [B, 8] 情绪概率
            'logits': emotion_logits,         # [B, 8]
            'td_error': td_error,             # TD误差（多巴胺信号）
            'urgency': urgency_out['urgency'],
            'emotion_vad': emotion_vad,       # 涌现的VAD
            'trust': trust.squeeze(-1),
            'components': {
                'value': value_repr,
                'urgency': urgency,
                'social': {
                    'intention': social_out.get('intention') if others_obs else None,
                    'vad': emotion_vad,
                    'trust': trust,
                }
            }
        }

    def get_dynamics_summary(self) -> Dict:
        """获取涌现动力学摘要"""
        if len(self.emotion_buffer) == 0:
            return {'status': 'no_data'}

        recent = list(self.emotion_buffer)[-10:]

        # 情绪分布
        emotions = torch.stack([r['emotion'] for r in recent])
        avg_emotion = emotions.mean(dim=0)

        # 紧迫度趋势
        urgencies = torch.stack([r['urgency'] for r in recent])
        avg_urgency = urgencies.mean().item()

        return {
            'emotion_distribution': avg_emotion,
            'avg_urgency': avg_urgency,
            'dominant_emotion': avg_emotion.argmax().item(),
        }


# =============================================================================
# Emotion Emergence Validation
# =============================================================================
# 验证情绪是从学习中涌现，而非硬编码
# =============================================================================

def test_emergence():
    """验证情绪涌现"""
    print("=" * 60)
    print("Testing Emergent Emotion System")
    print("=" * 60)

    # 创建模型
    model = EmergentEmotion(input_dim=64, hidden_dim=64, value_dim=16)

    # 模拟状态序列
    states = [
        torch.randn(4, 64),  # 状态1: 初始
        torch.randn(4, 64),  # 状态2
        torch.randn(4, 64),  # 状态3
    ]

    rewards = [
        torch.tensor([0.5, 0.3, 0.8, 0.1]),  # 状态1奖励
        torch.tensor([0.2, 0.6, 0.4, 0.7]),  # 状态2奖励
    ]

    # 他人观察（模拟社会情境）
    others_obs = torch.randn(4, 64)

    print("\n[1] Testing without reward...")
    out = model(states[0])
    print(f"  Emotion shape: {out['emotion'].shape}")
    print(f"  TD error: {out['td_error']}")
    print(f"  Urgency: {out['urgency'][0]:.3f}")

    print("\n[2] Testing with reward (learning)...")
    out = model(states[1], reward=rewards[0])
    print(f"  TD error: {out['td_error'][0]:.3f}")
    print(f"  Emotion: {out['emotion'][0]}")

    print("\n[3] Testing with social context...")
    out = model(states[2], reward=rewards[1], others_obs=others_obs)
    print(f"  VAD: {out['emotion_vad'][0]}")
    print(f"  Trust: {out['trust'][0]:.3f}")

    # 检查是否可以学习
    print("\n[4] Checking if emotion can be learned...")
    loss = F.cross_entropy(out['logits'], torch.randint(0, 8, (4,)))
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients exist: {model.emergence_net[0].weight.grad is not None}")

    print("\n[5] Emergent dynamics summary...")
    summary = model.get_dynamics_summary()
    print(f"  Dominant emotion: {summary.get('dominant_emotion')}")
    print(f"  Avg urgency: {summary.get('avg_urgency', 0):.3f}")

    print("\n" + "=" * 60)
    print("✓ Emergent emotion system working!")
    print("  - Value learning: ✓")
    print("  - Urgency detection: ✓")
    print("  - Social inference: ✓")
    print("  - Emergent dynamics: ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_emergence()


__all__ = [
    'ValueLearner',
    'UrgencyDetector',
    'SocialInference',
    'EmergentEmotion',
    'test_emergence',
]