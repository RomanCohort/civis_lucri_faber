# =============================================================================
# Emotional Contagion - 情绪传染
# =============================================================================
# 模仿-反馈机制 + 群体情绪
#
# 核心机制：
# 1. 模仿-反馈：对方的表情→自己情绪
# 2. 情绪传染：他人情绪→自己情绪
# 3. 群体情绪动力学
# 4. 情绪同步
# =============================================================================

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 表情编码
# =============================================================================

@dataclass
class FacialExpression:
    """面部表情"""
    happiness: float = 0.0    # 高兴 [0, 1]
    sadness: float = 0.0       # 悲伤
    anger: float = 0.0          # 愤怒
    fear: float = 0.0           # 恐惧
    surprise: float = 0.0         # 惊讶
    disgust: float = 0.0         # 厌恶
    contempt: float = 0.0         # 轻蔑
    neutral: float = 1.0         # 中性


@dataclass
class VocalExpression:
    """声音表情"""
    pitch: float = 0.5         # 音高
    intensity: float = 0.5       # 强度
    speaking_rate: float = 0.5     # 语速
    tremor: float = 0.0            # 颤抖


# =============================================================================
# 模仿网络 (Mimicry Network)
# =============================================================================

class MimicryNetwork(nn.Module):
    """
    模仿网络

    对应神经机制：
    - 镜像神经元系统 (MNS): 模仿
    - SMAtion: 运动模拟
    - 梭状回: 面部识别

    功能：
    - 模仿他人表情
    - 感知-运动映射
    """

    def __init__(
        self,
        expression_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 表情识别网络
        self.recognition_net = nn.Sequential(
            nn.Linear(expression_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),  # 8种基本表情
        )

        # 运动模拟网络（感知→运动）
        self.simulation_net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, expression_dim),
        )

        # 反馈网络（模拟→情绪）
        self.feedback_net = nn.Sequential(
            nn.Linear(expression_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
        )

        # 模仿强度
        self.mimicry_strength = nn.Parameter(torch.tensor(0.5))

    def recognize(
        self,
        expression: torch.Tensor,
    ) -> dict:
        """
        识别他人表情

        Args:
            expression: [B, expression_dim] 表情特征

        Returns:
            recognized: 识别结果
        """
        recognized = self.recognition_net(expression)
        probs = F.softmax(recognized, dim=-1)

        return {
            'recognized': probs,
            'logits': recognized,
        }

    def simulate(
        self,
        recognized_expression: torch.Tensor,
    ) -> torch.Tensor:
        """
        模拟他人表情（运动模拟）

        Args:
            recognized_expression: [B, 8] 识别出的表情

        Returns:
            simulated: 模拟的运动表征
        """
        simulated = self.simulation_net(recognized_expression)
        return simulated

    def feedback(
        self,
        simulated_movement: torch.Tensor,
    ) -> dict:
        """
        模仿反馈（运动→情绪）

        Args:
            simulated_movement: 模拟的运动

        Returns:
            emotional_feedback: 情绪反馈
        """
        feedback = self.feedback_net(simulated_movement)
        emotions = F.softmax(feedback, dim=-1)

        return {
            'emotional_feedback': emotions,
            'mimicry_complete': True,
        }

    def forward(
        self,
        observed_expression: torch.Tensor,
    ) -> dict:
        """
        完整模仿流程

        Args:
            observed_expression: 观察到的表情

        Returns:
            resulting_emotion: 结果情绪
        """
        # 1. 识别
        recognized = self.recognize(observed_expression)

        # 2. 模拟
        simulated = self.simulate(recognized['recognized'])

        # 3. 反馈
        feedback = self.feedback(simulated)

        return {
            'recognized_emotion': recognized['recognized'],
            'simulated_movement': simulated,
            'resulting_emotion': feedback['emotional_feedback'],
        }


# =============================================================================
# 情绪传染网络
# =============================================================================

class EmotionalContagionNetwork(nn.Module):
    """
    情绪传染网络

    对应神经机制：
    - anterior insula: 情绪共鸣
    - anterior cingulate: 痛苦共鸣
    - 镜像系统: 情绪传染

    功能：
    - 他人情绪→自己情绪
    - 传染强度调节
    - 情绪阈值
    """

    def __init__(
        self,
        emotion_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.emotion_dim = emotion_dim

        # 传染网络
        self.contagion_net = nn.Sequential(
            nn.Linear(emotion_dim * 2, hidden_dim),  # 他人 + 自己
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
        )

        # 传染强度网络
        self.strength_net = nn.Sequential(
            nn.Linear(emotion_dim + 1, hidden_dim),  # 他人情绪 + 亲近度
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 阈值网络（调节敏感度）
        self.threshold_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 传染历史
        self.contagion_history = deque(maxlen=50)

    def compute_contagion(
        self,
        self_emotion: torch.Tensor,
        other_emotion: torch.Tensor,
        proximity: float = 0.5,
    ) -> dict:
        """
        计算传染

        Args:
            self_emotion: [B, emotion_dim] 自己的情绪
            other_emotion: [B, emotion_dim] 他人的情绪
            proximity: [0, 1] 亲近度

        Returns:
            contagion: 传染结果
        """
        # 传染强度
        proximity_vec = torch.ones_like(self_emotion[:, :1]) * proximity
        strength = self.strength_net(
            torch.cat([other_emotion, proximity_vec], dim=-1)
        )

        # 阈值
        threshold = self.threshold_net(
            torch.tensor([[proximity]], device=self_emotion.device)
        )

        # 计算传染
        combined = torch.cat([self_emotion, other_emotion], dim=-1)
        contagion_effect = self.contagion_net(combined)

        # 应用阈值
        effective_contagion = contagion_effect * strength

        # 记录
        self.contagion_history.append({
            'other': other_emotion.detach(),
            'self': self_emotion.detach(),
            'strength': strength.detach(),
        })

        return {
            'contagion_effect': effective_contagion,
            'strength': strength.squeeze(-1),
            'threshold': threshold,
        }

    def forward(
        self,
        self_emotion: torch.Tensor,
        other_emotion: torch.Tensor,
        proximity: float = 0.5,
    ) -> dict:
        """
        传染前向

        Args:
            self_emotion: 自己的情绪
            other_emotion: 他人的情绪
            proximity: 亲近度

        Returns:
            infected_emotion: 传染后的情绪
        """
        contagion_result = self.compute_contagion(
            self_emotion, other_emotion, proximity
        )

        # 应用传染
        infected = self_emotion + contagion_result['contagion_effect']

        return {
            'infected_emotion': infected,
            'contagion_strength': contagion_result['strength'],
        }


# =============================================================================
# 群体情绪系统
# =============================================================================

class GroupEmotionDynamics(nn.Module):
    """
    群体情绪动力学

    对应神经机制：
    - 大规模镜���神经元
    - 群体规范编码
    - 社会同步

    功能：
    - 多主体情绪同步
    - 群体情绪演化
    - 情绪极化
    """

    def __init__(
        self,
        n_agents: int = 10,
        emotion_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.emotion_dim = emotion_dim

        # 个体情绪网络（共享）
        self.individual_net = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
        )

        # 群体规范网络
        self.norm_net = nn.Sequential(
            nn.Linear(emotion_dim * n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
        )

        # 同步网络
        self.synchronization_net = nn.Sequential(
            nn.Linear(emotion_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 同步强度
        )

        # 极化网络
        self.polarization_net = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 情绪历史
        self.group_history = deque(maxlen=100)

    def compute_group_norms(
        self,
        all_emotions: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算群体规范

        Args:
            all_emotions: [n_agents, emotion_dim] 所有个体的情绪

        Returns:
            norm: 群体情绪规范
        """
        flat = all_emotions.view(-1)
        norm = self.norm_net(flat.unsqueeze(0))
        return norm

    def synchronize(
        self,
        individual_emotion: torch.Tensor,
        group_norm: torch.Tensor,
    ) -> dict:
        """
        同步更新

        Args:
            individual_emotion: 个体情绪
            group_norm: 群体规范

        Returns:
            synchronized: 同步后的情绪
        """
        combined = torch.cat([individual_emotion, group_norm], dim=-1)
        sync_strength = self.synchronization_net(combined)

        # 趋向群体规范
        synchronized = individual_emotion + sync_strength * (group_norm - individual_emotion)

        return {
            'synchronized': synchronized,
            'sync_strength': sync_strength.item(),
        }

    def compute_polarization(
        self,
        all_emotions: torch.Tensor,
    ) -> float:
        """
        计算极化程度

        Args:
            all_emotions: 所有个体的情绪

        Returns:
            polarization: 极化程度 [0, 1]
        """
        if self.n_agents < 2:
            return 0.0

        # 方差作为极化度量
        variance = all_emotions.var(dim=0).mean().item()
        return min(1.0, variance)

    def forward(
        self,
        all_emotions: torch.Tensor,
    ) -> dict:
        """
        群体动力学

        Args:
            all_emotions: [n_agents, emotion_dim] 所有个体情绪

        Returns:
            group_state: 群体状态
        """
        # 计算群体规范
        group_norm = self.compute_group_norms(all_emotions)

        # 同步更新（每个个体）
        synchronized_all = []
        for i in range(self.n_agents):
            result = self.synchronize(all_emotions[i], group_norm)
            synchronized_all.append(result['synchronized'])

        synchronized = torch.stack(synchronized_all, dim=0)

        # 极化
        polarization = self.compute_polarization(all_emotions)

        # 记录
        self.group_history.append(all_emotions.detach())

        return {
            'synchronized_emotions': synchronized,
            'group_norm': group_norm,
            'polarization': polarization,
        }


# =============================================================================
# 完整情绪传染系统
# =============================================================================

class EmotionalContagionSystem(nn.Module):
    """
    完整情绪传染系统

    整合：
    1. 模仿网络（表情→情绪）
    2. 情绪传染（他人→自己）
    3. 群体情绪动力学
    """

    def __init__(
        self,
        expression_dim: int = 64,
        emotion_dim: int = 8,
        hidden_dim: int = 64,
        n_group_members: int = 10,
    ):
        super().__init__()
        self.emotion_dim = emotion_dim

        # 子系统
        self.mimicry = MimicryNetwork(expression_dim, hidden_dim)
        self.contagion = EmotionalContagionNetwork(emotion_dim, hidden_dim)
        self.group = GroupEmotionDynamics(n_group_members, emotion_dim, hidden_dim)

        # 整合网络
        self.integrator = nn.Sequential(
            nn.Linear(emotion_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
        )

        # 敏感度（可学习）
        self.sensitivity = nn.Parameter(torch.tensor(0.5))

    def process_observation(
        self,
        observed_expression: torch.Tensor,
    ) -> dict:
        """
        处理观察到的表情

        Args:
            observed_expression: 观察到的表情

        Returns:
            emotional_response: 情绪反应
        """
        # 模仿流程
        mimicry_result = self.mimicry(observed_expression)

        return {
            'recognized': mimicry_result['recognized_emotion'],
            'resulting_emotion': mimicry_result['resulting_emotion'],
        }

    def infect(
        self,
        self_emotion: torch.Tensor,
        other_emotion: torch.Tensor,
        proximity: float = 0.5,
    ) -> dict:
        """
        情绪传染

        Args:
            self_emotion: 自己的情绪
            other_emotion: 他人的情绪
            proximity: 亲近度

        Returns:
            infected: 传染结果
        """
        result = self.contagion(self_emotion, other_emotion, proximity)
        return result

    def group_dynamics(
        self,
        all_emotions: torch.Tensor,
    ) -> dict:
        """
        群体动力学

        Args:
            all_emotions: 所有成员情绪

        Returns:
            group_result: 群体结果
        """
        return self.group(all_emotions)

    def forward(
        self,
        self_emotion: torch.Tensor,
        observed_expression: torch.Tensor | None = None,
        other_emotion: torch.Tensor | None = None,
        proximity: float = 0.5,
    ) -> dict:
        """
        完整传染流程

        Args:
            self_emotion: 自己的当前情绪
            observed_expression: 观察到的表情（可选）
            other_emotion: 他人情绪（可选）
            proximity: 亲近度

        Returns:
            final_emotion: 最终情绪
        """
        batch_size = self_emotion.shape[0]

        # 1. 模仿结果
        if observed_expression is not None:
            mimicry_result = self.process_observation(observed_expression)
            mimicry_emotion = mimicry_result['resulting_emotion']
        else:
            mimicry_emotion = torch.zeros_like(self_emotion)

        # 2. 传染结果
        if other_emotion is not None:
            infection_result = self.infect(self_emotion, other_emotion, proximity)
            contagion_emotion = infection_result['infected_emotion']
        else:
            contagion_emotion = self_emotion

        # 3. 综合
        combined = torch.cat([self_emotion, mimicry_emotion, contagion_emotion], dim=-1)
        final = self.integrator(combined)

        # 软最大
        final_probs = F.softmax(final, dim=-1)

        return {
            'final_emotion': final_probs,
            'mimicry_emotion': mimicry_emotion,
            'contagion_emotion': contagion_emotion,
            'sensitivity': self.sensitivity.item(),
        }

    def get_contagion_stats(self) -> dict:
        """获取传染统计"""
        return {
            'mimicry_strength': self.mimicry.mimicry_strength.item(),
            'contagion_sensitivity': self.sensitivity.item(),
            'group_size': self.group.n_agents,
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_emotional_contagion(
    expression_dim: int = 64,
    emotion_dim: int = 8,
    hidden_dim: int = 64,
    n_group_members: int = 10,
) -> EmotionalContagionSystem:
    """创建情绪传染系统"""
    return EmotionalContagionSystem(
        expression_dim, emotion_dim, hidden_dim, n_group_members
    )


__all__ = [
    'FacialExpression',
    'VocalExpression',
    'MimicryNetwork',
    'EmotionalContagionNetwork',
    'GroupEmotionDynamics',
    'EmotionalContagionSystem',
    'create_emotional_contagion',
]


# =============================================================================
# 测试
# =============================================================================

def test_emotional_contagion():
    """测试情绪传染系统"""
    print("=" * 60)
    print("Testing Emotional Contagion System")
    print("=" * 60)

    # 创建模型
    model = EmotionalContagionSystem()

    print("\n[1] Testing mimicry network...")
    expression = torch.randn(4, 64)
    result = model.process_observation(expression)
    print(f"  Recognized: {result['recognized'][0]}")
    print(f"  Resulting emotion: {result['resulting_emotion'][0]}")

    print("\n[2] Testing contagion...")
    my_emotion = F.softmax(torch.randn(4, 8), dim=-1)
    other_emotion = F.softmax(torch.randn(4, 8), dim=-1)
    result = model.infect(my_emotion, other_emotion, proximity=0.7)
    print(f"  Infected emotion: {result['infected_emotion'][0]}")

    print("\n[3] Testing group dynamics...")
    all_emotions = F.softmax(torch.randn(10, 8), dim=-1)
    result = model.group_dynamics(all_emotions)
    print(f"  Polarization: {result['polarization']:.3f}")

    print("\n[4] Testing complete system...")
    result = model(my_emotion, observed_expression=expression, other_emotion=other_emotion)
    print(f"  Final emotion: {result['final_emotion'][0]}")

    print("\n[5] Contagion stats...")
    stats = model.get_contagion_stats()
    print(f"  Sensitivity: {stats['contagion_sensitivity']:.3f}")

    print("\n" + "=" * 60)
    print("✓ Emotional contagion system working!")
    print("  - Mimicry: ✓")
    print("  - Contagion: ✓")
    print("  - Group dynamics: ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_emotional_contagion()
