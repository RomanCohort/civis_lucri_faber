# =============================================================================
# Emotion Regulation - 情绪调节系统
# =============================================================================
# 自上而下的情绪调控 + 代谢调节 + 社会调节
#
# 核心机制：
# 1. 前额叶自上而下调控 (PFC → Amygdala inhibition)
# 2. 代谢调节 (metabolic state → emotion modulation)
# 3. 社会调节 (social support → emotion damping)
# 4. 认知重评 (cognitive reappraisal)
# 5. 表达抑制 (response suppression)
# =============================================================================

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 前额叶调控网络 (PFC Regulation)
# =============================================================================

class PrefrontalRegulation(nn.Module):
    """
    前额叶调控网络

    对应神经机制：
    - dorsolateral PFC (DLPFC): 工作记忆、计划
    - ventromedial PFC (VMPFC): 情绪调节、价值
    - orbitofrontal PFC (OFC): 决策、奖励预测

    功能：
    - 抑制杏仁核过度激活
    - 情绪再评估
    - 目标导向调节
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        emotion_dim: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 调控网络：输入当前情绪/状态 → 调控信号
        self.regulation_net = nn.Sequential(
            nn.Linear(input_dim + emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),  # 调控强度
        )

        # 抑制网络：PFC → Amygdala 抑制
        self.inhibition_net = nn.Sequential(
            nn.Linear(input_dim + emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 抑制强度 [0, 1]
        )

        # 再评估网络：生成新的情绪解释
        self.reappraisal_net = nn.Sequential(
            nn.Linear(input_dim + emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
        )

        # 目标网络：调节目标（想要的情绪状态）
        self.goal_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        state: torch.Tensor,
        current_emotion: torch.Tensor,
    ) -> dict:
        """
        前向调控

        Args:
            state: [B, input_dim] 当前状态
            current_emotion: [B, emotion_dim] 当前情绪分布

        Returns:
            regulation: 调控信号
            inhibition: 抑制强度
            reappraisal: 再评估后的情绪
            goal: 调节目标
        """
        # 拼接状态和情绪
        combined = torch.cat([state, current_emotion], dim=-1)

        # 调控信号
        regulation = self.regulation_net(combined)

        # 抑制强度（PFC → Amygdala）
        inhibition = self.inhibition_net(combined)

        # 再评估
        reappraisal = self.reappraisal_net(combined)
        reappraisal_probs = F.softmax(reappraisal, dim=-1)

        # 调节目标
        goal = self.goal_net(state)

        # 计算调节差距
        regulation_gap = (goal - current_emotion)

        return {
            'regulation': regulation,
            'inhibition': inhibition.squeeze(-1),
            'reappraisal': reappraisal_probs,
            'goal': goal,
            'gap': regulation_gap,
        }


# =============================================================================
# 代谢调节网络 (Metabolic Regulation)
# =============================================================================

@dataclass
class MetabolicState:
    """代谢状态"""
    glucose: float      # 血糖水平 [0, 1]
    energy: float     # 能量水平 [0, 1]
    fatigue: float     # 疲劳度 [0, 1]
    cortisol: float  # 皮质醇（应激）[0, 1]
    serotonin: float  # 血清素（快乐）[0, 1]


class MetabolicRegulation(nn.Module):
    """
    代谢调节网络

    对应神经机制：
    - 下丘脑 (Hypothalamus): 代谢平衡
    - 脑岛 (Insula): 内感受觉知
    - 脑干单胺系统: 代谢调节

    功能：
    - 血糖影响情绪稳定性
    - 疲劳降低情绪调节能力
    - 皮质醇增强负面情绪
    - 血清素缓冲负面情绪
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 代谢状态编码
        self.metabolic_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),  # 5个代谢指标
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 情绪影响网络
        self.emotion_influence_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),  # 对8种基础情绪的影响
        )

        # 调节能力网络（代谢状态→调节能力）
        self.regulation_capacity_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 代谢状态历史
        self.metabolic_history = deque(maxlen=100)

    def forward(
        self,
        state: torch.Tensor,
        metabolic_state: MetabolicState | None = None,
    ) -> dict:
        """
        代谢调节

        Args:
            state: [B, input_dim] 当前状态
            metabolic_state: 代谢状态

        Returns:
            emotion_influence: 对各情绪的影响
            regulation_capacity: 调节能力
            metabolic_context: 代谢上下文
        """
        if metabolic_state is None:
            # 默认为中性
            metabolic_state = MetabolicState(
                glucose=0.5,
                energy=0.5,
                fatigue=0.5,
                cortisol=0.5,
                serotonin=0.5,
            )

        # 编码代谢状态
        metabolic_vec = torch.tensor([
            metabolic_state.glucose,
            metabolic_state.energy,
            metabolic_state.fatigue,
            metabolic_state.cortisol,
            metabolic_state.serotonin,
        ], dtype=torch.float32, device=state.device)

        if state.shape[0] > 1:
            metabolic_vec = metabolic_vec.unsqueeze(0).expand(state.shape[0], -1)

        metabolic_enc = self.metabolic_encoder(metabolic_vec)

        # 情绪影响
        combined = torch.cat([state, metabolic_enc], dim=-1)
        emotion_influence = self.emotion_influence_net(combined)

        # 调节能力（能量高→调节能力强，皮质醇高→调节能力弱）
        regulation_capacity = self.regulation_capacity_net(metabolic_enc)

        # 记录
        self.metabolic_history.append(metabolic_vec.detach())

        return {
            'emotion_influence': emotion_influence,
            'regulation_capacity': regulation_capacity.squeeze(-1),
            'metabolic_state': {
                'glucose': metabolic_state.glucose,
                'energy': metabolic_state.energy,
                'cortisol': metabolic_state.cortisol,
                'serotonin': metabolic_state.serotonin,
            }
        }


# =============================================================================
# 社会调节网络 (Social Regulation)
# =============================================================================

class SocialRegulation(nn.Module):
    """
    社会调节网络

    对应神经机制：
    - 镜像神经元系统: 模仿
    - 前扣带回 (ACC): 社会监控
    - 颞顶联合区 (TPJ): 社会认知

    功能：
    - 社会支持缓冲负面情绪
    - 情绪传染（见另一模块）
    - 社会评价影响情绪
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim

        # 社会支持网络
        self.support_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # 状态 + 支持度
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),  # 对各情绪的缓冲
        )

        # 社会评价网络
        self.evaluation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # positive/negative/neutral
        )

        # 社会调节能力
        self.regulation_strength = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        state: torch.Tensor,
        social_support: float = 0.5,
    ) -> dict:
        """
        社会调节

        Args:
            state: [B, input_dim] 当前状态
            social_support: [0, 1] 社会支持度

        Returns:
            buffer: 缓冲强度
            evaluation: 社会评价
        """
        # 扩展social_support
        if state.shape[0] > 1:
            support_vec = torch.ones(state.shape[0], 1) * social_support
        else:
            support_vec = torch.tensor([[social_support]], dtype=torch.float32, device=state.device)

        # 社会支持缓冲
        combined = torch.cat([state, support_vec], dim=-1)
        buffer = self.support_net(combined)

        # 社会评价
        evaluation = self.evaluation_net(state)

        # 计算调节强度（社会支持越高，负面情绪缓冲越强）
        regulation = self.regulation_strength * social_support

        return {
            'buffer': buffer,
            'evaluation': evaluation,
            'regulation_strength': regulation,
            'social_support': social_support,
        }


# =============================================================================
# 完整情绪调节系统
# =============================================================================

class EmotionRegulationSystem(nn.Module):
    """
    完整情绪调节系统

    整合：
    1. 前额叶调控
    2. 代谢调节
    3. 社会调节
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        emotion_dim: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim

        # 各调节子系统
        self.pfc = PrefrontalRegulation(input_dim, hidden_dim, emotion_dim)
        self.metabolic = MetabolicRegulation(input_dim, hidden_dim)
        self.social = SocialRegulation(input_dim, hidden_dim)

        # 综合调节网络
        self.integrator = nn.Sequential(
            nn.Linear(emotion_dim * 3 + 3, hidden_dim),  # 3���调节 × 8 + 3
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emotion_dim),
        )

        # 调节策略选择
        self.strategy_selector = nn.Sequential(
            nn.Linear(emotion_dim + 1 + 1, 3),  # 情绪 + 代谢 + 社会支持
            nn.Softmax(dim=-1)
        )

        # 策略：reappraisal / suppression / acceptance
        self.strategies = ['reappraisal', 'suppression', 'acceptance']

    def forward(
        self,
        state: torch.Tensor,
        current_emotion: torch.Tensor,
        metabolic_state: MetabolicState | None = None,
        social_support: float = 0.5,
    ) -> dict:
        """
        综合调节

        Args:
            state: 当前状态
            current_emotion: 当前情绪
            metabolic_state: 代谢状态
            social_support: 社会支持度

        Returns:
            regulated_emotion: 调节后的情绪
            regulation_info: 调节信息
        """
        # 1. 前额叶调控
        pfc_out = self.pfc(state, current_emotion)

        # 2. 代谢调节
        metabolic_out = self.metabolic(state, metabolic_state)

        # 3. 社会调节
        social_out = self.social(state, social_support)

        # 策略选择
        # 处理 regulation_capacity 可能是标量的情况
        reg_cap = metabolic_out['regulation_capacity']
        if isinstance(reg_cap, float):
            reg_cap_t = torch.tensor([[reg_cap]], dtype=torch.float32, device=state.device)
        else:
            reg_cap_t = reg_cap[:1].unsqueeze(0) if reg_cap.dim() == 1 else reg_cap[:1]

        strategy_input = torch.cat([
            current_emotion.mean(dim=-1, keepdim=True),
            reg_cap_t,
            torch.tensor([[social_support]], dtype=torch.float32, device=state.device),
        ], dim=-1)
        strategy_weights = self.strategy_selector(strategy_input)

        # 综合调节
        # 应用各调节的影响
        pfc_effect = pfc_out['inhibition'].unsqueeze(-1)  # [B, 1]
        metabolic_effect = F.softmax(metabolic_out['emotion_influence'], dim=-1)  # [B, 8]
        social_effect = F.softmax(social_out['buffer'], dim=-1)  # [B, 8]

        # 加权组合
        combined = torch.cat([
            current_emotion * (1 - pfc_effect),  # PFC抑制
            metabolic_effect * metabolic_out['regulation_capacity'],
            social_effect * social_out['regulation_strength'].unsqueeze(-1),
        ], dim=-1)

        regulated_emotion = self.integrator(combined)

        # 软max化
        regulated_probs = F.softmax(regulated_emotion, dim=-1)

        return {
            'regulated_emotion': regulated_probs,
            'pfc_regulation': pfc_out,
            'metabolic_regulation': metabolic_out,
            'social_regulation': social_out,
            'strategy': self.strategies[strategy_weights.argmax().item()],
            'strategy_weights': strategy_weights,
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_emotion_regulation(
    input_dim: int = 64,
    hidden_dim: int = 128,
    emotion_dim: int = 8,
) -> EmotionRegulationSystem:
    """创建情绪调节系统"""
    return EmotionRegulationSystem(input_dim, hidden_dim, emotion_dim)


__all__ = [
    'PrefrontalRegulation',
    'MetabolicRegulation',
    'SocialRegulation',
    'MetabolicState',
    'EmotionRegulationSystem',
    'create_emotion_regulation',
]


# =============================================================================
# 测试
# =============================================================================

def test_emotion_regulation():
    """测试情绪调节系统"""
    print("=" * 60)
    print("Testing Emotion Regulation System")
    print("=" * 60)

    # 创建模型
    model = EmotionRegulationSystem(input_dim=64, hidden_dim=64, emotion_dim=8)

    # 模拟输入
    state = torch.randn(4, 64)
    emotion = F.softmax(torch.randn(4, 8), dim=-1)

    # 代谢状态
    metabolic = MetabolicState(
        glucose=0.7,
        energy=0.6,
        fatigue=0.3,
        cortisol=0.4,
        serotonin=0.5,
    )

    print("\n[1] Testing regulation...")
    out = model(state, emotion, metabolic, social_support=0.6)
    print(f"  Regulated emotion: {out['regulated_emotion'][0]}")
    print(f"  Strategy: {out['strategy']}")

    print("\n[2] PFC regulation components...")
    print(f"  Inhibition: {out['pfc_regulation']['inhibition'][0]:.3f}")
    print(f"  Goal: {out['pfc_regulation']['goal'][0]}")

    print("\n[3] Metabolic components...")
    print(f"  Regulation capacity: {out['metabolic_regulation']['regulation_capacity'][0]:.3f}")

    print("\n[4] Social components...")
    print(f"  Social support: {out['social_regulation']['social_support']:.3f}")

    print("\n" + "=" * 60)
    print("✓ Emotion regulation system working!")
    print("=" * 60)


if __name__ == "__main__":
    test_emotion_regulation()
