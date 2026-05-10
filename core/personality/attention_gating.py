"""
注意力门控 (Attention Gating)

对应脑科学的前额叶-边缘系统连接：
- 中科院自动化所、Nature子刊、复旦类脑研究院
- 连接强度决定气质：外向性(奖赏渴求)、神经质(风险规避)
- 二维人格空间：社会参与 ↔ 心智探索

功能：
1. 注意力资源动态分配
2. 认知风格标定
3. 可调的气质参数
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CognitiveStyle:
    """认知风格"""
    reward_seeking: float = 0.5     # 奖赏渴求 (外向性)
    risk_avoidance: float = 0.5    # 风险规避 (神经质)
    social_exploration: float = 0.5  # 社会参与
    mental_exploration: float = 0.5  # 心智探索

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.reward_seeking,
            self.risk_avoidance,
            self.social_exploration,
            self.mental_exploration,
        ])


class PrefrontalLimbicConnection(nn.Module):
    """
    前额叶-边缘系统连接

    控制注意力资源的分配：
    - 连接强度可调 (hyperparameter)
    - 决定"气质"
    """

    def __init__(self):
        super().__init__()
        # 连接强度 (可调hyperparameter)
        # 默认中等强度
        self.reward_connection = nn.Parameter(torch.tensor(0.5))   # 奖赏通路
        self.risk_connection = nn.Parameter(torch.tensor(0.5))  # 回避通路

    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """应用前额叶-边缘连接调制"""
        # 奖赏渴求 → 增强对积极刺激的关注
        # 风险规避 → 增强对消极刺激的关注
        modulated = attention_weights * (
            self.reward_connection - self.risk_connection
        ).abs()
        return modulated


class AttentionRouter(nn.Module):
    """
    注意力路由器

    动态调整：
    - 向外 (关注用户/环境)
    - 向内 (检索知识/反思)
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

        # 路由器网络
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 2),  # [向外, 向内]
        )

        # 初始权重
        self.external_weight = 0.7
        self.internal_weight = 0.3

    def forward(self, context: torch.Tensor) -> Tuple[float, float]:
        """计算注意力分配"""
        weights = self.router(context)
        weights = torch.softmax(weights, dim=-1)

        self.external_weight = weights[0, 0].item()
        self.internal_weight = weights[0, 1].item()

        return self.external_weight, self.internal_weight

    def get_weights(self) -> Tuple[float, float]:
        """获取当前权重"""
        return self.external_weight, self.internal_weight


class AttentionGating(nn.Module):
    """
    注意力门控系统

    整合：
    1. 前额叶-边缘连接 (气质控制)
    2. 注意力路由器 (动态路由)
    3. 认知风格标定
    """

    def __init__(
        self,
        dim: int = 128,
        reward_seeking: float = 0.5,
        risk_avoidance: float = 0.5,
    ):
        super().__init__()
        self.dim = dim

        # 基础认知风格
        self.style = CognitiveStyle(
            reward_seeking=reward_seeking,
            risk_avoidance=risk_avoidance,
        )

        # 前额叶-边缘连接
        self.pfc_limbic = PrefrontalLimbicConnection()

        # 注意力路由器
        self.router = AttentionRouter(dim)

        # 历史
        self.attention_history = []

    def gate(
        self,
        task_type: str,
        user_emotion: float = 0.0,
    ) -> Dict[str, float]:
        """
        门控计算

        根据任务类型和用户情绪动态调整
        """
        # 1. 任务类型影响
        if task_type == "creative":
            # 创意任务 → 增强内部注意力 (心智探索)
            target_external = 0.3
            target_internal = 0.7
        elif task_type == "safety":
            # 安全任务 → 增强外部注意力 (风险规避)
            target_external = 0.9
            target_internal = 0.1
        elif task_type == "emotional":
            # 情感任务 → 平衡
            target_external = 0.5
            target_internal = 0.5
        else:
            # 常规任务
            target_external = 0.7
            target_internal = 0.3

        # 2. 用户情绪调制
        # 负面情绪 → 增强风险规避
        if user_emotion < -0.3:
            self.style.risk_avoidance = min(1.0, self.style.risk_avoidance + 0.1)

        # 3. 应用气质调制
        final_external = target_external * (1 + self.style.risk_avoidance - 0.5)
        final_internal = target_internal * (1 + self.style.reward_seeking - 0.5)

        # 归一化
        total = final_external + final_internal
        final_external /= total
        final_internal /= total

        # 记录
        self.attention_history.append({
            'task_type': task_type,
            'external': final_external,
            'internal': final_internal,
        })

        return {
            'external': final_external,
            'internal': final_internal,
        }

    def set_style(self, reward_seeking: float = None, risk_avoidance: float = None):
        """设置认知风格"""
        if reward_seeking is not None:
            self.style.reward_seeking = np.clip(reward_seeking, 0, 1)
        if risk_avoidance is not None:
            self.style.risk_avoidance = np.clip(risk_avoidance, 0, 1)

        # 更新连接强度
        with torch.no_grad():
            self.pfc_limbic.reward_connection.data = torch.tensor(
                self.style.reward_seeking
            )
            self.pfc_limbic.risk_connection.data = torch.tensor(
                self.style.risk_avoidance
            )

    def get_style(self) -> CognitiveStyle:
        """获取认知风格"""
        return self.style

    def get_weights(self) -> Tuple[float, float]:
        """获取当前注意力权重"""
        return self.router.get_weights()

    def get_summary(self) -> Dict:
        """获取摘要"""
        style = self.style
        return {
            'reward_seeking': style.reward_seeking,
            'risk_avoidance': style.risk_avoidance,
            'social_exploration': style.social_exploration,
            'mental_exploration': style.mental_exploration,
            'history_size': len(self.attention_history),
        }


# ============ 便捷函数 ============

def create_attention_gating(
    dim: int = 128,
    style: str = "balanced",
) -> AttentionGating:
    """创建注意力门控"""

    presets = {
        "balanced": (0.5, 0.5),
        "extrovert": (0.8, 0.3),
        "introvert": (0.3, 0.7),
        "cautious": (0.3, 0.9),
        "adventurous": (0.9, 0.2),
    }

    rs, ra = presets.get(style, (0.5, 0.5))
    return AttentionGating(dim=dim, reward_seeking=rs, risk_avoidance=ra)


__all__ = [
    "CognitiveStyle",
    "PrefrontalLimbicConnection",
    "AttentionRouter",
    "AttentionGating",
    "create_attention_gating",
]