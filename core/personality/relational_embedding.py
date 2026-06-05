"""
多维关系嵌入 (Relational Embedding)

对应脑科学的社会认知图谱 (SCM)：
- 北京大学韩世辉课题组, Cell Reports 2026
- 将他人投射到二维空间：能力轴 × 慷慨/可信度轴
- 动态更新，指导信任决策

功能：
1. 用户画像向量
2. 动态关系更新
3. 交互模式自动切换

事件驱动:
    - 订阅 PERSONALITY_UPDATE: 收到人格更新事件时更新关系
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from core.events import PERSONALITY_UPDATE


@dataclass
class UserNode:
    """用户节点"""
    user_id: str
    # 二维坐标 (能力, 可信度)
    expertise: float = 0.5    # 专业度
    trustworthiness: float = 0.5   # 可信度
    # 统计
    interaction_count: int = 0
    last_interaction: float = 0.0
    avg_sentiment: float = 0.0


class SocialCognitiveMap(nn.Module):
    """
    社会认知图谱

    二维关系空间：
    - x轴：专业度 (expertise)
    - y轴：可信度/善意 (trustworthiness)
    """

    def __init__(self, dim: int = 2):
        super().__init__()
        self.dim = dim

        # 用户图谱
        self.users: dict[str, UserNode] = {}
        self.max_users = 1000

        # 图谱更新参数
        self.learning_rate = 0.05
        self.decay_factor = 0.95

        # 历史
        self.interaction_log = deque(maxlen=10000)

    def get_or_create_user(self, user_id: str) -> UserNode:
        """获取或创建用户"""
        if user_id not in self.users:
            self.users[user_id] = UserNode(user_id=user_id)
        return self.users[user_id]

    def update_user(
        self,
        user_id: str,
        sentiment: float = 0.0,
        is_expert: bool = False,
        is_trustworthy: bool = None,
    ):
        """
        更新用户在图谱上的位置

        根据交互反馈动态调整
        """
        user = self.get_or_create_user(user_id)
        user.interaction_count += 1
        user.last_interaction = time.time()

        # 更新专业度 (基于是否展示专业知识)
        if is_expert:
            user.expertise = min(1.0, user.expertise + self.learning_rate)
        else:
            # 自然衰减
            user.expertise = user.expertise * self.decay_factor

        # 更新可信度 (基于交互情绪)
        if is_trustworthy is not None:
            target = 1.0 if is_trustworthy else 0.0
            user.trustworthiness += self.learning_rate * (target - user.trustworthiness)
        elif sentiment != 0:
            user.trustworthiness = user.trustworthiness + sentiment * 0.1

        user.trustworthiness = np.clip(user.trustworthiness, 0, 1)

        # 更新平均情绪
        user.avg_sentiment = (
            user.avg_sentiment * 0.9 + sentiment * 0.1
        )

        # 记录
        self.interaction_log.append({
            'user_id': user_id,
            'sentiment': sentiment,
            'expertise': user.expertise,
            'trustworthiness': user.trustworthiness,
            'time': time.time(),
        })

    def get_user_profile(self, user_id: str) -> tuple[float, float]:
        """获取用户画像 (expertise, trustworthiness)"""
        user = self.get_or_create_user(user_id)
        return user.expertise, user.trustworthiness

    def get_interaction_mode(self, user_id: str) -> str:
        """
        获取交互模式

        根据用户在图谱上的位置自动切换：
        - 高 expertise + 低 trustworthiness → 专家模式 (精简严谨)
        - 高 expertise + 高 trustworthiness → 合作模式
        - 低 expertise + 高 trustworthiness → 朋友模式 (放松)
        - 低 expertise + 低 trustworthiness → 保守模式 (谨慎)
        """
        exp, trust = self.get_user_profile(user_id)

        if exp > 0.6 and trust < 0.4:
            return "expert_strict"  # 专家模式
        elif exp > 0.6 and trust >= 0.4:
            return "collaborative"  # 合作模式
        elif exp <= 0.6 and trust > 0.6:
            return "friendly"     # 朋友模式
        else:
            return "cautious"    # 保守模式

    def get_vector(self, user_id: str) -> np.ndarray:
        """获取用户向量嵌入"""
        exp, trust = self.get_user_profile(user_id)
        user = self.users.get(user_id)
        if user:
            return np.array([exp, trust, user.avg_sentiment, user.interaction_count / 100])
        return np.array([0.5, 0.5, 0.0, 0.0])


class RelationalEmbedding(nn.Module):
    """
    多维关系嵌入系统

    基于GNN的动态人际网络记忆
    """

    def __init__(self, embedding_dim: int = 64, event_bus=None):
        super().__init__()
        self.dim = embedding_dim

        # 社会认知图谱
        self.scm = SocialCognitiveMap(dim=2)

        # 嵌入网络
        self.embed_net = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # 事件总线
        self._bus = event_bus
        if self._bus is not None:
            self._bus.subscribe(PERSONALITY_UPDATE, self.on_personality_update, priority=2, name="relation")

    def update(self, user_id: str, **kwargs):
        """更新用户"""
        self.scm.update_user(user_id, **kwargs)

    def get_embedding(self, user_id: str) -> torch.Tensor:
        """获取用户嵌入"""
        vec = self.scm.get_vector(user_id)
        return self.embed_net(torch.tensor(vec, dtype=torch.float32))

    def get_mode(self, user_id: str) -> str:
        """获取交互模式"""
        return self.scm.get_interaction_mode(user_id)

    def get_all_users(self) -> list[str]:
        """获取所有用户ID"""
        return list(self.scm.users.keys())

    def get_summary(self) -> dict:
        """获取摘要"""
        return {
            'total_users': len(self.scm.users),
            'interaction_log_size': len(self.scm.interaction_log),
        }

    def on_personality_update(self, event) -> dict[str, Any]:
        """事件驱动: 响应 PERSONALITY_UPDATE"""
        user_id = event.data.get("user_id", "default")
        sentiment = event.data.get("sentiment", 0.1)
        self.update(user_id, sentiment=sentiment)
        return {"relation_updated": True}


# ============ 便捷函数 ============

def create_relational_embedding(dim: int = 64) -> RelationalEmbedding:
    """创建关系嵌入系统"""
    return RelationalEmbedding(dim=dim)


__all__ = [
    "UserNode",
    "SocialCognitiveMap",
    "RelationalEmbedding",
    "create_relational_embedding",
]
