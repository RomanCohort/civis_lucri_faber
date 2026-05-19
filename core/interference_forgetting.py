"""
干扰性遗忘系统 (Interference-Based Forgetting)

对应认知心理学的遗忘理论：
1. 前摄干扰 (Proactive Interference): 旧记忆干扰新记忆的学习
2. 倒摄干扰 (Retroactive Interference): 新学习削弱旧记忆
3. 情境保护 (Context Protection): 不同情境的记忆互不干扰
4. 相似性依赖衰减 (Similarity-Dependent Decay): 越相似干扰越强

替代原有的 FIFO 队列淘汰机制，实现更符合生物学的自然遗忘。

核心类：
1. InterferenceEngine - 干扰性遗忘引擎
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ForgettingStats:
    """遗忘统计"""
    total_proactive_interferences: int = 0
    total_retroactive_interferences: int = 0
    total_pruned: int = 0
    avg_importance_before: float = 0.0
    avg_importance_after: float = 0.0


class InterferenceEngine:
    """
    干扰性遗忘引擎

    核心原理：
    - 记忆不是按时间/重要性简单淘汰
    - 而是相似记忆之间产生互相干扰
    - 高相似度的记忆互相"竞争"，导致较弱的被遗忘
    - 不同情境（context）下的相似记忆可以共存

    参数：
        similarity_metric: 相似度度量 ('cosine' 或 'euclidean')
        decay_rate: 倒摄干扰的基础衰减率
        proactive_strength: 前摄干扰强度
        retroactive_strength: 倒摄干扰强度
        min_importance: 低于此阈值的记忆被淘汰
        context_sensitivity: 情境保护灵敏度 [0,1]
    """

    def __init__(
        self,
        similarity_metric: str = 'cosine',
        decay_rate: float = 0.01,
        proactive_strength: float = 0.3,
        retroactive_strength: float = 0.01,
        min_importance: float = 0.05,
        context_sensitivity: float = 0.5,
    ):
        self.similarity_metric = similarity_metric
        self.decay_rate = decay_rate
        self.proactive_strength = proactive_strength
        self.retroactive_strength = retroactive_strength
        self.min_importance = min_importance
        self.context_sensitivity = context_sensitivity

        self.stats = ForgettingStats()

    def compute_similarity(
        self,
        encoding1: np.ndarray,
        encoding2: np.ndarray,
    ) -> float:
        """计算两个编码之间的相似度"""
        if self.similarity_metric == 'cosine':
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            if norm1 < 1e-8 or norm2 < 1e-8:
                return 0.0
            return float(np.dot(encoding1, encoding2) / (norm1 * norm2))
        else:  # euclidean → 转换为相似度
            dist = np.linalg.norm(encoding1 - encoding2)
            return float(1.0 / (1.0 + dist))

    def compute_retroactive_interference(
        self,
        new_encoding: np.ndarray,
        existing_memories: List,
        context: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        倒摄干扰：新学习削弱相似的旧记忆

        Args:
            new_encoding: 新记忆的编码
            existing_memories: 现有记忆列表（需要有encoding和importance属性）
            context: 可选的情境向量

        Returns:
            各记忆的干扰强度列表
        """
        interferences = []
        for mem in existing_memories:
            encoding = mem.encoding if hasattr(mem, 'encoding') else mem
            similarity = self.compute_similarity(new_encoding, encoding)

            # 情境保护：不同情境下的记忆互不干扰
            context_protection = 1.0
            if context is not None and hasattr(mem, 'context'):
                if mem.context is not None:
                    ctx_sim = self.compute_similarity(context, mem.context)
                    # 情境差异越大，保护越强
                    context_protection = 1.0 - ctx_sim * self.context_sensitivity

            # 干扰强度 = 相似度 × 衰减率 × 情境保护倒数
            interference = similarity * self.decay_rate * self.retroactive_strength * context_protection

            # 应用衰减
            if hasattr(mem, 'importance'):
                mem.importance *= (1.0 - interference)

            interferences.append(interference)

        self.stats.total_retroactive_interferences += len(interferences)
        return interferences

    def compute_proactive_interference(
        self,
        old_memories: List,
        new_encoding: np.ndarray,
    ) -> float:
        """
        前摄干扰：旧记忆干扰新记忆的学习

        越多相似的旧记忆存在，新记忆的初始重要性越低。

        Args:
            old_memories: 现有记忆列表
            new_encoding: 新记忆的编码

        Returns:
            归一化的前摄干扰强度 [0, 1]
        """
        if not old_memories:
            return 0.0

        total_interference = 0.0
        for mem in old_memories:
            encoding = mem.encoding if hasattr(mem, 'encoding') else mem
            similarity = self.compute_similarity(new_encoding, encoding)
            importance = mem.importance if hasattr(mem, 'importance') else 1.0
            total_interference += similarity * importance

        # 归一化
        n = max(1, len(old_memories))
        normalized = min(1.0, total_interference / (n * self.proactive_strength))

        self.stats.total_proactive_interferences += 1
        return normalized

    def apply_forgetting(
        self,
        memories: List,
        new_encoding: Optional[np.ndarray] = None,
        context: Optional[np.ndarray] = None,
    ) -> List:
        """
        对记忆列表应用完整的干扰性遗忘流程

        Args:
            memories: 记忆列表
            new_encoding: 可选的新记忆编码（触发倒摄干扰）
            context: 可选的情境向量

        Returns:
            经过遗忘处理后的记忆列表
        """
        if not memories:
            return memories

        # 记录遗忘前的重要性
        importances_before = [
            m.importance if hasattr(m, 'importance') else 1.0
            for m in memories
        ]
        self.stats.avg_importance_before = (
            sum(importances_before) / len(importances_before)
        )

        # 1. 倒摄干扰（如果提供了新编码）
        if new_encoding is not None:
            self.compute_retroactive_interference(
                new_encoding, memories, context
            )

        # 2. 淘汰低于阈值的记忆
        pruned = []
        surviving = []
        for mem in memories:
            imp = mem.importance if hasattr(mem, 'importance') else 1.0
            if imp < self.min_importance:
                pruned.append(mem)
            else:
                surviving.append(mem)

        self.stats.total_pruned += len(pruned)

        # 记录遗忘后的重要性
        importances_after = [
            m.importance if hasattr(m, 'importance') else 1.0
            for m in surviving
        ]
        self.stats.avg_importance_after = (
            sum(importances_after) / len(importances_after) if importances_after else 0.0
        )

        return surviving

    def get_summary(self) -> Dict:
        """获取遗忘统计"""
        return {
            'proactive_interferences': self.stats.total_proactive_interferences,
            'retroactive_interferences': self.stats.total_retroactive_interferences,
            'total_pruned': self.stats.total_pruned,
            'avg_importance_before': self.stats.avg_importance_before,
            'avg_importance_after': self.stats.avg_importance_after,
        }


def create_interference_engine(**kwargs) -> InterferenceEngine:
    """创建干扰性遗忘引擎"""
    return InterferenceEngine(**kwargs)


__all__ = [
    'ForgettingStats',
    'InterferenceEngine',
    'create_interference_engine',
]
