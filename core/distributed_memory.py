"""
分布式记忆表征 (Distributed Memory Representation)

对应生物学的全脑分布式记忆机制：
- 单一记忆同时存储在多个脑区（海马、杏仁核、丘脑、前额叶）
- 每个脑区存储不同维度的信息片段
- 检索时从多个脑区组装完整记忆
- 支持部分损伤容错（某脑区"受损"时记忆质量下降但不消失）

核心类：
1. DistributedMemoryStore - 分布式记忆存储与检索
2. MemoryTrace - 跨脑区记忆痕迹索引
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import uuid
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MemoryFragment:
    """记忆片段：单个脑区的记忆表征"""
    fragment_id: str
    region: str               # 来源脑区
    encoding: np.ndarray      # 该脑区的编码
    timestamp: int
    quality: float = 1.0      # 片段质量 [0-1]，受损时下降


@dataclass
class MemoryTrace:
    """跨脑区记忆痕迹：链接同一经历在不同脑区的片段"""
    trace_id: str
    fragments: Dict[str, MemoryFragment] = field(default_factory=dict)
    importance: float = 1.0
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.0
    timestamp: int = 0

    @property
    def completeness(self) -> float:
        """记忆完整性：有多少脑区参与了存储"""
        if not self.fragments:
            return 0.0
        total_quality = sum(f.quality for f in self.fragments.values())
        return total_quality / max(len(self.fragments), 1)


class DistributedMemoryStore(nn.Module):
    """
    分布式记忆存储系统

    每次编码时，信息同时通过多个脑区处理并存储片段。
    检索时从多脑区组装，支持部分损伤容错。

    对应生物学：
    - 海马体：情景细节（what, where, when）
    - 杏仁核：情绪效价和唤醒度
    - 前额叶：目标和上下文
    - 丘脑：时间序列信息
    """

    def __init__(
        self,
        state_dim: int = 64,
        encoding_dim: int = 128,
        n_regions: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.encoding_dim = encoding_dim
        self.n_regions = n_regions

        # 各脑区的编码器
        self.region_encoders = nn.ModuleDict({
            'hippocampus': nn.Sequential(
                nn.Linear(state_dim, encoding_dim),
                nn.ReLU(),
                nn.Linear(encoding_dim, encoding_dim),
                nn.Tanh(),
            ),
            'amygdala': nn.Sequential(
                nn.Linear(state_dim + 2, encoding_dim),  # state + valence + arousal
                nn.ReLU(),
                nn.Linear(encoding_dim, encoding_dim),
                nn.Tanh(),
            ),
            'prefrontal': nn.Sequential(
                nn.Linear(state_dim, encoding_dim),
                nn.ReLU(),
                nn.Linear(encoding_dim, encoding_dim),
                nn.Tanh(),
            ),
            'thalamus': nn.Sequential(
                nn.Linear(state_dim, encoding_dim // 2),
                nn.ReLU(),
                nn.Linear(encoding_dim // 2, encoding_dim),
                nn.Tanh(),
            ),
        })

        # 跨区域注意力融合（用于检索时的多区域合并）
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=encoding_dim,
            num_heads=4,
            batch_first=True,
        )

        # 存储索引
        self.traces: Dict[str, MemoryTrace] = {}
        self._timeline: List[str] = []  # 时间顺序的trace_id

        # 各区域的独立片段索引
        self._region_index: Dict[str, List[str]] = defaultdict(list)

        # 容量控制
        self.max_traces = 1000

        # 脑区健康度（1.0=完全健康，0.0=完全受损）
        self.region_health: Dict[str, float] = {
            'hippocampus': 1.0,
            'amygdala': 1.0,
            'prefrontal': 1.0,
            'thalamus': 1.0,
        }

    def encode(
        self,
        state: np.ndarray,
        valence: float = 0.0,
        arousal: float = 0.0,
        importance: float = 1.0,
    ) -> str:
        """
        分布式编码：同时通过多个脑区编码

        Args:
            state: 输入状态
            valence: 情绪效价 [-1, 1]
            arousal: 唤醒度 [0, 1]
            importance: 重要性 [0, 1]

        Returns:
            trace_id: 跨脑区记忆痕迹ID
        """
        trace_id = str(uuid.uuid4())[:8]
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        fragments = {}

        # 海马体编码（情景细节）
        hc_enc = self.region_encoders['hippocampus'](state_t)
        fragments['hippocampus'] = MemoryFragment(
            fragment_id=f"{trace_id}_hc",
            region='hippocampus',
            encoding=hc_enc.detach().numpy()[0],
            timestamp=len(self._timeline),
        )

        # 杏仁核编码（情绪信息）
        emotion_input = torch.cat([
            state_t,
            torch.tensor([[valence, arousal]])
        ], dim=-1)
        amy_enc = self.region_encoders['amygdala'](emotion_input)
        fragments['amygdala'] = MemoryFragment(
            fragment_id=f"{trace_id}_amy",
            region='amygdala',
            encoding=amy_enc.detach().numpy()[0],
            timestamp=len(self._timeline),
        )

        # 前额叶编码（上下文/目标）
        pfc_enc = self.region_encoders['prefrontal'](state_t)
        fragments['prefrontal'] = MemoryFragment(
            fragment_id=f"{trace_id}_pfc",
            region='prefrontal',
            encoding=pfc_enc.detach().numpy()[0],
            timestamp=len(self._timeline),
        )

        # 丘脑编码（时间序列）
        thal_enc = self.region_encoders['thalamus'](state_t)
        fragments['thalamus'] = MemoryFragment(
            fragment_id=f"{trace_id}_thal",
            region='thalamus',
            encoding=thal_enc.detach().numpy()[0],
            timestamp=len(self._timeline),
        )

        # 创建跨脑区记忆痕迹
        trace = MemoryTrace(
            trace_id=trace_id,
            fragments=fragments,
            importance=importance,
            emotion_valence=valence,
            emotion_arousal=arousal,
            timestamp=len(self._timeline),
        )

        self.traces[trace_id] = trace
        self._timeline.append(trace_id)

        # 更新区域索引
        for region_name in fragments:
            self._region_index[region_name].append(trace_id)

        # 容量控制（按时间淘汰最旧记忆）
        if len(self.traces) > self.max_traces:
            oldest_id = self._timeline[0]
            self._remove_trace(oldest_id)

        return trace_id

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 5,
        valence: float = None,
    ) -> List[Dict]:
        """
        分布式检索：从多个脑区独立检索后合并

        Args:
            query: 查询状态
            top_k: 返回数量
            valence: 可选的情绪效价过滤

        Returns:
            检索结果列表，每个结果包含trace和完整性得分
        """
        if not self.traces:
            return []

        query_t = torch.tensor(query, dtype=torch.float32).unsqueeze(0)

        # 从每个健康脑区独立检索
        region_scores: Dict[str, Dict[str, float]] = {}

        for region_name, encoder in self.region_encoders.items():
            health = self.region_health.get(region_name, 1.0)
            if health < 0.1:
                continue  # 该脑区严重受损，跳过

            query_enc = encoder(query_t if region_name != 'amygdala' else
                               torch.cat([query_t, torch.zeros(1, 2)], dim=-1))
            query_np = query_enc.detach().numpy()[0]

            scores = {}
            for trace_id in self._region_index.get(region_name, []):
                trace = self.traces.get(trace_id)
                if trace and region_name in trace.fragments:
                    frag = trace.fragments[region_name]
                    # 余弦相似度
                    sim = float(np.dot(query_np, frag.encoding) /
                               (np.linalg.norm(query_np) * np.linalg.norm(frag.encoding) + 1e-8))
                    # 按脑区健康度加权
                    scores[trace_id] = sim * health * frag.quality

            region_scores[region_name] = scores

        # 跨区域排名聚合（投票式合并）
        aggregated: Dict[str, float] = defaultdict(float)
        for region_name, scores in region_scores.items():
            for trace_id, score in scores.items():
                aggregated[trace_id] += score

        # 情绪效价过滤
        if valence is not None:
            for trace_id in list(aggregated.keys()):
                trace = self.traces.get(trace_id)
                if trace and abs(trace.emotion_valence - valence) > 0.5:
                    aggregated[trace_id] *= 0.5  # 降低不匹配的分数

        # 按聚合分数排序
        ranked = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)

        results = []
        for trace_id, score in ranked[:top_k]:
            trace = self.traces.get(trace_id)
            if trace:
                results.append({
                    'trace_id': trace_id,
                    'score': score,
                    'completeness': trace.completeness,
                    'importance': trace.importance,
                    'valence': trace.emotion_valence,
                    'arousal': trace.emotion_arousal,
                    'n_regions': len(trace.fragments),
                })

        return results

    def simulate_lesion(self, region: str, severity: float = 0.0):
        """
        模拟脑区损伤

        Args:
            region: 脑区名称
            severity: 损伤严重度 [0, 1]，0=完全健康，1=完全损毁
        """
        if region in self.region_health:
            self.region_health[region] = 1.0 - severity

            # 损伤该脑区的所有片段质量
            if severity > 0.5:
                for trace in self.traces.values():
                    if region in trace.fragments:
                        trace.fragments[region].quality = max(0.0, 1.0 - severity)

    def heal_lesion(self, region: str):
        """恢复脑区健康"""
        if region in self.region_health:
            self.region_health[region] = 1.0
            for trace in self.traces.values():
                if region in trace.fragments:
                    trace.fragments[region].quality = 1.0

    def _remove_trace(self, trace_id: str):
        """移除一个记忆痕迹"""
        trace = self.traces.pop(trace_id, None)
        if trace:
            for region_name in trace.fragments:
                if trace_id in self._region_index.get(region_name, []):
                    self._region_index[region_name].remove(trace_id)
            if trace_id in self._timeline:
                self._timeline.remove(trace_id)

    def get_summary(self) -> Dict:
        """获取系统摘要"""
        completeness_scores = [t.completeness for t in self.traces.values()]
        return {
            'total_traces': len(self.traces),
            'region_health': dict(self.region_health),
            'avg_completeness': np.mean(completeness_scores) if completeness_scores else 0.0,
            'min_completeness': min(completeness_scores) if completeness_scores else 0.0,
            'capacity_usage': len(self.traces) / self.max_traces,
        }


def create_distributed_memory(
    state_dim: int = 64,
    encoding_dim: int = 128,
) -> DistributedMemoryStore:
    """创建分布式记忆存储系统"""
    return DistributedMemoryStore(state_dim, encoding_dim)


__all__ = [
    'MemoryFragment',
    'MemoryTrace',
    'DistributedMemoryStore',
    'create_distributed_memory',
]
