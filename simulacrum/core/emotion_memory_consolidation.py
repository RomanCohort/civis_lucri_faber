# =============================================================================
# Emotion Memory Consolidation - 情绪记忆巩固
# =============================================================================
# 睡眠依赖的记忆巩固 + 再consolidation + 恐惧消退
#
# 核心机制：
# 1. 睡眠依赖重播：海马-皮层记忆转移
# 2. 再consolidation：记忆再激活后的修改
# 3. 恐惧消退：Extinction learning
# 4. 情绪记忆特异性巩固
# =============================================================================

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 情绪记忆结构
# =============================================================================

@dataclass
class EmotionalMemoryTrace:
    """情绪记忆痕迹"""
    id: int
    content: np.ndarray      # 内容表征
    emotion: str           # 情绪类型
    valence: float         # 效价
    arousal: float        # 唤醒度
    importance: float     # 重要性
    timestamp: int        # 时间戳
    reactivations: int    # 重激活次数
    consolidation_level: float  # 巩固水平 [0-1]


@dataclass
class SleepStageData:
    """睡眠阶段数据（用于巩固系统的阶段权重）"""
    NREM1: float = 0.0   # N1 浅睡
    NREM2: float = 0.0   # N2 中睡
    NREM3: float = 0.0   # N3 深睡（慢波）
    REM: float = 0.0      # REM 快速眼动
    AWAKE: float = 1.0    # 清醒


# =============================================================================
# 睡眠依赖重播系统
# =============================================================================

class SleepDependentReplay(nn.Module):
    """
    睡眠依赖重播系统

    对应神经机制：
    - Sharp-Wave Ripples (SWR): 海马事件
    - 睡眠纺锤波: NREM N2
    - 皮层慢振荡: NREM N3
    - REM theta: REM睡眠

    功能：
    - 清醒时：事件编码到海马
    - NREM深睡：海马→皮层转移
    - REM：情绪记忆整合
    """

    def __init__(
        self,
        content_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim

        # 海马编码器
        self.hippocampus_encoder = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 皮层编码器
        self.cortex_encoder = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 重播网络（海马→皮层）
        self.replay_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, content_dim),
        )

        # 选择性网络（重要记忆优先重播）
        self.priority_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, 1),  # 内容 + 重要性
        )

        # 重播缓冲区
        self.replay_buffer = deque(maxlen=50)

        # 统计
        self.total_replays = 0
        self.cortical_transfers = 0

    def encode_to_hippocampus(
        self,
        content: torch.Tensor,
        emotion: float = 0.5,
    ) -> torch.Tensor:
        """编码到海马"""
        return self.hippocampus_encoder(content)

    def replay(
        self,
        memory: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """
        执行重播

        Args:
            memory: 待重播记忆
            target: 目标皮层表征

        Returns:
            replay_loss: 重播损失
            transferred: 是否转移成功
        """
        # 计算目标
        target_repr = self.cortex_encoder(target)

        # 使用priority_net计算重放优先级权重
        importance = torch.tensor([target.norm().item()]).unsqueeze(0)
        priority_input = torch.cat([target_repr, importance], dim=-1)
        priority_score = torch.sigmoid(self.priority_net(priority_input))

        # 重播（按优先级加权）
        replayed = self.replay_net(memory) * priority_score

        # 损失
        loss = F.mse_loss(replayed, target)

        # 估算转移
        similarity = F.cosine_similarity(
            replayed.unsqueeze(0),
            target.unsqueeze(0)
        ).item()

        transferred = similarity > 0.7

        # 记录
        self.replay_buffer.append({
            'memory': memory.detach(),
            'replayed': replayed.detach(),
            'similarity': similarity,
        })
        self.total_replays += 1
        if transferred:
            self.cortical_transfers += 1

        return {
            'loss': loss,
            'replayed': replayed,
            'similarity': similarity,
            'transferred': transferred,
        }

    def get_replay_stats(self) -> dict:
        """获取重播统计"""
        return {
            'total_replays': self.total_replays,
            'cortical_transfers': self.cortical_transfers,
            'transfer_rate': self.cortical_transfers / max(1, self.total_replays),
        }


# =============================================================================
# 再consolidation系统
# =============================================================================

class ReconsolidationSystem(nn.Module):
    """
    再consolidation系统

    对应神经机制：
    - 记忆再激活 → 记忆进入不稳定态
    - 重新consolidation → 新信息整合

    功能：
    - 检索时激活记忆
    - 不稳定化后可修改
    - 重新consolidation
    """

    def __init__(
        self,
        content_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim

        # 记忆检索网络
        self.retrieval_net = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, content_dim),
        )

        # 不稳定化网络（再激活触发）
        self.destabilization_net = nn.Sequential(
            nn.Linear(content_dim + 1, hidden_dim),  # 记忆 + 重激活强度
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 再consolidation网络
        self.reconsolidation_net = nn.Sequential(
            nn.Linear(content_dim + content_dim, hidden_dim),  # 旧记忆 + 新信息
            nn.ReLU(),
            nn.Linear(hidden_dim, content_dim),
        )

        # 活跃记忆
        self.active_memories: list[EmotionalMemoryTrace] = []

        # 不稳定记忆ID
        self.destabilized_ids: set = set()

    def retrieve(
        self,
        cue: torch.Tensor,
    ) -> torch.Tensor:
        """检索记忆"""
        return self.retrieval_net(cue)

    def reactivate(
        self,
        memory: torch.Tensor,
        strength: float = 1.0,
    ) -> dict:
        """
        重激活记忆

        Args:
            memory: 记忆表征
            strength: 重激活强度

        Returns:
            destabilized: 是否进入不稳定态
            decay_rate: 衰减率
        """
        combined = torch.cat([memory, torch.tensor([[strength]], device=memory.device)], dim=-1)
        instability = self.destabilization_net(combined).item()

        # 不稳定化阈值
        destabilized = instability > 0.5

        # 计算衰减
        decay_rate = instability * 0.1  # 不稳定态易衰减

        return {
            'destabilized': destabilized,
            'instability': instability,
            'decay_rate': decay_rate,
            'requires_reconsolidation': destabilized,
        }

    def reconsolidate(
        self,
        old_memory: torch.Tensor,
        new_information: torch.Tensor,
    ) -> torch.Tensor:
        """
        重新consolidation

        Args:
            old_memory: 原始记忆
            new_information: 新信息

        Returns:
            updated_memory: 更新后的记忆
        """
        combined = torch.cat([old_memory, new_information], dim=-1)
        updated = self.reconsolidation_net(combined)

        return updated


# =============================================================================
# 恐惧消退系统
# =============================================================================

class FearExtinctionSystem(nn.Module):
    """
    恐惧消退系统

    对应神经机制：
    - 恐惧消退发生在Infra-Prelimbic区
    - 条件性抑制学习
    - 情境依赖消退

    功能：
    - 新情境中消退
    - 渐进式降低恐惧
    - 消退恢复（ spontaneous recovery）
    """

    def __init__(
        self,
        cue_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.cue_dim = cue_dim
        self.hidden_dim = hidden_dim

        # 恐惧网络
        self.fear_net = nn.Sequential(
            nn.Linear(cue_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 消退网络（IPL）
        self.extinction_net = nn.Sequential(
            nn.Linear(cue_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 情境编码
        self.context_net = nn.Sequential(
            nn.Linear(cue_dim, hidden_dim),
            nn.ReLU(),
        )

        # 记忆网络（恐惧情境关联）
        self.fear_memory: dict[int, float] = {}  # context_id -> fear_strength
        self.extinction_memory: dict[int, float] = {}  # context_id -> extinction_strength

        # 消退参数
        self.extinction_threshold = 0.3  # 低于此值认为消退
        self.spontaneous_recovery_rate = 0.1

    def learn_fear(
        self,
        cue: torch.Tensor,
        US: float,  # 非条件刺激强度
        context_id: int,
    ) -> float:
        """
        学习恐惧

        Args:
            cue: 线索
            US: 非条件刺激强度
            context_id: 情境ID

        Returns:
            fear_strength: 恐惧强度
        """
        fear = self.fear_net(cue).item() * US
        self.fear_memory[context_id] = fear

        return fear

    def learn_extinction(
        self,
        cue: torch.Tensor,
        context_id: int,
        safe_exposure: float = 1.0,
    ) -> dict:
        """
        学习消退

        Args:
            cue: 线索
            context_id: 情境ID
            safe_exposure: 安全暴露次数

        Returns:
            extinction_strength: 消退强度
            fear_level: 当前恐惧水平
        """
        # 使用context_net编码情境信息
        context_encoding = self.context_net(cue)
        context_mod = context_encoding.norm().item() * 0.1

        extinction = self.extinction_net(cue).item() * safe_exposure + context_mod

        # 如果新情境，逐渐降低恐惧
        if context_id not in self.extinction_memory:
            self.extinction_memory[context_id] = extinction
        else:
            # 渐进消退
            self.extinction_memory[context_id] = (
                0.9 * self.extinction_memory[context_id] + 0.1 * extinction
            )

        # 恐惧水平 = 原始恐惧 - 消退
        original_fear = self.fear_memory.get(context_id, 0.5)
        extinction_str = self.extinction_memory.get(context_id, 0)
        fear_level = max(0, original_fear - extinction_str)

        return {
            'extinction_strength': extinction_str,
            'fear_level': fear_level,
            'extinguished': fear_level < self.extinction_threshold,
        }

    def predict_fear(
        self,
        cue: torch.Tensor,
        context_id: int,
    ) -> float:
        """
        预测恐惧

        Args:
            cue: 线索
            context_id: 情境ID

        Returns:
            fear: 恐惧水平
        """
        base_fear = self.fear_net(cue).item()
        # 使用context_net调制情境依赖的恐惧
        context_encoding = self.context_net(cue)
        context_mod = context_encoding.norm().item() * 0.1
        extinction = self.extinction_memory.get(context_id, 0)

        fear = max(0, base_fear - extinction - context_mod)

        return fear

    def spontaneous_recovery(
        self,
        context_id: int,
    ):
        """自发恢复（睡眠后可能恢复）"""
        if context_id in self.extinction_memory:
            # 消退记忆逐渐恢复
            self.extinction_memory[context_id] *= (1 - self.spontaneous_recovery_rate)


# =============================================================================
# ���绪记忆巩固系统
# =============================================================================

class EmotionalMemoryConsolidation(nn.Module):
    """
    完整情绪记忆巩固系统

    整合：
    1. 睡眠依赖重播
    2. 再consolidation
    3. 恐惧消退
    """

    def __init__(
        self,
        content_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.content_dim = content_dim

        # 子系统
        self.replay = SleepDependentReplay(content_dim, hidden_dim)
        self.reconsolidation = ReconsolidationSystem(content_dim, hidden_dim)
        self.extinction = FearExtinctionSystem(content_dim, hidden_dim)

        # 情绪记忆存储
        self.emotional_memories: list[EmotionalMemoryTrace] = []

        # 整合网络
        self.consolidation_gate = nn.Sequential(
            nn.Linear(content_dim + 2, 1),  # 内容 + 情绪(Valence, Arousal)
            nn.Sigmoid()
        )

    def encode_emotional_memory(
        self,
        content: torch.Tensor,
        emotion: str,
        valence: float,
        arousal: float,
        importance: float = 0.5,
    ) -> int:
        """
        编码情绪记忆

        Args:
            content: 内容
            emotion: 情绪类型
            valence: 效价
            arousal: 唤醒度
            importance: 重要性

        Returns:
            memory_id: 记忆ID
        """
        memory_id = len(self.emotional_memories)

        trace = EmotionalMemoryTrace(
            id=memory_id,
            content=content.detach().cpu().numpy(),
            emotion=emotion,
            valence=valence,
            arousal=arousal,
            importance=importance,
            timestamp=len(self.emotional_memories),
            reactivations=0,
            consolidation_level=0.0,
        )

        self.emotional_memories.append(trace)

        return memory_id

    def consolidate(
        self,
        memory_id: int,
        sleep_stage: SleepStageData,
    ) -> dict:
        """
        巩固记忆

        Args:
            memory_id: 记忆ID
            sleep_stage: 睡眠阶段

        Returns:
            consolidation_result: 巩固结果
        """
        if memory_id >= len(self.emotional_memories):
            return {'error': 'Invalid memory_id'}

        memory = self.emotional_memories[memory_id]
        content = torch.tensor(memory.content, dtype=torch.float32).unsqueeze(0)

        # 使用consolidation_gate门控巩固强度
        gate_input = torch.cat([
            content,
            torch.tensor([[memory.valence, memory.arousal]])
        ], dim=-1)
        gate_value = self.consolidation_gate(gate_input).item()

        # 编码到海马
        hippocampal = self.replay.encode_to_hippocampus(content, memory.arousal)

        # 根据睡眠阶段决定重播
        if sleep_stage.NREM3 > 0.5:
            # 慢波睡眠：海马→皮层转移（门控调制）
            target = content
            replay_result = self.replay.replay(hippocampal, target)
            memory.consolidation_level = min(
                1.0,
                memory.consolidation_level + 0.3 * gate_value
            )

        elif sleep_stage.REM > 0.5:
            # REM：情绪整合（门控调制）
            memory.consolidation_level += 0.1 * gate_value

        else:
            # 清醒：维持
            pass

        return {
            'memory_id': memory_id,
            'consolidation_level': memory.consolidation_level,
            'sleep_stage': sleep_stage,
        }

    def retrieve_emotion_memory(
        self,
        cue: torch.Tensor,
    ) -> dict:
        """
        检索情绪记忆

        Args:
            cue: 检索线索

        Returns:
            retrieved: 检索结果
        """
        # 检索
        retrieved = self.reconsolidation.retrieve(cue)

        # 重激活检查
        react_result = self.reconsolidation.reactivate(retrieved)

        return {
            'retrieved': retrieved,
            'reactivations': react_result.get('instability', 0),
            'requires_reconsolidation': react_result.get('requires_reconsolidation', False),
        }

    def update_memory(
        self,
        memory_id: int,
        new_information: torch.Tensor,
    ) -> torch.Tensor:
        """
        更新记忆（再consolidation）

        Args:
            memory_id: 记忆ID
            new_information: 新信息

        Returns:
            updated: 更新后的内容
        """
        if memory_id >= len(self.emotional_memories):
            return None

        old_content = torch.tensor(
            self.emotional_memories[memory_id].content,
            dtype=torch.float32
        ).unsqueeze(0)

        updated = self.reconsolidation.reconsolidate(old_content, new_information)

        # 更新存储
        self.emotional_memories[memory_id].content = updated.detach().cpu().numpy()[0]
        self.emotional_memories[memory_id].reactivations += 1

        return updated

    def get_memory_summary(self) -> dict:
        """获取记忆摘要"""
        if not self.emotional_memories:
            return {'status': 'no_memories'}

        return {
            'total_memories': len(self.emotional_memories),
            'consolidated': sum(1 for m in self.emotional_memories if m.consolidation_level > 0.5),
            'avg_consolidation': np.mean([m.consolidation_level for m in self.emotional_memories]),
            'replay_stats': self.replay.get_replay_stats(),
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_emotion_memory_consolidation(
    content_dim: int = 64,
    hidden_dim: int = 128,
) -> EmotionalMemoryConsolidation:
    """创建情绪记忆巩固系统"""
    return EmotionalMemoryConsolidation(content_dim, hidden_dim)


__all__ = [
    'EmotionalMemoryTrace',
    'SleepStageData',
    'SleepDependentReplay',
    'ReconsolidationSystem',
    'FearExtinctionSystem',
    'EmotionalMemoryConsolidation',
    'create_emotion_memory_consolidation',
]


# =============================================================================
# 测试
# =============================================================================

def test_emotion_memory_consolidation():
    """测试情绪记忆巩固"""
    print("=" * 60)
    print("Testing Emotion Memory Consolidation")
    print("=" * 60)

    # 创建模型
    model = EmotionalMemoryConsolidation()

    print("\n[1] Encoding emotional memory...")
    content = torch.randn(1, 64)
    memory_id = model.encode_emotional_memory(
        content,
        emotion='fear',
        valence=-0.7,
        arousal=0.8,
        importance=0.9,
    )
    print(f"  Memory ID: {memory_id}")

    print("\n[2] Consolidating during deep sleep (NREM3)...")
    sleep_stage = SleepStageData(NREM3=0.9, NREM1=0.1)
    result = model.consolidate(memory_id, sleep_stage)
    print(f"  Consolidation level: {result['consolidation_level']:.3f}")

    print("\n[3] Retrieving memory...")
    cue = torch.randn(1, 64)
    retrieved = model.retrieve_emotion_memory(cue)
    print(f"  Retrieved: {retrieved['retrieved'].shape}")
    print(f"  Requires reconsolidation: {retrieved['requires_reconsolidation']}")

    print("\n[4] Fear extinction...")
    cue = torch.randn(1, 64)
    fear_result = model.extinction.learn_extinction(cue, context_id=1)
    print(f"  Fear level: {fear_result['fear_level']:.3f}")
    print(f"  Extinguished: {fear_result['extinguished']}")

    print("\n[5] Memory summary...")
    summary = model.get_memory_summary()
    print(f"  Total: {summary['total_memories']}")
    print(f"  Consolidated: {summary['consolidated']}")

    print("\n" + "=" * 60)
    print("✓ Emotion memory consolidation working!")
    print("=" * 60)


if __name__ == "__main__":
    test_emotion_memory_consolidation()
