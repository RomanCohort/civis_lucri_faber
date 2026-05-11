"""
杏仁核与丘脑系统 (Amygdala + Thalamus)

对应生物学的：
1. Amygdala - 情绪学习、恐惧条件化
2. Thalamus - 感觉中继、时间信息流

核心功能：
1. 情绪编码与记忆
2. 恐惧条件化
3. 感觉中继
4. 时间信息流
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from collections import deque


# ============ 杏仁核 (Amygdala) ============

@dataclass
class EmotionalMemory:
    """情绪记忆"""
    state: np.ndarray
    emotion: str  # "fear", "joy", "anger", "sadness", "neutral"
    valence: float  # -1 ~ 1
    arousal: float  # 0 ~ 1
    intensity: float
    timestamp: int


@dataclass
class FearCondition:
    """恐惧条件"""
    cue: np.ndarray
    response: str
    strength: float


class AmygdalaNucleus(nn.Module):
    """
    杏仁核核心

    情感价值评估
    """

    def __init__(
        self,
        input_dim: int = 64,
    ):
        super().__init__()

        # 情绪网络
        self.emotion_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 5),  # 5种基本情绪
            nn.Softmax(dim=-1)
        )

        # 价值网络
        self.valence_net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # -1 ~ 1
        )

        # 唤醒度网络
        self.arousal_net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 0 ~ 1
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> Dict:
        """评估情绪"""
        emotion_probs = self.emotion_net(state)[0]
        valence = self.valence_net(state).item()
        arousal = self.arousal_net(state).item()

        # 情绪类别
        emotions = ["joy", "sadness", "anger", "fear", "neutral"]
        emotion = emotions[emotion_probs.argmax().item()]

        return {
            'emotion': emotion,
            'emotion_probs': emotion_probs.detach().numpy(),
            'valence': valence,
            'arousal': arousal,
            'intensity': valence * arousal,
        }


class BasolateralAmygdala(nn.Module):
    """
    基底外侧杏仁核 (BLA)

    情绪记忆形成
    """

    def __init__(
        self,
        input_dim: int = 64,
        memory_dim: int = 64,
    ):
        super().__init__()

        self.memory_dim = memory_dim
        self.input_dim = input_dim

        # 情绪记忆编码 (state + emotion one-hot = input_dim + 5)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 5, memory_dim),  # state + emotion (5 types)
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
        )

        # 记忆存储
        self.emotional_memories: List[EmotionalMemory] = []

    def encode(
        self,
        state: np.ndarray,
        emotion: str,
        valence: float,
    ) -> np.ndarray:
        """编码情绪记忆"""
        state_t = torch.tensor(state, dtype=torch.float32)

        # 情绪one-hot
        emotions = ["joy", "sadness", "anger", "fear", "neutral"]
        emotion_vec = torch.zeros(5)
        if emotion in emotions:
            emotion_vec[emotions.index(emotion)] = 1

        # 拼接编码
        combined = torch.cat([state_t, emotion_vec * abs(valence)])
        encoding = self.encoder(combined).detach().numpy()

        return encoding

    def retrieve_by_emotion(
        self,
        emotion: str,
    ) -> List[EmotionalMemory]:
        """按情绪检索"""
        return [m for m in self.emotional_memories if m.emotion == emotion]


class CentralNucleus(nn.Module):
    """
    中央杏仁核 (CeA)

    情绪反应输出
    """

    def __init__(self):
        super().__init__()

        # 反应网络
        self.response_net = nn.Sequential(
            nn.Linear(5, 16),  # 情绪输入
            nn.ReLU(),
            nn.Linear(16, 4),  # 反应输出
        )

        # 反应类型
        self.responses = ["fight", "flight", "freeze", "calm"]

    def compute_response(
        self,
        emotion: str,
    ) -> str:
        """计算情绪反应"""
        emotions = ["joy", "sadness", "anger", "fear", "neutral"]
        emotion_idx = emotions.index(emotion) if emotion in emotions else 4

        emotion_vec = torch.zeros(5)
        emotion_vec[emotion_idx] = 1

        response = self.response_net(emotion_vec)
        response_idx = response.argmax().item()

        return self.responses[response_idx]


class FearConditioning(nn.Module):
    """
    恐惧条件化系统
    """

    def __init__(self):
        super().__init__()

        # 条件反射网络
        self.condition_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 条件记忆
        self.fear_memories: List[FearCondition] = []
        self.fear_threshold = 0.7

    def learn_fear(
        self,
        cue: np.ndarray,
        US: float,  # Unconditioned Stimulus
    ):
        """学习恐惧条件"""
        if US < 0:  # 负性US
            cue_t = torch.tensor(cue, dtype=torch.float32).unsqueeze(0)
            fear_strength = self.condition_net(cue_t).item()

            if fear_strength > self.fear_threshold:
                self.fear_memories.append(FearCondition(
                    cue=cue,
                    response="fear",
                    strength=fear_strength,
                ))

    def detect_fear(
        self,
        cue: np.ndarray,
    ) -> float:
        """检测恐惧反应"""
        cue_t = torch.tensor(cue, dtype=torch.float32).unsqueeze(0)
        fear = self.condition_net(cue_t).item()

        return fear

    def extinguish_fear(
        self,
        safe_exposure: int,
    ):
        """恐惧消退学习"""
        if len(self.fear_memories) > safe_exposure:
            # 逐渐降低恐惧
            for mem in self.fear_memories[-safe_exposure:]:
                mem.strength *= 0.9


class Amygdala(nn.Module):
    """
    完整杏仁核系统
    """

    def __init__(
        self,
        input_dim: int = 64,
    ):
        super().__init__()

        self.input_dim = input_dim

        # 各子核
        self.nucleus = AmygdalaNucleus(input_dim)
        self.bla = BasolateralAmygdala(input_dim)
        self.central = CentralNucleus()
        self.fear = FearConditioning()

        # 情绪记忆
        self.emotion_history = deque(maxlen=100)

    def process(
        self,
        state: torch.Tensor,
        store: bool = True,
    ) -> Dict:
        """处理情绪"""
        # 核心评估
        result = self.nucleus(state)

        # 编码记忆
        if store:
            state_np = state.detach().numpy()[0]
            encoding = self.bla.encode(
                state_np,
                result['emotion'],
                result['valence'],
            )

            memory = EmotionalMemory(
                state=state_np,
                emotion=result['emotion'],
                valence=result['valence'],
                arousal=result['arousal'],
                intensity=result['intensity'],
                timestamp=len(self.emotion_history),
            )
            self.emotion_history.append(memory)

        # 反应
        response = self.central.compute_response(result['emotion'])

        return {
            'emotion': result['emotion'],
            'valence': result['valence'],
            'arousal': result['arousal'],
            'intensity': result['intensity'],
            'response': response,
        }

    def get_emotion_summary(self) -> Dict:
        """获取情绪摘要"""
        if not self.emotion_history:
            return {'emotion': 'neutral', 'valence': 0, 'arousal': 0}

        recent = list(self.emotion_history)[-10:]
        return {
            'emotion': recent[-1].emotion,
            'avg_valence': np.mean([m.valence for m in recent]),
            'avg_arousal': np.mean([m.arousal for m in recent]),
        }


# ============ 丘脑 (Thalamus) ============

class ThalamicRelay(nn.Module):
    """
    丘脑中继

    感觉信息路由
    """

    def __init__(
        self,
        input_dim: int = 64,
        n_senses: int = 4,  # 视觉、听觉、触觉、本体感觉
    ):
        super().__init__()

        self.n_senses = n_senses

        # 感觉网络
        self.sensory_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
            )
            for _ in range(n_senses)
        ])

        # 注意力门控
        self.attention_gate = nn.Parameter(torch.ones(n_senses))

    def relay(
        self,
        sensory_inputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        中继感觉信息
        """
        outputs = []
        for i, sensory in enumerate(sensory_inputs):
            if i < len(self.sensory_nets):
                out = self.sensory_nets[i](sensory)
                att = self.attention_gate[i].sigmoid()
                outputs.append(out * att)
            else:
                outputs.append(sensory)

        # 整合
        return torch.stack(outputs).mean(dim=0)


class MDNucleus(nn.Module):
    """
    丘脑内侧核 (MD)

    高级感觉整合 + 工作记忆
    """

    def __init__(
        self,
        input_dim: int = 64,
    ):
        super().__init__()

        # 整合网络
        self.integrator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # 工作记忆
        self.working_memory = deque(maxlen=5)

    def update_memory(
        self,
        state: torch.Tensor,
    ):
        """更新工作记忆"""
        self.working_memory.append(state.detach())

    def get_attended_state(
        self,
        current: torch.Tensor,
    ) -> torch.Tensor:
        """获取注意的状态"""
        if not self.working_memory:
            return current

        # 关注相关信息
        recent = torch.stack(list(self.working_memory))

        # 简单attention
        attention = F.softmax(recent @ current.unsqueeze(-1), dim=0)
        attended = (attention * recent).sum(dim=0)

        return attended


class PUL(nn.Module):
    """
    丘脑枕核 (PUL)

    视觉信息处理
    """

    def __init__(self):
        super().__init__()

        # 视觉处理
        self.visual_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def process_visual(
        self,
        visual_input: torch.Tensor,
    ) -> torch.Tensor:
        """处理视觉"""
        return self.visual_net(visual_input)


class Thalamus(nn.Module):
    """
    完整丘脑系统
    """

    def __init__(
        self,
        input_dim: int = 64,
    ):
        super().__init__()

        self.input_dim = input_dim

        # 子核
        self.relay = ThalamicRelay(input_dim)
        self.md = MDNucleus(input_dim)

        # 时间信息流
        self.temporal_buffer = deque(maxlen=20)

    def process(
        self,
        sensory_inputs: List[torch.Tensor],
        state: torch.Tensor = None,
    ) -> Dict:
        """处理信息流"""
        # 中继
        relayed = self.relay.relay(sensory_inputs)

        # 工作记忆更新
        if state is not None:
            self.md.update_memory(state)

        # 时间序列
        if state is not None:
            self.temporal_buffer.append(state.detach().clone())

        # 时间特征
        if len(self.temporal_buffer) > 1:
            temporal_features = self.temporal_buffer[-1] - self.temporal_buffer[0]
        else:
            temporal_features = torch.zeros_like(state) if state is not None else None

        return {
            'relayed': relayed,
            'attended': state,
            'temporal_features': temporal_features,
        }

    def get_timing_info(self) -> Dict:
        """获取时间信息"""
        return {
            'buffer_length': len(self.temporal_buffer),
            'time_since_start': len(self.temporal_buffer),
        }


# ============ 整合系统 ============

class LimbicSystem(nn.Module):
    """
    边缘系统整合

    Amygdala + Thalamus
    """

    def __init__(
        self,
        input_dim: int = 64,
    ):
        super().__init__()

        self.amygdala = Amygdala(input_dim)
        self.thalamus = Thalamus(input_dim)

    def forward(
        self,
        state: torch.Tensor,
        sensory_inputs: List[torch.Tensor] = None,
    ) -> Dict:
        """整合处理"""
        # 情绪
        emotion_result = self.amygdala.process(state)

        # 丘脑
        if sensory_inputs is not None:
            thalamus_result = self.thalamus.process(sensory_inputs, state)
        else:
            thalamus_result = {'relayed': state, 'temporal_features': None}

        # 整合
        # 情绪影响注意力
        emotional_attention = 1.0 + emotion_result['arousal'] * 0.5

        return {
            'emotion': emotion_result['emotion'],
            'valence': emotion_result['valence'],
            'arousal': emotion_result['arousal'],
            'response': emotion_result['response'],
            'attended_state': thalamus_result.get('attended'),
            'emotional_attention': emotional_attention,
        }

    def get_summary(self) -> Dict:
        """获取摘要"""
        return {
            'amygdala': self.amygdala.get_emotion_summary(),
            'thalamus': self.thalamus.get_timing_info(),
        }


# ============ 便捷函数 ============

def create_limbic_system(
    input_dim: int = 64,
) -> LimbicSystem:
    """创建边缘系统"""
    return LimbicSystem(input_dim)


__all__ = [
    'EmotionalMemory',
    'FearCondition',
    'AmygdalaNucleus',
    'BasolateralAmygdala',
    'CentralNucleus',
    'FearConditioning',
    'Amygdala',
    'ThalamicRelay',
    'MDNucleus',
    'Thalamus',
    'LimbicSystem',
    'create_limbic_system',
]