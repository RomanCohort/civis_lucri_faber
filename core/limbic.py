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
    ) -> Dict:
        """计算情绪反应 (连续概率, 非argmax离散选择)"""
        emotions = ["joy", "sadness", "anger", "fear", "neutral"]
        emotion_idx = emotions.index(emotion) if emotion in emotions else 4

        emotion_vec = torch.zeros(5)
        emotion_vec[emotion_idx] = 1

        response_logits = self.response_net(emotion_vec)
        response_probs = F.softmax(response_logits, dim=-1)

        # 派生主反应 (向后兼容)
        response_idx = response_probs.argmax().item()
        primary_response = self.responses[response_idx]

        return {
            'response': primary_response,
            'response_probs': response_probs.detach().numpy(),
        }


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
        """学习恐惧条件 (连续记忆强度, 非硬阈值)"""
        if US < 0:  # 负性US
            cue_t = torch.tensor(cue, dtype=torch.float32).unsqueeze(0)
            fear_strength = self.condition_net(cue_t).item()

            # sigmoid平滑替代硬阈值: 强度越高越可能被存储
            formation_prob = 1.0 / (1.0 + np.exp(-10 * (fear_strength - 0.6)))
            if np.random.random() < formation_prob:
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
        response_result = self.central.compute_response(result['emotion'])

        return {
            'emotion': result['emotion'],
            'valence': result['valence'],
            'arousal': result['arousal'],
            'intensity': result['intensity'],
            'response': response_result['response'],
            'response_probs': response_result['response_probs'],
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

    def filter_noise(
        self,
        sensory_inputs: List[torch.Tensor],
        noise_threshold: float = 0.3,
    ) -> List[torch.Tensor]:
        """
        Exp 7改进: 丘脑门控噪声过滤

        通过 attention_gate 参数实际过滤噪声信号:
        - 高 attention_gate 值 → 低过滤 → ADHD模式 (噪声淹没信号)
        - 低 attention_gate 值 → 高过滤 → 正常模式 (信号清晰)

        Args:
            sensory_inputs: 感觉输入张量列表
            noise_threshold: 噪声过滤阈值

        Returns:
            过滤后的感觉输入列表
        """
        filtered_inputs = []

        for i, sensory in enumerate(sensory_inputs):
            if i >= len(self.sensory_nets):
                filtered_inputs.append(sensory)
                continue

            # 获取门控值 (sigmoid输出)
            gate_value = self.attention_gate[i].sigmoid().item()

            # 计算噪声过滤强度 (高门控→低过滤, 低门控→高过滤)
            # 正常模式: gate=1.0 → filter_strength=0.3
            # ADHD模式: gate=2.0 → filter_strength=0.1 (过滤弱)
            filter_strength = noise_threshold * (1.5 - gate_value)
            filter_strength = max(0.05, min(0.5, filter_strength))

            # 应用噪声过滤: 保留信号成分，抑制高频噪声
            if sensory.dim() >= 1 and sensory.shape[-1] > 0:
                # 计算信号的标准差作为噪声估计
                signal_std = sensory.std().item()
                signal_mean = sensory.mean().item()

                # 噪声估计: 高频波动成分
                noise_estimate = (sensory - signal_mean).abs().std().item()

                # 信号-噪声比
                snr = signal_std / (noise_estimate + 1e-6)

                # 根据SNR和门控强度调整过滤
                if snr < 2.0:  # 低信噪比 → 噪声大
                    # ADHD模式: gate高→过滤弱→保留更多噪声
                    # Normal模式: gate低→过滤强→抑制噪声
                    filtered = sensory * (1 - filter_strength * (2.0 - snr) / 2.0)
                else:
                    filtered = sensory  # 高信噪比，无需过滤

                filtered_inputs.append(filtered)
            else:
                filtered_inputs.append(sensory)

        return filtered_inputs

    def get_attention_stats(self) -> Dict[str, float]:
        """获取门控统计信息"""
        gate_values = [g.sigmoid().item() for g in self.attention_gate]
        return {
            'attention_gate_avg': np.mean(gate_values),
            'attention_gate_min': min(gate_values),
            'attention_gate_max': max(gate_values),
            'noise_filtering_weak': np.mean(gate_values) > 0.85,  # Exp 7改进: 阈值从1.5调整为0.85
        }


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
        """获取注意的状态（通过integrator变换）"""
        if not self.working_memory:
            return self.integrator(current)

        # 关注相关信息
        recent = torch.stack(list(self.working_memory))

        # 注意力加权
        attention = F.softmax(recent @ current.unsqueeze(-1), dim=0)
        attended = (attention * recent).sum(dim=0)

        # 通过integrator变换输出
        return self.integrator(attended)


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
        event_bus=None,
    ):
        super().__init__()

        self.amygdala = Amygdala(input_dim)
        self.thalamus = Thalamus(input_dim)

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "sensory_process",
                self._handle_sensory_process,
                priority=0,
                name="limbic",
            )

    def _handle_sensory_process(self, event) -> Dict:
        """Event-driven handler for sensory_process events."""
        import torch as _torch
        state_tensor = event.data.get("state_tensor", _torch.randn(1, 64))
        result = self(
            state=state_tensor,
            sensory_inputs=[state_tensor],
        )

        state = event.data.get("internal_state", {})
        state["limbic_emotion"] = result["emotion"]
        state["limbic_valence"] = result["valence"]
        state["limbic_arousal"] = result["arousal"]
        state["limbic_response"] = result["response"]
        state["limbic_emotional_attention"] = result["emotional_attention"]

        # Set emotion_criticality if fear + high arousal
        if result["emotion"] == "fear" and result["arousal"] > 0.7:
            state["emotion_criticality"] = True

        return result

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
    'AmygdalaWithPrior',
    'ThalamicRelay',
    'MDNucleus',
    'Thalamus',
    'LimbicSystem',
    'create_limbic_system',
]


# ============ AmygdalaWithPrior - 面部区域先验 ============
# 对应Censor的AmygdalaWithPrior


class AmygdalaWithPrior(nn.Module):
    """
    杏仁核 + 面部区域先验（对应Censor的AmygdalaWithPrior）
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_h: int = 14,
        output_w: int = 14,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_h * output_w)
        self.prior_strength = nn.Parameter(torch.tensor(0.3))
        self.register_buffer('face_prior', self._create_face_prior())

    def _create_face_prior(self) -> torch.Tensor:
        h, w = 14, 14
        self.output_h = h
        self.output_w = w
        prior = torch.zeros(h, w)
        prior[h//6:h//3, w//4:3*w//4] = 1.0  # 眼
        prior[h//3:h//2, w//3:2*w//3] = 1.0  # 鼻
        prior[2*h//3:5*h//6, w//4:3*w//4] = 1.0  # 嘴
        return prior / (prior.sum() + 1e-8)

    def forward(self, fast_feat: torch.Tensor) -> Dict:
        h = torch.relu(self.fc1(fast_feat))
        learned_map = torch.sigmoid(self.fc2(h).view(-1, 1, 14, 14))
        attention_map = (1 - self.prior_strength) * learned_map + self.prior_strength * self.face_prior.view(1, 1, 14, 14)
        return {'attention_map': attention_map, 'prior_strength': self.prior_strength}