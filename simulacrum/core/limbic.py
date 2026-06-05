"""
杏仁核与丘脑系统 (Amygdala + Thalamus)

对应生物学的：
1. Amygdala - 情绪学习、恐惧条件化
2. Thalamus - 感觉中继、时间信息流
3. vmPFC - 恐惧消退学习 (新增)

核心功能：
1. 情绪编码与记忆
2. 恐惧条件化
3. 感觉中继
4. 时间信息流
5. 恐惧消退学习 - 新增
6. 杏仁核-海马情绪调制连接 - 新增

参考:
- LeDoux (2000) - 恐惧条件化的神经机制
- Milad & Quirk (2012) - vmPFC恐惧消退
- Richter-Levin & Akirav (2010) - Amygdala-Hippocampus交互
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ) -> dict:
        """评估情绪"""
        emotion_probs = self.emotion_net(state)[0]
        valence = self.valence_net(state).item()
        arousal = self.arousal_net(state).item()

        # 情绪类别
        emotions = ["joy", "sadness", "anger", "fear", "neutral"]
        emotion = emotions[emotion_probs.argmax().item()]

        return {
            'emotion': emotion,
            'emotion_probs': emotion_probs.detach().cpu().numpy(),
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
        self.emotional_memories: list[EmotionalMemory] = []

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
        encoding = self.encoder(combined).detach().cpu().numpy()

        return encoding

    def retrieve_by_emotion(
        self,
        emotion: str,
    ) -> list[EmotionalMemory]:
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
    ) -> dict:
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
            'response_probs': response_probs.detach().cpu().numpy(),
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
        self.fear_memories: list[FearCondition] = []
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
    ) -> dict:
        """处理情绪"""
        # 核心评估
        result = self.nucleus(state)

        # 编码记忆
        if store:
            state_np = state.detach().cpu().numpy()[0]
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

    def get_emotion_summary(self) -> dict:
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
    增强的丘脑中继

    感觉信息路由 + 动态唤醒度调制

    改进:
    - 动态门控: 根据唤醒度调制感觉吞吐量
    - 网状核抑制: 模拟丘脑网状核的抑制性门控
    - 新颖性放大: 增强新颖刺激的信号
    - 感觉特定增益: 每种感觉模态独立的增益控制
    """

    def __init__(
        self,
        input_dim: int = 64,
        n_senses: int = 4,  # 视觉、听觉、触觉、本体感觉
        n_modalities: int = 3,  # 模态数量 (visual, auditory, language)
    ):
        super().__init__()

        self.n_senses = n_senses
        self.n_modalities = n_modalities

        # 感觉网络
        self.sensory_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
            )
            for _ in range(n_senses)
        ])

        # 基础门控 (可学习基线，不是静态的)
        self.base_gate = nn.Parameter(torch.ones(n_senses) * 0.5)

        # 唤醒度增益: 高唤醒时增强威胁感觉
        # 视觉 > 听觉 > 语言
        self.arousal_gain = nn.Parameter(torch.tensor([1.2, 1.0, 0.8]))

        # 感觉特定增益 (每种模态独立的可学习增益)
        self.sensory_gains = nn.Parameter(torch.ones(n_modalities))

        # 丘脑网状核模拟 (抑制性中间神经元)
        # 学习基于注意力抑制特定感觉通道
        self.reticular_nucleus = nn.Sequential(
            nn.Linear(n_senses + 1, 16),  # +1 for arousal input
            nn.ReLU(),
            nn.Linear(16, n_senses),
            nn.Sigmoid(),  # 输出 0=完全抑制, 1=无抑制
        )

        # 新颖性检测
        self.novelty_threshold = 0.3
        self._previous_sensory_input = None

    def relay(
        self,
        sensory_inputs: list[torch.Tensor],
        arousal: float = 0.5,
    ) -> torch.Tensor:
        """
        中继感觉信息

        Args:
            sensory_inputs: 感觉输入张量列表
            arousal: 当前唤醒度水平 [0, 1]
        """
        outputs = []

        # 合并所有感觉输入用于处理
        batch_size = sensory_inputs[0].shape[0] if sensory_inputs else 1
        device = sensory_inputs[0].device if sensory_inputs else 'cpu'

        # 1. 基础门控 (可学习的基线)
        base_gated = torch.sigmoid(self.base_gate)  # [n_senses]

        # 2. 唤醒度调制
        # 高唤醒 -> 放大威胁感觉; 低唤醒 -> 抑制所有感觉
        arousal_tensor = torch.tensor(arousal, device=device)
        arousal_factor = 1.0 + (arousal_tensor - 0.5) * self.arousal_gain
        arousal_factor = torch.clamp(arousal_factor, 0.1, 3.0)

        # 扩展 arousal_factor 到所有感觉通道
        senses_per_modality = self.n_senses // self.n_modalities
        if senses_per_modality == 0:
            senses_per_modality = 1
        arousal_expanded = arousal_factor.repeat(senses_per_modality)
        if len(arousal_expanded) < self.n_senses:
            arousal_expanded = torch.cat([
                arousal_expanded,
                torch.ones(self.n_senses - len(arousal_expanded), device=device)
            ])
        elif len(arousal_expanded) > self.n_senses:
            arousal_expanded = arousal_expanded[:self.n_senses]

        # 3. 感觉特定增益
        gains_expanded = torch.sigmoid(self.sensory_gains).repeat(senses_per_modality)
        if len(gains_expanded) < self.n_senses:
            gains_expanded = torch.cat([
                gains_expanded,
                torch.ones(self.n_senses - len(gains_expanded), device=device)
            ])
        elif len(gains_expanded) > self.n_senses:
            gains_expanded = gains_expanded[:self.n_senses]

        # 4. 网状核抑制
        # 先将感觉输入投影到低维，再组合唤醒度
        combined_input = torch.stack(sensory_inputs).mean(dim=0)  # [batch, dim]
        # 投影到 n_senses 维度
        sensory_proj = nn.Linear(combined_input.shape[-1], self.n_senses).to(device)
        projected = sensory_proj(combined_input)  # [batch, n_senses]
        reticular_input = torch.cat([
            projected,
            arousal_tensor.expand(batch_size, 1)
        ], dim=-1)  # [batch, n_senses+1]
        inhibition = self.reticular_nucleus(reticular_input)  # [batch, n_senses]
        # 网状核输出是 [0,1]，1表示无抑制，需要反转
        inhibition = 1.0 - inhibition

        # 5. 新颖性放大
        novelty_gate = torch.ones(batch_size, self.n_senses, device=device)
        if self._previous_sensory_input is not None and len(self._previous_sensory_input) == len(sensory_inputs):
            for i, sensory in enumerate(sensory_inputs):
                if i < len(self._previous_sensory_input) and i < self.n_senses:
                    delta = torch.abs(sensory - self._previous_sensory_input[i])
                    # 计算新颖性比例: 超过阈值的元素比例
                    novelty_ratio = (delta > self.novelty_threshold).float().mean(dim=-1)  # [batch]
                    novelty_gate[:, i] = 1.0 + novelty_ratio * 0.5  # 新颖信号放大50%

        # 更新历史记录
        self._previous_sensory_input = [s.detach().clone() for s in sensory_inputs]

        # 6. 组合所有门控因子
        for i, sensory in enumerate(sensory_inputs):
            if i < len(self.sensory_nets):
                out = self.sensory_nets[i](sensory)

                # 组合门控: 基础 × 唤醒度 × 增益 × 网状核抑制 × 新颖性
                gate = (
                    base_gated[i] *
                    arousal_expanded[i] *
                    gains_expanded[i] *
                    inhibition[:, i].mean() *  # 取平均抑制
                    novelty_gate[:, i].mean()  # 取平均新颖性
                )
                gate = torch.clamp(gate, 0.0, 1.0)

                outputs.append(out * gate)
            else:
                outputs.append(sensory)

        # 整合
        return torch.stack(outputs).mean(dim=0)

    def filter_noise(
        self,
        sensory_inputs: list[torch.Tensor],
        noise_threshold: float = 0.3,
        arousal: float = 0.5,
    ) -> list[torch.Tensor]:
        """
        增强的噪声过滤: 根据唤醒度和网状核活动动态调整

        Args:
            sensory_inputs: 感觉输入张量列表
            noise_threshold: 噪声过滤阈值
            arousal: 当前唤醒度水平

        Returns:
            过滤后的感觉输入列表
        """
        filtered_inputs = []

        # 根据唤醒度调整过滤强度
        # 高唤醒 -> 过滤弱 (保留信号); 低唤醒 -> 过滤强
        arousal_filter_factor = 0.3 + (arousal - 0.5) * 0.2  # [0.2, 0.4]

        for i, sensory in enumerate(sensory_inputs):
            if i >= len(self.sensory_nets):
                filtered_inputs.append(sensory)
                continue

            # 获取基础门控值
            gate_value = torch.sigmoid(self.base_gate[i]).item()

            # 计算过滤强度
            # 高门控 -> 低过滤 (ADHD模式); 低门控 -> 高过滤 (正常模式)
            filter_strength = noise_threshold * (1.5 - gate_value) * arousal_filter_factor
            filter_strength = max(0.05, min(0.5, filter_strength))

            # 应用噪声过滤
            if sensory.dim() >= 1 and sensory.shape[-1] > 0:
                signal_std = sensory.std().item()
                signal_mean = sensory.mean().item()
                noise_estimate = (sensory - signal_mean).abs().std().item()
                snr = signal_std / (noise_estimate + 1e-6)

                if snr < 2.0:
                    # 低信噪比 -> 噪声过滤
                    filtered = sensory * (1 - filter_strength * (2.0 - snr) / 2.0)
                else:
                    filtered = sensory

                filtered_inputs.append(filtered)
            else:
                filtered_inputs.append(sensory)

        return filtered_inputs

    def get_attention_stats(self) -> dict[str, float]:
        """获取门控统计信息"""
        gate_values = torch.sigmoid(self.base_gate).detach().cpu().numpy()
        arousal_gains = torch.sigmoid(self.arousal_gain).detach().cpu().numpy()

        return {
            'attention_gate_avg': float(np.mean(gate_values)),
            'attention_gate_min': float(np.min(gate_values)),
            'attention_gate_max': float(np.max(gate_values)),
            'arousal_gain_visual': float(arousal_gains[0]),
            'arousal_gain_auditory': float(arousal_gains[1]),
            'arousal_gain_language': float(arousal_gains[2]),
            'dynamic_gating': True,  # 标记为动态门控模式
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
        self.relay = ThalamicRelay(input_dim, n_modalities=3)
        self.md = MDNucleus(input_dim)

        # 时间信息流
        self.temporal_buffer = deque(maxlen=20)

    def process(
        self,
        sensory_inputs: list[torch.Tensor],
        state: torch.Tensor = None,
        arousal: float = 0.5,
    ) -> dict:
        """处理信息流

        Args:
            sensory_inputs: 感觉输入张量列表
            state: 当前状态张量
            arousal: 当前唤醒度水平 [0, 1]
        """
        # 中继 (使用唤醒度调制)
        relayed = self.relay.relay(sensory_inputs, arousal=arousal)

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

    def get_timing_info(self) -> dict:
        """获取时间信息"""
        return {
            'buffer_length': len(self.temporal_buffer),
            'time_since_start': len(self.temporal_buffer),
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
    'EnhancedAmygdala',
    'vmPFCExtinction',
    'ExtinctionMemory',
    'AmygdalaHippocampusConnection',
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

    def forward(self, fast_feat: torch.Tensor) -> dict:
        h = torch.relu(self.fc1(fast_feat))
        learned_map = torch.sigmoid(self.fc2(h).view(-1, 1, 14, 14))
        attention_map = (1 - self.prior_strength) * learned_map + self.prior_strength * self.face_prior.view(1, 1, 14, 14)
        return {'attention_map': attention_map, 'prior_strength': self.prior_strength}


# ══════════════════════════════════════════════════════
# vmPFC恐惧消退机制 (新增)
# 参考: Milad & Quirk (2012) - vmPFC在恐惧消退中的作用
# ══════════════════════════════════════════════════════

@dataclass
class ExtinctionMemory:
    """消退记忆

    存储"安全"信号的记忆，用于抑制恐惧反应
    """
    cue: np.ndarray
    safety_context: str
    extinction_strength: float
    extinction_trials: int
    timestamp: int


class vmPFCExtinction(nn.Module):
    """腹内侧前额叶(vmPFC)恐惧消退学习

    vmPFC的功能:
    1. 存储恐惧消退记忆 (safe context)
    2. 抑制杏仁核的恐惧反应
    3. 检测安全信号，输出消退信号

    参考:
    - Milad & Quirk (2012): vmPFC的恐惧消退机制
    - Sotres-Bayon et al. (2009): 消退学习的神经回路
    """

    def __init__(
        self,
        input_dim: int = 64,
        extinction_rate: float = 0.05,
        consolidation_rate: float = 0.02,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.extinction_rate = extinction_rate
        self.consolidation_rate = consolidation_rate

        # 安全信号检测网络
        self.safety_detector = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 消退信号输出网络
        self.extinction_output = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 消退记忆存储
        self.extinction_memories: list[ExtinctionMemory] = []
        self.extinction_history: deque = deque(maxlen=50)

        # 当前消退强度
        self.current_extinction_signal = 0.0

    def detect_safety(
        self,
        context: torch.Tensor,
    ) -> float:
        """检测安全信号

        Args:
            context: 当前情境编码

        Returns:
            safety_level: 安全水平 [0, 1]
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)
        safety_level = self.safety_detector(context).squeeze().item()
        return safety_level

    def learn_extinction(
        self,
        fear_cue: np.ndarray,
        safety_context: str = "default",
        trial_count: int = 1,
    ) -> dict:
        """学习恐惧消退

        在安全暴露后，逐渐降低对恐惧cue的反应

        Args:
            fear_cue: 原恐惧条件刺激
            safety_context: 安全情境标识
            trial_count: 安全暴露次数

        Returns:
            extinction_result: 消退学习结果
        """
        # 检查是否已有消退记忆
        existing = None
        for mem in self.extinction_memories:
            if np.allclose(mem.cue, fear_cue, atol=0.1):
                existing = mem
                break

        if existing:
            # 增强现有消退记忆
            existing.extinction_strength += self.extinction_rate * trial_count
            existing.extinction_strength = min(1.0, existing.extinction_strength)
            existing.extinction_trials += trial_count
        else:
            # 创建新消退记忆
            new_mem = ExtinctionMemory(
                cue=fear_cue,
                safety_context=safety_context,
                extinction_strength=self.extinction_rate * trial_count,
                extinction_trials=trial_count,
                timestamp=len(self.extinction_memories),
            )
            self.extinction_memories.append(new_mem)

        # 记录消退历史
        self.extinction_history.append({
            'trial_count': trial_count,
            'safety_context': safety_context,
        })

        return {
            'extinction_strength': existing.extinction_strength if existing else self.extinction_rate * trial_count,
            'total_trials': existing.extinction_trials if existing else trial_count,
        }

    def output_extinction_signal(
        self,
        context: torch.Tensor,
        fear_memories: list[FearCondition],
    ) -> float:
        """输出消退信号，抑制杏仁核

        Args:
            context: 当前情境
            fear_memories: 杏仁核的恐惧记忆列表

        Returns:
            extinction_signal: 消退信号强度 [0, 1]
        """
        safety_level = self.detect_safety(context)

        # 计算消退强度：基于安全信号和消退记忆
        extinction_strength = safety_level

        # 检查是否有匹配的消退记忆
        context_np = context.squeeze(0).detach().cpu().numpy() if context.dim() > 1 else context.detach().cpu().numpy()
        for mem in self.extinction_memories:
            if np.allclose(mem.cue, context_np, atol=0.1):
                extinction_strength = max(extinction_strength, mem.extinction_strength)

        # vmPFC输出消退信号
        extinction_output = self.extinction_output(context).squeeze().item()
        self.current_extinction_signal = extinction_output * extinction_strength

        return self.current_extinction_signal

    def consolidate_extinction(self):
        """巩固消退记忆

        类似睡眠期间的记忆巩固
        """
        for mem in self.extinction_memories:
            mem.extinction_strength += self.consolidation_rate
            mem.extinction_strength = min(1.0, mem.extinction_strength)

    def get_extinction_stats(self) -> dict:
        """获取消退统计"""
        return {
            'n_extinction_memories': len(self.extinction_memories),
            'current_extinction_signal': self.current_extinction_signal,
            'avg_extinction_strength': np.mean([m.extinction_strength for m in self.extinction_memories]) if self.extinction_memories else 0.0,
            'total_extinction_trials': sum(m.extinction_trials for m in self.extinction_memories),
        }


# ══════════════════════════════════════════════════════
# 杏仁核-海马情绪调制连接 (新增)
# 参考: Richter-Levin & Akirav (2010)
# ══════════════════════════════════════════════════════

class AmygdalaHippocampusConnection(nn.Module):
    """杏仁核-海马情绪调制连接

    双向连接:
    1. Amygdala->HC: 情绪增强记忆编码 (情绪性记忆更强)
    2. HC->Amygdala: 情景记忆触发情绪反应 (回忆引发情绪)

    参考:
    - Richter-Levin & Akirav (2010): Amygdala-Hippocampus交互
    - McGaugh (2004): 情绪增强记忆
    """

    def __init__(
        self,
        amygdala_dim: int = 64,
        hippocampus_dim: int = 128,
        modulation_strength: float = 0.5,
    ):
        super().__init__()
        self.amygdala_dim = amygdala_dim
        self.hippocampus_dim = hippocampus_dim

        # Amygdala->HC: 情绪调制记忆编码强度
        self.emotion_to_memory = nn.Sequential(
            nn.Linear(amygdala_dim, hippocampus_dim),
            nn.Tanh(),  # 调制信号 [-1, 1]
        )

        # HC->Amygdala: 记忆触发情绪
        self.memory_to_emotion = nn.Sequential(
            nn.Linear(hippocampus_dim, amygdala_dim),
            nn.ReLU(),
        )

        # 调制强度
        self.modulation_strength = nn.Parameter(torch.tensor(modulation_strength))

        # 连接历史
        self.connection_history: deque = deque(maxlen=50)

    def modulate_memory_encoding(
        self,
        emotion_encoding: torch.Tensor,
        arousal: float,
        valence: float,
    ) -> torch.Tensor:
        """情绪调制记忆编码

        高唤醒/强情绪 -> 记忆编码更强

        Args:
            emotion_encoding: 杏仁核情绪编码 [amygdala_dim]
            arousal: 唤醒水平 [0, 1]
            valence: 效价 [-1, 1]

        Returns:
            memory_modulation: 记忆编码调制信号 [hippocampus_dim]
        """
        if emotion_encoding.dim() == 1:
            emotion_encoding = emotion_encoding.unsqueeze(0)

        # 情绪调制信号
        modulation = self.emotion_to_memory(emotion_encoding)

        # 基于唤醒和效价调整强度
        # 高唤醒增强，负效价也增强 (恐惧记忆更强)
        intensity_factor = arousal * (1.0 + 0.3 * abs(valence))
        modulation = modulation * intensity_factor * self.modulation_strength

        # 记录连接
        self.connection_history.append({
            'direction': 'emotion_to_memory',
            'arousal': arousal,
            'valence': valence,
        })

        return modulation

    def trigger_emotion_from_memory(
        self,
        memory_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """记忆触发情绪反应

        回忆特定情景 -> 触发相关情绪

        Args:
            memory_encoding: 海马记忆编码 [hippocampus_dim]

        Returns:
            emotion_signal: 情绪触发信号 [amygdala_dim]
        """
        if memory_encoding.dim() == 1:
            memory_encoding = memory_encoding.unsqueeze(0)

        # 记忆->情绪
        emotion_signal = self.memory_to_emotion(memory_encoding)
        emotion_signal = emotion_signal * self.modulation_strength

        # 记录连接
        self.connection_history.append({
            'direction': 'memory_to_emotion',
        })

        return emotion_signal

    def get_connection_stats(self) -> dict:
        """获取连接统计"""
        emotion_to_mem = sum(1 for h in self.connection_history if h['direction'] == 'emotion_to_memory')
        mem_to_emotion = sum(1 for h in self.connection_history if h['direction'] == 'memory_to_emotion')

        return {
            'modulation_strength': self.modulation_strength.item(),
            'emotion_to_memory_count': emotion_to_mem,
            'memory_to_emotion_count': mem_to_emotion,
        }


# ══════════════════════════════════════════════════════
# 增强Amygdala类 - 整合vmPFC和HC连接
# ══════════════════════════════════════════════════════

class EnhancedAmygdala(Amygdala):
    """增强版杏仁核 - 整合vmPFC消退和HC连接"""

    def __init__(
        self,
        input_dim: int = 64,
        hippocampus_dim: int = 128,
    ):
        super().__init__(input_dim)
        self.hippocampus_dim = hippocampus_dim

        # vmPFC消退系统
        self.vmpfc = vmPFCExtinction(input_dim)

        # HC连接
        self.hc_connection = AmygdalaHippocampusConnection(
            amygdala_dim=input_dim,
            hippocampus_dim=hippocampus_dim,
        )

        # 消退抑制状态
        self.extinction_suppression = 0.0

    def process_with_extinction(
        self,
        state: torch.Tensor,
        store: bool = True,
        apply_extinction: bool = True,
    ) -> dict:
        """处理情绪 + vmPFC消退调制

        Args:
            state: 输入状态
            store: 是否存储情绪记忆
            apply_extinction: 是否应用vmPFC消退

        Returns:
            result: 处理结果
        """
        # 基础情绪处理
        result = super().process(state, store)

        # vmPFC消退信号
        if apply_extinction and self.fear.fear_memories:
            extinction_signal = self.vmpfc.output_extinction_signal(
                state, self.fear.fear_memories
            )

            # 消退信号抑制恐惧反应
            if result['emotion'] == 'fear':
                # 消退调制：降低恐惧强度
                fear_suppression = extinction_signal * 0.5
                result['arousal'] = result['arousal'] * (1.0 - fear_suppression)
                result['intensity'] = result['intensity'] * (1.0 - fear_suppression)

                # 如果消退足够强，改变反应
                if extinction_signal > 0.7:
                    result['response'] = 'calm'

            self.extinction_suppression = extinction_signal
            result['extinction_signal'] = extinction_signal

        return result

    def get_enhanced_summary(self) -> dict:
        """获取增强版摘要"""
        base_summary = self.get_emotion_summary()
        return {
            **base_summary,
            'vmpfc_stats': self.vmpfc.get_extinction_stats(),
            'hc_connection_stats': self.hc_connection.get_connection_stats(),
            'extinction_suppression': self.extinction_suppression,
        }


# ══════════════════════════════════════════════════════
# 更新LimbicSystem以包含vmPFC
# ══════════════════════════════════════════════════════


class LimbicSystem(nn.Module):
    """
    边缘系统整合

    Amygdala + Thalamus + vmPFC (新增)
    """

    def __init__(
        self,
        input_dim: int = 64,
        hippocampus_dim: int = 128,
        event_bus=None,
    ):
        super().__init__()

        self.amygdala = EnhancedAmygdala(input_dim, hippocampus_dim)
        self.thalamus = Thalamus(input_dim)
        self.hippocampus_dim = hippocampus_dim

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "sensory_process",
                self._handle_sensory_process,
                priority=0,
                name="limbic",
            )

    def _handle_sensory_process(self, event) -> dict:
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
        if result["emotion"] == "fear" and result["arousal"] > 0.7:
            state["emotion_criticality"] = True
        return result

    def step(self, *args, **kwargs) -> dict:
        """One simulation step — delegates to forward()."""
        return self.forward(*args, **kwargs)

    @staticmethod
    def required_keys() -> list[str]:
        """Keys this region reads from the shared state."""
        return ["state_tensor", "hippo_retrieval", "dopamine_level"]

    @staticmethod
    def output_keys() -> list[str]:
        """Keys this region writes to the shared state."""
        return ["limbic_emotion", "limbic_arousal", "limbic_valence",
                "limbic_fear", "limbic_reward", "limbic_stress"]

    def forward(
        self,
        state: torch.Tensor,
        sensory_inputs: list[torch.Tensor] = None,
        apply_extinction: bool = True,
    ) -> dict:
        """整合处理"""
        emotion_result = self.amygdala.process_with_extinction(
            state, apply_extinction=apply_extinction,
        )
        arousal = emotion_result.get("arousal", 0.5)
        if sensory_inputs is not None:
            thalamus_result = self.thalamus.process(sensory_inputs, state, arousal=arousal)
        else:
            thalamus_result = {"relay_output": state, "attention_weights": None}
        return {
            "emotion": emotion_result.get("emotion", "neutral"),
            "valence": emotion_result.get("valence", 0.0),
            "arousal": arousal,
            "response": emotion_result.get("response", "calm"),
            "emotional_attention": emotion_result.get("emotional_attention", 0.5),
            "amygdala_output": emotion_result,
            "thalamus_output": thalamus_result,
        }

