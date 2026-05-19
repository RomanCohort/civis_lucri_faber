"""
仿生发音语言系统 (Bio-Inspired Vocalization System)

模拟人类从语言意图到声学输出的完整发声通路：

生物学基础：
  语言皮层 → Broca区 → 运动皮层 → 声道肌肉
     ↕                                    ↕
  听觉反馈 ← 声学输出 ← 声道滤波 ← 声带振动 ← 呼吸气流

核心模块：
1. VocalTract          — 声道发音器官模型（舌/唇/颌/软腭/声门）
2. ArticulatoryPlanner — 音素序列 → 发音器运动轨迹（CPG驱动）
3. FormantSynthesizer  — 发音器姿态 → 共振峰声学特征
4. SpeechProductionPipeline — 完整语音产生管线
5. VocalCortex         — 事件驱动的发声皮层协调器

通路（对应真实解剖）：
  LanguageCortex (颞叶语义) → ArticulatoryPlanner (Broca区)
    → VocalTract (运动皮层→面部/咽喉运动神经元)
    → FormantSynthesizer (声学输出，类比喉部声带+声道滤波)
    ← auditory feedback → ArticulatoryPlanner (听觉-运动校正)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


def _get_waveform_synthesizer():
    """延迟导入共振峰→波形合成器（避免循环依赖）"""
    try:
        from simulacrum.core.formant_synthesis import FormantToWaveform
    except ImportError:
        from core.formant_synthesis import FormantToWaveform
    return FormantToWaveform


# ============ ARPAbet音素表（与phonetic_perception.py对齐） ============

PHONEMES = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ay',
    'b', 'ch', 'd', 'dh',
    'eh', 'er', 'ey',
    'f',
    'g',
    'hh',
    'ih', 'iy',
    'jh',
    'k',
    'l',
    'm',
    'n', 'ng',
    'ow', 'oy',
    'p',
    'r',
    's', 'sh',
    't', 'th',
    'uh', 'uw',
    'v',
    'w',
    'y',
    'z', 'zh',
]

N_PHONEMES = len(PHONEMES)
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEMES)}

# 发音器维度定义（5个自由度，生物真实）
ARTICULATOR_DIMS = [
    'tongue_tip',       # 舌尖前后（-1后 ~ +1前）
    'tongue_body',      # 舌体上下（-1低 ~ +1高）
    'lip_spread',       # 唇展开度（-1圆唇 ~ +1展唇）
    'jaw_open',         # 下颌开合（-1闭 ~ +1开）
    'velum_open',       # 软腭开合（-1鼻腔关闭 ~ +1鼻腔开放）
]
N_ARTICULATORS = len(ARTICULATOR_DIMS)


# ============ 数据类 ============

@dataclass
class VocalizationOutput:
    """发声系统输出"""
    phoneme_sequence: List[str]           # 音素序列
    articulator_trajectory: np.ndarray    # [T, 5] 发音器轨迹
    formant_values: np.ndarray            # [T, 3] F1/F2/F3 共振峰
    acoustic_features: np.ndarray         # [T, 64] 声学特征向量
    voicing: np.ndarray                   # [T] 浊音/清音标记
    intensity: float                      # 整体响度 [0,1]
    duration_ms: float                    # 预估持续时间(毫秒)


# ============ 1. 声道发音器官模型 ============

class VocalTract(nn.Module):
    """
    声道发音器官物理模型

    模拟5个主要发音器官的运动学：
    - 舌尖(tongue_tip): 前后运动（齿龈音/卷舌音控制）
    - 舌体(tongue_body): 上下运动（元音高度、软腭音）
    - 唇(lip_spread): 圆唇/展唇（圆唇元音、双唇辅音）
    - 下颌(jaw): 开合（开口度、辅音阻碍）
    - 软腭(velum): 开合（鼻音/口音切换）

    生物学对应：面神经(VII) + 三叉神经(V) + 舌下神经(XII)
    控制下颌、舌、唇的肌肉群
    """

    def __init__(
        self,
        n_articulators: int = N_ARTICULATORS,
        hidden_dim: int = 64,
        dynamics_smooth: float = 0.3,   # 运动平滑系数（惯性模拟）
    ):
        super().__init__()
        self.n_articulators = n_articulators
        self.dynamics_smooth = dynamics_smooth

        # 目标姿态 → 实际姿态（带惯性的二阶动力学）
        self.position_encoder = nn.Sequential(
            nn.Linear(n_articulators, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_articulators),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

        # 运动速度（一阶导数）
        self.velocity_encoder = nn.Sequential(
            nn.Linear(n_articulators * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_articulators),
            nn.Tanh(),
        )

        # 肌肉协同模式（synergies）：减少独立自由度
        # 生物学中，5个发音器由约100块肌肉控制，
        # 但大脑通过"运动基元"将其简化为少量协同模式
        self.n_synergies = 8
        self.synergy_matrix = nn.Parameter(
            torch.randn(n_articulators, self.n_synergies) * 0.1
        )
        self.synergy_decoder = nn.Sequential(
            nn.Linear(self.n_synergies, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_articulators),
            nn.Tanh(),
        )

        # 当前状态
        self._position = torch.zeros(n_articulators)
        self._velocity = torch.zeros(n_articulators)

    def forward(
        self,
        target_positions: torch.Tensor,
        dt: float = 0.01,
    ) -> Dict:
        """
        推进发音器一步运动

        Args:
            target_positions: [B, N_ARTICULATORS] 目标姿态（来自ArticulatoryPlanner）
            dt: 时间步长

        Returns:
            current_position: [B, N_ARTICULATORS] 当前实际位置
            velocity: [B, N_ARTICULATORS] 当前速度
            target_error: [B] 到达目标的误差
        """
        B = target_positions.shape[0]

        # 编码目标
        encoded_target = self.position_encoder(target_positions)

        # 肌肉协同投影（降维后再解码，模拟运动基元）
        synergy_activation = torch.matmul(encoded_target, self.synergy_matrix)
        synergized_target = self.synergy_decoder(synergy_activation)

        # 二阶动力学：position += velocity * dt; velocity += acceleration * dt
        # 加速度 = (target - current) * spring - damping * velocity
        spring_force = 20.0   # 弹簧刚度（肌肉张力）
        damping = 4.0         # 阻尼（粘弹性）

        position_input = torch.cat([
            self._position.unsqueeze(0).expand(B, -1),
            self._velocity.unsqueeze(0).expand(B, -1)
        ], dim=-1)

        acceleration = spring_force * (synergized_target - self._position.unsqueeze(0).expand(B, -1)) \
                      - damping * self._velocity.unsqueeze(0).expand(B, -1)

        new_velocity = self._velocity.unsqueeze(0).expand(B, -1) + acceleration * dt
        new_position = self._position.unsqueeze(0).expand(B, -1) + new_velocity * dt

        # 夹紧到 [-1, 1]
        new_position = torch.clamp(new_position, -1.0, 1.0)
        new_velocity = torch.clamp(new_velocity, -2.0, 2.0)

        # 更新内部状态（取batch均值作为全局状态）
        self._position = new_position.mean(dim=0).detach()
        self._velocity = new_velocity.mean(dim=0).detach()

        target_error = torch.norm(synergized_target - new_position, dim=-1)

        return {
            'current_position': new_position,     # [B, N_ARTICULATORS]
            'velocity': new_velocity,              # [B, N_ARTICULATORS]
            'target_error': target_error,          # [B]
            'synergy_activation': synergy_activation,  # [B, n_synergies]
        }


# ============ 2. 发音运动计划器（CPG驱动） ============

class ArticulatoryPlanner(nn.Module):
    """
    发音运动计划器

    功能：音素序列 → 发音器时间轨迹

    生物学对应：
    - Broca区（BA44/45）：音素序列编码
    - pre-SMA：运动序列计划
    - 基底神经节→小脑回路：CPG产生节律性发音模式
    - 丘脑：感觉反馈中继

    关键机制：
    1. 音素到发音器目标姿态的映射（learned）
    2. CPG驱动的节律性发音模式（音节节奏）
    3. 协同发音（coarticulation）：相邻音素间的平滑过渡
    4. 听觉-运动校正：基于听觉反馈的实时修正
    """

    def __init__(
        self,
        n_phonemes: int = N_PHONEMES,
        n_articulators: int = N_ARTICULATORS,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        cpg_frequency: float = 4.0,     # 基础音节节奏 ~4 Hz
    ):
        super().__init__()
        self.n_phonemes = n_phonemes
        self.n_articulators = n_articulators

        # 音素嵌入（learned）
        self.phoneme_embedding = nn.Embedding(n_phonemes, embedding_dim)

        # 音素 → 发音器目标姿态映射（静态音素目标）
        self.phoneme_to_target = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_articulators),
            nn.Tanh(),
        )

        # CPG振荡器（用于音节节奏）
        # 每个发音器一个相位振荡器，耦合形成协同模式
        self.cpg_frequency = cpg_frequency
        self.cpg_phases = nn.Parameter(torch.zeros(n_articulators))
        self.cpg_amplitudes = nn.Parameter(torch.ones(n_articulators) * 0.3)
        self.cpg_coupling = nn.Parameter(
            torch.eye(n_articulators) * 0.2
        )

        # 协同发音模型（上下文感知的轨迹平滑）
        # 生物学：发音器运动有惯性，不可能瞬间跳变
        # 模型：BiLSTM捕捉音素序列上下文，输出平滑轨迹
        self.coarticulation = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_articulators),
            nn.Tanh(),
        )

        # 听觉反馈校正网络
        # 生物学：听觉皮层→Broca区反馈回路，用于在线修正发音
        self.feedback_corrector = nn.Sequential(
            nn.Linear(n_articulators + 64, hidden_dim),  # articulators + acoustic
            nn.ReLU(),
            nn.Linear(hidden_dim, n_articulators),
            nn.Tanh(),
        )
        self.feedback_gain = nn.Parameter(torch.tensor(0.3))

        # 时间编码
        self.time_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.SiLU(),
        )

    def _cpg_tick(self, dt: float = 0.01) -> torch.Tensor:
        """
        CPG振荡器推进一步

        Returns: [n_articulators] 当前CPG输出值
        """
        phases = self.cpg_phases.data
        # 相邻振荡器耦合
        coupling = torch.matmul(
            torch.sin(phases), self.cpg_coupling
        )
        new_phases = phases + (self.cpg_frequency * 2 * np.pi * dt + coupling) * dt
        self.cpg_phases.data = new_phases % (2 * np.pi)

        return self.cpg_amplitudes * torch.sin(self.cpg_phases.data)

    def forward(
        self,
        phoneme_indices: torch.Tensor,        # [B, T] 音素索引
        respiratory_phase: Optional[torch.Tensor] = None,  # [B] 呼吸相位
        auditory_feedback: Optional[torch.Tensor] = None,  # [B, 64] 听觉反馈
        dt: float = 0.01,
    ) -> Dict:
        """
        生成发音器运动轨迹

        Args:
            phoneme_indices: [B, T] 音素索引序列
            respiratory_phase: [B] 呼吸相位（0=呼气开始，1=吸气开始）
                               语音产生需要在呼气阶段
            auditory_feedback: [B, 64] 来自听觉皮层的反馈信号
            dt: 时间步长

        Returns:
            target_trajectory: [B, T, N_ARTICULATORS] 发音器目标轨迹
            cpg_output: [B, T, N_ARTICULATORS] CPG节律调制
            phoneme_embeddings: [B, T, emb_dim] 音素嵌入
        """
        B, T = phoneme_indices.shape

        # 1. 音素嵌入
        emb = self.phoneme_embedding(phoneme_indices)  # [B, T, emb_dim]

        # 2. 协同发音：BiLSTM捕捉上下文，输出平滑目标轨迹
        context_out, _ = self.coarticulation(emb)  # [B, T, hidden*2]
        target_base = self.trajectory_decoder(context_out)  # [B, T, n_art]

        # 3. CPG节律调制（音节节奏，约4Hz）
        cpg_values = []
        for t in range(T):
            cpg_tick = self._cpg_tick(dt)  # [n_art]
            cpg_values.append(cpg_tick)
        cpg_output = torch.stack(cpg_values, dim=0).unsqueeze(0).expand(B, -1, -1)  # [B, T, n_art]

        # CPG调制叠加到目标轨迹上
        target_trajectory = target_base + cpg_output * 0.15

        # 4. 呼吸门控：语音仅在呼气阶段产生
        if respiratory_phase is not None:
            # respiratory_phase: 0~1, 语音在0~0.6区间（呼气阶段）
            speech_gate = torch.sigmoid(
                (respiratory_phase.unsqueeze(-1).unsqueeze(-1) - 0.3) * 10.0
            )
            target_trajectory = target_trajectory * (1.0 - speech_gate)  # 呼气时激活

        # 5. 听觉-运动反馈校正
        if auditory_feedback is not None:
            last_pos = target_trajectory[:, -1, :]  # [B, n_art]
            feedback_input = torch.cat([last_pos, auditory_feedback], dim=-1)
            correction = self.feedback_corrector(feedback_input)  # [B, n_art]
            target_trajectory = target_trajectory + self.feedback_gain * correction.unsqueeze(1)

        # 夹紧
        target_trajectory = torch.clamp(target_trajectory, -1.0, 1.0)

        return {
            'target_trajectory': target_trajectory,   # [B, T, n_art]
            'cpg_output': cpg_output,                  # [B, T, n_art]
            'phoneme_embeddings': emb,                 # [B, T, emb_dim]
        }


# ============ 3. 共振峰声学合成器 ============

class FormantSynthesizer(nn.Module):
    """
    共振峰声学合成器

    将发音器姿态映射为声学特征（共振峰频率）

    生物学原理：
    - 声源-滤波器理论（Fant, 1960）：
      声带振动（声源） → 声道共振（滤波器） → 辐射输出
    - 声道形状决定共振峰（F1~F3）
    - 声带频率决定基频F0（音高）
    - 鼻腔耦合产生额外鼻共振峰

    共振峰-发音器关系（经验映射）：
    - F1（第一共振峰）↔ 下颌开合 + 舌体高度
    - F2（第二共振峰）↔ 舌体前后位置
    - F3（第三共振峰）↔ 舌尖/卷舌
    """

    def __init__(
        self,
        n_articulators: int = N_ARTICULATORS,
        hidden_dim: int = 64,
        n_formants: int = 3,         # F1, F2, F3
        output_dim: int = 64,        # 最终声学特征维度
    ):
        super().__init__()
        self.n_formants = n_formants

        # 发音器姿态 → 三个共振峰频率
        # F1范围: 200-1000 Hz, F2: 800-2600 Hz, F3: 2000-3500 Hz
        self.formant_predictor = nn.Sequential(
            nn.Linear(n_articulators, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_formants),
        )

        # 声源参数预测（F0基频 + 气声度）
        self.source_predictor = nn.Sequential(
            nn.Linear(n_articulators, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # [F0, breathiness]
        )

        # 共振峰 → 声学特征向量
        self.acoustic_encoder = nn.Sequential(
            nn.Linear(n_formants + 2 + n_articulators, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # 清/浊音判断
        self.voicing_predictor = nn.Sequential(
            nn.Linear(n_articulators, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 鼻音检测（软腭打开 = 鼻音）
        self.nasality_detector = nn.Sequential(
            nn.Linear(n_articulators, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # 共振峰平滑（模拟声道惯性）
        self._prev_formants = torch.zeros(n_formants)
        self.smoothing = 0.7

    def forward(self, articulator_positions: torch.Tensor) -> Dict:
        """
        Args:
            articulator_positions: [B, T, N_ARTICULATORS] 或 [B, N_ARTICULATORS]

        Returns:
            formants: [B, T, 3] F1/F2/F3 (Hz, normalized)
            acoustic_features: [B, T, 64] 完整声学特征
            voicing: [B, T, 1] 浊音概率
            nasality: [B, T, 1] 鼻音概率
            f0: [B, T, 1] 基频
        """
        squeeze = False
        if articulator_positions.dim() == 2:
            articulator_positions = articulator_positions.unsqueeze(1)
            squeeze = True

        B, T, _ = articulator_positions.shape
        flat = articulator_positions.reshape(-1, articulator_positions.shape[-1])

        # 共振峰预测
        raw_formants = self.formant_predictor(flat)  # [B*T, 3]
        # 用sigmoid + 范围映射确保在真实频率范围内
        formants_norm = torch.sigmoid(raw_formants)   # [0, 1]
        # F1: 200-1000, F2: 800-2600, F3: 2000-3500 Hz
        freq_ranges = torch.tensor(
            [[200, 800, 2000]], device=raw_formants.device
        )
        freq_scales = torch.tensor(
            [[800, 1800, 1500]], device=raw_formants.device
        )
        formants_hz = formants_norm * freq_scales + freq_ranges  # [B*T, 3]

        # 归一化到 [0, 1] 用于声学特征
        formants_normalized = (formants_hz - freq_ranges) / freq_scales

        # 声源参数
        source_params = torch.sigmoid(self.source_predictor(flat))  # [B*T, 2]
        f0_norm = source_params[:, 0:1]    # 基频归一化
        breathiness = source_params[:, 1:2]  # 气声度

        # 清/浊音 & 鼻音
        voicing = self.voicing_predictor(flat)   # [B*T, 1]
        nasality = self.nasality_detector(flat)  # [B*T, 1]

        # 拼接所有特征 → 声学特征向量
        acoustic_input = torch.cat([
            formants_normalized,    # 3
            f0_norm, breathiness,   # 2
            flat,                   # n_articulators
        ], dim=-1)
        acoustic_features = self.acoustic_encoder(acoustic_input)  # [B*T, 64]

        # Reshape回去
        formants_hz = formants_hz.reshape(B, T, 3)
        acoustic_features = acoustic_features.reshape(B, T, 64)
        voicing = voicing.reshape(B, T, 1)
        nasality = nasality.reshape(B, T, 1)
        f0_norm = f0_norm.reshape(B, T, 1)

        if squeeze:
            formants_hz = formants_hz.squeeze(1)
            acoustic_features = acoustic_features.squeeze(1)
            voicing = voicing.squeeze(1)
            nasality = nasality.squeeze(1)
            f0_norm = f0_norm.squeeze(1)

        return {
            'formants': formants_hz,          # [B, T, 3] Hz
            'acoustic_features': acoustic_features,  # [B, T, 64]
            'voicing': voicing,               # [B, T, 1]
            'nasality': nasality,             # [B, T, 1]
            'f0': f0_norm,                    # [B, T, 1]
        }


# ============ 4. 完整语音产生管线 ============

class SpeechProductionPipeline(nn.Module):
    """
    完整语音产生管线

    整合：语言意图 → 运动计划 → 声道执行 → 声学输出

    生物学通路（Dorsal stream + Ventral stream）：
    颞叶语言区（Wernicke）→ 弓状束 → Broca区（ArticulatoryPlanner）
      → 运动皮层 → 面神经/舌下神经 → VocalTract肌肉
      → 声带振动 + 声道滤波 → FormantSynthesizer
      → 听觉皮层（反馈）→ 校正回路

    参数共享：
    - 呼吸节律来自 Brainstem.RespiratoryRhythmGenerator
    - 音素序列来自 LanguageCortex 或 PhoneticPerception
    - 听觉反馈来自 AuditoryCortex
    """

    def __init__(
        self,
        n_phonemes: int = N_PHONEMES,
        n_articulators: int = N_ARTICULATORS,
        acoustic_dim: int = 64,
    ):
        super().__init__()

        # 子模块
        self.planner = ArticulatoryPlanner(
            n_phonemes=n_phonemes,
            n_articulators=n_articulators,
        )
        self.vocal_tract = VocalTract(
            n_articulators=n_articulators,
        )
        self.synthesizer = FormantSynthesizer(
            n_articulators=n_articulators,
            output_dim=acoustic_dim,
        )

        # 发声持续时间预测（每个音素的持续时间）
        self.duration_predictor = nn.Sequential(
            nn.Linear(64, 32),  # 输入：语言特征
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),     # 确保正数
        )

        # 响度控制（与情绪/意图耦合）
        self.intensity_controller = nn.Sequential(
            nn.Linear(64 + 1, 32),  # acoustic + arousal
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        phoneme_indices: torch.Tensor,              # [B, T_phonemes]
        language_features: Optional[torch.Tensor] = None,  # [B, 64]
        respiratory_phase: Optional[torch.Tensor] = None,   # [B]
        auditory_feedback: Optional[torch.Tensor] = None,   # [B, 64]
        arousal: float = 0.5,
        dt: float = 0.01,
    ) -> Dict:
        """
        执行完整的语音产生流程

        Args:
            phoneme_indices: 音素索引序列
            language_features: 来自LanguageCortex的特征
            respiratory_phase: 呼吸相位
            auditory_feedback: 听觉反馈
            arousal: 唤醒水平（影响响度）
            dt: 时间步长

        Returns:
            VocalizationOutput的所有字段
        """
        B, T_phonemes = phoneme_indices.shape

        # 1. 运动计划：音素 → 目标轨迹
        planner_out = self.planner(
            phoneme_indices=phoneme_indices,
            respiratory_phase=respiratory_phase,
            auditory_feedback=auditory_feedback,
            dt=dt,
        )
        target_trajectory = planner_out['target_trajectory']  # [B, T, n_art]

        # 2. 声道执行：目标轨迹 → 实际轨迹（带惯性）
        T_frames = target_trajectory.shape[1]
        actual_positions = []
        for t in range(T_frames):
            tract_out = self.vocal_tract(
                target_positions=target_trajectory[:, t, :],
                dt=dt,
            )
            actual_positions.append(tract_out['current_position'])
        actual_trajectory = torch.stack(actual_positions, dim=1)  # [B, T, n_art]

        # 3. 声学合成：实际轨迹 → 声学特征
        synth_out = self.synthesizer(actual_trajectory)

        # 4. 响度控制
        if language_features is not None:
            intensity_input = torch.cat([
                synth_out['acoustic_features'][:, -1, :],  # 最后帧特征
                torch.tensor([[arousal]], device=language_features.device).expand(B, -1),
            ], dim=-1)
            intensity = self.intensity_controller(intensity_input).squeeze(-1)  # [B]
        else:
            intensity = torch.ones(B, device=phoneme_indices.device) * 0.5

        # 应用响度到声学特征
        scaled_acoustic = synth_out['acoustic_features'] * intensity.unsqueeze(-1).unsqueeze(-1)

        # 5. 构造输出
        phoneme_names = [PHONEMES[idx] for idx in phoneme_indices[0].cpu().numpy()]

        # 转换 numpy → python list，避免 JSON 序列化问题
        result = {
            'phoneme_sequence': phoneme_names,
            'articulator_trajectory': actual_trajectory[0].detach().cpu().numpy().tolist(),
            'formant_values': synth_out['formants'][0].detach().cpu().numpy().tolist(),
            'acoustic_features': scaled_acoustic[0].detach().cpu().numpy().tolist(),
            'voicing': synth_out['voicing'][0].detach().cpu().numpy().tolist(),
            'intensity': intensity[0].item(),
            'target_trajectory': target_trajectory,
            'actual_trajectory': actual_trajectory,
            'all_formants': synth_out['formants'],
            'all_acoustic': scaled_acoustic,
            'f0': synth_out['f0'],
            'nasality': synth_out['nasality'],
            'planner_state': planner_out,
        }
        return result


# ============ 5. 事件驱动的发声皮层协调器 ============

class VocalCortex(nn.Module):
    """
    发声皮层（Vocal Cortex）

    对应大脑皮层中控制发声的区域：
    - Broca区 (BA44/45): 语言产生、语法编码
    - 前运动皮层 (BA6): 发音运动序列计划
    - 初级运动皮层 (BA4): 发音器肌肉执行指令
    - 岛叶 (BA13): 发声的内脏感觉
    - 扣带回 (BA24): 发声动机/情绪驱动

    事件驱动设计：
    - 订阅 VOCALIZATION_CONTROL 事件
    - 订阅 BRAIN_UPDATE 事件（获取呼吸相位、情绪状态）
    - 订阅 SENSORY_PROCESS 事件（获取听觉反馈）
    - 发布 VOCALIZATION_OUTPUT 事件（声学输出）

    集成点：
    - 从 LanguageCortex 获取语言特征/音素序列
    - 从 Brainstem 获取呼吸相位
    - 从 AuditoryCortex 获取听觉反馈
    - 从 Limbic/Emotion 获取情绪调制（响度/语调）
    - 向 Agent.step() 输出声学特征
    """

    def __init__(
        self,
        n_phonemes: int = N_PHONEMES,
        acoustic_dim: int = 64,
        hidden_dim: int = 128,
        event_bus=None,
    ):
        super().__init__()

        # 核心管线
        self.pipeline = SpeechProductionPipeline(
            n_phonemes=n_phonemes,
            acoustic_dim=acoustic_dim,
        )

        # 语言特征→音素序列的转换器
        self.feature_to_phoneme = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_phonemes),
        )

        # 情绪调制参数
        self.emotion_modulator = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 4),    # [tempo_scale, pitch_scale, intensity_scale, roughness]
            nn.Sigmoid(),
        )

        # 语音运动学习（通过听觉反馈的误差修正）
        self.error_learner = nn.Sequential(
            nn.Linear(acoustic_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_phonemes),
        )

        # 内部状态
        self._last_acoustic = None
        self._vocalization_count = 0
        self._cumulative_duration_ms = 0.0
        self._is_speaking = False
        self._speech_onset_threshold = 0.3

        # 事件总线：在 __init__ 中订阅（与其他模块一致）
        self.event_bus = event_bus
        if event_bus is not None:
            event_bus.subscribe(
                "vocalization_control",
                self._handle_vocalization_control,
                priority=5,
                name="vocal_cortex",
            )

    def _handle_vocalization_control(self, event) -> Dict:
        """Event-driven handler for VOCALIZATION_CONTROL events.

        从事件中提取音素序列、呼吸、情绪参数，执行发声管线。
        """
        state = event.data
        phoneme_indices = state.get("phoneme_indices")
        if phoneme_indices is None:
            return {"is_speaking": False}

        respiratory_rate = state.get("respiratory_rate", 12.0)
        respiratory_phase = state.get("respiratory_phase", 0.5)
        arousal = state.get("arousal", 0.5)
        emotion_vector = state.get("emotion_vector")

        result = self.forward(
            phoneme_indices=phoneme_indices,
            respiratory_rate=respiratory_rate,
            respiratory_phase=respiratory_phase,
            emotion_vector=emotion_vector,
            arousal=arousal,
        )

        # 将发声状态写回 internal_state 供下游模块使用
        internal_state = state.get("internal_state", {})
        internal_state["vocal_is_speaking"] = result.get("is_speaking", False)
        internal_state["vocal_intensity"] = result.get("intensity", 0.0)

        return result

    def forward(
        self,
        phoneme_indices: Optional[torch.Tensor] = None,
        language_features: Optional[torch.Tensor] = None,
        respiratory_rate: float = 12.0,
        respiratory_phase: float = 0.5,
        emotion_vector: Optional[torch.Tensor] = None,  # [B, 8]
        auditory_feedback: Optional[torch.Tensor] = None,
        arousal: float = 0.5,
        dt: float = 0.01,
    ) -> Dict:
        """
        处理发声请求

        优先级：
        1. 直接提供音素序列 → 直接使用
        2. 提供语言特征 → 推断音素序列
        3. 都没有 → 静默

        Args:
            phoneme_indices: [B, T] 音素序列
            language_features: [B, 64] 来自LanguageCortex的特征
            respiratory_rate: 呼吸频率（次/分钟）
            respiratory_phase: 当前呼吸相位 [0, 1]
            emotion_vector: [B, 8] 情绪向量（Plutchik 8维）
            auditory_feedback: [B, 64] 听觉反馈
            arousal: 唤醒水平
            dt: 时间步长
        """
        B = 1
        if phoneme_indices is not None:
            B = phoneme_indices.shape[0]
        elif language_features is not None:
            B = language_features.shape[0]

        # 如果没有音素输入，尝试从语言特征推断
        if phoneme_indices is None and language_features is not None:
            logits = self.feature_to_phoneme(language_features)  # [B, n_phonemes]
            phoneme_indices = logits.argmax(dim=-1)  # [B]
            # 扩展为序列（每个特征产生一个音素）
            phoneme_indices = phoneme_indices.unsqueeze(1)  # [B, 1]

        if phoneme_indices is None:
            return {'is_speaking': False, 'acoustic_features': None}

        # 情绪调制
        tempo_scale, pitch_scale, intensity_scale, roughness = 1.0, 1.0, 1.0, 0.0
        if emotion_vector is not None:
            if emotion_vector.dim() == 1:
                emotion_vector = emotion_vector.unsqueeze(0)
            mod_params = self.emotion_modulator(emotion_vector)  # [B, 4]
            tempo_scale = mod_params[0, 0].item()
            pitch_scale = mod_params[0, 1].item()
            intensity_scale = mod_params[0, 2].item()
            roughness = mod_params[0, 3].item()

        # 有效唤醒（情绪调制后的）
        effective_arousal = arousal * intensity_scale

        # 有效呼吸相位（考虑情绪）
        # 高唤醒 → 呼吸加速，声门下压力增大
        effective_respiratory_phase = respiratory_phase

        # 调整时间步长（情绪影响语速）
        effective_dt = dt / tempo_scale

        # 执行管线
        output = self.pipeline(
            phoneme_indices=phoneme_indices,
            language_features=language_features,
            respiratory_phase=torch.tensor(
                [effective_respiratory_phase], device=phoneme_indices.device, dtype=torch.float32
            ).expand(B),
            auditory_feedback=auditory_feedback,
            arousal=effective_arousal,
            dt=effective_dt,
        )

        # 更新状态
        self._last_acoustic = output['acoustic_features']
        self._is_speaking = output['intensity'] > self._speech_onset_threshold
        self._vocalization_count += 1

        # 计算持续时间（基于音素数量和语速）
        n_phonemes = phoneme_indices.shape[-1]
        avg_duration_per_phoneme_ms = 80.0 / tempo_scale  # 典型80ms/音素
        estimated_duration_ms = n_phonemes * avg_duration_per_phoneme_ms
        self._cumulative_duration_ms += estimated_duration_ms

        output['duration_ms'] = estimated_duration_ms
        output['emotion_modulation'] = {
            'tempo_scale': tempo_scale,
            'pitch_scale': pitch_scale,
            'intensity_scale': intensity_scale,
            'roughness': roughness,
        }
        output['is_speaking'] = self._is_speaking
        output['vocalization_count'] = self._vocalization_count

        # 波形合成：共振峰 → 可听音频
        try:
            waveform_synth = _get_waveform_synthesizer()()
            # 反归一化 F0 到 Hz 范围
            f0_hz_min = 80.0 * pitch_scale
            f0_hz_max = 300.0 * pitch_scale

            # 确保所有数据已 detach + 转为 numpy
            formants_np = np.array(output['formant_values'], dtype=np.float64)
            f0_tensor = output['f0']
            if isinstance(f0_tensor, torch.Tensor):
                f0_tensor = f0_tensor.detach().cpu()
            voicing_tensor = output['voicing']
            if isinstance(voicing_tensor, torch.Tensor):
                voicing_tensor = voicing_tensor.detach().cpu()

            waveform = waveform_synth.synthesize(
                formants=formants_np,                  # [T, 3] Hz
                f0=f0_tensor,                           # [T, 1] 归一化
                voicing=voicing_tensor,                 # [T, 1]
                intensity=float(output['intensity']),
                f0_range=(f0_hz_min, f0_hz_max),
            )
            output['waveform'] = waveform
            output['waveform_sample_rate'] = waveform_synth.sample_rate
        except Exception as _waveform_err:
            import traceback
            traceback.print_exc()
            output['waveform'] = None
            output['waveform_sample_rate'] = None

        # 听觉-运动学习（误差修正）
        if auditory_feedback is not None and self._last_acoustic is not None:
            error_input = torch.cat([
                output['acoustic_features'][:, -1, :].unsqueeze(1),
                auditory_feedback.unsqueeze(1),
            ], dim=-1)
            correction_logits = self.error_learner(error_input.squeeze(1))
            output['learning_signal'] = correction_logits

        return output

    def get_state_summary(self) -> Dict:
        """获取发声系统状态摘要"""
        return {
            'is_speaking': self._is_speaking,
            'vocalization_count': self._vocalization_count,
            'cumulative_duration_ms': self._cumulative_duration_ms,
            'has_last_acoustic': self._last_acoustic is not None,
        }


# ============ 辅助：音素到IPA/ARPAbet映射 ============

PHONEME_DESCRIPTIONS = {
    'aa': 'father (ɑ)',     'ae': 'cat (æ)',       'ah': 'but (ʌ)',
    'ao': 'thought (ɔ)',    'aw': 'cow (aʊ)',      'ay': 'ride (aɪ)',
    'b':  'bat (b)',        'ch': 'church (tʃ)',   'd':  'dog (d)',
    'dh': 'this (ð)',       'eh': 'bed (ɛ)',       'er': 'bird (ɝ)',
    'ey': 'say (eɪ)',       'f':  'fan (f)',       'g':  'go (ɡ)',
    'hh': 'hat (h)',        'ih': 'sit (ɪ)',       'iy': 'see (iː)',
    'jh': 'judge (dʒ)',     'k':  'key (k)',       'l':  'led (l)',
    'm':  'mat (m)',        'n':  'no (n)',        'ng': 'sing (ŋ)',
    'ow': 'go (oʊ)',        'oy': 'boy (ɔɪ)',      'p':  'pat (p)',
    'r':  'red (ɹ)',        's':  'say (s)',       'sh': 'shoe (ʃ)',
    't':  'tea (t)',        'th': 'think (θ)',     'uh': 'book (ʊ)',
    'uw': 'two (uː)',       'v':  'van (v)',       'w':  'we (w)',
    'y':  'yes (j)',        'z':  'zoo (z)',       'zh': 'measure (ʒ)',
}


def describe_phoneme(phoneme: str) -> str:
    """获取音素的IPA描述"""
    return PHONEME_DESCRIPTIONS.get(phoneme, f'unknown ({phoneme})')


def text_to_phoneme_indices(text: str) -> List[int]:
    """
    文本→音素索引转换

    策略: 常见英语单词查 CMU 风格词典, 未知词逐字母回退。
    """
    # 精选高频英语词汇发音字典 (ARPAbet)
    _PRON_DICT: Dict[str, List[str]] = {
        'the': ['dh', 'ah'],
        'a': ['ah'],
        'an': ['ae', 'n'],
        'is': ['ih', 'z'],
        'it': ['ih', 't'],
        'to': ['t', 'uw'],
        'of': ['ah', 'v'],
        'and': ['ae', 'n', 'd'],
        'in': ['ih', 'n'],
        'that': ['dh', 'ae', 't'],
        'have': ['hh', 'ae', 'v'],
        'i': ['ay'],
        'you': ['y', 'uw'],
        'do': ['d', 'uw'],
        'at': ['ae', 't'],
        'on': ['aa', 'n'],
        'he': ['hh', 'iy'],
        'she': ['sh', 'iy'],
        'we': ['w', 'iy'],
        'be': ['b', 'iy'],
        'me': ['m', 'iy'],
        'my': ['m', 'ay'],
        'no': ['n', 'ow'],
        'so': ['s', 'ow'],
        'go': ['g', 'ow'],
        'up': ['ah', 'p'],
        'or': ['ao', 'r'],
        'if': ['ih', 'f'],
        'as': ['ae', 'z'],
        'not': ['n', 'aa', 't'],
        'are': ['aa', 'r'],
        'but': ['b', 'ah', 't'],
        'can': ['k', 'ae', 'n'],
        'this': ['dh', 'ih', 's'],
        'what': ['w', 'ah', 't'],
        'with': ['w', 'ih', 'dh'],
        'all': ['ao', 'l'],
        'will': ['w', 'ih', 'l'],
        'there': ['dh', 'eh', 'r'],
        'one': ['w', 'ah', 'n'],
        'about': ['ah', 'b', 'aw', 't'],
        'how': ['hh', 'aw'],
        'their': ['dh', 'eh', 'r'],
        'than': ['dh', 'ae', 'n'],
        'its': ['ih', 't', 's'],
        'would': ['w', 'uh', 'd'],
        'like': ['l', 'ay', 'k'],
        'very': ['v', 'eh', 'r', 'iy'],
        'much': ['m', 'ah', 'ch'],
        'just': ['jh', 'ah', 's', 't'],
        'hello': ['hh', 'eh', 'l', 'ow'],
        'hi': ['hh', 'ay'],
        'yes': ['y', 'eh', 's'],
        'thanks': ['th', 'ae', 'ng', 'k', 's'],
        'thank': ['th', 'ae', 'ng', 'k'],
        'please': ['p', 'l', 'iy', 'z'],
        'sorry': ['s', 'aa', 'r', 'iy'],
        'help': ['hh', 'eh', 'l', 'p'],
        'good': ['g', 'uh', 'd'],
        'bad': ['b', 'ae', 'd'],
        'love': ['l', 'ah', 'v'],
        'want': ['w', 'aa', 'n', 't'],
        'need': ['n', 'iy', 'd'],
        'know': ['n', 'ow'],
        'think': ['th', 'ih', 'ng', 'k'],
        'see': ['s', 'iy'],
        'come': ['k', 'ah', 'm'],
        'take': ['t', 'ey', 'k'],
        'make': ['m', 'ey', 'k'],
        'time': ['t', 'ay', 'm'],
        'day': ['d', 'ey'],
        'now': ['n', 'aw'],
        'new': ['n', 'uw'],
        'here': ['hh', 'iy', 'r'],
        'where': ['w', 'eh', 'r'],
        'when': ['w', 'eh', 'n'],
        'why': ['w', 'ay'],
        'who': ['hh', 'uw'],
        'work': ['w', 'er', 'k'],
        'give': ['g', 'ih', 'v'],
        'get': ['g', 'eh', 't'],
        'got': ['g', 'aa', 't'],
        'say': ['s', 'ey'],
        'said': ['s', 'eh', 'd'],
        'tell': ['t', 'eh', 'l'],
        'ask': ['ae', 's', 'k'],
        'feel': ['f', 'iy', 'l'],
        'well': ['w', 'eh', 'l'],
        'also': ['ao', 'l', 's', 'ow'],
        'thing': ['th', 'ih', 'ng'],
        'way': ['w', 'ey'],
        'many': ['m', 'eh', 'n', 'iy'],
        'some': ['s', 'ah', 'm'],
        'any': ['eh', 'n', 'iy'],
        'find': ['f', 'ay', 'n', 'd'],
        'back': ['b', 'ae', 'k'],
        'only': ['ow', 'n', 'l', 'iy'],
        'still': ['s', 't', 'ih', 'l'],
        'other': ['ah', 'dh', 'er'],
        'should': ['sh', 'uh', 'd'],
        'could': ['k', 'uh', 'd'],
        'people': ['p', 'iy', 'p', 'ah', 'l'],
        'world': ['w', 'er', 'l', 'd'],
        'right': ['r', 'ay', 't'],
        'left': ['l', 'eh', 'f', 't'],
        'big': ['b', 'ih', 'g'],
        'small': ['s', 'm', 'ao', 'l'],
        'first': ['f', 'er', 's', 't'],
        'last': ['l', 'ae', 's', 't'],
        'long': ['l', 'ao', 'ng'],
        'great': ['g', 'r', 'ey', 't'],
        'little': ['l', 'ih', 't', 'ah', 'l'],
        'old': ['ow', 'l', 'd'],
        'same': ['s', 'ey', 'm'],
        'another': ['ah', 'n', 'ah', 'dh', 'er'],
        'high': ['hh', 'ay'],
        'low': ['l', 'ow'],
        'man': ['m', 'ae', 'n'],
        'woman': ['w', 'uh', 'm', 'ah', 'n'],
        'child': ['ch', 'ay', 'l', 'd'],
        'hand': ['hh', 'ae', 'n', 'd'],
        'case': ['k', 'ey', 's'],
        'week': ['w', 'iy', 'k'],
        'company': ['k', 'ah', 'm', 'p', 'ah', 'n', 'iy'],
        'system': ['s', 'ih', 's', 't', 'ah', 'm'],
        'program': ['p', 'r', 'ow', 'g', 'r', 'ae', 'm'],
        'question': ['k', 'w', 'eh', 's', 'ch', 'ah', 'n'],
        'city': ['s', 'ih', 't', 'iy'],
        'earth': ['er', 'th'],
        'eye': ['ay'],
        'face': ['f', 'ey', 's'],
        'hair': ['hh', 'eh', 'r'],
        'heart': ['hh', 'aa', 'r', 't'],
        'talk': ['t', 'ao', 'k'],
        'speak': ['s', 'p', 'iy', 'k'],
        'read': ['r', 'iy', 'd'],
        'write': ['r', 'ay', 't'],
        'play': ['p', 'l', 'ey'],
        'run': ['r', 'ah', 'n'],
        'drink': ['d', 'r', 'ih', 'ng', 'k'],
        'sleep': ['s', 'l', 'iy', 'p'],
        'wake': ['w', 'ey', 'k'],
        'understand': ['ah', 'n', 'd', 'er', 's', 't', 'ae', 'n', 'd'],
        'remember': ['r', 'ih', 'm', 'eh', 'm', 'b', 'er'],
        'believe': ['b', 'ih', 'l', 'iy', 'v'],
        'really': ['r', 'iy', 'ah', 'l', 'iy'],
        'because': ['b', 'ih', 'k', 'ah', 'z'],
        'different': ['d', 'ih', 'f', 'er', 'ah', 'n', 't'],
        'important': ['ih', 'm', 'p', 'ao', 'r', 't', 'ah', 'n', 't'],
        'computer': ['k', 'ah', 'm', 'p', 'y', 'uw', 't', 'er'],
        'artificial': ['aa', 'r', 't', 'ah', 'f', 'ih', 'sh', 'ah', 'l'],
        'intelligence': ['ih', 'n', 't', 'eh', 'l', 'ih', 'jh', 'ah', 'n', 's'],
        'language': ['l', 'ae', 'ng', 'g', 'w', 'ih', 'jh'],
        'model': ['m', 'aa', 'd', 'ah', 'l'],
        'neural': ['n', 'uw', 'r', 'ah', 'l'],
        'network': ['n', 'eh', 't', 'w', 'er', 'k'],
        'data': ['d', 'ae', 't', 'ah'],
        'information': ['ih', 'n', 'f', 'er', 'm', 'ey', 'sh', 'ah', 'n'],
        'memory': ['m', 'eh', 'm', 'er', 'iy'],
        'attention': ['ah', 't', 'eh', 'n', 'sh', 'ah', 'n'],
        'emotion': ['ih', 'm', 'ow', 'sh', 'ah', 'n'],
        'feeling': ['f', 'iy', 'l', 'ih', 'ng'],
        'happy': ['hh', 'ae', 'p', 'iy'],
        'sad': ['s', 'ae', 'd'],
        'angry': ['ae', 'ng', 'r', 'iy'],
        'afraid': ['ah', 'f', 'r', 'ey', 'd'],
        'curious': ['k', 'y', 'uh', 'r', 'iy', 'ah', 's'],
        'tired': ['t', 'ay', 'er', 'd'],
        'excited': ['ih', 'k', 's', 'ay', 't', 'ih', 'd'],
        'calm': ['k', 'aa', 'm'],
        'certain': ['s', 'er', 't', 'ah', 'n'],
        'interesting': ['ih', 'n', 't', 'er', 'eh', 's', 't', 'ih', 'ng'],
        'funny': ['f', 'ah', 'n', 'iy'],
        'beautiful': ['b', 'y', 'uw', 't', 'ah', 'f', 'ah', 'l'],
        'strong': ['s', 't', 'r', 'ao', 'ng'],
        'fast': ['f', 'ae', 's', 't'],
        'slow': ['s', 'l', 'ow'],
        'hard': ['hh', 'aa', 'r', 'd'],
        'easy': ['iy', 'z', 'iy'],
        'true': ['t', 'r', 'uw'],
        'false': ['f', 'ao', 'l', 's'],
        'wrong': ['r', 'ao', 'ng'],
        'solution': ['s', 'ah', 'l', 'uw', 'sh', 'ah', 'n'],
        'answer': ['ae', 'n', 's', 'er'],
        'simple': ['s', 'ih', 'm', 'p', 'ah', 'l'],
        'possible': ['p', 'aa', 's', 'ah', 'b', 'ah', 'l'],
        'something': ['s', 'ah', 'm', 'th', 'ih', 'ng'],
        'nothing': ['n', 'ah', 'th', 'ih', 'ng'],
        'everything': ['eh', 'v', 'r', 'iy', 'th', 'ih', 'ng'],
        'sometimes': ['s', 'ah', 'm', 't', 'ay', 'm', 'z'],
        'today': ['t', 'ah', 'd', 'ey'],
        'tomorrow': ['t', 'ah', 'm', 'aa', 'r', 'ow'],
        'tonight': ['t', 'ah', 'n', 'ay', 't'],
        'morning': ['m', 'ao', 'r', 'n', 'ih', 'ng'],
        'evening': ['iy', 'v', 'n', 'ih', 'ng'],
        'water': ['w', 'ao', 't', 'er'],
        'food': ['f', 'uw', 'd'],
        'money': ['m', 'ah', 'n', 'iy'],
        'color': ['k', 'ah', 'l', 'er'],
        'dark': ['d', 'aa', 'r', 'k'],
        'sound': ['s', 'aw', 'n', 'd'],
        'game': ['g', 'ey', 'm'],
        'home': ['hh', 'ow', 'm'],
        'room': ['r', 'uw', 'm'],
        'door': ['d', 'ao', 'r'],
        'window': ['w', 'ih', 'n', 'd', 'ow'],
        'table': ['t', 'ey', 'b', 'ah', 'l'],
        'chair': ['ch', 'eh', 'r'],
        'street': ['s', 't', 'r', 'iy', 't'],
        'tree': ['t', 'r', 'iy'],
        'flower': ['f', 'l', 'aw', 'er'],
        'bird': ['b', 'er', 'd'],
        'fish': ['f', 'ih', 'sh'],
        'cat': ['k', 'ae', 't'],
        'dog': ['d', 'ao', 'g'],
        'human': ['hh', 'y', 'uw', 'm', 'ah', 'n'],
        'friend': ['f', 'r', 'eh', 'n', 'd'],
        'family': ['f', 'ae', 'm', 'ah', 'l', 'iy'],
        'story': ['s', 't', 'ao', 'r', 'iy'],
        'history': ['hh', 'ih', 's', 't', 'er', 'iy'],
        'idea': ['ay', 'd', 'iy', 'ah'],
        'future': ['f', 'y', 'uw', 'ch', 'er'],
        'past': ['p', 'ae', 's', 't'],
        'present': ['p', 'r', 'eh', 'z', 'ah', 'n', 't'],
        'power': ['p', 'aw', 'er'],
        'energy': ['eh', 'n', 'er', 'jh', 'iy'],
        'change': ['ch', 'ey', 'n', 'jh'],
        'move': ['m', 'uw', 'v'],
        'stop': ['s', 't', 'aa', 'p'],
        'try': ['t', 'r', 'ay'],
        'use': ['y', 'uw', 'z'],
        'let': ['l', 'eh', 't'],
        'hear': ['hh', 'iy', 'r'],
        'turn': ['t', 'er', 'n'],
        'might': ['m', 'ay', 't'],
        'call': ['k', 'ao', 'l'],
        'each': ['iy', 'ch'],
        'follow': ['f', 'aa', 'l', 'ow'],
        'came': ['k', 'ey', 'm'],
        'every': ['eh', 'v', 'r', 'iy'],
        'live': ['l', 'ih', 'v'],
        'after': ['ae', 'f', 't', 'er'],
        'then': ['dh', 'eh', 'n'],
        'them': ['dh', 'eh', 'm'],
        'wrote': ['r', 'ow', 't'],
        'made': ['m', 'ey', 'd'],
        'down': ['d', 'aw', 'n'],
        'been': ['b', 'ih', 'n'],
        'before': ['b', 'ih', 'f', 'ao', 'r'],
        'must': ['m', 'ah', 's', 't'],
        'through': ['th', 'r', 'uw'],
        'years': ['y', 'ih', 'r', 'z'],
        'mean': ['m', 'iy', 'n'],
        'side': ['s', 'ay', 'd'],
        'five': ['f', 'ay', 'v'],
        'since': ['s', 'ih', 'n', 's'],
        'far': ['f', 'aa', 'r'],
        'art': ['aa', 'r', 't'],
        'without': ['w', 'ih', 'dh', 'aw', 't'],
        'again': ['ah', 'g', 'eh', 'n'],
        'watch': ['w', 'aa', 'ch'],
        'few': ['f', 'uw'],
        'open': ['ow', 'p', 'ah', 'n'],
        'together': ['t', 'ah', 'g', 'eh', 'dh', 'er'],
        'white': ['w', 'ay', 't'],
        'children': ['ch', 'ih', 'l', 'd', 'r', 'ah', 'n'],
        'walk': ['w', 'ao', 'k'],
        'ease': ['iy', 'z'],
        'paper': ['p', 'ey', 'p', 'er'],
        'trees': ['t', 'r', 'iy', 'z'],
        'forest': ['f', 'ao', 'r', 'ah', 's', 't'],
        'sitting': ['s', 'ih', 't', 'ih', 'ng'],
        'car': ['k', 'aa', 'r'],
        'river': ['r', 'ih', 'v', 'er'],
    }

    _CHAR_TO_PHONEME = {
        'a': PHONEME_TO_IDX['ae'],
        'b': PHONEME_TO_IDX['b'],
        'c': PHONEME_TO_IDX['k'],
        'd': PHONEME_TO_IDX['d'],
        'e': PHONEME_TO_IDX['eh'],
        'f': PHONEME_TO_IDX['f'],
        'g': PHONEME_TO_IDX['g'],
        'h': PHONEME_TO_IDX['hh'],
        'i': PHONEME_TO_IDX['iy'],
        'j': PHONEME_TO_IDX['jh'],
        'k': PHONEME_TO_IDX['k'],
        'l': PHONEME_TO_IDX['l'],
        'm': PHONEME_TO_IDX['m'],
        'n': PHONEME_TO_IDX['n'],
        'o': PHONEME_TO_IDX['ao'],
        'p': PHONEME_TO_IDX['p'],
        'q': PHONEME_TO_IDX['k'],
        'r': PHONEME_TO_IDX['r'],
        's': PHONEME_TO_IDX['s'],
        't': PHONEME_TO_IDX['t'],
        'u': PHONEME_TO_IDX['uw'],
        'v': PHONEME_TO_IDX['v'],
        'w': PHONEME_TO_IDX['w'],
        'x': PHONEME_TO_IDX['k'],
        'y': PHONEME_TO_IDX['y'],
        'z': PHONEME_TO_IDX['z'],
    }

    tokens = text.lower().split()
    indices: List[int] = []

    for token in tokens:
        word = ''.join(ch for ch in token if ch.isalpha())
        if not word:
            continue

        if word in _PRON_DICT:
            for ph in _PRON_DICT[word]:
                indices.append(PHONEME_TO_IDX.get(ph, PHONEME_TO_IDX['ah']))
        else:
            for ch in word:
                indices.append(_CHAR_TO_PHONEME.get(ch, PHONEME_TO_IDX['ah']))

    return indices if indices else [PHONEME_TO_IDX['ah']]
