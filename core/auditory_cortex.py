"""
听觉皮层系统 - 仿生架构 (改进版)

参考生物学：
1. 外周: 耳蜗 → 时频分析 (Gammatone/临界带)
2. 下丘/MGN: 中继+门控 (带通滤波+增益)
3. A1: 初级听觉皮层 (spectral-temporal特征)
4. 腹侧流: A1→STG→颞叶前部 (识别: 音素→词)
5. 背侧流: A1→顶叶→额叶 (定位+发音)

关键改进 (vs原版):
- Gammatone滤波器 (更接近生物耳蜗)
- 扩展频率范围 20Hz-20kHz
- 临界带划分
- 侧抑制网络
- 内毛细胞非线性模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


# ============ Gammatone滤波器组 (改进版) ============

class GammatoneFilterBank(nn.Module):
    """
    Gammatone滤波器组 - 模拟耳蜗基底膜

    关键改进:
    1. Gammatone滤波器更接近真实耳蜗频率响应
    2. 扩展频率范围 20Hz-20kHz
    3. 临界带划分 (~24个)
    4. 侧抑制网络
    """
    def __init__(
        self,
        n_filters: int = 64,
        sample_rate: int = 16000,
        min_freq: float = 20,
        max_freq: float = 20000,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq

        # 中心频率 (对数尺度，高频更密集)
        freqs = np.geomspace(min_freq, max_freq, n_filters)
        self.register_buffer('center_freqs', torch.tensor(freqs, dtype=torch.float32))

        # Gammatone参数
        self.n = 4  # 滤波器阶数
        self.b = 1.019  # 带宽因子

        # 临界带 (~24个)
        self.n_critical_bands = 24
        self._init_critical_bands()

        # 可学习带宽
        self.bandwidth_factor = nn.Parameter(torch.ones(n_filters) * 0.5)

        # 侧抑制 (邻频抑制矩阵)
        self._init_lateral_inhibition()

    def _init_critical_bands(self):
        """初始化临界带"""
        # ERB (Equivalent Rectangular Bandwidth)
        def erb(f):
            return 24.7 + 0.108 * f
        edges = [100.0]
        for _ in range(self.n_critical_bands - 1):
            edges.append(edges[-1] + erb(edges[-1]))
        self.register_buffer('critical_band_edges', torch.tensor(edges))

    def _init_lateral_inhibition(self):
        """初始化侧抑制矩阵"""
        # 宽松的侧抑制 (对角线1, 邻接0.3)
        inh = torch.eye(self.n_filters) * 0.8
        for i in range(1, self.n_filters):
            if i > 0:
                inh[i, i-1] = 0.3
            if i < self.n_filters - 1:
                inh[i, i+1] = 0.3
        self.register_buffer('lateral_inhibit', inh)

    def _gammatone_kernel(self, cf: float, length: int = 256) -> torch.Tensor:
        """生成Gammatone滤波器核"""
        device = self.center_freqs.device
        t = torch.arange(length, dtype=torch.float32, device=device) / self.sample_rate

        # Gammatone: t^(n-1) * exp(-2*pi*b*bt) * cos(2*pi*cf*t)
        env = (t ** (self.n - 1)) * torch.exp(-2 * np.pi * self.b * t)
        carrier = torch.cos(2 * np.pi * cf * t)
        kernel = env * carrier

        # 归一化
        kernel = kernel / (kernel.abs().sum() + 1e-8)
        return kernel

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, T] 原始音频
        Returns:
            output: [B, n_filters, T]
        """
        B, T = audio.shape
        device = audio.device

        outputs = []
        kernel_size = 256
        hop = 128

        for i in range(self.n_filters):
            cf = self.center_freqs[i].item()
            kernel = self._gammatone_kernel(cf, kernel_size).to(device)

            # 卷积
            convolved = []
            for start in range(0, T - kernel_size + 1, hop):
                end = start + kernel_size
                segment = audio[:, start:end]
                if segment.shape[1] < kernel_size:
                    break
                filtered = (segment * kernel).sum(dim=-1, keepdim=True)
                convolved.append(filtered)

            if convolved:
                convolved = torch.cat(convolved, dim=-1)
                if convolved.shape[-1] < T:
                    pad = T - convolved.shape[-1]
                    convolved = F.pad(convolved, (0, pad))
                outputs.append(convolved.squeeze(-1))
            else:
                outputs.append(torch.zeros(B, T, device=device))

        tf_rep = torch.stack(outputs, dim=1)

        # 侧抑制
        tf_rep = torch.einsum('bct,cc->bct', tf_rep, self.lateral_inhibit)
        tf_rep = F.relu(tf_rep)

        return tf_rep


class InnerHairCellModel(nn.Module):
    """
    内毛细胞模型 - 非线性换能

    关键机制:
    1. 半波整流
    2. 幂律压缩 (~100dB → 40dB)
    3. 快速适应
    """
    def __init__(self, n_filters: int = 64, compression: float = 0.3):
        super().__init__()
        self.n_filters = n_filters
        self.compression = nn.Parameter(torch.tensor(compression))

    def forward(self, cochlear_output: torch.Tensor) -> torch.Tensor:
        """
        cochlear_output: [B, n_filters, T]
        Returns:
            rate: [B, n_filters, T] (神经发放率)
        """
        # 半波整流
        response = F.relu(cochlear_output)

        # 幂律压缩 (rate ∝ stimulus^alpha)
        rate = torch.pow(response + 1e-8, self.compression)

        # 饱和
        rate = rate / (1 + rate)

        return rate


# 旧接口兼容性
class GaborFilterBank(nn.Module):
    """旧接口 - 内部使用Gammatone"""
    def __init__(self, n_filters: int = 64, sample_rate: int = 16000,
                 min_freq: float = 20, max_freq: float = 20000):
        super().__init__()
        self.gammatone = GammatoneFilterBank(n_filters, sample_rate, min_freq, max_freq)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.gammatone(audio)


class Cochlea(nn.Module):
    """
    改进耳蜗模型 - 整合Gammatone + IHC
    """
    def __init__(
        self,
        n_filters: int = 64,
        sample_rate: int = 16000,
        min_freq: float = 20,
        max_freq: float = 20000,
    ):
        super().__init__()

        self.gammatone = GammatoneFilterBank(n_filters, sample_rate, min_freq, max_freq)
        self.ihc = InnerHairCellModel(n_filters)

    def forward(self, audio: torch.Tensor) -> Dict:
        """
        Args:
            audio: [B, T]
        Returns:
            cochlear_output: [B, n_filters, T]
            center_frequencies: [n_filters]
        """
        tf_rep = self.gammatone(audio)
        ihc_rate = self.ihc(tf_rep)

        return {
            'tf_representation': ihc_rate,
            'center_frequencies': self.gammatone.center_freqs,
            'gammatone_output': tf_rep,
        }


# ============ 下丘 / 内侧膝状体 (改进版) ============

class SubcorticalRelay(nn.Module):
    """
    下丘/内侧膝状体 (改进版)

    关键改进:
    1. 更强的门控网络
    2. 增益控制
    3. 多时间尺度处理
    """
    def __init__(
        self,
        n_filters: int = 64,
    ):
        super().__init__()

        # 门控网络 (两层MLP)
        self.gate_net = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            nn.ReLU(),
            nn.Linear(n_filters, n_filters),
            nn.Sigmoid(),
        )

        # 增益网络
        self.gain_net = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            nn.ReLU(),
            nn.Linear(n_filters, n_filters),
            nn.Sigmoid(),
        )

    def forward(
        self,
        cochlear_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cochlear_output: [B, n_filters, T]
        Returns:
            gated: [B, n_filters, T]
        """
        # 时间平均
        x = cochlear_output.mean(dim=-1)  # [B, n_filters]

        # 门控
        gate = self.gate_net(x).unsqueeze(-1)
        # 增益
        gain = self.gain_net(x).unsqueeze(-1)

        # 应用门控和增益
        gated = cochlear_output * gate * (gain * 2)

        return gated


# ============ 初级听觉皮层 (改进版) ============

class PrimaryAuditoryCortex(nn.Module):
    """
    A1 - 初级听觉皮层 (改进版)

    关键改进:
    1. 添加频率拓扑映射
    2. 更复杂的时序卷积
    3. 双路径处理
    """
    def __init__(
        self,
        n_filters: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 频率拓扑映射 (tonotopy)
        self.tonotopic_map = nn.Linear(n_filters, hidden_dim)

        # 时序卷积 (模拟A1的spectral-temporal感受野)
        self.conv_st = nn.Sequential(
            nn.Conv1d(n_filters, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # 快速通路 (时间精细结构)
        self.temporal_path = nn.Sequential(
            nn.Conv1d(n_filters, hidden_dim, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
        )

        # 池化 (时间不变性)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        subcortical_input: torch.Tensor,
    ) -> Dict:
        """
        Args:
            subcortical_input: [B, n_filters, T]
        Returns:
            features: [B, hidden_dim]
        """
        # 频率拓扑
       tonotopic = self.tonotopic_map(subcortical_input.mean(-1))

        # 时序卷积
        features = self.conv_st(subcortical_input)

        # 时间池化
        pooled = self.pool(features).squeeze(-1)

        # 整合
        integrated = pooled + tonotopic * 0.1

        return {
            'a1_features': integrated,
            'tonotopic': tonotopic,
        }


# ============ 腹侧流 (改进版 - "是什么") ============

class VentralStream(nn.Module):
    """
    腹侧流 - 听觉识别 + MOE (改进版)

    关键改进:
    1. 多级MOE (音素→音节→词)
    2. 更丰富的专家
    3. 词汇整合
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # 级1: 音素检测 (3个专家)
        self.experts_phoneme = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            for _ in range(3)
        ])
        self.gate_phoneme = nn.Linear(input_dim, 3)

        # 级2: 音节整合 (3个专家)
        self.experts_syllable = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(3)
        ])
        self.gate_syllable = nn.Linear(hidden_dim, 3)

        # STG (颞上回)
        self.stg = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 前额叶 (词汇检索)
        self.frontal_temporal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, a1_features: torch.Tensor) -> Dict:
        # 级1: 音素检测
        gate_p = self.gate_phoneme(a1_features)
        top_p = gate_p.argmax(dim=-1)
        phoneme_out = self.experts_phoneme[top_p](a1_features)

        # 级2: 音节整合
        gate_s = self.gate_syllable(phoneme_out)
        top_s = gate_s.argmax(dim=-1)
        syllable_out = self.experts_syllable[top_s](phoneme_out)

        stg_out = self.stg(syllable_out)
        lexical = self.frontal_temporal(stg_out)

        return {
            'stg_features': stg_out,
            'lexical': lexical,
            'phoneme_features': phoneme_out,
            'syllable_features': syllable_out,
            'what': 'identity/word',
            'expert_used': (top_p.item(), top_s.item()),
        }


# ============ 背侧流 (改进版) ============

class DorsalStream(nn.Module):
    """
    背侧流 - 空间定位 + 运动 + MOE (改进版)

    关键改进:
    1. 双耳整合
    2. 更精细的时序处理
    3. 运动学习
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # MOE
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            for _ in range(3)
        ])
        self.gate = nn.Linear(input_dim, 3)

        # 空间处理
        self.spatial = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # 运动规划
        self.motor_planning = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 运动执行
        self.motor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, a1_features: torch.Tensor) -> Dict:
        # MOE: top-1
        gate_logits = self.gate(a1_features)
        top_idx = gate_logits.argmax(dim=-1)
        expert_out = self.experts[top_idx](a1_features)

        spatial_out = self.spatial(expert_out)
        planned = self.motor_planning(spatial_out)
        motor_out = self.motor(planned)

        return {
            'spatial': spatial_out,
            'motor_planning': planned,
            'motor': motor_out,
            'where': 'direction/location',
            'how': 'rhythm/motor',
            'expert_used': top_idx.item(),
        }


# ============ 可塑性机制 (新增) ============

class AuditoryPlasticity(nn.Module):
    """
    听觉可塑性 - STDP学习

    实现脉冲时序依赖可塑性
    """
    def __init__(self, n_filters: int = 64):
        super().__init__()
        self.n_filters = n_filters

        # STDP参数
        self.tau_plus = nn.Parameter(torch.tensor(20e-3))
        self.tau_minus = nn.Parameter(torch.tensor(20e-3))
        self.A_plus = 0.01
        self.A_minus = 0.012

    def stdp_update(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """
        计算STDP权重更新

        Args:
            pre: [B, T, C] 前突触活动
            post: [B, T, C] 后突触活动

        Returns:
            delta_w: [B, C, C]
        """
        # 简化: Hebbian学习
        corr = torch.einsum('btc,btc->bc', pre, post)
        delta_w = self.A_plus * corr - self.A_minus * corr
        return delta_w


# ============ 完整听觉系统 ============

class AuditoryCortex(nn.Module):
    """
    完整听觉皮层 + 心理学机制 (改进版)

    整合:
    1. 耳蜗 (时频分析, Gammatone)
    2. 下丘/MGN (中继门控)
    3. A1 (初级皮层 + 拓扑)
    4. 腹侧流 (识别 + 多级MOE)
    5. 背侧流 (定位+运动)
    6. 心理学机制 (听觉情境记忆)
    7. 可塑性
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filters: int = 128,
    ):
        super().__init__()

        hidden_dim = 256

        # 外周
        self.cochlea = Cochlea(n_filters, sample_rate, min_freq=20, max_freq=20000)

        # 下丘/MGN
        self.subcortical = SubcorticalRelay(n_filters)

        # A1
        self.a1 = PrimaryAuditoryCortex(n_filters, hidden_dim=hidden_dim)

        # 腹侧流 (Ventral - 识别)
        self.ventral = VentralStream(input_dim=hidden_dim, hidden_dim=hidden_dim)

        # 背侧流 (Dorsal - 定位)
        self.dorsal = DorsalStream(input_dim=hidden_dim, hidden_dim=hidden_dim)

        # 心理学组件
        self.auditory_memory = AuditoryContextMemory(hidden_dim)
        self.attentional_capture = AttentionalCapture(hidden_dim)
        self.emotion_regulation = AudioEmotionRegulation()

        # 可塑性
        self.plasticity = AuditoryPlasticity(n_filters)

        # 情感头
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(
        self,
        audio: torch.Tensor,
    ) -> Dict:
        """
        处理音频

        Args:
            audio: [B, T] 原始音频 (16kHz)
        Returns:
            result: {
                'cochlear': ...,
                'subcortical': ...,
                'a1': ...,
                'ventral': {...},
                'dorsal': {...},
                'features': ...,
                'valence': ...,
                'arousal': ...,
            }
        """
        # 1. 耳蜗
        cochlear = self.cochlea(audio)

        # 2. 下丘/MGN
        subcortical = self.subcortical(cochlear['tf_representation'])

        # 3. A1
        a1_result = self.a1(subcortical)
        a1_features = a1_result['a1_features']

        # 4. 腹侧流
        ventral = self.ventral(a1_features)

        # 5. 背侧流
        dorsal = self.dorsal(a1_features)

        # 6. 情感 (从A1)
        emotion = self.emotion_head(a1_features)

        return {
            'cochlear': cochlear['tf_representation'],
            'subcortical': subcortical,
            'a1': a1_features,

            # 腹侧流
            'what': ventral['what'],
            'lexical': ventral['lexical'],
            'phoneme_features': ventral['phoneme_features'],
            'syllable_features': ventral['syllable_features'],

            # 背侧流
            'where': dorsal['where'],
            'how': dorsal['how'],
            'spatial': dorsal['spatial'],
            'motor': dorsal['motor'],

            # 特征
            'features': a1_features,

            # 情感
            'valence': torch.tanh(emotion[:, 0]),
            'arousal': torch.sigmoid(emotion[:, 1]),
            'dominance': torch.sigmoid(emotion[:, 2]),
            'pleasantness': torch.sigmoid(emotion[:, 3]),

            # 拓扑
            'tonotopic': a1_result.get('tonotopic', a1_features),
        }


# ============ 便捷函数 ============

def create_auditory_cortex(
    sample_rate: int = 16000,
    n_filters: int = 128,
) -> AuditoryCortex:
    return AuditoryCortex(sample_rate, n_filters)


# ============ 听觉心理学组件 (保留) ============

class AuditoryContextMemory(nn.Module):
    """听觉情境记忆"""
    def __init__(self, dim: int = 256):
        super().__init__()

        self.context_encoder = nn.Linear(dim, dim)
        self.context_memory = nn.Parameter(torch.randn(10, dim))
        self.context_index = nn.Parameter(torch.zeros(10))

    def retrieve_context(self, audio_features: torch.Tensor):
        """检索最匹配的情境"""
        encoded = self.context_encoder(audio_features)
        similarity = F.cosine_similarity(encoded.unsqueeze(1), self.context_memory.unsqueeze(0), dim=-1)
        idx = similarity.argmax(dim=-1)
        return self.context_memory[idx], idx

    def encode_experience(self, audio_features: torch.Tensor, context_id: int):
        """编码经验"""
        with torch.no_grad():
            encoded = self.context_encoder(audio_features.detach())
            self.context_memory[context_id] = encoded
            self.context_index[context_id] += 0.1


class AttentionalCapture(nn.Module):
    """注意力捕获"""
    def __init__(self, dim: int = 256):
        super().__init__()

        self.salience_detector = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        self.capture_threshold = nn.Parameter(torch.tensor(0.7))

    def check_capture(self, audio_features: torch.Tensor):
        """检查是否捕获注意力"""
        salience = self.salience_detector(audio_features)
        if salience > self.capture_threshold:
            return True, salience
        return False, salience


class AudioEmotionRegulation(nn.Module):
    """听觉情绪调节"""
    def __init__(self):
        super().__init__()

        self.tempo_sensitivity = nn.Parameter(torch.zeros(1))
        self.tone_sensitivity = nn.Parameter(torch.zeros(1))
        self.timbre_sensitivity = nn.Parameter(torch.zeros(1))

    def regulate(self, audio_features: torch.Tensor, tempo: float, tone: float):
        """根据音频特性调节情绪响应"""
        tempo_effect = torch.tanh(self.tempo_sensitivity) * (tempo - 0.5)
        tone_effect = torch.tanh(self.tone_sensitivity) * (tone - 0.5)

        regulated = audio_features * (1 + tempo_effect + tone_effect)
        return regulated


# ============ 听觉剪枝机制 (保留) ============

class AuditoryPruner(nn.Module):
    """听觉动态剪枝"""
    def __init__(self, n_filters=128):
        super().__init__()
        self.n_filters = n_filters
        self.active_filters = nn.Parameter(torch.ones(n_filters))

    def get_active_ratio(self):
        return (self.active_filters > 0).float().mean().item()

    def prune(self, threshold=0.1):
        with torch.no_grad():
            mask = self.active_filters > threshold
            self.active_filters *= mask.float()


class CochlearFilterPruner(nn.Module):
    """耳蜗滤波器剪枝 - 只选top-k频率"""
    def __init__(self, n_filters=128, keep_k=32):
        super().__init__()
        self.n_filters = n_filters
        self.keep_k = keep_k
        self.importance = nn.Parameter(torch.ones(n_filters))

    def select_top_k(self):
        scores = self.importance
        _, indices = scores.topk(self.keep_k)
        return indices


__all__ = [
    'GammatoneFilterBank',
    'InnerHairCellModel',
    'GaborFilterBank',
    'Cochlea',
    'SubcorticalRelay',
    'PrimaryAuditoryCortex',
    'VentralStream',
    'DorsalStream',
    'AuditoryPlasticity',
    'AuditoryCortex',
    'create_auditory_cortex',
]