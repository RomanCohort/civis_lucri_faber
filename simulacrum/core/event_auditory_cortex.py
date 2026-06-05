"""
听觉系统 - 事件驱动 (Spiking) 版本

Simulacrum - Event-Driven Auditory Cortex

基于 SNN 架构:
1. Gammatone → Spiking Encoder
2. IHC → LIF 脉冲
3. 皮层 → Spiking Layers
4. 事件驱动计算
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.snn_core import SpikingLayer

# ============ 事件驱动耳蜗 ============

class SpikingGammatone(nn.Module):
    """
    Spiking Gammatone 滤波器组

    将音频转换为脉冲事件
    """
    def __init__(self, n_filters: int = 64, sample_rate: int = 16000,
                 min_freq: float = 20, max_freq: float = 20000):
        super().__init__()
        self.n_filters = n_filters
        self.sample_rate = sample_rate

        # 中心频率
        freqs = np.geomspace(min_freq, max_freq, n_filters)
        self.register_buffer('center_freqs', torch.tensor(freqs, dtype=torch.float32))

        # Gammatone 参数
        self.n = 4
        self.b = 1.019

        # 可学习参数
        self.bandwidth_factor = nn.Parameter(torch.ones(n_filters) * 0.5)
        self.gain = nn.Parameter(torch.ones(n_filters))

        # 侧抑制
        self._init_lateral_inhibition()

    def _init_lateral_inhibition(self):
        inh = torch.eye(self.n_filters) * 0.8
        for i in range(1, self.n_filters):
            if i > 0:
                inh[i, i-1] = 0.3
            if i < self.n_filters - 1:
                inh[i, i+1] = 0.3
        self.register_buffer('lateral_inhibit', inh)

    def _gammatone_kernel(self, cf: float, length: int = 256) -> torch.Tensor:
        device = self.center_freqs.device
        t = torch.arange(length, dtype=torch.float32, device=device) / self.sample_rate
        env = (t ** (self.n - 1)) * torch.exp(-2 * np.pi * self.b * t)
        carrier = torch.cos(2 * np.pi * cf * t)
        kernel = env * carrier
        kernel = kernel / (kernel.abs().sum() + 1e-8)
        return kernel

    def forward(self, audio: torch.Tensor) -> dict:
        """
        Args:
            audio: [B, T]

        Returns:
            spikes: [B, n_filters, T] 脉冲事件
            energies: [B, n_filters] 能量
        """
        B, T = audio.shape
        device = audio.device
        n = self.n_filters
        kernel_size = 256
        hop = 128

        # 时频分析
        outputs = []
        for i in range(n):
            cf = self.center_freqs[i].item()
            kernel = self._gammatone_kernel(cf, kernel_size).to(device)

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

        # 转换为脉冲 (阈值化)
        threshold = tf_rep.mean(dim=(1, 2), keepdim=True) * 0.5
        spikes = (tf_rep > threshold).float()

        # 能量
        energies = tf_rep.mean(-1)

        return {
            'spikes': spikes,
            'energies': energies,
            'tf_representation': tf_rep,
        }


class SpikingIHC(nn.Module):
    """
    Spiking 内毛细胞模型

    半波整流 + 幂律压缩 → LIF脉冲
    """
    def __init__(self, n_filters: int = 64,
                 compression: float = 0.3,
                 threshold: float = 0.5):
        super().__init__()
        self.n_filters = n_filters
        self.compression = nn.Parameter(torch.tensor(compression))
        self.threshold = nn.Parameter(torch.tensor(threshold))

        # 膜电位
        self.register_buffer('v_mem', torch.zeros(n_filters))

    def forward(self, cochlear_output: torch.Tensor) -> dict:
        """
        Args:
            cochlear_output: [B, n_filters, T]

        Returns:
            spikes: [B, n_filters, T]
            rates: [B, n_filters]
        """
        # 半波整流 + 幂律压缩
        response = F.relu(cochlear_output)
        rate = torch.pow(response + 1e-8, self.compression)
        rate = rate / (1 + rate)

        # 转换为脉冲事件
        spikes = (rate > self.threshold).float()

        # 发放率
        rates = spikes.mean(-1)

        return {
            'spikes': spikes,
            'rates': rates,
            'ihc_output': rate,
        }


class SpikingSubcortical(nn.Module):
    """
    Spiking 下丘/MGN

    门控 + 增益 + 脉冲编码
    """
    def __init__(self, n_filters: int = 64):
        super().__init__()
        self.n_filters = n_filters

        # 门控网络
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

    def forward(self, input_spikes: torch.Tensor) -> dict:
        """
        Args:
            input_spikes: [B, n_filters]

        Returns:
            gated_spikes: [B, n_filters]
        """
        gate = self.gate_net(input_spikes)
        gain = self.gain_net(input_spikes)

        gated = input_spikes * gate * (gain * 2)

        return {
            'gated_spikes': gated,
            'gate': gate,
            'gain': gain,
        }


class SpikingA1(nn.Module):
    """
    Spiking 初级听觉皮层

    时序卷积 → LIF脉冲
    """
    def __init__(self, n_filters: int = 64, hidden_dim: int = 64):
        super().__init__()

        # 卷积特征提取
        self.conv = nn.Sequential(
            nn.Conv1d(n_filters, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # LIF 神经元
        self.lif = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(2)
        ])

        # 膜电位
        self.register_buffer('v_mem', torch.zeros(hidden_dim))

    def forward(self, subcortical_spikes: torch.Tensor) -> dict:
        """
        Args:
            subcortical_spikes: [B, n_filters, T]

        Returns:
            a1_spikes: [B, hidden_dim]
            features: [B, hidden_dim]
        """
        B = subcortical_spikes.shape[0]

        # 卷积
        features = self.conv(subcortical_spikes)

        # 时间池化
        pooled = features.mean(-1)

        # LIF 发射
        for lif in self.lif:
            v = lif(pooled)
            spikes = (v > 0).float()
            pooled = pooled * (1 - spikes) + spikes * (-75e-3)

        return {
            'a1_spikes': pooled,
            'features': pooled,
        }


class SpikingVentralStream(nn.Module):
    """
    Spiking 腹侧流

    识别流: 音素 → 音节 → 词
    """
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()

        # 多专家 (3个)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            for _ in range(3)
        ])

        # 门控
        self.gate = nn.Linear(input_dim, 3)

        # LIF 层
        self.lif = SpikingLayer(input_dim, hidden_dim)

        # STG
        self.stg = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, a1_features: torch.Tensor) -> dict:
        gate_logits = self.gate(a1_features)
        top_idx = gate_logits.argmax(dim=-1)
        expert_out = self.experts[top_idx](a1_features)

        stg_out = self.stg(expert_out)

        # 脉冲化
        spikes = (stg_out > 0).float()

        return {
            'spikes': spikes,
            'lexical': stg_out,
            'stg_features': stg_out,
            'expert_used': top_idx.item(),
        }


class SpikingDorsalStream(nn.Module):
    """
    Spiking 背侧流

    空间定位 + 运动
    """
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()

        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            for _ in range(3)
        ])
        self.gate = nn.Linear(input_dim, 3)

        self.spatial = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.motor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, a1_features: torch.Tensor) -> dict:
        gate_logits = self.gate(a1_features)
        top_idx = gate_logits.argmax(dim=-1)
        expert_out = self.experts[top_idx](a1_features)

        spatial_out = self.spatial(expert_out)
        motor_out = self.motor(spatial_out)

        # 脉冲化
        spikes = (motor_out > 0).float()

        return {
            'spikes': spikes,
            'spatial': spatial_out,
            'motor': motor_out,
            'expert_used': top_idx.item(),
        }


# ============ 完整事件驱动听觉系统 ============

class EventDrivenAuditoryCortex(nn.Module):
    """
    事件驱动听觉皮层

    全脉冲架构:
    1. Gammatone (时频分析)
    2. IHC (非线性)
    3. Subcortical (门控)
    4. A1 (特征提取)
    5. Ventral (识别)
    6. Dorsal (定位)
    """
    def __init__(self, sample_rate: int = 16000,
                 n_filters: int = 128):
        super().__init__()

        hidden_dim = 256

        # 外周
        self.gammatone = SpikingGammatone(n_filters, sample_rate, 20, 20000)
        self.ihc = SpikingIHC(n_filters)

        # 下丘/MGN
        self.subcortical = SpikingSubcortical(n_filters)

        # A1
        self.a1 = SpikingA1(n_filters, hidden_dim)

        # 腹侧流
        self.ventral = SpikingVentralStream(hidden_dim, hidden_dim)

        # 背侧流
        self.dorsal = SpikingDorsalStream(hidden_dim, hidden_dim)

        # 情感头
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, audio: torch.Tensor) -> dict:
        """
        Args:
            audio: [B, T]

        Returns:
            {spikes, features, ...}
        """
        # 1. Gammatone
        gammatone_out = self.gammatone(audio)
        gammatone_spikes = gammatone_out['spikes']

        # 2. IHC
        ihc_out = self.ihc(gammatone_out['tf_representation'])
        ihc_spikes = ihc_out['spikes']

        # 3. 下丘
        # 时间平均
        subcortical_in = ihc_out['rates']
        subcortical_out = self.subcortical(subcortical_in)
        gated_spikes = subcortical_out['gated_spikes']

        # 4. A1
        a1_out = self.a1(gammatone_spikes)
        a1_features = a1_out['features']

        # 5. 腹侧流
        ventral = self.ventral(a1_features)

        # 6. 背侧流
        dorsal = self.dorsal(a1_features)

        # 7. 情感
        emotion = self.emotion_head(a1_features)

        return {
            # 脉冲
            'gammatone_spikes': gammatone_spikes,
            'ihc_spikes': ihc_spikes,
            'gated_spikes': gated_spikes,
            'a1_spikes': a1_out['a1_spikes'],
            'ventral_spikes': ventral['spikes'],
            'dorsal_spikes': dorsal['spikes'],

            # 特征
            'features': a1_features,
            'lexical': ventral['lexical'],
            'spatial': dorsal['spatial'],
            'motor': dorsal['motor'],

            # 情感
            'valence': torch.tanh(emotion[:, 0]),
            'arousal': torch.sigmoid(emotion[:, 1]),
            'dominance': torch.sigmoid(emotion[:, 2]),
            'pleasantness': torch.sigmoid(emotion[:, 3]),

            # 能量
            'energies': gammatone_out['energies'],
        }


def create_event_auditory(
    sample_rate: int = 16000,
    n_filters: int = 128,
) -> EventDrivenAuditoryCortex:
    return EventDrivenAuditoryCortex(sample_rate, n_filters)


# ============ 测试 ============

if __name__ == "__main__":
    print("=== Testing Event-Driven Auditory ===\n")

    # 测试脉冲Gammatone
    print("[1] Spiking Gammatone")
    gammatone = SpikingGammatone(64, 16000, 20, 20000)
    audio = torch.randn(1, 16000)
    result = gammatone(audio)
    print(f"  - spikes: {result['spikes'].shape}")
    print(f"  - sparsity: {(result['spikes'] > 0).float().mean():.4f}")

    # 测试IHC
    print("\n[2] Spiking IHC")
    ihc = SpikingIHC(64)
    output = ihc(result['tf_representation'])
    print(f"  - spikes: {output['spikes'].shape}")
    print(f"  - rates: {output['rates'].shape}")

    # 测试完整系统
    print("\n[3] Event-Driven Auditory")
    cortex = create_event_auditory(16000, 64)
    audio = torch.randn(2, 16000)
    result = cortex(audio)
    print(f"  - features: {result['features'].shape}")
    print(f"  - lexical: {result['lexical'].shape}")
    print(f"  - motor: {result['motor'].shape}")
    print(f"  - valence: {result['valence'].item():.3f}")

    print("\nAll tests passed!")
