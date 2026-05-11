"""
听觉皮层系统 - 仿生架构

参考生物学：
1. 外周: 耳蜗 → 时频分析 (Gabor/小波)
2. 下丘/MGN: 中继+门控 (带通滤波+增益)
3. A1: 初级听觉皮层 (spectral-temporal特征)
4. 腹侧流: A1→STG→颞叶前部 (识别: 音素→词)
5. 背侧流: A1→顶叶→额叶 (定位+发音)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


# ============ 耳蜗模型 (Cochlea) ============

class GaborFilterBank(nn.Module):
    """
    Gabor滤波器组 - 模拟耳蜗基底膜

    时频分析：不同中心频率的带通滤波器
    简化版：使用卷积核
    """
    def __init__(
        self,
        n_filters: int = 64,
        sample_rate: int = 16000,
        min_freq: float = 100,
        max_freq: float = 8000,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.sample_rate = sample_rate

        # 中心频率 (对数尺度)
        freqs = np.geomspace(min_freq, max_freq, n_filters)
        self.register_buffer('center_freqs', torch.tensor(freqs))

        # 带宽
        self.bandwidths = torch.tensor([f / 2.0 for f in freqs])

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, T] 原始音频
        Returns:
            output: [B, n_filters, T]
        """
        B, T = audio.shape
        device = audio.device

        # 简化：使用STFT风格的频域分析
        # 先做短时傅里叶变换的简化版
        # 用固定的短窗口做卷积

        outputs = []
        for i in range(self.n_filters):
            cf = self.center_freqs[i].item()

            # 固定窗口大小
            kernel_size = 256
            hop = 128

            # 创建带通滤波器 (复数小波)
            t = torch.arange(kernel_size, device=device) - kernel_size // 2
            sigma = kernel_size / 8.0
            omega = 2 * np.pi * cf / self.sample_rate

            kernel = torch.exp(-t**2 / (2 * sigma**2)) * torch.cos(omega * t)
            kernel = kernel / kernel.abs().sum() + 1e-8

            # 短时卷积 (1D卷积)
            # 步进式处理
            convolved = []
            for start in range(0, T - kernel_size + 1, hop):
                end = start + kernel_size
                segment = audio[:, start:end]  # [B, kernel_size]
                if segment.shape[1] < kernel_size:
                    break
                filtered = (segment * kernel).sum(dim=-1, keepdim=True)
                convolved.append(filtered)

            if convolved:
                convolved = torch.cat(convolved, dim=-1)
                # 插值回原始长度
                if convolved.shape[-1] < T:
                    pad_size = T - convolved.shape[-1]
                    convolved = F.pad(convolved, (0, pad_size))
                outputs.append(convolved.squeeze(-1))
            else:
                outputs.append(torch.zeros(B, T, device=device))

        return torch.stack(outputs, dim=1)


class Cochlea(nn.Module):
    """
    耳蜗模型

    将时域音频转换为时频表示
    """
    def __init__(
        self,
        n_filters: int = 64,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self.gabor = GaborFilterBank(n_filters, sample_rate)

        # 非线性压缩 (模拟毛细胞饱和)
        self.compression = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU(),
        )

    def forward(self, audio: torch.Tensor) -> Dict:
        """
        Args:
            audio: [B, T]
        Returns:
            cochlear_output: [B, n_filters, T]
        """
        # Gabor滤波
        tf_rep = self.gabor(audio)

        # 模拟毛细胞非线性和半波整流
        tf_rep = F.relu(tf_rep)
        tf_rep = torch.log1p(tf_rep)  # ���数压缩

        return {
            'tf_representation': tf_rep,
            'center_frequencies': self.gabor.center_freqs,
        }


# ============ 下丘 / 内侧膝状体 (Inferior Colliculus / MGN) ============

class SubcorticalRelay(nn.Module):
    """
    下丘/内侧膝状体

    中继 + 门控 + 增益控制
    """
    def __init__(
        self,
        n_filters: int = 64,
    ):
        super().__init__()

        # 门控网络 (注意力机制)
        self.gate_net = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            nn.Sigmoid(),
        )

        # 增益网络
        self.gain_net = nn.Sequential(
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
        gated = cochlear_output * gate * gain

        return gated


# ============ 初级听觉皮层 (A1) ============

class PrimaryAuditoryCortex(nn.Module):
    """
    A1 - 初级听觉皮层

    提取spectral-temporal特征
    """
    def __init__(
        self,
        n_filters: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 时序卷积 (模拟A1的spectral-temporal感受野)
        self.conv_st = nn.Sequential(
            nn.Conv1d(n_filters, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
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
        features = self.conv_st(subcortical_input)
        features = self.pool(features).squeeze(-1)

        return {
            'a1_features': features,
        }


# ============ 腹侧流 (Ventral Stream - "是什么") + MOE ============

class VentralStream(nn.Module):
    """
    腹侧流 - 听觉识别 + MOE选择

    A1 → STG → 颞叶前部
    功能：音素→音节→词形式→词汇条目
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # MOE: 多个专家
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            for _ in range(3)
        ])
        self.gate = nn.Linear(input_dim, 3)

        # STG
        self.stg = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.frontal_temporal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, a1_features: torch.Tensor) -> Dict:
        # MOE: top-1专家
        gate_logits = self.gate(a1_features)
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_idx = gate_weights.argmax(dim=-1)
        expert_out = self.experts[top_idx](a1_features)

        stg_out = self.stg(expert_out)
        lexical = self.frontal_temporal(stg_out)

        return {
            'stg_features': stg_out,
            'lexical': lexical,
            'what': 'identity/word',
            'expert_used': top_idx.item(),
        }


# ============ 背侧流 (Dorsal Stream) + MOE ============

class DorsalStream(nn.Module):
    """
    背侧流 - 空间定位 + 运动 + MOE
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

        self.spatial = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
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
        motor_out = self.motor(spatial_out)

        return {
            'spatial': spatial_out,
            'motor': motor_out,
            'where': 'direction/location',
            'how': 'rhythm/motor',
            'expert_used': top_idx.item(),
        }


# ============ 完整听觉系统 ============

class AuditoryCortex(nn.Module):
    """
    完整听觉皮层 + 心理学机制

    整合:
    1. 耳蜗 (时频分析)
    2. 下丘/MGN (中继门控)
    3. A1 (初级皮层)
    4. 腹侧流 (识别)
    5. 背侧流 (定位+运动)
    6. 心理学期声 (听觉情境记忆)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filters: int = 128,
    ):
        super().__init__()

        hidden_dim = 256

        # 外周
        self.cochlea = Cochlea(n_filters, sample_rate)

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
                'ventral': {'lexical': ..., 'what': ...},
                'dorsal': {'motor': ..., 'where': ..., 'how': ...},
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
        }


# ============ 便捷函数 ============

def create_auditory_cortex(
    sample_rate: int = 16000,
    n_filters: int = 128,  # 增大
) -> AuditoryCortex:
    return AuditoryCortex(sample_rate, n_filters)


# ============ 听觉心理学组件 ============

class AuditoryContextMemory(nn.Module):
    """
    听觉情境记忆 (Auditory Context Memory)

    心理学: 听觉线索触发情境记忆
    实现: 声音模式 → 情境联想
    """
    def __init__(self, dim: int = 256):
        super().__init__()

        self.context_encoder = nn.Linear(dim, dim)
        self.context_memory = nn.Parameter(torch.randn(10, dim))  # 10个情境原型
        self.context_index = nn.Parameter(torch.zeros(10))

    def retrieve_context(self, audio_features: torch.Tensor):
        """检索最匹配的情境"""
        encoded = self.context_encoder(audio_features)
        # 计算相似度
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
    """
    注意力捕获 (Attentional Capture)

    心理学: 意外声音自动捕获注意力
    实现: 显著性检测 → 强制注意
    """
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
    """
    听觉情绪调节

    心理声学: 节奏/音调影响情绪
    """
    def __init__(self):
        super().__init__()

        self.tempo_sensitivity = nn.Parameter(torch.zeros(1))
        self.tone_sensitivity = nn.Parameter(torch.zeros(1))
        self.timbre_sensitivity = nn.Parameter(torch.zeros(1))

    def regulate(self, audio_features: torch.Tensor, tempo: float, tone: float):
        """
        根据音频特性调节情绪响应
        """
        tempo_effect = torch.tanh(self.tempo_sensitivity) * (tempo - 0.5)
        tone_effect = torch.tanh(self.tone_sensitivity) * (tone - 0.5)

        regulated = audio_features * (1 + tempo_effect + tone_effect)
        return regulated


# ============ 听党剪枝机制 ============

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
    'GaborFilterBank',
    'Cochlea',
    'SubcorticalRelay',
    'PrimaryAuditoryCortex',
    'VentralStream',
    'DorsalStream',
    'AuditoryCortex',
    'create_auditory_cortex',
]