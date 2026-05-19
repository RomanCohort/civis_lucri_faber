"""
听觉系统神经拟真改进方案

Simulacrum - Spiking Auditory Cortex

从系统工程 → 神经拟真
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============== 阶段1: 生物物理 ==============

@dataclass
class CochleaParams:
    """耳蜗关键参数"""
    n_channels: int = 128          # 频率通道数 (~3500 IHC真实值)
    sample_rate: int = 16000       # 采样率
    length_mm: float = 35.0         # 基底膜长度 (mm)
    base_freq_hz: float = 20000     # 基部特征频率
    apex_freq_hz: float = 200      # 顶部特征频率
    compression_ratio: float = 0.3  # 毛细胞压缩比
    ohc_gain_db: float = 50        # OHC放大增益 (dB)


class BasilarMembrane(nn.Module):
    """
    基底膜 - 行波模型

    关键物理：
    1. 刚度梯度: 基部→顶部 刚度递减100倍
    2. 行波传播: 频率越高位置越靠近基部
    3. 延迟: 低频顶点延迟 (~10ms @ 200Hz)
    """

    def __init__(self, params: CochleaParams):
        super().__init__()
        self.p = params

        # 对数频率分布 (符合耳蜗)
        freqs = np.geomspace(params.apex_freq_hz, params.base_freq_hz, params.n_channels)
        self.register_buffer('cf_hz', torch.tensor(freqs))

        # 每个位置的带宽 (Q值 ~10)
        self.Q = 10.0
        self.bandwidths = freqs / self.Q

        # 延迟参数 (ms) - 低频延迟更长
        self.delays_ms = 12.0 / (freqs / 1000)  # ~12ms @ 1kHz基准

        # 空间扩散核 (simulate membrane mechanics)
        self.spatial_kernel_size = 5

    def forward(self, audio: torch.Tensor) -> Dict:
        """
        audio: [B, T]
        Returns:
            displacement: [B, T, n_channels]
        """
        B, T = audio.shape

        # STFT → 频域表示
        stft = torch.stft(
            audio,
            n_fft=512,
            hop_length=128,
            window=torch.hann_window(512),
            return_complex=True
        )

        F_ = stft.shape[-1]
        freqs = torch.fft.rfftfreq(512, 1.0/self.p.sample_rate)[:F_]

        # 每个channel的响应 (bandpass filter bank)
        outputs = []
        for i, cf in enumerate(self.cf_hz):
            bw = self.bandwidths[i]
            # 高斯带通
            response = torch.exp(-((freqs - cf) / bw) ** 2)
            response = response.view(1, 1, F_) * stft
            output = torch.istft(response, 512, 128,
                          window=torch.hann_window(512),
                          length=T)
            outputs.append(output)

        displacement = torch.stack(outputs, dim=-1)

        # 行波延迟 (简化: 低频更晚)
        displacement = self._apply_traveling_wave_delay(displacement)

        return {
            'displacement': displacement,
            'cf_hz': self.cf_hz,
            'pitch': self._track_pitch(displacement)
        }

    def _apply_traveling_wave_delay(self, x: torch.Tensor) -> torch.Tensor:
        """应用行波延迟"""
        # 简化为一个固定延迟 (转换为整数)
        delay_samples = int((self.delays_ms * self.p.sample_rate / 1000).mean())
        if delay_samples > 0 and x.shape[1] > delay_samples:
            x = x[:, delay_samples:, :]
        return x

    def _track_pitch(self, displacement: torch.Tensor) -> torch.Tensor:
        """追踪基频 (fundamental frequency)"""
        # 简化: 用最大能量通道
        energy = displacement.abs().mean(-2)
        peak_cf_idx = energy.argmax(-1)
        return self.cf_hz[peak_cf_idx]


class InnerHairCellNonlinear(nn.Module):
    """
    内毛细胞 - 非线性换能

    关键机制：
    1. 半波整流 (只有伸长产生信号)
    2. 对数压缩 (动态范围约100dB → 40dB)
    3. 快速适应 (20ms时间常数)
    """

    def __init__(self, n_channels: int, compression_ratio: float = 0.3):
        super().__init__()
        self.n_channels = n_channels
        self.compression_ratio = nn.Parameter(torch.tensor(compression_ratio))

        # 适应网络 (fast adaptation)
        self.adapt_tau = 20e-3  # 20ms
        self.adapt_state = None

    def forward(self, displacement: torch.Tensor) -> torch.Tensor:
        """
        displacement: [B, T, n_channels]
        Returns:
            firing_rate: [B, T, n_channels]  (spike rate, Hz)
        """
        # 1. 半波整流
        response = F.relu(displacement)

        # 2. 非线性压缩 (power-law)
        # rate ∝ stimulus^0.3 (真实IHC特性)
        rate = torch.pow(response + 1e-8, self.compression_ratio)

        # 3. 快速适应 (简化)
        if self.adapt_state is None:
            self.adapt_state = torch.zeros_like(rate[:, :1, :])

        # 适应: 快速衰减
        adapted = rate - self.adapt_state * 0.1
        adapted = F.relu(adapted)

        # 更新适应状态
        self.adapt_state = adapted[:, -1:, :].detach()

        return adapted


class OuterHairCellAmplifier(nn.Module):
    """
    外毛细胞 - 主动放大器

    关键机制：
    1. 电致伸缩 (prestin蛋白)
    2. 高速放大 (~100kHz)
    3. 压缩动态范围
    4. OAE产生
    """

    def __init__(self, n_channels: int, gain_db: float = 50.0):
        super().__init__()
        self.n_channels = n_channels
        self.gain_db = gain_db
        self.gain_linear = 10 ** (gain_db / 20)  # dB → linear

        # 饱和非线性 (防止过载)
        self.saturation_level = 0.5

    def amplify(self, displacement: torch.Tensor,
              feedback: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        主动放大

        displacement: [B, T, n_channels]
        feedback: [B, T, n_channels] (来自脑干反馈)
        Returns:
            amplified: [B, T, n_channels]
        """
        # 基础增益
        amplified = displacement * self.gain_linear

        # 脑干反馈调制 (抑制过强信号)
        if feedback is not None:
            amplified = amplified * (1 - feedback * 0.5)

        # 饱和限制
        amplified = torch.tanh(amplated / self.saturation_level) * self.saturation_level

        return amplified

    def generate_oae(self, amplified: torch.Tensor) -> torch.Tensor:
        """
        产生耳声发射 (OAE)

        用于监测放大器状态
        """
        # 简化: 畸变产物 OAEs (DPOAE)
        # f2 - f1 产生
        return amplified.sum(-1) / self.n_channels


class CochleaModel(nn.Module):
    """
    完整生物拟真耳蜗
    """

    def __init__(self, params: Optional[CochleaParams] = None):
        super().__init__()
        self.p = params or CochleaParams()

        self.basilar = BasilarMembrane(self.p)
        self.ihc = InnerHairCellNonlinear(
            self.p.n_channels,
            self.p.compression_ratio
        )
        self.ohc = OuterHairCellAmplifier(
            self.p.n_channels,
            self.p.ohc_gain_db
        )

    def forward(self, audio: torch.Tensor) -> Dict:
        # 行波 → 毛细胞响应
        bm = self.basilar(audio)

        # IHC换能
        ihc_rate = self.ihc(bm['displacement'])

        # OHC放大 (简化, 无反馈)
        oae = self.ohc.generate_oae(bm['displacement'])

        return {
            'displacement': bm['displacement'],  # 基底膜位移
            'ihc_rate': ihc_rate,         # 神经发放率
            'pitch': bm['pitch'],          # 基频
            'oae': oae,                # 耳声发射
            'cf_hz': bm['cf_hz']         # 特征频率
        }


# ============== 1.2 脉冲神经网络编码 ==============

@dataclass
class LIFParams:
    """Leaky Integrate-and-Fire 参数"""
    tau_mem: float = 10e-3      # 膜时间常数
    v_thresh: float = -55e-3     # 阈值电位 (mV)
    v_rest: float = -70e-3       # 静息电位
    v_reset: float = -75e-3      # 重置电位
    tau_ref: float = 1e-3        # 不应期


class SpikingAuditoryEncoder(nn.Module):
    """
    脉冲编码器 - 将耳蜗输出转为脉冲序列

    使用 Leaky Integrate-and-Fire (LIF)
    关键: phase-locking encoding
    """

    def __init__(self, n_channels: int = 128, params: Optional[LIFParams] = None):
        super().__init__()
        self.p = params or LIFParams()
        self.n_channels = n_channels

        # 阈值适应 (expand to match channels)
        self.thresh_adapt = nn.Parameter(torch.full((n_channels,), self.p.v_thresh))

        # 侧抑制
        self.k_winners = int(n_channels * 0.05)  # 5%

    def forward(self, ihc_rate: torch.Tensor) -> Dict:
        """
        ihc_rate: [B, T, n_channels] (Hz firing rate)

        Returns:
            spikes: [B, T, n_channels] (binary 0/1)
            phase_lock: [B, n_channels] (同步度)
            latency: [B, n_channels] (首_spike时间)
        """
        B, T, C = ihc_rate.shape

        # 率 → 电流 (poisson process)
        # 简化: 确定性近似
        dt = 0.001  # 假设1ms frame

        # 积分
        V = torch.cumsum(ihc_rate * dt, dim=1)

        # 阈值
        thresh = self.thresh_adapt.view(1, 1, C).expand(B, T, C)

        # 发spike
        spikes = (V > thresh).float()

        # 重置
        for t in range(1, T):
            reset_mask = spikes[:, t-1:t, :].expand(B, 1, C)
            V[:, t:, :] = V[:, t:, :] * (1 - reset_mask) + self.p.v_reset * reset_mask

        # 侧抑制
        spikes = self._lateral_inhibition(spikes)

        return {
            'spikes': spikes,
            'phase_lock': self._compute_phase_lock(ihc_rate, spikes),
            'latency': self._compute_latency(spikes),
            'rate': spikes.sum(-2) / (T * dt)
        }

    def _lateral_inhibition(self, spikes: torch.Tensor) -> torch.Tensor:
        """侧抑制 - 只保留5%最强"""
        B, T, C = spikes.shape

        # 每时刻选择top-k
        for t in range(T):
            frame = spikes[:, t, :]
            if frame.sum() > self.k_winners:
                values, indices = frame.topk(self.k_winners)
                new_frame = torch.zeros_like(frame)
                new_frame.scatter_(1, indices, values)
                spikes[:, t, :] = new_frame

        return spikes

    def _compute_phase_lock(self, rate: torch.Tensor,
                          spikes: torch.Tensor) -> torch.Tensor:
        """
        计算相位锁定 (phase locking)

        低频(<2kHz)神经元与声波周期同步
        """
        B, T, C = rate.shape

        # 简化: 用发放率的标准差
        phase_lock = rate.std(-2) / (rate.mean(-2) + 1e-8)

        return phase_lock

    def _compute_latency(self, spikes: torch.Tensor) -> torch.Tensor:
        """首 spike 潜伏期"""
        B, T, C = spikes.shape

        latency = torch.full((B, C), T, dtype=torch.float32)

        for t in range(T):
            for b in range(B):
                fired = spikes[b, t, :]
                for c in range(C):
                    if fired[c] > 0 and latency[b, c] == T:
                        latency[b, c] = t

        return latency


# ============== 1.3 双耳整合 ==============

class BinauralIntegration(nn.Module):
    """
    双耳整合 - ITD / ILD

    关键机制：
    1. ITD: 互相关 (interaural time difference)
    2. ILD: 强度差 (interaural level difference)
    3. HRTF: 头相关传输函数
    """

    def __init__(self, n_channels: int = 128):
        super().__init__()
        self.n_channels = n_channels

        # 简化HRTF (实际应加载真实HRTF)
        self.hrtf_left = nn.Parameter(torch.randn(n_channels, 64))
        self.hrtf_right = nn.Parameter(torch.randn(n_channels, 64))

        # ITD检测 (cross-correlation)
        self.max_itd_us = 700  # 最大ITD (μs), 头宽~18cm

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> Dict:
        """
        left/right: [B, T, n_channels]

        Returns:
            itd: [B, n_channels] (μs)
            ild: [B, n_channels] (dB)
            azimuth: [B] (度)
            elevation: [B] (度)
        """
        # ITD: 互相关
        itd = self._compute_itd(left, right)

        # ILD: 能量比
        ild = self._compute_ild(left, right)

        # 方位 (简化)
        azimuth = torch.atan2(itd, 0.1) * 180 / np.pi

        return {
            'itd': itd,
            'ild': ild,
            'azimuth': azimuth,
            'elevation': torch.zeros_like(azimuth)
        }

    def _compute_itd(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """互相关计算ITD"""
        # 简化: 延迟求和
        B, T, C = left.shape

        itd = torch.zeros(B, C)

        for b in range(B):
            for c in range(C):
                # 粗略搜索
                best_delay = 0
                best_corr = -1
                for delay in range(-10, 10):
                    if delay < 0:
                        shifted = right[b, -delay:, c]
                        ref = left[b, :T+delay, c]
                    else:
                        shifted = left[b, delay:, c]
                        ref = right[b, :T-delay, c]

                    if len(shifted) > 0 and len(ref) > 0:
                        corr = (shifted * ref).mean()
                        if corr > best_corr:
                            best_corr = corr
                            best_delay = delay

                itd[b, c] = best_delay * 1000  # 假设1ms帧 → μs

        return itd

    def _compute_ild(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """ILD = 20log10(P_left / P_right)"""
        power_left = (left ** 2).mean(-2)
        power_right = (right ** 2).mean(-2)

        ild = 20 * torch.log10(power_left / (power_right + 1e-8) + 1e-8)

        return ild


# ============== 1.4 可塑性 (STDP) ==============

class STDPPlasticity(nn.Module):
    """
    脉冲时序依赖可塑性 (Spike-Timing-Dependent Plasticity)

    Hebbian: .fire together, wire together_

    关键规则：
    1. pre→post先: LTP (增强)
    2. post→pre先: LTD (减弱)
    3. 时间窗: ±20ms
    """

    def __init__(self, tau_plus: float = 20e-3, tau_minus: float = 20e-3):
        super().__init__()
        self.tau_plus = tau_plus   # LTP时间常数
        self.tau_minus = tau_minus # LTD时间常数

        self.A_plus = 0.01      # LTP幅度
        self.A_minus = 0.012     # LTD幅度 (稍大)

    def forward(self, pre_spikes: torch.Tensor,
               post_spikes: torch.Tensor) -> Dict:
        """
        pre/post: [B, T, n_neurons]

        Returns:
            weight_change: [B, n_neurons, n_neurons]
        """
        B, T, N = pre_spikes.shape

        # 简化: 计算时间差
        # 找到pre和post spikes的时间

        delta_t = self._find_temporal_gap(pre_spikes, post_spikes)

        # STDP曲线
        # LTP: δt > 0 (pre before post) → strengthen
        ltp = self.A_plus * torch.exp(-delta_t.abs() / self.tau_plus)
        ltp = ltp * (delta_t > 0).float()

        # LTD: δt < 0 (post before pre) → weaken
        ltd = self.A_minus * torch.exp(-delta_t.abs() / self.tau_minus)
        ltd = ltd * (delta_t < 0).float()

        weight_change = ltp - ltd

        return {'delta_w': weight_change}

    def _find_temporal_gap(self, pre: torch.Tensor,
                     post: torch.Tensor) -> torch.Tensor:
        """简化: 返回时间差"""
        # 简化实现
        return torch.zeros_like(pre.sum(-1, keepdim=True).squeeze(-1))


# ============== 完整系统 ==============

class SpikingAuditoryCortex(nn.Module):
    """
    神经拟真听觉皮层 - 完整系统

    整合:
    1. 生物拟真耳蜗
    2. 脉冲编码
    3. 双耳整合
    4. STDP可塑性
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 n_channels: int = 128):
        super().__init__()

        # 耳蜗
        self.cochlea = CochleaModel()

        # 脉冲编码
        self.encoder = SpikingAuditoryEncoder(n_channels)

        # 双耳
        self.binaural = BinauralIntegration(n_channels) if hasattr(torch.cuda, 'is_available') else None

        # 可塑性 (可选, 训练时启用)
        self.stdp = STDPPlasticity()

        # 输出 (处理spikes)
        self.cortex = nn.Sequential(
            nn.Conv1d(n_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, audio_left: torch.Tensor,
               audio_right: Optional[torch.Tensor] = None) -> Dict:
        """
        audio_left: [B, T]
        audio_right: [B, T] (可选, 双耳)

        Returns:
            spikes: [B, T, n_channels]
            features: [B, 256]
            pitch: [B]
            azimuth: [B]
        """
        # 耳蜗
        cochlear = self.cochlea(audio_left)
        ihc_rate = cochlear['ihc_rate']

        # 脉冲编码
        spikes = self.encoder(ihc_rate)

        # 特征
        features = self.cortex(spikes['spikes'].transpose(1, 2)).squeeze(-1)

        result = {
            'spikes': spikes['spikes'],
            'features': features,
            'pitch': cochlear['pitch'],
            'phase_lock': spikes['phase_lock'],
            'latency': spikes['latency'],
            'oae': cochlear['oae']
        }

        # 双耳 (如果有)
        if audio_right is not None and self.binaural is not None:
            cochlear_r = self.cochlea(audio_right)
            binaural = self.binaural(
                cochlear['ihc_rate'],
                cochlear_r['ihc_rate']
            )
            result['azimuth'] = binaural['azimuth']
            result['itd'] = binaural['itd']
            result['ild'] = binaural['ild']

        return result


def create_spiking_auditory_cortex() -> SpikingAuditoryCortex:
    return SpikingAuditoryCortex()


# ============== 测试 ==============

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 测试耳蜗
    cochlea = CochleaModel()
    audio = torch.randn(1, 16000)  # 1秒音频
    out = cochlea(audio)

    print("=== Cochlea Test ===")
    print(f"displacement shape: {out['displacement'].shape}")
    print(f"IHC rate shape: {out['ihc_rate'].shape}")
    print(f"pitch: {out['pitch']}")

    # 测试完整系统
    cortex = create_spiking_auditory_cortex()
    result = cortex(audio)

    print("\n=== Spiking Auditory Cortex ===")
    print(f"spikes shape: {result['spikes'].shape}")
    print(f"features shape: {result['features'].shape}")
    print(f"phase_lock: {result['phase_lock'].shape}")
    print(f"avg spike rate: {result['spikes'].mean():.2f} Hz")