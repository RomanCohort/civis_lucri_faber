"""
脉冲神经网络 (SNN) 基础架构 - 事件驱动

Simulacrum - SNN Core

特点:
1. Leaky Integrate-and-Fire (LIF) 神经元
2. 事件驱动计算 (稀疏spike)
3. 异步处理
4. 低功耗模拟
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ 基础组件 ============

@dataclass
class SNNConfig:
    """SNN配置"""
    tau_mem: float = 10e-3      # 膜时间常数 (10ms)
    v_thresh: float = -55e-3     # 阈值电位 (mV)
    v_rest: float = -70e-3       # 静息电位 (mV)
    v_reset: float = -75e-3    # 重置电位 (mV)
    tau_ref: float = 1e-3      # 不应期 (1ms)
    tau_adapt: float = 100e-3   # 适应性时间常数 (新增, 100ms)
    adapt_rate: float = 0.1     # 适应性发放率 (新增)


class LeakyIntegrateAndFire(nn.Module):
    """
    泄漏积分触发器 (LIF)

    核心Spiking神经元模型:
    1. tau * dV/dt = -(V - V_rest) + I*R
    2. V > V_thresh → spike → V_reset

    新增功能:
    3. 不应期 (refractory period) - spike后一段时间内不响应输入
    4. 适应性发放 (adaptation) - 随着连续spike阈值逐渐升高
    """
    def __init__(self, n_neurons: int, config: SNNConfig | None = None):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config or SNNConfig()

        # 膜电位状态
        self.register_buffer('v_mem', torch.full((n_neurons,), self.config.v_rest))
        self.register_buffer('last_spike', torch.zeros(n_neurons, dtype=torch.long))

        # 不应期状态 (新增)
        self.register_buffer('in_refractory', torch.zeros(n_neurons, dtype=torch.bool))

        # 适应性电流 (新增)
        self.register_buffer('adaptation_current', torch.zeros(n_neurons))
        self.adapt_increment = self.config.adapt_rate  # 每次spike增加的适应性

        # 可学习参数
        self.weight = nn.Parameter(torch.randn(n_neurons) * 0.1)
        self.threshold = nn.Parameter(torch.tensor(self.config.v_thresh))

        # 时间计数 (用于不应期计算)
        self.current_time_step = 0

    def forward(self, current: torch.Tensor) -> dict:
        """
        输入电流 → spike输出

        Args:
            current: [B, n_neurons] 输入电流

        Returns:
            {spikes, v_mem, in_refractory, adaptation}
        """
        B = current.shape[0]
        self.current_time_step += 1

        # 初始化膜电位
        if self.v_mem.dim() == 1:
            v_mem = self.v_mem.unsqueeze(0).expand(B, -1)
        else:
            v_mem = self.v_mem

        # ── 不应期检查 (新增) ──
        # 计算从上次spike到现在的步数
        time_since_spike = self.current_time_step - self.last_spike
        refractory_steps = int(self.config.tau_ref * 1000)  # 转换ms到steps

        # 创建不应期mask: 时间 < refractory_steps → 不响应
        refractory_mask = (time_since_spike < refractory_steps).float()
        self.in_refractory = refractory_mask.bool()

        # ── 适应性阈值调整 (新增) ──
        # 有效阈值 = base_threshold + adaptation_current
        effective_threshold = self.threshold + self.adaptation_current

        # ── 膜电位更新 ──
        # 1. Leak衰减
        v_mem = v_mem * np.exp(-1 / (self.config.tau_mem * 1000))

        # 2. 输入驱动 (但在不应期内被阻断)
        current_masked = current * (1 - refractory_mask.unsqueeze(0))
        v_mem = v_mem + current_masked * self.weight

        # 3. 适应性电流衰减
        self.adaptation_current = self.adaptation_current * np.exp(-1 / (self.config.tau_adapt * 1000))

        # ── Spike生成 ──
        # 使用有效阈值
        spikes = (v_mem > effective_threshold.unsqueeze(0)).float()

        # ── Spike后处理 ──
        # 1. 重置触发spike的神经元
        v_mem = v_mem * (1 - spikes) + spikes * self.config.v_reset

        # 2. 记录spike时间 (用于不应期)
        new_spike_mask = spikes[0] > 0  # [n_neurons]
        self.last_spike = torch.where(
            new_spike_mask,
            torch.full_like(self.last_spike, self.current_time_step),
            self.last_spike
        )

        # 3. 增加适应性电流 (每次spike后增加)
        self.adaptation_current = torch.where(
            new_spike_mask,
            self.adaptation_current + self.adapt_increment,
            self.adaptation_current
        )

        # 更新状态
        self.v_mem = v_mem.mean(0).detach()

        return {
            'spikes': spikes,
            'v_mem': v_mem,
            'in_refractory': self.in_refractory,
            'adaptation': self.adaptation_current,
            'effective_threshold': effective_threshold,
        }

    def reset_adaptation(self):
        """重置适应性电流"""
        self.adaptation_current.zero_()

    def get_firing_rate(self, window: int = 100) -> float:
        """获取最近window步的平均发放率"""
        # 简化: 返回当前适应性电流的倒数作为发放率估计
        return 1.0 / (1.0 + self.adaptation_current.mean().item())


class SpikingLayer(nn.Module):
    """
    Spiking全连接层

    使用surrogate gradient简化反向传播
    """
    def __init__(self, in_features: int, out_features: int,
                 tau_mem: float = 10e-3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau_mem = tau_mem

        # 权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # 膜电位状态
        self.register_buffer('v_mem', torch.zeros(out_features))
        self.register_buffer('spike_counts', torch.zeros(out_features))

    def forward(self, input_spikes: torch.Tensor) -> dict:
        """
        Args:
            input_spikes: [B, in_features] binary spikes

        Returns:
            {spikes, v_mem, rates}
        """
        B = input_spikes.shape[0]

        # 突触电流
        current = F.linear(input_spikes, self.weight, self.bias)

        # 膜电位更新
        v_mem = self.v_mem.unsqueeze(0).expand(B, -1)
        decay = np.exp(-1 / (self.tau_mem * 1000))
        v_mem = v_mem * decay + current

        # Spike发射 (使用surrogate gradient)
        spike_mask = (v_mem > 0).float()
        spikes = spike_mask * (v_mem > 0).float().detach()

        # 重置
        v_mem = v_mem * (1 - spikes) - spikes * 75e-3

        # 更新状态
        self.v_mem = v_mem.mean(0).detach()
        self.spike_counts = spikes.sum(0).detach()

        rates = spikes.sum(-1) / B  # 发放率

        return {
            'spikes': spikes,
            'v_mem': v_mem,
            'rates': rates,
        }


class SpikingConv1d(nn.Module):
    """
    Spiking卷积层

    1D卷积的spiking版本
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, tau_mem: float = 10e-3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.tau_mem = tau_mem

        # 卷积核
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                           padding=kernel_size//2)

        # 膜电位
        self.register_buffer('v_mem', torch.zeros(out_channels))

    def forward(self, input_spikes: torch.Tensor) -> dict:
        """
        Args:
            input_spikes: [B, C, T]

        Returns:
            {spikes, v_mem}
        """
        # 卷积
        conv_out = self.conv(input_spikes)

        # 膜电位更新
        B, C, T = conv_out.shape
        v_mem = self.v_mem.unsqueeze(0).unsqueeze(-1).expand(B, C, T)
        decay = np.exp(-1 / (self.tau_mem * 1000))
        v_mem = v_mem * decay + conv_out

        # Spike
        spikes = (v_mem > 0).float()
        v_mem = v_mem * (1 - spikes) - spikes * 75e-3

        return {
            'spikes': spikes,
            'v_mem': v_mem,
        }


class SpikingPooling(nn.Module):
    """
    Spiking池化层

    时间窗口内spike计数
    """
    def __init__(self, window_size: int = 10):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('counter', torch.zeros(1))

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: [B, C, T]

        Returns:
            rates: [B, C]
        """
        # 时间窗内计数
        rates = spikes.sum(-1) / self.window_size
        return rates


class LateralInhibition(nn.Module):
    """
    侧抑制 - winner-take-all

    竞争机制: 增强的spike抑制弱的
    """
    def __init__(self, n_neurons: int, k_winners: int = 1):
        super().__init__()
        self.n_neurons = n_neurons
        self.k_winners = k_winners

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: [B, n_neurons]

        Returns:
            inhibited: [B, n_neurons]
        """
        B, C = spikes.shape

        # 每样本选择top-k
        for b in range(B):
            values, indices = spikes[b].topk(self.k_winners)
            new_spikes = torch.zeros_like(spikes[b])
            new_spikes[indices] = values
            spikes[b] = new_spikes

        return spikes


class DenseToSparse(nn.Module):
    """
    密集 → 稀疏转换

    将率编码转换为spike时空稀疏编码
    """
    def __init__(self, n_neurons: int, rate_scale: float = 100.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.rate_scale = rate_scale

    def forward(self, rates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rates: [B, n_neurons] 发放率 (0-1)

        Returns:
            spikes: [B, n_neurons, T] 稀疏spike
        """
        B, C = rates.shape

        # Poisson spike生成
        probs = rates * self.rate_scale / 1000  # 转换为概率
        rand = torch.rand(B, C, 1)
        spikes = (rand < probs.unsqueeze(-1)).float()

        # 稀疏: 每个神经元最多1个spike
        spikes = spikes * (spikes.sum(-1, keepdim=True) > 0).float()

        return spikes


class SpikeEncode(nn.Module):
    """
    率 → Spike 编码器

    多种编码策略:
    1. Poisson - 随机
    2. Rate - 时间编码
    3. Temporal - 首次spike时间
    """
    def __init__(self, encoding: str = 'poisson', tau: float = 10e-3):
        super().__init__()
        self.encoding = encoding
        self.tau = tau

    def poisson_encode(self, rates: torch.Tensor) -> torch.Tensor:
        """
        Poisson编码

        rate高 → spike密集
        """
        B, C = rates.shape
        device = rates.device

        # 随机spike
        prob = rates * self.tau  # spike概率
        rand = torch.rand(B, C, 1, device=device)
        spikes = (rand < prob).float()

        return spikes

    def temporal_encode(self, rates: torch.Tensor) -> torch.Tensor:
        """
        时间编码 (latency)

        rate → 首个spike的延迟
        """
        # 延迟 = 1/rate
        latency = 1 / (rates + 1e-8)
        # 转换为spike timing (简化)
        T = 100
        timing = (latency / latency.max() * T).long()
        timing = timing.clamp(0, T-1)

        spikes = torch.zeros(rates.shape[0], rates.shape[1], T)
        for b in range(rates.shape[0]):
            for c in range(rates.shape[1]):
                t = timing[b, c]
                spikes[b, c, t:] = 1

        return spikes

    def rate_encode(self, rates: torch.Tensor) -> torch.Tensor:
        """
        率编码 (时间平均spike数)
        """
        spike_count = rates * self.tau * 1000  # Hz
        return spike_count

    def forward(self, rates: torch.Tensor) -> dict:
        """
        编码

        Args:
            rates: [B, C] 发放率

        Returns:
            {spikes, encoding_type}
        """
        if self.encoding == 'poisson':
            spikes = self.poisson_encode(rates)
        elif self.encoding == 'temporal':
            spikes = self.temporal_encode(rates)
        else:
            spikes = self.rate_encode(rates)

        return {
            'spikes': spikes,
            'encoding': self.encoding,
        }


class SpikeDecode(nn.Module):
    """
    Spike → 率 解码器

    从spike序列重建发放率
    """
    def __init__(self, window_size: int = 10):
        super().__init__()
        self.window_size = window_size

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: [B, C, T] or [B, C]

        Returns:
            rates: [B, C]
        """
        if spikes.dim() == 3:
            # 时间窗口内计数
            rates = spikes.sum(-1) / self.window_size
        else:
            rates = spikes

        return rates


# ============ SNN模块 (兼容Dense接口) ============

class SNN密集层(nn.Module):
    """兼容Dense接口的SNN层"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lif = SpikingLayer(in_features, out_features)
        self.encoder = SpikeEncode('poisson')

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, in_features]

        Returns:
            {spikes, rates}
        """
        # 率编码
        encoded = self.encoder(x)
        spikes = encoded['spikes']

        # LIF
        out = self.lif(spikes)
        rates = out['rates']

        return {'spikes': out['spikes'], 'rates': rates}


# ============ 网络构建器 ============

class SNNBuilder(nn.Module):
    """
    SNN网络构建器

    便捷函数
    """
    @staticmethod
    def build_dense(in_dim: int, out_dim: int,
                    hidden_dims: list = None) -> nn.Module:
        """构建全连接SNN"""
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            layers.append(SpikingLayer(prev_dim, h_dim))
            prev_dim = h_dim

        layers.append(SpikingLayer(prev_dim, out_dim))

        return nn.Sequential(*layers)


def create_snn_layer(in_features: int, out_features: int) -> SpikingLayer:
    return SpikingLayer(in_features, out_features)


def create_encoder(encoding: str = 'poisson') -> SpikeEncode:
    return SpikeEncode(encoding)


# ============ 测试 ============

if __name__ == "__main__":
    print("=== Testing SNN Core ===\n")

    # 测试LIF
    print("[1] LIF Neuron")
    lif = LeakyIntegrateAndFire(64)
    current = torch.randn(2, 64) * 0.5
    result = lif(current)
    print(f"  - spikes: {result['spikes'].shape}")
    print(f"  - v_mem: {result['v_mem'].shape}")

    # 测试SpikingLayer
    print("\n[2] Spiking Layer")
    layer = SpikingLayer(128, 64)
    input_spikes = (torch.rand(2, 128) > 0.8).float()
    result = layer(input_spikes)
    print(f"  - spikes: {result['spikes'].shape}")
    print(f"  - rates: {result['rates'].shape}")

    # 测试Spike编码
    print("\n[3] Spike Encode")
    encoder = SpikeEncode('poisson')
    rates = torch.rand(2, 64)
    result = encoder(rates)
    print(f"  - encoding: {result['encoding']}")
    print(f"  - spikes: {result['spikes'].shape}")
    print(f"  - spike rate: {result['spikes'].mean():.3f}")

    # 测试DenseToSparse
    print("\n[4] Dense to Sparse")
    d2s = DenseToSparse(64)
    rates = torch.rand(2, 64)
    spikes = d2s(rates)
    print(f"  - spikes shape: {spikes.shape}")
    print(f"  - sparsity: {(spikes > 0).float().mean():.3f}")

    print("\nAll SNN tests passed!")
