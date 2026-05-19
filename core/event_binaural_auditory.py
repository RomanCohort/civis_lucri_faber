"""
双耳听觉 - 事件驱动版本

Civis Lucri-Faber - Event-Driven Binaural

基于脉冲神经网络:
1. ITD → Spiking Cross-Correlation
2. ILD → Spike Rates
3. 空间定位 → Event-Based
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict


class SpikingITD(nn.Module):
    """
    Spiking ITD 计算

    使用脉冲事件延迟相关
    """
    def __init__(self, n_channels: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.max_itd_us = 700

        # 延迟线
        self.delay_line = nn.Parameter(torch.zeros(1, n_channels))

        # ITD估计器
        self.itd_estimator = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute_itd(self, left_spikes: torch.Tensor,
                   right_spikes: torch.Tensor) -> torch.Tensor:
        """
        计算ITD

        Args:
            left_spikes: [B, T, C]
            right_spikes: [B, T, C]

        Returns:
            itd: [B] (μs)
        """
        B, T, C = left_spikes.shape
        max_delay = min(10, T - 1)

        itd = torch.zeros(B, device=left_spikes.device)

        for b in range(B):
            best_corr = -1.0
            best_delay = 0

            for delay in range(-max_delay, max_delay + 1):
                if delay < 0:
                    shifted = right_spikes[b, -delay:, :]
                    ref = left_spikes[b, :T+delay, :]
                elif delay > 0:
                    shifted = left_spikes[b, delay:, :]
                    ref = right_spikes[b, :T-delay, :]
                else:
                    shifted = left_spikes[b]
                    ref = right_spikes[b]

                if len(shifted) > 0 and len(ref) > 0:
                    corr = (shifted * ref).mean()
                    if corr > best_corr:
                        best_corr = corr
                        best_delay = delay

            itd[b] = best_delay * 1000

        return itd

    def forward(self, left_spikes: torch.Tensor,
              right_spikes: torch.Tensor) -> Dict:
        """
        Args:
            left_spikes: [B, T, C]
            right_spikes: [B, T, C]

        Returns:
            {itd, correlation}
        """
        itd = self.compute_itd(left_spikes, right_spikes)

        # 合并脉冲做相关性
        merged = torch.cat([left_spikes, right_spikes], dim=1)
        corr = self.itd_estimator(merged.mean(1, keepdim=True))

        return {
            'itd': itd,
            'correlation': corr,
        }


class SpikingILD(nn.Module):
    """
    Spiking ILD 计算

    基于脉冲发放率差异
    """
    def __init__(self, n_channels: int = 128):
        super().__init__()
        self.n_channels = n_channels

        # ILD估计
        self.ild_estimator = nn.Sequential(
            nn.Linear(n_channels, 64),
            nn.ReLU(),
            nn.Linear(64, n_channels),
        )

    def compute_ild(self, left_spikes: torch.Tensor,
                 right_spikes: torch.Tensor) -> torch.Tensor:
        """
        计算ILD

        Args:
            left/right_spikes: [B, T, C]

        Returns:
            ild: [B, C] (dB)
        """
        # 发放率
        rate_left = left_spikes.mean(1)
        rate_right = right_spikes.mean(1)

        # 强度比 (dB)
        ild = 10 * torch.log10(rate_left / (rate_right + 1e-8) + 1e-8)

        return ild

    def forward(self, left_spikes: torch.Tensor,
              right_spikes: torch.Tensor) -> Dict:
        """
        Returns:
            {ild, rate_left, rate_right}
        """
        ild = self.compute_ild(left_spikes, right_spikes)
        rate_left = left_spikes.mean(1)
        rate_right = right_spikes.mean(1)

        return {
            'ild': ild,
            'rate_left': rate_left,
            'rate_right': rate_right,
        }


class SpikingHRTF(nn.Module):
    """
    Spiking HRTF

    简化头相关传输函数
    """
    def __init__(self, n_channels: int = 128, n_directions: int = 36):
        super().__init__()
        self.n_channels = n_channels
        self.n_directions = n_directions

        # 可学习HRTF
        self.hrtf_coeffs = nn.Parameter(
            torch.randn(n_directions, n_channels) * 0.1
        )

        # 方向编码
        azimuth = np.linspace(-180, 180, n_directions)
        elevation = np.zeros(n_directions)
        dirs = np.stack([azimuth, elevation], axis=-1)
        self.register_buffer('directions', torch.tensor(dirs))

    def apply_hrtf(self, spikes: torch.Tensor,
                  azimuth: float) -> torch.Tensor:
        """
        应用HRTF

        Args:
            spikes: [B, T, C]
            azimuth: 度

        Returns:
            filtered: [B, T, C]
        """
        # 查找最近方向
        dirs = self.directions[:, 0]
        az_idx = (dirs - azimuth).abs().argmin()

        # 应用HRTF
        hrtf = self.hrtf_coeffs[az_idx]
        filtered = spikes * (1 + hrtf)

        return filtered

    def forward(self, spikes: torch.Tensor,
              azimuth: float = 0) -> Dict:
        """
        Args:
            spikes: [B, T, C]
            azimuth: 默认0度

        Returns:
            {filtered, hrtf}
        """
        filtered = self.apply_hrtf(spikes, azimuth)

        return {
            'filtered': filtered,
            'hrtf': self.hrtf_coeffs,
        }


class SpikingSpatialLocalizer(nn.Module):
    """
    Spiking 空间定位器

    从双耳脉冲定位
    """
    def __init__(self, n_channels: int = 128):
        super().__init__()

        # 双耳整合
        self.binaural_proj = nn.Linear(n_channels * 2, 128)

        # 空间映射
        self.spatial_map = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # azimuth, elevation
        )

    def forward(self, left_spikes: torch.Tensor,
              right_spikes: torch.Tensor) -> Dict:
        """
        Args:
            left/right_spikes: [B, C]

        Returns:
            {azimuth, elevation}
        """
        # 合并
        binaural = torch.cat([left_spikes, right_spikes], dim=-1)
        binaural = self.binaural_proj(binaural)

        location = self.spatial_map(binaural)
        azimuth = location[:, 0] * 180
        elevation = location[:, 1] * 90

        return {
            'azimuth': azimuth,
            'elevation': elevation,
        }


class EventDrivenBinaural(nn.Module):
    """
    事件驱动双耳听觉

    整合:
    1. Spiking ITD
    2. Spiking ILD
    3. Spiking HRTF
    4. Spiking Spatial Localization
    """
    def __init__(self, n_channels: int = 128):
        super().__init__()

        self.itd = SpikingITD(n_channels)
        self.ild = SpikingILD(n_channels)
        self.hrtf = SpikingHRTF(n_channels)
        self.localizer = SpikingSpatialLocalizer(n_channels)

    def forward(self, left_spikes: torch.Tensor,
              right_spikes: torch.Tensor,
              azimuth: float = 0) -> Dict:
        """
        Args:
            left_spikes: [B, T, C] 或 [B, C]
            right_spikes: [B, T, C] 或 [B, C]
            azimuth: 声源方向

        Returns:
            {itd, ild, azimuth, elevation}
        """
        # ITD
        itd_result = self.itd(left_spikes, right_spikes)
        itd = itd_result['itd']

        # ILD
        ild_result = self.ild(left_spikes, right_spikes)
        ild = ild_result['ild']

        # HRTF
        hrtf_result = self.hrtf(left_spikes, azimuth)
        filtered = hrtf_result['filtered']

        # 空间定位
        # 适配输入维度
        if left_spikes.dim() == 3:
            left_rate = left_spikes.mean(1)
            right_rate = right_spikes.mean(1)
        else:
            left_rate = left_spikes
            right_rate = right_spikes

        location = self.localizer(left_rate, right_rate)

        return {
            'itd': itd,
            'ild': ild,
            'azimuth': location['azimuth'],
            'elevation': location['elevation'],
            'filtered': filtered,
            'rate_left': ild_result.get('rate_left'),
            'rate_right': ild_result.get('rate_right'),
        }


def create_event_binaural(n_channels: int = 128) -> EventDrivenBinaural:
    return EventDrivenBinaural(n_channels)


# ============ 测试 ============

if __name__ == "__main__":
    print("=== Testing Event-Driven Binaural ===\n")

    # 测试 Spiking ITD
    print("[1] Spiking ITD")
    itd = SpikingITD(64)
    left = (torch.rand(2, 100, 64) > 0.9).float()
    right = (torch.rand(2, 100, 64) > 0.9).float()
    result = itd(left, right)
    print(f"  - ITD: {result['itd'].shape}")

    # 测试 Spiking ILD
    print("\n[2] Spiking ILD")
    ild = SpikingILD(64)
    result = ild(left, right)
    print(f"  - ILD: {result['ild'].shape}")
    print(f"  - Rate L: {result['rate_left'].shape}")

    # 测试完整双耳
    print("\n[3] Event-Driven Binaural")
    binaural = create_event_binaural(64)
    result = binaural(left, right, 45)
    print(f"  - ITD: {result['itd'].shape}")
    print(f"  - ILD: {result['ild'].shape}")
    print(f"  - Azimuth: {result['azimuth'].shape}")

    print("\nAll tests passed!")