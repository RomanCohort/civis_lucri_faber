"""
双耳听觉系统 - ITD/ILD/HRTF

Simulacrum - 双耳听觉机制

关键机制:
1. ITD: 互相关计算时间差
2. ILD: 强度差
3. HRTF: 头相关传输函数
4. 空间定位
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class HRTFModel(nn.Module):
    """
    头部相关传输函数 (HRTF)

    简化模型 - 可学习参数
    实际应加载真实HRTF数据
    """
    def __init__(self, n_channels: int = 128, n_directions: int = 36):
        super().__init__()
        self.n_channels = n_channels
        self.n_directions = n_directions

        # 可学习的HRTF (方向 × 频率)
        self.hrtf_coeffs = nn.Parameter(
            torch.randn(n_directions, n_channels) * 0.1
        )

        # 方向编码 (方位角0-360, 仰角-90~90)
        self._init_directions()

    def _init_directions(self):
        """初始化方向网格"""
        azimuth = np.linspace(-180, 180, self.n_directions)
        elevation = np.zeros(self.n_directions)
        dirs = np.stack([azimuth, elevation], axis=-1)
        self.register_buffer('directions', torch.tensor(dirs))

    def forward(self, azimuth: float, elevation: float = 0) -> torch.Tensor:
        """
        获取指定方向的HRTF

        Args:
            azimuth: 方位角 (度)
            elevation: 仰角 (度)

        Returns:
            hrtf: [n_channels]
        """
        # 找到最近方向
        dirs = self.directions[:, 0]
        az = torch.tensor(azimuth) if not isinstance(azimuth, torch.Tensor) else azimuth
        az_rad = az * np.pi / 180

        # 简化: 正弦响应
        phase = dirs * np.pi / 180
        hrtf = self.hrtf_coeffs * torch.cos(phase.unsqueeze(-1) - az_rad)
        return hrtf.mean(0)


class BinauralProcessor(nn.Module):
    """
    双耳处理器 - ITD + ILD + 空间定位
    """
    def __init__(self, n_channels: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.max_itd_us = 700  # 最大ITD (μs), 头宽~18cm

        # HRTF
        self.hrtf = HRTFModel(n_channels)

        # ITD估计器
        self.itd_estimator = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # ILD估计器
        self.ild_estimator = nn.Sequential(
            nn.Linear(n_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 空间定位 - 输入是 binaural_proj 的输出 128
        self.spatial_localizer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # azimuth, elevation
        )

        # 双耳投影
        self.binaural_proj = nn.Linear(n_channels * 2, 128)

    def compute_itd(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        计算ITD (Interaural Time Difference)

        Args:
            left: [B, T, C]
            right: [B, T, C]

        Returns:
            itd: [B] (μs)
        """
        # 简化: 延迟搜索
        B, T, C = left.shape
        max_delay = min(10, T - 1)

        itd = torch.zeros(B, device=left.device)

        for b in range(B):
            best_corr = -1.0
            best_delay = 0
            for delay in range(-max_delay, max_delay + 1):
                if delay < 0:
                    shifted = right[b, -delay:, :]
                    ref = left[b, :T+delay, :]
                elif delay > 0:
                    shifted = left[b, delay:, :]
                    ref = right[b, :T-delay, :]
                else:
                    shifted = left[b]
                    ref = right[b]

                if len(shifted) > 0:
                    corr = (shifted * ref).mean()
                    if corr > best_corr:
                        best_corr = corr
                        best_delay = delay

            # 假设1ms帧 → 1000μs
            itd[b] = best_delay * 1000

        return itd

    def compute_ild(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        计算ILD (Interaural Level Difference)

        Args:
            left/right: [B, T, C]

        Returns:
            ild: [B, C] (dB)
        """
        power_left = (left ** 2).mean(dim=1)  # [B, C]
        power_right = (right ** 2).mean(dim=1)

        ild = 10 * torch.log10(power_left / (power_right + 1e-8) + 1e-8)
        return ild

    def compute_spatial_location(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Dict:
        """
        计算空间位置

        Args:
            left/right: [B, T, C]

        Returns:
            location: {azimuth, elevation, distance}
        """
        # 合并双耳信息: [B, 2C]
        binaural = torch.cat([left, right], dim=-1).mean(dim=1)  # [B, 2C]
        binaural = self.binaural_proj(binaural)

        location = self.spatial_localizer(binaural)
        azimuth = location[:, 0] * 180  # 缩放到度
        elevation = location[:, 1] * 90

        return {
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': torch.ones_like(azimuth),
        }

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Dict:
        """
        处理双耳音频

        Args:
            left: [B, T, C]
            right: [B, T, C]

        Returns:
            binaural: {itd, ild, azimuth, elevation}
        """
        itd = self.compute_itd(left, right)
        ild = self.compute_ild(left, right)
        location = self.compute_spatial_location(left, right)

        return {
            'itd': itd,
            'ild': ild,
            'azimuth': location['azimuth'],
            'elevation': location['elevation'],
            'distance': location['distance'],
        }


def create_binaural_processor(n_channels: int = 128) -> BinauralProcessor:
    return BinauralProcessor(n_channels)


if __name__ == "__main__":
    print("=== Testing Binaural Processor ===")

    # 测试
    processor = BinauralProcessor(64)
    left = torch.randn(2, 100, 64)
    right = torch.randn(2, 100, 64)

    result = processor(left, right)

    print(f"ITD: {result['itd']}")
    print(f"ILD shape: {result['ild'].shape}")
    print(f"Azimuth: {result['azimuth']}")
    print(f"Elevation: {result['elevation']}")

    print("\n✓ Binaural test passed!")