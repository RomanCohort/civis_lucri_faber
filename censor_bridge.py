"""
Censor桥接模块 (更新版)

直接调用D:\censor项目的双通路架构:
- Fast pathway: 3D ResNet (快速通路)
- Slow pathway: 3D Swin (慢速通路)
"""
import sys
import os

# 添加censor路径
CENSOR_PATH = r"D:\censor"
if CENSOR_PATH not in sys.path:
    sys.path.insert(0, CENSOR_PATH)

# 尝试直接加载配置和数据
try:
    from config.defaults import FAST_PATHWAY_CONFIG, SLOW_PATHWAY_CONFIG
    CENSOR_CONFIG_AVAILABLE = True
except ImportError:
    CENSOR_CONFIG_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FastPathway(nn.Module):
    """
    快速通路 (3D ResNet)

    模拟丘脑-杏仁核快速通路:
    - 上丘 -> 枕核 -> 杏仁核
    - 用于快速视觉检测
    """

    def __init__(
        self,
        in_channels: int = 2,  # TV-L1 optical flow
        base_channels: int = 32,
    ):
        super().__init__()

        # 3D卷积
        self.conv1 = nn.Conv3d(in_channels, base_channels, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(3, stride=2, padding=1)

        # 残差块
        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = []
        layers.append(nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm3d(out_c))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(nn.Conv3d(out_c, out_c, 3, padding=1))
            layers.append(nn.BatchNorm3d(out_c))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class SlowPathway(nn.Module):
    """
    慢速通路 (3D Swin-like)

    模拟皮层视觉通路的慢速精细识别:
    - LGN -> V1 -> V2 -> V4 -> IT
    - 需要更多计算但更精确
    """

    def __init__(
        self,
        in_channels: int = 6,  # RGB + rPPG
        base_channels: int = 64,
    ):
        super().__init__()

        # 3D卷积
        self.conv1 = nn.Conv3d(in_channels, base_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # 层级
        self.layer1 = self._make_layer(base_channels, base_channels * 2, 2)
        self.layer2 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, in_c, out_c, blocks, stride=1):
        layers = []
        layers.append(nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm3d(out_c))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(nn.Conv3d(out_c, out_c, 3, padding=1))
            layers.append(nn.BatchNorm3d(out_c))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class DualPathwayCortex(nn.Module):
    """
    双通路视觉皮层

    整合快速通路 + 慢速通路
    等生物学:
    - Fast: 丘脑-杏仁核 (快速无意识)
    - Slow: 皮层视觉 (精细识别)
    """

    def __init__(
        self,
        fast_channels: int = 2,
        slow_channels: int = 6,
    ):
        super().__init__()

        # 两条通路
        self.fast = FastPathway(fast_channels)
        self.slow = SlowPathway(slow_channels)

        # 融合
        # Fast: 32 -> 64 (layer2输出) = 64
        # Slow: 64 -> 128 -> 256 (layer2输出) = 256
        # Total: 64 + 256 = 320
        self.fusion = nn.Sequential(
            nn.Linear(64 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(
        self,
        fast_input: torch.Tensor = None,
        slow_input: torch.Tensor = None,
    ) -> Dict:
        """
        Args:
            fast_input: [B, 2, T, H, W] 光流
            slow_input: [B, 6, T, H, W] RGB+PPG
        """
        features = []

        # 快速通路
        if fast_input is not None:
            fast_feat = self.fast(fast_input)
            features.append(fast_feat)

        # 慢速通路
        if slow_input is not None:
            slow_feat = self.slow(slow_input)
            features.append(slow_feat)

        if not features:
            return {'embedding': None, 'salience': 0.5}

        # 拼接融合
        if len(features) == 1:
            embedding = features[0]
        else:
            embedding = torch.cat(features, dim=-1)

        embedding = self.fusion(embedding)

        salience = torch.sigmoid(embedding.mean())

        return {
            'embedding': embedding,
            'salience': salience.item(),
            'fast_features': fast_feat if fast_input is not None else None,
            'slow_features': slow_feat if slow_input is not None else None,
        }


class CensorVision(nn.Module):
    """
    Censor视觉 + 心理学机制
    """

    def __init__(
        self,
        use_fast: bool = True,
        use_slow: bool = True,
    ):
        super().__init__()

        self.use_fast = use_fast
        self.use_slow = use_slow

        if use_fast and use_slow:
            self.pathway = DualPathwayCortex()
        elif use_slow:
            self.pathway = SlowPathway()
        else:
            self.pathway = FastPathway()

        # 心理学组件
        self.visual_attention = VisualAttentionalBias()
        self.emotion_detection = VisualEmotionDetector()
        self.scene_memory = VisualSceneMemory()
        self.threat_detection = ThreatDetector()

    def forward(
        self,
        optical_flow: torch.Tensor = None,  # [B, 2, T, H, W]
        rgb_ppg: torch.Tensor = None,  # [B, 6, T, H, W]
    ) -> Dict:
        """前向传播"""
        if self.use_fast and self.use_slow:
            result = self.pathway(optical_flow, rgb_ppg)
        elif self.use_slow:
            result = {'embedding': self.pathway(rgb_ppg)}
        else:
            result = {'embedding': self.pathway(optical_flow)}

        if result.get('embedding') is not None:
            result['salience'] = torch.sigmoid(result['embedding'].mean()).item()

        return result


# ============ 视觉心理学组件 ============

class VisualAttentionalBias(nn.Module):
    """
    视觉注意偏差

    心理学: 整体优先 (Global Precedence)
    实现: 先看整体再看细节
    """
    def __init__(self):
        super().__init__()
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.detail_bias = nn.Parameter(torch.zeros(1))

    def process(self, image_features: torch.Tensor):
        """整体→细节处理"""
        # 整体提取
        global_out = image_features.mean(dim=[-2, -1])
        # 细节提取
        detail_out = F.adaptive_max_pool2d(image_features, 1)
        return global_out + detail_out


class VisualEmotionDetector(nn.Module):
    """
    视觉情绪检测

    心理学: 面孔/动作情绪识别
    """
    def __init__(self):
        super().__init__()
        self.emotion_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 6种情绪
        )

    def detect(self, visual_features: torch.Tensor):
        """检测情绪"""
        return self.emotion_classifier(visual_features)


class VisualSceneMemory(nn.Module):
    """
    视觉场景记忆

    心理学: 场景识别与位置记忆
    """
    def __init__(self):
        super().__init__()
        self.scene_prototypes = nn.Parameter(torch.randn(20, 64))
        self.location_memory = nn.Parameter(torch.randn(20, 64))

    def recognize_scene(self, features: torch.Tensor):
        """场景识别"""
        sim = F.cosine_similarity(features.unsqueeze(0), self.scene_prototypes.unsqueeze(1), dim=-1)
        return sim.argmax(dim=-1)

    def remember_location(self, features: torch.Tensor):
        """位置记忆"""
        return features


class ThreatDetector(nn.Module):
    """
    威胁检测器

    心理学: 快速威胁检测 (恐惧系统)
    实现: 快速通路 → 杏仁核
    """
    def __init__(self):
        super().__init__()
        self.threat_classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.threat_threshold = nn.Parameter(torch.tensor(0.6))

    def detect(self, visual_features: torch.Tensor):
        """快速威胁检测"""
        threat_level = self.threat_classifier(visual_features)
        is_threat = threat_level > self.threat_threshold
        return is_threat, threat_level


# ============ 视觉剪枝机制 ============

class VisualSpatialPruner(nn.Module):
    """视觉空间剪枝 - 只处理显著区域"""
    def __init__(self):
        super().__init__()
        self.salience_threshold = nn.Parameter(torch.tensor(0.5))

    def compute_salience(self, features):
        return features.abs().mean(dim=[-2, -1])

    def select_roi(self, features, top_k=4):
        """选择显著ROI"""
        sal = self.compute_salience(features)
        _, indices = sal.view(-1).topk(top_k)
        return indices


class ChannelPruner(nn.Module):
    """通道剪枝"""
    def __init__(self, n_channels=64):
        super().__init__()
        self.importance = nn.Parameter(torch.ones(n_channels))

    def prune(self, keep_ratio=0.5):
        n_keep = int(self.importance.numel() * keep_ratio)
        _, indices = self.importance.topk(n_keep)
        return indices

    def get_sparsity(self):
        return (self.importance == 0).float().mean().item()


def create_censor_vision(
    mode: str = "dual",  # "dual", "fast", "slow"
) -> CensorVision:
    """创建Censor视觉处理器"""
    if mode == "dual":
        return CensorVision(use_fast=True, use_slow=True)
    elif mode == "fast":
        return CensorVision(use_fast=True, use_slow=False)
    else:
        return CensorVision(use_fast=False, use_slow=True)


__all__ = [
    'FastPathway',
    'SlowPathway',
    'DualPathwayCortex',
    'CensorVision',
    'create_censor_vision',
]