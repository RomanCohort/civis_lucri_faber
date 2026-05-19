"""
言语感知机制 - 麦格尔效应/语音切分/范畴感知

Civis Lucri-Faber - 言语感知

关键机制:
1. 麦格尔效应 (Meyg effect) - 视觉辅助语音感知
2. 语音切分 - 检测词边界
3. 范畴感知 - 音素范畴化
4. 音素识别
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


# 音素表 (ARPAbet简化)
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


class PhonemeRecognizer(nn.Module):
    """
    音素识别器

    从acoustic features识别音素
    """
    def __init__(self, input_dim: int = 256, n_phones: int = 40):
        super().__init__()
        self.n_phones = n_phones

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 音素分类
        self.classifier = nn.Linear(256, n_phones)

    def forward(self, features: torch.Tensor) -> Dict:
        """
        Args:
            features: [B, input_dim]

        Returns:
            phonemes: [B, n_phones] (logits)
            top_phoneme: [B]
            confidence: [B]
        """
        encoded = self.encoder(features)
        logits = self.classifier(encoded)

        probs = F.softmax(logits, dim=-1)
        top_idx = probs.argmax(dim=-1)
        confidence = probs.gather(1, top_idx.unsqueeze(-1)).squeeze(-1)

        return {
            'logits': logits,
            'phoneme': top_idx,
            'confidence': confidence,
        }


class CategoricalPerception(nn.Module):
    """
    范畴感知

    关键机制: 连续语音流 → 离散范畴
    实现: 音素边界检测 + 范畴化
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()

        # 边界检测
        self.boundary_detector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 范畴化
        self.categorizer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 40),  # 40个音素范畴
        )

    def detect_boundaries(self, features: torch.Tensor) -> torch.Tensor:
        """
        检测音素边界

        features: [B, T, D]
        Returns: [B, T] 边界概率
        """
        B, T, D = features.shape
        boundaries = []

        for t in range(T):
            boundary = self.boundary_detector(features[:, t, :])
            boundaries.append(boundary)

        return torch.cat(boundaries, dim=1)

    def categorize(self, features: torch.Tensor) -> Dict:
        """
        范畴化

        Args:
            features: [B, D]

        Returns:
            category: [B] 音素索引
            category_onehot: [B, 40]
        """
        logits = self.categorizer(features)
        category = logits.argmax(dim=-1)

        onehot = F.one_hot(category, num_classes=40).float()

        return {
            'category': category,
            'onehot': onehot,
            'logits': logits,
        }

    def forward(self, features: torch.Tensor) -> Dict:
        """
        范畴感知前向

        Args:
            features: [B, T, D] 或 [B, D]

        Returns:
            boundaries: [B, T]
            categories: [B] 或 [B, T]
        """
        if features.dim() == 3:
            # 多帧
            boundaries = self.detect_boundaries(features)
            # 取最后一帧的范畴 (简化)
            features_last = features[:, -1, :]
            cat_result = self.categorize(features_last)
            return {
                'boundaries': boundaries,
                'category': cat_result['category'],
                'onehot': cat_result['onehot'],
            }
        else:
            return self.categorize(features)


class SpeechSegmentation(nn.Module):
    """
    语音切分

    检测词边界、短语边界
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()

        # 能量检测
        self.energy_detector = nn.Linear(input_dim, 1)

        # 零交叉率
        self.zcr_detector = nn.Linear(input_dim, 1)

        # 边界分类
        self.boundary_classifier = nn.Sequential(
            nn.Linear(input_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 无边界 / 音素边界 / 词边界
        )

    def compute_energy(self, audio: torch.Tensor) -> torch.Tensor:
        """计算短时能量"""
        return (audio ** 2).mean(dim=-1)

    def compute_zcr(self, audio: torch.Tensor) -> torch.Tensor:
        """计算零交叉率"""
        sign = torch.sign(audio)
        zcr = (sign[:, 1:] != sign[:, :-1]).float().mean(dim=-1)
        return zcr

    def detect_boundaries(
        self,
        features: torch.Tensor,
        energy: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        检测边界

        Args:
            features: [B, T, D]
            energy: [B, T]

        Returns:
            boundaries: 类型
        """
        B, T, D = features.shape

        # 简化: 基于能量变化
        if energy is None:
            energy = self.compute_energy(features)

        # 能量差分
        energy_diff = energy[:, 1:] - energy[:, :-1]

        # 边界分数
        scores = self.boundary_classifier(features.mean(dim=1))
        boundary_type = scores.argmax(dim=-1)

        return {
            'boundary_type': boundary_type,
            'scores': scores,
            'energy_diff': energy_diff if energy is not None else None,
        }

    def forward(self, features: torch.Tensor) -> Dict:
        return self.detect_boundaries(features)


class McGurkEffect(nn.Module):
    """
    麦格尔效应

    视觉-听觉整合增强语音感知
    对应: 颞上回 + 视觉皮层
    """
    def __init__(self, audio_dim: int = 256, visual_dim: int = 256):
        super().__init__()

        # 听觉编码
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.ReLU(),
        )

        # 视觉编码
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 128),
            nn.ReLU(),
        )

        # 整合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.Sigmoid(),
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 40),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        麦格尔效应

        Args:
            audio_features: [B, audio_dim]
            visual_features: [B, visual_dim] (可选)

        Returns:
            fused: 融合后的音素预测
        """
        audio_enc = self.audio_encoder(audio_features)

        if visual_features is not None:
            visual_enc = self.visual_encoder(visual_features)

            # 门控融合
            combined = torch.cat([audio_enc, visual_enc], dim=-1)
            gate = self.fusion_gate(combined)

            fused = self.fusion(combined)

            return {
                'fused': fused,
                'audio_only': audio_enc,
                'visual_only': visual_enc,
                'gate': gate,
            }
        else:
            # 仅听觉
            return {
                'fused': audio_enc,
                'audio_only': audio_enc,
                'visual_only': None,
                'gate': torch.ones_like(audio_enc[:, :1]),
            }


class PhoneticPerception(nn.Module):
    """
    完整言语感知系统

    整合:
    1. 音素识别
    2. 范畴感知
    3. 语音切分
    4. 麦格尔效应
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()

        self.phoneme_recognizer = PhonemeRecognizer(input_dim)
        self.categorical = CategoricalPerception(input_dim)
        self.segmentation = SpeechSegmentation(input_dim)
        self.mcgurk = McGurkEffect(input_dim, input_dim)

    def forward(
        self,
        features: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        言语感知

        Args:
            features: [B, input_dim]
            visual_features: [B, input_dim] (可选)

        Returns:
            phoneme: 识别结果
            category: 范畴
            boundaries: 边界
            fused: 融合结果
        """
        # 1. 音素识别
        phoneme_result = self.phoneme_recognizer(features)

        # 2. 范畴感知
        category = self.categorical(features)

        # 3. 麦格尔效应
        mcgurk_result = self.mcgurk(features, visual_features)

        return {
            'phoneme': phoneme_result['phoneme'],
            'phoneme_confidence': phoneme_result['confidence'],
            'category': category['category'],
            'boundaries': category.get('boundaries', None),
            'fused': mcgurk_result['fused'],
            'audio_features': mcgurk_result['audio_only'],
            'visual_features': mcgurk_result['visual_only'],
        }


def create_phonetic_perception(input_dim: int = 256) -> PhoneticPerception:
    return PhoneticPerception(input_dim)


if __name__ == "__main__":
    print("=== Testing Phonetic Perception ===")

    # 测试音素识别
    print("[1] Phoneme Recognizer")
    recognizer = PhonemeRecognizer(256)
    features = torch.randn(2, 256)
    result = recognizer(features)
    print(f"  - phoneme: {result['phoneme']}")
    print(f"  - confidence: {result['confidence']}")

    # 测试范畴感知
    print("\n[2] Categorical Perception")
    categorical = CategoricalPerception(256)
    result = categorical(features)
    print(f"  - category: {result['category']}")

    # 测试麦格尔
    print("\n[3] McGurk Effect")
    mcgurk = McGurkEffect(256, 256)
    audio = torch.randn(2, 256)
    visual = torch.randn(2, 256)
    result = mcgurk(audio, visual)
    print(f"  - gate shape: {result['gate'].shape}")

    # 测试完整系统
    print("\n[4] Full Phonetic Perception")
    phonetic = create_phonetic_perception(256)
    result = phonetic(features, visual)
    print(f"  - phoneme: {result['phoneme']}")
    print(f"  - fused: {result['fused'].shape}")

    print("\n✓ Phonetic perception tests passed!")