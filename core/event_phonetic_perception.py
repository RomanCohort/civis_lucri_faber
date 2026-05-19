"""
言语感知 - 事件驱动版本

Simulacrum - Event-Driven Phonetic Perception

基于脉冲神经网络:
1. 音素识别 → Spiking
2. 范畴感知 → LIF
3. 语音切分 → Spike Events
4. 麦格尔效应 → Spiking Fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class SpikingPhonemeRecognizer(nn.Module):
    """
    Spiking 音素识别器

    从脉冲特征识别音素
    """
    def __init__(self, input_dim: int = 256, n_phones: int = 40):
        super().__init__()
        self.n_phones = n_phones

        # 编码层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # LIF 神经元
        self.lif = nn.Linear(128, n_phones)

    def forward(self, features: torch.Tensor) -> Dict:
        """
        Args:
            features: [B, input_dim]

        Returns:
            {phoneme, confidence, spikes}
        """
        # 编码
        encoded = self.encoder(features)

        # LIF 发射
        v_mem = encoded
        spikes = (v_mem > 0).float()
        v_mem = v_mem * (1 - spikes) - spikes * 75e-3

        # 音素分类
        logits = self.lif(v_mem)
        probs = F.softmax(logits, dim=-1)
        top_idx = probs.argmax(dim=-1)
        confidence = probs.gather(1, top_idx.unsqueeze(-1)).squeeze(-1)

        return {
            'phoneme': top_idx,
            'confidence': confidence,
            'logits': logits,
            'spikes': spikes,
        }


class SpikingCategoricalPerception(nn.Module):
    """
    Spiking 范畴感知

    脉冲边界检测 + 范畴化
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

        # 范畴化 LIF
        self.categorizer_lif = nn.Linear(input_dim, 40)

    def detect_boundaries(self, features: torch.Tensor) -> torch.Tensor:
        """检测音素边界"""
        return self.boundary_detector(features)

    def categorize(self, features: torch.Tensor) -> Dict:
        """范畴化"""
        # LIF 发射
        v_mem = features
        spikes = (v_mem > 0).float()
        v_mem = v_mem * (1 - spikes) - spikes * 75e-3

        logits = self.categorizer_lif(v_mem)
        category = logits.argmax(dim=-1)

        return {
            'category': category,
            'spikes': spikes,
            'logits': logits,
        }

    def forward(self, features: torch.Tensor) -> Dict:
        """范畴感知"""
        boundaries = self.detect_boundaries(features)
        category_result = self.categorize(features)

        return {
            'boundaries': boundaries,
            'category': category_result['category'],
            'spikes': category_result['spikes'],
        }


class SpikingSpeechSegmentation(nn.Module):
    """
    Spiking 语音切分

    基于脉冲的边界检测
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()

        # 能量检测
        self.energy_detector = nn.Linear(input_dim, 1)

        # 边界分类器
        self.boundary_classifier = nn.Sequential(
            nn.Linear(input_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 边界类型
        )

    def compute_energy_from_spikes(self, spikes: torch.Tensor) -> torch.Tensor:
        """从脉冲计算能量"""
        energy = (spikes ** 2).sum(-1)
        return energy

    def forward(self, features: torch.Tensor) -> Dict:
        """
        检测边界

        Args:
            features: [B, D]

        Returns:
            {boundary_type, scores}
        """
        # 简化: 基于能量变化
        energy = self.energy_detector(features)

        # 边界类型 (LIF)
        boundary_type = (energy.squeeze() > 0.5).long()

        return {
            'boundary_type': boundary_type,
            'energy': energy,
        }


class SpikingMcGurkEffect(nn.Module):
    """
    Spiking 麦格尔效应

    视觉-听觉脉冲整合
    """
    def __init__(self, audio_dim: int = 256, visual_dim: int = 256):
        super().__init__()

        # 听觉 LIF
        self.audio_lif = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.ReLU(),
        )

        # 视觉 LIF
        self.visual_lif = nn.Sequential(
            nn.Linear(visual_dim, 128),
            nn.ReLU(),
        )

        # 融合门控 (LIF)
        self.fusion_gate = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.Sigmoid(),
        )

        # 融合 LIF
        self.fusion_lif = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 40),
        )

    def forward(self,
              audio_features: torch.Tensor,
              visual_features: Optional[torch.Tensor] = None) -> Dict:
        """
        麦格尔效应

        Args:
            audio_features: [B, audio_dim]
            visual_features: [B, visual_dim] (可选)

        Returns:
            {fused, audio_spikes, visual_spikes, gate}
        """
        # 听觉 LIF
        audio_enc = self.audio_lif(audio_features)
        audio_spikes = (audio_enc > 0).float()
        audio_enc = audio_enc * (1 - audio_spikes)

        if visual_features is not None:
            # 视觉 LIF
            visual_enc = self.visual_lif(visual_features)
            visual_spikes = (visual_enc > 0).float()
            visual_enc = visual_enc * (1 - visual_spikes)

            # 融合门控
            combined = torch.cat([audio_enc, visual_enc], dim=-1)
            gate = self.fusion_gate(combined)

            # 融合 LIF
            fused = self.fusion_lif(combined)

            return {
                'fused': fused,
                'audio_spikes': audio_spikes,
                'visual_spikes': visual_spikes,
                'gate': gate,
            }
        else:
            return {
                'fused': audio_enc,
                'audio_spikes': audio_spikes,
                'visual_spikes': None,
                'gate': torch.ones_like(audio_enc[:, :1]),
            }


class EventDrivenPhoneticPerception(nn.Module):
    """
    事件驱动言语感知

    整合:
    1. Spiking 音素识别
    2. Spiking 范畴感知
    3. Spiking 语音切分
    4. Spiking 麦格尔效应
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()

        self.phoneme_recognizer = SpikingPhonemeRecognizer(input_dim)
        self.categorical = SpikingCategoricalPerception(input_dim)
        self.segmentation = SpikingSpeechSegmentation(input_dim)
        self.mcgurk = SpikingMcGurkEffect(input_dim, input_dim)

    def forward(self,
              features: torch.Tensor,
              visual_features: Optional[torch.Tensor] = None) -> Dict:
        """
        言语感知

        Args:
            features: [B, input_dim]
            visual_features: [B, input_dim] (可选)

        Returns:
            {phoneme, category, boundaries, fused}
        """
        # 1. 音素识别
        phoneme_result = self.phoneme_recognizer(features)

        # 2. 范畴感知
        categorical = self.categorical(features)

        # 3. 麦格尔效应
        mcgurk_result = self.mcgurk(features, visual_features)

        return {
            'phoneme': phoneme_result['phoneme'],
            'phoneme_confidence': phoneme_result['confidence'],
            'category': categorical['category'],
            'boundaries': categorical.get('boundaries'),
            'fused': mcgurk_result['fused'],
            'audio_spikes': mcgurk_result['audio_spikes'],
            'visual_spikes': mcgurk_result.get('visual_spikes'),
            'gate': mcgurk_result['gate'],
        }


def create_event_phonetic(input_dim: int = 256) -> EventDrivenPhoneticPerception:
    return EventDrivenPhoneticPerception(input_dim)


# ============ 测试 ============

if __name__ == "__main__":
    print("=== Testing Event-Driven Phonetic Perception ===\n")

    # 测试音素识别
    print("[1] Spiking Phoneme Recognizer")
    recognizer = SpikingPhonemeRecognizer(256)
    features = torch.randn(2, 256)
    result = recognizer(features)
    print(f"  - phoneme: {result['phoneme'].shape}")
    print(f"  - confidence: {result['confidence'].shape}")
    print(f"  - spikes: {result['spikes'].shape}")

    # 测试范畴感知
    print("\n[2] Spiking Categorical Perception")
    categorical = SpikingCategoricalPerception(256)
    result = categorical(features)
    print(f"  - category: {result['category'].shape}")
    print(f"  - spikes: {result['spikes'].shape}")

    # 测试麦格尔
    print("\n[3] Spiking McGurk Effect")
    mcgurk = SpikingMcGurkEffect(256, 256)
    audio = torch.randn(2, 256)
    visual = torch.randn(2, 256)
    result = mcgurk(audio, visual)
    print(f"  - gate: {result['gate'].shape}")
    print(f"  - fused: {result['fused'].shape}")

    # 测试完整系统
    print("\n[4] Event-Driven Phonetic Perception")
    phonetic = create_event_phonetic(256)
    result = phonetic(features, visual)
    print(f"  - phoneme: {result['phoneme'].shape}")
    print(f"  - category: {result['category'].shape}")
    print(f"  - fused: {result['fused'].shape}")

    print("\nAll tests passed!")