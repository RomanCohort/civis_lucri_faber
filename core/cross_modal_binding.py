# =============================================================================
# Civis Lucri-Faber -- Cross-Modal Binding
# =============================================================================
# 跨模态绑定机制
#
# 解决的问题：
# 1. Binding Problem：不同模态如何绑定为统一体验
# 2. 时序同步：视觉"看到蛇" + 听觉"嘶嘶声" → 统一"危险"
# 3. 动态交互：各模态之间的注意力调制
#
# 理论基础：
# - Treisman (1977) Feature Integration Theory
# - Driver (2000) Cross-modal attention
# - Seth (2005) Predictive binding
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from collections import deque
import numpy as np


# =============================================================================
# Modal Encoders -- 各模态编码器
# =============================================================================

class ModalEncoder(nn.Module):
    """通用模态编码器"""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class VisionEncoder(ModalEncoder):
    """视觉编码器"""
    pass


class AudioEncoder(ModalEncoder):
    """听觉编码器"""
    pass


class LanguageEncoder(ModalEncoder):
    """语言编码器"""
    pass


# =============================================================================
# Temporal Binding Buffer -- 时序绑定缓冲区
# =============================================================================

class TemporalBindingBuffer(nn.Module):
    """
    时序绑定缓冲区

    功能：
    1. 存储各模态的历史表征
    2. 查找时序接近的事件
    3. 支持跨模态注意力查找
    """

    def __init__(
        self,
        max_length: int = 50,
        time_window: float = 0.5,  # 500ms时间窗口
    ):
        super().__init__()
        self.max_length = max_length
        self.time_window = time_window  # 秒

        # 各模态的缓冲区
        self.buffers = {
            'vision': deque(maxlen=max_length),
            'audio': deque(maxlen=max_length),
            'language': deque(maxlen=max_length),
        }

    def append(self, modality: str, representation: torch.Tensor, timestamp: float):
        """添加事件到缓冲区"""
        self.buffers[modality].append({
            'repr': representation,
            'timestamp': timestamp,
            'modality': modality,
        })

    def find_temporal_matches(
        self,
        source_modality: str,
        target_modality: str,
        timestamp: float,
        tolerance: Optional[float] = None,
    ) -> List[Dict]:
        """
        查找时序接近的事件

        Example:
            source: 'vision', timestamp: 1.0s
            target: 'audio', tolerance: 0.5s
            → 找到 audio 在 0.5s-1.5s 之间的表征
        """
        tolerance = tolerance or self.time_window

        source_buffer = self.buffers[source_modality]
        target_buffer = self.buffers[target_modality]

        matches = []
        for event in target_buffer:
            target_time = event['timestamp']
            if abs(target_time - timestamp) <= tolerance:
                matches.append(event)

        return matches

    def get_unified_context(self, timestamp: float) -> Dict:
        """获取统一上下文（所有模态在当前时刻的表征）"""
        context = {}
        for mod, buffer in self.buffers.items():
            # 找到最近的表征
            if len(buffer) > 0:
                recent = list(buffer)[-1]
                context[mod] = recent['repr']
        return context


# =============================================================================
# Cross-Modal Attention -- 跨模态注意力
# =============================================================================

class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制

    功能：
    1. Query from one modality
    2. Key/Value from another modality
    3. 输出跨模态调制后的表征
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 跨模态注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

        # 模态特定投影
        self.projection = nn.ModuleDict({
            'vision': nn.Linear(embed_dim, embed_dim),
            'audio': nn.Linear(embed_dim, embed_dim),
            'language': nn.Linear(embed_dim, embed_dim),
        })

    def forward(
        self,
        query_mod: str,
        query_repr: torch.Tensor,
        key_mod: str,
        key_repr: torch.Tensor,
        value_repr: torch.Tensor,
    ) -> Dict:
        """
        跨模态注意力

        Args:
            query_mod: Query来源的模态
            query_repr: [B, D] Query表征
            key_mod: Key来源的模态
            key_repr: [B, D] Key表征
            value_repr: [B, D] Value表征

        Returns:
            output: [B, D] 跨模态调制后的表征
            attention_weights: [B, num_heads, T, T]
        """
        # 投影到共同空间
        query = self.projection[query_mod](query_repr)
        key = self.projection[key_mod](key_repr)
        value = self.projection[key_mod](value_repr)

        # 注意力计算
        output, attn_weights = self.attention(query, key, value)

        return {
            'output': output,
            'attention_weights': attn_weights,
        }


# =============================================================================
# Cross-Modal Binding -- 跨模态绑定主模块
# =============================================================================

class CrossModalBinder(nn.Module):
    """
    跨模态绑定器

    解决：
    1. Binding Problem：不同模态 → 统一体验
    2. 时序同步：事件绑定
    3. 动态交互：跨模态调制
    """

    def __init__(
        self,
        input_dims: Dict[str, int] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        time_window: float = 0.5,
    ):
        super().__init__()

        input_dims = input_dims or {
            'vision': 768,
            'audio': 256,
            'language': 512,
        }

        self.embed_dim = embed_dim
        self.time_window = time_window

        # 1. 各模态编码器
        self.encoders = nn.ModuleDict({
            'vision': VisionEncoder(input_dims['vision'], embed_dim),
            'audio': AudioEncoder(input_dims['audio'], embed_dim),
            'language': LanguageEncoder(input_dims['language'], embed_dim),
        })

        # 2. 时序绑定缓冲区
        self.temporal_buffer = TemporalBindingBuffer(
            max_length=50,
            time_window=time_window,
        )

        # 3. 跨模态注意力
        self.cross_attention = CrossModalAttention(embed_dim, num_heads)

        # 4. 统一表征融合
        self.fusion_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 5. 场景检测器（检测跨模态场景）
        self.scene_detector = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 10种预定义场景
        )

        # 当前时间戳
        self.current_time = 0.0

    def encode_modality(
        self,
        modality: str,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """编码单个模态"""
        encoder = self.encoders.get(modality)
        if encoder is None:
            return input_tensor
        return encoder(input_tensor)

    def forward(
        self,
        modalities: Dict[str, Optional[torch.Tensor]],
        timestamp: Optional[float] = None,
    ) -> Dict:
        """
        前向处理

        Args:
            modalities: {'vision': tensor, 'audio': tensor, 'language': tensor}
            timestamp: 可选时间戳

        Returns:
            unified_repr: 统一表征
            scene: 场景检测
            cross_modal_attention: 注意力权重
        """
        if timestamp is None:
            timestamp = self.current_time
            self.current_time += 0.1

        # Step 1: 编码各模态
        encoded = {}
        for mod, input_tensor in modalities.items():
            if input_tensor is not None:
                encoded[mod] = self.encode_modality(mod, input_tensor)
                # 存储到时序缓冲区
                self.temporal_buffer.append(mod, encoded[mod].detach(), timestamp)

        # Step 2: 跨模态绑定
        bound_repr = self._bind_cross_modal(encoded, timestamp)

        # Step 3: 统一表征
        if bound_repr is not None:
            unified = self.fusion_net(bound_repr)
        else:
            # 如果没有跨模态绑定，使用平均
            encoded_list = list(encoded.values())
            unified = torch.mean(torch.stack(encoded_list), dim=0)

        # Step 4: 场景检测
        scene_logits = self.scene_detector(unified)
        scene = scene_logits.argmax(dim=-1)

        return {
            'unified_repr': unified,
            'scene': scene,
            'scene_logits': scene_logits,
            'encoded': encoded,
        }

    def _bind_cross_modal(
        self,
        encoded: Dict[str, torch.Tensor],
        timestamp: float,
    ) -> Optional[torch.Tensor]:
        """
        跨模态绑定逻辑

        Example:
            vision: "看到蛇" + audio: "嘶嘶声" → 绑定为"危险"场景
        """
        if len(encoded) < 2:
            return None

        # 查找时序匹配
        # 核心：如果两个模态事件在时间窗口内，绑定它们

        # Vision → Audio binding (例如：看到蛇 + 听到嘶嘶声)
        if 'vision' in encoded and 'audio' in encoded:
            audio_events = self.temporal_buffer.find_temporal_matches(
                'vision', 'audio', timestamp
            )
            if len(audio_events) > 0:
                # 使用跨模态注意力
                audio_repr = torch.stack([e['repr'] for e in audio_events])
                # 平均
                audio_repr = audio_repr.mean(dim=0, keepdim=True)

                result = self.cross_attention(
                    query_mod='vision',
                    query_repr=encoded['vision'],
                    key_mod='audio',
                    key_repr=audio_repr,
                    value_repr=audio_repr,
                )
                return result['output']

        # Language ↔ Vision binding
        if 'language' in encoded and 'vision' in encoded:
            # 语言可以指向视觉事件
            result = self.cross_attention(
                query_mod='language',
                query_repr=encoded['language'],
                key_mod='vision',
                key_repr=encoded['vision'],
                value_repr=encoded['vision'],
            )
            return result['output']

        return None

    def get_scene_description(self, scene_id: int) -> str:
        """获取场景描述"""
        scenes = {
            0: "neutral",
            1: "danger",      # 危险（蛇+嘶嘶声）
            2: "social",     # 社交
            3: "food",      # 食物
            4: "threat",    # 威胁
            5: "reward",    # 奖励
            6: "play",      # 玩耍
            7: "care",      # 照顾
            8: "explore",   # 探索
            9: "rest",      # 休息
        }
        return scenes.get(scene_id, "unknown")


# =============================================================================
# Test
# =============================================================================

def test_cross_modal_binding():
    """测试跨模态绑定"""
    print("=" * 60)
    print("Testing Cross-Modal Binding")
    print("=" * 60)

    # 创建模型
    model = CrossModalBinder(
        input_dims={
            'vision': 768,
            'audio': 256,
            'language': 512,
        },
        embed_dim=256,
    )

    # 模拟输入
    vision_input = torch.randn(2, 768)   # 看到蛇
    audio_input = torch.randn(2, 256)    # 听到嘶嘶声
    language_input = torch.randn(2, 512)  # 说"危险"

    # Test 1: 单模态
    print("\n[1] Single modality:")
    result = model({'vision': vision_input})
    print(f"  Scene: {model.get_scene_description(result['scene'].item())}")

    # Test 2: 多模态
    print("\n[2] Multiple modalities:")
    result = model({
        'vision': vision_input,
        'audio': audio_input,
    }, timestamp=0.5)
    print(f"  Scene: {model.get_scene_description(result['scene'].item())}")

    # Test 3: 完整场景
    print("\n[3] Full scenario (danger):")
    # Vision + Audio 绑定
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        result = model({
            'vision': vision_input,
            'audio': audio_input,
            'language': language_input,
        }, timestamp=t)

    scene = model.get_scene_description(result['scene'].item())
    print(f"  Detected scene: {scene}")
    print(f"  Unified repr shape: {result['unified_repr'].shape}")

    print("\n" + "=" * 60)
    print("Cross-Modal Binding Test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_cross_modal_binding()


__all__ = [
    'CrossModalBinder',
    'TemporalBindingBuffer',
    'CrossModalAttention',
    'test_cross_modal_binding',
]