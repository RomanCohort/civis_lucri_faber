"""
角回模块 — 跨模态翻译器 (Angular Gyrus)

位于顶叶、颞叶和枕叶交界处，大脑的"跨模态翻译器"。
负责将视觉、听觉和概念完美融合。

核心功能：
1. 模态投影 — 将不同维度的模态输入映射到统一空间
2. NxN 翻译矩阵 — 任何模态可以翻译成任何其他模态
3. 共享语义空间 (Interlingua) — 所有模态汇聚的统一表征
4. 预测性跨模态补全 — 给定一个模态，预测缺失的其他模态
5. 时序绑定 — 时间窗口内的跨模态事件自动关联

参考:
  - Binder et al. (2005) - Angular Gyrus and semantic processing
  - Seghier (2013) - The Angular Gyrus: multiple functions
  - Bonner et al. (2013) - Cross-modal representation in Angular Gyrus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from collections import deque
from dataclasses import dataclass, field
import time

MODALITIES = ['vision', 'audio', 'language']


# ============ 模态投影器 ============

class ModalityProjector(nn.Module):
    """将各模态的不同维度投影到统一的 embed_dim"""

    def __init__(self, input_dims: Dict[str, int], embed_dim: int = 256):
        super().__init__()
        self.projectors = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for mod, dim in input_dims.items()
        })

    def forward(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        return self.projectors[modality](x)


# ============ NxN 翻译矩阵 ============

class TranslationMatrix(nn.Module):
    """
    跨模态翻译矩阵

    3×3 个 MultiheadAttention，实现任何模态对之间的双向翻译。
    对角线 = 自注意力（同模态精炼）。
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.translators = nn.ModuleDict()
        for src in MODALITIES:
            for tgt in MODALITIES:
                key = f"{src}_to_{tgt}"
                self.translators[key] = nn.MultiheadAttention(
                    embed_dim=embed_dim, num_heads=num_heads,
                    batch_first=True, dropout=0.1,
                )

    def translate(self, query_mod: str, kv_mod: str,
                  query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        将 kv_mod 模态的信息翻译到 query_mod 的视角

        Args:
            query_mod: 目标模态（查询来源）
            kv_mod: 源模态（键值来源）
            query: [B, 1, D]
            key_value: [B, 1, D]

        Returns:
            translated: [B, 1, D]
        """
        key = f"{query_mod}_to_{kv_mod}"
        out, _ = self.translators[key](query, key_value, key_value)
        return out

    def translate_all(self, projected: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        执行所有模态对之间的翻译

        Returns:
            translations[src][tgt] = 用 tgt 查询 src 的结果
        """
        present = list(projected.keys())
        results = {}
        for src in present:
            results[src] = {}
            src_repr = projected[src].unsqueeze(1) if projected[src].dim() == 2 else projected[src]
            for tgt in present:
                tgt_repr = projected[tgt].unsqueeze(1) if projected[tgt].dim() == 2 else projected[tgt]
                translated = self.translate(tgt, src, tgt_repr, src_repr)
                results[src][tgt] = translated.squeeze(1)
        return results


# ============ 共享语义空间 ============

class SemanticInterlingua(nn.Module):
    """
    共享语义空间 (Interlingua)

    将所有模态的翻译结果汇聚为统一的语义表征。
    所有模态在此空间中对齐——语义等价的输入应接近。
    """

    def __init__(self, embed_dim: int = 256, n_modalities: int = 3):
        super().__init__()
        # 每个模态对翻译结果的加权
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, embed_dim),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Linear(embed_dim * n_modalities, embed_dim)

    def forward(self, projected: Dict[str, torch.Tensor],
                translations: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        将投影 + 翻译结果融合为统一表征

        对每个模态（包括缺失的）：原始投影 + 所有翻译结果的平均。
        缺失模态用零向量填充，保证 concat 维度恒定。
        """
        # 取一个参考张量用于形状/设备
        ref = next(iter(projected.values()))
        batch_size = ref.shape[0]
        zero = torch.zeros(batch_size, self.fusion_proj.in_features // len(MODALITIES),
                           device=ref.device, dtype=ref.dtype)

        present = list(projected.keys())
        per_modality = []

        for mod in MODALITIES:
            if mod not in projected:
                per_modality.append(zero)
                continue
            # 原始投影
            orig = projected[mod]
            # 该模态从所有其他模态翻译来的信息
            translated_from_others = []
            for other in present:
                if other != mod and other in translations and mod in translations[other]:
                    translated_from_others.append(translations[other][mod])

            if translated_from_others:
                cross_info = torch.stack(translated_from_others).mean(dim=0)
                combined = orig + cross_info
            else:
                combined = orig
            per_modality.append(combined)

        # 拼接所有模态（始终 3 × embed_dim）
        concat = torch.cat(per_modality, dim=-1)
        gate = self.fusion_gate(concat)
        unified = self.fusion_proj(concat) * gate + self.fusion_proj(concat) * (1 - gate) * 0.5

        return unified


# ============ 跨模态预测器 ============

class CrossModalPredictor(nn.Module):
    """
    预测性跨模态补全

    给定存在的模态，预测缺失模态的表征。
    模拟：听到嘶嘶声 → 预测蛇的视觉形象。
    """

    def __init__(self, embed_dim: int = 256, output_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        output_dims = output_dims or {'vision': 768, 'audio': 256, 'language': 128}
        self.predictors = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, dim),
            )
            for mod, dim in output_dims.items()
        })

    def forward(self, unified_repr: torch.Tensor,
                present_modalities: List[str]) -> Dict[str, torch.Tensor]:
        """
        从统一表征预测缺失模态

        Args:
            unified_repr: [B, embed_dim]
            present_modalities: 当前存在的模态列表

        Returns:
            predictions: 缺失模态的预测表征（原始维度）
        """
        predictions = {}
        for mod in MODALITIES:
            if mod not in present_modalities:
                predictions[mod] = self.predictors[mod](unified_repr)
        return predictions


# ============ 时序绑定缓冲区 ============

class TemporalBindingBuffer:
    """
    时序绑定缓冲区

    存储各模态的历史表征，查找时间窗口内的跨模态事件。
    """

    def __init__(self, max_length: int = 50, time_window: float = 0.5):
        self.max_length = max_length
        self.time_window = time_window
        self.buffers: Dict[str, deque] = {mod: deque(maxlen=max_length) for mod in MODALITIES}

    def append(self, modality: str, repr_tensor: torch.Tensor, timestamp: float):
        self.buffers[modality].append({
            'repr': repr_tensor.detach(),
            'timestamp': timestamp,
        })

    def find_temporal_matches(self, source_modality: str, target_modality: str,
                               timestamp: float) -> List[Dict]:
        tolerance = self.time_window
        matches = []
        for event in self.buffers[target_modality]:
            if abs(event['timestamp'] - timestamp) <= tolerance:
                matches.append(event)
        return matches

    def get_recent_context(self) -> Dict[str, Optional[torch.Tensor]]:
        """获取每个模态最近的表征"""
        context = {}
        for mod, buf in self.buffers.items():
            if buf:
                context[mod] = buf[-1]['repr']
            else:
                context[mod] = None
        return context


# ============ 场景检测器 ============

class SceneDetector(nn.Module):
    """
    跨模态场景检测

    基于统一表征判断当前场景类型。
    """

    SCENES = ['neutral', 'danger', 'social', 'food', 'nature',
              'music', 'reading', 'conversation', 'math', 'creative']

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, len(self.SCENES))
        )

    def forward(self, unified: torch.Tensor) -> Dict:
        logits = self.net(unified)
        probs = F.softmax(logits, dim=-1)
        scene_idx = logits.argmax(dim=-1)
        top_prob = probs.max(dim=-1).values
        return {
            'scene_logits': logits,
            'scene_probs': probs,
            'scene_index': scene_idx,
            'scene_name': self.SCENES[scene_idx.item()] if scene_idx.numel() == 1 else [self.SCENES[i] for i in scene_idx.tolist()],
            'confidence': top_prob,
        }


# ============ 主类：角回 ============

class AngularGyrus(nn.Module):
    """
    角回 — 跨模态翻译器

    大脑的"跨模态翻译器"，位于顶叶/颞叶/枕叶交界处。
    将视觉（看到的文字）、听觉（听到的声音）和概念（懂的意思）完美融合。

    功能：
    1. NxN 跨模态翻译 — 任何模态 ↔ 任何模态
    2. 共享语义空间 — 统一表征
    3. 预测性补全 — 缺失模态预测
    4. 时序绑定 — 时间窗口关联
    5. 场景检测 — 跨模态场景分类

    Usage:
        ag = AngularGyrus()
        result = ag({
            'vision': visual_tensor,      # [B, 768]
            'audio': audio_tensor,         # [B, 256]
            'language': language_tensor,   # [B, 128]  或 None
        })
        unified = result['unified_repr']   # [B, 256]
    """

    def __init__(
        self,
        input_dims: Dict[str, int] = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        output_dims: Dict[str, int] = None,
        time_window: float = 0.5,
        event_bus=None,
    ):
        super().__init__()
        input_dims = input_dims or {'vision': 768, 'audio': 256, 'language': 128}
        output_dims = output_dims or {'vision': 768, 'audio': 256, 'language': 128}
        self.embed_dim = embed_dim

        # 子系统
        self.projector = ModalityProjector(input_dims, embed_dim)
        self.translator = TranslationMatrix(embed_dim, num_heads)
        self.interlingua = SemanticInterlingua(embed_dim, len(input_dims))
        self.predictor = CrossModalPredictor(embed_dim, output_dims)
        self.scene_detector = SceneDetector(embed_dim)

        # 时序绑定
        self.temporal_buffer = TemporalBindingBuffer(time_window=time_window)

        # 自上而下调制门控
        self.topdown_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

        # 时间戳
        self.current_time = 0.0

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "sensory_process",
                self._handle_sensory_process,
                priority=2,
                name="angular_gyrus",
            )

    def _handle_sensory_process(self, event) -> Dict:
        """Event-driven handler for sensory_process events."""
        import torch as _torch
        state = event.data.get("internal_state", {})
        state_tensor = event.data.get("state_tensor", _torch.randn(1, 64))

        # Build modalities dict — project real state vector to each modality's expected dim
        modalities = {}

        # Language: project 64-d state → 128-d via learned projection if available
        lang_feat = state.get("language_features")
        if lang_feat is not None and isinstance(lang_feat, _torch.Tensor):
            modalities["language"] = lang_feat
        else:
            # 从真实状态向量派生语言模态 (重复填充到128维)
            modalities["language"] = state_tensor.repeat(1, 2)[:, :128] if state_tensor.shape[-1] == 64 else _torch.randn(1, 128)

        # Vision: project 64-d state → 768-d
        vision_feat = event.data.get("vision_input")
        if vision_feat is not None and isinstance(vision_feat, _torch.Tensor):
            modalities["vision"] = vision_feat
        else:
            modalities["vision"] = state_tensor.repeat(1, 12)[:, :768] if state_tensor.shape[-1] == 64 else _torch.randn(1, 768)

        # Audio: project 64-d state → 256-d
        audio_feat = event.data.get("audio_input")
        if audio_feat is not None and isinstance(audio_feat, _torch.Tensor):
            modalities["audio"] = audio_feat
        else:
            modalities["audio"] = state_tensor.repeat(1, 4)[:, :256] if state_tensor.shape[-1] == 64 else _torch.randn(1, 256)

        # Optional top-down signal from PFC
        top_down_signal = event.data.get("top_down_signal")

        result = self(
            modalities=modalities,
            timestamp=None,
            top_down_signal=top_down_signal,
        )

        state["ag_unified_repr"] = result["unified_repr"]
        state["ag_scene"] = result["scene"].get("scene_name", "neutral")
        state["ag_n_present"] = result.get("n_present", 0)
        state["ag_n_predicted"] = result.get("n_predicted", 0)
        state["ag_present_modalities"] = result.get("present_modalities", [])

        return result

    def forward(
        self,
        modalities: Dict[str, Optional[torch.Tensor]],
        timestamp: Optional[float] = None,
        top_down_signal: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        角回前向传播

        Args:
            modalities: 各模态输入 {'vision': [B,D], 'audio': [B,D], 'language': [B,D]}
                        缺失模态传 None 或不包含该键
            timestamp: 时间戳（用于时序绑定），None 则自动递增
            top_down_signal: 来自 PFC 的自上而下调制信号 [B, embed_dim]

        Returns:
            unified_repr: [B, embed_dim] 统一语义表征
            translations: 所有模态对的翻译结果
            predictions: 缺失模态的预测
            scene: 场景检测结果
        """
        if timestamp is None:
            timestamp = self.current_time
            self.current_time += 0.1

        # Step 1: 过滤有效模态
        present = {mod: tensor for mod, tensor in modalities.items()
                    if tensor is not None}
        if not present:
            # 没有任何输入，返回零向量
            batch_size = 1
            zero = torch.zeros(batch_size, self.embed_dim, device=next(self.parameters()).device)
            return {
                'unified_repr': zero,
                'translations': {},
                'predictions': {},
                'scene': {'scene_name': 'neutral', 'confidence': torch.tensor(0.0)},
                'present_modalities': [],
            }

        # Step 2: 模态投影
        projected = {}
        for mod, tensor in present.items():
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            projected[mod] = self.projector(mod, tensor)

        # Step 3: 时序绑定 — 写入缓冲区
        for mod, repr_t in projected.items():
            self.temporal_buffer.append(mod, repr_t, timestamp)

        # Step 4: NxN 翻译
        translations = self.translator.translate_all(projected)

        # Step 5: 共享语义空间
        unified = self.interlingua(projected, translations)

        # Step 6: 自上而下调制（如果 PFC 提供信号）
        if top_down_signal is not None:
            if top_down_signal.dim() == 1:
                top_down_signal = top_down_signal.unsqueeze(0)
            if top_down_signal.shape[-1] != self.embed_dim:
                top_down_signal = F.adaptive_avg_pool1d(
                    top_down_signal.unsqueeze(1), self.embed_dim
                ).squeeze(1)
            gate = self.topdown_gate(torch.cat([unified, top_down_signal], dim=-1))
            unified = unified * gate + top_down_signal * (1 - gate)

        # Step 7: 预测缺失模态
        predictions = self.predictor(unified, list(present.keys()))

        # Step 8: 场景检测
        scene = self.scene_detector(unified)

        return {
            'unified_repr': unified,
            'translations': translations,
            'predictions': predictions,
            'scene': scene,
            'present_modalities': list(present.keys()),
            'n_present': len(present),
            'n_predicted': len(predictions),
        }

    def get_summary(self) -> Dict:
        """获取系统摘要"""
        ctx = self.temporal_buffer.get_recent_context()
        return {
            'embed_dim': self.embed_dim,
            'modalities': MODALITIES,
            'temporal_buffer_sizes': {mod: len(self.temporal_buffer.buffers[mod]) for mod in MODALITIES},
            'recent_context': {mod: 'available' if v is not None else 'empty' for mod, v in ctx.items()},
        }


__all__ = [
    'AngularGyrus',
    'ModalityProjector',
    'TranslationMatrix',
    'SemanticInterlingua',
    'CrossModalPredictor',
    'TemporalBindingBuffer',
    'SceneDetector',
    'MODALITIES',
]
