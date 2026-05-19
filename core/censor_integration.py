"""Censor 微表情感知集成模块

将 D:\\censor 仿生双通路微表情识别系统实时集成进 Simulacrum 认知架构。

生物对应:
    - 快通路 (3D ResNet-18) → 皮下通路 (杏仁核快速威胁检测)
    - 慢通路 (3D Swin-Transformer) → 皮层通路 (FFA 面孔精细识别)
    - AU 强度 → 面部动作编码系统 (FACS) → 情绪推断
    - ME logits → 微表情分类 → 隐藏情绪揭示
    - Apex 检测 → 情绪峰值定位 → 情绪时序建模

事件驱动:
    - 订阅 MICRO_EXPRESSION_PROCESS: 接收视频帧，运行 Censor 推理
    - 发布 MICRO_EXPRESSION_DETECTED: 输出 AU/ME/情绪结果

集成点:
    1. AU intensities → Simulacrum 情绪系统 (AdvancedEmotionModule)
    2. ME logits → Simulacrum 边缘系统 (LimbicSystem 杏仁核威胁评估)
    3. Apex scores → Simulacrum 海马体 (Hippocampus 情绪事件标记)
    4. Emotion reports → Simulacrum 语言皮层 (LanguageCortex 情绪词汇)
    5. Expert gates → Simulacrum 基底神经节 (BasalGanglia 决策路由)
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from simulacrum.core.events import (
    MICRO_EXPRESSION_PROCESS,
    MICRO_EXPRESSION_DETECTED,
    SENSORY_PROCESS,
)


# ===== 微表情结果数据结构 =====

@dataclass
class MicroExpressionResult:
    """Censor 单次推理结果

    封装所有 Censor 输出，供 Simulacrum 各脑区消费。
    """
    # 微表情分类 logits (7类或11类)
    me_logits: np.ndarray          # (num_me_classes,)
    me_predicted: int              # 预测类别索引
    me_confidence: float           # 预测置信度

    # AU 强度 (28个 FACS Action Units)
    au_intensities: np.ndarray     # (28,) 帧平均强度
    au_active: List[int]           # 激活的 AU 索引列表
    au_dominant: int               # 最强 AU 索引
    au_dominant_intensity: float   # 最强 AU 强度值

    # AU 时序 (Onset-Peak-Decay)
    au_opd: np.ndarray             # (28, 3) onset/peak/decay 坐标

    # Apex 帧检测
    apex_scores: np.ndarray        # (T_apa,) 各帧 apex 分数
    apex_frame: int                # 最可能的 apex 帧索引

    # MoE 专家门控
    expert_gates: np.ndarray       # (3,) 各专家权重
    dominant_expert: int           # 主导专家索引

    # 个性化特征
    adapted_feat: np.ndarray       # (1024,) 个性化后特征

    # 情绪报告
    template_report: str           # 结构化临床报告
    llm_report: str                # 自由文本报告 (placeholder)

    # 原始 Censor 输出 (完整字典)
    raw_output: Dict[str, Any] = field(default_factory=dict)

    # 元信息
    inference_time_ms: float = 0.0
    frame_count: int = 16


# ===== Censor 包装器 =====

class CensorPerceptionModule:
    """Censor 微表情感知模块 (事件驱动)

    包装 Censor 模型，提供:
    1. 惰性初始化 (首次推理时才加载模型)
    2. 事件驱动接口 (订阅/发布)
    3. 降级回退 (Censor 不可用时返回默认值)
    4. AU → 情绪映射 (FACS → 情绪维度)
    5. 状态缓存 (避免重复推理)

    用法:
        censor = CensorPerceptionModule(event_bus=bus)
        # 事件驱动: 自动响应 MICRO_EXPRESSION_PROCESS
        # 手动调用:
        result = censor.process_video(video_tensor)
    """

    # FACS AU → 基础情绪映射 (Ekman 6 + Contempt)
    AU_EMOTION_MAP = {
        'happiness':   [6, 12],       # AU6 (Cheek Raiser) + AU12 (Lip Corner Puller)
        'sadness':     [1, 4, 15],    # AU1 + AU4 + AU15
        'anger':       [4, 5, 7, 23], # AU4 + AU5 + AU7 + AU23
        'fear':        [1, 2, 4, 5, 20, 26],  # AU1+2+4+5+20+26
        'disgust':     [9, 10, 17],   # AU9 + AU10 + AU17
        'surprise':    [1, 2, 5, 26, 27],  # AU1+2+5+26+27
        'contempt':    [12, 14],      # AU12 (单侧) + AU14
    }

    # ME 类别名称 (7类标准)
    ME_CATEGORIES_7 = [
        "happiness", "sadness", "surprise", "fear",
        "disgust", "anger", "contempt",
    ]

    # ME 类别名称 (11类扩展)
    ME_CATEGORIES_11 = [
        "Happiness (Duchenne)", "Happiness (Non-Duchenne)",
        "Surprise (Strong)", "Surprise (Weak)",
        "Fear", "Disgust (Strong)", "Disgust (Weak)",
        "Anger (Strong)", "Anger (Weak)",
        "Sadness", "Contempt",
    ]

    def __init__(
        self,
        event_bus=None,
        censor_path: str = r"D:\censor",
        device: str = "auto",
        au_threshold: float = 0.3,
        enable_lazy_init: bool = True,
    ):
        self._bus = event_bus
        self._censor_path = censor_path
        self._au_threshold = au_threshold
        self._enable_lazy_init = enable_lazy_init

        # 模型引用 (惰性加载)
        self._model: Optional[nn.Module] = None
        self._initialized = False
        self._init_failed = False

        # 设备
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # 缓存最近一次结果
        self._last_result: Optional[MicroExpressionResult] = None
        self._last_frame_hash: Optional[int] = None

        # 事件订阅
        if self._bus is not None:
            self._bus.subscribe(
                MICRO_EXPRESSION_PROCESS,
                self._on_micro_expression_process,
                priority=5,
                name="censor_perception",
            )
            # 也订阅 SENSORY_PROCESS，在感知阶段自动触发
            self._bus.subscribe(
                SENSORY_PROCESS,
                self._on_sensory_process,
                priority=10,  # 低优先级，不阻塞主感知流
                name="censor_sensory_bridge",
            )

        print(f"[Censor] Perception Module initialized (path={censor_path}, "
              f"device={self._device}, lazy={enable_lazy_init})")

    # ===== 惰性初始化 =====

    def _ensure_initialized(self) -> bool:
        """确保 Censor 模型已加载"""
        if self._initialized:
            return not self._init_failed
        if self._init_failed:
            return False

        try:
            self._initialize_censor()
            self._initialized = True
            return True
        except Exception as e:
            print(f"[Censor] Initialization failed: {e}")
            self._init_failed = True
            return False

    def _initialize_censor(self):
        """加载 Censor 模型"""
        censor_dir = Path(self._censor_path)
        if not censor_dir.exists():
            raise FileNotFoundError(f"Censor directory not found: {self._censor_path}")

        # 将 Censor 路径加入 sys.path
        censor_str = str(censor_dir)
        if censor_str not in sys.path:
            sys.path.insert(0, censor_str)

        # 导入 Censor 主模型
        from main import Censor

        self._model = Censor()
        self._model.to(self._device)
        self._model.eval()

        # 统计参数量
        total_params = sum(p.numel() for p in self._model.parameters())
        print(f"[Censor] Model loaded successfully ({total_params:,} parameters, device={self._device})")

    # ===== 事件处理器 =====

    def _on_micro_expression_process(self, event) -> Optional[Dict[str, Any]]:
        """响应 MICRO_EXPRESSION_PROCESS 事件

        事件数据:
            video: torch.Tensor (B, 3, T, H, W) 或 None
            video_path: str 视频文件路径 (备选)
            force: bool 是否强制重新推理
        """
        data = event.data if hasattr(event, 'data') else event
        video = data.get("video")
        force = data.get("force", False)

        if video is None:
            return None

        result = self.process_video(video, force=force)
        if result is None:
            return None

        # 发布检测结果事件
        if self._bus is not None:
            self._bus.publish(
                MICRO_EXPRESSION_DETECTED,
                {
                    "me_predicted": result.me_predicted,
                    "me_confidence": result.me_confidence,
                    "me_category": self._get_me_category(result.me_predicted),
                    "au_active": result.au_active,
                    "au_dominant": result.au_dominant,
                    "au_dominant_intensity": result.au_dominant_intensity,
                    "apex_frame": result.apex_frame,
                    "expert_gates": result.expert_gates.tolist(),
                    "dominant_expert": result.dominant_expert,
                    "template_report": result.template_report,
                    "emotion_map": self._compute_emotion_map(result.au_intensities),
                    "adapted_feat_mean": float(result.adapted_feat.mean()),
                },
                source="censor_perception",
            )

        return {
            "censor_perception": {
                "me_predicted": result.me_predicted,
                "me_confidence": result.me_confidence,
                "au_active": result.au_active,
                "au_dominant": result.au_dominant,
                "apex_frame": result.apex_frame,
                "emotion_map": self._compute_emotion_map(result.au_intensities),
                "result": result,
            }
        }

    def _on_sensory_process(self, event) -> Optional[Dict[str, Any]]:
        """响应 SENSORY_PROCESS 事件 — 自动桥接

        当 Simulacrum 感知阶段有视频输入时，自动触发 Censor 推理。
        不阻塞主感知流 (低优先级)。
        """
        data = event.data if hasattr(event, 'data') else event
        video = data.get("video")
        if video is None:
            return None

        # 异步触发微表情处理 (不等待结果)
        result = self.process_video(video)
        if result is not None and self._bus is not None:
            self._bus.publish(
                MICRO_EXPRESSION_DETECTED,
                {
                    "me_predicted": result.me_predicted,
                    "me_confidence": result.me_confidence,
                    "me_category": self._get_me_category(result.me_predicted),
                    "au_active": result.au_active,
                    "emotion_map": self._compute_emotion_map(result.au_intensities),
                    "source": "sensory_bridge",
                },
                source="censor_perception",
            )
        return None  # 不返回结果，避免阻塞主感知流

    # ===== 核心推理 =====

    @torch.no_grad()
    def process_video(
        self,
        video: torch.Tensor,
        force: bool = False,
    ) -> Optional[MicroExpressionResult]:
        """处理视频帧，返回微表情结果

        Args:
            video: (B, 3, T, H, W) RGB 视频张量，或 (3, T, H, W) 单样本
            force: 是否强制重新推理 (忽略缓存)

        Returns:
            MicroExpressionResult 或 None (模型不可用时)
        """
        import time
        t0 = time.time()

        # 确保模型已加载
        if not self._ensure_initialized():
            return self._fallback_result(video)

        # 维度标准化
        if video.dim() == 4:
            video = video.unsqueeze(0)  # (3,T,H,W) → (1,3,T,H,W)
        B, C, T, H, W = video.shape

        # 缓存检查
        frame_hash = hash(video.data_ptr()) if video.is_contiguous() else hash(tuple(video.shape))
        if not force and self._last_result is not None and frame_hash == self._last_frame_hash:
            return self._last_result

        # 推理
        try:
            video = video.to(self._device)
            # Censor forward 期望 (B, 3, T=16, H=224, W=224)
            # 如果尺寸不匹配，进行插值
            if H != 224 or W != 224:
                video = nn.functional.interpolate(
                    video.view(B * C, T, H, W),
                    size=(T, 224, 224),
                    mode='trilinear',
                    align_corners=False,
                ).view(B, C, T, 224, 224)
            if T != 16:
                # 时间维度插值
                video = nn.functional.interpolate(
                    video.permute(0, 1, 3, 4, 2),  # (B,C,H,W,T)
                    size=(224, 224, 16),
                    mode='trilinear',
                    align_corners=False,
                ).permute(0, 1, 4, 2, 3)  # (B,C,T,H,W)

            outputs = self._model(video)
        except Exception as e:
            print(f"[Censor] Forward pass failed: {e}")
            return self._fallback_result(video)

        inference_ms = (time.time() - t0) * 1000

        # 解析输出
        result = self._parse_censor_output(outputs, B, T, inference_ms)
        self._last_result = result
        self._last_frame_hash = frame_hash

        return result

    def _parse_censor_output(
        self,
        outputs: Dict[str, Any],
        batch_size: int,
        temporal_frames: int,
        inference_ms: float,
    ) -> MicroExpressionResult:
        """解析 Censor 原始输出为 MicroExpressionResult

        取 batch 中第一个样本的结果。
        """
        # ME logits → 预测
        me_logits = outputs['me_logits'][0].cpu().numpy()  # (num_classes,)
        me_predicted = int(np.argmax(me_logits))
        me_confidence = float(np.softmax(me_logits).max())

        # AU intensities → 帧平均
        au_raw = outputs['au_intensities'][0].cpu().numpy()  # (T, 28)
        au_intensities = au_raw.mean(axis=0)  # (28,) 帧平均
        au_active = [i for i in range(28) if au_intensities[i] > self._au_threshold]
        au_dominant = int(np.argmax(au_intensities))
        au_dominant_intensity = float(au_intensities[au_dominant])

        # AU OPD
        au_opd = outputs['au_opd'][0].cpu().numpy()  # (28, 3)

        # Apex scores
        apex_scores = outputs['apex_scores'][0].cpu().numpy()
        apex_frame = int(np.argmax(apex_scores))

        # Expert gates
        expert_gates = outputs['expert_gates'][0].cpu().numpy()  # (3,)
        dominant_expert = int(np.argmax(expert_gates))

        # Adapted features
        adapted_feat = outputs['adapted_feat'][0].cpu().numpy()  # (1024,)

        # Reports
        template_reports = outputs.get('template_report', [''])
        template_report = template_reports[0] if template_reports else ''
        llm_reports = outputs.get('llm_report', [''])
        llm_report = llm_reports[0] if llm_reports else ''

        return MicroExpressionResult(
            me_logits=me_logits,
            me_predicted=me_predicted,
            me_confidence=me_confidence,
            au_intensities=au_intensities,
            au_active=au_active,
            au_dominant=au_dominant,
            au_dominant_intensity=au_dominant_intensity,
            au_opd=au_opd,
            apex_scores=apex_scores,
            apex_frame=apex_frame,
            expert_gates=expert_gates,
            dominant_expert=dominant_expert,
            adapted_feat=adapted_feat,
            template_report=template_report,
            llm_report=llm_report,
            raw_output={k: v.shape if isinstance(v, torch.Tensor) else v
                        for k, v in outputs.items()},
            inference_time_ms=inference_ms,
            frame_count=temporal_frames,
        )

    # ===== AU → 情绪映射 =====

    def _compute_emotion_map(self, au_intensities: np.ndarray) -> Dict[str, float]:
        """从 AU 强度计算基础情绪激活度

        每个情绪的激活度 = 其关联 AU 强度的加权平均。
        权重: AU 强度 × 1/N (N=该情绪关联的 AU 数量)

        Returns:
            {"happiness": 0.72, "sadness": 0.15, ...}
        """
        emotion_map = {}
        for emotion, au_indices in self.AU_EMOTION_MAP.items():
            # AU 索引从1开始，数组从0开始
            values = [au_intensities[au - 1] for au in au_indices if au - 1 < len(au_intensities)]
            if values:
                emotion_map[emotion] = float(np.mean(values))
            else:
                emotion_map[emotion] = 0.0
        return emotion_map

    def _get_me_category(self, me_idx: int) -> str:
        """获取 ME 类别名称"""
        if me_idx < len(self.ME_CATEGORIES_7):
            return self.ME_CATEGORIES_7[me_idx]
        elif me_idx < len(self.ME_CATEGORIES_11):
            return self.ME_CATEGORIES_11[me_idx]
        return f"unknown_{me_idx}"

    # ===== 降级回退 =====

    def _fallback_result(self, video: torch.Tensor = None) -> MicroExpressionResult:
        """Censor 不可用时的降级结果

        返回全零/中性结果，不阻塞 Simulacrum 主流程。
        """
        num_classes = 7
        num_aus = 28

        return MicroExpressionResult(
            me_logits=np.zeros(num_classes),
            me_predicted=0,
            me_confidence=0.0,
            au_intensities=np.zeros(num_aus),
            au_active=[],
            au_dominant=0,
            au_dominant_intensity=0.0,
            au_opd=np.zeros((num_aus, 3)),
            apex_scores=np.zeros(1),
            apex_frame=0,
            expert_gates=np.ones(3) / 3.0,
            dominant_expert=0,
            adapted_feat=np.zeros(1024),
            template_report="[Censor unavailable] No micro-expression analysis.",
            llm_report="",
            raw_output={},
            inference_time_ms=0.0,
            frame_count=0,
        )

    # ===== 状态向量注入 =====

    def get_state_vector(self) -> np.ndarray:
        """提取 16 维 Censor 状态向量，注入 Simulacrum _build_state_vector()

        维度分配:
            [0]  me_confidence     微表情置信度
            [1]  me_predicted_norm 预测类别归一化
            [2]  au_active_ratio   AU激活比例
            [3]  au_dominant_int   最强AU强度
            [4]  au_mean_intensity AU平均强度
            [5]  apex_score_max    Apex帧最高分
            [6]  expert_gate_max   主导专家权重
            [7]  expert_gate_entropy 专家门控熵
            [8]  emotion_happiness 快乐激活度
            [9]  emotion_sadness   悲伤激活度
            [10] emotion_anger     愤怒激活度
            [11] emotion_fear      恐惧激活度
            [12] emotion_disgust   厌恶激活度
            [13] emotion_surprise  惊讶激活度
            [14] emotion_contempt  蔑视激活度
            [15] adapted_feat_norm 个性化特征范数
        """
        if self._last_result is None:
            return np.zeros(16, dtype=np.float32)

        r = self._last_result
        emotion_map = self._compute_emotion_map(r.au_intensities)

        # 专家门控熵
        gates = r.expert_gates + 1e-10
        gate_entropy = -float(np.sum(gates * np.log(gates)))

        return np.array([
            np.clip(r.me_confidence, 0, 1),
            np.clip(r.me_predicted / max(len(r.me_logits) - 1, 1), 0, 1),
            np.clip(len(r.au_active) / 28.0, 0, 1),
            np.clip(r.au_dominant_intensity, 0, 1),
            np.clip(float(r.au_intensities.mean()), 0, 1),
            np.clip(float(r.apex_scores.max()), 0, 1),
            np.clip(float(r.expert_gates.max()), 0, 1),
            np.clip(gate_entropy / np.log(3), 0, 1),  # 归一化到 [0,1]
            np.clip(emotion_map.get('happiness', 0), 0, 1),
            np.clip(emotion_map.get('sadness', 0), 0, 1),
            np.clip(emotion_map.get('anger', 0), 0, 1),
            np.clip(emotion_map.get('fear', 0), 0, 1),
            np.clip(emotion_map.get('disgust', 0), 0, 1),
            np.clip(emotion_map.get('surprise', 0), 0, 1),
            np.clip(emotion_map.get('contempt', 0), 0, 1),
            np.clip(float(np.linalg.norm(r.adapted_feat)) / 100.0, 0, 1),  # 范数归一化
        ], dtype=np.float32)

    # ===== 辅助方法 =====

    def get_last_result(self) -> Optional[MicroExpressionResult]:
        """获取最近一次推理结果"""
        return self._last_result

    def is_available(self) -> bool:
        """Censor 模型是否可用"""
        return self._initialized and not self._init_failed

    def get_summary(self) -> str:
        """获取最近一次推理的文本摘要"""
        if self._last_result is None:
            return "[Censor] No analysis yet."

        r = self._last_result
        me_cat = self._get_me_category(r.me_predicted)
        emotion_map = self._compute_emotion_map(r.au_intensities)
        dominant_emotion = max(emotion_map, key=emotion_map.get)

        return (
            f"[Censor] ME={me_cat} (conf={r.me_confidence:.2f}), "
            f"AU active={len(r.au_active)}/28 (dominant=AU{r.au_dominant+1}:{r.au_dominant_intensity:.2f}), "
            f"Emotion={dominant_emotion}, "
            f"Apex@frame{r.apex_frame}, "
            f"Expert={r.dominant_expert}, "
            f"Inference={r.inference_time_ms:.1f}ms"
        )
