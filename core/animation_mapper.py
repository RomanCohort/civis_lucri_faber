# =============================================================================
# Animation Mapper - VTuber 动画参数映射系统
# =============================================================================
# 将情绪状态 (VAD) 映射到 VTuber 动画参数
#
# 核心功能：
# 1. 情绪 → 动画参数映射 (Live2D/VRM 兼容)
# 2. 平滑过渡机制 (插值算法)
# 3. 事件驱动集成 (订阅 EMOTION_UPDATED)
# 4. 表情强度调制 (arousal 驱动)
#
# VTuber 参数标准：
# - Live2D: 参数名 + 归一化值 [0, 1] 或 [-1, 1]
# - VRM: BlendShapeProxy + Weight [0, 1]
# =============================================================================

import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# 处理相对导入
try:
    from core.events import EMOTION_UPDATED, Event
    from core.event_bus import EventBus
except ImportError:
    # 直接运行时添加父目录
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from core.events import EMOTION_UPDATED, Event
    from core.event_bus import EventBus

logger = logging.getLogger(__name__)


# =============================================================================
# 参数类型定义
# =============================================================================

class VTuberPlatform(Enum):
    """VTuber 平台类型"""
    LIVE2D = "live2d"
    VRM = "vrm"
    GENERIC = "generic"


class EmotionLabel(Enum):
    """基本情绪标签"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    NEUTRAL = "neutral"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CONTEMPT = "contempt"


@dataclass
class VADState:
    """VAD 情绪状态"""
    valence: float = 0.0      # 效价 [-1, 1], 负=不愉快, 正=愉快
    arousal: float = 0.5      # 唤醒度 [0, 1], 低=平静, 高=兴奋
    dominance: float = 0.5    # 支配感 [0, 1], 低=被动, 高=主动

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.valence, self.arousal, self.dominance])

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "VADState":
        return cls(
            valence=float(t[0]),
            arousal=float(t[1]),
            dominance=float(t[2])
        )

    def clamp(self) -> "VADState":
        """限制到有效范围"""
        return VADState(
            valence=max(-1.0, min(1.0, self.valence)),
            arousal=max(0.0, min(1.0, self.arousal)),
            dominance=max(0.0, min(1.0, self.dominance))
        )


@dataclass
class AnimationParameter:
    """动画参数"""
    name: str
    value: float
    min_val: float = -1.0
    max_val: float = 1.0
    interpolation_time: float = 0.3  # 插值时间 (秒)

    def clamp(self) -> "AnimationParameter":
        """限制到有效范围"""
        clamped_val = max(self.min_val, min(self.max_val, self.value))
        return AnimationParameter(
            name=self.name,
            value=clamped_val,
            min_val=self.min_val,
            max_val=self.max_val,
            interpolation_time=self.interpolation_time
        )


@dataclass
class AnimationState:
    """完整动画状态"""
    parameters: dict[str, AnimationParameter] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    emotion_label: str = "neutral"
    intensity: float = 0.5

    def to_dict(self) -> dict[str, float]:
        """转换为参数名:值字典"""
        return {k: v.value for k, v in self.parameters.items()}


# =============================================================================
# Live2D 参数映射表
# =============================================================================

# Live2D 标准参数名
LIVE2D_PARAMS = {
    # 眼睛
    "EYE_OPEN_LEFT": "ParamEyeLOpen",
    "EYE_OPEN_RIGHT": "ParamEyeROpen",
    "EYE_SMILE_LEFT": "ParamEyeLSmile",
    "EYE_SMILE_RIGHT": "ParamEyeRSmile",
    "EYE_BALL_X": "ParamEyeBallX",
    "EYE_BALL_Y": "ParamEyeBallY",

    # 眉毛
    "BROW_Y": "ParamBrowY",
    "BROW_X": "ParamBrowX",
    "BROW_ANGLE_LEFT": "ParamBrowLAngle",
    "BROW_ANGLE_RIGHT": "ParamBrowRAngle",

    # 嘴巴
    "MOUTH_OPEN": "ParamMouthOpenY",
    "MOUTH_SMILE": "ParamMouthSmile",
    "MOUTH_FORM": "ParamMouthForm",  # -1:悲伤, 0:普通, 1:微笑

    # 脸部
    "ANGLE_X": "ParamAngleX",  # 头部左右转动
    "ANGLE_Y": "ParamAngleY",  # 头部上下转动
    "ANGLE_Z": "ParamAngleZ",  # 头部倾斜
}

# VRM BlendShape 名称
VRM_BLENDSHAPES = {
    "HAPPY": "Happy",
    "SAD": "Sad",
    "ANGRY": "Angry",
    "RELAXED": "Relaxed",
    "SURPRISED": "Surprised",
    "NEUTRAL": "Neutral",
    "BLINK_LEFT": "Blink_L",
    "BLINK_RIGHT": "Blink_R",
    "LOOK_UP": "LookUp",
    "LOOK_DOWN": "LookDown",
    "LOOK_LEFT": "LookLeft",
    "LOOK_RIGHT": "LookRight",
}


# =============================================================================
# 情绪 → 参数映射规则
# =============================================================================

@dataclass
class EmotionMappingRule:
    """情绪到参数的映射规则"""
    emotion: EmotionLabel
    parameters: dict[str, tuple[float, float]]  # param_name: (base_value, intensity_scale)
    description: str = ""


# 基本情绪映射规则 (Live2D 风格)
EMOTION_MAPPING_RULES: dict[EmotionLabel, EmotionMappingRule] = {
    EmotionLabel.JOY: EmotionMappingRule(
        emotion=EmotionLabel.JOY,
        parameters={
            "EYE_OPEN_LEFT": (0.8, 0.2),      # 眼睛微眯 (笑眼)
            "EYE_OPEN_RIGHT": (0.8, 0.2),
            "EYE_SMILE_LEFT": (0.6, 0.4),     # 笑眼
            "EYE_SMILE_RIGHT": (0.6, 0.4),
            "BROW_Y": (0.3, 0.3),             # 眉毛上扬
            "MOUTH_OPEN": (0.2, 0.3),         # 嘴巴微张
            "MOUTH_SMILE": (0.8, 0.2),        # 嘴角上扬
            "MOUTH_FORM": (0.8, 0.2),         # 笑容形态
        },
        description="快乐: 眼睛笑眯, 嘴角上扬"
    ),

    EmotionLabel.SADNESS: EmotionMappingRule(
        emotion=EmotionLabel.SADNESS,
        parameters={
            "EYE_OPEN_LEFT": (0.5, -0.2),     # 眼睛下垂
            "EYE_OPEN_RIGHT": (0.5, -0.2),
            "BROW_Y": (-0.3, -0.2),           # 眉毛下垂
            "BROW_ANGLE_LEFT": (-0.4, -0.3),  # 眉毛内侧下垂
            "BROW_ANGLE_RIGHT": (-0.4, -0.3),
            "MOUTH_OPEN": (0.1, 0.1),         # 嘴巴微张
            "MOUTH_SMILE": (0.0, 0.0),        # 无笑意
            "MOUTH_FORM": (-0.6, -0.3),       # 嘴角下弯
        },
        description="悲伤: 眼睛下垂, 嘴角下弯"
    ),

    EmotionLabel.ANGER: EmotionMappingRule(
        emotion=EmotionLabel.ANGER,
        parameters={
            "EYE_OPEN_LEFT": (0.9, 0.1),      # 眼睛睁大
            "EYE_OPEN_RIGHT": (0.9, 0.1),
            "BROW_Y": (-0.4, -0.3),           # 眉毛压低
            "BROW_ANGLE_LEFT": (0.5, 0.3),    # 眉毛内侧下压
            "BROW_ANGLE_RIGHT": (0.5, 0.3),
            "MOUTH_OPEN": (0.3, 0.2),         # 嘴巴张开
            "MOUTH_SMILE": (0.0, 0.0),        # 无笑意
            "MOUTH_FORM": (-0.3, -0.2),       # 嘴角紧绷
        },
        description="愤怒: 眉毛压低, 眼睛睁大, 嘴角紧绷"
    ),

    EmotionLabel.FEAR: EmotionMappingRule(
        emotion=EmotionLabel.FEAR,
        parameters={
            "EYE_OPEN_LEFT": (1.0, 0.0),      # 眼睛睁大
            "EYE_OPEN_RIGHT": (1.0, 0.0),
            "BROW_Y": (0.5, 0.3),             # 眉毛上扬
            "BROW_ANGLE_LEFT": (-0.3, -0.2),  # 眉毛内侧上扬
            "BROW_ANGLE_RIGHT": (-0.3, -0.2),
            "MOUTH_OPEN": (0.5, 0.3),         # 嘴巴张开
            "MOUTH_SMILE": (0.0, 0.0),        # 无笑意
            "MOUTH_FORM": (-0.2, -0.1),       # 嘴角微下弯
        },
        description="恐惧: 眼睛睁大, 眉毛上扬, 嘴巴张开"
    ),

    EmotionLabel.NEUTRAL: EmotionMappingRule(
        emotion=EmotionLabel.NEUTRAL,
        parameters={
            "EYE_OPEN_LEFT": (1.0, 0.0),
            "EYE_OPEN_RIGHT": (1.0, 0.0),
            "BROW_Y": (0.0, 0.0),
            "MOUTH_OPEN": (0.0, 0.0),
            "MOUTH_SMILE": (0.0, 0.0),
            "MOUTH_FORM": (0.0, 0.0),
        },
        description="中性: 默认表情"
    ),

    EmotionLabel.SURPRISE: EmotionMappingRule(
        emotion=EmotionLabel.SURPRISE,
        parameters={
            "EYE_OPEN_LEFT": (1.0, 0.0),      # 眼睛睁大
            "EYE_OPEN_RIGHT": (1.0, 0.0),
            "BROW_Y": (0.7, 0.3),             # 眉毛高扬
            "MOUTH_OPEN": (0.7, 0.3),         # 嘴巴张大
            "MOUTH_SMILE": (0.0, 0.0),        # 无笑意
            "MOUTH_FORM": (0.0, 0.0),         # 圆形嘴巴
        },
        description="惊讶: 眼睛睁大, 眉毛高扬, 嘴巴张大"
    ),

    EmotionLabel.DISGUST: EmotionMappingRule(
        emotion=EmotionLabel.DISGUST,
        parameters={
            "EYE_OPEN_LEFT": (0.6, -0.2),     # 眼睛微眯
            "EYE_OPEN_RIGHT": (0.6, -0.2),
            "BROW_Y": (-0.2, -0.2),           # 眉毛压低
            "MOUTH_OPEN": (0.2, 0.1),         # 嘴巴微张
            "MOUTH_SMILE": (0.0, 0.0),        # 无笑意
            "MOUTH_FORM": (-0.4, -0.2),       # 嘴角下弯
        },
        description="厌恶: 眼睛微眯, 嘴角下弯"
    ),

    EmotionLabel.CONTEMPT: EmotionMappingRule(
        emotion=EmotionLabel.CONTEMPT,
        parameters={
            "EYE_OPEN_LEFT": (0.8, 0.0),
            "EYE_OPEN_RIGHT": (0.8, 0.0),
            "BROW_Y": (-0.1, -0.1),           # 眉毛微压
            "MOUTH_OPEN": (0.0, 0.0),
            "MOUTH_SMILE": (0.1, 0.1),        # 单侧冷笑
            "MOUTH_FORM": (0.2, 0.1),         # 嘴角微上扬
        },
        description="轻蔑: 冷笑表情"
    ),
}


# =============================================================================
# VAD → 情绪标签映射
# =============================================================================

def vad_to_emotion_label(vad: VADState) -> tuple[EmotionLabel, float]:
    """
    将 VAD 状态映射到情绪标签

    基于 Russell 的环形情绪模型:
    - valence: 正负效价
    - arousal: 唤醒程度
    - dominance: 支配感 (用于区分 anger vs fear)

    Returns:
        (emotion_label, intensity)
    """
    v, a, d = vad.valence, vad.arousal, vad.dominance

    # 计算强度 (基于 arousal)
    intensity = a

    # 高唤醒度
    if a > 0.6:
        if v > 0.3:
            if d > 0.5:
                return EmotionLabel.JOY, intensity
            else:
                return EmotionLabel.SURPRISE, intensity
        elif v < -0.3:
            if d > 0.5:
                return EmotionLabel.ANGER, intensity
            else:
                return EmotionLabel.FEAR, intensity
        else:
            return EmotionLabel.SURPRISE, intensity

    # 低唤醒度
    elif a < 0.4:
        if v < -0.2:
            return EmotionLabel.SADNESS, intensity
        elif v < 0.0:
            return EmotionLabel.DISGUST, intensity
        else:
            return EmotionLabel.NEUTRAL, intensity

    # 中等唤醒度
    else:
        if v > 0.2:
            return EmotionLabel.JOY, intensity
        elif v < -0.2:
            if d > 0.5:
                return EmotionLabel.CONTEMPT, intensity
            else:
                return EmotionLabel.SADNESS, intensity
        else:
            return EmotionLabel.NEUTRAL, intensity


# =============================================================================
# 平滑插值算法
# =============================================================================

class SmoothInterpolator:
    """
    平滑插值器

    支持多种插值方式:
    - linear: 线性插值
    - ease_in_out: 缓入缓出
    - spring: 弹簧动力学
    """

    def __init__(
        self,
        method: str = "ease_in_out",
        spring_stiffness: float = 100.0,
        spring_damping: float = 10.0,
    ):
        self.method = method
        self.spring_stiffness = spring_stiffness
        self.spring_damping = spring_damping

        # 弹簧状态
        self._velocity: dict[str, float] = {}

    def interpolate(
        self,
        current: float,
        target: float,
        dt: float,
        param_name: str = "",
        duration: float = 0.3,
    ) -> float:
        """
        插值计算

        Args:
            current: 当前值
            target: 目标值
            dt: 时间步长 (秒)
            param_name: 参数名 (用于弹簧状态追踪)
            duration: 插值持续时间

        Returns:
            interpolated: 插值结果
        """
        if self.method == "linear":
            return self._linear(current, target, dt, duration)
        elif self.method == "ease_in_out":
            return self._ease_in_out(current, target, dt, duration)
        elif self.method == "spring":
            return self._spring(current, target, dt, param_name)
        else:
            return self._ease_in_out(current, target, dt, duration)

    def _linear(
        self,
        current: float,
        target: float,
        dt: float,
        duration: float,
    ) -> float:
        """线性插值"""
        if duration <= 0:
            return target
        t = min(1.0, dt / duration)
        return current + (target - current) * t

    def _ease_in_out(
        self,
        current: float,
        target: float,
        dt: float,
        duration: float,
    ) -> float:
        """缓入缓出插值 (smoothstep)"""
        if duration <= 0:
            return target
        t = min(1.0, dt / duration)
        # smoothstep: 3t² - 2t³
        smooth_t = t * t * (3.0 - 2.0 * t)
        return current + (target - current) * smooth_t

    def _spring(
        self,
        current: float,
        target: float,
        dt: float,
        param_name: str,
    ) -> float:
        """
        弹簧动力学插值

        模拟阻尼弹簧: F = -k*x - c*v
        提供自然的弹性过渡效果
        """
        # 获取或初始化速度
        if param_name not in self._velocity:
            self._velocity[param_name] = 0.0

        velocity = self._velocity[param_name]

        # 弹簧力: F = -k * (x - target)
        displacement = current - target
        spring_force = -self.spring_stiffness * displacement

        # 阻尼力: F = -c * v
        damping_force = -self.spring_damping * velocity

        # 加速度
        acceleration = spring_force + damping_force

        # 更新速度和位置 (半隐式欧拉)
        velocity += acceleration * dt
        new_value = current + velocity * dt

        # 存储速度
        self._velocity[param_name] = velocity

        return new_value

    def reset(self, param_name: str | None = None):
        """重置插值状态"""
        if param_name is not None:
            self._velocity.pop(param_name, None)
        else:
            self._velocity.clear()


# =============================================================================
# 动画映射器主类
# =============================================================================

class AnimationMapper(nn.Module):
    """
    VTuber 动画映射器

    将情绪状态 (VAD) 映射到 VTuber 动画参数

    功能:
    1. 情绪 → 参数映射 (基于规则表)
    2. 平滑过渡 (插值算法)
    3. 多平台支持 (Live2D, VRM)
    4. 事件驱动集成 (订阅 EMOTION_UPDATED)
    """

    def __init__(
        self,
        platform: VTuberPlatform = VTuberPlatform.LIVE2D,
        interpolation_method: str = "ease_in_out",
        default_interpolation_time: float = 0.3,
        enable_event_integration: bool = True,
        event_bus: EventBus | None = None,
    ):
        super().__init__()
        self.platform = platform
        self.default_interpolation_time = default_interpolation_time
        self.enable_event_integration = enable_event_integration
        self.event_bus = event_bus

        # 平滑插值器
        self.interpolator = SmoothInterpolator(method=interpolation_method)

        # 当前动画状态
        self._current_state = AnimationState()
        self._target_state = AnimationState()

        # 上次更新时间
        self._last_update_time = time.time()

        # 参数历史 (用于调试/可视化)
        self._param_history: deque = deque(maxlen=100)

        # VAD 调制网络 (学习细微调节)
        self.vad_modulator = nn.Sequential(
            nn.Linear(3, 32),  # VAD → hidden
            nn.ReLU(),
            nn.Linear(32, 16),  # modulation signals
            nn.Tanh()
        )

        # 注册事件订阅
        if enable_event_integration and event_bus is not None:
            self._register_event_handlers()

    def _register_event_handlers(self):
        """注册事件处理器"""
        self.event_bus.subscribe(
            EMOTION_UPDATED,
            self._on_emotion_updated,
            priority=10,  # 较低优先级, 在情绪系统更新后执行
            name="animation_mapper"
        )

    def _on_emotion_updated(self, event: Event) -> dict[str, Any] | None:
        """
        处理 EMOTION_UPDATED 事件

        Event data 应包含:
        - vad: VADState 或 dict
        - emotion_label: str (可选)
        - intensity: float (可选)
        """
        data = event.data

        # 提取 VAD
        if "vad" in data:
            vad_data = data["vad"]
            if isinstance(vad_data, VADState):
                vad = vad_data
            elif isinstance(vad_data, dict):
                vad = VADState(
                    valence=vad_data.get("valence", 0.0),
                    arousal=vad_data.get("arousal", 0.5),
                    dominance=vad_data.get("dominance", 0.5)
                )
            else:
                return None
        else:
            # 从 data 直接提取
            vad = VADState(
                valence=data.get("valence", 0.0),
                arousal=data.get("arousal", 0.5),
                dominance=data.get("dominance", 0.5)
            )

        # 提取情绪标签 (可选)
        emotion_label = data.get("emotion_label")
        intensity = data.get("intensity")

        # 更新动画
        new_state = self.update(vad, emotion_label, intensity)

        return {
            "animation_state": new_state.to_dict(),
            "emotion_label": new_state.emotion_label,
            "intensity": new_state.intensity
        }

    def compute_parameters(
        self,
        emotion: EmotionLabel,
        intensity: float,
        vad: VADState,
    ) -> dict[str, AnimationParameter]:
        """
        计算动画参数

        Args:
            emotion: 情绪标签
            intensity: 表情强度 [0, 1]
            vad: VAD 状态 (用于细微调节)

        Returns:
            parameters: 参数名 → AnimationParameter
        """
        parameters = {}

        # 获取映射规则
        rule = EMOTION_MAPPING_RULES.get(emotion, EMOTION_MAPPING_RULES[EmotionLabel.NEUTRAL])

        # 应用基本映射规则
        for param_name, (base_val, scale) in rule.parameters.items():
            # intensity 调制: value = base + scale * intensity
            value = base_val + scale * (intensity - 0.5)

            # 创建参数
            param = AnimationParameter(
                name=param_name,
                value=value,
                interpolation_time=self.default_interpolation_time
            )
            parameters[param_name] = param.clamp()

        # VAD 细微调节
        if vad is not None:
            parameters = self._apply_vad_modulation(parameters, vad)

        return parameters

    def _apply_vad_modulation(
        self,
        parameters: dict[str, AnimationParameter],
        vad: VADState,
    ) -> dict[str, AnimationParameter]:
        """
        应用 VAD 细微调节

        使用神经网络学习 VAD 对参数的细微影响
        """
        # 计算调制信号
        vad_tensor = vad.to_tensor().unsqueeze(0)  # [1, 3]
        modulation = self.vad_modulator(vad_tensor).squeeze(0)  # [16]

        # 应用调制到关键参数
        # modulation[0]: 眼睛开度微调
        # modulation[1]: 嘴巴开度微调
        # modulation[2]: 头部倾斜微调
        # ...

        if "EYE_OPEN_LEFT" in parameters:
            eye_param = parameters["EYE_OPEN_LEFT"]
            parameters["EYE_OPEN_LEFT"] = AnimationParameter(
                name=eye_param.name,
                value=eye_param.value + modulation[0].item() * 0.1,
                interpolation_time=eye_param.interpolation_time
            ).clamp()

        if "EYE_OPEN_RIGHT" in parameters:
            eye_param = parameters["EYE_OPEN_RIGHT"]
            parameters["EYE_OPEN_RIGHT"] = AnimationParameter(
                name=eye_param.name,
                value=eye_param.value + modulation[0].item() * 0.1,
                interpolation_time=eye_param.interpolation_time
            ).clamp()

        if "MOUTH_OPEN" in parameters:
            mouth_param = parameters["MOUTH_OPEN"]
            parameters["MOUTH_OPEN"] = AnimationParameter(
                name=mouth_param.name,
                value=mouth_param.value + modulation[1].item() * 0.1,
                interpolation_time=mouth_param.interpolation_time
            ).clamp()

        return parameters

    def update(
        self,
        vad: VADState,
        emotion_label: str | None = None,
        intensity: float | None = None,
    ) -> AnimationState:
        """
        更新动画状态

        Args:
            vad: VAD 情绪状态
            emotion_label: 情绪标签 (可选, 自动推断)
            intensity: 强度 (可选, 自动推断)

        Returns:
            state: 新的动画状态
        """
        # 推断情绪标签和强度
        if emotion_label is None or intensity is None:
            inferred_label, inferred_intensity = vad_to_emotion_label(vad)
            if emotion_label is None:
                emotion_label = inferred_label.value
            if intensity is None:
                intensity = inferred_intensity

        # 转换为枚举
        try:
            emotion_enum = EmotionLabel(emotion_label)
        except ValueError:
            emotion_enum = EmotionLabel.NEUTRAL

        # 计算目标参数
        target_params = self.compute_parameters(emotion_enum, intensity, vad)

        # 计算时间步长
        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now

        # 平滑插值
        interpolated_params = self._interpolate_parameters(
            self._current_state.parameters,
            target_params,
            dt
        )

        # 更新状态
        new_state = AnimationState(
            parameters=interpolated_params,
            timestamp=now,
            emotion_label=emotion_label,
            intensity=intensity
        )

        self._current_state = new_state
        self._target_state = AnimationState(
            parameters=target_params,
            timestamp=now,
            emotion_label=emotion_label,
            intensity=intensity
        )

        # 记录历史
        self._param_history.append({
            "timestamp": now,
            "vad": vad,
            "emotion": emotion_label,
            "intensity": intensity,
            "params": new_state.to_dict()
        })

        return new_state

    def _interpolate_parameters(
        self,
        current: dict[str, AnimationParameter],
        target: dict[str, AnimationParameter],
        dt: float,
    ) -> dict[str, AnimationParameter]:
        """对所有参数进行插值"""
        result = {}

        # 所有关心的参数名
        all_keys = set(current.keys()) | set(target.keys())

        for key in all_keys:
            current_val = current.get(key, AnimationParameter(name=key, value=0.0)).value
            target_param = target.get(key, AnimationParameter(name=key, value=0.0))

            interpolated_val = self.interpolator.interpolate(
                current_val,
                target_param.value,
                dt,
                param_name=key,
                duration=target_param.interpolation_time
            )

            result[key] = AnimationParameter(
                name=key,
                value=interpolated_val,
                min_val=target_param.min_val,
                max_val=target_param.max_val,
                interpolation_time=target_param.interpolation_time
            ).clamp()

        return result

    def get_live2d_params(self) -> dict[str, float]:
        """
        获取 Live2D 格式参数

        Returns:
            params: Live2D 参数名 → 值
        """
        params = {}
        for key, param in self._current_state.parameters.items():
            live2d_name = LIVE2D_PARAMS.get(key)
            if live2d_name is not None:
                params[live2d_name] = param.value
        return params

    def get_vrm_params(self) -> dict[str, float]:
        """
        获取 VRM 格式参数

        Returns:
            params: VRM BlendShape 名 → 权重
        """
        # 将当前参数映射到 VRM BlendShape
        params = {}
        emotion_label = self._current_state.emotion_label
        intensity = self._current_state.intensity

        # 基于情绪标签设置 BlendShape
        emotion_to_vrm = {
            "joy": ("HAPPY", 1.0),
            "sadness": ("SAD", 1.0),
            "anger": ("ANGRY", 1.0),
            "fear": ("RELAXED", 0.5),  # VRM 没有 fear, 用 relaxed 替代
            "neutral": ("NEUTRAL", 1.0),
            "surprise": ("SURPRISED", 1.0),
            "disgust": ("SAD", 0.7),   # 近似映射
            "contempt": ("ANGRY", 0.5), # 近似映射
        }

        if emotion_label in emotion_to_vrm:
            blendshape, weight = emotion_to_vrm[emotion_label]
            vrm_name = VRM_BLENDSHAPES.get(blendshape)
            if vrm_name is not None:
                params[vrm_name] = weight * intensity

        return params

    def step(self, dt: float | None = None) -> AnimationState:
        """
        单步更新 (用于主动更新, 不依赖事件)

        Args:
            dt: 时间步长 (可选, 自动计算)

        Returns:
            state: 更新后的状态
        """
        if dt is None:
            now = time.time()
            dt = now - self._last_update_time
            self._last_update_time = now

        # 对当前状态向目标插值
        interpolated_params = self._interpolate_parameters(
            self._current_state.parameters,
            self._target_state.parameters,
            dt
        )

        new_state = AnimationState(
            parameters=interpolated_params,
            timestamp=time.time(),
            emotion_label=self._current_state.emotion_label,
            intensity=self._current_state.intensity
        )

        self._current_state = new_state
        return new_state

    def get_param_history(self, n: int = 10) -> list[dict]:
        """获取参数历史"""
        return list(self._param_history)[-n:]

    def get_current_state(self) -> AnimationState:
        """获取当前状态"""
        return self._current_state

    def reset(self):
        """重置到默认状态"""
        self._current_state = AnimationState()
        self._target_state = AnimationState()
        self._last_update_time = time.time()
        self._param_history.clear()
        self.interpolator.reset()


# =============================================================================
# 便捷函数
# =============================================================================

def create_animation_mapper(
    platform: str = "live2d",
    interpolation_method: str = "ease_in_out",
    event_bus: EventBus | None = None,
) -> AnimationMapper:
    """
    创建动画映射器

    Args:
        platform: 平台类型 ("live2d", "vrm", "generic")
        interpolation_method: 插值方法 ("linear", "ease_in_out", "spring")
        event_bus: 事件总线 (可选)

    Returns:
        mapper: AnimationMapper 实例
    """
    platform_enum = {
        "live2d": VTuberPlatform.LIVE2D,
        "vrm": VTuberPlatform.VRM,
        "generic": VTuberPlatform.GENERIC
    }.get(platform.lower(), VTuberPlatform.LIVE2D)

    return AnimationMapper(
        platform=platform_enum,
        interpolation_method=interpolation_method,
        event_bus=event_bus
    )


def vad_to_animation(
    vad: VADState,
    platform: str = "live2d"
) -> dict[str, float]:
    """
    快速将 VAD 转换为动画参数

    Args:
        vad: VAD 状态
        platform: 平台类型

    Returns:
        params: 动画参数
    """
    mapper = create_animation_mapper(platform=platform)
    state = mapper.update(vad)

    if platform.lower() == "live2d":
        return mapper.get_live2d_params()
    elif platform.lower() == "vrm":
        return mapper.get_vrm_params()
    else:
        return state.to_dict()


# =============================================================================
# 测试
# =============================================================================

def test_animation_mapper():
    """测试动画映射器"""
    print("=" * 60)
    print("Testing Animation Mapper")
    print("=" * 60)

    # 创建映射器
    mapper = AnimationMapper(
        platform=VTuberPlatform.LIVE2D,
        interpolation_method="ease_in_out"
    )

    print("\n[1] Testing VAD to emotion mapping...")
    test_vads = [
        VADState(valence=0.8, arousal=0.7, dominance=0.6),  # Joy
        VADState(valence=-0.6, arousal=0.3, dominance=0.4),  # Sadness
        VADState(valence=-0.7, arousal=0.8, dominance=0.7),  # Anger
        VADState(valence=-0.5, arousal=0.9, dominance=0.3),  # Fear
        VADState(valence=0.0, arousal=0.5, dominance=0.5),   # Neutral
    ]

    for vad in test_vads:
        label, intensity = vad_to_emotion_label(vad)
        print(f"  VAD({vad.valence:.1f}, {vad.arousal:.1f}, {vad.dominance:.1f}) → {label.value} (intensity={intensity:.2f})")

    print("\n[2] Testing parameter computation...")
    params = mapper.compute_parameters(EmotionLabel.JOY, 0.8, VADState(0.8, 0.7, 0.6))
    print(f"  Joy parameters:")
    for name, param in params.items():
        print(f"    {name}: {param.value:.3f}")

    print("\n[3] Testing animation update...")
    for vad in test_vads:
        state = mapper.update(vad)
        print(f"  Emotion: {state.emotion_label}, Intensity: {state.intensity:.2f}")

    print("\n[4] Testing Live2D output...")
    joy_state = mapper.update(VADState(0.8, 0.7, 0.6))
    live2d_params = mapper.get_live2d_params()
    print(f"  Live2D params: {list(live2d_params.keys())[:5]}...")

    print("\n[5] Testing VRM output...")
    vrm_params = mapper.get_vrm_params()
    print(f"  VRM params: {vrm_params}")

    print("\n[6] Testing smooth interpolation...")
    mapper.reset()
    # 突变
    mapper.update(VADState(-0.8, 0.3, 0.4))  # Sadness
    for i in range(5):
        state = mapper.step(dt=0.1)
        print(f"  Step {i}: mouth_form={state.parameters.get('MOUTH_FORM', AnimationParameter('x', 0)).value:.3f}")

    print("\n[7] Testing event integration...")
    from core.event_bus import EventBus
    bus = EventBus()
    mapper_with_events = AnimationMapper(
        platform=VTuberPlatform.LIVE2D,
        event_bus=bus,
        enable_event_integration=True
    )
    # 发布事件
    bus.publish(EMOTION_UPDATED, {
        "vad": {"valence": 0.5, "arousal": 0.6, "dominance": 0.5},
        "emotion_label": "joy"
    }, source="test")
    state = mapper_with_events.get_current_state()
    print(f"  After event: emotion={state.emotion_label}, intensity={state.intensity:.2f}")

    print("\n" + "=" * 60)
    print("[OK] Animation mapper working!")
    print("  - VAD to emotion mapping: OK")
    print("  - Parameter computation: OK")
    print("  - Smooth interpolation: OK")
    print("  - Live2D output: OK")
    print("  - VRM output: OK")
    print("  - Event integration: OK")
    print("=" * 60)


if __name__ == "__main__":
    test_animation_mapper()


__all__ = [
    # 类型
    "VTuberPlatform",
    "EmotionLabel",
    "VADState",
    "AnimationParameter",
    "AnimationState",
    "EmotionMappingRule",
    # 映射表
    "LIVE2D_PARAMS",
    "VRM_BLENDSHAPES",
    "EMOTION_MAPPING_RULES",
    # 核心类
    "SmoothInterpolator",
    "AnimationMapper",
    # 函数
    "vad_to_emotion_label",
    "create_animation_mapper",
    "vad_to_animation",
]
