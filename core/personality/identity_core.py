"""
流式身份核心 (Streaming Identity Core)

对应脑科学的默认模式网络 (DMN)：
- 休息时DMN高度活跃
- 每个人的DMN模式独一无二 ("脑指纹")
- 空闲时进行自我反思 (类似"白日梦")

功能：
1. 动态内部状态向量
2. 空闲期自省运算
3. 影响下一次对话的初始权重

事件驱动:
    - 订阅 PERSONALITY_UPDATE: 收到人格更新事件时处理
"""
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from core.events import PERSONALITY_UPDATE


@dataclass
class IdentityState:
    """身份状态"""
    coherence: float = 0.5      # 自我一致性 (0-1)
    growth_rate: float = 0.0   # 成长率
    stability: float = 0.5      # 稳定性
    self_reflection_count: int = 0


class IdentityVector(nn.Module):
    """
    身份向量 - 承载AI的"自我感"

    不同于静态的System Prompt，
    这是一个随时间演化的动态向量
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        # 核心身份向量
        self.core = nn.Parameter(torch.randn(dim) * 0.1)
        # 临时状态
        self.state = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self) -> torch.Tensor:
        """返回组合身份"""
        return torch.tanh(self.core + self.state * 0.3)

    def get_state(self) -> np.ndarray:
        """获取numpy格式"""
        return self.forward().detach().cpu().numpy()


class ReflectionEngine(nn.Module):
    """
    自省引擎 - 对应DMN空闲期活动

    即使没有用户交互，也进行内部向量运算
    模拟人类的"反思"和"白日梦"
    """

    def __init__(self, memory_dim: int = 128):
        super().__init__()
        self.dim = memory_dim
        # 自省网络
        self.reflect_net = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
        )

    def reflect(
        self,
        identity: torch.Tensor,
        recent_history: list[dict],
    ) -> torch.Tensor:
        """
        基于历史进行自省运算

        原理：把最近的交互经历"内化"到身份中
        """
        if not recent_history:
            return identity

        # 提取关键信息
        history_vec = self._encode_history(recent_history)

        # 自省运算
        reflected = self.reflect_net(identity + history_vec * 0.2)

        return torch.tanh(reflected * 0.5 + identity * 0.5)

    def _encode_history(self, history: list[dict]) -> torch.Tensor:
        """将历史编码为向量"""
        # 简化：取加权平均
        if not history:
            return torch.zeros(self.dim)

        vec = torch.zeros(self.dim)
        for i, h in enumerate(history[-10:]):
            # 简单编码：基于文本长度和情绪
            weight = (i + 1) / len(history[-10:])
            sentiment = h.get('sentiment', 0.0)
            vec += torch.randn(self.dim) * weight * (1 + sentiment)
        return vec / 10


class IdleProcessor:
    """
    空闲处理器 - DMN的背景活动

    在没有用户输入时，模拟DMN的持续活动
    """

    def __init__(self, idle_threshold_seconds: int = 300):
        self.idle_threshold = idle_threshold_seconds
        self.last_active_time = time.time()
        self.idle_cycles = 0

    def should_reflect(self) -> bool:
        """判断是否应该进行自省"""
        idle_time = time.time() - self.last_active_time

        if idle_time > self.idle_threshold:
            self.idle_cycles += 1
            return True
        return False

    def mark_active(self):
        """标记活跃"""
        self.last_active_time = time.time()
        self.idle_cycles = 0


class StreamingIdentityCore(nn.Module):
    """
    流式身份核心

    核心特性：
    1. 动态演化的身份向量
    2. 空闲期自省
    3. 累积的"自我一致性"
    """

    def __init__(
        self,
        dim: int = 128,
        idle_threshold: int = 300,
        event_bus=None,
    ):
        super().__init__()
        self.dim = dim

        # 身份向量
        self.identity = IdentityVector(dim)

        # 自省引擎
        self.reflection = ReflectionEngine(dim)

        # 空闲处理器
        self.idle_processor = IdleProcessor(idle_threshold)

        # 事件总线
        self._bus = event_bus
        if self._bus is not None:
            self._bus.subscribe(PERSONALITY_UPDATE, self.on_personality_update, priority=1, name="identity_core")

        # 历史
        self.interaction_history = []
        self.max_history = 1000

        # 状态
        self.state = IdentityState()

    def forward(self, context: dict) -> torch.Tensor:
        """前向：获取当前身份向量"""
        return self.identity()

    def process_input(self, text: str, sentiment: float = 0.0) -> torch.Tensor:
        """
        处理输入，更新身份状态
        """
        # 记录交互
        self.interaction_history.append({
            'text': text,
            'sentiment': sentiment,
            'time': time.time(),
        })

        # 裁剪历史
        if len(self.interaction_history) > self.max_history:
            self.interaction_history.pop(0)

        # 标记活跃
        self.idle_processor.mark_active()

        # 返回当前身份
        return self.forward({})

    def process_idle(self) -> torch.Tensor | None:
        """
        空闲期处理：自省运算

        对应DMN的背景活动
        """
        if not self.idle_processor.should_reflect():
            return None

        # 获取身份向量
        current_identity = self.identity()

        # 自省运算
        reflected = self.reflection.reflect(
            current_identity,
            self.interaction_history[-50:]
        )

        # 更新身份 (软更新)
        with torch.no_grad():
            self.identity.core.data *= 0.95
            self.identity.core.data += reflected * 0.05

        # 更新状态
        self.state.self_reflection_count += 1
        self.state.growth_rate = self._compute_growth()

        return self.identity()

    def _compute_growth(self) -> float:
        """计算成长率"""
        if len(self.interaction_history) < 10:
            return 0.0

        # 基于历史多样性
        recent = self.interaction_history[-50:]
        sentiments = [h.get('sentiment', 0.0) for h in recent]
        return np.std(sentiments) if sentiments else 0.0

    def get_embedding(self) -> np.ndarray:
        """获取身份嵌入向量"""
        return self.identity.get_state()

    def compute_coherence(self) -> float:
        """
        计算自我一致性

        对应DMN研究的"脑指纹" uniqueness
        """
        if len(self.interaction_history) < 10:
            return 0.5

        # 基于历史一致性
        recent = self.interaction_history[-20:]
        texts = [h['text'][:20] for h in recent]

        # 简单：文本重复度
        unique = len(set(texts)) / len(texts)
        return min(1.0, unique)

    def get_state(self) -> IdentityState:
        """获取状态"""
        self.state.coherence = self.compute_coherence()
        self.state.stability = 1.0 - self.state.growth_rate
        return self.state

    def on_personality_update(self, event) -> dict[str, Any]:
        """事件驱动: 响应 PERSONALITY_UPDATE"""
        text = event.data.get("text", "")
        sentiment = event.data.get("sentiment", 0.1)
        result = self.process_input(text, sentiment=sentiment)
        return {"identity_updated": True}

    def get_summary(self) -> dict:
        """获取摘要"""
        return {
            'coherence': self.state.coherence,
            'growth_rate': self.state.growth_rate,
            'stability': self.state.stability,
            'reflection_count': self.state.self_reflection_count,
            'history_size': len(self.interaction_history),
            'idle_cycles': self.idle_processor.idle_cycles,
        }


# ============ 便捷函数 ============

def create_identity_core(dim: int = 128) -> StreamingIdentityCore:
    """创建流式身份核心"""
    return StreamingIdentityCore(dim=dim)


__all__ = [
    "IdentityState",
    "IdentityVector",
    "ReflectionEngine",
    "IdleProcessor",
    "StreamingIdentityCore",
    "create_identity_core",
]
