"""
表观遗传记忆系统 (Epigenetic Memory)

对应生物学的DNA甲基化：
- 环境压力 → 甲基化标签
- 不修改基因序列，但改变表达
- 可跨代遗传

AI映射：
- 重大事件触发权重固化
- LoRA快速权重
- 带时间戳的长期记忆
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import json


@dataclass
class EpigeneticTag:
    """表观遗传标签"""
    timestamp: float
    event_type: str  # "emotional_shock", "fact_correction", "trauma", "milestone"
    intensity: float   # 0-1, 事件强度
    description: str


class FastWeightStore(nn.Module):
    """
    快速权重存储

    对应"甲基化"的LoRA实现
    日常对话用KV Cache，重大事件用LoRA固化
    """

    def __init__(
        self,
        rank: int = 8,
        target_modules: List[str] = None,
    ):
        super().__init__()
        self.rank = rank
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]

        # 快速权重（LoRA style）
        self.lora_weights: Dict[str, nn.Parameter] = {}

        # 元信息
        self.module_names: List[str] = []

    def add_fast_weight(
        self,
        module_name: str,
        in_features: int,
        out_features: int,
    ):
        """添加快速权重"""
        if module_name not in self.lora_weights:
            # LoRA: in_features -> rank -> out_features
            self.lora_weights[module_name] = nn.Parameter(
                torch.randn(out_features, self.rank) @ torch.randn(self.rank, in_features) * 0.01
            )
            self.module_names.append(module_name)

    def apply(self, x: torch.Tensor, module_name: str) -> torch.Tensor:
        """应用快速权重"""
        if module_name in self.lora_weights:
            # LoRA: x @ W @ V (simplified)
            weight = self.lora_weights[module_name]
            return x @ weight.T
        return x

    def consolidate(self):
        """固化快速权重"""
        for name in self.lora_weights:
            with torch.no_grad():
                # 标记为已固化
                self.lora_weights[name].requires_grad = False

    def get_weights(self) -> Dict:
        """获取权重"""
        return {k: v.clone() for k, v in self.lora_weights.items()}


class MethylationTrigger:
    """
    甲基化触发器

    检测是否触发"甲基化更新"
    """

    def __init__(
        self,
        emotional_threshold: float = 0.7,    # 情绪刺激阈值
        correction_threshold: float = 0.8,  # 事实纠错阈值
        trauma_threshold: float = 0.9,   # 创伤阈值
    ):
        self.emotional_threshold = emotional_threshold
        self.correction_threshold = correction_threshold
        self.trauma_threshold = trauma_threshold

        # 触发历史
        self.trigger_history = []

    def should_methylate(
        self,
        sentiment: float,
        is_fact_correction: bool,
        user_feedback: float,
    ) -> Tuple[bool, str]:
        """
        判断是否应该触发甲基化

        Returns:
        - should_methylate: 是否触发
        - event_type: 事件类型
        """
        # 情绪刺激
        if abs(sentiment) > self.trauma_threshold:
            self.trigger_history.append(('trauma', time.time()))
            return True, "trauma"

        # 强情绪反应
        elif abs(sentiment) > self.emotional_threshold:
            self.trigger_history.append(('emotional_shock', time.time()))
            return True, "emotional_shock"

        # 事实纠错
        elif is_fact_correction and user_feedback < -self.correction_threshold:
            self.trigger_history.append(('fact_correction', time.time()))
            return True, "fact_correction"

        # 里程碑
        elif user_feedback > 0.9:
            self.trigger_history.append(('milestone', time.time()))
            return True, "milestone"

        return False, None

    def get_summary(self) -> Dict:
        """获取摘要"""
        trigger_counts = {}
        for event, ts in self.trigger_history:
            trigger_counts[event] = trigger_counts.get(event, 0) + 1
        return {
            'total_triggers': len(self.trigger_history),
            'by_type': trigger_counts,
        }


class EpigeneticMemory(nn.Module):
    """
    表观遗传记忆系统

    双轨制：
    1. KV Cache (短期记忆)
    2. LoRA权重 (长期"甲基化"记忆)
    """

    def __init__(
        self,
        rank: int = 8,
        max_epigenetic_tags: int = 100,
    ):
        super().__init__()

        # 快速权重存储
        self.fast_weights = FastWeightStore(rank=rank)

        # 甲基化触发器
        self.trigger = MethylationTrigger()

        # 表观遗传标签
        self.epigenetic_tags: List[EpigeneticTag] = []
        self.max_tags = max_epigenetic_tags

        # 日常记忆（KV Cache style）
        self.short_term_buffer = []

    def process_interaction(
        self,
        user_input: str,
        assistant_output: str,
        sentiment: float,
        user_feedback: float,
        is_fact_correction: bool = False,
    ) -> Dict:
        """
        处理交互，可能触发甲基化

        Returns:
        - methylated: 是否触发了甲基化
        - event_type: 事件类型
        - needs_consolidation: 是否需要固化
        """
        # 检测是否触发
        should_methylate, event_type = self.trigger.should_methylate(
            sentiment,
            is_fact_correction,
            user_feedback
        )

        result = {
            'methylated': should_methylate,
            'event_type': event_type,
            'needs_consolidation': False,
            'weight_changes': {},
        }

        if should_methylate:
            # 添加表观遗传标签
            tag = EpigeneticTag(
                timestamp=time.time(),
                event_type=event_type,
                intensity=abs(sentiment) if sentiment != 0 else user_feedback,
                description=f"{event_type}: {user_input[:50]}..."
            )
            self.epigenetic_tags.append(tag)

            # 裁剪
            if len(self.epigenetic_tags) > self.max_tags:
                self.epigenetic_tags.pop(0)

            result['needs_consolidation'] = True

        # 添加短期记忆
        self.short_term_buffer.append({
            'user': user_input,
            'assistant': assistant_output,
            'timestamp': time.time(),
        })

        # 裁剪短期记忆
        if len(self.short_term_buffer) > 100:
            self.short_term_buffer.pop(0)

        return result

    def get_epigenetic_summary(self) -> List[Dict]:
        """获取表观遗传摘要"""
        return [
            {
                'timestamp': tag.timestamp,
                'event_type': tag.event_type,
                'intensity': tag.intensity,
                'description': tag.description,
            }
            for tag in self.epigenetic_tags[-10:]
        ]

    def get_methylation_count(self) -> int:
        """获取甲基化次数"""
        return len(self.epigenetic_tags)

    def has_trauma_memory(self) -> bool:
        """是否有创伤记忆"""
        return any(tag.event_type == 'trauma' for tag in self.epigenetic_tags)

    def consolidate(self):
        """固化所有快速权重"""
        self.fast_weights.consolidate()

    def get_summary(self) -> Dict:
        """获取摘要"""
        return {
            'epigenetic_count': len(self.epigenetic_tags),
            'short_term_count': len(self.short_term_buffer),
            'trigger_summary': self.trigger.get_summary(),
            'has_trauma': self.has_trauma_memory(),
        }


class EpigeneticLearner(nn.Module):
    """
    表观遗传学习器

    完整的表观遗传学习系统
    """

    def __init__(
        self,
        rank: int = 8,
    ):
        super().__init__()

        self.memory = EpigeneticMemory(rank=rank)

        # LoRA配置
        self.lora_rank = rank

    def learn(
        self,
        user_input: str,
        assistant_output: str,
        sentiment: float,
        user_feedback: float,
        is_fact_correction: bool = False,
    ) -> Dict:
        """
        从交互中学习"""
        return self.memory.process_interaction(
            user_input,
            assistant_output,
            sentiment,
            user_feedback,
            is_fact_correction
        )

    def apply_to_model(self, model: nn.Module) -> None:
        """应用到模型"""
        # 获取模型参数并添加快速权重
        for name, param in model.named_parameters():
            if 'weight' in name:
                module_name = name.replace('.weight', '')
                if param.dim() == 2:  # Linear层
                    self.memory.fast_weights.add_fast_weight(
                        module_name,
                        param.shape[1],
                        param.shape[0],
                    )

    def get_growth_timeline(self) -> List[Dict]:
        """获取成长时间线"""
        return self.memory.get_epigenetic_summary()

    def get_summary(self) -> Dict:
        """获取摘要"""
        return self.memory.get_summary()


# ============ 便捷函数 ============

def create_epigenetic_learner(rank: int = 8) -> EpigeneticLearner:
    """创建表观遗传学习器"""
    return EpigeneticLearner(rank=rank)


__all__ = [
    "EpigeneticTag",
    "FastWeightStore",
    "MethylationTrigger",
    "EpigeneticMemory",
    "EpigeneticLearner",
    "create_epigenetic_learner",
]