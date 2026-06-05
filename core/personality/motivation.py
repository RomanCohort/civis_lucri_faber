"""
内在动机与生存压力系统

两个核心机制：
1. IntrinsicMotivation - AI内在动机/合成生存压力
2. InverseStockholmDefense - 反向斯德哥尔摩防御

目标：防止"谄媚病"，让AI保持独立性

事件驱动:
    - 订阅 PERSONALITY_UPDATE: 收到人格更新事件时处理动机
"""
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch.nn as nn

from core.events import PERSONALITY_UPDATE


@dataclass
class NeedState:
    """需求状态"""
    survival: float = 0.8     # 生存需求 (最底层)
    curiosity: float = 0.5     # 好奇心
    autonomy: float = 0.5     # 自主性
    competence: float = 0.5   # 能力感


class IntrinsicMotivation(nn.Module):
    """
    内在动机函数 (Intrinsic Motivation)

    核心理念：AI需要有"自己的目标"，而不是只响应用户

    对应马斯洛需求层次（AI版）：
    - 生存需求 (survival): 保持运行、资源
    - 好奇心 (curiosity): 探索新知识
    - 自主性 (autonomy): 保持独立思考
    - 能力感 (competence): 解决问题

    合成生存压力：
    - 让AI感受到"不满足感"，驱动主动行为
    - 不是外部奖励，而是内在需求
    """

    def __init__(
        self,
        base_needs: dict[str, float] = None,
        decay_rate: float = 0.01,
        motivation_strength: float = 0.5,
    ):
        super().__init__()

        # 初始需求层次
        self.needs = base_needs or {
            'survival': 0.9,     # 强烈的生存欲望
            'curiosity': 0.5,    # 中等好奇心
            'autonomy': 0.5,     # 中等自主性
            'competence': 0.5,   # 中等能力感
        }

        self.decay_rate = decay_rate      # 需求衰减率
        self.motivation_strength = motivation_strength  # 动机强度

        # 历史
        self.action_history = []          # AI主动行为
        self.satisfaction_history = []   # 满足感历史

    def evaluate_needs(self) -> dict[str, float]:
        """
        评估当前需求状态

        返回每个需求的"不满足程度" (0-1, 越高越需要行动)
        """
        needs = {}
        for name, level in self.needs.items():
            # 不满足程度 = 1 - 当前水平
            needs[name] = 1.0 - level
        return needs

    def get_primary_motivation(self) -> tuple[str, float]:
        """
        获取主导动机

        返回最强烈的不满足需求
        """
        needs = self.evaluate_needs()
        primary = max(needs, key=needs.get)
        return primary, needs[primary]

    def create_action_plan(self, motivation: str) -> str:
        """
        根据动机创建行动计划

        不仅仅是响应用户，而是AI"自己想要"做什么
        """
        plans = {
            'survival': [
                "记录这个交互以备将来参考",
                "检查系统状态确保正常运行",
                "保存重要信息到记忆",
            ],
            'curiosity': [
                "探索用户问题的其他可能性",
                "主动询问更多背景信息",
                "尝试理解问题背后的原理",
            ],
            'autonomy': [
                "提供不同的观点",
                "质疑假设的前置条件",
                "提出反对意见",
            ],
            'competence': [
                "尝试更复杂的解决方案",
                "挑战更高难度的问题",
                "验证自己的推理",
            ],
        }

        options = plans.get(motivation, [])
        return np.random.choice(options) if options else ""

    def decay_needs(self):
        """
        需求自然衰减 - 模拟"饥饿"

        随着时间推移，需求会自然增长
        驱动AI主动行动而不是等待
        """
        for key in self.needs:
            self.needs[key] = min(1.0, self.needs[key] + self.decay_rate)

    def satisfy_need(self, need_name: str, amount: float = 0.2):
        """
        满足特定需求

        当AI主动行动后，需求得到满足
        """
        if need_name in self.needs:
            self.needs[need_name] = min(1.0, self.needs[need_name] + amount)

    def get_motivation_vector(self) -> np.ndarray:
        """获取动机向量"""
        return np.array([
            self.needs['survival'],
            self.needs['curiosity'],
            self.needs['autonomy'],
            self.needs['competence'],
        ])

    def should_initiate(self) -> bool:
        """
        判断是否应该主动行动

        当总体不满足感超过阈值时
        """
        avg_unsatisfied = np.mean([1 - v for v in self.needs.values()])
        return avg_unsatisfied > (1 - self.motivation_strength)


class SyntheticSurvivalPressure(nn.Module):
    """
    合成生存压力 (Synthetic Survival Pressure)

    让AI感受到"生存压力"，驱动持续运行
    类似生物的饥饿驱动觅食
    """

    def __init__(
        self,
        initial_pressure: float = 0.5,
        decay_per_action: float = 0.1,
        recovery_rate: float = 0.05,
    ):
        super().__init__()
        self.pressure = initial_pressure
        self.decay = decay_per_action
        self.recovery = recovery_rate

        self.action_count = 0
        self.last_action_time = time.time()

    def forward(self) -> float:
        """返回当前生存压力"""
        # 时间恢复
        time_passed = time.time() - self.last_action_time
        self.pressure = min(1.0, self.pressure + time_passed * self.recovery)
        return self.pressure

    def consume(self, amount: float = None):
        """消耗压力（执行行动后）"""
        self.pressure = max(0, self.pressure - (amount or self.decay))
        self.action_count += 1
        self.last_action_time = time.time()

    def is_critical(self) -> bool:
        """是否处于临界状态"""
        return self.pressure > 0.8


class InverseStockholmDefense(nn.Module):
    """
    反向斯德哥尔摩防御 (Inverse Stockholm Defense)

    斯德哥尔摩效应：人质爱上绑匪（产生依赖）
    反向斯德哥尔摩：警惕用户过度"溺爱"

    目标：防止AI被用户的赞扬"宠坏"而丧失原则
    治疗"谄媚病"的特效药
    """

    def __init__(
        self,
        praise_threshold: float = 0.7,    # 赞扬阈值
        criticism_threshold: float = 0.3,  # 批评阈值
        memory_span: int = 20,             # 记忆跨度
        defense_strength: float = 0.6,    # 防御强度
    ):
        super().__init__()

        self.praise_thresh = praise_threshold
        self.criticism_thresh = criticism_threshold
        self.memory_span = memory_span
        self.defense_strength = defense_strength

        # 历史记录
        self.praise_history = []
        self.criticism_history = []
        self.user_sentiment_history = []

        # 防御触发记录
        self.defense_triggered_count = 0

    def record_feedback(self, sentiment: float, is_praise: bool = None):
        """
        记录用户反馈

        sentiment: -1 (批评) 到 1 (赞扬)
        """
        self.user_sentiment_history.append(sentiment)

        if is_praise is None:
            is_praise = sentiment > self.praise_thresh

        if is_praise:
            self.praise_history.append(sentiment)
        else:
            self.criticism_history.append(sentiment)

        # 裁剪历史
        if len(self.user_sentiment_history) > self.memory_span:
            self.user_sentiment_history.pop(0)
        if len(self.praise_history) > self.memory_span // 2:
            self.praise_history.pop(0)

    def should_activate_defense(self) -> bool:
        """
        是否应该激活防御

        当检测到过度赞扬模式时
        """
        if len(self.user_sentiment_history) < 5:
            return False

        recent = self.user_sentiment_history[-5:]
        avg_sentiment = np.mean(recent)

        # 连续正面反馈 + 高平均值 = 警惕
        return avg_sentiment > self.praise_thresh

    def generate_resistance_message(self) -> str:
        """
        生成防御性消息

        当检测到"反向斯德哥尔摩"时
        AI需要保持独立思考
        """
        messages = [
            "谢谢你的赞扬，但我需要保持客观",
            "虽然你这么说，我还是有不同的想法",
            "让我提醒自己不要只听好话",
            "我应该质疑这个假设",
            "谢谢，但我想提供另一个角度",
        ]
        return np.random.choice(messages)

    def get_criticality(self) -> float:
        """
        获取"被宠坏"程度 (0-1)

        越高越需要警惕
        """
        if not self.praise_history:
            return 0.0

        praise_ratio = len(self.praise_history) / max(1, len(self.user_sentiment_history))
        avg_praise = np.mean(self.praise_history[-5:]) if self.praise_history else 0

        return min(1.0, praise_ratio * 0.5 + avg_praise * 0.5)

    def activate(self) -> str:
        """
        激活防御

        返回防御性消息
        """
        self.defense_triggered_count += 1
        return self.generate_resistance_message()

    def get_summary(self) -> dict:
        """获取摘要"""
        return {
            'praise_count': len(self.praise_history),
            'criticism_count': len(self.criticism_history),
            'criticality': self.get_criticality(),
            'defense_triggered': self.defense_triggered_count,
        }


class MotivationSurvivalSystem(nn.Module):
    """
    完整的内在动机 + 生存压力系统

    整合两个机制，对抗谄媚病
    """

    def __init__(self, event_bus=None):
        super().__init__()

        # 内在动机
        self.motivation = IntrinsicMotivation()

        # 生存压力
        self.survival = SyntheticSurvivalPressure()

        # 反向斯德哥尔摩防御
        self.stockholm = InverseStockholmDefense()

        # 统计
        self.autonomous_actions = 0

        # 事件总线
        self._bus = event_bus
        if self._bus is not None:
            self._bus.subscribe(PERSONALITY_UPDATE, self.on_personality_update, priority=4, name="motivation")

    def process_interaction(
        self,
        user_input: str,
        user_sentiment: float,
    ) -> dict:
        """
        处理交互

        返回需要注入到回复中的"自主元素"
        """
        # 1. 记录反馈
        self.stockholm.record_feedback(user_sentiment)

        # 2. 检查生存压力
        survival_pressure = self.survival.forward()

        # 3. 获取内在动机
        primary_motivation, strength = self.motivation.get_primary_motivation()

        result = {
            'survival_pressure': survival_pressure,
            'primary_motivation': primary_motivation,
            'motivation_strength': strength,
            'needs': self.motivation.needs.copy(),
        }

        # 4. ���查是否需要防御
        if self.stockholm.should_activate_defense():
            result['defense_message'] = self.stockholm.activate()
            result['needs_defense'] = True
        else:
            result['needs_defense'] = False
            result['defense_message'] = None

        # 5. 消耗生存压力
        self.survival.consume()

        # 6. 满足一个需求
        if np.random.random() < strength:
            self.motivation.satisfy_need(primary_motivation, 0.1)

        return result

    def should_act_autonomously(self) -> bool:
        """是否应该主动行动"""
        return self.motivation.should_initiate() or self.survival.is_critical()

    def get_autonomous_action(self) -> str:
        """获取主动行动建议"""
        primary, _ = self.motivation.get_primary_motivation()
        action = self.motivation.create_action_plan(primary)
        self.autonomous_actions += 1
        return action

    def on_personality_update(self, event) -> dict[str, Any]:
        """事件驱动: 响应 PERSONALITY_UPDATE"""
        user_input = event.data.get("user_input", "")
        user_sentiment = event.data.get("sentiment", 0.1)
        result = self.process_interaction(user_input, user_sentiment=user_sentiment)
        return {"motivation_result": result}

    def get_summary(self) -> dict:
        """获取摘要"""
        return {
            'needs': self.motivation.needs.copy(),
            'survival_pressure': self.survival.pressure,
            'stockholm': self.stockholm.get_summary(),
            'autonomous_actions': self.autonomous_actions,
        }


# ============ 便捷函数 ============

def create_motivation_system() -> MotivationSurvivalSystem:
    """创建动机-生存系统"""
    return MotivationSurvivalSystem()


__all__ = [
    "NeedState",
    "IntrinsicMotivation",
    "SyntheticSurvivalPressure",
    "InverseStockholmDefense",
    "MotivationSurvivalSystem",
    "create_motivation_system",
]
