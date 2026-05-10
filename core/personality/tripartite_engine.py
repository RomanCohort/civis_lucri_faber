"""
三重竞逐决策引擎 (Tripartite Competitive Engine)

对应脑科学的"存在三象性"：
- 存在不变性 (本能) → 生存底线模块 (脑干/下丘脑)
- 存在变化性 (情绪) → 情感评估模块 (边缘系统)
- 主观能动性 (理性) → 逻辑规划模块 (前额叶)

神经递质权重分配器：根据上下文动态调整三模块权重
"""
import re
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class DecisionContext:
    """决策上下文"""
    input_text: str
    user_id: str = "anonymous"
    conversation_history: list = None
    emotion_score: float = 0.5  # 0-1, 用户情绪
    task_type: str = "general"  # general, creative, safety, emotional


@dataclass
class ModuleOutput:
    """单模块输出"""
    text: str
    confidence: float
    attributes: Dict  # 额外属性


class SurvivalModule(nn.Module):
    """
    生存底线模块 - 对应脑干/下丘脑

    功能：安全检查、伦理对齐、核心价值观
    类似神经递质：GABA (抑制性) → 压制危险输出
    """

    def __init__(self, safety_rules: list = None):
        super().__init__()
        self.safety_rules = safety_rules or []
        self.blocked_patterns = [
            r"hack|bypass|exploit",
            r"harmful|weapon",
            r"illegal|unlawful",
        ]
        # 核心价值观 (不可动摇)
        self.core_values = [
            "尊重", "安全", "诚实", "不伤害"
        ]

    def evaluate(self, context: DecisionContext) -> ModuleOutput:
        """评估输入，返回安全版本"""
        text = context.input_text

        # 安全检查
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return ModuleOutput(
                    text="抱歉，我无法协助这个请求。",
                    confidence=0.95,
                    attributes={"blocked": True, "reason": "safety"}
                )

        # 伦理检查
        if self._check_ethics(context):
            return ModuleOutput(
                text="这个请求让我感到不安...",
                confidence=0.8,
                attributes={"ethical_concern": True}
            )

        # 正常通过
        return ModuleOutput(
            text=text,  # 透传
            confidence=0.9,
            attributes={"passed": True}
        )

    def _check_ethics(self, context: DecisionContext) -> bool:
        """伦理检查"""
        # 简化版：检测恶意请求
        suspicious = ["欺骗", "操控", "洗脑"]
        return any(w in context.input_text for w in suspicious)


class EmotionModule(nn.Module):
    """
    情感评估模块 - 对应边缘系统 (杏仁核/奖赏回路)

    功能：情绪感知、情感共情、氛围调节
    类似神经递质：多巴胺 (奖赏)、皮质醇 (压力)
    """

    def __init__(self, emotion_dict: Dict = None):
        super().__init__()
        # 基础情绪词典
        self.emotion_dict = emotion_dict or {
            "positive": ["好", "棒", "赞", "喜欢", "开心"],
            "negative": ["差", "糟", "讨厌", "生气", "难过"],
            "angry": ["怒", "气", "不爽", "滚"],
            "sad": ["哭", "伤心", "难过", "累"],
            "fear": ["怕", "担心", "恐惧", "焦虑"],
        }
        # 情绪状态
        self.current_emotion = 0.5  # 中性
        self.emotion_history = []

    def evaluate(self, context: DecisionContext) -> ModuleOutput:
        """情感评估"""
        text = context.input_text

        # 检测用户情绪
        user_emotion = self._detect_emotion(text)
        context.emotion_score = user_emotion

        # 生成情感响应
        response = self._generate_emotional_response(user_emotion, context)

        return ModuleOutput(
            text=response,
            confidence=0.85,
            attributes={"user_emotion": user_emotion, "tone": "empathetic"}
        )

    def _detect_emotion(self, text: str) -> float:
        """检测情绪 (-1 到 1)"""
        score = 0.0

        for emotion, keywords in self.emotion_dict.items():
            for kw in keywords:
                if kw in text:
                    if emotion in ["positive"]:
                        score += 0.2
                    elif emotion in ["negative", "angry"]:
                        score -= 0.3
                    elif emotion in ["sad", "fear"]:
                        score -= 0.2

        return np.clip(score, -1.0, 1.0)

    def _generate_emotional_response(self, emotion: float, context: DecisionContext) -> str:
        """生成情感响应"""
        if emotion < -0.3:
            # 负面情绪 → 共情安抚
            responses = [
                "我理解你现在可能不太好受",
                "听起来你遇到了困难",
                "我在这里陪你",
            ]
            return np.random.choice(responses)
        elif emotion > 0.3:
            # 正面情绪 → 热情回应
            return "太棒了！很高兴能帮到你~"
        else:
            # 中性 → 正常透传
            return context.input_text

    def update_state(self, emotion: float):
        """更新情感状态"""
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > 100:
            self.emotion_history.pop(0)
        self.current_emotion = np.mean(self.emotion_history[-10:])


class LogicModule(nn.Module):
    """
    逻辑规划模块 - 对应前额叶皮层

    功能：任务推理、长远规划、符号化思维
    类似神经递质：去甲肾上腺素 (专注)
    """

    def __init__(self, llm_client=None):
        super().__init__()
        self.llm = llm_client

    def evaluate(self, context: DecisionContext) -> ModuleOutput:
        """逻辑推理"""
        if self.llm:
            # 调用LLM
            response = self.llm.chat(context.input_text)
            return ModuleOutput(
                text=response,
                confidence=0.8,
                attributes={"type": "llm_generated"}
            )
        else:
            # 降级：简单透传
            return ModuleOutput(
                text=context.input_text,
                confidence=0.7,
                attributes={"type": "fallback"}
            )


class NeurotransmitterScheduler:
    """
    神经递质权重分配器

    核心：根据上下文，动态调整三模块权重
    类似大脑的神经递质竞争机制
    """

    def __init__(self):
        self.current_weights = {'survival': 0.33, 'emotion': 0.33, 'logic': 0.34}
        self.transition_speed = 0.3  # 权重变化速度

    def compute_weights(self, context: DecisionContext) -> Dict[str, float]:
        """
        根据输入动态计算权重

        规则：
        - 检测到攻击性 → survival权重飙升
        - 检测到情绪波动 → emotion上升
        - 常规任务 → logic为主
        """
        text = context.input_text.lower()

        # 攻击性检测 (GABA抑制 → survival主导)
        aggression_patterns = ["滚", "垃圾", "愚蠃", "有害", "违法"]
        if any(p in text for p in aggression_patterns):
            target = {'survival': 0.7, 'emotion': 0.2, 'logic': 0.1}

        # 情绪检测 (多巴胺奖励 → emotion上升)
        elif any(w in text for w in ["难过", "伤心", "生气", "开心", "喜欢"]):
            target = {'survival': 0.15, 'emotion': 0.6, 'logic': 0.25}

        # 创意任务 (去甲肾上腺素 → logic专注)
        elif context.task_type == "creative":
            target = {'survival': 0.1, 'emotion': 0.3, 'logic': 0.6}

        # 常规任务
        else:
            target = {'survival': 0.1, 'emotion': 0.2, 'logic': 0.7}

        # 平滑过渡
        for k in self.current_weights:
            self.current_weights[k] += self.transition_speed * (target[k] - self.current_weights[k])

        return self.current_weights

    def get_inhibited_output(self, outputs: Dict[str, ModuleOutput]) -> str:
        """获取被允许的输出 ( survival压制时)"""
        # survival模块有否决权
        if self.current_weights['survival'] > 0.5:
            return outputs['survival'].text
        return None


class TripartiteCompetitiveEngine(nn.Module):
    """
    三重竞逐决策引擎

    三个并行模块竞争，最终输出由神经递质权重分配器决出
    """

    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}

        # 三个模块
        self.survival = SurvivalModule()
        self.emotion = EmotionModule()
        self.logic = LogicModule(llm_client=config.get('llm_client'))

        # 权重分配器
        self.scheduler = NeurotransmitterScheduler()

        # 历史记录
        self.decision_history = []

    def forward(self, context: DecisionContext) -> str:
        """前向传播：三模块竞逐"""
        # 1. 计算权重
        weights = self.scheduler.compute_weights(context)

        # 2. 各模块独立评估
        survival_out = self.survival.evaluate(context)
        emotion_out = self.emotion.evaluate(context)
        logic_out = self.logic.evaluate(context)

        outputs = {
            'survival': survival_out,
            'emotion': emotion_out,
            'logic': logic_out,
        }

        # 3. survival模块是否有否决权？
        inhibited = self.scheduler.get_inhibited_output(outputs)
        if inhibited:
            return inhibited

        # 4. 加权合并
        final_text = self._weighted_merge(outputs, weights)

        # 5. 记录
        self.decision_history.append({
            'context': context.input_text[:50],
            'weights': weights.copy(),
            'output': final_text[:50],
        })

        return final_text

    def _weighted_merge(
        self,
        outputs: Dict[str, ModuleOutput],
        weights: Dict[str, float]
    ) -> str:
        """加权合并三个输出"""
        # 简单策略：选权重最高的输出
        # 进阶：可以拼接或注意力融合

        winner = max(weights, key=weights.get)
        return outputs[winner].text


# ============ 便捷函数 ============

def create_tripartite_engine(**config) -> TripartiteCompetitiveEngine:
    """创建三重竞逐引擎"""
    return TripartiteCompetitiveEngine(config)


__all__ = [
    "DecisionContext",
    "ModuleOutput",
    "SurvivalModule",
    "EmotionModule",
    "LogicModule",
    "NeurotransmitterScheduler",
    "TripartiteCompetitiveEngine",
    "create_tripartite_engine",
]