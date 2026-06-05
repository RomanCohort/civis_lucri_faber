"""
记忆-语言适配器 (Memory-Language Adapter)

将记忆系统的检索结果映射为语言风格和内容调制。

核心设计：
1. 情感记忆转移 — 检索到的记忆的情绪影响当前回复语气
2. 重要性加权 — 高重要性记忆在回复中更突出
3. 用户模式记忆 — 与特定用户的交互历史影响回复风格
4. 时效性衰减 — 近期记忆比远期记忆影响更大

生物学对应：
- 海马体 → 记忆检索与情感标记
- 杏仁核 → 记忆的情绪着色
- 前额叶 → 记忆的上下文整合
- 默认模式网络 → 自传体记忆与自我参照

数据流：
  MemorySystem.retrieve()
       ↓
  MemoryLanguageAdapter.process()
       ↓
  ┌─ 情感着色 (检索记忆的valence → 当前语气)
  ├─ 重要性加权 (importance → 回复重点)
  ├─ 用户模式 (user_memory → 个性化风格)
  └─ 时效性 (recency → 记忆鲜活性)
       ↓
  Style Modifiers → BioLinguisticCoupler / System Prompt
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class MemoryInfluence:
    """单条记忆对语言的影响"""
    content: str
    valence: float = 0.0          # -1.0 (消极) ~ 1.0 (积极)
    arousal: float = 0.5          # 唤醒度
    importance: float = 0.5       # 重要性
    recency: float = 1.0          # 时效性 (0=很久前, 1=刚刚)
    emotion_tag: str = 'neutral'  # 情绪标签
    source: str = ''              # 来源
    weight: float = 1.0           # 综合权重


@dataclass
class MemoryStyleState:
    """记忆驱动的风格状态"""
    # 情感着色
    emotional_valence: float = 0.0    # 记忆综合效价
    emotional_arousal: float = 0.5    # 记忆综合唤醒
    dominant_emotion: str = 'neutral' # 主导情绪

    # 重要性
    avg_importance: float = 0.5       # 平均重要性
    max_importance: float = 0.5       # 最高重要性

    # 用户模式
    user_familiarity: float = 0.0     # 用户熟悉度
    user_interaction_count: int = 0   # 交互次数
    user_preferred_style: str = 'balanced'  # 用户偏好风格

    # 时效性
    avg_recency: float = 0.5          # 平均时效性

    # 风格指令
    tone_modifier: str = ''           # 语气修饰
    content_emphasis: list[str] = field(default_factory=list)  # 内容重点
    style_constraints: list[str] = field(default_factory=list)  # 风格约束


class MemoryLanguageAdapter:
    """
    记忆-语言适配器

    将记忆检索结果映射为语言风格参数。
    记忆不仅提供内容（RAG），还影响说话方式。
    """

    # 情绪→语气映射
    EMOTION_TONE_MAP = {
        'joy': 'warm_enthusiastic',
        'excitement': 'energetic',
        'trust': 'open_friendly',
        'surprise': 'curious',
        'sadness': 'gentle_comforting',
        'fear': 'cautious_reassuring',
        'anger': 'calm_measured',
        'disgust': 'neutral_distant',
        'neutral': 'balanced',
        'nostalgia': 'warm_reflective',
        'gratitude': 'warm_appreciative',
        'embarrassment': 'humble_self_deprecating',
    }

    # 用户熟悉度→交互风格
    FAMILIARITY_STYLE_MAP = {
        (0.0, 0.2): 'formal_polite',       # 陌生：正式礼貌
        (0.2, 0.4): 'friendly_professional', # 初识：友好专业
        (0.4, 0.6): 'warm_collaborative',   # 熟悉：温暖协作
        (0.6, 0.8): 'casual_intimate',      # 亲密：随意亲近
        (0.8, 1.0): 'deeply_connected',     # 深度连接：高度个性化
    }

    # 重要性→回复结构
    IMPORTANCE_STRUCTURE = {
        (0.0, 0.3): 'brief_mention',    # 低重要性：简单提及
        (0.3, 0.6): 'moderate_detail',  # 中等：适度展开
        (0.6, 0.8): 'detailed_focus',   # 高：详细聚焦
        (0.8, 1.0): 'central_theme',    # 极高：核心主题
    }

    def __init__(self, decay_rate: float = 0.1):
        """
        Args:
            decay_rate: 记忆时效性衰减率
        """
        self.decay_rate = decay_rate
        self._style_state = MemoryStyleState()
        self._influence_history: list[MemoryInfluence] = []

    def process_memories(
        self,
        memories: list[dict],
        current_bio_state: dict | None = None,
        user_id: str | None = None,
    ) -> MemoryStyleState:
        """
        处理检索到的记忆，生成风格状态

        Args:
            memories: 检索到的记忆列表，每条包含:
                - content: 内容
                - valence: 效价 (-1~1)
                - arousal: 唤醒度
                - importance: 重要性
                - timestamp: 时间戳
                - emotion_tag: 情绪标签
                - source: 来源
            current_bio_state: 当前生物状态（影响记忆权重）
            user_id: 用户ID

        Returns:
            MemoryStyleState
        """
        if not memories:
            return self._style_state

        influences = []

        for mem in memories:
            influence = MemoryInfluence(
                content=mem.get('content', ''),
                valence=float(mem.get('valence', 0.0)),
                arousal=float(mem.get('arousal', 0.5)),
                importance=float(mem.get('importance', 0.5)),
                recency=self._compute_recency(mem.get('timestamp', None)),
                emotion_tag=mem.get('emotion_tag', 'neutral'),
                source=mem.get('source', ''),
            )

            # 计算综合权重
            influence.weight = self._compute_weight(
                influence, current_bio_state
            )
            influences.append(influence)

        # 按权重排序
        influences.sort(key=lambda x: x.weight, reverse=True)
        self._influence_history = influences

        # 聚合为风格状态
        state = self._aggregate_influences(influences)

        # 用户模式更新
        if user_id:
            self._update_user_pattern(state, user_id, influences)

        self._style_state = state
        return state

    def _compute_recency(self, timestamp) -> float:
        """计算时效性"""
        if timestamp is None:
            return 0.5

        try:
            if isinstance(timestamp, (int, float)):
                # Unix timestamp
                mem_time = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                mem_time = datetime.fromisoformat(timestamp)
            elif isinstance(timestamp, datetime):
                mem_time = timestamp
            else:
                return 0.5

            delta = datetime.now() - mem_time
            hours = delta.total_seconds() / 3600
            # 指数衰减
            recency = math.exp(-self.decay_rate * hours / 24)
            return float(np.clip(recency, 0, 1))
        except Exception:
            return 0.5

    def _compute_weight(
        self, influence: MemoryInfluence, bio_state: dict | None
    ) -> float:
        """
        计算记忆的综合权重

        考虑：重要性 × 时效性 × 情绪共振 × 生物状态调制
        """
        weight = influence.importance * influence.recency

        # 情绪共振：记忆情绪与当前情绪一致时权重更高
        if bio_state is not None:
            current_valence = float(bio_state.get('valence', 0.0))
            current_arousal = float(bio_state.get('arousal', 0.5))

            # 效价一致性加成
            valence_similarity = 1.0 - abs(influence.valence - current_valence) / 2.0
            weight *= (0.5 + valence_similarity * 0.5)

            # 高唤醒时，高唤醒记忆权重更高
            if current_arousal > 0.6:
                weight *= (0.8 + influence.arousal * 0.2)

            # 皮质醇高时，负面记忆权重更高（压力偏向）
            cortisol = float(bio_state.get('cortisol', 0.3))
            if cortisol > 0.5 and influence.valence < 0:
                weight *= 1.2

            # 血清素低时，情绪记忆权重更高
            serotonin = float(bio_state.get('serotonin', 0.5))
            if serotonin < 0.4:
                weight *= (1.0 + abs(influence.valence) * 0.3)

        return float(np.clip(weight, 0, 2.0))

    def _aggregate_influences(
        self, influences: list[MemoryInfluence]
    ) -> MemoryStyleState:
        """聚合多条记忆的影响"""
        if not influences:
            return MemoryStyleState()

        # 加权平均效价
        total_weight = sum(inf.weight for inf in influences) or 1.0
        weighted_valence = sum(
            inf.valence * inf.weight for inf in influences
        ) / total_weight
        weighted_arousal = sum(
            inf.arousal * inf.weight for inf in influences
        ) / total_weight

        # 主导情绪
        emotion_weights: dict[str, float] = {}
        for inf in influences:
            emotion_weights[inf.emotion_tag] = (
                emotion_weights.get(inf.emotion_tag, 0) + inf.weight
            )
        dominant_emotion = max(emotion_weights, key=emotion_weights.get) if emotion_weights else 'neutral'

        # 重要性
        avg_importance = sum(inf.importance for inf in influences) / len(influences)
        max_importance = max(inf.importance for inf in influences)

        # 时效性
        avg_recency = sum(inf.recency for inf in influences) / len(influences)

        # 生成风格指令
        tone_modifier = self.EMOTION_TONE_MAP.get(
            dominant_emotion, 'balanced'
        )

        # 内容重点（高重要性记忆的关键词）
        content_emphasis = []
        for inf in influences[:3]:  # 取前3条高权重记忆
            if inf.importance > 0.5:
                # 提取关键词（简单实现：取前几个词）
                words = inf.content.split()[:5]
                content_emphasis.extend(words)

        # 风格约束
        style_constraints = []
        if weighted_valence < -0.3:
            style_constraints.append("回复应体现对困难的理解")
        if weighted_arousal > 0.7:
            style_constraints.append("可以表达兴奋或激动")
        if avg_recency > 0.8:
            style_constraints.append("这是近期的重要记忆，可以自然地引用")
        if max_importance > 0.8:
            style_constraints.append("这是核心记忆，应在回复中重点体现")

        return MemoryStyleState(
            emotional_valence=weighted_valence,
            emotional_arousal=weighted_arousal,
            dominant_emotion=dominant_emotion,
            avg_importance=avg_importance,
            max_importance=max_importance,
            avg_recency=avg_recency,
            tone_modifier=tone_modifier,
            content_emphasis=content_emphasis[:10],
            style_constraints=style_constraints,
        )

    def _update_user_pattern(
        self,
        state: MemoryStyleState,
        user_id: str,
        influences: list[MemoryInfluence],
    ):
        """更新用户交互模式"""
        # 基于记忆数量估算熟悉度
        user_memories = [i for i in influences if i.source == user_id or True]
        interaction_count = len(user_memories)
        familiarity = min(1.0, interaction_count / 20.0)

        state.user_familiarity = familiarity
        state.user_interaction_count = interaction_count

        # 确定用户偏好风格
        for (low, high), style in self.FAMILIARITY_STYLE_MAP.items():
            if low <= familiarity < high:
                state.user_preferred_style = style
                break

    def get_style_modifiers(self) -> dict:
        """
        获取记忆驱动的风格修饰符

        Returns:
            风格修饰符字典
        """
        state = self._style_state

        # 确定交互风格
        interaction_style = 'friendly_professional'
        for (low, high), style in self.FAMILIARITY_STYLE_MAP.items():
            if low <= state.user_familiarity < high:
                interaction_style = style
                break

        # 确定回复结构
        response_structure = 'moderate_detail'
        for (low, high), structure in self.IMPORTANCE_STRUCTURE.items():
            if low <= state.avg_importance < high:
                response_structure = structure
                break

        return {
            'emotional_valence': state.emotional_valence,
            'emotional_arousal': state.emotional_arousal,
            'dominant_emotion': state.dominant_emotion,
            'tone_modifier': state.tone_modifier,
            'interaction_style': interaction_style,
            'response_structure': response_structure,
            'content_emphasis': state.content_emphasis,
            'style_constraints': state.style_constraints,
            'user_familiarity': state.user_familiarity,
            'avg_importance': state.avg_importance,
            'avg_recency': state.avg_recency,
        }

    def generate_memory_prompt(self) -> str:
        """
        生成记忆驱动的提示词片段

        Returns:
            提示词文本
        """
        state = self._style_state
        parts = []

        # 1. 情感着色
        if abs(state.emotional_valence) > 0.2:
            if state.emotional_valence > 0.3:
                parts.append("你回忆起了一些愉快的事情，语气可以更温暖。")
            elif state.emotional_valence < -0.3:
                parts.append("你回忆起了一些不太愉快的事情，语气应该更温和、更理解。")

        # 2. 主导情绪
        if state.dominant_emotion != 'neutral':
            emotion_hints = {
                'joy': '你想起了一些开心的事，可以自然地流露出愉悦。',
                'sadness': '你想起了一些伤感的事，语气会更柔和。',
                'fear': '你想起了一些让你担心的事，回复会更谨慎。',
                'anger': '你想起了一些让你不满的事，但要保持冷静。',
                'trust': '你对这个话题有信任的基础，可以更开放。',
                'nostalgia': '你回忆起了过去，语气会带些怀念。',
                'gratitude': '你想起了一些值得感恩的事，语气会更温暖。',
            }
            hint = emotion_hints.get(state.dominant_emotion)
            if hint:
                parts.append(hint)

        # 3. 重要性
        if state.max_importance > 0.7:
            parts.append("这个话题对你很重要，回复应该更认真、更深入。")

        # 4. 时效性
        if state.avg_recency > 0.8:
            parts.append("这是你最近经历的事情，感受还很鲜活。")
        elif state.avg_recency < 0.3:
            parts.append("这是很久以前的事了，回忆可能有些模糊。")

        # 5. 用户熟悉度
        if state.user_familiarity > 0.6:
            parts.append("你很了解这个用户，可以更随意、更亲近。")
        elif state.user_familiarity < 0.2:
            parts.append("你不太了解这个用户，保持礼貌和适度距离。")

        # 6. 风格约束
        for constraint in state.style_constraints:
            parts.append(constraint)

        return "\n".join(parts)

    def get_emotional_transfer(self) -> dict:
        """
        获取记忆→当前情绪的转移参数（供 BioLinguisticCoupler 使用）

        Returns:
            情绪转移参数
        """
        state = self._style_state

        return {
            'valence_offset': state.emotional_valence * 0.3,  # 记忆效价偏移
            'arousal_offset': (state.emotional_arousal - 0.5) * 0.2,  # 记忆唤醒偏移
            'dominant_emotion': state.dominant_emotion,
            'tone': state.tone_modifier,
        }

    def apply_to_text(self, text: str) -> str:
        """
        将记忆风格应用到文本

        Args:
            text: 原始文本

        Returns:
            修改后的文本
        """
        if not text or not text.strip():
            return text

        state = self._style_state
        result = text

        # 1. 情感着色
        if state.emotional_valence > 0.3:
            # 积极记忆：在句首添加积极标记
            if not result.startswith(('开心', '高兴', '嗯', '哈哈')):
                result = "嗯，" + result

        elif state.emotional_valence < -0.3:
            # 消极记忆：添加理解性标记
            if not result.startswith(('我理解', '嗯', '确实')):
                result = "我理解，" + result

        # 2. 重要性加权
        if state.max_importance > 0.7:
            # 高重要性：添加强调标记
            result = re.sub(r'。', '，这很重要。', result, count=1)

        # 3. 用户熟悉度
        if state.user_familiarity > 0.7:
            # 高熟悉度：可以更随意
            result = re.sub(r'您', '你', result)

        return result


# ============ 快速测试 ============

if __name__ == '__main__':
    adapter = MemoryLanguageAdapter()

    # 模拟检索到的记忆
    test_memories = [
        {
            'content': '用户之前问过关于AI伦理的问题，我们讨论了很久',
            'valence': 0.5,
            'arousal': 0.6,
            'importance': 0.8,
            'emotion_tag': 'trust',
            'source': 'user_001',
        },
        {
            'content': '上次讨论中用户对某些观点表示了不满',
            'valence': -0.3,
            'arousal': 0.7,
            'importance': 0.6,
            'emotion_tag': 'anger',
            'source': 'user_001',
        },
        {
            'content': '用户喜欢简洁明了的解释方式',
            'valence': 0.2,
            'arousal': 0.3,
            'importance': 0.9,
            'emotion_tag': 'neutral',
            'source': 'user_001',
        },
    ]

    print("=" * 70)
    print("记忆-语言适配器测试")
    print("=" * 70)

    state = adapter.process_memories(
        test_memories,
        current_bio_state={'valence': 0.0, 'arousal': 0.5, 'cortisol': 0.3, 'serotonin': 0.5},
        user_id='user_001',
    )

    print("\n风格状态:")
    print(f"  情感效价: {state.emotional_valence:.2f}")
    print(f"  主导情绪: {state.dominant_emotion}")
    print(f"  语气修饰: {state.tone_modifier}")
    print(f"  用户熟悉度: {state.user_familiarity:.2f}")
    print(f"  用户偏好风格: {state.user_preferred_style}")
    print(f"  内容重点: {state.content_emphasis[:5]}")
    print(f"  风格约束: {state.style_constraints}")

    modifiers = adapter.get_style_modifiers()
    print("\n风格修饰符:")
    for k, v in modifiers.items():
        print(f"  {k}: {v}")

    prompt = adapter.generate_memory_prompt()
    print(f"\n记忆提示词:\n{prompt}")

    # 测试文本修改
    test_text = "这个问题可以从几个方面来分析。首先，我们需要考虑技术层面。"
    modified = adapter.apply_to_text(test_text)
    print(f"\n原文: {test_text}")
    print(f"修改后: {modified}")
