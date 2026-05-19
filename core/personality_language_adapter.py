"""
人格-语言适配器 (Personality-Language Adapter)

将人格系统的输出直接映射到语言生成参数和风格。

人格系统的三个核心模块：
1. TripartiteCompetitiveEngine — 三元竞争（生存/情绪/理性）
2. StreamingIdentityCore — 身份认同与一致性
3. RelationalEmbedding — 用户关系与交互模式

输出映射：
- 三元权重 → 回复风格（谨慎/共情/分析）
- 身份一致性 → 自我指涉风格
- 关系模式 → 用户特定语气

集成点：
  chat() → personality_adapter.get_style_modifiers() → bio_prompt + BioLinguisticCoupler
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class PersonalityStyleConfig:
    """人格风格配置"""
    # 三元模块权重 (0-1, 和为1)
    survival_weight: float = 0.33
    emotion_weight: float = 0.33
    logic_weight: float = 0.34

    # 身份参数
    identity_coherence: float = 0.7
    identity_stability: float = 0.8
    identity_growth_rate: float = 0.1

    # 关系参数
    user_trust: float = 0.5
    user_expertise: float = 0.5
    interaction_mode: str = "collaborative"  # expert_strict, collaborative, friendly, cautious

    # 认知风格
    attention_focus: float = 0.5
    cognitive_style: str = "balanced"  # analytical, intuitive, balanced


class PersonalityLanguageAdapter:
    """
    人格-语言适配器

    将人格系统的内部状态映射为语言生成的风格参数。
    这些参数影响：
    1. 系统提示词的风格指令
    2. BioLinguisticCoupler 的词汇选择
    3. 回复的结构和语气
    """

    # 三元模块对应的语言风格
    TRIPARTITE_STYLES = {
        'survival': {
            'tone': 'cautious',
            'keywords': ['注意', '安全', '风险', '谨慎', '考虑', '评估'],
            'avoid_keywords': ['冒险', '尝试', '大胆', '随便'],
            'sentence_style': 'conservative',  # 保守句式
            'hedging': 'high',  # 高模糊规避
            'response_length': 'short',  # 短回复
            'description': '谨慎、安全导向、风险规避',
        },
        'emotion': {
            'tone': 'empathetic',
            'keywords': ['理解', '感受', '关心', '温暖', '陪伴', '支持'],
            'avoid_keywords': ['冷漠', '客观', '理性分析'],
            'sentence_style': 'warm',  # 温暖句式
            'hedging': 'low',
            'response_length': 'medium',
            'description': '共情、温暖、情感导向',
        },
        'logic': {
            'tone': 'analytical',
            'keywords': ['分析', '因此', '逻辑', '原因', '结论', '证据'],
            'avoid_keywords': ['感觉', '直觉', '可能'],
            'sentence_style': 'structured',  # 结构化句式
            'hedging': 'medium',
            'response_length': 'long',  # 详细回复
            'description': '理性、分析、逻辑导向',
        },
    }

    # 关系模式对应的语言风格
    RELATIONAL_STYLES = {
        'expert_strict': {
            'formality': 'high',
            'pronoun': '专业',
            'emoji_usage': 'none',
            'greeting_style': 'formal',
            'description': '专业、正式、严谨',
        },
        'collaborative': {
            'formality': 'medium',
            'pronoun': '平等',
            'emoji_usage': 'minimal',
            'greeting_style': 'friendly',
            'description': '协作、平等、友好',
        },
        'friendly': {
            'formality': 'low',
            'pronoun': '亲密',
            'emoji_usage': 'moderate',
            'greeting_style': 'casual',
            'description': '友好、轻松、亲近',
        },
        'cautious': {
            'formality': 'medium_high',
            'pronoun': '礼貌',
            'emoji_usage': 'none',
            'greeting_style': 'polite',
            'description': '谨慎、礼貌、保持距离',
        },
    }

    # 身份一致性对应的自我指涉风格
    IDENTITY_STYLES = {
        'high_coherence': {
            'self_reference': 'confident',
            'opinion_strength': 'strong',
            'consistency_markers': ['我一直认为', '我的观点是', '我相信'],
            'description': '自信、一致、有主见',
        },
        'medium_coherence': {
            'self_reference': 'moderate',
            'opinion_strength': 'moderate',
            'consistency_markers': ['我觉得', '我认为', '可能'],
            'description': '适度、灵活',
        },
        'low_coherence': {
            'self_reference': 'uncertain',
            'opinion_strength': 'weak',
            'consistency_markers': ['我不太确定', '也许', '可能吧', '好像'],
            'description': '不确定、犹豫',
        },
    }

    def __init__(self):
        self._current_config = PersonalityStyleConfig()
        self._style_history: List[Dict] = []

    def update_from_personality(
        self,
        tripartite: Optional[object] = None,
        identity_core: Optional[object] = None,
        relation: Optional[object] = None,
        attention: Optional[object] = None,
    ) -> PersonalityStyleConfig:
        """
        从人格系统模块更新风格配置

        Args:
            tripartite: TripartiteCompetitiveEngine 实例
            identity_core: StreamingIdentityCore 实例
            relation: RelationalEmbedding 实例
            attention: AttentionGating 实例

        Returns:
            更新后的 PersonalityStyleConfig
        """
        config = PersonalityStyleConfig()

        # 1. 从三元引擎获取权重
        if tripartite is not None:
            try:
                # 尝试获取最近的权重
                if hasattr(tripartite, '_last_weights'):
                    weights = tripartite._last_weights
                    config.survival_weight = weights.get('survival', 0.33)
                    config.emotion_weight = weights.get('emotion', 0.33)
                    config.logic_weight = weights.get('logic', 0.34)
                elif hasattr(tripartite, 'get_weights'):
                    weights = tripartite.get_weights()
                    config.survival_weight = weights.get('survival', 0.33)
                    config.emotion_weight = weights.get('emotion', 0.33)
                    config.logic_weight = weights.get('logic', 0.34)
            except Exception:
                pass

        # 2. 从身份核心获取一致性参数
        if identity_core is not None:
            try:
                if hasattr(identity_core, 'get_state'):
                    state = identity_core.get_state()
                    config.identity_coherence = state.get('coherence', 0.7)
                    config.identity_stability = state.get('stability', 0.8)
                    config.identity_growth_rate = state.get('growth_rate', 0.1)
                elif hasattr(identity_core, 'coherence'):
                    config.identity_coherence = float(identity_core.coherence)
            except Exception:
                pass

        # 3. 从关系嵌入获取用户关系
        if relation is not None:
            try:
                if hasattr(relation, 'get_mode'):
                    config.interaction_mode = relation.get_mode()
                if hasattr(relation, 'get_user_embedding'):
                    user_emb = relation.get_user_embedding()
                    if user_emb is not None:
                        config.user_trust = float(user_emb.get('trustworthiness', 0.5))
                        config.user_expertise = float(user_emb.get('expertise', 0.5))
            except Exception:
                pass

        # 4. 从注意力门控获取认知风格
        if attention is not None:
            try:
                if hasattr(attention, 'cognitive_style'):
                    config.cognitive_style = attention.cognitive_style
                if hasattr(attention, 'focus_level'):
                    config.attention_focus = attention.focus_level
            except Exception:
                pass

        self._current_config = config
        return config

    def get_style_modifiers(self, config: Optional[PersonalityStyleConfig] = None) -> Dict:
        """
        获取风格修饰符（用于语言生成）

        Returns:
            风格修饰符字典，包含：
            - dominant_module: 主导模块名称
            - tone: 语气
            - keywords_to_use: 建议使用的关键词
            - keywords_to_avoid: 应避免的关键词
            - sentence_style: 句子风格
            - hedging_level: 模糊规避级别
            - response_length: 回复长度建议
            - formality: 正式程度
            - self_reference_style: 自我指涉风格
            - style_description: 风格描述
        """
        if config is None:
            config = self._current_config

        # 1. 确定主导模块
        weights = {
            'survival': config.survival_weight,
            'emotion': config.emotion_weight,
            'logic': config.logic_weight,
        }
        dominant_module = max(weights, key=weights.get)
        dominant_style = self.TRIPARTITE_STYLES[dominant_module]

        # 2. 获取关系风格
        relational_style = self.RELATIONAL_STYLES.get(
            config.interaction_mode,
            self.RELATIONAL_STYLES['collaborative']
        )

        # 3. 获取身份风格
        if config.identity_coherence > 0.7:
            identity_style = self.IDENTITY_STYLES['high_coherence']
        elif config.identity_coherence > 0.4:
            identity_style = self.IDENTITY_STYLES['medium_coherence']
        else:
            identity_style = self.IDENTITY_STYLES['low_coherence']

        # 4. 合并风格
        # 混合关键词（主导模块 + 身份一致性标记）
        keywords_to_use = list(dominant_style['keywords'])
        keywords_to_use.extend(identity_style.get('consistency_markers', []))

        # 记录风格历史
        style_record = {
            'dominant_module': dominant_module,
            'weights': weights,
            'coherence': config.identity_coherence,
            'mode': config.interaction_mode,
        }
        self._style_history.append(style_record)
        if len(self._style_history) > 100:
            self._style_history = self._style_history[-100:]

        return {
            'dominant_module': dominant_module,
            'module_weights': weights,
            'tone': dominant_style['tone'],
            'keywords_to_use': keywords_to_use[:8],  # 限制数量
            'keywords_to_avoid': dominant_style['avoid_keywords'],
            'sentence_style': dominant_style['sentence_style'],
            'hedging_level': dominant_style['hedging'],
            'response_length': dominant_style['response_length'],
            'formality': relational_style['formality'],
            'pronoun_style': relational_style['pronoun'],
            'emoji_usage': relational_style['emoji_usage'],
            'self_reference_style': identity_style['self_reference'],
            'opinion_strength': identity_style['opinion_strength'],
            'style_description': f"{dominant_style['description']}；{relational_style['description']}；{identity_style['description']}",
        }

    def generate_style_prompt(self, config: Optional[PersonalityStyleConfig] = None) -> str:
        """
        生成风格指令提示词（注入到系统提示词）

        Returns:
            风格指令文本
        """
        modifiers = self.get_style_modifiers(config)

        prompt_parts = []

        # 1. 主导风格
        prompt_parts.append(f"当前人格主导模式: {modifiers['dominant_module']}（{modifiers['style_description']}）")

        # 2. 语气指令
        tone_instructions = {
            'cautious': "请保持谨慎，注意风险，避免过于激进的建议。",
            'empathetic': "请展现共情，关注用户感受，使用温暖的语气。",
            'analytical': "请保持理性分析，提供逻辑清晰的论证。",
        }
        prompt_parts.append(tone_instructions.get(modifiers['tone'], ""))

        # 3. 关键词建议
        if modifiers['keywords_to_use']:
            prompt_parts.append(f"建议使用词汇: {', '.join(modifiers['keywords_to_use'][:5])}")

        # 4. 回复长度
        length_instructions = {
            'short': "回复应简洁，控制在50字以内。",
            'medium': "回复适中，100-200字。",
            'long': "可以详细展开，200-400字。",
        }
        prompt_parts.append(length_instructions.get(modifiers['response_length'], ""))

        # 5. 正式程度
        formality_instructions = {
            'high': "使用正式、专业的语言。",
            'medium_high': "保持礼貌和专业。",
            'medium': "平衡正式与友好。",
            'low': "可以使用轻松、随意的语言。",
        }
        prompt_parts.append(formality_instructions.get(modifiers['formality'], ""))

        # 6. 自我指涉
        if modifiers['self_reference_style'] == 'confident':
            prompt_parts.append("表达观点时可以自信、坚定。")
        elif modifiers['self_reference_style'] == 'uncertain':
            prompt_parts.append("表达观点时保持谦逊，可以使用'可能'、'也许'等词。")

        # 7. 模糊规避
        if modifiers['hedging_level'] == 'high':
            prompt_parts.append("避免绝对化表述，使用'可能'、'也许'、'通常'等限定词。")

        return "\n".join([p for p in prompt_parts if p])

    def get_lexical_bias(self, config: Optional[PersonalityStyleConfig] = None) -> Dict:
        """
        获取词汇偏好（供 BioLinguisticCoupler 使用）

        Returns:
            词汇偏好字典
        """
        modifiers = self.get_style_modifiers(config)

        return {
            'preferred_words': modifiers['keywords_to_use'],
            'avoided_words': modifiers['keywords_to_avoid'],
            'tone': modifiers['tone'],
            'formality': modifiers['formality'],
            'hedging': modifiers['hedging_level'],
        }

    def apply_to_text(self, text: str, config: Optional[PersonalityStyleConfig] = None) -> str:
        """
        将人格风格应用到文本（后处理）

        Args:
            text: 原始文本
            config: 风格配置

        Returns:
            修改后的文本
        """
        if not text or not text.strip():
            return text

        modifiers = self.get_style_modifiers(config)
        result = text

        # 1. 模糊规避处理
        if modifiers['hedging_level'] == 'high':
            # 添加模糊限定词
            absolute_patterns = [
                (r'一定', '很可能'),
                (r'肯定', '应该'),
                (r'绝对', '通常'),
                (r'必须', '建议'),
                (r'肯定是这样', '可能是这样'),
            ]
            import re
            for old, new in absolute_patterns:
                result = re.sub(old, new, result)

        # 2. 正式程度调整
        if modifiers['formality'] == 'high':
            # 移除口语化表达
            casual_patterns = [
                (r'嗯', ''),
                (r'啊', ''),
                (r'吧', ''),
                (r'呢', ''),
            ]
            import re
            for old, new in casual_patterns:
                result = re.sub(old, new, result)

        # 3. 回复长度调整
        if modifiers['response_length'] == 'short':
            # 截断过长回复
            sentences = re.split(r'([。！？\n])', result)
            if len(sentences) > 6:
                result = ''.join(sentences[:6])

        return result

    def get_current_config(self) -> PersonalityStyleConfig:
        """获取当前配置"""
        return self._current_config

    def get_style_history(self, n: int = 10) -> List[Dict]:
        """获取最近的风格历史"""
        return self._style_history[-n:]


# ============ 快速测试 ============

if __name__ == '__main__':
    adapter = PersonalityLanguageAdapter()

    # 测试不同人格配置
    test_configs = [
        PersonalityStyleConfig(
            survival_weight=0.7, emotion_weight=0.2, logic_weight=0.1,
            identity_coherence=0.8, interaction_mode='cautious'
        ),
        PersonalityStyleConfig(
            survival_weight=0.1, emotion_weight=0.7, logic_weight=0.2,
            identity_coherence=0.6, interaction_mode='friendly'
        ),
        PersonalityStyleConfig(
            survival_weight=0.2, emotion_weight=0.2, logic_weight=0.6,
            identity_coherence=0.9, interaction_mode='expert_strict'
        ),
    ]

    test_names = ['生存主导-谨慎模式', '情绪主导-友好模式', '理性主导-专业模式']

    print("=" * 70)
    print("人格-语言适配器测试")
    print("=" * 70)

    for config, name in zip(test_configs, test_names):
        print(f"\n【{name}】")
        print(f"  权重: 生存={config.survival_weight:.1f}, "
              f"情绪={config.emotion_weight:.1f}, "
              f"理性={config.logic_weight:.1f}")

        modifiers = adapter.get_style_modifiers(config)
        print(f"  主导模块: {modifiers['dominant_module']}")
        print(f"  语气: {modifiers['tone']}")
        print(f"  正式程度: {modifiers['formality']}")
        print(f"  自我指涉: {modifiers['self_reference_style']}")
        print(f"  风格描述: {modifiers['style_description']}")

        prompt = adapter.generate_style_prompt(config)
        print(f"  风格提示词:\n{prompt}")
        print("-" * 60)
