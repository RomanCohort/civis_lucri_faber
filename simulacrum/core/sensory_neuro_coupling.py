"""
感觉-神经递质耦合系统 (Sensory-Neurotransmitter Coupling)

将感觉输入映射到神经递质和激素系统的释放。

生物基础:
- 视觉威胁 → 肾上腺素/去甲肾上腺素 (战斗或逃跑)
- 愉快的声音 → 多巴胺 (奖励)
- 语言负面效价 → 血清素减少 (情绪)
- 新颖刺激 → 乙酰胆碱 (注意力)

功能:
- 从视觉、听觉、语言感觉流计算显著性
- 应用耦合矩阵计算神经递质释放
- 实现神经化学水平的衰减动力学
- 通过事件总线与神经递质/激素系统通信
"""

from dataclasses import dataclass

import numpy as np

try:
    from core.event_bus import EventBus
    from core.events import SENSORY_PROCESS
except ImportError:
    from event_bus import EventBus
    from events import SENSORY_PROCESS


@dataclass
class NeurochemicalState:
    """神经化学状态快照"""
    sensory_adrenaline: float = 0.0  # 感觉驱动的肾上腺素释放
    sensory_dopamine: float = 0.0    # 感觉驱动的多巴胺释放
    sensory_cortisol: float = 0.0    # 感觉驱动的皮质醇释放
    sensory_acetylcholine: float = 0.0  # 感觉驱动的乙酰胆碱释放


class SensoryNeuroCoupling:
    """
    感觉-神经递质耦合

    桥接感觉输入事件和神经递质/激素系统。

    耦合矩阵定义了每种感觉模态如何驱动每种神经化学物质的释放。
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        # 订阅 SENSORY_PROCESS 事件
        self.event_bus.subscribe(
            SENSORY_PROCESS,
            self._on_sensory_process,
            priority=3,  # 在 limbic/language 处理之后
            name="sensory_neuro_coupling"
        )

        # 耦合矩阵: 感觉模态 → 神经化学物质强度
        # 值表示该模态驱动该神经化学物质释放的强度 [0, 1]
        self.coupling_matrix = {
            # 视觉感觉: 威胁检测 → 肾上腺素/皮质醇, 奖励刺激 → 多巴胺
            'visual': {
                'sensory_adrenaline': 0.7,    # 高: 视觉威胁 → 战斗或逃跑
                'sensory_dopamine': 0.1,      # 低: 视觉奖励 → 多巴胺
                'sensory_cortisol': 0.5,       # 中: 视觉压力 → 皮质醇
                'sensory_acetylcholine': 0.4,  # 中: 视觉注意 → 乙酰胆碱
            },
            # 听觉感觉: 愉快声音 → 多巴胺, 不愉快声音 → 皮质醇
            'auditory': {
                'sensory_adrenaline': 0.2,     # 低: 听觉威胁 → 肾上腺素
                'sensory_dopamine': 0.6,       # 高: 愉快声音 → 多巴胺奖励
                'sensory_cortisol': 0.1,       # 低: 听觉压力
                'sensory_acetylcholine': 0.5,  # 中: 听觉注意
            },
            # 语言感觉: 负面语言 → 皮质醇, 正面语言 → 多巴胺
            'language': {
                'sensory_adrenaline': 0.3,     # 中: 语言压力 → 肾上腺素
                'sensory_dopamine': 0.4,       # 中: 正面语言 → 多巴胺
                'sensory_cortisol': 0.6,       # 高: 负面语言 → 皮质醇
                'sensory_acetylcholine': 0.3,  # 低: 语言理解 → 乙酰胆碱
            },
        }

        # 衰减率: 每步衰减，模拟神经化学物质的代谢清除
        # 皮质醇衰减慢 (0.95)，其他较快
        self.decay_rates = {
            'sensory_adrenaline': 0.85,      # 快速衰减
            'sensory_dopamine': 0.90,         # 中速衰减
            'sensory_cortisol': 0.95,         # 慢速衰减 (应激激素)
            'sensory_acetylcholine': 0.88,    # 中速衰减
        }

        # 当前神经化学水平
        self.levels = NeurochemicalState()

        # 上一帧的感觉输入 (用于新颖性检测)
        self._previous_sensory_input: dict[str, float] | None = None

    def _on_sensory_process(self, event) -> dict:
        """处理 SENSORY_PROCESS 事件并计算神经化学释放。

        从各感觉模态提取 VAD 值，计算显著性，然后应用耦合矩阵。
        """
        data = event.data
        internal_state = data.get('internal_state', {})

        # 提取各模态的 VAD
        visual_vad = self._extract_visual_vad(internal_state)
        auditory_vad = self._extract_auditory_vad(internal_state)
        language_vad = self._extract_language_vad(internal_state)

        # 计算每种模态的显著性
        visual_salience = self._compute_salience(visual_vad)
        auditory_salience = self._compute_salience(auditory_vad)
        language_salience = self._compute_salience(language_vad)

        saliences = {
            'visual': visual_salience,
            'auditory': auditory_salience,
            'language': language_salience,
        }

        # 计算新颖性因子 (新颖刺激产生更强的神经化学释放)
        novelty_factor = self._compute_novelty(saliences)

        # 应用耦合矩阵: 对每个模态的每个神经化学物质进行计算
        for modality, salience in saliences.items():
            coupling = self.coupling_matrix[modality]
            for neurochem, strength in coupling.items():
                # 释放 = 显著性 × 耦合强度 × 0.1 (缩放因子)
                release = salience * strength * novelty_factor * 0.1

                # 更新对应水平
                if neurochem == 'sensory_adrenaline':
                    self.levels.sensory_adrenaline = min(
                        1.0, self.levels.sensory_adrenaline + release
                    )
                elif neurochem == 'sensory_dopamine':
                    self.levels.sensory_dopamine = min(
                        1.0, self.levels.sensory_dopamine + release
                    )
                elif neurochem == 'sensory_cortisol':
                    self.levels.sensory_cortisol = min(
                        1.0, self.levels.sensory_cortisol + release
                    )
                elif neurochem == 'sensory_acetylcholine':
                    self.levels.sensory_acetylcholine = min(
                        1.0, self.levels.sensory_acetylcholine + release
                    )

        # 应用衰减
        self._apply_decay()

        # 写入到 internal_state (供 _build_state_vector 使用)
        internal_state['sensory_neurochemical'] = {
            'sensory_adrenaline': self.levels.sensory_adrenaline,
            'sensory_dopamine': self.levels.sensory_dopamine,
            'sensory_cortisol': self.levels.sensory_cortisol,
            'sensory_acetylcholine': self.levels.sensory_acetylcholine,
        }

        return {
            'sensory_neuro_levels': {
                'adrenaline': self.levels.sensory_adrenaline,
                'dopamine': self.levels.sensory_dopamine,
                'cortisol': self.levels.sensory_cortisol,
                'acetylcholine': self.levels.sensory_acetylcholine,
            },
            'saliences': saliences,
            'novelty_factor': novelty_factor,
        }

    def _extract_visual_vad(self, internal_state: dict) -> dict:
        """从 internal_state 提取视觉 VAD。

        优先级:
        1. censor_vad (微表情驱动的 VAD)
        2. limbic_valence/arousal (边缘系统驱动的情绪)
        3. 默认值
        """
        # 尝试从 Censor 微表情获取视觉情绪
        if 'censor_happiness' in internal_state or 'censor_anger' in internal_state:
            # 微表情驱动的视觉情绪
            happiness = internal_state.get('censor_happiness', 0.0)
            sadness = internal_state.get('censor_sadness', 0.0)
            anger = internal_state.get('censor_anger', 0.0)
            fear = internal_state.get('censor_fear', 0.0)
            surprise = internal_state.get('censor_surprise', 0.0)

            # 效价: 快乐增加效价，悲伤/愤怒/恐惧降低效价
            valence = (happiness - anger - fear) * 0.5 + 0.5

            # 唤醒度: 恐惧/惊讶增加唤醒度
            arousal = (fear + surprise + anger) * 0.5

            return {
                'valence': valence,
                'arousal': arousal,
                'dominance': 0.5,
                'pleasantness': happiness,
            }

        # 尝试从边缘系统获取
        limbic_valence = internal_state.get('limbic_valence', 0.0)
        limbic_arousal = internal_state.get('limbic_arousal', 0.5)

        return {
            'valence': (limbic_valence + 1.0) / 2.0,  # [-1,1] → [0,1]
            'arousal': limbic_arousal,
            'dominance': 0.5,
            'pleasantness': 0.5,
        }

    def _extract_auditory_vad(self, internal_state: dict) -> dict:
        """从 internal_state 提取听觉 VAD。

        优先级:
        1. auditory_vad (听觉皮层直接输出)
        2. limbic_valence/arousal (边缘系统情绪)
        3. 默认值
        """
        auditory_vad = internal_state.get('auditory_vad', {})

        if auditory_vad:
            return {
                'valence': auditory_vad.get('valence', 0.5),
                'arousal': auditory_vad.get('arousal', 0.0),
                'dominance': auditory_vad.get('dominance', 0.5),
                'pleasantness': auditory_vad.get('pleasantness', 0.5),
            }

        # 回退到边缘系统
        limbic_valence = internal_state.get('limbic_valence', 0.0)
        limbic_arousal = internal_state.get('limbic_arousal', 0.5)

        return {
            'valence': (limbic_valence + 1.0) / 2.0,
            'arousal': limbic_arousal,
            'dominance': 0.5,
            'pleasantness': 0.5,
        }

    def _extract_language_vad(self, internal_state: dict) -> dict:
        """从 internal_state 提取语言 VAD。

        优先级:
        1. language_semantic (语言语义向量)
        2. language_valence/arousal (语言皮层输出)
        3. mood_valence/arousal (情绪系统)
        4. 默认值
        """
        language_semantic = internal_state.get('language_semantic', {})

        if language_semantic:
            return {
                'valence': language_semantic.get('language_valence', 0.5),
                'arousal': language_semantic.get('language_arousal', 0.0),
                'dominance': language_semantic.get('language_dominance', 0.5),
                'pleasantness': language_semantic.get('language_valence', 0.5),
            }

        # 尝试从语言皮层获取
        language_valence = internal_state.get('language_valence')
        language_arousal = internal_state.get('language_arousal')

        if language_valence is not None:
            return {
                'valence': language_valence,
                'arousal': language_arousal if language_arousal is not None else 0.0,
                'dominance': 0.5,
                'pleasantness': language_valence,
            }

        # 回退到情绪系统
        mood_valence = internal_state.get('mood_valence', 0.5)
        mood_arousal = internal_state.get('mood_arousal', 0.5)

        return {
            'valence': mood_valence,
            'arousal': mood_arousal,
            'dominance': 0.5,
            'pleasantness': mood_valence,
        }

    def _compute_salience(self, vad: dict) -> float:
        """从 VAD 计算显著性。

        显著性由唤醒度和效价偏差决定:
        - 高唤醒 + 极端效价 = 高显著性
        - 中性效价 + 低唤醒 = 低显著性
        """
        arousal = vad.get('arousal', 0.0)
        valence = vad.get('valence', 0.5)

        # 效价偏差: 0.5 是中性，0 或 1 是极端
        valence_deviation = abs(valence - 0.5) * 2  # [0, 1] 范围

        # 显著性 = 唤醒度 * 0.6 + 效价偏差 * 0.3 + 支配性 * 0.1
        dominance = vad.get('dominance', 0.5)
        salience = arousal * 0.6 + valence_deviation * 0.3 + dominance * 0.1

        return np.clip(salience, 0, 1)

    def _compute_novelty(self, current_saliences: dict[str, float]) -> float:
        """计算新颖性因子。

        如果感觉输入与上一帧显著不同，产生更强的神经化学释放。
        """
        if self._previous_sensory_input is None:
            self._previous_sensory_input = current_saliences.copy()
            return 1.0  # 第一帧不施加新颖性加成

        # 计算与上一帧的总差异
        total_delta = 0.0
        for modality, salience in current_saliences.items():
            prev = self._previous_sensory_input.get(modality, 0.0)
            total_delta += abs(salience - prev)

        # 平均差异
        avg_delta = total_delta / len(current_saliences)

        # 新颖性阈值: 超过 0.2 认为是新颖刺激
        if avg_delta > 0.2:
            novelty_factor = 1.5  # 新颖刺激增强 50%
        else:
            novelty_factor = 1.0

        # 更新历史记录
        self._previous_sensory_input = current_saliences.copy()

        return novelty_factor

    def _apply_decay(self) -> None:
        """应用衰减到神经化学水平。

        模拟神经化学物质的代谢清除和重摄取。
        """
        self.levels.sensory_adrenaline *= self.decay_rates['sensory_adrenaline']
        self.levels.sensory_dopamine *= self.decay_rates['sensory_dopamine']
        self.levels.sensory_cortisol *= self.decay_rates['sensory_cortisol']
        self.levels.sensory_acetylcholine *= self.decay_rates['sensory_acetylcholine']

    def get_state(self) -> NeurochemicalState:
        """获取当前神经化学状态。"""
        return self.levels

    def reset(self) -> None:
        """重置所有神经化学水平。"""
        self.levels = NeurochemicalState()
        self._previous_sensory_input = None


def create_sensory_neuro_coupling(event_bus: EventBus) -> SensoryNeuroCoupling:
    """工厂函数: 创建感觉-神经递质耦合系统。"""
    return SensoryNeuroCoupling(event_bus)
