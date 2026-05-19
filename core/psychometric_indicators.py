"""心理测量指标系统 — 从 _internal_state 提取临床量表模拟评分。

将agent内部状态映射为标准化临床量表的计算模拟版本:
  - PHQ-9 模拟 (抑郁严重度, 0-10)
  - GAD-7 模拟 (焦虑水平, 0-10)
  - MoCA 模拟 (认知功能, 0-10)
  - DERS 模拟 (情绪调节困难, 0-10)
  - 社会功能量表模拟 (0-10)

每个量表由多个 _internal_state 键加权求和计算，权重参考
临床量表的条目与脑区/神经递质的对应关系。

参考文献:
  - Kroenke et al. (2001) PHQ-9: J Gen Intern Med 16:606-613
  - Spitzer et al. (2006) GAD-7: Arch Intern Med 166:1092-1097
  - Nasreddine et al. (2005) MoCA: JAGS 53:695-699
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PsychometricSnapshot:
    """一个时间点的心理测量指标快照。"""
    step: int
    time_h: float

    # 临床量表模拟 (0-10, 10=最严重/最失调)
    depression_severity: float    # PHQ-9 analog
    anxiety_level: float          # GAD-7 analog
    cognitive_function: float     # MoCA analog (反转: 10=正常, 0=严重损伤)
    emotional_regulation: float   # DERS analog (0=良好调节, 10=严重失调)
    social_functioning: float     # 社会功能 (10=正常, 0=严重缺损)

    # 脑区指标
    pfc_maturity: float
    limbic_valence: float
    limbic_arousal: float
    hpa_reactivity: float

    # 药物指标
    pkpd_effect: float
    drug_conc: float

    # 疗法指标
    therapy_skill: float
    synergy_factor: float

    # 综合严重度
    global_symptom_severity: float  # 0-1 综合


# ── 量表条目映射 ──
# 格式: {state_key: (weight, center)}
# weight: 正=该值越高→量表分越高(更严重), 负=该值越高→量表分越低(更轻微)
# center: 正常基线值

_PHQ9_ITEMS: Dict[str, tuple] = {
    # 快乐感缺失 → DA低
    "nt_dopamine":                             (-1.5, 0.5),
    # 心境低落 → 5-HT低
    "nt_serotonin":                            (-1.8, 0.5),
    # 情绪效价 → 负效价=抑郁
    "limbic_valence":                          (-1.2, 0.0),
    # 精力下降 → BDNF低
    "plasticity_bdnf":                         (-1.0, 0.5),
    # 认知功能 → PFC低
    "prefrontal_maturity":                     (-1.0, 0.5),
    # 躯体症状 → Cortisol高
    "cortisol_level":                          (1.5, 0.3),
    # 情绪调节能力
    "emotion_regulation_regulation_capacity":  (-1.2, 0.5),
    # 自我一致感
    "self_awareness_coherence":                (-0.8, 0.5),
    # 社会兴趣
    "social_cognition_affective_empathy":      (-0.5, 0.5),
}

_GAD7_ITEMS: Dict[str, tuple] = {
    # 神经紧张 → 唤醒高
    "limbic_arousal":                          (2.0, 0.5),
    # NE高 → 焦虑
    "nt_norepinephrine":                       (2.0, 0.3),
    # HPA应激反应性高
    "hpa_axis_stress_reactivity_mult":         (1.5, 1.0),
    # Cortisol高
    "cortisol_level":                          (1.5, 0.3),
    # GABA低 → 抑制不足
    "nt_gaba":                                 (-1.5, 0.5),
    # PFC抑制低 → 无法抑制焦虑
    "prefrontal_inhibition":                   (-1.0, 0.5),
    # 迷走神经张力低
    "ans_hrv":                                 (-0.8, 0.5),
}

_MOCA_ITEMS: Dict[str, tuple] = {
    # PFC成熟度 → 执行功能
    "prefrontal_maturity":                     (1.5, 0.5),
    # ACh → 注意/记忆
    "nt_acetylcholine":                        (1.5, 0.5),
    # BDNF → 神经可塑性
    "plasticity_bdnf":                         (1.2, 0.5),
    # 海马编码 → 记忆
    "hippocampus_encoding_modulation":         (1.3, 0.5),
    # DA → 工作记忆/动机
    "nt_dopamine":                             (1.0, 0.5),
    # 觉醒度 → 意识水平
    "brainstem_arousal_setpoint":              (0.8, 0.5),
    # 自省深度 → 元认知
    "self_awareness_introspection_depth":      (0.7, 0.5),
}

_DERS_ITEMS: Dict[str, tuple] = {
    # 情绪调节能力 (反转: 高=好)
    "emotion_regulation_regulation_capacity":  (-1.5, 0.5),
    # PFC抑制 (反转: 高=好)
    "prefrontal_inhibition":                   (-1.0, 0.5),
    # 情绪波动性
    "mood_system_volatility_mult":             (1.5, 1.0),
    # 唤醒水平过高
    "limbic_arousal":                          (1.2, 0.5),
    # 自我一致感
    "self_awareness_coherence":                (-0.8, 0.5),
    # 效价波动
    "limbic_valence":                          (-0.5, 0.0),
}

_SOCIAL_ITEMS: Dict[str, tuple] = {
    # 情感共情
    "social_cognition_affective_empathy":      (1.5, 0.5),
    # 认知共情
    "social_cognition_cognitive_empathy":      (1.3, 0.5),
    # 催产素 → 社会联结
    "hormone_oxytocin":                        (1.2, 0.4),
    # 情绪感染 (适度)
    "social_cognition_contagion":              (0.5, 0.3),
    # PFC成熟 → 社会认知
    "prefrontal_maturity":                     (0.8, 0.5),
    # 5-HT → 社会支配/信心
    "nt_serotonin":                            (0.7, 0.5),
}


class PsychometricIndicatorTracker:
    """从 _internal_state 提取心理测量指标。"""

    def compute_snapshot(
        self,
        state: Dict[str, Any],
        step: int,
        time_h: float,
        pkpd_effect: float = 0.0,
        drug_conc: float = 0.0,
        therapy_skill: float = 0.0,
        synergy_factor: float = 0.0,
    ) -> PsychometricSnapshot:
        """计算所有心理测量指标。"""
        phq9 = self._compute_scale(state, _PHQ9_ITEMS, max_score=10.0)
        gad7 = self._compute_scale(state, _GAD7_ITEMS, max_score=10.0)
        cognitive = self._compute_scale(state, _MOCA_ITEMS, max_score=10.0)
        emotion_reg = self._compute_scale(state, _DERS_ITEMS, max_score=10.0)
        social = self._compute_scale(state, _SOCIAL_ITEMS, max_score=10.0)

        # 综合严重度: PHQ-9和GAD-7的加权平均 (归一化到0-1)
        global_severity = np.clip(
            (phq9 * 0.4 + gad7 * 0.3 +
             (10 - cognitive) * 0.15 + emotion_reg * 0.15) / 10.0,
            0.0, 1.0
        )

        return PsychometricSnapshot(
            step=step,
            time_h=time_h,
            depression_severity=float(np.clip(phq9, 0, 10)),
            anxiety_level=float(np.clip(gad7, 0, 10)),
            cognitive_function=float(np.clip(cognitive, 0, 10)),
            emotional_regulation=float(np.clip(emotion_reg, 0, 10)),
            social_functioning=float(np.clip(social, 0, 10)),
            pfc_maturity=float(state.get("prefrontal_maturity", 0.5)),
            limbic_valence=float(state.get("limbic_valence", 0.0)),
            limbic_arousal=float(state.get("limbic_arousal", 0.5)),
            hpa_reactivity=float(state.get("hpa_axis_stress_reactivity_mult", 1.0)),
            pkpd_effect=pkpd_effect,
            drug_conc=drug_conc,
            therapy_skill=therapy_skill,
            synergy_factor=synergy_factor,
            global_symptom_severity=float(global_severity),
        )

    def _compute_scale(
        self,
        state: Dict[str, Any],
        items: Dict[str, tuple],
        max_score: float = 10.0,
    ) -> float:
        """通用量表计算: 加权求和。

        对每个条目:
          contribution = weight × (value - center)
        总分 = sum(contributions), 归一化到 [0, max_score]
        """
        raw_score = 0.0
        total_weight = 0.0

        for state_key, (weight, center) in items.items():
            val = float(state.get(state_key, center))
            contribution = weight * (val - center)
            raw_score += contribution
            total_weight += abs(weight)

        if total_weight == 0:
            return max_score / 2  # 无法计算时返回中间值

        # 归一化: 每个权重贡献最多 max_score/N
        normalized = raw_score / total_weight * max_score / 2 + max_score / 2
        return float(np.clip(normalized, 0, max_score))

    def compute_baseline(self, state: Dict[str, Any]) -> Dict[str, float]:
        """计算基线指标 (疾病治疗前)。"""
        snap = self.compute_snapshot(state, step=0, time_h=0.0)
        return {
            "phq9_baseline": snap.depression_severity,
            "gad7_baseline": snap.anxiety_level,
            "cognitive_baseline": snap.cognitive_function,
            "emotion_reg_baseline": snap.emotional_regulation,
            "social_baseline": snap.social_functioning,
        }

    def classify_severity(self, phq9: float) -> str:
        """PHQ-9严重度分级。"""
        if phq9 <= 4:
            return "minimal"
        elif phq9 <= 9:
            return "mild"
        elif phq9 <= 14:
            return "moderate"
        elif phq9 <= 19:
            return "moderately_severe"
        else:
            return "severe"

    def classify_remission(self, phq9: float) -> str:
        """缓解状态判定。"""
        if phq9 < 5:
            return "full_remission"
        elif phq9 < 10:
            return "partial_remission"
        else:
            return "no_remission"


__all__ = [
    "PsychometricSnapshot",
    "PsychometricIndicatorTracker",
]