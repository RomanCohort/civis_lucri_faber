"""精神疾病与情绪状态模拟器

核心原理: 精神疾病不是新代码，而是现有 30+ 子系统的特定参数配置组合。
本模块是 "Profile Applicator" — 一次性跨子系统设置参数，
让现有动力学自然涌现病理行为。

生物对应:
    - 每种疾病对应一组跨脑区的参数偏移 (如同真实精神疾病的神经生物学基础)
    - 渐变 onset/offset 模拟疾病自然病程
    - 共病 = 多组偏移的叠加/竞争
    - 治疗响应 = 药物预设对抗疾病偏移

事件驱动:
    - 发布 PSYCHIATRIC_CONDITION_CHANGE: 条件变化时通知
    - 订阅 NEURAL_REGULATION: 在神经调节后推进渐变

可模拟疾病 (26种):
    MDD, Bipolar-Mania, Bipolar-Depression, GAD, Panic Disorder,
    Social Anxiety, PTSD, OCD, Schizophrenia-Positive, Schizophrenia-Negative,
    Autism Spectrum, ADHD, BPD, ASPD, NPD, Dissociative, Burnout,
    Cyclothymia, Dysthymia, Specific Phobia, Agoraphobia,
    Somatic Symptom, Anorexia Tendencies, Substance Use,
    Prolonged Grief, Delirium

可模拟情绪状态 (7种):
    Emotional Blunting, Emotional Lability, Alexithymia,
    Mixed Emotions, Contagion Hypersensitivity, Anhedonia,
    Emotional Dysregulation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.events import PSYCHIATRIC_CONDITION_CHANGE

# ===== 枚举与数据结构 =====

class OnsetMode(Enum):
    """发病模式"""
    INSTANT = "instant"         # 立即 (急性)
    RAPID = "rapid"             # 快速 (~5步)
    GRADUAL = "gradual"         # 渐进 (~50步)
    INSIDIOUS = "insidious"     # 隐袭 (~200步)


class OffsetMode(Enum):
    """恢复模式"""
    INSTANT = "instant"
    TREATMENT_RESPONSE = "treatment_response"   # 药物加速 (~20步)
    NATURAL_REMISSION = "natural_remission"      # 自然缓解 (~50步)
    SLOW_RECOVERY = "slow_recovery"              # 缓慢恢复 (~100步)


class Severity(Enum):
    """严重程度"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class ActiveCondition:
    """当前活跃的精神疾病状态"""
    condition_id: str
    severity: Severity
    onset_mode: OnsetMode
    offset_mode: OffsetMode
    progress: float = 0.0           # 0.0→1.0 onset, 1.0→0.0 offset
    is_onsetting: bool = True       # True=正在发病, False=正在恢复
    is_active: bool = True
    applied_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    baseline_snapshot: dict[str, dict[str, float]] = field(default_factory=dict)
    step_count: int = 0
    comorbid_with: list[str] = field(default_factory=list)


# ===== 渐变速度 =====

ONSET_SPEEDS = {
    OnsetMode.INSTANT: 1.0,
    OnsetMode.RAPID: 0.2,
    OnsetMode.GRADUAL: 0.02,
    OnsetMode.INSIDIOUS: 0.005,
}

OFFSET_SPEEDS = {
    OffsetMode.INSTANT: 1.0,
    OffsetMode.TREATMENT_RESPONSE: 0.05,
    OffsetMode.NATURAL_REMISSION: 0.02,
    OffsetMode.SLOW_RECOVERY: 0.01,
}

SEVERITY_MULTIPLIERS = {
    Severity.MILD: 0.3,
    Severity.MODERATE: 0.6,
    Severity.SEVERE: 1.0,
}


# ===== 26种精神疾病 Profile =====

PSYCHIATRIC_PROFILES = {
    # ──────────── 情感障碍 ────────────
    "MDD": {
        "name": "Major Depressive Disorder",
        "description": "重度抑郁症 — 持续低落、快感缺失、认知迟缓",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.15, "serotonin": 0.15, "bdnf": 0.2},
            "hpa_axis": {"stress_reactivity_mult": 1.5, "feedback_strength_mult": 0.5, "cortisol_bias": 0.3},
            "ans": {"baseline_vagal_tone": 0.2, "sympathetic_reactivity_mult": 1.3},
            "mood_system": {"mean_shift": [-0.5, -0.3, -0.4, -0.6, -0.5], "volatility_mult": 0.4},
            "self_awareness": {"agency": 0.25, "coherence": 0.3},
            "predictive_coding": {"precision_mult": 0.5, "free_energy_bias": 0.2},
            "glial": {"neuroinflammation": 0.5, "pruning_rate_mult": 1.5},
            "emotion_regulation": {"regulation_capacity": 0.2, "inhibition": 0.2},
            "social_cognition": {"affective_empathy": 0.3, "contagion": 0.3},
            "brainstem": {"arousal_setpoint": 0.3, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.1, "cortisol": 0.7},
            "scn": {"cortisol_peak_shift": -2, "melatonin_amplitude_mult": 0.6},
            "hippocampus": {"encoding_modulation": 0.2},
            "prefrontal": {"maturity": 0.2, "inhibition": 0.2},
            "limbic": {"valence": -0.5, "arousal": 0.2},
            "neuroplasticity": {"bdnf": 0.2},
            "interoception": {"gut_serotonin": 0.2, "inflammation": 0.5},
            "basal_ganglia": {"habit_strength": 0.3, "td_error_mult": 0.5},
        },
        "onset_default": "gradual",
        "offset_default": "slow_recovery",
        "comorbidities": ["GAD", "PTSD"],
        "treatment_responses": {"antidepressant": 0.7, "stimulant": 0.3, "anxiolytic": 0.2},
    },

    "bipolar_mania": {
        "name": "Bipolar Disorder — Manic Episode",
        "description": "双相障碍躁狂发作 — 情感高涨、思维奔逸、冲动行为",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.9, "serotonin": 0.3, "norepinephrine": 0.85, "bdnf": 0.8},
            "hpa_axis": {"stress_reactivity_mult": 0.5, "cortisol_bias": -0.2},
            "ans": {"baseline_vagal_tone": 0.2, "sympathetic_reactivity_mult": 2.0},
            "mood_system": {"mean_shift": [0.6, 0.7, 0.5, 0.8, 0.6], "volatility_mult": 3.0},
            "self_awareness": {"agency": 0.9, "coherence": 0.3, "self_boundary": 0.4},
            "predictive_coding": {"precision_mult": 2.0, "free_energy_bias": -0.3},
            "glial": {"neuroinflammation": 0.2, "pruning_rate_mult": 0.5},
            "emotion_regulation": {"regulation_capacity": 0.15, "inhibition": 0.1},
            "social_cognition": {"affective_empathy": 0.4, "contagion": 0.7},
            "brainstem": {"arousal_setpoint": 0.9, "default_defense": "fight"},
            "hormone": {"oxytocin": 0.5, "cortisol": 0.3, "testosterone": 0.7},
            "scn": {"cortisol_peak_shift": 3, "melatonin_amplitude_mult": 0.3},
            "hippocampus": {"encoding_modulation": 0.9},
            "prefrontal": {"maturity": 0.15, "inhibition": 0.1},
            "limbic": {"valence": 0.7, "arousal": 0.9},
            "neuroplasticity": {"bdnf": 0.8, "ltp_rate_mult": 1.5},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.1},
            "basal_ganglia": {"habit_strength": 0.8, "td_error_mult": 2.0},
        },
        "onset_default": "rapid",
        "offset_default": "treatment_response",
        "comorbidities": ["bipolar_depression", "ASPD"],
        "treatment_responses": {"mood_stabilizer": 0.8, "antipsychotic": 0.7, "anxiolytic": 0.3},
    },

    "bipolar_depression": {
        "name": "Bipolar Disorder — Depressive Episode",
        "description": "双相障碍抑郁发作 — 比MDD更深的低落，伴精神运动迟滞",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.1, "serotonin": 0.1, "norepinephrine": 0.15, "bdnf": 0.15},
            "hpa_axis": {"stress_reactivity_mult": 1.8, "feedback_strength_mult": 0.3, "cortisol_bias": 0.4},
            "ans": {"baseline_vagal_tone": 0.15, "sympathetic_reactivity_mult": 1.5},
            "mood_system": {"mean_shift": [-0.6, -0.4, -0.5, -0.7, -0.6], "volatility_mult": 0.3},
            "self_awareness": {"agency": 0.15, "coherence": 0.2},
            "predictive_coding": {"precision_mult": 0.3, "free_energy_bias": 0.4},
            "glial": {"neuroinflammation": 0.6, "pruning_rate_mult": 1.8},
            "emotion_regulation": {"regulation_capacity": 0.15, "inhibition": 0.15},
            "social_cognition": {"affective_empathy": 0.2, "contagion": 0.2},
            "brainstem": {"arousal_setpoint": 0.2, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.05, "cortisol": 0.8},
            "scn": {"cortisol_peak_shift": -3, "melatonin_amplitude_mult": 0.5},
            "hippocampus": {"encoding_modulation": 0.1},
            "prefrontal": {"maturity": 0.15, "inhibition": 0.15},
            "limbic": {"valence": -0.7, "arousal": 0.15},
            "neuroplasticity": {"bdnf": 0.15},
            "interoception": {"gut_serotonin": 0.15, "inflammation": 0.6},
            "basal_ganglia": {"habit_strength": 0.2, "td_error_mult": 0.3},
        },
        "onset_default": "gradual",
        "offset_default": "slow_recovery",
        "comorbidities": ["GAD", "MDD"],
        "treatment_responses": {"antidepressant": 0.5, "mood_stabilizer": 0.6, "stimulant": 0.2},
    },

    "GAD": {
        "name": "Generalized Anxiety Disorder",
        "description": "广泛性焦虑障碍 — 持续担忧、过度警觉、肌肉紧张",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.4, "serotonin": 0.3, "norepinephrine": 0.8, "gaba": 0.2},
            "hpa_axis": {"stress_reactivity_mult": 2.0, "feedback_strength_mult": 0.4, "cortisol_bias": 0.3},
            "ans": {"baseline_vagal_tone": 0.25, "sympathetic_reactivity_mult": 1.8},
            "mood_system": {"mean_shift": [-0.2, 0.1, -0.1, -0.3, 0.3], "volatility_mult": 1.8},
            "self_awareness": {"agency": 0.4, "coherence": 0.5, "self_boundary": 0.6},
            "predictive_coding": {"precision_mult": 1.8, "free_energy_bias": 0.5},
            "glial": {"neuroinflammation": 0.3, "pruning_rate_mult": 1.2},
            "emotion_regulation": {"regulation_capacity": 0.3, "inhibition": 0.3},
            "social_cognition": {"affective_empathy": 0.5, "contagion": 0.6},
            "brainstem": {"arousal_setpoint": 0.7, "default_defense": "flight"},
            "hormone": {"oxytocin": 0.3, "cortisol": 0.6, "adrenaline": 0.7},
            "scn": {"cortisol_peak_shift": 1, "melatonin_amplitude_mult": 0.7},
            "hippocampus": {"encoding_modulation": 0.6},
            "prefrontal": {"maturity": 0.4, "inhibition": 0.3},
            "limbic": {"valence": -0.2, "arousal": 0.7},
            "neuroplasticity": {"bdnf": 0.5},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.3},
            "basal_ganglia": {"habit_strength": 0.6, "td_error_mult": 1.5},
        },
        "onset_default": "gradual",
        "offset_default": "natural_remission",
        "comorbidities": ["MDD", "Social_Anxiety", "Panic_Disorder"],
        "treatment_responses": {"anxiolytic": 0.8, "antidepressant": 0.6, "beta_blocker": 0.4},
    },

    "Panic_Disorder": {
        "name": "Panic Disorder",
        "description": "惊恐障碍 — 反复惊恐发作、窒息感、濒死感",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.5, "serotonin": 0.3, "norepinephrine": 0.9, "gaba": 0.15},
            "hpa_axis": {"stress_reactivity_mult": 2.5, "feedback_strength_mult": 0.2, "cortisol_bias": 0.5},
            "ans": {"baseline_vagal_tone": 0.15, "sympathetic_reactivity_mult": 2.5},
            "mood_system": {"mean_shift": [-0.1, 0.3, -0.2, -0.4, 0.5], "volatility_mult": 4.0},
            "self_awareness": {"agency": 0.3, "coherence": 0.3, "self_boundary": 0.4},
            "predictive_coding": {"precision_mult": 2.5, "free_energy_bias": 0.7},
            "glial": {"neuroinflammation": 0.2, "pruning_rate_mult": 1.0},
            "emotion_regulation": {"regulation_capacity": 0.15, "inhibition": 0.1},
            "social_cognition": {"affective_empathy": 0.4, "contagion": 0.5},
            "brainstem": {"arousal_setpoint": 0.8, "default_defense": "flight"},
            "hormone": {"oxytocin": 0.2, "cortisol": 0.8, "adrenaline": 0.9},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.8},
            "hippocampus": {"encoding_modulation": 0.7},
            "prefrontal": {"maturity": 0.3, "inhibition": 0.2},
            "limbic": {"valence": -0.3, "arousal": 0.9},
            "neuroplasticity": {"bdnf": 0.4},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.2},
            "basal_ganglia": {"habit_strength": 0.5, "td_error_mult": 2.0},
        },
        "onset_default": "rapid",
        "offset_default": "treatment_response",
        "comorbidities": ["GAD", "Agoraphobia"],
        "treatment_responses": {"anxiolytic": 0.9, "antidepressant": 0.5, "beta_blocker": 0.6},
    },

    "Social_Anxiety": {
        "name": "Social Anxiety Disorder",
        "description": "社交焦虑障碍 — 社交情境中的强烈恐惧与回避",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.3, "serotonin": 0.25, "norepinephrine": 0.7, "gaba": 0.25},
            "hpa_axis": {"stress_reactivity_mult": 1.8, "feedback_strength_mult": 0.5},
            "ans": {"baseline_vagal_tone": 0.25, "sympathetic_reactivity_mult": 1.6},
            "mood_system": {"mean_shift": [-0.2, 0.0, -0.1, -0.3, 0.2], "volatility_mult": 1.5},
            "self_awareness": {"agency": 0.35, "coherence": 0.4, "self_boundary": 0.5},
            "predictive_coding": {"precision_mult": 1.5, "free_energy_bias": 0.4},
            "glial": {"neuroinflammation": 0.2, "pruning_rate_mult": 1.1},
            "emotion_regulation": {"regulation_capacity": 0.3, "inhibition": 0.3},
            "social_cognition": {"affective_empathy": 0.7, "contagion": 0.8, "social_threat_sensitivity": 0.9},
            "brainstem": {"arousal_setpoint": 0.6, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.2, "cortisol": 0.6},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.8},
            "hippocampus": {"encoding_modulation": 0.5},
            "prefrontal": {"maturity": 0.4, "inhibition": 0.35},
            "limbic": {"valence": -0.2, "arousal": 0.6},
            "neuroplasticity": {"bdnf": 0.5},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.2},
            "basal_ganglia": {"habit_strength": 0.5, "td_error_mult": 1.3},
        },
        "onset_default": "gradual",
        "offset_default": "natural_remission",
        "comorbidities": ["GAD", "MDD"],
        "treatment_responses": {"anxiolytic": 0.7, "antidepressant": 0.6, "beta_blocker": 0.5},
    },

    "PTSD": {
        "name": "Post-Traumatic Stress Disorder",
        "description": "创伤后应激障碍 — 闪回、回避、高警觉、负性认知",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.3, "serotonin": 0.2, "norepinephrine": 0.85, "gaba": 0.15, "bdnf": 0.3},
            "hpa_axis": {"stress_reactivity_mult": 2.5, "feedback_strength_mult": 0.2, "cortisol_bias": 0.4},
            "ans": {"baseline_vagal_tone": 0.15, "sympathetic_reactivity_mult": 2.2},
            "mood_system": {"mean_shift": [-0.3, 0.1, -0.2, -0.5, 0.4], "volatility_mult": 3.0},
            "self_awareness": {"agency": 0.2, "coherence": 0.25, "self_boundary": 0.3},
            "predictive_coding": {"precision_mult": 2.0, "free_energy_bias": 0.6},
            "glial": {"neuroinflammation": 0.5, "pruning_rate_mult": 1.3},
            "emotion_regulation": {"regulation_capacity": 0.15, "inhibition": 0.15},
            "social_cognition": {"affective_empathy": 0.3, "contagion": 0.4},
            "brainstem": {"arousal_setpoint": 0.8, "default_defense": "fight"},
            "hormone": {"oxytocin": 0.15, "cortisol": 0.75, "adrenaline": 0.8},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.5},
            "hippocampus": {"encoding_modulation": 0.8},
            "prefrontal": {"maturity": 0.25, "inhibition": 0.2},
            "limbic": {"valence": -0.4, "arousal": 0.8},
            "neuroplasticity": {"bdnf": 0.3, "ltp_rate_mult": 0.5},
            "interoception": {"gut_serotonin": 0.2, "inflammation": 0.4},
            "basal_ganglia": {"habit_strength": 0.7, "td_error_mult": 1.8},
        },
        "onset_default": "rapid",
        "offset_default": "slow_recovery",
        "comorbidities": ["MDD", "GAD", "Substance_Use"],
        "treatment_responses": {"antidepressant": 0.5, "anxiolytic": 0.6, "mood_stabilizer": 0.3},
    },

    "OCD": {
        "name": "Obsessive-Compulsive Disorder",
        "description": "强迫症 — 侵入性思维 + 仪式性行为",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.6, "serotonin": 0.2, "gaba": 0.2, "glutamate": 0.8},
            "hpa_axis": {"stress_reactivity_mult": 1.5, "feedback_strength_mult": 0.5},
            "ans": {"baseline_vagal_tone": 0.3, "sympathetic_reactivity_mult": 1.4},
            "mood_system": {"mean_shift": [-0.1, 0.0, -0.1, -0.2, 0.1], "volatility_mult": 1.2},
            "self_awareness": {"agency": 0.3, "coherence": 0.4},
            "predictive_coding": {"precision_mult": 2.5, "free_energy_bias": 0.6},
            "glial": {"neuroinflammation": 0.3, "pruning_rate_mult": 0.7},
            "emotion_regulation": {"regulation_capacity": 0.25, "inhibition": 0.2},
            "social_cognition": {"affective_empathy": 0.5, "contagion": 0.4},
            "brainstem": {"arousal_setpoint": 0.6, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.3, "cortisol": 0.5},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.8},
            "hippocampus": {"encoding_modulation": 0.5},
            "prefrontal": {"maturity": 0.5, "inhibition": 0.2},
            "limbic": {"valence": -0.1, "arousal": 0.5},
            "neuroplasticity": {"bdnf": 0.5, "ltp_rate_mult": 0.7},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.2},
            "basal_ganglia": {"habit_strength": 0.9, "td_error_mult": 0.3},
        },
        "onset_default": "insidious",
        "offset_default": "slow_recovery",
        "comorbidities": ["GAD", "MDD"],
        "treatment_responses": {"antidepressant": 0.6, "anxiolytic": 0.5, "antipsychotic": 0.3},
    },

    "schizophrenia_positive": {
        "name": "Schizophrenia — Positive Symptoms",
        "description": "精神分裂症阳性症状 — 幻觉、妄想、思维形式障碍",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.85, "serotonin": 0.4, "glutamate": 0.7, "gaba": 0.2, "bdnf": 0.3},
            "hpa_axis": {"stress_reactivity_mult": 1.8, "feedback_strength_mult": 0.4},
            "ans": {"baseline_vagal_tone": 0.3, "sympathetic_reactivity_mult": 1.5},
            "mood_system": {"mean_shift": [0.0, 0.2, -0.1, -0.2, 0.3], "volatility_mult": 2.5},
            "self_awareness": {"agency": 0.15, "coherence": 0.15, "self_boundary": 0.2},
            "predictive_coding": {"precision_mult": 2.5, "free_energy_bias": 0.3},
            "glial": {"neuroinflammation": 0.6, "pruning_rate_mult": 2.0},
            "emotion_regulation": {"regulation_capacity": 0.2, "inhibition": 0.15},
            "social_cognition": {"affective_empathy": 0.2, "contagion": 0.2},
            "brainstem": {"arousal_setpoint": 0.6, "default_defense": "fight"},
            "hormone": {"oxytocin": 0.1, "cortisol": 0.6},
            "scn": {"cortisol_peak_shift": 1, "melatonin_amplitude_mult": 0.6},
            "hippocampus": {"encoding_modulation": 0.6},
            "prefrontal": {"maturity": 0.15, "inhibition": 0.1},
            "limbic": {"valence": 0.0, "arousal": 0.6},
            "neuroplasticity": {"bdnf": 0.3, "ltp_rate_mult": 0.5},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.5},
            "basal_ganglia": {"habit_strength": 0.6, "td_error_mult": 1.5},
        },
        "onset_default": "gradual",
        "offset_default": "treatment_response",
        "comorbidities": ["MDD", "GAD"],
        "treatment_responses": {"antipsychotic": 0.8, "mood_stabilizer": 0.4, "anxiolytic": 0.3},
    },

    "schizophrenia_negative": {
        "name": "Schizophrenia — Negative Symptoms",
        "description": "精神分裂症阴性症状 — 情感淡漠、意志缺乏、社交退缩",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.15, "serotonin": 0.3, "glutamate": 0.3, "bdnf": 0.2},
            "hpa_axis": {"stress_reactivity_mult": 0.5, "feedback_strength_mult": 0.8},
            "ans": {"baseline_vagal_tone": 0.5, "sympathetic_reactivity_mult": 0.5},
            "mood_system": {"mean_shift": [-0.3, -0.4, -0.3, -0.4, -0.3], "volatility_mult": 0.2},
            "self_awareness": {"agency": 0.2, "coherence": 0.3},
            "predictive_coding": {"precision_mult": 0.3, "free_energy_bias": 0.1},
            "glial": {"neuroinflammation": 0.5, "pruning_rate_mult": 2.0},
            "emotion_regulation": {"regulation_capacity": 0.5, "inhibition": 0.7},
            "social_cognition": {"affective_empathy": 0.15, "contagion": 0.1},
            "brainstem": {"arousal_setpoint": 0.2, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.05, "cortisol": 0.3},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.7},
            "hippocampus": {"encoding_modulation": 0.2},
            "prefrontal": {"maturity": 0.2, "inhibition": 0.5},
            "limbic": {"valence": -0.2, "arousal": 0.15},
            "neuroplasticity": {"bdnf": 0.2, "ltp_rate_mult": 0.3},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.4},
            "basal_ganglia": {"habit_strength": 0.2, "td_error_mult": 0.3},
        },
        "onset_default": "insidious",
        "offset_default": "slow_recovery",
        "comorbidities": ["MDD", "schizophrenia_positive"],
        "treatment_responses": {"antipsychotic": 0.3, "stimulant": 0.4, "antidepressant": 0.3},
    },

    "ASD": {
        "name": "Autism Spectrum Disorder",
        "description": "自闭症谱系障碍 — 社交沟通困难、刻板行为、感觉过敏",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.5, "serotonin": 0.6, "gaba": 0.4, "glutamate": 0.7},
            "hpa_axis": {"stress_reactivity_mult": 1.5, "feedback_strength_mult": 0.6},
            "ans": {"baseline_vagal_tone": 0.3, "sympathetic_reactivity_mult": 1.5},
            "mood_system": {"mean_shift": [-0.1, 0.0, -0.1, -0.1, 0.0], "volatility_mult": 1.0},
            "self_awareness": {"agency": 0.5, "coherence": 0.5, "self_boundary": 0.8},
            "predictive_coding": {"precision_mult": 1.8, "free_energy_bias": 0.5},
            "glial": {"neuroinflammation": 0.3, "pruning_rate_mult": 0.6},
            "emotion_regulation": {"regulation_capacity": 0.3, "inhibition": 0.4},
            "social_cognition": {"affective_empathy": 0.2, "cognitive_empathy": 0.3, "contagion": 0.15},
            "brainstem": {"arousal_setpoint": 0.5, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.15, "cortisol": 0.4},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.7},
            "hippocampus": {"encoding_modulation": 0.5},
            "prefrontal": {"maturity": 0.4, "inhibition": 0.4},
            "limbic": {"valence": 0.0, "arousal": 0.4},
            "neuroplasticity": {"bdnf": 0.5, "ltp_rate_mult": 0.8},
            "interoception": {"gut_serotonin": 0.5, "inflammation": 0.3},
            "basal_ganglia": {"habit_strength": 0.7, "td_error_mult": 0.8},
        },
        "onset_default": "insidious",
        "offset_default": "slow_recovery",
        "comorbidities": ["GAD", "OCD"],
        "treatment_responses": {"anxiolytic": 0.4, "antidepressant": 0.3, "stimulant": 0.2},
    },

    "ADHD": {
        "name": "Attention Deficit Hyperactivity Disorder",
        "description": "注意力缺陷多动障碍 — 注意力分散、冲动、多动",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.3, "norepinephrine": 0.3, "bdnf": 0.5},
            "hpa_axis": {"stress_reactivity_mult": 1.2, "feedback_strength_mult": 0.7},
            "ans": {"baseline_vagal_tone": 0.35, "sympathetic_reactivity_mult": 1.3},
            "mood_system": {"mean_shift": [0.0, 0.1, 0.0, -0.1, 0.1], "volatility_mult": 2.0},
            "self_awareness": {"agency": 0.4, "coherence": 0.4},
            "predictive_coding": {"precision_mult": 0.5, "free_energy_bias": 0.1},
            "glial": {"neuroinflammation": 0.1, "pruning_rate_mult": 0.8},
            "emotion_regulation": {"regulation_capacity": 0.25, "inhibition": 0.2},
            "social_cognition": {"affective_empathy": 0.5, "contagion": 0.5},
            "brainstem": {"arousal_setpoint": 0.4, "default_defense": "flight"},
            "hormone": {"oxytocin": 0.4, "cortisol": 0.4},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.8},
            "hippocampus": {"encoding_modulation": 0.4},
            "prefrontal": {"maturity": 0.2, "inhibition": 0.15},
            "limbic": {"valence": 0.0, "arousal": 0.5},
            "neuroplasticity": {"bdnf": 0.5, "ltp_rate_mult": 0.7},
            "interoception": {"gut_serotonin": 0.4, "inflammation": 0.1},
            "basal_ganglia": {"habit_strength": 0.3, "td_error_mult": 0.5},
        },
        "onset_default": "insidious",
        "offset_default": "treatment_response",
        "comorbidities": ["GAD", "MDD", "ASD"],
        "treatment_responses": {"stimulant": 0.8, "antidepressant": 0.3, "anxiolytic": 0.2},
    },

    "BPD": {
        "name": "Borderline Personality Disorder",
        "description": "边缘型人格障碍 — 情绪不稳定、自我认同混乱、害怕被抛弃",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.5, "serotonin": 0.2, "norepinephrine": 0.7, "gaba": 0.2},
            "hpa_axis": {"stress_reactivity_mult": 2.0, "feedback_strength_mult": 0.3},
            "ans": {"baseline_vagal_tone": 0.2, "sympathetic_reactivity_mult": 2.0},
            "mood_system": {"mean_shift": [-0.1, 0.0, -0.1, -0.2, 0.1], "volatility_mult": 5.0},
            "self_awareness": {"agency": 0.2, "coherence": 0.15, "self_boundary": 0.15},
            "predictive_coding": {"precision_mult": 1.5, "free_energy_bias": 0.4},
            "glial": {"neuroinflammation": 0.3, "pruning_rate_mult": 1.2},
            "emotion_regulation": {"regulation_capacity": 0.1, "inhibition": 0.1},
            "social_cognition": {"affective_empathy": 0.8, "contagion": 0.9, "abandonment_sensitivity": 0.9},
            "brainstem": {"arousal_setpoint": 0.6, "default_defense": "fight"},
            "hormone": {"oxytocin": 0.6, "cortisol": 0.6},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.7},
            "hippocampus": {"encoding_modulation": 0.6},
            "prefrontal": {"maturity": 0.15, "inhibition": 0.1},
            "limbic": {"valence": -0.1, "arousal": 0.7},
            "neuroplasticity": {"bdnf": 0.4, "ltp_rate_mult": 0.6},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.3},
            "basal_ganglia": {"habit_strength": 0.5, "td_error_mult": 1.5},
        },
        "onset_default": "gradual",
        "offset_default": "slow_recovery",
        "comorbidities": ["MDD", "PTSD", "Substance_Use", "Eating_Disorder"],
        "treatment_responses": {"mood_stabilizer": 0.6, "antidepressant": 0.4, "anxiolytic": 0.3},
    },

    "ASPD": {
        "name": "Antisocial Personality Disorder",
        "description": "反社会型人格障碍 — 冷漠、操纵、冲动、缺乏共情",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.7, "serotonin": 0.3, "norepinephrine": 0.6, "gaba": 0.4},
            "hpa_axis": {"stress_reactivity_mult": 0.5, "feedback_strength_mult": 0.9},
            "ans": {"baseline_vagal_tone": 0.6, "sympathetic_reactivity_mult": 0.7},
            "mood_system": {"mean_shift": [0.0, 0.1, 0.0, 0.0, 0.1], "volatility_mult": 1.0},
            "self_awareness": {"agency": 0.7, "coherence": 0.6, "self_boundary": 0.8},
            "predictive_coding": {"precision_mult": 0.6, "free_energy_bias": -0.1},
            "glial": {"neuroinflammation": 0.1, "pruning_rate_mult": 0.8},
            "emotion_regulation": {"regulation_capacity": 0.5, "inhibition": 0.3},
            "social_cognition": {"affective_empathy": 0.1, "cognitive_empathy": 0.7, "contagion": 0.1},
            "brainstem": {"arousal_setpoint": 0.5, "default_defense": "fight"},
            "hormone": {"oxytocin": 0.05, "cortisol": 0.2, "testosterone": 0.8},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.9},
            "hippocampus": {"encoding_modulation": 0.5},
            "prefrontal": {"maturity": 0.3, "inhibition": 0.2},
            "limbic": {"valence": 0.1, "arousal": 0.4},
            "neuroplasticity": {"bdnf": 0.5},
            "interoception": {"gut_serotonin": 0.4, "inflammation": 0.1},
            "basal_ganglia": {"habit_strength": 0.6, "td_error_mult": 1.2},
        },
        "onset_default": "insidious",
        "offset_default": "slow_recovery",
        "comorbidities": ["Substance_Use", "bipolar_mania"],
        "treatment_responses": {"mood_stabilizer": 0.3, "antipsychotic": 0.2},
    },

    "NPD": {
        "name": "Narcissistic Personality Disorder",
        "description": "自恋型人格障碍 — 夸大自我、需要赞美、缺乏共情",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.7, "serotonin": 0.4, "norepinephrine": 0.5},
            "hpa_axis": {"stress_reactivity_mult": 1.3, "feedback_strength_mult": 0.5},
            "ans": {"baseline_vagal_tone": 0.4, "sympathetic_reactivity_mult": 1.2},
            "mood_system": {"mean_shift": [0.1, 0.2, 0.1, 0.0, 0.2], "volatility_mult": 1.5},
            "self_awareness": {"agency": 0.8, "coherence": 0.4, "self_boundary": 0.7},
            "predictive_coding": {"precision_mult": 0.8, "free_energy_bias": -0.2},
            "glial": {"neuroinflammation": 0.1, "pruning_rate_mult": 0.9},
            "emotion_regulation": {"regulation_capacity": 0.4, "inhibition": 0.3},
            "social_cognition": {"affective_empathy": 0.15, "cognitive_empathy": 0.6, "contagion": 0.2},
            "brainstem": {"arousal_setpoint": 0.5, "default_defense": "fight"},
            "hormone": {"oxytocin": 0.1, "cortisol": 0.4},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.9},
            "hippocampus": {"encoding_modulation": 0.5},
            "prefrontal": {"maturity": 0.4, "inhibition": 0.3},
            "limbic": {"valence": 0.2, "arousal": 0.4},
            "neuroplasticity": {"bdnf": 0.5},
            "interoception": {"gut_serotonin": 0.4, "inflammation": 0.1},
            "basal_ganglia": {"habit_strength": 0.5, "td_error_mult": 1.0},
        },
        "onset_default": "insidious",
        "offset_default": "slow_recovery",
        "comorbidities": ["MDD", "bipolar_mania"],
        "treatment_responses": {"antidepressant": 0.3, "mood_stabilizer": 0.2},
    },

    "Dissociative": {
        "name": "Dissociative Disorder",
        "description": "解离性障碍 — 自我感丧失、现实解体、记忆断层",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.3, "serotonin": 0.3, "norepinephrine": 0.2, "gaba": 0.7, "glutamate": 0.2},
            "hpa_axis": {"stress_reactivity_mult": 0.3, "feedback_strength_mult": 0.9},
            "ans": {"baseline_vagal_tone": 0.7, "sympathetic_reactivity_mult": 0.3},
            "mood_system": {"mean_shift": [-0.1, -0.2, -0.1, -0.1, -0.1], "volatility_mult": 0.3},
            "self_awareness": {"agency": 0.1, "coherence": 0.1, "self_boundary": 0.1},
            "predictive_coding": {"precision_mult": 0.2, "free_energy_bias": 0.1},
            "glial": {"neuroinflammation": 0.2, "pruning_rate_mult": 1.0},
            "emotion_regulation": {"regulation_capacity": 0.6, "inhibition": 0.8},
            "social_cognition": {"affective_empathy": 0.2, "contagion": 0.1},
            "brainstem": {"arousal_setpoint": 0.2, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.1, "cortisol": 0.2},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.8},
            "hippocampus": {"encoding_modulation": 0.1},
            "prefrontal": {"maturity": 0.3, "inhibition": 0.6},
            "limbic": {"valence": -0.1, "arousal": 0.15},
            "neuroplasticity": {"bdnf": 0.4, "ltp_rate_mult": 0.4},
            "interoception": {"gut_serotonin": 0.4, "inflammation": 0.2},
            "basal_ganglia": {"habit_strength": 0.3, "td_error_mult": 0.4},
        },
        "onset_default": "rapid",
        "offset_default": "slow_recovery",
        "comorbidities": ["PTSD", "BPD"],
        "treatment_responses": {"anxiolytic": 0.4, "antidepressant": 0.3},
    },

    "Burnout": {
        "name": "Burnout Syndrome",
        "description": "倦怠综合征 — 情感耗竭、去人格化、成就感降低",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.2, "serotonin": 0.25, "norepinephrine": 0.3, "bdnf": 0.3},
            "hpa_axis": {"stress_reactivity_mult": 1.5, "feedback_strength_mult": 0.5, "cortisol_bias": 0.2},
            "ans": {"baseline_vagal_tone": 0.3, "sympathetic_reactivity_mult": 1.3},
            "mood_system": {"mean_shift": [-0.3, -0.2, -0.3, -0.4, -0.2], "volatility_mult": 0.5},
            "self_awareness": {"agency": 0.3, "coherence": 0.4},
            "predictive_coding": {"precision_mult": 0.4, "free_energy_bias": 0.3},
            "glial": {"neuroinflammation": 0.4, "pruning_rate_mult": 1.3},
            "emotion_regulation": {"regulation_capacity": 0.3, "inhibition": 0.3},
            "social_cognition": {"affective_empathy": 0.2, "contagion": 0.2},
            "brainstem": {"arousal_setpoint": 0.3, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.2, "cortisol": 0.6},
            "scn": {"cortisol_peak_shift": -1, "melatonin_amplitude_mult": 0.7},
            "hippocampus": {"encoding_modulation": 0.3},
            "prefrontal": {"maturity": 0.3, "inhibition": 0.3},
            "limbic": {"valence": -0.3, "arousal": 0.3},
            "neuroplasticity": {"bdnf": 0.3, "ltp_rate_mult": 0.5},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.4},
            "basal_ganglia": {"habit_strength": 0.3, "td_error_mult": 0.4},
        },
        "onset_default": "gradual",
        "offset_default": "natural_remission",
        "comorbidities": ["MDD", "GAD"],
        "treatment_responses": {"antidepressant": 0.5, "anxiolytic": 0.3},
    },

    "Cyclothymia": {
        "name": "Cyclothymic Disorder",
        "description": "环形心境障碍 — 轻度躁狂与轻度抑郁交替",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.5, "serotonin": 0.4, "norepinephrine": 0.5},
            "hpa_axis": {"stress_reactivity_mult": 1.2, "feedback_strength_mult": 0.6},
            "ans": {"baseline_vagal_tone": 0.35, "sympathetic_reactivity_mult": 1.3},
            "mood_system": {"mean_shift": [0.0, 0.0, 0.0, 0.0, 0.0], "volatility_mult": 3.0},
            "self_awareness": {"agency": 0.5, "coherence": 0.5},
            "predictive_coding": {"precision_mult": 1.0, "free_energy_bias": 0.0},
            "glial": {"neuroinflammation": 0.1, "pruning_rate_mult": 1.0},
            "emotion_regulation": {"regulation_capacity": 0.4, "inhibition": 0.35},
            "social_cognition": {"affective_empathy": 0.5, "contagion": 0.5},
            "brainstem": {"arousal_setpoint": 0.5, "default_defense": "flight"},
            "hormone": {"oxytocin": 0.4, "cortisol": 0.4},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.8},
            "hippocampus": {"encoding_modulation": 0.5},
            "prefrontal": {"maturity": 0.4, "inhibition": 0.35},
            "limbic": {"valence": 0.0, "arousal": 0.5},
            "neuroplasticity": {"bdnf": 0.5},
            "interoception": {"gut_serotonin": 0.4, "inflammation": 0.1},
            "basal_ganglia": {"habit_strength": 0.5, "td_error_mult": 1.0},
        },
        "onset_default": "insidious",
        "offset_default": "natural_remission",
        "comorbidities": ["bipolar_mania", "MDD"],
        "treatment_responses": {"mood_stabilizer": 0.7, "antidepressant": 0.3},
    },

    "Dysthymia": {
        "name": "Persistent Depressive Disorder (Dysthymia)",
        "description": "持续性抑郁障碍 — 长期轻度抑郁，>2年",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.3, "serotonin": 0.3, "bdnf": 0.4},
            "hpa_axis": {"stress_reactivity_mult": 1.3, "feedback_strength_mult": 0.6},
            "ans": {"baseline_vagal_tone": 0.35, "sympathetic_reactivity_mult": 1.2},
            "mood_system": {"mean_shift": [-0.25, -0.15, -0.2, -0.3, -0.2], "volatility_mult": 0.5},
            "self_awareness": {"agency": 0.35, "coherence": 0.4},
            "predictive_coding": {"precision_mult": 0.6, "free_energy_bias": 0.2},
            "glial": {"neuroinflammation": 0.3, "pruning_rate_mult": 1.2},
            "emotion_regulation": {"regulation_capacity": 0.35, "inhibition": 0.35},
            "social_cognition": {"affective_empathy": 0.4, "contagion": 0.4},
            "brainstem": {"arousal_setpoint": 0.35, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.2, "cortisol": 0.45},
            "scn": {"cortisol_peak_shift": -1, "melatonin_amplitude_mult": 0.75},
            "hippocampus": {"encoding_modulation": 0.35},
            "prefrontal": {"maturity": 0.35, "inhibition": 0.35},
            "limbic": {"valence": -0.25, "arousal": 0.3},
            "neuroplasticity": {"bdnf": 0.4},
            "interoception": {"gut_serotonin": 0.35, "inflammation": 0.3},
            "basal_ganglia": {"habit_strength": 0.4, "td_error_mult": 0.6},
        },
        "onset_default": "insidious",
        "offset_default": "slow_recovery",
        "comorbidities": ["MDD", "GAD"],
        "treatment_responses": {"antidepressant": 0.6, "anxiolytic": 0.3},
    },

    "Specific_Phobia": {
        "name": "Specific Phobia",
        "description": "特定恐惧症 — 对特定物体/情境的强烈恐惧",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.4, "serotonin": 0.4, "norepinephrine": 0.7, "gaba": 0.3},
            "hpa_axis": {"stress_reactivity_mult": 1.8, "feedback_strength_mult": 0.5},
            "ans": {"baseline_vagal_tone": 0.3, "sympathetic_reactivity_mult": 1.6},
            "mood_system": {"mean_shift": [-0.1, 0.0, -0.1, -0.2, 0.2], "volatility_mult": 1.5},
            "self_awareness": {"agency": 0.4, "coherence": 0.5},
            "predictive_coding": {"precision_mult": 1.8, "free_energy_bias": 0.4},
            "glial": {"neuroinflammation": 0.1, "pruning_rate_mult": 1.0},
            "emotion_regulation": {"regulation_capacity": 0.35, "inhibition": 0.3},
            "social_cognition": {"affective_empathy": 0.5, "contagion": 0.5},
            "brainstem": {"arousal_setpoint": 0.6, "default_defense": "flight"},
            "hormone": {"oxytocin": 0.3, "cortisol": 0.5},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.85},
            "hippocampus": {"encoding_modulation": 0.5},
            "prefrontal": {"maturity": 0.4, "inhibition": 0.35},
            "limbic": {"valence": -0.15, "arousal": 0.55},
            "neuroplasticity": {"bdnf": 0.5},
            "interoception": {"gut_serotonin": 0.4, "inflammation": 0.1},
            "basal_ganglia": {"habit_strength": 0.5, "td_error_mult": 1.2},
        },
        "onset_default": "rapid",
        "offset_default": "natural_remission",
        "comorbidities": ["GAD", "Agoraphobia"],
        "treatment_responses": {"anxiolytic": 0.6, "beta_blocker": 0.5},
    },

    "Agoraphobia": {
        "name": "Agoraphobia",
        "description": "广场恐惧症 — 对开放空间/难以逃离情境的恐惧",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.3, "serotonin": 0.3, "norepinephrine": 0.7, "gaba": 0.25},
            "hpa_axis": {"stress_reactivity_mult": 1.8, "feedback_strength_mult": 0.4},
            "ans": {"baseline_vagal_tone": 0.25, "sympathetic_reactivity_mult": 1.7},
            "mood_system": {"mean_shift": [-0.2, 0.0, -0.1, -0.3, 0.2], "volatility_mult": 1.5},
            "self_awareness": {"agency": 0.3, "coherence": 0.4},
            "predictive_coding": {"precision_mult": 1.6, "free_energy_bias": 0.5},
            "glial": {"neuroinflammation": 0.2, "pruning_rate_mult": 1.1},
            "emotion_regulation": {"regulation_capacity": 0.3, "inhibition": 0.25},
            "social_cognition": {"affective_empathy": 0.4, "contagion": 0.5},
            "brainstem": {"arousal_setpoint": 0.65, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.2, "cortisol": 0.55},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.8},
            "hippocampus": {"encoding_modulation": 0.5},
            "prefrontal": {"maturity": 0.35, "inhibition": 0.3},
            "limbic": {"valence": -0.2, "arousal": 0.6},
            "neuroplasticity": {"bdnf": 0.5},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.2},
            "basal_ganglia": {"habit_strength": 0.5, "td_error_mult": 1.2},
        },
        "onset_default": "gradual",
        "offset_default": "natural_remission",
        "comorbidities": ["Panic_Disorder", "GAD"],
        "treatment_responses": {"anxiolytic": 0.7, "antidepressant": 0.5},
    },

    "Somatic_Symptom": {
        "name": "Somatic Symptom Disorder",
        "description": "躯体症状障碍 — 过度关注身体症状，健康焦虑",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.4, "serotonin": 0.3, "norepinephrine": 0.6, "gaba": 0.3},
            "hpa_axis": {"stress_reactivity_mult": 1.5, "feedback_strength_mult": 0.5},
            "ans": {"baseline_vagal_tone": 0.3, "sympathetic_reactivity_mult": 1.4},
            "mood_system": {"mean_shift": [-0.15, 0.0, -0.1, -0.2, 0.1], "volatility_mult": 1.3},
            "self_awareness": {"agency": 0.4, "coherence": 0.4},
            "predictive_coding": {"precision_mult": 1.8, "free_energy_bias": 0.5},
            "glial": {"neuroinflammation": 0.3, "pruning_rate_mult": 1.1},
            "emotion_regulation": {"regulation_capacity": 0.3, "inhibition": 0.3},
            "social_cognition": {"affective_empathy": 0.5, "contagion": 0.5},
            "brainstem": {"arousal_setpoint": 0.55, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.3, "cortisol": 0.5},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.8},
            "hippocampus": {"encoding_modulation": 0.5},
            "prefrontal": {"maturity": 0.4, "inhibition": 0.35},
            "limbic": {"valence": -0.15, "arousal": 0.5},
            "neuroplasticity": {"bdnf": 0.5},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.5, "interoceptive_salience": 0.9},
            "basal_ganglia": {"habit_strength": 0.5, "td_error_mult": 1.0},
        },
        "onset_default": "gradual",
        "offset_default": "natural_remission",
        "comorbidities": ["GAD", "MDD"],
        "treatment_responses": {"antidepressant": 0.5, "anxiolytic": 0.4},
    },

    "Anorexia_Tendencies": {
        "name": "Anorexia Nervosa Tendencies",
        "description": "神经性厌食倾向 — 体像扭曲、限制饮食、过度运动",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.3, "serotonin": 0.7, "norepinephrine": 0.5, "gaba": 0.3},
            "hpa_axis": {"stress_reactivity_mult": 1.5, "feedback_strength_mult": 0.5},
            "ans": {"baseline_vagal_tone": 0.3, "sympathetic_reactivity_mult": 1.3},
            "mood_system": {"mean_shift": [-0.1, 0.0, -0.1, -0.2, 0.0], "volatility_mult": 0.8},
            "self_awareness": {"agency": 0.5, "coherence": 0.3, "body_image_distortion": 0.9},
            "predictive_coding": {"precision_mult": 1.5, "free_energy_bias": 0.4},
            "glial": {"neuroinflammation": 0.3, "pruning_rate_mult": 1.2},
            "emotion_regulation": {"regulation_capacity": 0.3, "inhibition": 0.5},
            "social_cognition": {"affective_empathy": 0.4, "contagion": 0.4},
            "brainstem": {"arousal_setpoint": 0.4, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.2, "cortisol": 0.6, "estrogen": 0.2},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.7},
            "hippocampus": {"encoding_modulation": 0.4},
            "prefrontal": {"maturity": 0.4, "inhibition": 0.5},
            "limbic": {"valence": -0.1, "arousal": 0.4},
            "neuroplasticity": {"bdnf": 0.3},
            "interoception": {"gut_serotonin": 0.5, "inflammation": 0.4, "interoceptive_salience": 0.3},
            "basal_ganglia": {"habit_strength": 0.7, "td_error_mult": 0.8},
        },
        "onset_default": "insidious",
        "offset_default": "slow_recovery",
        "comorbidities": ["MDD", "OCD", "BPD"],
        "treatment_responses": {"antidepressant": 0.4, "anxiolytic": 0.3},
    },

    "Substance_Use": {
        "name": "Substance Use Disorder",
        "description": "物质使用障碍 — 成瘾、耐受、戒断",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.8, "serotonin": 0.3, "gaba": 0.3, "glutamate": 0.7, "bdnf": 0.3},
            "hpa_axis": {"stress_reactivity_mult": 1.5, "feedback_strength_mult": 0.4},
            "ans": {"baseline_vagal_tone": 0.3, "sympathetic_reactivity_mult": 1.4},
            "mood_system": {"mean_shift": [0.0, 0.1, 0.0, -0.1, 0.2], "volatility_mult": 2.0},
            "self_awareness": {"agency": 0.25, "coherence": 0.35},
            "predictive_coding": {"precision_mult": 0.5, "free_energy_bias": 0.3},
            "glial": {"neuroinflammation": 0.4, "pruning_rate_mult": 1.3},
            "emotion_regulation": {"regulation_capacity": 0.2, "inhibition": 0.15},
            "social_cognition": {"affective_empathy": 0.3, "contagion": 0.4},
            "brainstem": {"arousal_setpoint": 0.5, "default_defense": "flight"},
            "hormone": {"oxytocin": 0.2, "cortisol": 0.5},
            "scn": {"cortisol_peak_shift": 0, "melatonin_amplitude_mult": 0.6},
            "hippocampus": {"encoding_modulation": 0.4},
            "prefrontal": {"maturity": 0.2, "inhibition": 0.15},
            "limbic": {"valence": 0.0, "arousal": 0.5},
            "neuroplasticity": {"bdnf": 0.3, "ltp_rate_mult": 0.6},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.4},
            "basal_ganglia": {"habit_strength": 0.9, "td_error_mult": 2.0},
        },
        "onset_default": "gradual",
        "offset_default": "slow_recovery",
        "comorbidities": ["MDD", "PTSD", "BPD", "ASPD"],
        "treatment_responses": {"mood_stabilizer": 0.5, "antidepressant": 0.4, "anxiolytic": 0.3},
    },

    "Prolonged_Grief": {
        "name": "Prolonged Grief Disorder",
        "description": "延长哀伤障碍 — 持续强烈哀伤、身份认同混乱",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.2, "serotonin": 0.2, "norepinephrine": 0.4, "bdnf": 0.3},
            "hpa_axis": {"stress_reactivity_mult": 1.5, "feedback_strength_mult": 0.5},
            "ans": {"baseline_vagal_tone": 0.3, "sympathetic_reactivity_mult": 1.2},
            "mood_system": {"mean_shift": [-0.4, -0.2, -0.3, -0.5, -0.3], "volatility_mult": 0.6},
            "self_awareness": {"agency": 0.25, "coherence": 0.3},
            "predictive_coding": {"precision_mult": 0.5, "free_energy_bias": 0.3},
            "glial": {"neuroinflammation": 0.4, "pruning_rate_mult": 1.3},
            "emotion_regulation": {"regulation_capacity": 0.3, "inhibition": 0.3},
            "social_cognition": {"affective_empathy": 0.6, "contagion": 0.5},
            "brainstem": {"arousal_setpoint": 0.35, "default_defense": "freeze"},
            "hormone": {"oxytocin": 0.3, "cortisol": 0.5},
            "scn": {"cortisol_peak_shift": -1, "melatonin_amplitude_mult": 0.7},
            "hippocampus": {"encoding_modulation": 0.7},
            "prefrontal": {"maturity": 0.35, "inhibition": 0.35},
            "limbic": {"valence": -0.4, "arousal": 0.3},
            "neuroplasticity": {"bdnf": 0.3},
            "interoception": {"gut_serotonin": 0.3, "inflammation": 0.3},
            "basal_ganglia": {"habit_strength": 0.4, "td_error_mult": 0.5},
        },
        "onset_default": "rapid",
        "offset_default": "slow_recovery",
        "comorbidities": ["MDD", "PTSD"],
        "treatment_responses": {"antidepressant": 0.5, "anxiolytic": 0.3},
    },

    "Delirium": {
        "name": "Delirium",
        "description": "谵妄 — 急性意识模糊、注意力障碍、感知障碍",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.7, "serotonin": 0.5, "norepinephrine": 0.8, "acetylcholine": 0.1, "glutamate": 0.8},
            "hpa_axis": {"stress_reactivity_mult": 2.0, "feedback_strength_mult": 0.3},
            "ans": {"baseline_vagal_tone": 0.2, "sympathetic_reactivity_mult": 2.0},
            "mood_system": {"mean_shift": [0.0, 0.2, 0.0, -0.1, 0.3], "volatility_mult": 5.0},
            "self_awareness": {"agency": 0.1, "coherence": 0.1, "self_boundary": 0.15},
            "predictive_coding": {"precision_mult": 2.0, "free_energy_bias": 0.5},
            "glial": {"neuroinflammation": 0.7, "pruning_rate_mult": 1.5},
            "emotion_regulation": {"regulation_capacity": 0.1, "inhibition": 0.1},
            "social_cognition": {"affective_empathy": 0.2, "contagion": 0.3},
            "brainstem": {"arousal_setpoint": 0.7, "default_defense": "fight"},
            "hormone": {"oxytocin": 0.1, "cortisol": 0.7, "adrenaline": 0.8},
            "scn": {"cortisol_peak_shift": 2, "melatonin_amplitude_mult": 0.3},
            "hippocampus": {"encoding_modulation": 0.1},
            "prefrontal": {"maturity": 0.1, "inhibition": 0.1},
            "limbic": {"valence": 0.0, "arousal": 0.8},
            "neuroplasticity": {"bdnf": 0.2, "ltp_rate_mult": 0.3},
            "interoception": {"gut_serotonin": 0.2, "inflammation": 0.6},
            "basal_ganglia": {"habit_strength": 0.4, "td_error_mult": 1.5},
        },
        "onset_default": "rapid",
        "offset_default": "treatment_response",
        "comorbidities": ["Dissociative", "schizophrenia_positive"],
        "treatment_responses": {"antipsychotic": 0.7, "anxiolytic": 0.5},
    },

    # ── 新增: 失眠障碍 ──
    "Insomnia_Disorder": {
        "name": "Insomnia Disorder",
        "description": "失眠障碍 — 持续入睡/维持困难，非自愿早醒",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"gaba": 0.25, "norepinephrine": 0.7, "serotonin": 0.4},
            "hpa_axis": {"cortisol_bias": 0.3, "stress_reactivity_mult": 1.5},
            "ans": {"baseline_vagal_tone": 0.2, "sympathetic_reactivity_mult": 1.6},
            "scn": {"melatonin_amplitude_mult": 0.4, "cortisol_peak_shift": -1},
            "brainstem": {"arousal_setpoint": 0.7},
            "limbic": {"arousal": 0.6, "valence": -0.2},
            "prefrontal": {"inhibition": 0.3},
            "interoception": {"interoceptive_salience": 0.6},
        },
        "onset_default": "gradual",
        "offset_default": "treatment_response",
        "comorbidities": ["GAD", "MDD", "Panic_Disorder"],
        "treatment_responses": {"sedative": 0.7, "antidepressant": 0.4, "dora": 0.8},
    },

    # ── 新增: 神经感染障碍 ──
    "Neuroinfectious": {
        "name": "Neuroinfectious Disorder",
        "description": "病原体触发神经炎症 — 感染介导的认知/情绪/运动/睡眠障碍",
        "severity_multipliers": {"mild": 0.3, "moderate": 0.6, "severe": 1.0},
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.3, "serotonin": 0.3, "acetylcholine": 0.3, "bdnf": 0.2},
            "glial": {"neuroinflammation": 0.7, "pruning_rate_mult": 2.0},
            "hpa_axis": {"stress_reactivity_mult": 1.8, "cortisol_bias": 0.3},
            "prefrontal": {"maturity": 0.3, "inhibition": 0.3},
            "hippocampus": {"encoding_modulation": 0.3},
            "limbic": {"arousal": 0.6, "valence": -0.2},
            "interoception": {"inflammation": 0.7},
            "brainstem": {"arousal_setpoint": 0.6},
        },
        "onset_default": "rapid",
        "offset_default": "slow_recovery",
        "comorbidities": ["MDD", "GAD", "Delirium"],
        "treatment_responses": {"antibiotic": 0.8, "antidepressant": 0.3, "antipsychotic": 0.2},
    },
}


# ===== 7种情绪状态 Profile =====

EMOTION_STATE_PROFILES = {
    "emotional_blunting": {
        "name": "Emotional Blunting",
        "description": "情感迟钝 — 情绪强度普遍降低，对刺激反应减弱",
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.2, "serotonin": 0.3, "norepinephrine": 0.3},
            "mood_system": {"volatility_mult": 0.2},
            "emotion_regulation": {"regulation_capacity": 0.8, "inhibition": 0.9},
            "predictive_coding": {"precision_mult": 0.3},
            "limbic": {"arousal": 0.2},
            "social_cognition": {"affective_empathy": 0.2, "contagion": 0.15},
            "interoception": {"interoceptive_salience": 0.2},
        },
    },

    "emotional_lability": {
        "name": "Emotional Lability",
        "description": "情感不稳 — 情绪快速剧烈波动，难以预测",
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.6, "serotonin": 0.2, "gaba": 0.15, "norepinephrine": 0.7},
            "mood_system": {"volatility_mult": 4.0},
            "emotion_regulation": {"regulation_capacity": 0.1, "inhibition": 0.1},
            "predictive_coding": {"precision_mult": 1.5},
            "limbic": {"arousal": 0.7},
            "prefrontal": {"maturity": 0.2, "inhibition": 0.15},
            "hpa_axis": {"stress_reactivity_mult": 1.8},
        },
    },

    "alexithymia": {
        "name": "Alexithymia",
        "description": "述情障碍 — 难以识别和描述自己的情绪",
        "subsystem_overrides": {
            "neurotransmitter": {"serotonin": 0.4, "dopamine": 0.4},
            "mood_system": {"volatility_mult": 0.5},
            "self_awareness": {"agency": 0.4, "coherence": 0.3, "introspection_depth": 0.1},
            "emotion_regulation": {"regulation_capacity": 0.3},
            "predictive_coding": {"precision_mult": 0.4},
            "interoception": {"interoceptive_salience": 0.2},
            "social_cognition": {"affective_empathy": 0.2},
            "limbic": {"arousal": 0.3},
        },
    },

    "mixed_emotions": {
        "name": "Mixed Emotions",
        "description": "混合情绪 — 同时体验矛盾情绪 (如悲喜交加)",
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.5, "serotonin": 0.4, "norepinephrine": 0.6},
            "mood_system": {"volatility_mult": 2.0},
            "emotion_regulation": {"regulation_capacity": 0.3, "inhibition": 0.2},
            "predictive_coding": {"precision_mult": 1.2},
            "limbic": {"valence": 0.0, "arousal": 0.6},
            "self_awareness": {"coherence": 0.3},
        },
    },

    "contagion_hypersensitivity": {
        "name": "Emotional Contagion Hypersensitivity",
        "description": "情绪传染超敏 — 极易受他人情绪影响",
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.5, "serotonin": 0.4, "oxytocin": 0.9},
            "mood_system": {"volatility_mult": 2.0},
            "emotion_regulation": {"regulation_capacity": 0.3},
            "social_cognition": {"affective_empathy": 0.9, "contagion": 0.95},
            "self_awareness": {"self_boundary": 0.2},
            "hormone": {"oxytocin": 0.9},
            "limbic": {"arousal": 0.6},
        },
    },

    "anhedonia": {
        "name": "Anhedonia",
        "description": "快感缺失 — 无法体验愉悦，奖励系统钝化",
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.1, "serotonin": 0.2, "bdnf": 0.2},
            "mood_system": {"mean_shift": [-0.4, -0.3, -0.3, -0.5, -0.3], "volatility_mult": 0.3},
            "emotion_regulation": {"regulation_capacity": 0.4, "inhibition": 0.6},
            "predictive_coding": {"precision_mult": 0.3},
            "limbic": {"valence": -0.4, "arousal": 0.2},
            "basal_ganglia": {"td_error_mult": 0.2, "habit_strength": 0.2},
            "hormone": {"oxytocin": 0.1},
            "social_cognition": {"affective_empathy": 0.2},
        },
    },

    "emotional_dysregulation": {
        "name": "Emotional Dysregulation",
        "description": "情绪失调 — 情绪调节能力严重不足",
        "subsystem_overrides": {
            "neurotransmitter": {"dopamine": 0.5, "serotonin": 0.2, "gaba": 0.15, "norepinephrine": 0.7},
            "mood_system": {"volatility_mult": 3.5},
            "emotion_regulation": {"regulation_capacity": 0.1, "inhibition": 0.1},
            "predictive_coding": {"precision_mult": 1.5},
            "limbic": {"arousal": 0.7},
            "prefrontal": {"maturity": 0.15, "inhibition": 0.1},
            "hpa_axis": {"stress_reactivity_mult": 2.0, "feedback_strength_mult": 0.3},
            "ans": {"baseline_vagal_tone": 0.2, "sympathetic_reactivity_mult": 1.8},
        },
    },
}


# ===== 模拟器主类 =====

class PsychiatricConditionSimulator:
    """精神疾病与情绪状态模拟器

    核心原理: 精神疾病 = 现有 30+ 子系统的特定参数配置组合
    不是新代码，而是让现有动力学自然涌现病理行为。

    用法:
        sim = PsychiatricConditionSimulator(agent)
        sim.apply_condition("MDD", severity="moderate", onset="gradual")
        # 运行 step() 若干步后，agent 自然涌现抑郁行为
        sim.remove_condition("MDD", offset="slow_recovery")
    """

    def __init__(self, agent=None, event_bus=None):
        self._agent = agent
        self._bus = event_bus or (agent.bus if agent else None)
        self.active_conditions: dict[str, ActiveCondition] = {}
        self._step_count = 0

    # ===== 应用条件 =====

    def apply_condition(
        self,
        condition_id: str,
        severity: str = "moderate",
        onset: str = "gradual",
    ) -> bool:
        """应用精神疾病或情绪状态 profile

        Args:
            condition_id: 疾病/状态 ID (如 "MDD", "emotional_blunting")
            severity: "mild" / "moderate" / "severe"
            onset: "instant" / "rapid" / "gradual" / "insidious"

        Returns:
            是否成功应用
        """
        # 查找 profile
        profile = PSYCHIATRIC_PROFILES.get(condition_id) or EMOTION_STATE_PROFILES.get(condition_id)
        if profile is None:
            print(f"[PsychSim] Unknown condition: {condition_id}")
            return False

        # 如果已存在同条件，先移除
        if condition_id in self.active_conditions:
            self.remove_condition(condition_id, offset="instant")

        # 解析参数
        sev = Severity(severity)
        onset_mode = OnsetMode(onset)
        profile_onset = profile.get("onset_default", onset)
        if onset == "gradual":
            onset_mode = OnsetMode(profile_onset)

        # 保存 baseline 快照
        baseline = self._snapshot_subsystem_state(profile["subsystem_overrides"])

        # 创建活跃条件
        condition = ActiveCondition(
            condition_id=condition_id,
            severity=sev,
            onset_mode=onset_mode,
            offset_mode=OffsetMode(profile.get("offset_default", "slow_recovery")),
            progress=0.0,
            is_onsetting=True,
            is_active=True,
            applied_overrides=profile["subsystem_overrides"],
            baseline_snapshot=baseline,
        )
        self.active_conditions[condition_id] = condition

        # 如果是 instant onset，立即应用
        if onset_mode == OnsetMode.INSTANT:
            condition.progress = 1.0
            self._apply_overrides(condition)

        # 发布事件
        if self._bus is not None:
            self._bus.publish(
                PSYCHIATRIC_CONDITION_CHANGE,
                {
                    "action": "apply",
                    "condition_id": condition_id,
                    "severity": severity,
                    "onset": onset,
                    "progress": condition.progress,
                },
                source="psychiatric_simulator",
            )

        print(f"[PsychSim] Applied {profile['name']} (severity={severity}, onset={onset})")
        return True

    def remove_condition(
        self,
        condition_id: str,
        offset: str = None,
    ) -> bool:
        """移除精神疾病效果

        Args:
            condition_id: 条件 ID
            offset: "instant" / "treatment_response" / "natural_remission" / "slow_recovery"
        """
        if condition_id not in self.active_conditions:
            print(f"[PsychSim] Condition not active: {condition_id}")
            return False

        condition = self.active_conditions[condition_id]

        if offset is not None:
            condition.offset_mode = OffsetMode(offset)

        if condition.offset_mode == OffsetMode.INSTANT:
            # 立即恢复 baseline
            self._restore_baseline(condition)
            del self.active_conditions[condition_id]
        else:
            # 开始渐变恢复
            condition.is_onsetting = False

        # 发布事件
        if self._bus is not None:
            self._bus.publish(
                PSYCHIATRIC_CONDITION_CHANGE,
                {
                    "action": "remove",
                    "condition_id": condition_id,
                    "offset": offset or condition.offset_mode.value,
                    "progress": condition.progress,
                },
                source="psychiatric_simulator",
            )

        print(f"[PsychSim] Removing {condition_id} (offset={offset or condition.offset_mode.value})")
        return True

    def apply_treatment(self, drug_preset: str, condition_id: str = None) -> bool:
        """应用药物治疗

        利用 neuro_pharmacology 的药物预设来对抗疾病偏移。
        如果指定 condition_id，则加速该条件的 offset。
        否则对所有活跃条件应用。

        Args:
            drug_preset: 药物预设名 (如 "antidepressant", "stimulant")
            condition_id: 目标条件 ID (None=全部)
        """
        if condition_id is not None:
            if condition_id not in self.active_conditions:
                return False
            condition = self.active_conditions[condition_id]
            profile = PSYCHIATRIC_PROFILES.get(condition_id, {})
            treatment_response = profile.get("treatment_responses", {}).get(drug_preset, 0.3)
            # 加速 offset
            condition.offset_mode = OffsetMode.TREATMENT_RESPONSE
            condition.is_onsetting = False
            # 治疗响应系数影响 offset 速度
            condition._treatment_speedup = treatment_response
            print(f"[PsychSim] Treatment '{drug_preset}' applied to {condition_id} "
                  f"(response={treatment_response:.1f})")
        else:
            for cid, cond in self.active_conditions.items():
                profile = PSYCHIATRIC_PROFILES.get(cid, {})
                treatment_response = profile.get("treatment_responses", {}).get(drug_preset, 0.3)
                if treatment_response > 0.3:
                    cond.offset_mode = OffsetMode.TREATMENT_RESPONSE
                    cond.is_onsetting = False
                    cond._treatment_speedup = treatment_response

        return True

    # ===== 每步推进 =====

    def step(self) -> dict[str, Any]:
        """每步推进所有活跃条件的 onset/offset 渐变

        渐变公式:
            current = baseline + (target - baseline) * progress * severity_mult

        Returns:
            各条件的当前状态摘要
        """
        self._step_count += 1
        summary = {}

        to_remove = []
        for cid, condition in self.active_conditions.items():
            if condition.is_onsetting:
                # 发病中: progress → 1.0
                speed = ONSET_SPEEDS.get(condition.onset_mode, 0.02)
                condition.progress = min(1.0, condition.progress + speed)
            else:
                # 恢复中: progress → 0.0
                speed = OFFSET_SPEEDS.get(condition.offset_mode, 0.01)
                # 治疗加速
                treatment_speedup = getattr(condition, '_treatment_speedup', 1.0)
                effective_speed = speed * (1.0 + treatment_speedup)
                condition.progress = max(0.0, condition.progress - effective_speed)

                if condition.progress <= 0.0:
                    # 完全恢复
                    self._restore_baseline(condition)
                    to_remove.append(cid)
                    continue

            # 应用当前进度的偏移
            self._apply_overrides(condition)
            condition.step_count += 1

            # 记录摘要
            sev_mult = SEVERITY_MULTIPLIERS.get(condition.severity, 0.6)
            summary[cid] = {
                "progress": condition.progress,
                "severity": condition.severity.value,
                "effective_strength": condition.progress * sev_mult,
                "phase": "onset" if condition.is_onsetting else "offset",
                "steps": condition.step_count,
            }

        # 清除已恢复的条件
        for cid in to_remove:
            del self.active_conditions[cid]
            print(f"[PsychSim] {cid} fully recovered")

        return summary

    # ===== 内部方法 =====

    def _apply_overrides(self, condition: ActiveCondition):
        """将 profile overrides 应用到 agent 的 internal_state

        插值公式: current = baseline + (target - baseline) * progress * severity_mult
        共病处理: 同一 key 取偏离 baseline 最远的值
        """
        if self._agent is None:
            return

        sev_mult = SEVERITY_MULTIPLIERS.get(condition.severity, 0.6)
        effective = condition.progress * sev_mult

        for subsystem, overrides in condition.applied_overrides.items():
            baseline = condition.baseline_snapshot.get(subsystem, {})
            for key, target_val in overrides.items():
                base_val = baseline.get(key, 0.5)  # 默认 baseline 为 0.5

                # 插值
                if isinstance(target_val, list):
                    # 向量值 (如 mood mean_shift)
                    current = [
                        b + (t - b) * effective
                        for b, t in zip([base_val] * len(target_val), target_val)
                    ]
                else:
                    current = base_val + (target_val - base_val) * effective

                # 写入 internal_state
                state_key = f"{subsystem}_{key}"
                existing = self._agent._internal_state.get(state_key)

                # 共病处理: 取偏离 0.5 最远的值
                if existing is not None and len(self.active_conditions) > 1:
                    if isinstance(current, (int, float)):
                        if abs(existing - 0.5) > abs(current - 0.5):
                            continue  # 保留更极端的值

                self._agent._internal_state[state_key] = current

        # 合并别名键到规范键 (如 neurotransmitter_dopamine → nt_dopamine)
        try:
            from core.state_key_mapping import UnifiedStateMapping
            UnifiedStateMapping.harmonize_dict(self._agent._internal_state)
        except ImportError:
            pass

    def _snapshot_subsystem_state(self, overrides: dict) -> dict[str, dict[str, float]]:
        """保存当前子系统状态作为 baseline"""
        if self._agent is None:
            return {}

        snapshot = {}
        for subsystem, keys in overrides.items():
            snapshot[subsystem] = {}
            for key in keys:
                state_key = f"{subsystem}_{key}"
                snapshot[subsystem][key] = self._agent._internal_state.get(state_key, 0.5)
        return snapshot

    def _restore_baseline(self, condition: ActiveCondition):
        """恢复 baseline 状态"""
        if self._agent is None:
            return

        for subsystem, baseline in condition.baseline_snapshot.items():
            for key, value in baseline.items():
                state_key = f"{subsystem}_{key}"
                self._agent._internal_state[state_key] = value

    # ===== 监测涌现行为 =====

    def monitor_emergent_behavior(self) -> dict[str, dict[str, Any]]:
        """监测各活跃条件的涌现行为指标

        检查 agent 的 internal_state 是否表现出预期的病理特征。

        Returns:
            {condition_id: {metric: value, expected: range, match: bool}}
        """
        if self._agent is None:
            return {}

        results = {}
        s = self._agent._internal_state

        for cid, condition in self.active_conditions.items():
            metrics = {}

            if cid == "MDD" or cid == "bipolar_depression" or cid == "Dysthymia":
                metrics["mood_valence"] = {
                    "value": s.get("mood_system_mean_shift", [0])[0] if isinstance(s.get("mood_system_mean_shift"), list) else s.get("limbic_valence", 0),
                    "expected": "< -0.2",
                    "match": s.get("limbic_valence", 0) < -0.2,
                }
                metrics["hrv"] = {
                    "value": s.get("ans_baseline_vagal_tone", 0.5),
                    "expected": "< 0.35",
                    "match": s.get("ans_baseline_vagal_tone", 0.5) < 0.35,
                }

            elif cid == "bipolar_mania":
                metrics["mood_activation"] = {
                    "value": s.get("limbic_arousal", 0.5),
                    "expected": "> 0.7",
                    "match": s.get("limbic_arousal", 0.5) > 0.7,
                }
                metrics["impulsivity"] = {
                    "value": s.get("prefrontal_inhibition", 0.5),
                    "expected": "< 0.2",
                    "match": s.get("prefrontal_inhibition", 0.5) < 0.2,
                }

            elif cid == "GAD" or cid == "Social_Anxiety":
                metrics["cortisol"] = {
                    "value": s.get("hormone_cortisol", 0.3),
                    "expected": "> 0.5",
                    "match": s.get("hormone_cortisol", 0.3) > 0.5,
                }
                metrics["sympathetic"] = {
                    "value": s.get("ans_sympathetic_reactivity_mult", 1.0),
                    "expected": "> 1.4",
                    "match": s.get("ans_sympathetic_reactivity_mult", 1.0) > 1.4,
                }

            elif cid == "BPD":
                metrics["emotion_volatility"] = {
                    "value": s.get("mood_system_volatility_mult", 1.0),
                    "expected": "> 3.0",
                    "match": s.get("mood_system_volatility_mult", 1.0) > 3.0,
                }
                metrics["self_coherence"] = {
                    "value": s.get("self_awareness_coherence", 0.7),
                    "expected": "< 0.3",
                    "match": s.get("self_awareness_coherence", 0.7) < 0.3,
                }

            elif cid == "schizophrenia_positive":
                metrics["aberrant_precision"] = {
                    "value": s.get("predictive_coding_precision_mult", 1.0),
                    "expected": "> 1.8",
                    "match": s.get("predictive_coding_precision_mult", 1.0) > 1.8,
                }
                metrics["self_boundary"] = {
                    "value": s.get("self_awareness_self_boundary", 0.7),
                    "expected": "< 0.3",
                    "match": s.get("self_awareness_self_boundary", 0.7) < 0.3,
                }

            elif cid == "PTSD":
                metrics["hyperarousal"] = {
                    "value": s.get("brainstem_arousal_setpoint", 0.5),
                    "expected": "> 0.7",
                    "match": s.get("brainstem_arousal_setpoint", 0.5) > 0.7,
                }
                metrics["threat_response"] = {
                    "value": s.get("hpa_axis_stress_reactivity_mult", 1.0),
                    "expected": "> 1.8",
                    "match": s.get("hpa_axis_stress_reactivity_mult", 1.0) > 1.8,
                }

            results[cid] = metrics

        return results

    # ===== 查询接口 =====

    def get_active_conditions(self) -> list[str]:
        """获取当前活跃的条件列表"""
        return list(self.active_conditions.keys())

    def get_condition_state(self, condition_id: str) -> dict | None:
        """获取特定条件的当前状态"""
        if condition_id not in self.active_conditions:
            return None
        c = self.active_conditions[condition_id]
        sev_mult = SEVERITY_MULTIPLIERS.get(c.severity, 0.6)
        return {
            "condition_id": c.condition_id,
            "severity": c.severity.value,
            "progress": c.progress,
            "effective_strength": c.progress * sev_mult,
            "phase": "onset" if c.is_onsetting else "offset",
            "steps": c.step_count,
            "onset_mode": c.onset_mode.value,
            "offset_mode": c.offset_mode.value,
        }

    def get_available_conditions(self) -> dict[str, str]:
        """获取所有可用的条件 ID 和名称"""
        result = {}
        for cid, profile in PSYCHIATRIC_PROFILES.items():
            result[cid] = profile["name"]
        for cid, profile in EMOTION_STATE_PROFILES.items():
            result[cid] = profile["name"]
        return result

    def get_comorbidities(self, condition_id: str) -> list[str]:
        """获取某条件的常见共病"""
        profile = PSYCHIATRIC_PROFILES.get(condition_id, {})
        return profile.get("comorbidities", [])

    def get_treatment_responses(self, condition_id: str) -> dict[str, float]:
        """获取某条件的治疗响应"""
        profile = PSYCHIATRIC_PROFILES.get(condition_id, {})
        return profile.get("treatment_responses", {})
