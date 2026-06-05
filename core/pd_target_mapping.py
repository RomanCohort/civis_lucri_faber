"""PD靶点映射 — 将药物 target_neurotransmitters 分配为具体PD效应。

每种药物有多个神经递质靶点，其效应强度不同。此模块将
DrugDefinition.target_neurotransmitters 转换为 PDTarget 列表，
供 TherapeuticExperiment 在每步计算 PD 效应时使用。

效应分配规则:
  - 主靶点: effect_strength = 0.80
  - 次靶点 (slight_*): effect_strength = 0.20
  - 抑制靶点 (decrease): effect_strength = 0.90, type="decrease"
  - 多靶点均分余量

参考文献:
  - Stahl's Essential Psychopharmacology (2021) — 靶点效价分配
"""

from __future__ import annotations

from dataclasses import dataclass

from core.state_key_mapping import UnifiedStateMapping


@dataclass
class PDTarget:
    """一个神经递质靶点的PD效应描述。"""
    neurotransmitter: str       # 通用名 "serotonin"
    canonical_key: str          # 规范状态键 "nt_serotonin"
    effect_type: str            # "increase" / "decrease"
    effect_strength: float      # 0-1, 该靶点占总效应的比例
    ec50_modifier: float = 1.0  # 对全局EC50的修正倍率
    baseline: float = 0.5       # 基线NT水平


# ── 效应强度分配规则 ──
_EFFECT_STRENGTH_RULES: dict[str, float] = {
    "increase":         0.80,   # 主激动效应
    "slight_increase":  0.20,   # 轻微激动
    "moderate":         0.50,   # 中等效应
    "decrease":         0.90,   # 主抑制效应
    "slight_decrease":  0.15,   # 轻微抑制
}

# 各药物类别的默认基线NT水平 (治疗前的"病态"基线)
_DRUG_CLASS_BASELINES: dict[str, dict[str, float]] = {
    "SSRI": {
        "serotonin": 0.15,      # 抑郁症5-HT偏低
        "dopamine":  0.20,
    },
    "stimulant": {
        "dopamine":       0.20,
        "norepinephrine": 0.25,
    },
    "sedative": {
        "gaba":  0.25,
    },
    "antipsychotic": {
        "dopamine": 0.70,       # 精神分裂症DA偏高
    },
    "mood_stabilizer": {
        "serotonin": 0.30,
        "dopamine":  0.50,
        "gaba":      0.25,
    },
    "hallucinogen": {
        "glutamate": 0.50,
        "dopamine":  0.30,
    },
    "opioid": {
        "dopamine": 0.20,
        "gaba":     0.50,
    },
    "default": {},
}


def build_pd_targets(
    target_neurotransmitters: dict[str, str],
    drug_class: str = "default",
) -> list[PDTarget]:
    """Build PD target list from drug's target_neurotransmitters dict.

    Args:
        target_neurotransmitters: e.g. {"serotonin": "increase", "dopamine": "slight_increase"}
        drug_class: for baseline lookup

    Returns:
        List of PDTarget with normalized effect_strengths (sum to 1.0)
    """
    baselines = _DRUG_CLASS_BASELINES.get(drug_class, _DRUG_CLASS_BASELINES["default"])
    raw_targets: list[PDTarget] = []

    for nt_name, effect_type_raw in target_neurotransmitters.items():
        # 解析 effect_type
        effect_type, ec50_mod = _parse_effect_type(effect_type_raw)

        # 获取效应强度
        strength = _EFFECT_STRENGTH_RULES.get(effect_type_raw, 0.50)

        # 解析规范键
        canonical_key = UnifiedStateMapping.resolve(nt_name)

        # 基线
        baseline = baselines.get(nt_name, 0.50)

        raw_targets.append(PDTarget(
            neurotransmitter=nt_name,
            canonical_key=canonical_key,
            effect_type=effect_type,
            effect_strength=strength,
            ec50_modifier=ec50_mod,
            baseline=baseline,
        ))

    # 归一化: 所有靶点的 effect_strength 之和 = 1.0
    total = sum(t.effect_strength for t in raw_targets)
    if total > 0:
        for t in raw_targets:
            t.effect_strength /= total

    return raw_targets


def _parse_effect_type(effect_str: str) -> tuple:
    """Parse effect string to (type, ec50_modifier)."""
    s = effect_str.lower().strip()

    if s in ("increase", "slight_increase"):
        return ("increase", 1.0)
    elif s in ("decrease", "slight_decrease"):
        return ("decrease", 1.0)
    elif s == "moderate":
        return ("increase", 1.5)     # 中等效应需更高浓度
    else:
        return ("increase", 1.0)


def compute_pd_deltas(
    targets: list[PDTarget],
    global_effect: float,
) -> dict[str, float]:
    """Distribute global PD effect across targets.

    Args:
        targets: List of PDTarget with normalized strengths
        global_effect: E = Emax * C^Hill / (EC50^Hill + C^Hill), range [0, Emax]

    Returns:
        Dict of {canonical_key: delta_value}
        delta > 0 means increase, delta < 0 means decrease
    """
    deltas: dict[str, float] = {}

    for target in targets:
        # 该靶点的效应份额
        target_effect = global_effect * target.effect_strength

        # 方向: increase → positive delta, decrease → negative delta
        if target.effect_type == "decrease":
            delta = -target_effect
        else:
            delta = target_effect

        # 累加 (同一NT可能被多个靶点影响)
        current = deltas.get(target.canonical_key, 0.0)
        deltas[target.canonical_key] = current + delta

    return deltas


__all__ = [
    "PDTarget",
    "build_pd_targets",
    "compute_pd_deltas",
]
