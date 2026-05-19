"""统一状态键映射 — 解决 _internal_state 命名不一致问题。

问题:
  1. 疾病档案用 neurotransmitter_dopamine，药物用 nt_dopamine
  2. GABA 和 NE 都代理到 ans_hrv，无法独立操控

方案:
  - 所有神经递质统一为 nt_* 前缀
  - 新增 nt_gaba 和 nt_norepinephrine 独立键
  - ans_hrv 改为派生计算: f(nt_gaba, nt_norepinephrine)
  - harmonize_dict() 合并别名键到规范键
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class UnifiedStateMapping:
    """Canonical state key registry for CLF _internal_state."""

    # 规范键 — 所有模块应使用这些键
    CANONICAL: Dict[str, str] = {
        "dopamine":       "nt_dopamine",
        "serotonin":      "nt_serotonin",
        "norepinephrine": "nt_norepinephrine",
        "gaba":           "nt_gaba",
        "glutamate":      "nt_glutamate",
        "acetylcholine":  "nt_acetylcholine",
        "cortisol":       "cortisol_level",
        "bdnf":           "plasticity_bdnf",
        "oxytocin":       "hormone_oxytocin",
        "adrenaline":     "hormone_adrenaline",
        "melatonin":      "scn_melatonin",
        # ── 受体亚型键 ──
        "5ht1a":      "rct_5ht1a",
        "5ht2a":      "rct_5ht2a",
        "5ht2c":      "rct_5ht2c",
        "5ht3":       "rct_5ht3",
        "sert":       "rct_sert",
        "d1":         "rct_d1",
        "d2":         "rct_d2",
        "d3":         "rct_d3",
        "dat":        "rct_dat",
        "gaba_a":     "rct_gabaa",
        "gaba_b":     "rct_gabab",
        "nmda":       "rct_nmda",
        "ampa":       "rct_ampa",
        "mglur2":     "rct_mglur2",
        "mglur5":     "rct_mglur5",
        "alpha1":     "rct_alpha1",
        "alpha2":     "rct_alpha2",
        "beta1":      "rct_beta1",
        "net":        "rct_net",
        "mu_opioid":  "rct_muopioid",
        "delta":      "rct_delta",
        "kappa":      "rct_kappa",
        "nachr":      "rct_nachr",
        "machr_m1":   "rct_machrm1",
        "orexin1":    "rct_orexin1",
        "orexin2":    "rct_orexin2",
        # ── 成瘾/症状键 ──
        "orexin_level":       "orexin_level",
        "craving_level":      "craving_level",
        "withdrawal_severity":"withdrawal_severity",
        "tolerance_factor":   "tolerance_factor",
        "pathogen_load":      "pathogen_load",
        "bbb_disruption":     "bbb_disruption",
        "interoceptive_pe":   "interoceptive_pe",
    }

    # 所有已知别名 → 规范键
    ALIASES: Dict[str, str] = {
        # 神经递质 — 疾病档案格式
        "neurotransmitter_dopamine":      "nt_dopamine",
        "neurotransmitter_serotonin":     "nt_serotonin",
        "neurotransmitter_norepinephrine": "nt_norepinephrine",
        "neurotransmitter_gaba":         "nt_gaba",
        "neurotransmitter_glutamate":    "nt_glutamate",
        "neurotransmitter_acetylcholine": "nt_acetylcholine",
        # 缩写
        "DA":  "nt_dopamine",
        "5-HT": "nt_serotonin",
        "5ht": "nt_serotonin",
        "NE":  "nt_norepinephrine",
        "ACh": "nt_acetylcholine",
        # 激素别名
        "hormone_cortisol":  "cortisol_level",
        "neurotransmitter_bdnf": "plasticity_bdnf",
        # 旧代理键 (保留映射但不推荐写入)
        # "gaba" 旧映射到 "ans_hrv" — 现在映射到独立键
        # ── 受体亚型别名 ──
        "5-HT1A":    "rct_5ht1a",
        "5-HT2A":    "rct_5ht2a",
        "5-HT2C":    "rct_5ht2c",
        "5-HT3":     "rct_5ht3",
        "mu-opioid": "rct_muopioid",
        "GABA-A":    "rct_gabaa",
        "GABA-B":    "rct_gabab",
        "nAChR":     "rct_nachr",
        "mAChR-M1":  "rct_machrm1",
        "Orexin-1":  "rct_orexin1",
        "Orexin-2":  "rct_orexin2",
    }

    # 反向映射: 规范键 → 所有别名 (含自身)
    _REVERSE: Dict[str, List[str]] = {}

    @classmethod
    def _build_reverse(cls) -> None:
        if cls._REVERSE:
            return
        rev: Dict[str, List[str]] = {}
        for canonical in cls.CANONICAL.values():
            rev.setdefault(canonical, [])
        for alias, canonical in cls.ALIASES.items():
            rev.setdefault(canonical, []).append(alias)
        for canonical in cls.CANONICAL.values():
            if canonical not in rev.get(canonical, []):
                rev[canonical].append(canonical)
        cls._REVERSE = rev

    @classmethod
    def resolve(cls, name: str) -> str:
        """Resolve any variant name to canonical key.

        Examples:
            resolve("neurotransmitter_dopamine") → "nt_dopamine"
            resolve("DA") → "nt_dopamine"
            resolve("nt_dopamine") → "nt_dopamine"
            resolve("unknown_key") → "unknown_key"  (passthrough)
        """
        if name in cls.CANONICAL.values():
            return name
        lower = name.lower().replace("-", "").replace("_", "").replace(" ", "")
        # 先查 CANONICAL (简短名)
        for short, canonical in cls.CANONICAL.items():
            if lower == short.lower().replace("-", "").replace("_", ""):
                return canonical
        # 再查 ALIASES
        for alias, canonical in cls.ALIASES.items():
            if lower == alias.lower().replace("-", "").replace("_", "").replace(" ", ""):
                return canonical
        return name

    @classmethod
    def harmonize_dict(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fix a state dict: merge variant keys into canonical ones.

        If both 'neurotransmitter_dopamine' and 'nt_dopamine' exist,
        take the value furthest from 0.5 (most extreme) and store
        under 'nt_dopamine'. Remove the variant key.

        Returns the same dict (modified in-place and returned).
        """
        cls._build_reverse()
        for canonical, aliases in cls._REVERSE.items():
            values = []
            if canonical in state:
                values.append((canonical, state[canonical]))
            for alias in aliases:
                if alias in state and alias != canonical:
                    values.append((alias, state[alias]))
            if len(values) <= 1:
                continue
            # 选择离0.5最远的值 (最极端的)
            best_key, best_val = max(values, key=lambda kv: abs(_to_float(kv[1]) - 0.5))
            state[canonical] = best_val
            # 删除别名键
            for key, _ in values:
                if key != canonical and key in state:
                    del state[key]
        return state

    @classmethod
    def derive_ans_hrv(cls, state: Dict[str, Any]) -> float:
        """Compute ans_hrv from dedicated nt_gaba and nt_norepinephrine keys.

        Formula: ans_hrv = 0.3 + 0.4 * nt_gaba - 0.2 * nt_norepinephrine
        Clipped to [0.05, 0.95].

        This replaces the old proxy where both GABA and NE mapped to
        ans_hrv directly with opposing effects.
        """
        gaba = _to_float(state.get("nt_gaba", 0.5))
        ne = _to_float(state.get("nt_norepinephrine", 0.3))
        hrv = 0.3 + 0.4 * gaba - 0.2 * ne
        return max(0.05, min(0.95, hrv))

    @classmethod
    def update_derived_keys(cls, state: Dict[str, Any]) -> None:
        """Update all derived keys in state dict after NT changes.

        Currently only ans_hrv is derived. Call this after any
        modification to nt_gaba or nt_norepinephrine.
        """
        state["ans_hrv"] = cls.derive_ans_hrv(state)

    @classmethod
    def all_canonical_nt_keys(cls) -> List[str]:
        """Return all canonical neurotransmitter state keys."""
        return [v for k, v in cls.CANONICAL.items()
                if v.startswith("nt_")]


def _to_float(val: Any) -> float:
    """Safely convert a value to float for comparison."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.5


__all__ = [
    "UnifiedStateMapping",
]