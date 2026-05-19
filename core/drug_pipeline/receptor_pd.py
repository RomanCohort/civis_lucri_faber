"""受体亚型药效动力学 — Receptor-subtype pharmacodynamics.

将药物效应从神经递质层面下沉到受体亚型层面:
  旧: sertraline → "serotonin: increase"
  新: sertraline → SERT inhibition(+1.0, Ki=0.14nM) → 5-HT突触间隙↑

这使得:
  - 同一NT的不同受体亚型可以有相反效应 (5-HT1A自受体 vs 5-HT2A)
  - DDI可以在受体层面建模 (buspirone 5-HT1A部分激动 + SSRI SERT抑制)
  - 耐受可以在受体层面建模 (mu-opioid下调)

Ki数据来源: PDSP Ki Database (UNC), IUPHAR Guide to Pharmacology
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────
# 受体亚型定义
# ──────────────────────────────────────────────────────

@dataclass
class ReceptorSubtype:
    """受体亚型 — 药理学的最小作用单元。"""
    name: str                    # "5-HT1A", "D2", "GABA-A", "NMDA"...
    neurotransmitter: str        # 父NT: "serotonin", "dopamine"...
    coupling: str                # "Gs"/"Gi"/"Gq"/"ionotropic"/"transporter"/"pam"
    signal_direction: float      # 激活时对父NT净效应方向: +1.0=增强, -1.0=抑制
    density_weight: float        # 对父NT总效应的贡献权重 (0-1, 同NT下归一化)
    is_autoreceptor: bool = False  # 自受体: 激活时抑制自身NT释放
    location: str = "postsynaptic" # "presynaptic"/"postsynaptic"/"somatodendritic"


# 25个关键受体亚型注册表
RECEPTOR_REGISTRY: Dict[str, ReceptorSubtype] = {
    # ── 5-HT系 ──
    "5-HT1A": ReceptorSubtype(
        "5-HT1A", "serotonin", "Gi", -1.0, 0.20,
        is_autoreceptor=True, location="somatodendritic",
    ),
    "5-HT2A": ReceptorSubtype(
        "5-HT2A", "serotonin", "Gq", +1.0, 0.20,
    ),
    "5-HT2C": ReceptorSubtype(
        "5-HT2C", "serotonin", "Gq", +0.7, 0.10,
    ),
    "5-HT3": ReceptorSubtype(
        "5-HT3", "serotonin", "ionotropic", +0.7, 0.10,
    ),
    "SERT": ReceptorSubtype(
        "SERT", "serotonin", "transporter", +1.0, 0.30,
    ),
    # ── DA系 ──
    "D1": ReceptorSubtype(
        "D1", "dopamine", "Gs", +1.0, 0.25,
    ),
    "D2": ReceptorSubtype(
        "D2", "dopamine", "Gi", -1.0, 0.25,
        is_autoreceptor=True, location="presynaptic",
    ),
    "D3": ReceptorSubtype(
        "D3", "dopamine", "Gi", -0.5, 0.10,
    ),
    "DAT": ReceptorSubtype(
        "DAT", "dopamine", "transporter", +1.0, 0.30,
    ),
    # ── GABA系 ──
    "GABA-A": ReceptorSubtype(
        "GABA-A", "gaba", "ionotropic", +1.0, 0.60,
    ),
    "GABA-B": ReceptorSubtype(
        "GABA-B", "gaba", "Gi", +0.8, 0.30,
    ),
    # ── Glu系 ──
    "NMDA": ReceptorSubtype(
        "NMDA", "glutamate", "ionotropic", +1.0, 0.30,
    ),
    "AMPA": ReceptorSubtype(
        "AMPA", "glutamate", "ionotropic", +0.8, 0.25,
    ),
    "mGluR2": ReceptorSubtype(
        "mGluR2", "glutamate", "Gi", -0.5, 0.10,
        is_autoreceptor=True, location="presynaptic",
    ),
    "mGluR5": ReceptorSubtype(
        "mGluR5", "glutamate", "Gq", +0.7, 0.15,
    ),
    # ── NE系 ──
    "alpha1": ReceptorSubtype(
        "alpha1", "norepinephrine", "Gq", +1.0, 0.20,
    ),
    "alpha2": ReceptorSubtype(
        "alpha2", "norepinephrine", "Gi", -1.0, 0.20,
        is_autoreceptor=True, location="presynaptic",
    ),
    "beta1": ReceptorSubtype(
        "beta1", "norepinephrine", "Gs", +1.0, 0.20,
    ),
    "NET": ReceptorSubtype(
        "NET", "norepinephrine", "transporter", +1.0, 0.30,
    ),
    # ── Opioid系 ──
    "mu-opioid": ReceptorSubtype(
        "mu-opioid", "endorphin", "Gi", +0.8, 0.40,
    ),
    "delta": ReceptorSubtype(
        "delta", "endorphin", "Gi", +0.5, 0.25,
    ),
    "kappa": ReceptorSubtype(
        "kappa", "endorphin", "Gi", -0.3, 0.20,
    ),
    # ── ACh系 ──
    "nAChR": ReceptorSubtype(
        "nAChR", "acetylcholine", "ionotropic", +1.0, 0.40,
    ),
    "mAChR-M1": ReceptorSubtype(
        "mAChR-M1", "acetylcholine", "Gq", +0.8, 0.35,
    ),
    # ── Orexin系 ──
    "orexin1": ReceptorSubtype(
        "orexin1", "orexin", "Gq", +1.0, 0.50,
    ),
    "orexin2": ReceptorSubtype(
        "orexin2", "orexin", "Gq", +1.0, 0.50,
    ),
}


# ──────────────────────────────────────────────────────
# 药物-受体亲和力图谱 (Ki, nM)
# ──────────────────────────────────────────────────────

# Ki越小亲和力越强; effect_type描述药物在该位点的药理作用
DRUG_RECEPTOR_AFFINITY: Dict[str, Dict[str, Dict]] = {
    # ── SSRI ──
    "sertraline": {
        "SERT":   {"ki_nm": 0.14, "effect_type": "antagonist"},    # SERT抑制
        "5-HT2A": {"ki_nm": 325,  "effect_type": "antagonist"},    # 弱5-HT2A拮抗
        "DAT":    {"ki_nm": 25,   "effect_type": "antagonist"},    # 弱DAT抑制
    },
    "fluoxetine": {
        "SERT":   {"ki_nm": 1.0,  "effect_type": "antagonist"},
        "5-HT2C": {"ki_nm": 8,    "effect_type": "antagonist"},
        "NET":    {"ki_nm": 1000, "effect_type": "antagonist"},    # 极弱
    },
    "paroxetine": {
        "SERT":   {"ki_nm": 0.06, "effect_type": "antagonist"},    # 最强SERT
        "NET":    {"ki_nm": 45,   "effect_type": "antagonist"},
    },
    # ── 5-HT1A部分激动剂 ──
    "buspirone": {
        "5-HT1A": {"ki_nm": 21,   "effect_type": "partial_agonist", "intrinsic_activity": 0.4},
        "D2":     {"ki_nm": 100,  "effect_type": "antagonist"},
    },
    # ── 抗精神病 ──
    "haloperidol": {
        "D2":     {"ki_nm": 0.5,  "effect_type": "antagonist"},
        "5-HT2A": {"ki_nm": 30,   "effect_type": "antagonist"},
        "alpha1": {"ki_nm": 10,   "effect_type": "antagonist"},
    },
    # ── 阿片类 ──
    "morphine": {
        "mu-opioid": {"ki_nm": 1.8,  "effect_type": "agonist"},
        "delta":     {"ki_nm": 200,  "effect_type": "agonist"},
        "kappa":     {"ki_nm": 50,   "effect_type": "agonist"},
    },
    "buprenorphine": {
        "mu-opioid": {"ki_nm": 0.8,  "effect_type": "partial_agonist", "intrinsic_activity": 0.6},
        "kappa":     {"ki_nm": 1.5,  "effect_type": "antagonist"},     # kappa拮抗!
    },
    # ── 苯二氮卓 ──
    "diazepam": {
        "GABA-A": {"ki_nm": 0.05, "effect_type": "pam"},  # 正向别构调节
    },
    # ── NMDA拮抗剂 ──
    "ketamine": {
        "NMDA": {"ki_nm": 0.5,  "effect_type": "antagonist"},
        "AMPA": {"ki_nm": 5000, "effect_type": "antagonist"},  # 极弱
    },
    # ── 兴奋剂 ──
    "amphetamine": {
        "DAT":    {"ki_nm": 0.1, "effect_type": "agonist"},   # 促释放+抑制再摄取
        "NET":    {"ki_nm": 0.1, "effect_type": "agonist"},
        "SERT":   {"ki_nm": 50,  "effect_type": "agonist"},   # 弱5-HT效应
        "5-HT2A": {"ki_nm": 200, "effect_type": "agonist"},
    },
    # ── β-阻滞剂 ──
    "propranolol": {
        "beta1":  {"ki_nm": 1.5, "effect_type": "antagonist"},
        "beta2":  {"ki_nm": 1.0, "effect_type": "antagonist"},
        "5-HT1A": {"ki_nm": 10,  "effect_type": "antagonist"},
    },
    # ── Orexin受体拮抗剂 (DORA) ──
    "suvorexant": {
        "orexin1": {"ki_nm": 5.0, "effect_type": "antagonist"},
        "orexin2": {"ki_nm": 5.0, "effect_type": "antagonist"},
    },
    # ── 抗炎抗生素 (TLR2抑制) ──
    "minocycline": {},  # 无直接受体亲和力, 作用于TLR2/小胶质细胞
    # ── 锂盐 ──
    "lithium": {},      # 离子通道调节, 无经典受体
}


# ──────────────────────────────────────────────────────
# 受体PD目标
# ──────────────────────────────────────────────────────

@dataclass
class ReceptorPDTarget:
    """受体层面的PD目标 — 替代PDTarget用于有亲和力数据的药物。"""
    receptor_name: str           # "5-HT1A"
    canonical_key: str           # "rct_5ht1a"
    parent_nt_key: str           # "nt_serotonin"
    effect_type: str             # "agonist"/"antagonist"/"partial_agonist"/"pam"/"nam"
    ki_nm: float                 # 亲和力 (nM)
    intrinsic_activity: float = 1.0  # 部分激动剂内在活性 (0-1)
    signal_direction: float = 0.0    # 从ReceptorSubtype继承
    density_weight: float = 0.0      # 从ReceptorSubtype继承
    is_autoreceptor: bool = False


def build_receptor_pd_targets(
    drug_name: str,
    drug_effect: float,
) -> Tuple[List[ReceptorPDTarget], Dict[str, float]]:
    """构建受体层面的PD目标列表。

    Returns:
        (targets, receptor_deltas) — 目标列表和每个受体的效应份额
    """
    affinity = DRUG_RECEPTOR_AFFINITY.get(drug_name, {})
    if not affinity:
        return [], {}

    # 按亲和力归一化效应份额 (Ki越小→份额越大)
    total_affinity = sum(1.0 / max(info["ki_nm"], 0.001) for info in affinity.values())
    receptor_deltas: Dict[str, float] = {}

    for rct_name, info in affinity.items():
        ki = max(info["ki_nm"], 0.001)
        weight = (1.0 / ki) / total_affinity  # 归一化权重
        effect_share = drug_effect * weight
        receptor_deltas[rct_name] = effect_share

    # 构建ReceptorPDTarget列表
    targets: List[ReceptorPDTarget] = []
    for rct_name, info in affinity.items():
        rct_def = RECEPTOR_REGISTRY.get(rct_name)
        if rct_def is None:
            continue

        canonical = "rct_" + rct_name.lower().replace("-", "").replace(" ", "")
        effect_type = info.get("effect_type", "agonist")
        intrinsic = info.get("intrinsic_activity", 1.0)

        # 计算信号方向
        if effect_type == "antagonist":
            sig_dir = -rct_def.signal_direction  # 拮抗=反转方向
        elif effect_type == "pam":
            sig_dir = +rct_def.signal_direction  # PAM=增强内源配体
        elif effect_type == "nam":
            sig_dir = -rct_def.signal_direction  # NAM=减弱内源配体
        elif effect_type == "partial_agonist":
            sig_dir = rct_def.signal_direction * intrinsic
        else:  # agonist
            sig_dir = rct_def.signal_direction

        targets.append(ReceptorPDTarget(
            receptor_name=rct_name,
            canonical_key=canonical,
            parent_nt_key=f"nt_{rct_def.neurotransmitter}",
            effect_type=effect_type,
            ki_nm=info["ki_nm"],
            intrinsic_activity=intrinsic,
            signal_direction=sig_dir,
            density_weight=rct_def.density_weight,
            is_autoreceptor=rct_def.is_autoreceptor,
        ))

    return targets, receptor_deltas


def compute_receptor_deltas(
    targets: List[ReceptorPDTarget],
    receptor_shares: Dict[str, float],
    tolerance_factors: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """计算每个受体亚型的delta值。

    Args:
        targets: 受体PD目标列表
        receptor_shares: 每个受体的效应份额 {receptor_name: share}
        tolerance_factors: 耐受因子 {receptor_name: factor} (1.0=无耐受, <1=下调)

    Returns:
        {canonical_key: delta_value}
    """
    deltas: Dict[str, float] = {}
    tol = tolerance_factors or {}

    for t in targets:
        share = receptor_shares.get(t.receptor_name, 0.0)
        tol_factor = tol.get(t.receptor_name, 1.0)

        # 受体delta = 份额 × 信号方向 × 耐受因子
        delta = share * t.signal_direction * tol_factor

        # 自受体特殊处理: 急性期自受体激活 → 抑制NT释放
        # 但长期脱敏后此效应减弱 (由tolerance_factor控制)
        if t.is_autoreceptor and t.effect_type in ("agonist", "partial_agonist"):
            # 自受体激动 → 抑制释放 → 负向delta
            # 脱敏后 (tol_factor↓) → 自受体效应减弱 → 释放恢复
            pass  # signal_direction已经包含自受体的-1.0

        deltas[t.canonical_key] = delta

    return deltas


def aggregate_receptor_to_nt(
    receptor_deltas: Dict[str, float],
    targets: List[ReceptorPDTarget],
) -> Dict[str, float]:
    """将受体层面delta聚合到父NT层面。

    按density_weight加权: nt_delta = Σ(rct_delta_i × density_weight_i)
    """
    nt_deltas: Dict[str, float] = {}

    for t in targets:
        rct_delta = receptor_deltas.get(t.receptor_name, 0.0)
        weighted = rct_delta * t.density_weight

        nt_key = t.parent_nt_key
        nt_deltas[nt_key] = nt_deltas.get(nt_key, 0.0) + weighted

    return nt_deltas


def get_receptor_time_profile(
    drug_name: str,
    step: int,
    total_steps: int,
) -> Dict[str, float]:
    """受体效应的时间动态修正。

    某些受体效应有时间依赖性:
    - 5-HT1A自受体: 急性期抑制5-HT释放, 2-4周后脱敏→协同
    - NMDA拮抗: 急性期谷氨酸暴发, 后期mTOR激活→突触可塑性
    """
    modifiers: Dict[str, float] = {}
    affinity = DRUG_RECEPTOR_AFFINITY.get(drug_name, {})

    # 时间进度 (0→1)
    progress = step / max(total_steps, 1)

    for rct_name, info in affinity.items():
        effect_type = info.get("effect_type", "agonist")

        if rct_name == "5-HT1A" and effect_type == "partial_agonist":
            # Buspirone: 急性期5-HT1A自受体激活→抑制5-HT释放
            # 2-4周后自受体脱敏→5-HT释放恢复→与SERT协同
            if progress < 0.15:  # 前15%时间: 急性抑制
                modifiers[rct_name] = 0.5  # 自受体效应减半(脱敏中)
            elif progress < 0.35:  # 15-35%: 过渡期
                modifiers[rct_name] = 0.5 + 0.5 * (progress - 0.15) / 0.20
            else:  # 35%后: 完全脱敏
                modifiers[rct_name] = 1.0

        elif rct_name == "NMDA" and effect_type == "antagonist":
            # Ketamine: 急性NMDA阻断→谷氨酸暴发→AMPA激活→BDNF→突触可塑性
            # 前期效应强, 后期维持
            if progress < 0.05:
                modifiers[rct_name] = 1.5  # 急性谷氨酸暴发
            else:
                modifiers[rct_name] = 1.0

    return modifiers


__all__ = [
    "ReceptorSubtype",
    "RECEPTOR_REGISTRY",
    "DRUG_RECEPTOR_AFFINITY",
    "ReceptorPDTarget",
    "build_receptor_pd_targets",
    "compute_receptor_deltas",
    "aggregate_receptor_to_nt",
    "get_receptor_time_profile",
]
