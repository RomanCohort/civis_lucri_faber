"""药物相互作用 (DDI) 评估系统 — CYP介导的PK相互作用 + PD靶点交互。

核心功能:
  1. PK层面: CYP抑制/诱导 → 动态ke修正 → 浓度曲线变化
  2. PD层面: 同靶点药物用Bliss独立模型(非朴素求和) → 避免虚假饱和
  3. 禁忌检测: 基于药物类别和临床规则的禁忌/警告

参考文献:
  - Flockhart DA (2007) Drug Interactions: Cytochrome P450 Drug Interaction Table
  - FDA (2020) Clinical Drug Interaction Studies — Cytochrome P450 Enzyme-
    and Transporter-Mediated Drug Interactions Guidance
  - Bliss CI (1939) The toxicity of poisons applied jointly. Ann Appl Biol 26:585-615
  - Loewe S (1953) The problem of synergism and antagonism. Arzneimittelforschung 3:285-290
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class DDIPairRecord:
    """一对药物的DDI记录。"""
    drug_a: str
    drug_b: str
    severity: str          # "contraindicated" / "major" / "moderate" / "minor" / "none"
    mechanism: str         # "pk_cyp_inhibition" / "pk_cyp_induction" / "pd_synergistic" / ...
    clinical_description: str
    affected_cyp: str | None = None
    ke_modifier_a: float = 1.0   # drug_a的ke修正倍率 (<1=清除减慢)
    ke_modifier_b: float = 1.0   # drug_b的ke修正倍率
    pd_interaction_model: str | None = None  # "bliss" / "synergistic" / "antagonistic" / "additive"
    pd_targets: list[str] = field(default_factory=list)
    pd_factor: float = 1.0       # PD交互因子 (>1=超加性, <1=拮抗)


@dataclass
class DDIResult:
    """多药DDI评估结果。"""
    drug_ids: list[str]
    pair_records: list[DDIPairRecord]
    ke_modifiers: dict[str, float]       # {drug_id: ke_multiplier}
    pd_interaction_map: dict[str, str]   # {nt_key: interaction_model}
    contraindications: list[DDIPairRecord]
    warnings: list[str]
    max_severity: str


# ══════════════════════════════════════════════════════════════
# CYP底物档案 (Flockhart表 + FDA DDI指南)
# ══════════════════════════════════════════════════════════════
# 格式: {drug_name: {CYP_isoform: fraction_metabolized}}
# fraction_metabolized: 该CYP途径占总清除的比例 (fm)
# 参考: Flockhart Table (2007), FDA DDI Guidance (2020)

CYP_SUBSTRATE_PROFILES: dict[str, dict[str, float]] = {
    # ── SSRI ──
    "sertraline": {
        "CYP2C19": 0.40,   # 主要代谢途径
        "CYP3A4":  0.30,
        "CYP2D6":  0.10,
        # 剩余0.20: 其他途径/直接排泄
    },
    "fluoxetine": {
        "CYP2D6":  0.50,   # 主要代谢途径
        "CYP2C19": 0.20,
        "CYP3A4":  0.10,
    },
    "paroxetine": {
        "CYP2D6":  0.70,   # 高度依赖CYP2D6
        "CYP3A4":  0.15,
        "CYP2C19": 0.10,
    },
    # ── 兴奋剂 ──
    "amphetamine": {
        "CYP2D6":  0.60,   # 主要代谢途径
        "CYP3A4":  0.20,
        # 剩余0.20: 直接肾排泄
    },
    # ── 镇静剂 ──
    "diazepam": {
        "CYP3A4":  0.60,   # 主要代谢途径
        "CYP2C19": 0.30,
        "CYP2C9":  0.05,
    },
    # ── 抗精神病 ──
    "haloperidol": {
        "CYP2D6":  0.50,
        "CYP3A4":  0.30,
        "CYP1A2":  0.10,
    },
    # ── 心境稳定剂 ──
    "lithium": {
        # 锂几乎完全经肾排泄，无CYP代谢
    },
    # ── 致幻剂/NMDA拮抗 ──
    "ketamine": {
        "CYP3A4":  0.50,
        "CYP2C9":  0.10,
        "CYP2B6":  0.10,
    },
    # ── 阿片 ──
    "morphine": {
        "CYP3A4":  0.40,   # 主要代谢为M3G/M6G
        "CYP2D6":  0.30,   # 少量代谢
        # 剩余0.30: 直接肾排泄
    },
    # ── 5-HT1A部分激动剂 ──
    "buspirone": {
        "CYP3A4":  0.60,   # 主要代谢途径
        "CYP2D6":  0.10,
    },
    # ── 阿片部分激动剂 ──
    "buprenorphine": {
        "CYP3A4":  0.70,   # 主要代谢途径 (N-dealkylation)
        "CYP2C8":  0.20,
    },
    # ── β-受体阻滞剂 ──
    "propranolol": {
        "CYP2D6":  0.50,
        "CYP1A2":  0.30,
        "CYP3A4":  0.10,
    },
    # ── DORA ──
    "suvorexant": {
        "CYP3A4":  0.80,   # 高度依赖CYP3A4
        "CYP2C19": 0.10,
    },
    # ── 抗生素 ──
    "minocycline": {
        # 微量CYP代谢，主要经肝/肾排泄
    },
    # ── α2激动剂 ──
    "clonidine": {
        "CYP2D6":  0.50,
        "CYP3A4":  0.20,
    },
}

# ══════════════════════════════════════════════════════════════
# CYP抑制分类 (Flockhart分类)
# ══════════════════════════════════════════════════════════════
# 格式: {drug_name: {CYP_isoform: "strong"/"moderate"/"weak"}}
# strong: AUC增加≥5倍; moderate: AUC增加2-5倍; weak: AUC增加1.25-2倍

CYP_INHIBITOR_CLASSIFICATION: dict[str, dict[str, str]] = {
    "sertraline": {
        "CYP2D6": "moderate",   # sertraline中度抑制CYP2D6
        "CYP3A4": "weak",
    },
    "fluoxetine": {
        "CYP2D6": "strong",     # fluoxetine是强CYP2D6抑制剂
        "CYP3A4": "moderate",
        "CYP2C19": "moderate",  # norfluoxetine代谢物
    },
    "paroxetine": {
        "CYP2D6": "strong",     # paroxetine是最强CYP2D6抑制剂之一
        "CYP2C19": "moderate",
    },
    "haloperidol": {
        "CYP2D6": "moderate",
        "CYP3A4": "weak",
    },
    # 其余药物: 无临床显著CYP抑制
    # amphetamine: 无显著CYP抑制
    # diazepam: 无显著CYP抑制 (是底物，不是抑制剂)
    # lithium: 无CYP相互作用
    # ketamine: 无显著CYP抑制
    # morphine: 无显著CYP抑制
    # ── 新增药物 ──
    "buspirone": {},               # 无显著CYP抑制
    "buprenorphine": {
        "CYP3A4": "moderate",
        "CYP2D6": "weak",
    },
    "propranolol": {
        "CYP2D6": "moderate",
    },
    "suvorexant": {},              # 无显著CYP抑制
    "minocycline": {
        "CYP3A4": "weak",
    },
    "clonidine": {},               # 无显著CYP抑制
}

# Flockhart抑制倍率 (R值)
# AUC_ratio = 1 + (R - 1) × concentration_factor
FLOCKHART_R: dict[str, float] = {
    "strong": 5.0,
    "moderate": 2.5,
    "weak": 1.5,
    "none": 1.0,
}


# ══════════════════════════════════════════════════════════════
# PD相互作用规则
# ══════════════════════════════════════════════════════════════

# 临床已知的PD交互覆盖 (优先于通用规则)
# 格式: {(drug_a, drug_b): {model, targets, factor, severity}}
PD_CLINICAL_OVERRIDES: dict[tuple[str, str], dict] = {
    # ── 双SSRI: Bliss独立 (SERT占有率饱和) ──
    ("sertraline", "fluoxetine"): {
        "model": "bliss",
        "targets": ["nt_serotonin"],
        "factor": 1.0,
        "severity": "moderate",
        "description": "双重SERT抑制: 5-HT综合征风险增加",
    },
    ("sertraline", "paroxetine"): {
        "model": "bliss",
        "targets": ["nt_serotonin"],
        "factor": 1.0,
        "severity": "major",
        "description": "强CYP2D6抑制+双重SERT: 高5-HT综合征风险",
    },
    ("fluoxetine", "paroxetine"): {
        "model": "bliss",
        "targets": ["nt_serotonin"],
        "factor": 1.0,
        "severity": "major",
        "description": "双强CYP2D6抑制+双重SERT: 极高5-HT综合征风险",
    },
    # ── DA拮抗: haloperidol拮抗amphetamine ──
    ("haloperidol", "amphetamine"): {
        "model": "antagonistic",
        "targets": ["nt_dopamine"],
        "factor": 0.7,
        "severity": "moderate",
        "description": "DA拮抗: haloperidol阻断amphetamine的DA效应",
    },
    # ── CNS抑制协同: BZD+阿片 ──
    ("diazepam", "morphine"): {
        "model": "synergistic",
        "targets": ["nt_gaba", "nt_dopamine"],
        "factor": 1.5,
        "severity": "contraindicated",
        "description": "FDA黑框警告: BZD+阿片→致命呼吸抑制",
    },
    # ── NMDA+DA: ketamine增强amphetamine ──
    ("ketamine", "amphetamine"): {
        "model": "synergistic",
        "targets": ["nt_dopamine", "nt_glutamate"],
        "factor": 1.2,
        "severity": "moderate",
        "description": "NMDA拮抗增强DA释放: 精神病风险增加",
    },
    # ── SSRI+ketamine: 快速抗抑郁协同 ──
    ("sertraline", "ketamine"): {
        "model": "synergistic",
        "targets": ["nt_serotonin", "nt_glutamate"],
        "factor": 1.15,
        "severity": "minor",
        "description": "SSRI+NMDA拮抗: 可能增强快速抗抑郁效应",
    },
    # ── 锂+SSRI: 增强抗抑郁 + 5-HT风险 ──
    ("lithium", "sertraline"): {
        "model": "synergistic",
        "targets": ["nt_serotonin"],
        "factor": 1.1,
        "severity": "minor",
        "description": "锂增强5-HT效应: 可能增强抗抑郁，轻微5-HT风险",
    },
    # ── 受体层DDI覆盖 ──
    # buspirone+SSRI: 急性5-HT1A自受体拮抗 → 5-HT↓ → 后期脱敏→协同
    ("buspirone", "sertraline"): {
        "model": "time_dependent",
        "targets": ["rct_5ht1a", "nt_serotonin"],
        "factor": 1.0,           # 初始factor=1.0, 后期→1.15
        "severity": "minor",
        "description": "buspirone急性期5-HT1A自受体激活→抑制5-HT释放(拮抗SERT); 2-4周后自受体脱敏→协同SERT",
        "time_profile": {
            "0_14_days": {"model": "antagonistic", "factor": 0.85},
            "14_28_days": {"model": "additive", "factor": 1.0},
            "28+_days": {"model": "synergistic", "factor": 1.15},
        },
    },
    ("buspirone", "fluoxetine"): {
        "model": "time_dependent",
        "targets": ["rct_5ht1a", "nt_serotonin"],
        "factor": 1.0,
        "severity": "minor",
        "description": "buspirone+fluoxetine: 同sertraline时间依赖模式",
        "time_profile": {
            "0_14_days": {"model": "antagonistic", "factor": 0.85},
            "14_28_days": {"model": "additive", "factor": 1.0},
            "28+_days": {"model": "synergistic", "factor": 1.15},
        },
    },
    # buprenorphine+morphine: 天花板效应 → 减少过量风险
    ("buprenorphine", "morphine"): {
        "model": "ceiling",
        "targets": ["rct_muopioid", "nt_dopamine"],
        "factor": 0.6,
        "severity": "moderate",
        "description": "buprenorphine部分激动mu-opioid → 天花板效应 → 阻断morphine完全激动; MAT核心机制",
    },
    # propranolol+amphetamine: 外周β-阻断拮抗NE效应
    ("propranolol", "amphetamine"): {
        "model": "antagonistic",
        "targets": ["rct_beta1", "rct_beta2", "nt_norepinephrine"],
        "factor": 0.75,
        "severity": "moderate",
        "description": "propranolol阻断外周β-adrenergic → 减少amphetamine的躯体焦虑症状(心悸/震颤)",
    },
    # minocycline+SSRI: 微胶质协同抗炎 → 增强抗抑郁
    ("minocycline", "sertraline"): {
        "model": "synergistic",
        "targets": ["nt_serotonin"],
        "factor": 1.1,
        "severity": "minor",
        "description": "minocycline抑制TLR2/4→降低微胶质炎症→减少IDO通路→5-HT耗竭减少→协同SSRI",
    },
    # suvorexant+diazepam: 双重促睡眠 → 协同但非禁忌
    ("suvorexant", "diazepam"): {
        "model": "synergistic",
        "targets": ["rct_orexin1", "rct_orexin2", "nt_gaba"],
        "factor": 1.2,
        "severity": "moderate",
        "description": "DORA降低觉醒驱动+BZD增强GABA抑制: 协同促睡眠，但过度镇静风险",
    },
    # clonidine+propranolol: 双重降低NE → 严重低血压风险
    ("clonidine", "propranolol"): {
        "model": "synergistic",
        "targets": ["rct_alpha2", "rct_beta1", "nt_norepinephrine"],
        "factor": 1.3,
        "severity": "major",
        "description": "α2激动+β阻断: 双重降低交感输出→严重低血压/心动过缓风险",
    },
}

# 药物类别禁忌规则
CONTRAINDICATION_RULES: list[dict] = [
    {
        "drug_classes": ["sedative", "opioid"],
        "severity": "contraindicated",
        "reason": "FDA黑框: BZD+阿片→致命呼吸抑制",
        "targets": ["nt_gaba"],
    },
    {
        "drug_classes": ["SSRI", "SSRI"],
        "severity": "moderate",
        "reason": "双重SERT抑制→5-HT综合征风险",
        "targets": ["nt_serotonin"],
    },
    {
        "drug_classes": ["stimulant", "antipsychotic"],
        "severity": "moderate",
        "reason": "DA激动+拮抗: 效应抵消/精神病风险",
        "targets": ["nt_dopamine"],
    },
    {
        "drug_classes": ["hallucinogen", "stimulant"],
        "severity": "moderate",
        "reason": "NMDA拮抗+DA释放: 精神病/兴奋性毒性风险",
        "targets": ["nt_dopamine", "nt_glutamate"],
    },
    {
        "drug_classes": ["beta_blocker", "alpha2_agonist"],
        "severity": "major",
        "reason": "双重降低交感输出→严重低血压/心动过缓",
        "targets": ["nt_norepinephrine"],
    },
    {
        "drug_classes": ["sedative", "dora"],
        "severity": "moderate",
        "reason": "BZD+DORA: 过度镇静风险",
        "targets": ["nt_gaba"],
    },
]

# 药物→类别映射
DRUG_CLASS_MAP: dict[str, str] = {
    "sertraline": "SSRI",
    "fluoxetine": "SSRI",
    "paroxetine": "SSRI",
    "amphetamine": "stimulant",
    "diazepam": "sedative",
    "haloperidol": "antipsychotic",
    "lithium": "mood_stabilizer",
    "ketamine": "hallucinogen",
    "morphine": "opioid",
    "buspirone": "anxiolytic",
    "buprenorphine": "opioid",
    "propranolol": "beta_blocker",
    "suvorexant": "dora",
    "minocycline": "antibiotic",
    "clonidine": "alpha2_agonist",
}


# ══════════════════════════════════════════════════════════════
# DDI评估核心
# ══════════════════════════════════════════════════════════════

def assess_ddi(
    drug_ids: list[str],
    drug_defs: dict[str, Any] | None = None,
    drug_admet: dict[str, dict] | None = None,
) -> DDIResult:
    """评估多药DDI。

    Args:
        drug_ids: 药物ID列表 (如 ["sertraline", "fluoxetine"])
        drug_defs: {drug_id: DrugDefinition} (可选，用于读取CYP字段)
        drug_admet: {drug_id: ADMET预测结果} (可选，用于读取CYP抑制分数)

    Returns:
        DDIResult with ke_modifiers, PD interaction map, contraindications, warnings
    """
    if len(drug_ids) < 2:
        return DDIResult(
            drug_ids=drug_ids,
            pair_records=[],
            ke_modifiers={d: 1.0 for d in drug_ids},
            pd_interaction_map={},
            contraindications=[],
            warnings=[],
            max_severity="none",
        )

    pair_records: list[DDIPairRecord] = []
    ke_modifiers: dict[str, float] = {d: 1.0 for d in drug_ids}
    pd_interaction_map: dict[str, str] = {}
    contraindications: list[DDIPairRecord] = []
    warnings: list[str] = []
    max_sev = "none"

    severity_order = ["none", "minor", "moderate", "major", "contraindicated"]

    # ── 1. 逐对评估 ──
    for i, drug_a in enumerate(drug_ids):
        for drug_b in drug_ids[i + 1:]:
            # 检查临床覆盖
            override = _get_clinical_override(drug_a, drug_b)

            # PK: CYP抑制
            pk_records = _assess_cyp_interaction(drug_a, drug_b)
            for rec in pk_records:
                pair_records.append(rec)
                # 累积ke修正 (取最小值 = 最强抑制)
                ke_modifiers[rec.drug_a] = min(ke_modifiers[rec.drug_a], rec.ke_modifier_a)
                ke_modifiers[rec.drug_b] = min(ke_modifiers[rec.drug_b], rec.ke_modifier_b)

            # PD: 靶点交互
            pd_record = _assess_pd_interaction(drug_a, drug_b, override)
            if pd_record:
                pair_records.append(pd_record)
                for t in pd_record.pd_targets:
                    if t not in pd_interaction_map:
                        pd_interaction_map[t] = pd_record.pd_interaction_model or "additive"

            # 临床覆盖的severity
            if override:
                sev = override.get("severity", "moderate")
                if severity_order.index(sev) > severity_order.index(max_sev):
                    max_sev = sev
                if sev in ("contraindicated", "major"):
                    desc = override.get("description", f"{drug_a}+{drug_b} interaction")
                    contraindications.append(DDIPairRecord(
                        drug_a=drug_a, drug_b=drug_b,
                        severity=sev,
                        mechanism="pd_" + (override.get("model", "synergistic")),
                        clinical_description=desc,
                        pd_interaction_model=override.get("model"),
                        pd_targets=override.get("targets", []),
                        pd_factor=override.get("factor", 1.0),
                    ))
                    warnings.append(f"[{sev.upper()}] {desc}")

    # ── 2. 类别禁忌规则 ──
    classes = [DRUG_CLASS_MAP.get(d, "unknown") for d in drug_ids]
    for rule in CONTRAINDICATION_RULES:
        rule_classes = rule["drug_classes"]
        # 检查是否所有指定类别都出现在当前药物中
        matched = []
        for rc in rule_classes:
            for d, c in zip(drug_ids, classes):
                if c == rc and d not in matched:
                    matched.append(d)
                    break
        if len(matched) >= len(rule_classes):
            sev = rule["severity"]
            if severity_order.index(sev) > severity_order.index(max_sev):
                max_sev = sev
            warnings.append(f"[{sev.upper()}] {rule['reason']}")

    # ── 3. 从pair_records提取最大severity ──
    for rec in pair_records:
        if severity_order.index(rec.severity) > severity_order.index(max_sev):
            max_sev = rec.severity

    return DDIResult(
        drug_ids=drug_ids,
        pair_records=pair_records,
        ke_modifiers=ke_modifiers,
        pd_interaction_map=pd_interaction_map,
        contraindications=contraindications,
        warnings=warnings,
        max_severity=max_sev,
    )


def _get_clinical_override(drug_a: str, drug_b: str) -> dict | None:
    """查找临床PD交互覆盖。"""
    key1 = (drug_a, drug_b)
    key2 = (drug_b, drug_a)
    if key1 in PD_CLINICAL_OVERRIDES:
        return PD_CLINICAL_OVERRIDES[key1]
    if key2 in PD_CLINICAL_OVERRIDES:
        return PD_CLINICAL_OVERRIDES[key2]
    return None


def _assess_cyp_interaction(drug_a: str, drug_b: str) -> list[DDIPairRecord]:
    """评估CYP介导的PK相互作用 (双向)。"""
    records: list[DDIPairRecord] = []

    # A抑制B的CYP → B清除减慢
    rec_ab = _compute_cyp_ke_modifier(drug_a, drug_b)
    if rec_ab:
        records.append(rec_ab)

    # B抑制A的CYP → A清除减慢
    rec_ba = _compute_cyp_ke_modifier(drug_b, drug_a)
    if rec_ba:
        records.append(rec_ba)

    return records


def _compute_cyp_ke_modifier(
    inhibitor: str,
    substrate: str,
) -> DDIPairRecord | None:
    """计算inhibitor对substrate的ke修正。

    公式 (FDA DDI指南):
      AUC_ratio = 1 / (1 - fm × (1 - 1/R))
      ke_modifier = 1 / AUC_ratio

    其中:
      fm = substrate经该CYP代谢的清除分数
      R = Flockhart抑制倍率 (strong=5, moderate=2.5, weak=1.5)

    Returns:
        DDIPairRecord if interaction exists, else None
    """
    inhibitor_classes = CYP_INHIBITOR_CLASSIFICATION.get(inhibitor, {})
    substrate_profile = CYP_SUBSTRATE_PROFILES.get(substrate, {})

    if not inhibitor_classes or not substrate_profile:
        return None

    # 找到共同的CYP: inhibitor抑制的CYP恰好是substrate的代谢途径
    shared_cyps = set(inhibitor_classes.keys()) & set(substrate_profile.keys())
    if not shared_cyps:
        return None

    # 计算累积AUC_ratio (多CYP叠加)
    total_auc_ratio = 1.0
    strongest_cyp = None
    strongest_severity = "none"

    for cyp in shared_cyps:
        fm = substrate_profile[cyp]  # 该CYP途径的清除分数
        inhibitor_class = inhibitor_classes[cyp]
        R = FLOCKHART_R.get(inhibitor_class, 1.0)

        # AUC_ratio for this CYP pathway
        # AUC_ratio = 1 / (1 - fm * (1 - 1/R))
        denominator = 1.0 - fm * (1.0 - 1.0 / R)
        if denominator <= 0.01:
            denominator = 0.01  # 防止除零/极端值
        auc_ratio_cyp = 1.0 / denominator

        # 多CYP叠加: 用乘法模型
        # 总AUC_ratio ≈ product of individual AUC_ratios (保守估计)
        total_auc_ratio *= auc_ratio_cyp

        # 记录最强的CYP交互
        if FLOCKHART_R.get(inhibitor_class, 1.0) > FLOCKHART_R.get(strongest_severity, 1.0):
            strongest_cyp = cyp
            strongest_severity = inhibitor_class

    if total_auc_ratio <= 1.05:
        return None  # <5% AUC变化，无临床意义

    ke_modifier = 1.0 / total_auc_ratio

    # severity基于AUC变化
    if total_auc_ratio >= 5.0:
        severity = "major"
    elif total_auc_ratio >= 2.0:
        severity = "moderate"
    elif total_auc_ratio >= 1.25:
        severity = "minor"
    else:
        severity = "none"

    return DDIPairRecord(
        drug_a=inhibitor,
        drug_b=substrate,
        severity=severity,
        mechanism="pk_cyp_inhibition",
        clinical_description=(
            f"{inhibitor} inhibits {strongest_cyp or 'CYP'} → "
            f"{substrate} clearance ↓{((1 - ke_modifier) * 100):.0f}% "
            f"(AUC ↑{((total_auc_ratio - 1) * 100):.0f}%)"
        ),
        affected_cyp=strongest_cyp,
        ke_modifier_a=1.0,       # inhibitor自身ke不变
        ke_modifier_b=ke_modifier,  # substrate ke减慢
    )


def _assess_pd_interaction(
    drug_a: str,
    drug_b: str,
    override: dict | None = None,
) -> DDIPairRecord | None:
    """评估PD靶点交互。"""
    if override:
        return DDIPairRecord(
            drug_a=drug_a,
            drug_b=drug_b,
            severity=override.get("severity", "moderate"),
            mechanism="pd_" + override.get("model", "additive"),
            clinical_description=override.get("description", ""),
            pd_interaction_model=override.get("model"),
            pd_targets=override.get("targets", []),
            pd_factor=override.get("factor", 1.0),
        )

    # 无临床覆盖时: 不产生PD交互记录 (使用默认additive)
    return None


# ══════════════════════════════════════════════════════════════
# PD效应组合 (替代朴素求和)
# ══════════════════════════════════════════════════════════════

def combine_pd_deltas(
    per_drug_deltas: dict[str, dict[str, float]],
    pd_targets_per_drug: dict[str, list],
    ddi_result: DDIResult | None = None,
) -> tuple[dict[str, float], list[str]]:
    """组合多药的PD delta，考虑DDI。

    Args:
        per_drug_deltas: {drug_id: {nt_key: delta_value}}
        pd_targets_per_drug: {drug_id: [PDTarget, ...]}
        ddi_result: DDI评估结果 (可选)

    Returns:
        (combined_deltas, warnings)
    """
    if len(per_drug_deltas) <= 1:
        # 单药: 直接返回
        for deltas in per_drug_deltas.values():
            return deltas, []

    combined: dict[str, float] = {}
    warnings: list[str] = []

    # 收集每个NT的所有delta
    nt_deltas: dict[str, list[tuple[str, float]]] = {}  # {nt_key: [(drug_id, delta)]}
    for drug_id, deltas in per_drug_deltas.items():
        for nt_key, delta in deltas.items():
            nt_deltas.setdefault(nt_key, []).append((drug_id, delta))

    # 获取DDI交互模型
    pd_interaction_map = ddi_result.pd_interaction_map if ddi_result else {}

    for nt_key, drug_delta_list in nt_deltas.items():
        if len(drug_delta_list) == 1:
            # 只有一个药物影响该NT
            combined[nt_key] = drug_delta_list[0][1]
            continue

        # 多药影响同一NT → 需要交互模型
        model = pd_interaction_map.get(nt_key, "additive")

        if model == "bliss":
            # Bliss独立模型: E = E_A + E_B - E_A * E_B
            # 适用于同靶点竞争性结合 (如两个SERT抑制剂)
            # 注意: delta可能为负 (decrease)
            positive_deltas = [d for _, d in drug_delta_list if d > 0]
            negative_deltas = [d for _, d in drug_delta_list if d < 0]

            # 正向delta用Bliss
            bliss_result = 0.0
            for d in positive_deltas:
                bliss_result = bliss_result + d - bliss_result * d

            # 负向delta用Bliss (反转)
            bliss_neg = 0.0
            for d in negative_deltas:
                d_abs = abs(d)
                bliss_neg = bliss_neg + d_abs - bliss_neg * d_abs

            combined[nt_key] = bliss_result - bliss_neg

            drug_names = [did for did, _ in drug_delta_list]
            warnings.append(
                f"Bliss model applied for {nt_key}: "
                f"{'+'.join(drug_names)} (SERT occupancy saturation)"
            )

        elif model == "synergistic":
            # 超加性: 先additive再乘factor
            factor = 1.0
            if ddi_result:
                for rec in ddi_result.pair_records:
                    if nt_key in rec.pd_targets and rec.pd_interaction_model == "synergistic":
                        factor = max(factor, rec.pd_factor)

            additive = sum(d for _, d in drug_delta_list)
            combined[nt_key] = additive * factor

        elif model == "antagonistic":
            # 拮抗: 效应部分抵消
            factor = 1.0
            if ddi_result:
                for rec in ddi_result.pair_records:
                    if nt_key in rec.pd_targets and rec.pd_interaction_model == "antagonistic":
                        factor = rec.pd_factor  # <1

            # 取绝对值较大的方向，乘以factor
            positive_sum = sum(d for _, d in drug_delta_list if d > 0)
            negative_sum = sum(d for _, d in drug_delta_list if d < 0)
            net = positive_sum + negative_sum
            combined[nt_key] = net * factor

        else:
            # additive (默认): 朴素求和
            combined[nt_key] = sum(d for _, d in drug_delta_list)

    return combined, warnings


# ══════════════════════════════════════════════════════════════
# 动态ke修正 (浓度依赖)
# ══════════════════════════════════════════════════════════════

def compute_step_ke_modifiers(
    drug_ids: list[str],
    drug_concentrations: dict[str, float],
    drug_cmax_reference: dict[str, float],
    ddi_result: DDIResult,
) -> dict[str, float]:
    """计算当前步的浓度依赖ke修正。

    抑制剂浓度达到治疗浓度(Cmax)时 = 全效;
    低于Cmax时按比例缩放。

    Args:
        drug_ids: 药物ID列表
        drug_concentrations: {drug_id: current_conc_mg_per_L}
        drug_cmax_reference: {drug_id: therapeutic_Cmax_mg_per_L}
        ddi_result: DDI评估结果

    Returns:
        {drug_id: ke_modifier} (1.0 = 无变化, <1.0 = 清除减慢)
    """
    if not ddi_result or not ddi_result.pair_records:
        return {d: 1.0 for d in drug_ids}

    ke_mods: dict[str, float] = {d: 1.0 for d in drug_ids}

    for rec in ddi_result.pair_records:
        if rec.mechanism != "pk_cyp_inhibition":
            continue

        inhibitor = rec.drug_a
        substrate = rec.drug_b

        # 抑制剂的浓度依赖缩放
        conc_inhibitor = drug_concentrations.get(inhibitor, 0.0)
        cmax_ref = drug_cmax_reference.get(inhibitor, 1.0)

        if cmax_ref <= 0:
            concentration_factor = 1.0
        else:
            # 浓度达到Cmax时全效，超过Cmax不再增加
            concentration_factor = min(conc_inhibitor / cmax_ref, 1.0)

        # 基础ke_modifier × 浓度依赖缩放
        # 当concentration_factor=0时，ke_modifier=1.0 (无抑制)
        # 当concentration_factor=1时，ke_modifier=rec.ke_modifier_b (全抑制)
        base_modifier = rec.ke_modifier_b  # <1.0
        step_modifier = 1.0 - (1.0 - base_modifier) * concentration_factor

        # 累积: 取最小值 (最强抑制)
        ke_mods[substrate] = min(ke_mods.get(substrate, 1.0), step_modifier)

    return ke_mods


__all__ = [
    "DDIPairRecord",
    "DDIResult",
    "CYP_SUBSTRATE_PROFILES",
    "CYP_INHIBITOR_CLASSIFICATION",
    "FLOCKHART_R",
    "PD_CLINICAL_OVERRIDES",
    "CONTRAINDICATION_RULES",
    "DRUG_CLASS_MAP",
    "assess_ddi",
    "combine_pd_deltas",
    "compute_step_ke_modifiers",
]
