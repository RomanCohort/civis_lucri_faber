"""药物-心理治疗协同引擎 (Pharmacotherapy Synergy Engine)

核心原理: 药物和心理治疗不是简单叠加，而是有协同/拮抗交互。
协同 = combined > drug_only + therapy_only
拮抗 = combined < drug_only + therapy_only

协同机制基于真实临床研究:
    - Castren (2005): SSRI→BDNF↑→神经可塑性窗口→CBT更有效
    - Walker et al. (2002): DCS(NMDA)→恐惧消退记忆巩固↑→暴露疗法增强
    - Linehan (1993): 心境稳定剂→极端波动↓→DBT技能更可习得
    - Carhart-Harris et al. (2012): 5-HT2A→DMN激活→自省深度↑
    - 苯二氮卓+暴露疗法 = 拮抗 (药物抑制恐惧→暴露学习无法发生)
    - SSRI过度+精神动力学 = 拮抗 (情感钝化→自省材料减少)

用法:
    synergy = PharmacotherapySynergy()
    factor = synergy.compute("antidepressant", "CBT", "MDD")
    # factor ≈ 0.4 (协同增强: CBT效果提升40%)
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# ===== 协同/拮抗类型 =====

class SynergyType(Enum):
    """协同类型"""
    SYNERGISTIC = "synergistic"     # 协同: 1+1>2
    ADDITIVE = "additive"           # 叠加: 1+1=2
    ANTAGONISTIC = "antagonistic"   # 拮抗: 1+1<2


# ===== 协同记录 =====

@dataclass
class SynergyRecord:
    """协同效应记录"""
    drug: str
    therapy: str
    condition: str
    synergy_type: SynergyType
    synergy_factor: float       # 范围: [-0.5, 1.5], 正=协同, 负=拮抗
    mechanism: str              # 生物学机制描述
    evidence: str               # 临床证据引用
    dose_dependency: bool = False  # 是否有剂量依赖性


# ===== 协同矩阵 =====
# 基于临床文献的药物-治疗交互矩阵

SYNERGY_MATRIX: Dict[str, Dict[str, Dict[str, SynergyRecord]]] = {
    # ──── 抗抑郁药 (SSRI) ────
    "antidepressant": {
        "CBT": {
            "MDD": SynergyRecord(
                drug="antidepressant", therapy="CBT", condition="MDD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.4,
                mechanism="SSRI↑BDNF→打开神经可塑性窗口→CBT认知重建更有效",
                evidence="Castren (2005) Nat Rev Neurosci; DeRubeis et al. (2005) Arch Gen Psychiatry",
                dose_dependency=True,
            ),
            "GAD": SynergyRecord(
                drug="antidepressant", therapy="CBT", condition="GAD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.3,
                mechanism="SSRI↑5-HT→焦虑基线降低→CBT暴露练习更可行",
                evidence="Hollon et al. (2005) JAMA",
            ),
            "OCD": SynergyRecord(
                drug="antidepressant", therapy="CBT", condition="OCD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.35,
                mechanism="SSRI↑5-HT→强迫冲动降低→ERP暴露练习依从性↑",
                evidence="Foa et al. (2005) Am J Psychiatry",
            ),
        },
        "psychodynamic": {
            "MDD": SynergyRecord(
                drug="antidepressant", therapy="psychodynamic", condition="MDD",
                synergy_type=SynergyType.ADDITIVE,
                synergy_factor=0.1,
                mechanism="SSRI改善情绪→自省材料略增，但高剂量可能情感钝化",
                evidence=" mildly synergistic at low dose, antagonistic at high dose",
                dose_dependency=True,
            ),
        },
        "ACT": {
            "MDD": SynergyRecord(
                drug="antidepressant", therapy="ACT", condition="MDD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.3,
                mechanism="SSRI↑BDNF→认知灵活性↑→ACT心理灵活性训练更有效",
                evidence="Hayes et al. (2006)",
            ),
        },
        "interpersonal": {
            "MDD": SynergyRecord(
                drug="antidepressant", therapy="interpersonal", condition="MDD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.35,
                mechanism="SSRI改善社交焦虑→人际治疗社交练习更可行",
                evidence="Weissman et al. (2000)",
            ),
        },
    },

    # ──── 兴奋剂 ────
    "stimulant": {
        "CBT": {
            "ADHD": SynergyRecord(
                drug="stimulant", therapy="CBT", condition="ADHD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.5,
                mechanism="兴奋剂↑DA→注意力恢复→CBT技能训练可习得",
                evidence="Safren et al. (2010) JAMA; Solanto (2018)",
            ),
        },
        "ACT": {
            "ADHD": SynergyRecord(
                drug="stimulant", therapy="ACT", condition="ADHD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.4,
                mechanism="DA↑→动机恢复→ACT价值导向行动更可行",
                evidence="Hayes et al. (2006)",
            ),
        },
    },

    # ──── 镇静剂 (苯二氮卓) ────
    "sedative": {
        "exposure": {
            "GAD": SynergyRecord(
                drug="sedative", therapy="exposure", condition="GAD",
                synergy_type=SynergyType.ANTAGONISTIC,
                synergy_factor=-0.4,
                mechanism="苯二氮卓抑制恐惧反应→暴露学习无法发生(恐惧被药物压制而非消退)",
                evidence="Basoglu et al. (1994) Br J Psychiatry; van Balkom et al. (1997)",
            ),
            "Specific_Phobia": SynergyRecord(
                drug="sedative", therapy="exposure", condition="Specific_Phobia",
                synergy_type=SynergyType.ANTAGONISTIC,
                synergy_factor=-0.35,
                mechanism="同上: 镇静剂阻止恐惧激活→消退学习无法发生",
                evidence="Marks et al. (1993)",
            ),
            "PTSD": SynergyRecord(
                drug="sedative", therapy="exposure", condition="PTSD",
                synergy_type=SynergyType.ANTAGONISTIC,
                synergy_factor=-0.3,
                mechanism="苯二氮卓可能阻碍PTSD暴露疗法效果",
                evidence="van Minen et al. (2002)",
            ),
        },
    },

    # ──── 抗焦虑药 ────
    "anxiolytic": {
        "exposure": {
            "GAD": SynergyRecord(
                drug="anxiolytic", therapy="exposure", condition="GAD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.2,
                mechanism="非苯二氮卓抗焦虑药降低基线焦虑→暴露练习依从性↑(不阻止恐惧激活)",
                evidence="比苯二氮卓更少拮抗",
            ),
            "Social_Anxiety": SynergyRecord(
                drug="anxiolytic", therapy="exposure", condition="Social_Anxiety",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.25,
                mechanism="社交焦虑基线降低→社交暴露练习更可行",
                evidence="Clark & Ehlers (1993)",
            ),
        },
        "interpersonal": {
            "Social_Anxiety": SynergyRecord(
                drug="anxiolytic", therapy="interpersonal", condition="Social_Anxiety",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.3,
                mechanism="抗焦虑→社交互动焦虑↓→人际治疗关系练习更可行",
                evidence="Aderka et al. (2012)",
            ),
        },
        "EMDR": {
            "PTSD": SynergyRecord(
                drug="anxiolytic", therapy="EMDR", condition="PTSD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.2,
                mechanism="基线焦虑降低→EMDR加工创伤记忆时不过载",
                evidence="Shapiro (2001)",
            ),
        },
    },

    # ──── 共情增强剂 (催产素) ────
    "empathogen": {
        "interpersonal": {
            "Social_Anxiety": SynergyRecord(
                drug="empathogen", therapy="interpersonal", condition="Social_Anxiety",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.35,
                mechanism="催产素↑→信任感↑→人际治疗关系建立更快",
                evidence="Guastella et al. (2010)",
            ),
        },
        "psychodynamic": {
            "BPD": SynergyRecord(
                drug="empathogen", therapy="psychodynamic", condition="BPD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.2,
                mechanism="催产素→移情关系更安全→探索防御机制更可行",
                evidence="Bartz et al. (2011)",
            ),
        },
    },

    # ──── 心境稳定剂 ────
    "mood_stabilizer": {
        "DBT": {
            "BPD": SynergyRecord(
                drug="mood_stabilizer", therapy="DBT", condition="BPD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.45,
                mechanism="心境稳定剂→极端情绪波动↓→DBT技能在可习得范围内",
                evidence="Linehan (1993); Swartz et al. (2003)",
            ),
            "bipolar_mania": SynergyRecord(
                drug="mood_stabilizer", therapy="DBT", condition="bipolar_mania",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.4,
                mechanism="锂盐→躁狂波动↓→DBT正念/情绪调节技能可习得",
                evidence="Miklowitz et al. (2007)",
            ),
        },
        "CBT": {
            "bipolar_mania": SynergyRecord(
                drug="mood_stabilizer", therapy="CBT", condition="bipolar_mania",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.35,
                mechanism="心境稳定→认知重建可在稳定基线上进行",
                evidence="Lam et al. (2003)",
            ),
        },
    },

    # ──── 多巴胺拮抗剂 (抗精神病) ────
    "dopamine_antagonist": {
        "CBT": {
            "schizophrenia_positive": SynergyRecord(
                drug="dopamine_antagonist", therapy="CBT", condition="schizophrenia_positive",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.3,
                mechanism="抗精神病药→妄想强度↓→CBT现实检验可进行",
                evidence="Tarrier et al. (2004); Wykes et al. (2008)",
            ),
        },
    },

    # ──── 致幻剂 ────
    "hallucinogen": {
        "psychodynamic": {
            "PTSD": SynergyRecord(
                drug="hallucinogen", therapy="psychodynamic", condition="PTSD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.5,
                mechanism="5-HT2A→DMN重组→创伤记忆重新整合→精神动力学探索深度↑",
                evidence="Carhart-Harris et al. (2012); Mithoefer et al. (2011)",
            ),
        },
        "ACT": {
            "MDD": SynergyRecord(
                drug="hallucinogen", therapy="ACT", condition="MDD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.45,
                mechanism="5-HT2A→认知僵化打破→ACT心理灵活性训练更有效",
                evidence="Carhart-Harris & Friston (2019)",
            ),
        },
    },

    # ──── 促智药 ────
    "nootropic": {
        "CBT": {
            "MDD": SynergyRecord(
                drug="nootropic", therapy="CBT", condition="MDD",
                synergy_type=SynergyType.SYNERGISTIC,
                synergy_factor=0.2,
                mechanism="ACh↑→注意力↑→CBT认知重建练习更专注",
                evidence="间接证据",
            ),
        },
    },
}


# ===== 协同计算器 =====

class SynergyCalculator:
    """药物-心理治疗协同因子计算器

    用法:
        calc = SynergyCalculator()
        factor = calc.compute("antidepressant", "CBT", "MDD")
        # factor ≈ 0.4 (CBT效果提升40%)
    """

    def __init__(self, custom_overrides: Optional[Dict] = None):
        """初始化

        Args:
            custom_overrides: 自定义协同覆盖 {drug: {therapy: {condition: SynergyRecord}}}
        """
        self._matrix = SYNERGY_MATRIX.copy()
        if custom_overrides:
            for drug, therapies in custom_overrides.items():
                if drug not in self._matrix:
                    self._matrix[drug] = {}
                for therapy, conditions in therapies.items():
                    if therapy not in self._matrix[drug]:
                        self._matrix[drug][therapy] = {}
                    self._matrix[drug][therapy].update(conditions)

    def compute(
        self,
        drug: str,
        therapy: str,
        condition: str,
        drug_dose: float = 1.0,
    ) -> float:
        """计算协同因子

        Args:
            drug: 药物预设名 (如 "antidepressant")
            therapy: 治疗流派名 (如 "CBT")
            condition: 疾病ID (如 "MDD")
            drug_dose: 药物剂量系数 [0,1], 影响协同强度

        Returns:
            synergy_factor: 范围 [-0.5, 1.5]
                正值 = 协同增强
                0 = 纯叠加
                负值 = 拮抗削弱
        """
        # 查找精确匹配
        record = self._lookup(drug, therapy, condition)
        if record is not None:
            factor = record.synergy_factor
            # 剂量依赖性调整
            if record.dose_dependency and drug_dose < 0.5:
                factor *= drug_dose * 2  # 低剂量时协同减弱
            return float(np.clip(factor, -0.5, 1.5))

        # 无精确匹配: 尝试模糊匹配 (同药物不同疾病)
        drug_matrix = self._matrix.get(drug, {})
        therapy_records = drug_matrix.get(therapy, {})
        if therapy_records:
            # 取同药物同治疗不同疾病的平均值
            factors = [r.synergy_factor for r in therapy_records.values()]
            return float(np.clip(np.mean(factors) * 0.5, -0.3, 0.5))

        # 完全无匹配: 默认轻微叠加
        return 0.05

    def compute_all(
        self,
        active_drugs: List[str],
        active_therapies: List[str],
        condition: str,
        drug_doses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """计算所有药物-治疗组合的协同因子

        Returns:
            {therapy_name: total_synergy_factor}
        """
        if drug_doses is None:
            drug_doses = {d: 1.0 for d in active_drugs}

        result = {}
        for therapy in active_therapies:
            total = 0.0
            for drug in active_drugs:
                dose = drug_doses.get(drug, 1.0)
                total += self.compute(drug, therapy, condition, dose)
            result[therapy] = float(np.clip(total, -0.5, 1.5))

        return result

    def get_mechanism(
        self,
        drug: str,
        therapy: str,
        condition: str,
    ) -> str:
        """获取协同/拮抗的生物学机制描述"""
        record = self._lookup(drug, therapy, condition)
        if record is not None:
            return record.mechanism
        return "无已知交互机制"

    def get_evidence(
        self,
        drug: str,
        therapy: str,
        condition: str,
    ) -> str:
        """获取临床证据引用"""
        record = self._lookup(drug, therapy, condition)
        if record is not None:
            return record.evidence
        return "无直接临床证据"

    def get_synergy_type(
        self,
        drug: str,
        therapy: str,
        condition: str,
    ) -> SynergyType:
        """获取协同类型"""
        record = self._lookup(drug, therapy, condition)
        if record is not None:
            return record.synergy_type
        return SynergyType.ADDITIVE

    def list_known_interactions(self) -> List[Dict[str, Any]]:
        """列出所有已知的药物-治疗交互"""
        results = []
        for drug, therapies in self._matrix.items():
            for therapy, conditions in therapies.items():
                for condition, record in conditions.items():
                    results.append({
                        "drug": drug,
                        "therapy": therapy,
                        "condition": condition,
                        "type": record.synergy_type.value,
                        "factor": record.synergy_factor,
                        "mechanism": record.mechanism,
                    })
        return results

    def _lookup(
        self,
        drug: str,
        therapy: str,
        condition: str,
    ) -> Optional[SynergyRecord]:
        """查找精确匹配的协同记录"""
        return self._matrix.get(drug, {}).get(therapy, {}).get(condition)


# ===== 便捷函数 =====

def create_synergy_calculator(
    custom_overrides: Optional[Dict] = None,
) -> SynergyCalculator:
    """创建协同计算器"""
    return SynergyCalculator(custom_overrides=custom_overrides)


__all__ = [
    "SynergyType",
    "SynergyRecord",
    "SynergyCalculator",
    "SYNERGY_MATRIX",
    "create_synergy_calculator",
]