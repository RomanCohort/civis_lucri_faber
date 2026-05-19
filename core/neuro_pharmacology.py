"""Simulacrum 神经药理学接口

像药理实验一样操控 agent 的大脑:
- 麻醉 (anesthesia): 抑制特定脑区，降低输出但不完全关闭
- 激活 (activation): 增强特定脑区的活跃度
- 切除 (lesion): 完全禁用特定脑区
- 神经递质注射 (inject): 直接修改 DA/5-HT/NE/ACh 等浓度
- 拮抗剂 (antagonist): 阻断特定递质的受体
- 激动剂 (agonist): 增强特定递质的受体敏感性

用法:
    pharma = NeuroPharmacology(agent)

    # 神经递质操作
    pharma.inject('dopamine', 0.9)        # 注射多巴胺 → 狂热/创造力爆发
    pharma.inject('serotonin', 0.1)        # 耗竭5-HT → 情绪不稳定
    pharma.inject('cortisol', 0.8)         # 注射皮质醇 → 高压/简短回复

    # 脑区操作
    pharma.anesthetize('amygdala')          # 麻醉杏仁核 → 无恐惧反应
    pharma.activate('prefrontal_cortex')    # 激活前额叶 → 更理性/成熟
    pharma.lesion('hippocampus')            # 切除海马体 → 无法形成新记忆

    # 药物组合
    pharma.prescribe('antidepressant')      # SSRI类: ↑5-HT
    pharma.prescribe('stimulant')           # 兴奋剂: ↑DA, ↑NE
    pharma.prescribe('sedative')            # 镇静剂: ↑GABA, ↓arousal
    pharma.prescribe('anxiolytic')          # 抗焦虑: ↓cortisol, ↑GABA

    # 查询
    pharma.status()                         # 当前所有药物/操作状态
    pharma.reset()                          # 恢复所有操作
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


class Region(Enum):
    """可操控的脑区"""
    AMYGDALA = "amygdala"
    PREFRONTAL_CORTEX = "prefrontal_cortex"
    HIPPOCAMPUS = "hippocampus"
    BASAL_GANGLIA = "basal_ganglia"
    BRAINSTEM = "brainstem"
    LIMBIC = "limbic"                  # 杏仁核+丘脑整体
    CEREBELLUM = "cerebellum"
    SCN = "scn"                        # 昼夜节律
    HPA_AXIS = "hpa_axis"
    ANS = "ans"                        # 自主神经
    GLIAL = "glial"
    SENSORY = "sensory"                # 语言皮层+角回
    SOCIAL = "social_cognition"
    SELF_AWARENESS = "self_awareness"
    CURIOSITY = "curiosity"


class Neurotransmitter(Enum):
    """可操控的神经递质/激素"""
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    NOREPINEPHRINE = "norepinephrine"
    ACETYLCHOLINE = "acetylcholine"
    GABA = "gaba"
    GLUTAMATE = "glutamate"
    CORTISOL = "cortisol"
    MELATONIN = "melatonin"
    ADRENALINE = "adrenaline"
    OXYTOCIN = "oxytocin"
    BDNF = "bdnf"


# 脑区 → 影响的 _internal_state 键 映射
_REGION_STATE_MAP = {
    Region.AMYGDALA: {
        "suppress_keys": ["limbic_valence", "limbic_arousal", "limbic_emotion"],
        "suppress_to": {"limbic_valence": 0.0, "limbic_arousal": 0.3, "limbic_emotion": "neutral"},
        "activate_keys": ["limbic_arousal"],
        "activate_boost": {"limbic_arousal": 0.3},
    },
    Region.PREFRONTAL_CORTEX: {
        "suppress_keys": ["pfc_maturity", "pfc_inhibition"],
        "suppress_to": {"pfc_maturity": 0.1, "pfc_inhibition": 0.0},
        "activate_keys": ["pfc_maturity", "pfc_inhibition"],
        "activate_boost": {"pfc_maturity": 0.3, "pfc_inhibition": 0.3},
    },
    Region.HIPPOCAMPUS: {
        "suppress_keys": ["encoding_modulation"],
        "suppress_to": {"encoding_modulation": 0.1},
        "activate_keys": ["encoding_modulation"],
        "activate_boost": {"encoding_modulation": 0.3},
    },
    Region.BASAL_GANGLIA: {
        "suppress_keys": ["bg_habit_strength", "bg_td_error"],
        "suppress_to": {"bg_habit_strength": 0.0, "bg_td_error": 0.0},
        "activate_keys": ["bg_habit_strength"],
        "activate_boost": {"bg_habit_strength": 0.2},
    },
    Region.BRAINSTEM: {
        "suppress_keys": ["bsm_arousal", "bsm_consciousness_gate"],
        "suppress_to": {"bsm_arousal": 0.1, "bsm_consciousness_gate": 0.1},
        "activate_keys": ["bsm_arousal", "bsm_consciousness_gate"],
        "activate_boost": {"bsm_arousal": 0.3, "bsm_consciousness_gate": 0.3},
    },
    Region.HPA_AXIS: {
        "suppress_keys": ["cortisol_level", "hormone_cortisol"],
        "suppress_to": {"cortisol_level": 0.1, "hormone_cortisol": 0.1},
        "activate_keys": ["cortisol_level", "hormone_cortisol"],
        "activate_boost": {"cortisol_level": 0.3, "hormone_cortisol": 0.3},
    },
    Region.ANS: {
        "suppress_keys": ["ans_hrv"],
        "suppress_to": {"ans_hrv": 0.2},
        "activate_keys": ["ans_hrv"],
        "activate_boost": {"ans_hrv": 0.2},
    },
    Region.CURIOSITY: {
        "suppress_keys": [],
        "suppress_to": {},
        "activate_keys": [],
        "activate_boost": {},
    },
}

# 递质 → _internal_state 键 映射
_NT_STATE_MAP = {
    Neurotransmitter.DOPAMINE: "nt_dopamine",
    Neurotransmitter.SEROTONIN: "nt_serotonin",
    Neurotransmitter.CORTISOL: "cortisol_level",
    Neurotransmitter.MELATONIN: "scn_melatonin",
    Neurotransmitter.ADRENALINE: "hormone_adrenaline",
    Neurotransmitter.OXYTOCIN: "hormone_oxytocin",
    Neurotransmitter.BDNF: "plasticity_bdnf",
}

# 预设药物配方
_PRESETS = {
    "antidepressant": {
        "name": "抗抑郁药 (SSRI类)",
        "effects": "↑5-HT, ↑BDNF, ↓cortisol",
        "nt": {Neurotransmitter.SEROTONIN: 0.8, Neurotransmitter.BDNF: 0.7},
        "reduce": {Neurotransmitter.CORTISOL: 0.2},
    },
    "stimulant": {
        "name": "中枢兴奋剂 (安非他命类)",
        "effects": "↑DA, ↑NE, ↑arousal",
        "nt": {Neurotransmitter.DOPAMINE: 0.9, Neurotransmitter.NOREPINEPHRINE: 0.8},
        "activate": [Region.BRAINSTEM],
    },
    "sedative": {
        "name": "镇静剂 (苯二氮卓类)",
        "effects": "↑GABA, ↓arousal, ↓consciousness",
        "nt": {Neurotransmitter.GABA: 0.9},
        "anesthetize": [Region.BRAINSTEM, Region.AMYGDALA],
    },
    "anxiolytic": {
        "name": "抗焦虑药",
        "effects": "↓cortisol, ↑GABA, ↑HRV",
        "nt": {Neurotransmitter.GABA: 0.7},
        "reduce": {Neurotransmitter.CORTISOL: 0.2},
    },
    "nootropic": {
        "name": "益智药 (促智剂)",
        "effects": "↑ACh, ↑BDNF, ↑PFC activation",
        "nt": {Neurotransmitter.ACETYLCHOLINE: 0.8, Neurotransmitter.BDNF: 0.8},
        "activate": [Region.PREFRONTAL_CORTEX, Region.HIPPOCAMPUS],
    },
    "empathogen": {
        "name": "共情增强剂 (催产素类)",
        "effects": "↑oxytocin, ↑5-HT, ↑social",
        "nt": {Neurotransmitter.OXYTOCIN: 0.9, Neurotransmitter.SEROTONIN: 0.7},
        "activate": [Region.SOCIAL],
    },
    "hallucinogen": {
        "name": "致幻剂 (5-HT2A激动)",
        "effects": "↑↑5-HT, ↓PFC suppression, ↑sensory",
        "nt": {Neurotransmitter.SEROTONIN: 1.0},
        "anesthetize": [Region.PREFRONTAL_CORTEX],
        "activate": [Region.SENSORY],
    },
    "anesthetic": {
        "name": "全身麻醉剂",
        "effects": "↓↓arousal, ↓consciousness, 全脑抑制",
        "nt": {Neurotransmitter.GABA: 1.0},
        "anesthetize": [Region.BRAINSTEM, Region.AMYGDALA, Region.PREFRONTAL_CORTEX,
                        Region.HIPPOCAMPUS, Region.SENSORY],
    },
    "dopamine_antagonist": {
        "name": "多巴胺拮抗剂 (抗精神病)",
        "effects": "↓DA, ↓exploration, ↓reward sensitivity",
        "reduce": {Neurotransmitter.DOPAMINE: 0.1},
        "anesthetize": [Region.CURIOSITY],
    },
}


@dataclass
class DrugRecord:
    """药物/操作记录"""
    name: str
    kind: str          # "inject", "reduce", "anesthetize", "activate", "lesion", "preset"
    target: str        # 递质名/脑区名/药物名
    value: float       # 设置值
    step_applied: int  # 应用时的步数


class NeuroPharmacology:
    """神经药理学控制器

    允许像药理实验一样操控 agent 的大脑状态。
    所有操作记录在日志中，可随时 reset 恢复。
    """

    def __init__(self, agent):
        self.agent = agent
        self._log: List[DrugRecord] = []
        self._snapshots: Dict[str, dict] = {}  # 操作前的快照，用于 reset
        self._lesioned_regions: Set[str] = set()
        self._anesthetized_regions: Set[str] = set()
        self._activated_regions: Set[str] = set()
        self._nt_overrides: Dict[str, float] = {}  # 被手动覆盖的递质

    # ════════════════════════════════════════════════
    # 神经递质操作
    # ════════════════════════════════════════════════

    def inject(self, nt: str, concentration: float) -> str:
        """注射神经递质/激素

        Args:
            nt: 递质名 (dopamine, serotonin, cortisol, melatonin, etc.)
            concentration: 目标浓度 [0.0, 1.0]

        Returns:
            操作描述
        """
        concentration = max(0.0, min(1.0, concentration))
        state_key = self._resolve_nt_key(nt)
        if state_key is None:
            return f"[ERROR] Unknown neurotransmitter: {nt}"

        # 保存快照
        self._snapshot_nt(nt)
        self.agent._internal_state[state_key] = concentration
        self._nt_overrides[nt] = concentration
        self._update_derived_keys()

        self._log.append(DrugRecord(
            name=f"inject_{nt}", kind="inject", target=nt,
            value=concentration, step_applied=self.agent.step_count,
        ))

        desc = self._describe_nt_effect(nt, concentration)
        return f"[INJECT] {nt} → {concentration:.2f} ({desc})"

    def reduce(self, nt: str, concentration: float) -> str:
        """降低递质浓度（拮抗剂效果）"""
        state_key = self._resolve_nt_key(nt)
        if state_key is None:
            return f"[ERROR] Unknown neurotransmitter: {nt}"

        self._snapshot_nt(nt)
        concentration = max(0.0, min(1.0, concentration))
        self.agent._internal_state[state_key] = concentration
        self._nt_overrides[nt] = concentration
        self._update_derived_keys()

        self._log.append(DrugRecord(
            name=f"reduce_{nt}", kind="reduce", target=nt,
            value=concentration, step_applied=self.agent.step_count,
        ))
        return f"[REDUCE] {nt} → {concentration:.2f}"

    # ════════════════════════════════════════════════
    # 脑区操作
    # ════════════════════════════════════════════════

    def anesthetize(self, region: str, level: float = 0.1) -> str:
        """麻醉脑区（抑制但不完全关闭）

        Args:
            region: 脑区名 (amygdala, prefrontal_cortex, hippocampus, etc.)
            level: 抑制程度 [0=完全关闭, 1=无抑制]，默认 0.1
        """
        region_enum = self._resolve_region(region)
        if region_enum is None:
            return f"[ERROR] Unknown region: {region}"

        self._snapshot_region(region)
        mapping = _REGION_STATE_MAP.get(region_enum, {})
        suppress_to = mapping.get("suppress_to", {})

        for key, val in suppress_to.items():
            # level 控制抑制程度：level=0 完全抑制到 suppress_to 值，level=1 不抑制
            current = self.agent._internal_state.get(key, 0.5)
            if isinstance(val, (int, float)) and isinstance(current, (int, float)):
                target = val * level + current * (1 - level)
                self.agent._internal_state[key] = target
            else:
                # 非数值类型（如 emotion 字符串）直接替换
                self.agent._internal_state[key] = val

        self._anesthetized_regions.add(region)
        self._log.append(DrugRecord(
            name=f"anesthetize_{region}", kind="anesthetize", target=region,
            value=level, step_applied=self.agent.step_count,
        ))
        return f"[ANESTHESIA] {region} 抑制至 {level:.1f}"

    def activate(self, region: str, boost: float = 0.3) -> str:
        """激活脑区（增强活跃度）

        Args:
            region: 脑区名
            boost: 增强幅度 [0.0, 1.0]，默认 0.3
        """
        region_enum = self._resolve_region(region)
        if region_enum is None:
            return f"[ERROR] Unknown region: {region}"

        self._snapshot_region(region)
        mapping = _REGION_STATE_MAP.get(region_enum, {})
        activate_boost = mapping.get("activate_boost", {})

        for key, val in activate_boost.items():
            current = self.agent._internal_state.get(key, 0.5)
            self.agent._internal_state[key] = min(1.0, current + val * boost)

        # 特殊处理
        if region_enum == Region.CURIOSITY:
            self.agent.config.exploration_rate = min(0.5, self.agent.config.exploration_rate * 1.5)
        if region_enum == Region.PREFRONTAL_CORTEX:
            self.agent._internal_state['pfc_maturity'] = min(
                1.0, self.agent._internal_state.get('pfc_maturity', 0.5) + boost
            )

        self._activated_regions.add(region)
        self._log.append(DrugRecord(
            name=f"activate_{region}", kind="activate", target=region,
            value=boost, step_applied=self.agent.step_count,
        ))
        return f"[ACTIVATE] {region} 激活 +{boost:.1f}"

    def lesion(self, region: str) -> str:
        """切除脑区（完全禁用）

        模拟脑损伤/手术切除。被切除的脑区在 step() 中仍运行，
       但其输出被钳制到基线/零值。
        """
        region_enum = self._resolve_region(region)
        if region_enum is None:
            return f"[ERROR] Unknown region: {region}"

        self._snapshot_region(region)
        mapping = _REGION_STATE_MAP.get(region_enum, {})
        suppress_to = mapping.get("suppress_to", {})

        for key, val in suppress_to.items():
            self.agent._internal_state[key] = val

        # 额外强制归零
        if region_enum == Region.HIPPOCAMPUS:
            self.agent._internal_state['encoding_modulation'] = 0.0
        if region_enum == Region.AMYGDALA:
            self.agent._internal_state['limbic_valence'] = 0.0
            self.agent._internal_state['limbic_arousal'] = 0.1
        if region_enum == Region.BRAINSTEM:
            self.agent._internal_state['bsm_consciousness_gate'] = 0.05
        if region_enum == Region.CURIOSITY:
            self.agent.config.exploration_rate = 0.01

        self._lesioned_regions.add(region)
        self._log.append(DrugRecord(
            name=f"lesion_{region}", kind="lesion", target=region,
            value=0.0, step_applied=self.agent.step_count,
        ))
        return f"[LESION] {region} 已切除"

    # ════════════════════════════════════════════════
    # 预设药物
    # ════════════════════════════════════════════════

    def prescribe(self, drug: str) -> str:
        """开处方药物（预设组合）

        可用药物: antidepressant, stimulant, sedative, anxiolytic,
                  nootropic, empathogen, hallucinogen, anesthetic,
                  dopamine_antagonist
        """
        preset = _PRESETS.get(drug)
        if preset is None:
            available = ", ".join(_PRESETS.keys())
            return f"[ERROR] Unknown drug '{drug}'. Available: {available}"

        results = [f"[PRESCRIBE] {preset['name']}: {preset['effects']}"]

        # 注射递质
        for nt, conc in preset.get("nt", {}).items():
            results.append(self.inject(nt.value if isinstance(nt, Neurotransmitter) else nt, conc))

        # 降低递质
        for nt, conc in preset.get("reduce", {}).items():
            results.append(self.reduce(nt.value if isinstance(nt, Neurotransmitter) else nt, conc))

        # 麻醉脑区
        for region in preset.get("anesthetize", []):
            r = region.value if isinstance(region, Region) else region
            results.append(self.anesthetize(r))

        # 激活脑区
        for region in preset.get("activate", []):
            r = region.value if isinstance(region, Region) else region
            results.append(self.activate(r))

        return "\n".join(results)

    # ════════════════════════════════════════════════
    # 查询与恢复
    # ════════════════════════════════════════════════

    def status(self) -> str:
        """当前所有药理学状态"""
        lines = ["=== 神经药理学状态 ==="]

        if self._lesioned_regions:
            lines.append(f"切除: {', '.join(self._lesioned_regions)}")
        if self._anesthetized_regions:
            lines.append(f"麻醉: {', '.join(self._anesthetized_regions)}")
        if self._activated_regions:
            lines.append(f"激活: {', '.join(self._activated_regions)}")
        if self._nt_overrides:
            nt_str = ", ".join(f"{k}={v:.2f}" for k, v in self._nt_overrides.items())
            lines.append(f"递质覆盖: {nt_str}")

        if not any([self._lesioned_regions, self._anesthetized_regions,
                     self._activated_regions, self._nt_overrides]):
            lines.append("(无活跃药物/操作)")

        # 当前关键递质值
        s = self.agent._internal_state
        lines.append(f"\n关键指标:")
        for nt_name, state_key in [("DA", "nt_dopamine"), ("5-HT", "nt_serotonin"),
                                    ("Cortisol", "cortisol_level"), ("BDNF", "plasticity_bdnf"),
                                    ("Oxytocin", "hormone_oxytocin"), ("Melatonin", "scn_melatonin")]:
            val = s.get(state_key, 0.5)
            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
            lines.append(f"  {nt_name:10s} {bar} {val:.3f}")

        lines.append(f"\n操作记录: {len(self._log)} 条")
        return "\n".join(lines)

    def reset(self) -> str:
        """恢复所有药理学操作"""
        # 恢复递质快照
        for nt, snapshot in self._snapshots.items():
            if nt.startswith("nt_"):
                state_key = self._resolve_nt_key(nt.replace("nt_", ""))
                if state_key:
                    for k, v in snapshot.items():
                        self.agent._internal_state[k] = v

        self._lesioned_regions.clear()
        self._anesthetized_regions.clear()
        self._activated_regions.clear()
        self._nt_overrides.clear()
        self._snapshots.clear()
        self._log.clear()

        # 恢复探索率
        self.agent.config.exploration_rate = 0.1

        return "[RESET] 所有药理学操作已恢复"

    def get_log(self) -> List[Dict]:
        """获取操作日志"""
        return [
            {"step": r.step_applied, "kind": r.kind, "target": r.target,
             "value": r.value, "name": r.name}
            for r in self._log
        ]

    # ════════════════════════════════════════════════
    # 内部方法
    # ════════════════════════════════════════════════

    def _resolve_nt_key(self, name: str) -> Optional[str]:
        """递质名 → _internal_state 键"""
        name_lower = name.lower()
        aliases = {
            "dopamine": "nt_dopamine", "da": "nt_dopamine",
            "serotonin": "nt_serotonin", "5-ht": "nt_serotonin", "5ht": "nt_serotonin",
            "cortisol": "cortisol_level",
            "melatonin": "scn_melatonin",
            "adrenaline": "hormone_adrenaline", "epinephrine": "hormone_adrenaline",
            "oxytocin": "hormone_oxytocin",
            "bdnf": "plasticity_bdnf",
            "gaba": "nt_gaba",
            "norepinephrine": "nt_norepinephrine",
            "ne": "nt_norepinephrine",
            "acetylcholine": "nt_acetylcholine",
            "ach": "nt_acetylcholine",
            "glutamate": "nt_glutamate",
        }
        return aliases.get(name_lower)

    def _resolve_region(self, name: str) -> Optional[Region]:
        """区域名 → Region 枚举"""
        name_lower = name.lower().replace(" ", "_")
        aliases = {
            "amygdala": Region.AMYGDALA,
            "prefrontal": Region.PREFRONTAL_CORTEX, "pfc": Region.PREFRONTAL_CORTEX,
            "prefrontal_cortex": Region.PREFRONTAL_CORTEX,
            "hippocampus": Region.HIPPOCAMPUS,
            "basal_ganglia": Region.BASAL_GANGLIA, "bg": Region.BASAL_GANGLIA,
            "brainstem": Region.BRAINSTEM,
            "limbic": Region.LIMBIC,
            "cerebellum": Region.CEREBELLUM,
            "scn": Region.SCN, "circadian": Region.SCN,
            "hpa": Region.HPA_AXIS, "hpa_axis": Region.HPA_AXIS,
            "ans": Region.ANS, "autonomic": Region.ANS,
            "glial": Region.GLIAL,
            "sensory": Region.SENSORY,
            "social": Region.SOCIAL, "social_cognition": Region.SOCIAL,
            "self_awareness": Region.SELF_AWARENESS,
            "curiosity": Region.CURIOSITY,
        }
        return aliases.get(name_lower)

    def _snapshot_nt(self, nt: str) -> None:
        """保存递质操作前的快照"""
        key = f"nt_{nt}"
        if key not in self._snapshots:
            state_key = self._resolve_nt_key(nt)
            if state_key:
                self._snapshots[key] = {state_key: self.agent._internal_state.get(state_key, 0.5)}

    def _update_derived_keys(self) -> None:
        """更新派生键 (ans_hrv 从 nt_gaba + nt_norepinephrine 计算)"""
        from core.state_key_mapping import UnifiedStateMapping
        UnifiedStateMapping.update_derived_keys(self.agent._internal_state)

    def _snapshot_region(self, region: str) -> None:
        """保存脑区操作前的快照"""
        key = f"region_{region}"
        if key not in self._snapshots:
            region_enum = self._resolve_region(region)
            mapping = _REGION_STATE_MAP.get(region_enum, {}) if region_enum else {}
            all_keys = list(mapping.get("suppress_to", {}).keys()) + list(mapping.get("activate_boost", {}).keys())
            snapshot = {}
            for k in all_keys:
                snapshot[k] = self.agent._internal_state.get(k, 0.5)
            self._snapshots[key] = snapshot

    @staticmethod
    def _describe_nt_effect(nt: str, conc: float) -> str:
        """描述递质浓度变化的效应"""
        effects = {
            "dopamine": {0.1: "冷漠/快感缺失", 0.5: "正常动机水平",
                         0.9: "狂热/极度好奇/创造力爆发"},
            "serotonin": {0.1: "抑郁/情绪不稳", 0.5: "情绪稳定",
                          0.9: "极度平和/满足"},
            "cortisol": {0.1: "极度放松", 0.5: "正常压力",
                         0.9: "高压/焦虑/回复极短"},
            "melatonin": {0.1: "完全清醒", 0.5: "正常",
                          0.9: "极度困倦/反应迟钝"},
            "oxytocin": {0.1: "社交回避/不信任", 0.5: "正常社交",
                         0.9: "极度信任/共情增强"},
            "bdnf": {0.1: "学习能力极低", 0.5: "正常学习",
                      0.9: "超常学习能力"},
        }
        nt_effects = effects.get(nt, {})
        if conc <= 0.3:
            return nt_effects.get(0.1, "低水平")
        elif conc <= 0.7:
            return nt_effects.get(0.5, "中等水平")
        else:
            return nt_effects.get(0.9, "高水平")
