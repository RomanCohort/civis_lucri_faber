"""计算精神药理学治疗实验 — TherapeuticExperiment

桥接PK/PD管线与agent _internal_state的核心实验框架。
支持药物单用、心理治疗单用、联合疗法、多药物组合。

数据流:
  DrugConfig → ADMET → PK/PD params → simulate_pkpd → C(t)
  step loop: C(t) → E(t) via Hill equation → PDTargets → _internal_state
  + therapy sessions with synergy modulation
  + disease dynamics
  + psychometric indicator tracking
  + LLM evaluation at key timepoints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from simulacrum.core.state_key_mapping import UnifiedStateMapping
from simulacrum.core.pd_target_mapping import PDTarget, build_pd_targets, compute_pd_deltas


# ══════════════════════════════════════════════════════════════
# 配置数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class DrugConfig:
    """药物配置 — 支持注册名或SMILES。"""
    name_or_smiles: str
    dose_mg: float
    freq_per_day: float = 1.0
    start_step: int = 0
    end_step: Optional[int] = None


@dataclass
class TherapyConfig:
    """心理治疗配置。"""
    modality: str           # "CBT" / "exposure" / "DBT" / "EMDR" / ...
    frequency: str = "weekly"
    intensity: float = 0.7
    start_step: int = 0
    end_step: Optional[int] = None


@dataclass
class ExperimentConfig:
    """完整实验配置。"""
    condition: str          # "MDD" / "GAD" / "PTSD" / ...
    severity: str = "moderate"  # "mild" / "moderate" / "severe"
    drugs: List[DrugConfig] = field(default_factory=list)
    therapies: List[TherapyConfig] = field(default_factory=list)
    duration_steps: int = 2000
    follow_up_steps: int = 500
    observation_interval: int = 10
    llm_evaluation_timepoints: Optional[List[int]] = None
    steps_per_hour: float = 10.0  # 模拟步数/小时

    def __post_init__(self):
        if self.llm_evaluation_timepoints is None:
            d = self.duration_steps
            f = self.follow_up_steps
            self.llm_evaluation_timepoints = [
                0,
                d // 4,
                d // 2,
                3 * d // 4,
                d,
                d + f,
            ]


# ══════════════════════════════════════════════════════════════
# 时间点快照
# ══════════════════════════════════════════════════════════════

@dataclass
class TreatmentTimepoint:
    """一个观测时间点的完整快照。"""
    step: int
    time_h: float
    phase: str              # "treatment" / "follow_up"
    # 药物状态
    drug_concentrations: Dict[str, float]   # {drug_id: C(mg/L)}
    drug_pd_effects: Dict[str, float]       # {drug_id: E(0-Emax)}
    # 神经递质
    neurotransmitters: Dict[str, float]     # {canonical_key: value}
    # 脑区
    brain_regions: Dict[str, float]
    # 症状严重度 (由 PsychometricIndicatorTracker 填充)
    psychometrics: Optional[Dict[str, float]] = None
    # 疗法
    therapy_sessions_total: int = 0
    therapy_skill: float = 0.0
    synergy_factor: float = 0.0
    # 病理检测
    warnings: List[str] = field(default_factory=list)
    # 症状追踪 (SymptomTracker)
    symptom_snapshot: Optional[Any] = None
    # 成瘾动力学
    tolerance_factors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    craving_levels: Dict[str, float] = field(default_factory=dict)
    withdrawal_severity: float = 0.0


@dataclass
class TherapeuticResult:
    """完整治疗实验结果。"""
    config: ExperimentConfig
    trajectory: List[TreatmentTimepoint] = field(default_factory=list)
    # 汇总指标
    remission_rate: float = 0.0        # 症状<0.3的时间占比
    relapse_rate: float = 0.0          # 随访期症状反弹率
    peak_drug_effect: float = 0.0
    total_therapy_sessions: int = 0
    # LLM评估结果
    llm_evaluations: List[Dict] = field(default_factory=list)
    # 药物信息追踪
    drug_info: Dict[str, Dict] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# 主实验类
# ══════════════════════════════════════════════════════════════

class TherapeuticExperiment:
    """计算精神药理学治疗实验。

    桥接PK/PD管线与agent _internal_state，整合疗法与协同效应，
    追踪心理测量指标，支持LLM评估。
    """

    def __init__(
        self,
        agent: Optional[Any] = None,
        config: Optional[ExperimentConfig] = None,
    ):
        self.agent = agent
        self.config = config or ExperimentConfig(
            condition="MDD",
            drugs=[],
            therapies=[],
        )
        self._state: Dict[str, Any] = {}
        self._state_mapping = UnifiedStateMapping()

        # 每个药物的PK/PD数据
        self._pk_curves: Dict[str, pd.DataFrame] = {}
        self._pkpd_params: Dict[str, Any] = {}        # PKPDParams per drug
        self._pd_targets: Dict[str, List[PDTarget]] = {}
        self._drug_defs: Dict[str, Any] = {}
        self._drug_admet: Dict[str, Any] = {}
        self._drug_ic50: Dict[str, Any] = {}

        # 疗法状态
        self._therapy_session_count: int = 0
        self._therapy_skill: float = 0.0
        self._therapy_alliance: float = 0.3

        # 药物效应追踪
        self._current_pd_effects: Dict[str, float] = {}
        self._current_concentrations: Dict[str, float] = {}
        self._current_synergy: float = 0.0

        # DDI状态
        self._ddi_result: Optional[Any] = None
        self._ke_modifiers_baseline: Dict[str, float] = {}
        self._current_ke_modifiers: Dict[str, float] = {}

        # 受体层PD状态
        self._receptor_pd_targets: Dict[str, List] = {}  # {drug_id: [ReceptorPDTarget]}

        # 成瘾动力学引擎
        self._addiction_engine: Optional[Any] = None

        # 症状追踪器
        self._symptom_tracker: Optional[Any] = None

        # 病原体炎症引擎
        self._pathogen_engine: Optional[Any] = None

    def prepare(self) -> None:
        """预计算PK曲线和PD靶点映射。

        对每个DrugConfig:
        1. 从注册表或SMILES解析药物
        2. ADMET预测
        3. 推导PK/PD参数
        4. 运行simulate_pkpd获取C(t)
        5. 构建PD靶点映射

        初始化agent状态:
        - 获取或创建 _internal_state
        - 应用精神疾病档案
        """
        from core.drug_pipeline.admet import predict_admet
        from core.drug_pipeline.pkpd import simulate_pkpd
        from core.drug_pipeline.risk_to_ic50 import (
            admet_to_ic50,
            smiles_to_pkpd_params,
        )
        from core.drug_pipeline.drug_registry import get_drug
        from core.drug_pipeline.pkpd import PKPDParams

        # 获取agent状态
        if self.agent is not None and hasattr(self.agent, "_internal_state"):
            self._state = self.agent._internal_state
        else:
            self._state = self._default_state()

        # 对每个药物准备PK/PD
        for i, drug_cfg in enumerate(self.config.drugs):
            drug_id = f"drug_{i}"
            name = drug_cfg.name_or_smiles.lower().strip()

            # 尝试从注册表获取
            drug_def = get_drug(name)
            if drug_def is None:
                # 当作SMILES处理
                from core.drug_pipeline.drug_registry import DrugDefinition
                drug_def = DrugDefinition(
                    name=name,
                    smiles=name,
                    drug_class="default",
                    therapeutic_dose_mg=drug_cfg.dose_mg,
                    freq_per_day=drug_cfg.freq_per_day,
                    target_neurotransmitters={"dopamine": "increase"},
                    therapeutic_ed50_mgkg=2.0,
                )

            self._drug_defs[drug_id] = drug_def

            # ADMET预测
            admet_result = predict_admet(drug_def.smiles)
            self._drug_admet[drug_id] = admet_result

            # PK/PD参数
            pkpd_params = smiles_to_pkpd_params(
                drug_def.smiles, drug_cfg.dose_mg, admet_result,
            )
            self._pkpd_params[drug_id] = pkpd_params

            # IC50档案
            ic50 = admet_to_ic50(admet_result, drug_def.drug_class)
            self._drug_ic50[drug_id] = ic50

            # PK曲线 — 模拟完整实验时长
            total_hours = (self.config.duration_steps + self.config.follow_up_steps) / self.config.steps_per_hour
            horizon = max(int(total_hours), 24)
            curve = simulate_pkpd(
                dose_mg=drug_cfg.dose_mg,
                freq_per_day=drug_cfg.freq_per_day,
                params=pkpd_params,
                horizon=horizon,
                dt=0.5,
            )
            self._pk_curves[drug_id] = curve

            # PD靶点映射
            self._pd_targets[drug_id] = build_pd_targets(
                drug_def.target_neurotransmitters,
                drug_def.drug_class,
            )

            # 受体层PD (当药物有receptor_targets时)
            if drug_def.receptor_targets:
                from core.drug_pipeline.receptor_pd import build_receptor_pd_targets
                rct_targets, rct_deltas = build_receptor_pd_targets(
                    name, drug_effect=0.5
                )
                self._receptor_pd_targets[drug_id] = rct_targets

        # 应用疾病档案
        self._apply_condition()

        # ── DDI评估 (多药时) ──
        if len(self.config.drugs) > 1:
            from core.drug_pipeline.ddi import assess_ddi
            drug_names = []
            for i, drug_cfg in enumerate(self.config.drugs):
                did = f"drug_{i}"
                ddef = self._drug_defs[did]
                drug_names.append(ddef.name.split()[0].lower())
            self._ddi_result = assess_ddi(
                drug_names, self._drug_defs, self._drug_admet,
            )
            self._ke_modifiers_baseline = dict(self._ddi_result.ke_modifiers)
            if self._ddi_result.warnings:
                print("\n[DDI Assessment]")
                for w in self._ddi_result.warnings:
                    print(f"  ⚠ {w}")

    def run(self) -> TherapeuticResult:
        """执行完整治疗实验。

        主循环每步:
        1. 计算PD效应 (从C(t)插值 → Hill方程 → 靶点分配)
        2. 应用PD效应到 _internal_state
        3. 疗法session (若到时间)
        4. 疾病动力学
        5. 记录指标
        """
        total_steps = self.config.duration_steps + self.config.follow_up_steps
        result = TherapeuticResult(config=self.config)

        # 保存baseline
        baseline_nt = self._snapshot_neurotransmitters()

        # 疗法频率 → 步间隔
        therapy_interval = self._therapy_frequency_to_steps(
            self.config.therapies[0].frequency if self.config.therapies else "weekly"
        )

        symptom_below_threshold_count = 0
        follow_up_symptom_rebound = False
        last_treatment_symptom = 0.5

        for step in range(total_steps):
            time_h = step / self.config.steps_per_hour
            is_follow_up = step >= self.config.duration_steps
            phase = "follow_up" if is_follow_up else "treatment"

            # ── 1. 药物PD效应 ──
            per_drug_deltas: Dict[str, Dict[str, float]] = {}
            for drug_id, drug_cfg in enumerate(self.config.drugs):
                did = f"drug_{drug_id}"

                # 随访期: 不再给药但PK继续衰减
                # (simulate_pkpd已经包含多次给药的完整曲线)
                # 检查药物是否在有效期内
                drug_active = (step >= drug_cfg.start_step and
                               (drug_cfg.end_step is None or step < drug_cfg.end_step))

                if not drug_active and not is_follow_up:
                    self._current_concentrations[did] = 0.0
                    self._current_pd_effects[did] = 0.0
                    continue

                # 从PK曲线插值 (DDI感知)
                conc, pd_effect = self._interpolate_pk(did, time_h)
                self._current_concentrations[did] = conc
                self._current_pd_effects[did] = pd_effect

                # 随访期不应用新效应
                if is_follow_up and step == self.config.duration_steps:
                    last_treatment_symptom = self._compute_symptom_severity()

                if not is_follow_up:
                    # 分配PD效应到各NT靶点
                    targets = self._pd_targets.get(did, [])
                    deltas = compute_pd_deltas(targets, pd_effect)
                    per_drug_deltas[did] = deltas

            # DDI: 浓度依赖ke修正
            if self._ddi_result and len(self.config.drugs) > 1:
                from core.drug_pipeline.ddi import compute_step_ke_modifiers
                cmax_refs = {}
                conc_by_name = {}
                for i, drug_cfg in enumerate(self.config.drugs):
                    did = f"drug_{i}"
                    ddef = self._drug_defs[did]
                    drug_name = ddef.name.split()[0].lower()
                    params = self._pkpd_params.get(did)
                    if params:
                        cmax_refs[drug_name] = drug_cfg.dose_mg / max(params.vd_l_per_kg * 70, 1.0)
                    else:
                        cmax_refs[drug_name] = 1.0
                    conc_by_name[drug_name] = self._current_concentrations.get(did, 0.0)
                drug_names = [self._drug_defs[f"drug_{i}"].name.split()[0].lower()
                              for i in range(len(self.config.drugs))]
                ke_mods_by_name = compute_step_ke_modifiers(
                    drug_names, conc_by_name, cmax_refs, self._ddi_result,
                )
                # 映射 drug_name → drug_i ID
                self._current_ke_modifiers = {}
                for i in range(len(self.config.drugs)):
                    did = f"drug_{i}"
                    drug_name = self._drug_defs[did].name.split()[0].lower()
                    self._current_ke_modifiers[did] = ke_mods_by_name.get(drug_name, 1.0)

            # 组合PD效应 (DDI感知: Bliss/拮抗/协同 替代朴素求和)
            if not is_follow_up:
                if len(per_drug_deltas) > 1 and self._ddi_result:
                    from core.drug_pipeline.ddi import combine_pd_deltas
                    all_pd_deltas, pd_warnings = combine_pd_deltas(
                        per_drug_deltas, self._pd_targets, self._ddi_result,
                    )
                else:
                    # 单药或无DDI: 直接合并
                    all_pd_deltas: Dict[str, float] = {}
                    for deltas in per_drug_deltas.values():
                        for k, v in deltas.items():
                            all_pd_deltas[k] = all_pd_deltas.get(k, 0.0) + v
                    pd_warnings = []

                self._apply_pd_effects(all_pd_deltas)

            # ── 2. 疗法 ──
            if self.config.therapies and not is_follow_up:
                therapy_cfg = self.config.therapies[0]
                if (step >= therapy_cfg.start_step and
                        step % therapy_interval == 0):
                    # 计算synergy
                    avg_pd = np.mean(list(self._current_pd_effects.values())) if self._current_pd_effects else 0.0
                    self._current_synergy = self._compute_synergy(avg_pd)

                    # 应用疗法
                    self._apply_therapy_session(therapy_cfg)
                    self._therapy_session_count += 1

            # ── 3. 疾病动力学 ──
            self._step_disease_dynamics()

            # ── 4. 更新派生键 ──
            UnifiedStateMapping.update_derived_keys(self._state)

            # ── 5. 随访期Ebbinghaus衰减 ──
            if is_follow_up:
                decay = np.exp(-0.002 * (step - self.config.duration_steps))
                self._therapy_skill *= decay

            # ── 6. 记录指标 ──
            if step % self.config.observation_interval == 0:
                symptom = self._compute_symptom_severity()
                if symptom < 0.3:
                    symptom_below_threshold_count += 1

                tp = TreatmentTimepoint(
                    step=step,
                    time_h=time_h,
                    phase=phase,
                    drug_concentrations=dict(self._current_concentrations),
                    drug_pd_effects=dict(self._current_pd_effects),
                    neurotransmitters=self._snapshot_neurotransmitters(),
                    brain_regions=self._snapshot_brain_regions(),
                    therapy_sessions_total=self._therapy_session_count,
                    therapy_skill=self._therapy_skill,
                    synergy_factor=self._current_synergy,
                    warnings=self._detect_pathological_states(),
                )
                result.trajectory.append(tp)

            # ── 7. LLM评估 (稍后集成) ──
            # if step in self.config.llm_evaluation_timepoints:
            #     evaluation = self._llm_evaluate(tp)
            #     result.llm_evaluations.append(evaluation)

        # 汇总
        treatment_obs = sum(1 for tp in result.trajectory if tp.phase == "treatment")
        if treatment_obs > 0:
            result.remission_rate = symptom_below_threshold_count / treatment_obs
        result.total_therapy_sessions = self._therapy_session_count
        result.peak_drug_effect = max(
            (max(e.values()) if e else 0.0)
            for tp in result.trajectory
            for e in [tp.drug_pd_effects]
        )

        # 药物信息
        for did, ddef in self._drug_defs.items():
            result.drug_info[did] = {
                "name": ddef.name,
                "smiles": ddef.smiles,
                "class": ddef.drug_class,
                "dose_mg": self.config.drugs[int(did.split("_")[1])].dose_mg,
            }

        return result

    # ── PK插值 ──

    def _interpolate_pk(self, drug_id: str, time_h: float) -> tuple:
        """从预计算的PK曲线插值当前浓度和PD效应。

        DDI: ke修正 → 浓度升高 → PD效应重算
        """
        curve = self._pk_curves.get(drug_id)
        if curve is None or curve.empty:
            return 0.0, 0.0

        conc = float(np.interp(time_h, curve["time_h"], curve["pkpd_conc_mg_per_l"]))
        effect = float(np.interp(time_h, curve["time_h"], curve["pkpd_effect"]))

        # DDI: ke修正 → 浓度升高 → 重算PD效应
        ke_mod = self._current_ke_modifiers.get(drug_id, 1.0)
        if ke_mod < 1.0:
            # AUC增加 = 浓度增加: conc_modified = conc / ke_modifier
            conc_modified = conc / ke_mod
            params = self._pkpd_params.get(drug_id)
            if params:
                h = max(getattr(params, "hill", 1.0), 0.1)
                ec50 = max(getattr(params, "ec50_mg_per_l", 1.0), 1e-6)
                emax = getattr(params, "emax", 1.0)
                ec50_h = ec50 ** h
                effect = emax * (conc_modified ** h) / (ec50_h + conc_modified ** h)
            conc = conc_modified

        return max(0.0, conc), max(0.0, effect)

    # ── PD效应应用 ──

    def _apply_pd_effects(self, deltas: Dict[str, float]) -> None:
        """将PD delta值应用到_internal_state。"""
        for canonical_key, delta in deltas.items():
            current = float(self._state.get(canonical_key, 0.5))
            # delta是效应方向和强度的乘积
            new_val = current + delta * 0.3  # 缩放因子防止过冲
            self._state[canonical_key] = max(0.0, min(1.0, new_val))

    # ── 疗法 ──

    def _apply_therapy_session(self, cfg: TherapyConfig) -> None:
        """简化版疗法session — 调整脑区状态。"""
        from core.psychotherapy import THERAPY_TARGETS

        modality = cfg.modality.lower()
        targets = THERAPY_TARGETS.get(modality, THERAPY_TARGETS.get(cfg.modality, {}))
        if not targets:
            return

        compliance = max(0.3, 1.0 - 0.3 * (1.0 - self._therapy_alliance))
        synergy_bonus = max(0.0, self._current_synergy)

        for state_key, delta in targets.items():
            resolved = UnifiedStateMapping.resolve(state_key)
            current = float(self._state.get(resolved, 0.5))
            adjustment = delta * cfg.intensity * compliance * (1.0 + synergy_bonus)
            new_val = current + adjustment
            self._state[resolved] = max(0.0, min(1.0, new_val))

        # 更新联盟和技能
        self._therapy_alliance = min(1.0, self._therapy_alliance + 0.01)
        self._therapy_skill = min(1.0, self._therapy_skill + 0.02 * cfg.intensity)

    # ── 疾病动力学 ──

    def _step_disease_dynamics(self) -> None:
        """简化的疾病压力+自然回归。"""
        from core.psychiatric_simulation import PSYCHIATRIC_PROFILES, SEVERITY_MULTIPLIERS

        profile = PSYCHIATRIC_PROFILES.get(self.config.condition)
        if profile is None:
            return

        severity_mult = SEVERITY_MULTIPLIERS.get(self.config.severity, 0.6)
        overrides = profile.get("subsystem_overrides", {})

        for subsystem, params in overrides.items():
            for key, target_val in params.items():
                state_key = f"{subsystem}_{key}"
                resolved = UnifiedStateMapping.resolve(state_key)
                current = float(self._state.get(resolved, 0.5))

                # 疾病压力 (向目标漂移)
                if isinstance(target_val, (int, float)):
                    disease_pressure = (float(target_val) - current) * 0.005 * severity_mult
                    # 自然回归 (向0.5)
                    natural_regression = (0.5 - current) * 0.003
                    new_val = current + disease_pressure + natural_regression
                    self._state[resolved] = max(0.0, min(1.0, new_val))

    # ── 症状严重度 ──

    def _compute_symptom_severity(self) -> float:
        """计算当前症状严重度 [0, 1]。"""
        from core.psychiatric_simulation import PSYCHIATRIC_PROFILES

        condition = self.config.condition
        s = self._state

        if condition in ("MDD", "Dysthymia", "bipolar_depression",
                          "Burnout", "Prolonged_Grief", "Cyclothymia"):
            valence = float(s.get("limbic_valence", 0.0))
            da = float(s.get("nt_dopamine", 0.5))
            sht = float(s.get("nt_serotonin", 0.5))
            cortisol = float(s.get("cortisol_level", 0.3))
            return np.clip((-valence + (1 - da) + (1 - sht) + (cortisol - 0.3)) / 4, 0, 1)

        elif condition in ("GAD", "Panic_Disorder", "Social_Anxiety",
                           "Specific_Phobia", "Agoraphobia"):
            arousal = float(s.get("limbic_arousal", 0.5))
            ne = float(s.get("nt_norepinephrine", 0.3))
            cortisol = float(s.get("cortisol_level", 0.3))
            return np.clip(((arousal - 0.5) + ne + (cortisol - 0.3)) / 3 * 2, 0, 1)

        elif condition == "PTSD":
            arousal = float(s.get("limbic_arousal", 0.5))
            stress = float(s.get("hpa_axis_stress_reactivity_mult", 1.0))
            pfc = float(s.get("prefrontal_inhibition", 0.5))
            return np.clip(((arousal - 0.5) + (stress - 1) + (1 - pfc)) / 3 * 2, 0, 1)

        elif condition == "BPD":
            volatility = float(s.get("mood_system_volatility_mult", 1.0))
            coherence = float(s.get("self_awareness_coherence", 0.5))
            regulation = float(s.get("emotion_regulation_regulation_capacity", 0.5))
            return np.clip((volatility - 1 + (1 - coherence) + (1 - regulation)) / 3, 0, 1)

        elif condition in ("OCD",):
            precision = float(s.get("predictive_coding_precision_mult", 1.0))
            pfc = float(s.get("prefrontal_inhibition", 0.5))
            da = float(s.get("nt_dopamine", 0.5))
            return np.clip(((precision - 1) + (1 - pfc) + (da - 0.5)) / 3 * 2, 0, 1)

        else:
            # 通用严重度
            sht = float(s.get("nt_serotonin", 0.5))
            da = float(s.get("nt_dopamine", 0.5))
            valence = float(s.get("limbic_valence", 0.0))
            return np.clip((1 - sht + 1 - da - valence) / 3, 0, 1)

    def _compute_symptom_snapshot(self, current_step: int = 0, time_h: float = 0.0) -> Any:
        """使用SymptomTracker计算多症状快照。

        Returns:
            SymptomSnapshot with per-symptom detection and composite scores
        """
        from core.symptom_tracker import SymptomTracker

        if self._symptom_tracker is None:
            self._symptom_tracker = SymptomTracker()

        return self._symptom_tracker.step(
            state=self._state,
            current_step=current_step,
            time_h=time_h,
        )

    def _compute_addiction_step(self, current_step: int = 0) -> Dict[str, Any]:
        """更新成瘾动力学 (耐受/戒断/渴求)。

        Returns:
            {tolerance_factors, withdrawal_deltas, craving_levels}
        """
        from core.addiction_dynamics import AddictionDynamicsEngine

        if self._addiction_engine is None:
            self._addiction_engine = AddictionDynamicsEngine()
            # 注册所有有addiction_risk的药物
            for drug_id, drug_def in self._drug_defs.items():
                if drug_def.addiction_risk not in ("low", ""):
                    self._addiction_engine.register_drug(
                        drug_id, drug_def.drug_class
                    )

        if not self._addiction_engine.profiles:
            return {"tolerance_factors": {}, "withdrawal_deltas": {}, "craving_levels": {}}

        concentrations = {}
        effects = {}
        for drug_id, conc in self._current_concentrations.items():
            concentrations[drug_id] = conc
            effects[drug_id] = self._current_pd_effects.get(drug_id, 0.0)

        tol, wd, cr = self._addiction_engine.step(concentrations, effects)

        # 应用戒断delta到state
        for k, v in wd.items():
            if k in self._state:
                self._state[k] = float(np.clip(self._state[k] + v * 0.05, 0.0, 1.0))

        # 应用渴求到state
        for drug_id, level in cr.items():
            self._state["craving_level"] = float(np.clip(level, 0.0, 1.0))

        return {
            "tolerance_factors": tol,
            "withdrawal_deltas": wd,
            "craving_levels": cr,
        }

    def _compute_pathogen_step(self, treatment_efficacy: Dict[str, float] = None) -> Dict[str, Any]:
        """更新病原体神经炎症引擎。

        Args:
            treatment_efficacy: {pathogen_name: efficacy (0-1)}

        Returns:
            {damage_signal, cytokine_boost, state_deltas}
        """
        from core.pathogen_neuroinflammation import PathogenTriggeredInflammationEngine

        if self._pathogen_engine is None:
            self._pathogen_engine = PathogenTriggeredInflammationEngine()

        if not self._pathogen_engine.states:
            return {"damage_signal": 0.0, "cytokine_boost": {}, "state_deltas": {}}

        damage, cytokines, deltas = self._pathogen_engine.step(treatment_efficacy)

        # 应用state deltas
        for k, v in deltas.items():
            if k in self._state:
                self._state[k] = float(np.clip(self._state[k] + v * 0.05, 0.0, 1.0))

        return {
            "damage_signal": damage,
            "cytokine_boost": cytokines,
            "state_deltas": deltas,
        }

    # ── 协同计算 ──

    def _compute_synergy(self, drug_effect_level: float) -> float:
        """基于当前药物效应水平计算协同因子。"""
        from core.pharmacotherapy_synergy import SynergyCalculator

        if not self.config.therapies or not self.config.drugs:
            return 0.0

        therapy_mod = self.config.therapies[0].modality
        drug_def = list(self._drug_defs.values())[0] if self._drug_defs else None
        if drug_def is None:
            return 0.0

        calc = SynergyCalculator()
        try:
            synergy = calc.compute(
                drug_class=drug_def.drug_class,
                therapy_modality=therapy_mod,
                condition=self.config.condition,
            )
            base_synergy = synergy.synergy_factor if hasattr(synergy, "synergy_factor") else 0.3
        except Exception:
            base_synergy = 0.3

        # 调制: 药物效应水平越高，协同越强
        modulated = base_synergy * (0.5 + 0.5 * drug_effect_level)
        return max(-0.5, min(1.5, modulated))

    # ── 病理检测 ──

    def _detect_pathological_states(self) -> List[str]:
        """检测异常状态（5-HT综合征、兴奋性毒性等）。

        DDI增强: 当存在5-HT相关DDI时，5-HT综合征阈值降低。
        """
        warnings = []
        s = self._state

        # DDI: 检查是否存在5-HT相关交互
        sht_threshold = 0.9  # 默认阈值
        if self._ddi_result:
            for rec in self._ddi_result.pair_records:
                if (rec.severity in ("contraindicated", "major") and
                        "serotonin" in rec.clinical_description.lower()):
                    sht_threshold = 0.7  # DDI存在时降低阈值
                    break

        sht = float(s.get("nt_serotonin", 0.5))
        if sht > sht_threshold:
            tag = "SEROTONIN SYNDROME ALERT" if sht_threshold < 0.9 else "5-HT syndrome risk"
            warnings.append(f"{tag}: serotonin > {sht_threshold:.1f}")

        da = float(s.get("nt_dopamine", 0.5))
        if da > 0.85:
            warnings.append("DA excitotoxicity risk: dopamine > 0.85")

        # DDI: BZD+阿片 → CNS呼吸抑制检测
        if self._ddi_result:
            for rec in self._ddi_result.contraindications:
                if rec.severity == "contraindicated":
                    gaba = float(s.get("nt_gaba", 0.5))
                    if gaba > 0.7:
                        warnings.append(
                            f"CNS DEPRESSION ALERT: {rec.drug_a}+{rec.drug_b} "
                            f"→ GABA={gaba:.2f} (respiratory depression risk)"
                        )

        arousal = float(s.get("limbic_arousal", 0.5))
        cortisol = float(s.get("cortisol_level", 0.3))
        if arousal > 0.8 and cortisol > 0.7:
            warnings.append("Excitotoxic cascade: arousal + cortisol critically elevated")

        consciousness = float(s.get("brainstem_arousal_setpoint", 0.5))
        if consciousness < 0.1:
            warnings.append("Consciousness gate failure (kernel panic analog)")

        return warnings

    # ── 快照工具 ──

    def _snapshot_neurotransmitters(self) -> Dict[str, float]:
        """快照当前神经递质状态。"""
        keys = [
            "nt_dopamine", "nt_serotonin", "nt_norepinephrine",
            "nt_gaba", "nt_glutamate", "nt_acetylcholine",
            "cortisol_level", "plasticity_bdnf", "hormone_oxytocin",
        ]
        return {k: float(self._state.get(k, 0.5)) for k in keys}

    def _snapshot_brain_regions(self) -> Dict[str, float]:
        """快照当前脑区状态。"""
        keys = [
            "prefrontal_maturity", "prefrontal_inhibition",
            "limbic_valence", "limbic_arousal",
            "hpa_axis_stress_reactivity_mult",
            "ans_hrv", "brainstem_arousal_setpoint",
            "hippocampus_encoding_modulation",
            "self_awareness_coherence", "self_awareness_agency",
        ]
        return {k: float(self._state.get(k, 0.5)) for k in keys}

    # ── 辅助 ──

    def _apply_condition(self) -> None:
        """应用精神疾病档案到状态。"""
        from core.psychiatric_simulation import PSYCHIATRIC_PROFILES, SEVERITY_MULTIPLIERS

        profile = PSYCHIATRIC_PROFILES.get(self.config.condition)
        if profile is None:
            return

        severity_mult = SEVERITY_MULTIPLIERS.get(self.config.severity, 0.6)
        overrides = profile.get("subsystem_overrides", {})

        for subsystem, params in overrides.items():
            for key, target_val in params.items():
                if isinstance(target_val, (int, float)):
                    state_key = f"{subsystem}_{key}"
                    resolved = UnifiedStateMapping.resolve(state_key)
                    baseline = self._state.get(resolved, 0.5)
                    target = float(baseline) + (float(target_val) - float(baseline)) * severity_mult
                    self._state[resolved] = max(0.0, min(1.0, target))

    def _therapy_frequency_to_steps(self, freq: str) -> int:
        """将疗法频率转换为步间隔。"""
        steps_per_hour = self.config.steps_per_hour
        mapping = {
            "biweekly": int(14 * 24 * steps_per_hour),
            "weekly": int(7 * 24 * steps_per_hour),
            "biweekly_month": int(14 * 24 * steps_per_hour),
            "monthly": int(30 * 24 * steps_per_hour),
        }
        return mapping.get(freq, int(7 * 24 * steps_per_hour))

    def _default_state(self) -> Dict[str, Any]:
        """默认状态 (无agent时使用)。"""
        return {
            "nt_dopamine": 0.5, "nt_serotonin": 0.5, "nt_norepinephrine": 0.3,
            "nt_gaba": 0.5, "nt_glutamate": 0.5, "nt_acetylcholine": 0.5,
            "cortisol_level": 0.3, "plasticity_bdnf": 0.5,
            "hormone_oxytocin": 0.4, "hormone_adrenaline": 0.2,
            "limbic_valence": 0.0, "limbic_arousal": 0.5,
            "prefrontal_maturity": 0.5, "prefrontal_inhibition": 0.5,
            "hpa_axis_stress_reactivity_mult": 1.0,
            "hpa_axis_feedback_strength_mult": 1.0,
            "ans_hrv": 0.5, "brainstem_arousal_setpoint": 0.5,
            "hippocampus_encoding_modulation": 0.5,
            "self_awareness_coherence": 0.5, "self_awareness_agency": 0.5,
            "self_awareness_introspection_depth": 0.5,
            "emotion_regulation_regulation_capacity": 0.5,
            "emotion_regulation_inhibition": 0.5,
            "mood_system_volatility_mult": 1.0,
            "predictive_coding_precision_mult": 1.0,
            "predictive_coding_free_energy_bias": 0.3,
            "social_cognition_affective_empathy": 0.5,
            "social_cognition_cognitive_empathy": 0.5,
            "social_cognition_contagion": 0.3,
            "therapy_alliance": 0.3,
        }


__all__ = [
    "DrugConfig",
    "TherapyConfig",
    "ExperimentConfig",
    "TreatmentTimepoint",
    "TherapeuticResult",
    "TherapeuticExperiment",
]