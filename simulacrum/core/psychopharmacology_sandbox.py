"""计算精神药理学沙盒 (Psychopharmacology Sandbox)

顶层实验编排器，整合:
    - PsychiatricConditionSimulator (精神疾病模拟)
    - NeuroPharmacology (药物干预)
    - PsychotherapySystem (心理治疗)
    - SynergyCalculator (协同引擎)

5种实验设计模式:
1. 单药对照: drug_only vs therapy_only vs combined vs placebo
2. 时序探索: drug-first vs therapy-first vs simultaneous
3. 剂量-频率矩阵: 不同药物剂量 × 不同治疗频率
4. 共病处理: 多疾病联合治疗方案
5. 复发预防: 治疗结束后的维持期模拟

用法:
    from core.psychopharmacology_sandbox import PsychopharmacologySandbox

    sandbox = PsychopharmacologySandbox()
    experiment = sandbox.design_experiment(
        condition="MDD", severity="moderate",
        treatment_plan={"drug": "antidepressant", "therapy": "CBT"},
        control_groups=["drug_only", "therapy_only", "no_treatment"],
        duration=200,
    )
    results = sandbox.run_experiment(experiment)
    analysis = sandbox.analyze_results(results)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

try:
    from core.events import (
        EXPERIMENT_END,
        EXPERIMENT_START,
        PSYCHIATRIC_CONDITION_CHANGE,
    )
    from core.neuro_pharmacology import _PRESETS, NeuroPharmacology
    from core.pharmacotherapy_synergy import (
        SynergyCalculator,
        SynergyType,
    )
    from core.psychiatric_simulation import (
        EMOTION_STATE_PROFILES,
        PSYCHIATRIC_PROFILES,
        PsychiatricConditionSimulator,
    )
    from core.psychotherapy import (
        THERAPY_TARGETS,
        PsychotherapySystem,
        TherapyModality,
        TherapyPhase,
    )
except ImportError:
    from core.neuro_pharmacology import _PRESETS
    from core.pharmacotherapy_synergy import (
        SynergyCalculator,
    )
    from core.psychiatric_simulation import (
        EMOTION_STATE_PROFILES,
        PSYCHIATRIC_PROFILES,
    )
    from core.psychotherapy import (
        THERAPY_TARGETS,
        PsychotherapySystem,
        TherapyModality,
    )


# ===== 实验设计 =====

class ExperimentMode(Enum):
    """实验设计模式"""
    CONTROLLED = "controlled"                # 单药对照
    TEMPORAL = "temporal"                    # 时序探索
    DOSE_FREQUENCY = "dose_frequency"        # 剂量-频率矩阵
    COMORBID = "comorbid"                    # 共病处理
    RELAPSE_PREVENTION = "relapse_prevention"  # 复发预防


@dataclass
class TreatmentArm:
    """实验分组 (arm)"""
    name: str
    drug: str | None = None
    therapy: str | None = None
    drug_start_step: int = 0
    therapy_start_step: int = 0
    drug_dose: float = 1.0
    therapy_frequency: str = "weekly"
    therapy_intensity: float = 0.7
    conditions: list[str] = field(default_factory=list)
    severity: str = "moderate"


@dataclass
class ExperimentDesign:
    """实验设计"""
    mode: ExperimentMode
    condition: str
    severity: str
    arms: list[TreatmentArm]
    duration: int = 200
    follow_up: int = 50          # 治疗结束后的追踪步数
    name: str = ""


@dataclass
class ArmResult:
    """单组实验结果"""
    arm_name: str
    symptom_trajectory: list[float] = field(default_factory=list)     # 症状严重度轨迹
    treatment_progress: list[float] = field(default_factory=list)     # 治疗进展轨迹
    skill_trajectory: list[float] = field(default_factory=list)       # 习得技能轨迹
    synergy_factors: list[float] = field(default_factory=list)        # 协同因子轨迹
    side_effects: list[dict] = field(default_factory=list)            # 副作用记录
    emergent_behaviors: list[dict] = field(default_factory=list)      # 涌现行为
    final_symptom: float = 0.0
    remission_rate: float = 0.0         # 缓解率 (症状<0.3的比例)
    relapse_rate: float = 0.0           # 复发率 (追踪期症状反弹)
    mean_synergy: float = 0.0


@dataclass
class ExperimentResult:
    """实验结果"""
    design: ExperimentDesign
    arms: dict[str, ArmResult] = field(default_factory=dict)
    synergy_analysis: dict[str, Any] = field(default_factory=dict)
    completed: bool = False


# ===== 沙盒主类 =====

class PsychopharmacologySandbox:
    """计算精神药理学沙盒

    整合精神疾病模拟、药物干预、心理治疗、协同引擎，
    提供对照实验设计和结果分析。
    """

    def __init__(self, agent=None):
        self._agent = agent
        self._synergy_calc = SynergyCalculator()
        self._experiments: dict[str, ExperimentResult] = {}

    # ════════════════════════════════════════════════
    # 实验设计
    # ════════════════════════════════════════════════

    def design_experiment(
        self,
        condition: str,
        severity: str = "moderate",
        treatment_plan: dict | None = None,
        control_groups: list[str] | None = None,
        duration: int = 200,
        mode: str = "controlled",
        follow_up: int = 50,
        name: str = "",
    ) -> ExperimentDesign:
        """设计实验

        Args:
            condition: 目标疾病ID (如 "MDD", "GAD", "PTSD")
            severity: 严重度 ("mild", "moderate", "severe")
            treatment_plan: 治疗方案 {
                "drug": "antidepressant",
                "therapy": "CBT",
                "drug_start": 10,
                "therapy_start": 30,
                "drug_dose": 1.0,
                "therapy_frequency": "weekly",
                "therapy_intensity": 0.7,
            }
            control_groups: 对照组 ["drug_only", "therapy_only", "no_treatment"]
            duration: 实验总步数
            mode: 实验模式 ("controlled", "temporal", "dose_frequency", "comorbid", "relapse_prevention")
            follow_up: 追踪期步数
            name: 实验名称

        Returns:
            ExperimentDesign
        """
        if treatment_plan is None:
            treatment_plan = {}
        if control_groups is None:
            control_groups = ["drug_only", "therapy_only", "no_treatment"]

        mode_enum = ExperimentMode(mode)
        arms = []

        drug = treatment_plan.get("drug")
        therapy = treatment_plan.get("therapy")
        drug_start = treatment_plan.get("drug_start", 0)
        therapy_start = treatment_plan.get("therapy_start", 0)
        drug_dose = treatment_plan.get("drug_dose", 1.0)
        therapy_freq = treatment_plan.get("therapy_frequency", "weekly")
        therapy_int = treatment_plan.get("therapy_intensity", 0.7)

        # 联合治疗组 (always present)
        arms.append(TreatmentArm(
            name="combined",
            drug=drug, therapy=therapy,
            drug_start_step=drug_start, therapy_start_step=therapy_start,
            drug_dose=drug_dose, therapy_frequency=therapy_freq,
            therapy_intensity=therapy_int,
            conditions=[condition], severity=severity,
        ))

        # 对照组
        for cg in control_groups:
            if cg == "drug_only":
                arms.append(TreatmentArm(
                    name="drug_only",
                    drug=drug, therapy=None,
                    drug_start_step=drug_start,
                    drug_dose=drug_dose,
                    conditions=[condition], severity=severity,
                ))
            elif cg == "therapy_only":
                arms.append(TreatmentArm(
                    name="therapy_only",
                    drug=None, therapy=therapy,
                    therapy_start_step=therapy_start,
                    therapy_frequency=therapy_freq,
                    therapy_intensity=therapy_int,
                    conditions=[condition], severity=severity,
                ))
            elif cg == "no_treatment":
                arms.append(TreatmentArm(
                    name="no_treatment",
                    conditions=[condition], severity=severity,
                ))
            elif cg == "placebo":
                arms.append(TreatmentArm(
                    name="placebo",
                    conditions=[condition], severity=severity,
                ))

        design = ExperimentDesign(
            mode=mode_enum,
            condition=condition,
            severity=severity,
            arms=arms,
            duration=duration,
            follow_up=follow_up,
            name=name or f"{condition}_{mode}",
        )

        return design

    def design_temporal_experiment(
        self,
        condition: str,
        drug: str,
        therapy: str,
        severity: str = "moderate",
        duration: int = 200,
    ) -> ExperimentDesign:
        """设计时序探索实验

        比较: 药物先→治疗先→同步→无治疗
        """
        arms = [
            TreatmentArm(name="drug_first", drug=drug, therapy=therapy,
                        drug_start_step=0, therapy_start_step=50,
                        conditions=[condition], severity=severity),
            TreatmentArm(name="therapy_first", drug=drug, therapy=therapy,
                        drug_start_step=50, therapy_start_step=0,
                        conditions=[condition], severity=severity),
            TreatmentArm(name="simultaneous", drug=drug, therapy=therapy,
                        drug_start_step=0, therapy_start_step=0,
                        conditions=[condition], severity=severity),
            TreatmentArm(name="no_treatment",
                        conditions=[condition], severity=severity),
        ]
        return ExperimentDesign(
            mode=ExperimentMode.TEMPORAL,
            condition=condition, severity=severity,
            arms=arms, duration=duration,
            name=f"{condition}_temporal",
        )

    def design_dose_frequency_experiment(
        self,
        condition: str,
        drug: str,
        therapy: str,
        doses: list[float] | None = None,
        frequencies: list[str] | None = None,
        severity: str = "moderate",
        duration: int = 200,
    ) -> ExperimentDesign:
        """设计剂量-频率矩阵实验

        探索不同药物剂量 × 不同治疗频率的组合效果
        """
        if doses is None:
            doses = [0.3, 0.6, 1.0]
        if frequencies is None:
            frequencies = ["biweekly", "weekly", "biweekly_month"]

        arms = []
        for dose in doses:
            for freq in frequencies:
                arms.append(TreatmentArm(
                    name=f"drug{dose:.1f}_{freq}",
                    drug=drug, therapy=therapy,
                    drug_dose=dose, therapy_frequency=freq,
                    conditions=[condition], severity=severity,
                ))

        return ExperimentDesign(
            mode=ExperimentMode.DOSE_FREQUENCY,
            condition=condition, severity=severity,
            arms=arms, duration=duration,
            name=f"{condition}_dose_freq",
        )

    # ════════════════════════════════════════════════
    # 实验运行
    # ════════════════════════════════════════════════

    def run_experiment(
        self,
        design: ExperimentDesign,
        agent=None,
        verbose: bool = True,
    ) -> ExperimentResult:
        """运行实验

        对每个arm，创建独立的模拟环境并运行。
        由于Simulacrum agent是有状态的，这里用简化模拟：
        直接操作 internal_state 字典来模拟各arm的轨迹。

        Args:
            design: 实验设计
            agent: Simulacrum agent (用于获取初始状态)
            verbose: 是否打印进度

        Returns:
            ExperimentResult
        """
        base_agent = agent or self._agent
        if base_agent is None:
            # 无agent时使用默认初始状态
            base_state = self._default_initial_state()
        else:
            base_state = dict(base_agent._internal_state)

        result = ExperimentResult(design=design)

        for arm in design.arms:
            if verbose:
                print(f"\n[Sandbox] Running arm: {arm.name}")

            arm_result = self._run_arm(
                arm=arm,
                base_state=base_state,
                duration=design.duration,
                follow_up=design.follow_up,
                condition=design.condition,
                verbose=verbose,
            )
            result.arms[arm.name] = arm_result

        # 协同分析
        result.synergy_analysis = self._analyze_synergy(result)
        result.completed = True

        # 保存
        self._experiments[design.name] = result

        return result

    def run_single(
        self,
        condition: str,
        drug: str | None = None,
        therapy: str | None = None,
        severity: str = "moderate",
        duration: int = 100,
    ) -> ArmResult:
        """快速运行单个治疗方案

        便捷方法，不需要设计完整实验。
        """
        arm = TreatmentArm(
            name=f"{drug or 'no_drug'}+{therapy or 'no_therapy'}",
            drug=drug, therapy=therapy,
            conditions=[condition], severity=severity,
        )
        base_state = self._default_initial_state()
        if self._agent is not None:
            base_state = dict(self._agent._internal_state)

        return self._run_arm(
            arm=arm, base_state=base_state,
            duration=duration, follow_up=30,
            condition=condition, verbose=False,
        )

    # ════════════════════════════════════════════════
    # 结果分析
    # ════════════════════════════════════════════════

    def analyze_results(self, result: ExperimentResult) -> dict[str, Any]:
        """分析实验结果

        Returns:
            {
                "synergy_quantification": combined vs drug_only+therapy_only,
                "best_arm": 最佳组名,
                "remission_comparison": 各组缓解率,
                "relapse_comparison": 各组复发率,
                "temporal_analysis": 时序分析 (如果有),
            }
        """
        analysis = {}

        # 1. 协同量化
        analysis["synergy_quantification"] = result.synergy_analysis

        # 2. 最佳组
        best_arm = None
        best_remission = -1
        for name, arm in result.arms.items():
            if arm.remission_rate > best_remission:
                best_remission = arm.remission_rate
                best_arm = name
        analysis["best_arm"] = best_arm

        # 3. 缓解率比较
        analysis["remission_comparison"] = {
            name: arm.remission_rate for name, arm in result.arms.items()
        }

        # 4. 复发率比较
        analysis["relapse_comparison"] = {
            name: arm.relapse_rate for name, arm in result.arms.items()
        }

        # 5. 最终症状比较
        analysis["final_symptom_comparison"] = {
            name: arm.final_symptom for name, arm in result.arms.items()
        }

        # 6. 协同因子比较
        analysis["mean_synergy_comparison"] = {
            name: arm.mean_synergy for name, arm in result.arms.items()
        }

        return analysis

    def get_experiment(self, name: str) -> ExperimentResult | None:
        """获取已运行的实验结果"""
        return self._experiments.get(name)

    def list_experiments(self) -> list[str]:
        """列出所有已运行的实验"""
        return list(self._experiments.keys())

    # ════════════════════════════════════════════════
    # 内部方法
    # ════════════════════════════════════════════════

    def _run_arm(
        self,
        arm: TreatmentArm,
        base_state: dict[str, Any],
        duration: int,
        follow_up: int,
        condition: str,
        verbose: bool = False,
    ) -> ArmResult:
        """运行单个实验组"""
        state = dict(base_state)
        result = ArmResult(arm_name=arm.name)

        # 应用疾病 profile
        profile = PSYCHIATRIC_PROFILES.get(condition) or EMOTION_STATE_PROFILES.get(condition)
        if profile:
            overrides = profile.get("subsystem_overrides", {})
            for subsystem, params in overrides.items():
                for key, val in params.items():
                    state_key = f"{subsystem}_{key}"
                    if isinstance(val, list):
                        state[state_key] = val
                    else:
                        state[state_key] = val

        # 初始化治疗系统
        therapy_system = None
        if arm.therapy:
            therapy_system = PsychotherapySystem()
            therapy_system.start_treatment(arm.therapy, arm.therapy_frequency)

        # 运行主治疗期
        for step in range(duration):
            # 1. 药物效应 (到达起始步后应用)
            if arm.drug and step >= arm.drug_start_step:
                self._apply_drug_effects(state, arm.drug, arm.drug_dose)

            # 2. 心理治疗 (到达起始步后执行session)
            if arm.therapy and step >= arm.therapy_start_step and therapy_system:
                # 计算协同
                synergy_bonus = 0.0
                if arm.drug:
                    synergy_bonus = self._synergy_calc.compute(
                        arm.drug, arm.therapy, condition, arm.drug_dose
                    )
                # 每10步做一次session
                if step % 10 == 0:
                    # 解析治疗流派枚举
                    mod_enum = None
                    for m in TherapyModality:
                        if m.value == arm.therapy:
                            mod_enum = m
                            break
                    if mod_enum is None:
                        mod_enum = TherapyModality.CBT

                    phase = TherapyPhase.ACTIVE
                    if therapy_system.active_treatments.get(arm.therapy):
                        phase = therapy_system.active_treatments[arm.therapy].phase

                    # 手动应用治疗效应到state (不传agent)
                    targets = THERAPY_TARGETS.get(mod_enum, {}).get("primary_targets", {})
                    compliance = 0.7  # 默认配合度
                    for state_key, delta in targets.items():
                        current = state.get(state_key, 0.5)
                        adjustment = delta * arm.therapy_intensity * compliance * (1.0 + synergy_bonus)
                        new_val = current + adjustment
                        if "mult" in state_key:
                            new_val = max(0.05, new_val)
                        else:
                            new_val = float(np.clip(new_val, 0.0, 1.0))
                        state[state_key] = new_val

            # 3. 疾病动力学 (自然进展/恢复)
            self._step_disease_dynamics(state, condition)

            # 4. 记录轨迹
            symptom = self._compute_symptom_severity(state, condition)
            result.symptom_trajectory.append(symptom)

            if therapy_system and arm.therapy:
                tp = therapy_system.active_treatments.get(arm.therapy)
                result.skill_trajectory.append(tp.current_skill if tp else 0.0)
                result.treatment_progress.append(tp.total_effect if tp else 0.0)
            else:
                result.skill_trajectory.append(0.0)
                result.treatment_progress.append(0.0)

            if arm.drug and arm.therapy:
                sf = self._synergy_calc.compute(arm.drug, arm.therapy, condition, arm.drug_dose)
                result.synergy_factors.append(sf)
            else:
                result.synergy_factors.append(0.0)

        # 追踪期 (治疗停止后)
        for step in range(follow_up):
            # 技能衰减 (Ebbinghaus)
            if result.skill_trajectory:
                last_skill = result.skill_trajectory[-1]
                decayed = last_skill * np.exp(-0.002 * (step + 1))
                result.skill_trajectory.append(decayed)

            # 疾病动力学继续
            self._step_disease_dynamics(state, condition)
            symptom = self._compute_symptom_severity(state, condition)
            result.symptom_trajectory.append(symptom)
            result.treatment_progress.append(result.treatment_progress[-1] if result.treatment_progress else 0.0)
            result.synergy_factors.append(0.0)

        # 计算汇总指标
        if result.symptom_trajectory:
            result.final_symptom = result.symptom_trajectory[-1]
            result.remission_rate = sum(1 for s in result.symptom_trajectory if s < 0.3) / len(result.symptom_trajectory)

            # 复发率: 追踪期症状是否反弹
            if len(result.symptom_trajectory) > follow_up:
                treatment_end = len(result.symptom_trajectory) - follow_up
                min_symptom = min(result.symptom_trajectory[treatment_end:])
                max_followup = max(result.symptom_trajectory[treatment_end:])
                if min_symptom < 0.3 and max_followup > 0.5:
                    result.relapse_rate = (max_followup - min_symptom) / max_followup

        if result.synergy_factors:
            result.mean_synergy = np.mean(result.synergy_factors)

        return result

    def _apply_drug_effects(self, state: dict, drug: str, dose: float = 1.0):
        """应用药物效应到状态字典"""
        preset = _PRESETS.get(drug)
        if preset is None:
            return

        # 注射递质
        for nt, conc in preset.get("nt", {}).items():
            nt_name = nt.value if hasattr(nt, 'value') else str(nt)
            state_key = f"nt_{nt_name}"
            state[state_key] = conc * dose

        # 降低递质
        for nt, conc in preset.get("reduce", {}).items():
            nt_name = nt.value if hasattr(nt, 'value') else str(nt)
            state_key = f"nt_{nt_name}"
            state[state_key] = conc * dose

    def _step_disease_dynamics(self, state: dict, condition: str):
        """推进疾病动力学一步

        疾病参数有持续压力向profile目标值偏移 (疾病维持力)，
        同时健康参数有自然回归趋势 (0.5)。
        两者的平衡决定了疾病的自然病程。
        """
        profile = PSYCHIATRIC_PROFILES.get(condition) or EMOTION_STATE_PROFILES.get(condition)
        if profile:
            overrides = profile.get("subsystem_overrides", {})
            # 疾病维持力: 向profile目标值缓慢偏移
            disease_pressure = 0.005  # 疾病持续偏移力
            for subsystem, params in overrides.items():
                for key, target_val in params.items():
                    state_key = f"{subsystem}_{key}"
                    current = state.get(state_key, 0.5)
                    if isinstance(target_val, list):
                        continue  # 向量值不动态调整
                    if not isinstance(target_val, (int, float)):
                        continue  # 非数值类型 (如 "freeze") 不动态调整
                    # 疾病压力: 向目标值偏移
                    state[state_key] = current + (target_val - current) * disease_pressure

        # 自然恢复力: 非疾病目标参数向0.5回归
        natural_regression = 0.003  # 自然恢复力 (比疾病压力弱)
        profile_keys = set()
        if profile:
            for subsystem, params in profile.get("subsystem_overrides", {}).items():
                for key in params:
                    profile_keys.add(f"{subsystem}_{key}")

        for key in list(state.keys()):
            val = state[key]
            if isinstance(val, (int, float)) and key not in profile_keys and key not in ("therapy_session_count", "therapy_alliance"):
                state[key] = val + (0.5 - val) * natural_regression

    def _compute_symptom_severity(self, state: dict, condition: str) -> float:
        """计算症状严重度 [0,1]

        基于疾病类型选择关键指标
        """
        if condition in ("MDD", "bipolar_depression", "Dysthymia"):
            # 抑郁: 低效价 + 低DA + 低5-HT + 高皮质醇
            valence = -state.get("limbic_valence", 0)  # 负效价→高严重度
            da = 1.0 - state.get("nt_dopamine", 0.5)
            serotonin = 1.0 - state.get("nt_serotonin", 0.5)
            cortisol = state.get("hormone_cortisol", 0.3) - 0.3
            return float(np.clip((valence + da + serotonin + cortisol) / 4, 0, 1))

        elif condition in ("GAD", "Social_Anxiety", "Panic_Disorder"):
            # 焦虑: 高唤醒 + 高NE + 高皮质醇 + 低GABA
            arousal = state.get("limbic_arousal", 0.5) - 0.5
            ne = state.get("nt_norepinephrine", 0.3) - 0.3
            cortisol = state.get("hormone_cortisol", 0.3) - 0.3
            return float(np.clip((arousal + ne + cortisol) / 3 * 2, 0, 1))

        elif condition == "PTSD":
            # PTSD: 高唤醒 + 高应激反应性 + 低PFC抑制
            arousal = state.get("limbic_arousal", 0.5) - 0.5
            stress = state.get("hpa_axis_stress_reactivity_mult", 1.0) - 1.0
            pfc = 1.0 - state.get("prefrontal_inhibition", 0.5)
            return float(np.clip((arousal + stress + pfc) / 3 * 2, 0, 1))

        elif condition == "BPD":
            # BPD: 高波动 + 低自我连贯 + 低调节容量
            volatility = state.get("mood_system_volatility_mult", 1.0) - 1.0
            coherence = 1.0 - state.get("self_awareness_coherence", 0.7)
            regulation = 1.0 - state.get("emotion_regulation_regulation_capacity", 0.5)
            return float(np.clip((volatility + coherence + regulation) / 3, 0, 1))

        elif condition in ("schizophrenia_positive",):
            # 精分阳性: 高DA + 高精确度 + 低自我边界
            da = state.get("nt_dopamine", 0.5) - 0.5
            precision = state.get("predictive_coding_precision_mult", 1.0) - 1.0
            boundary = 1.0 - state.get("self_awareness_self_boundary", 0.7)
            return float(np.clip((da + precision + boundary) / 3 * 2, 0, 1))

        else:
            # 通用: 基于效价和唤醒
            valence = -state.get("limbic_valence", 0)
            arousal = state.get("limbic_arousal", 0.5) - 0.5
            return float(np.clip((valence + arousal) / 2 + 0.3, 0, 1))

    def _analyze_synergy(self, result: ExperimentResult) -> dict[str, Any]:
        """分析协同效应"""
        analysis = {}

        combined = result.arms.get("combined")
        drug_only = result.arms.get("drug_only")
        therapy_only = result.arms.get("therapy_only")
        no_treatment = result.arms.get("no_treatment")

        if combined and drug_only and therapy_only:
            # 协同量化: combined的缓解率 vs drug_only+therapy_only的预期叠加
            combined_rem = combined.remission_rate
            drug_rem = drug_only.remission_rate
            therapy_rem = therapy_only.remission_rate

            # 预期叠加 (简单相加，上限1.0)
            expected_additive = min(1.0, drug_rem + therapy_rem)

            synergy_excess = combined_rem - expected_additive
            analysis["synergy_excess"] = synergy_excess
            analysis["is_synergistic"] = synergy_excess > 0.05
            analysis["is_antagonistic"] = synergy_excess < -0.05
            analysis["combined_remission"] = combined_rem
            analysis["drug_only_remission"] = drug_rem
            analysis["therapy_only_remission"] = therapy_rem
            analysis["expected_additive"] = expected_additive

        if combined and no_treatment:
            analysis["treatment_effect_size"] = (
                no_treatment.remission_rate - combined.remission_rate
            )

        # 时序分析
        if result.design.mode == ExperimentMode.TEMPORAL:
            temporal = {}
            for name in ("drug_first", "therapy_first", "simultaneous"):
                arm = result.arms.get(name)
                if arm:
                    temporal[name] = {
                        "final_symptom": arm.final_symptom,
                        "remission_rate": arm.remission_rate,
                        "relapse_rate": arm.relapse_rate,
                    }
            analysis["temporal_comparison"] = temporal

        return analysis

    @staticmethod
    def _default_initial_state() -> dict[str, float]:
        """默认初始状态 (无agent时使用)"""
        return {
            "limbic_valence": 0.0,
            "limbic_arousal": 0.5,
            "limbic_emotion": "neutral",
            "nt_dopamine": 0.5,
            "nt_serotonin": 0.5,
            "nt_norepinephrine": 0.3,
            "nt_acetylcholine": 0.5,
            "nt_gaba": 0.5,
            "cortisol_level": 0.3,
            "hormone_cortisol": 0.3,
            "hormone_oxytocin": 0.3,
            "hormone_adrenaline": 0.2,
            "prefrontal_maturity": 0.5,
            "prefrontal_inhibition": 0.5,
            "emotion_regulation_regulation_capacity": 0.5,
            "emotion_regulation_inhibition": 0.5,
            "mood_system_volatility_mult": 1.0,
            "self_awareness_coherence": 0.7,
            "self_awareness_agency": 0.7,
            "self_awareness_self_boundary": 0.7,
            "predictive_coding_precision_mult": 1.0,
            "predictive_coding_free_energy_bias": 0.0,
            "hpa_axis_stress_reactivity_mult": 1.0,
            "hpa_axis_feedback_strength_mult": 1.0,
            "ans_baseline_vagal_tone": 0.5,
            "ans_sympathetic_reactivity_mult": 1.0,
            "brainstem_arousal_setpoint": 0.5,
            "hippocampus_encoding_modulation": 0.5,
            "social_cognition_affective_empathy": 0.5,
            "social_cognition_cognitive_empathy": 0.5,
            "social_cognition_contagion": 0.5,
            "plasticity_bdnf": 0.5,
            "therapy_alliance": 0.3,
            "therapy_session_count": 0,
        }


# ===== 便捷函数 =====

def create_sandbox(agent=None) -> PsychopharmacologySandbox:
    """创建精神药理学沙盒"""
    return PsychopharmacologySandbox(agent=agent)


__all__ = [
    "ExperimentMode",
    "TreatmentArm",
    "ExperimentDesign",
    "ArmResult",
    "ExperimentResult",
    "PsychopharmacologySandbox",
    "create_sandbox",
]
