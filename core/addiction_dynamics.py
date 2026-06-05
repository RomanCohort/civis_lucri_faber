"""成瘾动力学引擎 — Tolerance, Withdrawal, Craving, Sensitization.

基于四个核心机制:
1. 耐受 (Tolerance): 慢性用药 → 受体下调/脱敏 → 同等剂量效应减弱
2. 戒断 (Withdrawal): 停药 → 对抗过程暴露 → 与药物效应方向相反的症状
3. 渴求 (Craving): 动机显著性放大 → wanting与liking分离 → 强迫性用药
4. 致敏化 (Sensitization): 反复暴露 → 神经适应进行性增强 → 点燃效应

理论框架:
- 对抗过程理论 (Solomon & Corbit, 1974): a-process(药物效应) vs b-process(对抗过程)
- 动机显著性理论 (Berridge & Robinson, 1998): wanting ≠ liking
- 点燃假说 (Post, 1980): 反复发作使阈值进行性降低
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────

@dataclass
class ToleranceState:
    """单个受体上的耐受状态。"""
    receptor_name: str
    downregulation_factor: float = 1.0     # 1.0=正常, <1=下调 (如0.6=40%下调)
    desensitization_factor: float = 1.0    # 1.0=正常, <1=脱敏
    adaptation_rate: float = 0.002         # 每步下调速率
    recovery_rate: float = 0.001           # 每步恢复速率
    cumulative_exposure: float = 0.0       # 累积暴露量


@dataclass
class WithdrawalState:
    """戒断状态 — 对抗过程模型。"""
    drug_name: str
    is_active: bool = False
    current_severity: float = 0.0
    peak_severity: float = 0.0
    steps_since_onset: int = 0
    a_process_amplitude: float = 0.0       # 药物效应振幅 (耐受后的)
    b_process_amplitude: float = 0.0       # 对抗过程振幅
    b_process_decay: float = 0.02          # 对抗过程衰减速率
    expected_duration_steps: int = 0


@dataclass
class CravingState:
    """渴求状态 — wanting/liking分离。"""
    drug_name: str
    current_level: float = 0.0             # 当前渴求强度 (0-1+)
    sensitization_factor: float = 1.0      # 致敏化因子 (进行性增长)
    wanting_amplification: float = 1.0     # wanting放大倍率
    liking_separation: float = 0.0         # wanting-liking分离度 (0=未分离)
    cue_reactivity: float = 0.0            # 线索反应性


@dataclass
class AddictionProfile:
    """单个药物的完整成瘾档案。"""
    drug_name: str
    drug_class: str
    tolerance: dict[str, ToleranceState] = field(default_factory=dict)
    withdrawal: WithdrawalState | None = None
    craving: CravingState | None = None
    total_chronic_steps: int = 0
    is_dependent: bool = False
    peak_blood_concentration: float = 0.0


# ──────────────────────────────────────────────────────
# 药物类别成瘾参数
# ──────────────────────────────────────────────────────

DRUG_CLASS_ADDICTION_PARAMS: dict[str, dict] = {
    "opioid": {
        "receptors": ["mu-opioid", "delta"],
        "tolerance_rate": 0.003,        # 快速耐受
        "withdrawal_peak": 0.85,        # 严重戒断
        "withdrawal_onset_steps": 30,   # 停药后30步开始
        "withdrawal_duration_steps": 800,
        "sensitization_rate": 0.001,
        "wanting_max": 3.0,             # 高渴求
        "liking_decay_rate": 0.005,     # liking快速衰减
    },
    "stimulant": {
        "receptors": ["D2", "DAT", "NET"],
        "tolerance_rate": 0.001,        # 中等耐受
        "withdrawal_peak": 0.30,        # 轻度戒断 (crash)
        "withdrawal_onset_steps": 10,
        "withdrawal_duration_steps": 300,
        "sensitization_rate": 0.003,    # 高致敏化
        "wanting_max": 5.0,             # 极高渴求
        "liking_decay_rate": 0.003,
    },
    "sedative": {
        "receptors": ["GABA-A"],
        "tolerance_rate": 0.002,        # 中等耐受
        "withdrawal_peak": 0.90,        # 极严重戒断 (癫痫风险)
        "withdrawal_onset_steps": 50,
        "withdrawal_duration_steps": 1200,
        "sensitization_rate": 0.001,
        "wanting_max": 2.0,             # 中等渴求
        "liking_decay_rate": 0.002,
    },
    "SSRI": {
        "receptors": ["SERT"],
        "tolerance_rate": 0.0005,       # 极慢耐受
        "withdrawal_peak": 0.25,        # 轻度戒断 (脑内闪电感)
        "withdrawal_onset_steps": 100,
        "withdrawal_duration_steps": 500,
        "sensitization_rate": 0.0002,
        "wanting_max": 1.0,             # 无显著渴求
        "liking_decay_rate": 0.0,
    },
    "hallucinogen": {
        "receptors": ["NMDA", "5-HT2A"],
        "tolerance_rate": 0.005,        # 快速耐受 (tachyphylaxis)
        "withdrawal_peak": 0.10,        # 极轻戒断
        "withdrawal_onset_steps": 50,
        "withdrawal_duration_steps": 200,
        "sensitization_rate": 0.0005,
        "wanting_max": 1.5,             # 低渴求
        "liking_decay_rate": 0.0,
    },
    "antipsychotic": {
        "receptors": ["D2", "5-HT2A"],
        "tolerance_rate": 0.001,
        "withdrawal_peak": 0.20,
        "withdrawal_onset_steps": 200,
        "withdrawal_duration_steps": 600,
        "sensitization_rate": 0.0003,
        "wanting_max": 1.0,
        "liking_decay_rate": 0.0,
    },
    "mood_stabilizer": {
        "receptors": [],
        "tolerance_rate": 0.0,
        "withdrawal_peak": 0.15,
        "withdrawal_onset_steps": 200,
        "withdrawal_duration_steps": 400,
        "sensitization_rate": 0.0,
        "wanting_max": 1.0,
        "liking_decay_rate": 0.0,
    },
    "beta_blocker": {
        "receptors": ["beta1", "beta2"],
        "tolerance_rate": 0.001,
        "withdrawal_peak": 0.30,        # 反跳性心动过速
        "withdrawal_onset_steps": 50,
        "withdrawal_duration_steps": 300,
        "sensitization_rate": 0.0,
        "wanting_max": 1.0,
        "liking_decay_rate": 0.0,
    },
    "dora": {  # 双orexin受体拮抗剂
        "receptors": ["orexin1", "orexin2"],
        "tolerance_rate": 0.0005,
        "withdrawal_peak": 0.10,
        "withdrawal_onset_steps": 100,
        "withdrawal_duration_steps": 200,
        "sensitization_rate": 0.0,
        "wanting_max": 1.0,
        "liking_decay_rate": 0.0,
    },
}


# ──────────────────────────────────────────────────────
# 成瘾动力学引擎
# ──────────────────────────────────────────────────────

class AddictionDynamicsEngine:
    """成瘾动力学引擎 — 管理耐受、戒断、渴求、致敏化的动态过程。"""

    def __init__(self) -> None:
        self.profiles: dict[str, AddictionProfile] = {}
        self._drug_concentrations: dict[str, float] = {}
        self._was_drug_active: dict[str, bool] = {}

    def register_drug(
        self,
        drug_name: str,
        drug_class: str,
    ) -> None:
        """注册一个药物到成瘾追踪系统。"""
        params = DRUG_CLASS_ADDICTION_PARAMS.get(drug_class, {})
        receptors = params.get("receptors", [])

        tolerance_states = {}
        tol_rate = params.get("tolerance_rate", 0.001)
        for rct in receptors:
            tolerance_states[rct] = ToleranceState(
                receptor_name=rct,
                adaptation_rate=tol_rate,
                recovery_rate=tol_rate * 0.5,  # 恢复比适应慢
            )

        profile = AddictionProfile(
            drug_name=drug_name,
            drug_class=drug_class,
            tolerance=tolerance_states,
            withdrawal=WithdrawalState(drug_name=drug_name),
            craving=CravingState(drug_name=drug_name),
        )
        self.profiles[drug_name] = profile
        self._was_drug_active[drug_name] = True

    def step(
        self,
        drug_concentrations: dict[str, float],
        drug_effects: dict[str, float],
    ) -> tuple[
        dict[str, dict[str, float]],  # tolerance_factors {drug: {receptor: factor}}
        dict[str, float],              # withdrawal_deltas {nt_key: delta}
        dict[str, float],              # craving_levels {drug: level}
    ]:
        """每步更新成瘾动力学。

        Args:
            drug_concentrations: {drug_name: concentration_mg_per_l}
            drug_effects: {drug_name: pd_effect (0-1)}

        Returns:
            (tolerance_factors, withdrawal_deltas, craving_levels)
        """
        self._drug_concentrations = drug_concentrations

        all_tolerance: dict[str, dict[str, float]] = {}
        all_withdrawal: dict[str, float] = {}
        all_craving: dict[str, float] = {}

        for drug_name, profile in self.profiles.items():
            conc = drug_concentrations.get(drug_name, 0.0)
            effect = drug_effects.get(drug_name, 0.0)
            is_active = conc > 0.01  # 有药物存在

            # ── 1. 耐受更新 ──
            tol_factors = self._update_tolerance(profile, conc, effect)
            all_tolerance[drug_name] = tol_factors

            # ── 2. 戒断检测与更新 ──
            was_active = self._was_drug_active.get(drug_name, True)
            if was_active and not is_active and profile.total_chronic_steps > 50:
                # 药物刚停 → 触发戒断
                self._trigger_withdrawal(profile, effect)
            wd_deltas = self._update_withdrawal(profile, is_active)
            for k, v in wd_deltas.items():
                all_withdrawal[k] = all_withdrawal.get(k, 0.0) + v

            # ── 3. 渴求更新 ──
            craving = self._update_craving(profile, conc, effect)
            all_craving[drug_name] = craving

            # ── 4. 累积暴露 ──
            if is_active:
                profile.total_chronic_steps += 1
                profile.peak_blood_concentration = max(
                    profile.peak_blood_concentration, conc
                )

            # 依赖判定
            params = DRUG_CLASS_ADDICTION_PARAMS.get(profile.drug_class, {})
            if profile.total_chronic_steps > 200 and params.get("withdrawal_peak", 0) > 0.3:
                profile.is_dependent = True

            self._was_drug_active[drug_name] = is_active

        return all_tolerance, all_withdrawal, all_craving

    def _update_tolerance(
        self,
        profile: AddictionProfile,
        concentration: float,
        effect: float,
    ) -> dict[str, float]:
        """更新耐受状态 — 慢性暴露 → 受体下调/脱敏。"""
        factors: dict[str, float] = {}

        for rct_name, state in profile.tolerance.items():
            if concentration > 0.01:
                # 有药物 → 下调/脱敏
                state.cumulative_exposure += concentration * 0.01
                downreg = state.adaptation_rate * concentration
                desens = state.adaptation_rate * effect * 0.8

                state.downregulation_factor = max(
                    0.3,  # 最多70%下调
                    state.downregulation_factor - downreg
                )
                state.desensitization_factor = max(
                    0.3,
                    state.desensitization_factor - desens
                )
            else:
                # 无药物 → 缓慢恢复
                state.downregulation_factor = min(
                    1.0,
                    state.downregulation_factor + state.recovery_rate
                )
                state.desensitization_factor = min(
                    1.0,
                    state.desensitization_factor + state.recovery_rate
                )

            factors[rct_name] = (
                state.downregulation_factor * state.desensitization_factor
            )

        return factors

    def _trigger_withdrawal(
        self,
        profile: AddictionProfile,
        last_effect: float,
    ) -> None:
        """触发戒断 — 对抗过程暴露。"""
        params = DRUG_CLASS_ADDICTION_PARAMS.get(profile.drug_class, {})
        wd = profile.withdrawal
        if wd is None:
            return

        # b-process振幅 = 基于峰值效应和耐受程度 (而非当前零效应)
        # 对抗过程振幅与慢性暴露期间的a-process振幅成正比
        avg_tol = 1.0
        if profile.tolerance:
            avg_tol = sum(
                s.downregulation_factor for s in profile.tolerance.values()
            ) / len(profile.tolerance)

        # 使用峰值效应而非当前零效应
        peak_effect = profile.peak_blood_concentration  # 峰值暴露
        wd.is_active = True
        wd.a_process_amplitude = peak_effect
        wd.b_process_amplitude = peak_effect * (2.0 - avg_tol)  # 耐受越大→b越大
        wd.peak_severity = params.get("withdrawal_peak", 0.3) * wd.b_process_amplitude
        wd.current_severity = 0.0
        wd.steps_since_onset = 0
        wd.expected_duration_steps = params.get("withdrawal_duration_steps", 500)

    def _update_withdrawal(
        self,
        profile: AddictionProfile,
        is_drug_active: bool,
    ) -> dict[str, float]:
        """更新戒断状态 — 产生与药物效应方向相反的NT delta。"""
        wd = profile.withdrawal
        if wd is None or not wd.is_active:
            return {}

        wd.steps_since_onset += 1

        # 戒断严重度曲线: 快速上升 → 峰值 → 指数衰减
        onset_steps = DRUG_CLASS_ADDICTION_PARAMS.get(
            profile.drug_class, {}
        ).get("withdrawal_onset_steps", 50)

        if wd.steps_since_onset < onset_steps:
            # 上升期
            wd.current_severity = wd.peak_severity * (
                wd.steps_since_onset / onset_steps
            )
        else:
            # 衰减期
            decay = wd.b_process_decay * (wd.steps_since_onset - onset_steps)
            wd.current_severity = wd.peak_severity * max(0.0, 1.0 - decay)

        # 戒断结束
        if wd.current_severity < 0.01:
            wd.is_active = False
            wd.current_severity = 0.0
            return {}

        # 如果重新用药 → 戒断缓解
        if is_drug_active:
            wd.current_severity *= 0.3  # 快速缓解
            if wd.current_severity < 0.05:
                wd.is_active = False

        # 戒断产生的NT delta (与药物效应方向相反)
        params = DRUG_CLASS_ADDICTION_PARAMS.get(profile.drug_class, {})
        receptors = params.get("receptors", [])
        wd_deltas: dict[str, float] = {}

        # 简化映射: 受体 → 父NT → 反向delta
        RECEPTOR_NT_MAP = {
            "mu-opioid": ("nt_dopamine", +1.0),   # 阿片戒断 → DA↓ (烦躁)
            "delta":     ("nt_dopamine", +0.3),
            "GABA-A":    ("nt_gaba", +1.0),        # BZD戒断 → GABA↓ (焦虑/癫痫)
            "D2":        ("nt_dopamine", -0.5),    # 兴奋剂戒断 → DA↓ (快感缺失)
            "DAT":       ("nt_dopamine", -0.5),
            "NET":       ("nt_norepinephrine", -0.3),
            "SERT":      ("nt_serotonin", -0.3),   # SSRI戒断 → 5-HT↓
            "beta1":     ("nt_norepinephrine", +0.5),  # β阻滞剂戒断 → NE↑ (反跳)
            "beta2":     ("nt_norepinephrine", +0.3),
        }

        for rct in receptors:
            if rct in RECEPTOR_NT_MAP:
                nt_key, direction = RECEPTOR_NT_MAP[rct]
                # 戒断delta = 严重度 × 反向 (与药物效应相反)
                wd_deltas[nt_key] = wd_deltas.get(nt_key, 0.0) + (
                    -direction * wd.current_severity * 0.3
                )

        return wd_deltas

    def _update_craving(
        self,
        profile: AddictionProfile,
        concentration: float,
        effect: float,
    ) -> float:
        """更新渴求状态 — wanting/liking分离 + 致敏化。"""
        craving = profile.craving
        if craving is None:
            return 0.0

        params = DRUG_CLASS_ADDICTION_PARAMS.get(profile.drug_class, {})
        sens_rate = params.get("sensitization_rate", 0.001)
        wanting_max = params.get("wanting_max", 1.0)

        # 致敏化: 每次暴露递增
        if concentration > 0.01:
            craving.sensitization_factor = min(
                wanting_max,
                craving.sensitization_factor + sens_rate
            )

        # Wanting = 基础渴求 × 致敏化因子
        # 基础渴求: 戒断时最高, 用药时降低
        wd = profile.withdrawal
        if wd and wd.is_active:
            base_craving = 0.5 + wd.current_severity * 0.5
        elif concentration > 0.01:
            base_craving = 0.1  # 用药时渴求低
        else:
            base_craving = 0.3  # 中等渴求

        craving.current_level = min(
            wanting_max,
            base_craving * craving.sensitization_factor
        )

        # Liking衰减: 慢性用药 → 享乐效应减弱
        liking_rate = params.get("liking_decay_rate", 0.0)
        if concentration > 0.01 and liking_rate > 0:
            craving.liking_separation = min(
                0.8,
                craving.liking_separation + liking_rate
            )

        # Wanting放大
        craving.wanting_amplification = craving.sensitization_factor

        return craving.current_level

    def get_state_summary(self) -> dict[str, dict]:
        """获取所有药物的成瘾状态摘要。"""
        summary = {}
        for name, profile in self.profiles.items():
            tol = {}
            for rct, state in profile.tolerance.items():
                tol[rct] = {
                    "downreg": round(state.downregulation_factor, 3),
                    "desens": round(state.desensitization_factor, 3),
                }
            wd = profile.withdrawal
            cr = profile.craving
            summary[name] = {
                "chronic_steps": profile.total_chronic_steps,
                "is_dependent": profile.is_dependent,
                "tolerance": tol,
                "withdrawal_active": wd.is_active if wd else False,
                "withdrawal_severity": round(wd.current_severity, 3) if wd else 0.0,
                "craving_level": round(cr.current_level, 3) if cr else 0.0,
                "sensitization": round(cr.sensitization_factor, 3) if cr else 1.0,
                "liking_separation": round(cr.liking_separation, 3) if cr else 0.0,
            }
        return summary


__all__ = [
    "ToleranceState",
    "WithdrawalState",
    "CravingState",
    "AddictionProfile",
    "DRUG_CLASS_ADDICTION_PARAMS",
    "AddictionDynamicsEngine",
]
