"""
激素调制系统 (Hormone Modulation System) — v2 动态版

对应生物学的内分泌-神经系统交互，含:
1. 皮质醇 — 单一来源(HPA axis) + 超日脉冲 + 倒U型记忆编码调制
2. 肾上腺素 — 二阶动力学(快升慢降) + SAM-HPA 耦合
3. 催产素 — 阈值激活 + 持续释放 + 社会缓冲
4. 褪黑素 — SCN 驱动 + 皮质醇交叉抑制
5. 血清素 — 肠脑轴双向耦合 + 皮质醇反馈抑制
6. 激素间交叉作用: 皮质醇⇄催产素, 皮质醇⇄褪黑素, 血清素→催产素
7. GR/MR 受体敏感性动态 (慢性应激下调)
8. 超日脉冲 (~90 min 周期)

核心原则:
- 皮质醇不再自行计算，从 HPA axis 统一来源读取
- 所有激素有独立的非线性动力学 (非简单一阶滤波器)
- 交叉作用在所有激素更新后统一应用

核心类:
1. HormoneSystem - 激素系统（内含SCN）
"""

import math
import numpy as np
from typing import Dict
from dataclasses import dataclass

try:
    from simulacrum.core.scn import (
        SuprachiasmaticNucleus,
        LightType,
        CircadianOutput,
        create_scn,
    )
except ImportError:
    from scn import (
        SuprachiasmaticNucleus,
        LightType,
        CircadianOutput,
        create_scn,
    )


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


@dataclass
class HormoneLevels:
    """激素水平快照 + 受体敏感性"""
    cortisol: float = 0.3
    melatonin: float = 0.2
    adrenaline: float = 0.2
    oxytocin: float = 0.3
    dopamine: float = 0.5       # 由 NT 系统驱动，此处仅作快照
    serotonin: float = 0.5
    # 受体敏感性
    gr_sensitivity: float = 1.0  # 糖皮质激素受体 (反馈效率)
    mr_sensitivity: float = 1.0  # 盐皮质激素受体 (基线稳定性)


class HormoneSystem:
    """
    激素调制系统 v2

    关键改进 vs v1:
    - 皮质醇从 HPA axis 读取，不再自行计算 (单一来源)
    - 肾上腺素: 二阶动力学 (快速上升 + 慢恢复)
    - 催产素: 阈值激活 + 非线性释放
    - 交叉作用: 皮质醇抑制催产素/褪黑素, 血清素促进催产素
    - GR/MR 受体敏感性: 慢性应激下调
    - 超日脉冲: 皮质醇 ~90 min 周期叠加
    - 肠脑轴双向: 皮质醇抑制肠道血清素合成
    """

    def __init__(self, chronotype_offset: float = 0.0, event_bus=None):
        self.levels = HormoneLevels()

        # SCN — 昼夜节律起搏器
        self.scn = create_scn(
            intrinsic_period=24.2,
            chronotype_offset=chronotype_offset,
        )
        self.circadian_phase = 0.0

        # 肠脑轴基线 (默认血清素合成率)
        self._gut_serotonin_baseline = 0.5

        # 历史
        self._history = []
        self._max_history = 100

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "brain_update",
                self._handle_brain_update,
                priority=2,
                name="hormones",
            )
            event_bus.subscribe(
                "sensory_neuro_update",
                self._handle_sensory_neuro_update,
                priority=3,
                name="hormones_sensory",
            )

    def _handle_sensory_neuro_update(self, event) -> Dict:
        """处理感觉-神经递质耦合事件。

        从感觉驱动的神经化学释放更新激素水平。
        """
        sensory_neuro = event.data.get('sensory_neurochemical', {})

        # 感觉驱动的肾上腺素 → 激素肾上腺素系统
        sensory_adrenaline = sensory_neuro.get('sensory_adrenaline', 0.0)
        if sensory_adrenaline > 0.1:
            # 触发激素肾上腺素激增
            current_adrenaline = self.levels.adrenaline
            # 快上升 (SAM 直接激活)
            self.levels.adrenaline = min(1.0, current_adrenaline + sensory_adrenaline * 0.2)

        # 感觉驱动的皮质醇 → 激素皮质醇系统
        sensory_cortisol = sensory_neuro.get('sensory_cortisol', 0.0)
        if sensory_cortisol > 0.1:
            # 触发激素皮质醇增加 (HPA 轴激活)
            current_cortisol = self.levels.cortisol
            self.levels.cortisol = min(1.0, current_cortisol + sensory_cortisol * 0.15)

        return {}

    # ── Event Handler ─────────────────────────────────────

    def _handle_brain_update(self, event) -> Dict:
        """Event-driven handler for brain_update events."""
        state = event.data.get("internal_state", {})
        result = self.step(internal_state=state)
        return result

    # ── 昼夜节律 ─────────────────────────────────────────

    def update_circadian(
        self,
        step: int,
        light_intensity: float = 0.3,
        light_type: LightType = LightType.INDOOR,
        is_awake: bool = True,
    ) -> float:
        """SCN → 松果体 → 褪黑素"""
        self.scn.step(
            light_input=light_intensity,
            light_type=light_type,
            is_awake=is_awake,
        )
        self.levels.melatonin = self.scn.output.melatonin
        self.circadian_phase = self.scn.molecular_clock.state.phase
        return self.levels.melatonin

    # ── 肾上腺素: 二阶动力学 ─────────────────────────────

    def _update_adrenaline(self, stress_input: float):
        """
        肾上腺素: 快速上升 (SAM 直接释放嗜铬细胞) + 慢恢复 (COMT/MAO 清除)

        生物学对应:
        - 上升: 蓝斑→肾上腺髓质, 几秒内释放, 时间常数 ~2-3 步
        - 恢复: 酶促降解, 半衰期 ~20-30 秒, 时间常数 ~15-20 步
        """
        target = 0.2 + 0.8 * _clamp(stress_input * 1.5)

        if target > self.levels.adrenaline:
            # 上升: 快 (SAM 直接激活)
            rate = 0.3
        else:
            # 恢复: 慢 (酶促清除)
            rate = 0.05

        self.levels.adrenaline += rate * (target - self.levels.adrenaline)
        self.levels.adrenaline = _clamp(self.levels.adrenaline)

    # ── 催产素: 阈值激活 ─────────────────────────────────

    def _update_oxytocin(self, social_valence: float):
        """
        催产素: 有激活阈值 + 释放后持续效应

        生物学对应:
        - 下丘脑室旁核/视上核释放 OT 需要足够社交刺激
        - OT 脉冲释放后有数分钟持续效应
        - 高皮质醇抑制 OT 释放 (交叉作用在 _apply_cross_talk)
        """
        threshold = 0.3  # 激活阈值
        if social_valence > threshold:
            # 超阈值: 释放 (非线性增强)
            excess = social_valence - threshold
            boost = excess * excess * 0.8  # 平方增强: 强社交刺激产生更强 OT 释放
            self.levels.oxytocin += 0.1 * boost
        else:
            # 低于阈值: 缓慢衰减 (OT 清除半衰期 ~3-5 min)
            self.levels.oxytocin *= 0.995

        self.levels.oxytocin = _clamp(self.levels.oxytocin)

    # ── 血清素: 肠脑双向 ─────────────────────────────────

    def _update_serotonin(self, gut_serotonin: float, cortisol_level: float = 0.3):
        """
        血清素: 肠脑轴双向耦合

        生物学对应:
        - 肠道产生 ~90% 血清素前体 (5-HTP)
        - 肠道血清素通过迷走神经和血液循环影响脑内 5-HT
        - 高皮质醇抑制肠道色氨酸羟化酶 (TPH1) → 减少 5-HT 合成
        """
        # 皮质醇对肠脑轴的抑制
        cortisol_suppress = max(0, cortisol_level - 0.5) * 0.25
        effective_gut = gut_serotonin * (1.0 - cortisol_suppress)

        # 耦合: 肠道信号 + 脑内维持
        coupling = 0.3
        self.levels.serotonin = (
            (1 - coupling) * self.levels.serotonin +
            coupling * effective_gut
        )

    # ── 受体敏感性动态 ───────────────────────────────────

    def _update_receptor_sensitivity(self, allostatic_load: float):
        """
        GR/MR 受体脱敏: 慢性应激降低受体敏感性

        生物学对应:
        - GR 下调: 慢性高皮质醇 → GR 内化/降解 → 反馈减弱 → 皮质醇更难被抑制
        - MR 相对稳定, 仅在极端应激时受影响
        - GR 恢复: 停止应激后 GR 逐渐上调
        """
        # GR 下调 + 恢复
        gr_downreg = 0.008 * max(0, self.levels.cortisol - 0.5)
        gr_recovery = 0.004 * (1.0 - self.levels.gr_sensitivity)
        self.levels.gr_sensitivity = _clamp(
            self.levels.gr_sensitivity - gr_downreg + gr_recovery,
            0.3, 1.0
        )

        # MR 仅在极端时受影响
        mr_stress = 0.003 * max(0, allostatic_load - 0.7)
        mr_recovery = 0.001 * (1.0 - self.levels.mr_sensitivity)
        self.levels.mr_sensitivity = _clamp(
            self.levels.mr_sensitivity - mr_stress + mr_recovery,
            0.5, 1.0
        )

    # ── 交叉作用 ─────────────────────────────────────────

    def _apply_cross_talk(self):
        """
        激素间交叉作用 (所有激素更新后统一应用)

        生物学对应:
        1. 皮质醇 ⊣ 催产素: GR 激活抑制 PVN 的 OT 神经元
        2. 皮质醇 ⊣ 褪黑素: 皮质醇抑制 AANAT 酶 (N-乙酰转移酶)
        3. 血清素 → 催产素: 5-HT2A 受体促进 OT 释放
        4. 催产素 ⊣ 皮质醇: OT 增强 HPA 负反馈 (已在 HPA 的 social_buffer)
        """
        # 1. 皮质醇抑制催产素
        cortisol_ot_inhib = max(0, self.levels.cortisol - 0.4) * 0.15
        self.levels.oxytocin *= (1.0 - cortisol_ot_inhib)

        # 2. 皮质醇抑制褪黑素 (AANAT 抑制)
        cortisol_melatonin_inhib = max(0, self.levels.cortisol - 0.5) * 0.12
        self.levels.melatonin *= (1.0 - cortisol_melatonin_inhib)

        # 3. 血清素促进催产素 (5-HT2A → PVN)
        serotonin_ot_boost = max(0, self.levels.serotonin - 0.5) * 0.02
        self.levels.oxytocin += serotonin_ot_boost

        # 4. 血清素稳定 (脑内 5-HT 维持稳态)
        self.levels.serotonin = _clamp(self.levels.serotonin, 0.1, 0.95)

    # ── 超日脉冲 ─────────────────────────────────────────

    def _cortisol_ultradian_pulse(self, base_cortisol: float, step_count: int) -> float:
        """
        皮质醇超日脉冲: ~90 分钟周期

        生物学对应:
        - 皮质醇以 ~60-90 min 脉冲分泌
        - 脉冲幅度与基线成正比 (应激时脉冲更大)
        - 早晨脉冲最强, 夜间最弱 (受昼夜节律调制)
        """
        period_steps = 90  # 90 步 ~ 90 分钟
        phase = (step_count % period_steps) / period_steps
        # 正弦脉冲: 只在正半周
        pulse_shape = max(0, math.sin(2 * math.pi * phase))
        # 脉冲幅度: 基线越高脉冲越大, 但有上限
        amplitude = min(0.15, base_cortisol * 0.2)
        return base_cortisol + amplitude * pulse_shape

    # ── 调制函数 ─────────────────────────────────────────

    def get_memory_encoding_modulation(self) -> float:
        """皮质醇倒U型记忆编码调制 (Yerkes-Dodson)"""
        c = self.levels.cortisol
        if c < 0.4:
            return 0.8 + c * 0.5
        elif c < 0.6:
            return 1.0 + (0.6 - abs(c - 0.5)) * 0.5
        else:
            return max(0.5, 1.0 - (c - 0.6) * 1.5)

    def get_exploration_modulation(self) -> float:
        """肾上腺素对探索行为的调制"""
        a = self.levels.adrenaline
        if a < 0.5:
            return 1.0 + a * 0.4
        else:
            return max(0.5, 1.0 - (a - 0.5) * 1.0)

    def get_social_modulation(self) -> float:
        """催产素对社交行为的调制"""
        return self.levels.oxytocin

    # ── 主步进 ───────────────────────────────────────────

    def step(
        self,
        step_count: int = 0,
        internal_state: Dict = None,
        **kwargs,
    ) -> Dict:
        """
        完整的激素系统步进

        关键变化:
        - 皮质醇从 internal_state["cortisol"] 读取 (HPA axis 单一来源)
        - 超日脉冲叠加在皮质醇上
        - 所有激素独立动力学 + 交叉作用
        """
        if internal_state is None:
            internal_state = {}

        # 0. 读取外部输入
        step_count = internal_state.get('step', step_count)
        gut_serotonin = internal_state.get('gut_serotonin', 0.5)
        social_engagement = internal_state.get('social_engagement', 0.5)
        allostatic_load = internal_state.get('allostatic_load', 0.0)
        light_intensity = internal_state.get('scn_alertness', 0.3)
        emotion_criticality = internal_state.get('emotion_criticality', 0.0)
        alignment_score = internal_state.get('alignment_score', 0.5)

        # 1. SCN 昼夜节律 (→ 褪黑素)
        self.update_circadian(step_count, light_intensity)

        # 2. 皮质醇: 从 HPA axis 读取 (单一来源)
        hpa_cortisol = internal_state.get('cortisol', 0.3)
        self.levels.cortisol = hpa_cortisol

        # 3. 超日脉冲叠加
        effective_cortisol = self._cortisol_ultradian_pulse(
            self.levels.cortisol, step_count
        )

        # 4. 肾上腺素: 二阶动力学
        stress_input = emotion_criticality * 0.6 + (1.0 - alignment_score) * 0.4
        self._update_adrenaline(stress_input)

        # 5. 催产素: 阈值激活
        self._update_oxytocin(social_engagement)

        # 6. 血清素: 肠脑双向 (含皮质醇反馈抑制)
        self._update_serotonin(gut_serotonin, effective_cortisol)

        # 7. 受体敏感性
        self._update_receptor_sensitivity(allostatic_load)

        # 8. 交叉作用 (在所有独立更新之后)
        self._apply_cross_talk()

        # 9. 写回 internal_state
        internal_state['hormone_cortisol'] = effective_cortisol
        internal_state['hormone_adrenaline'] = self.levels.adrenaline
        internal_state['hormone_melatonin'] = self.levels.melatonin
        internal_state['hormone_oxytocin'] = self.levels.oxytocin
        internal_state['encoding_modulation'] = self.get_memory_encoding_modulation()
        internal_state['exploration_modulation'] = self.get_exploration_modulation()
        internal_state['gr_sensitivity'] = self.levels.gr_sensitivity
        internal_state['mr_sensitivity'] = self.levels.mr_sensitivity

        # 10. 记录历史
        scn_out = self.scn.output
        snapshot = {
            'cortisol': self.levels.cortisol,
            'effective_cortisol': effective_cortisol,
            'melatonin': self.levels.melatonin,
            'adrenaline': self.levels.adrenaline,
            'oxytocin': self.levels.oxytocin,
            'serotonin': self.levels.serotonin,
            'gr_sensitivity': self.levels.gr_sensitivity,
            'mr_sensitivity': self.levels.mr_sensitivity,
            'encoding_modulation': self.get_memory_encoding_modulation(),
            'exploration_modulation': self.get_exploration_modulation(),
            'social_modulation': self.get_social_modulation(),
            'circadian_hour': self.scn.get_circadian_hour(),
            'core_temperature': scn_out.core_temperature,
            'alertness': scn_out.alertness,
            'sleep_pressure': scn_out.sleep_pressure,
            'wake_drive': scn_out.wake_drive,
        }
        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return snapshot

    def get_summary(self) -> Dict:
        """获取激素系统摘要"""
        return {
            'cortisol': self.levels.cortisol,
            'melatonin': self.levels.melatonin,
            'adrenaline': self.levels.adrenaline,
            'oxytocin': self.levels.oxytocin,
            'serotonin': self.levels.serotonin,
            'gr_sensitivity': self.levels.gr_sensitivity,
            'mr_sensitivity': self.levels.mr_sensitivity,
            'encoding_modulation': self.get_memory_encoding_modulation(),
            'exploration_modulation': self.get_exploration_modulation(),
            'circadian_phase': self.circadian_phase,
            'scn': self.scn.get_summary(),
        }


def create_hormone_system() -> HormoneSystem:
    """创建激素系统"""
    return HormoneSystem()


__all__ = [
    'HormoneLevels',
    'HormoneSystem',
    'create_hormone_system',
]
