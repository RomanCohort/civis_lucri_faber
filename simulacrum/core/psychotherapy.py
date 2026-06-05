"""心理治疗模拟器 (Psychotherapy Simulator)

核心原理: 心理治疗不是新代码，而是对现有子系统的定向训练/调节。
与药物（即时参数覆盖）互补，治疗是渐进式微调（慢起慢落但持久）。

7种治疗流派:
1. CBT (认知行为治疗) — PFC→杏仁核抑制通路
2. 暴露疗法 — 杏仁核恐惧消退
3. DBT (辩证行为治疗) — PFC+边缘系统整合
4. EMDR (眼动脱敏再加工) — 双侧刺激→海马体再编码
5. 精神动力学 — DMN自省→潜意识整合
6. ACT (接纳承诺治疗) — 预测编码→认知灵活性
7. 人际取向 — 社会认知→关系修复

事件驱动:
    - 发布 THERAPY_SESSION_START / THERAPY_SESSION_END
    - 发布 THERAPY_PROGRESS_UPDATE
    - 订阅 PSYCHIATRIC_CONDITION_CHANGE: 疾病变化时调整治疗策略

生物参考文献:
    - Beck (1979): CBT for depression
    - Foa & Kozak (1986): 情绪处理理论 (暴露疗法)
    - Linehan (1993): DBT for BPD
    - Shapiro (2001): EMDR
    - Hayes et al. (2006): ACT
    - Klerman et al. (1984): 人际心理治疗
    - Castren (2005): 神经可塑性与心理治疗
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

try:
    from core.events import (
        THERAPY_PROGRESS_UPDATE,
        THERAPY_SESSION_END,
        THERAPY_SESSION_START,
    )
except ImportError:
    from core.events import (
        THERAPY_PROGRESS_UPDATE,
        THERAPY_SESSION_END,
    )


# ===== 枚举 =====

class TherapyModality(Enum):
    """治疗流派"""
    CBT = "CBT"
    EXPOSURE = "exposure"
    DBT = "DBT"
    EMDR = "EMDR"
    PSYCHODYNAMIC = "psychodynamic"
    ACT = "ACT"
    INTERPERSONAL = "interpersonal"


class SessionFrequency(Enum):
    """治疗频率"""
    DAILY = "daily"
    BIWEEKLY = "biweekly"   # 每两周
    WEEKLY = "weekly"
    BIWEEKLY_MONTH = "biweekly_month"  # 每月两次


class TherapyPhase(Enum):
    """治疗阶段"""
    ENGAGEMENT = "engagement"       # 建立治疗关系
    ACTIVE = "active"               # 活跃治疗
    MAINTENANCE = "maintenance"     # 维持期
    TERMINATION = "termination"     # 结束期


# ===== 数据结构 =====

@dataclass
class TherapySession:
    """单次治疗session记录"""
    session_id: int
    modality: TherapyModality
    intensity: float          # 治疗强度 [0,1]
    compliance: float         # 患者配合度 [0,1]
    synergy_bonus: float      # 药物协同加成 [0,1]
    phase: TherapyPhase       # 治疗阶段
    step: int                 # 发生时的步数
    effects_applied: dict[str, float] = field(default_factory=dict)
    duration_minutes: float = 50.0  # 标准治疗时长


@dataclass
class TherapyProgress:
    """治疗进展追踪"""
    modality: TherapyModality
    total_sessions: int = 0
    total_effect: float = 0.0     # 累积效应 [0, ∞)
    current_skill: float = 0.0    # 习得技能水平 [0,1]
    decay_rate: float = 0.002     # 技能衰减率 (Ebbinghaus遗忘)
    last_session_step: int = 0
    phase: TherapyPhase = TherapyPhase.ENGAGEMENT
    therapeutic_alliance: float = 0.3  # 治疗关系质量 [0,1]
    resistance_level: float = 0.0      # 治疗阻抗 [0,1]
    sessions: list[TherapySession] = field(default_factory=list)


# ===== 治疗流派调节目标 =====

THERAPY_TARGETS = {
    TherapyModality.CBT: {
        "name": "认知行为治疗 (CBT)",
        "description": "PFC→杏仁核抑制通路: 认知重评→情绪调节",
        "primary_targets": {
            "prefrontal_maturity": +0.05,       # PFC成熟度↑
            "prefrontal_inhibition": +0.05,     # PFC抑制力↑
            "emotion_regulation_regulation_capacity": +0.04,  # 调节容量↑
            "emotion_regulation_inhibition": +0.03,           # 抑制↑
            "limbic_valence": +0.03,            # 效价↑ (更积极)
            "predictive_coding_precision_mult": -0.02,  # 精确度↓ (减少灾难化)
        },
        "indicated_for": ["MDD", "GAD", "OCD", "Social_Anxiety", "PTSD", "Insomnia"],
        "optimal_drug_synergy": ["antidepressant", "stimulant"],
        "antagonistic_drugs": [],
    },
    TherapyModality.EXPOSURE: {
        "name": "暴露疗法 (Exposure Therapy)",
        "description": "杏仁核恐惧消退: 反复暴露→恐惧记忆再巩固→消退",
        "primary_targets": {
            "hpa_axis_stress_reactivity_mult": -0.04,  # 应激反应性↓
            "hpa_axis_feedback_strength_mult": +0.03,  # 负反馈↑
            "limbic_arousal": -0.03,                   # 唤醒度↓
            "ans_sympathetic_reactivity_mult": -0.03,  # 交感反应性↓
            "ans_baseline_vagal_tone": +0.03,          # 迷走张力↑
            "brainstem_arousal_setpoint": -0.02,       # 唤醒设定点↓
        },
        "indicated_for": ["Specific_Phobia", "Social_Anxiety", "PTSD", "Panic_Disorder", "OCD", "Agoraphobia"],
        "optimal_drug_synergy": ["anxiolytic"],  # D-cycloserine ideal but not in presets
        "antagonistic_drugs": ["sedative"],  # 苯二氮卓拮抗暴露学习
    },
    TherapyModality.DBT: {
        "name": "辩证行为治疗 (DBT)",
        "description": "PFC+边缘系统整合: 情绪调节技能+正念+人际效能",
        "primary_targets": {
            "emotion_regulation_regulation_capacity": +0.06,  # 调节容量↑↑
            "emotion_regulation_inhibition": +0.04,           # 抑制↑
            "prefrontal_inhibition": +0.04,                   # PFC抑制↑
            "mood_system_volatility_mult": -0.03,             # 波动性↓
            "self_awareness_coherence": +0.03,                # 自我连贯性↑
            "social_cognition_affective_empathy": +0.02,      # 情感共情↑
        },
        "indicated_for": ["BPD", "emotional_dysregulation", "emotional_lability", "MDD", "Substance_Use"],
        "optimal_drug_synergy": ["antidepressant", "anxiolytic"],
        "antagonistic_drugs": [],
    },
    TherapyModality.EMDR: {
        "name": "眼动脱敏再加工 (EMDR)",
        "description": "双侧刺激→海马体再编码: 创伤记忆脱敏+适应性信息加工",
        "primary_targets": {
            "hippocampus_encoding_modulation": +0.05,         # 海马编码↑
            "hpa_axis_stress_reactivity_mult": -0.03,        # 应激反应↓
            "limbic_arousal": -0.03,                          # 唤醒↓
            "self_awareness_coherence": +0.03,                # 自我连贯↑
            "self_awareness_agency": +0.03,                   # 主体感↑
            "predictive_coding_free_energy_bias": -0.02,     # 自由能偏置↓
        },
        "indicated_for": ["PTSD", "Specific_Phobia", "Panic_Disorder", "GAD"],
        "optimal_drug_synergy": ["anxiolytic"],
        "antagonistic_drugs": ["sedative"],
    },
    TherapyModality.PSYCHODYNAMIC: {
        "name": "精神动力学治疗 (Psychodynamic)",
        "description": "DMN自省→潜意识整合: 探索防御机制+移情+内在冲突",
        "primary_targets": {
            "self_awareness_coherence": +0.04,                # 自我连贯↑
            "self_awareness_agency": +0.03,                   # 主体感↑
            "self_awareness_introspection_depth": +0.05,      # 内省深度↑↑
            "emotion_regulation_regulation_capacity": +0.03,  # 调节容量↑
            "social_cognition_affective_empathy": +0.03,      # 情感共情↑
            "limbic_valence": +0.02,                          # 效价↑
        },
        "indicated_for": ["MDD", "Dysthymia", "BPD", "Prolonged_Grief", "Dissociative"],
        "optimal_drug_synergy": ["antidepressant"],
        "antagonistic_drugs": [],  # SSRI过度→情感钝化→拮抗，但由synergy模块处理
    },
    TherapyModality.ACT: {
        "name": "接纳承诺治疗 (ACT)",
        "description": "预测编码→认知灵活性: 心理灵活性+价值导向行动+接纳",
        "primary_targets": {
            "predictive_coding_precision_mult": -0.03,        # 精确度↓ (减少僵化)
            "predictive_coding_free_energy_bias": -0.02,     # 自由能↓
            "emotion_regulation_regulation_capacity": +0.03,  # 调节容量↑
            "self_awareness_agency": +0.04,                   # 主体感↑
            "prefrontal_maturity": +0.03,                     # PFC成熟度↑
            "mood_system_volatility_mult": -0.02,             # 波动性↓
        },
        "indicated_for": ["GAD", "MDD", "OCD", "Substance_Use", "Burnout", "emotional_dysregulation"],
        "optimal_drug_synergy": ["antidepressant", "stimulant"],
        "antagonistic_drugs": [],
    },
    TherapyModality.INTERPERSONAL: {
        "name": "人际取向治疗 (Interpersonal)",
        "description": "社会认知→关系修复: 改善人际模式+社会支持+角色转换",
        "primary_targets": {
            "social_cognition_affective_empathy": +0.05,      # 情感共情↑↑
            "social_cognition_cognitive_empathy": +0.04,      # 认知共情↑
            "social_cognition_contagion": +0.03,              # 情绪传染↑ (适度)
            "hormone_oxytocin": +0.04,                        # 催产素↑
            "self_awareness_coherence": +0.03,                # 自我连贯↑
            "limbic_valence": +0.02,                          # 效价↑
        },
        "indicated_for": ["MDD", "Social_Anxiety", "Prolonged_Grief", "BPD", "Dysthymia"],
        "optimal_drug_synergy": ["empathogen", "anxiolytic"],
        "antagonistic_drugs": [],
    },
}


# ===== 治疗阻抗计算 =====

def compute_resistance(
    severity: float,
    pfc_maturity: float,
    therapeutic_alliance: float,
    personality_traits: dict | None = None,
) -> float:
    """计算治疗阻抗

    阻抗来源:
    1. 疾病严重度 → 高严重度→高阻抗
    2. PFC不成熟 → 执行功能差→难以参与治疗
    3. 治疗关系差 → 低联盟→低配合
    4. 人格特质 → 高神经质→阻抗↑

    Returns:
        resistance [0, 1]
    """
    base = 0.1
    severity_component = severity * 0.3
    pfc_component = (1.0 - pfc_maturity) * 0.2
    alliance_component = (1.0 - therapeutic_alliance) * 0.3

    personality_component = 0.0
    if personality_traits:
        neuroticism = personality_traits.get("neuroticism", 0.5)
        personality_component = neuroticism * 0.1

    resistance = base + severity_component + pfc_component + alliance_component + personality_component
    return float(np.clip(resistance, 0.0, 0.95))


def compute_compliance(resistance: float, session_intensity: float) -> float:
    """计算患者配合度

    compliance = (1 - resistance) * intensity_adjustment
    """
    base = 1.0 - resistance
    # 高强度治疗可能降低配合度 (过载)
    intensity_penalty = max(0, (session_intensity - 0.8) * 0.3)
    return float(np.clip(base - intensity_penalty, 0.05, 1.0))


# ===== 单次治疗 Session =====

def conduct_therapy_session(
    modality: TherapyModality,
    agent,
    session_intensity: float = 0.7,
    synergy_bonus: float = 0.0,
    phase: TherapyPhase = TherapyPhase.ACTIVE,
    step: int = 0,
    event_bus=None,
) -> TherapySession:
    """执行一次治疗session

    流程: 评估→选择干预→应用微调→记录

    Args:
        modality: 治疗流派
        agent: Simulacrum智能体
        session_intensity: 治疗强度 [0,1]
        synergy_bonus: 药物协同加成 [0,1]
        phase: 治疗阶段
        step: 当前步数
        event_bus: 事件总线

    Returns:
        TherapySession 记录
    """
    if agent is None:
        return TherapySession(
            session_id=0, modality=modality, intensity=session_intensity,
            compliance=0.0, synergy_bonus=synergy_bonus, phase=phase, step=step,
        )

    targets = THERAPY_TARGETS.get(modality, {})
    primary_targets = targets.get("primary_targets", {})

    # 1. 评估当前状态 → 计算阻抗
    s = agent._internal_state
    severity = float(np.clip(
        max(0, -s.get("limbic_valence", 0)) + s.get("limbic_arousal", 0.5) * 0.5,
        0, 1
    ))
    pfc_maturity = s.get("prefrontal_maturity", 0.5)
    alliance = s.get("therapy_alliance", 0.3)

    resistance = compute_resistance(severity, pfc_maturity, alliance)
    compliance = compute_compliance(resistance, session_intensity)

    # 2. 应用渐进式微调
    effects_applied = {}
    for state_key, delta in primary_targets.items():
        current = s.get(state_key, 0.5)

        # 核心公式: current + delta * intensity * compliance * (1 + synergy)
        adjustment = delta * session_intensity * compliance * (1.0 + synergy_bonus)

        # 维持期效果减半
        if phase == TherapyPhase.MAINTENANCE:
            adjustment *= 0.5
        # 结束期效果再减半
        elif phase == TherapyPhase.TERMINATION:
            adjustment *= 0.25

        new_val = current + adjustment
        # 钳制到合理范围
        if "mult" in state_key:
            new_val = max(0.05, new_val)  # 乘数不能为0
        else:
            new_val = float(np.clip(new_val, 0.0, 1.0))

        s[state_key] = new_val
        effects_applied[state_key] = adjustment

    # 3. 更新治疗关系质量 (每次session微增)
    alliance_delta = 0.02 * compliance * session_intensity
    s["therapy_alliance"] = float(np.clip(
        s.get("therapy_alliance", 0.3) + alliance_delta, 0.0, 1.0
    ))

    # 4. 创建session记录
    session = TherapySession(
        session_id=int(s.get("therapy_session_count", 0)),
        modality=modality,
        intensity=session_intensity,
        compliance=compliance,
        synergy_bonus=synergy_bonus,
        phase=phase,
        step=step,
        effects_applied=effects_applied,
    )
    s["therapy_session_count"] = s.get("therapy_session_count", 0) + 1

    # 5. 发布事件
    if event_bus is not None:
        event_bus.publish(
            THERAPY_SESSION_END,
            {
                "modality": modality.value,
                "intensity": session_intensity,
                "compliance": compliance,
                "synergy_bonus": synergy_bonus,
                "effects": effects_applied,
                "phase": phase.value,
            },
            source="psychotherapy",
        )

    return session


# ===== 心理治疗系统主类 =====

class PsychotherapySystem:
    """心理治疗系统

    管理7种治疗流派的治疗进展、session调度、技能衰减。

    用法:
        therapy = PsychotherapySystem(agent)
        therapy.start_treatment("CBT", frequency="weekly")
        # 在agent.step()中调用 therapy.step() 推进治疗
        therapy.stop_treatment("CBT")
    """

    def __init__(self, agent=None, event_bus=None):
        self._agent = agent
        self._bus = event_bus or (agent.bus if hasattr(agent, 'bus') else None)
        self.active_treatments: dict[str, TherapyProgress] = {}
        self._step_count = 0

        # 频率→步间隔映射 (假设1步=1秒, 1天=86400步)
        self._frequency_intervals = {
            SessionFrequency.DAILY: 86400,
            SessionFrequency.BIWEEKLY: 86400 * 3,     # 每3天
            SessionFrequency.WEEKLY: 86400 * 7,
            SessionFrequency.BIWEEKLY_MONTH: 86400 * 14,
        }

    def start_treatment(
        self,
        modality: str,
        frequency: str = "weekly",
        initial_intensity: float = 0.7,
    ) -> bool:
        """开始心理治疗

        Args:
            modality: 流派名 ("CBT", "exposure", "DBT", "EMDR", "psychodynamic", "ACT", "interpersonal")
            frequency: 治疗频率 ("daily", "biweekly", "weekly", "biweekly_month")
            initial_intensity: 初始治疗强度 [0,1]
        """
        mod_enum = self._resolve_modality(modality)
        if mod_enum is None:
            print(f"[Therapy] Unknown modality: {modality}")
            return False

        freq_enum = SessionFrequency(frequency)
        key = mod_enum.value

        self.active_treatments[key] = TherapyProgress(
            modality=mod_enum,
            phase=TherapyPhase.ENGAGEMENT,
        )

        if self._agent is not None:
            self._agent._internal_state[f"therapy_{key}_active"] = 1.0
            self._agent._internal_state[f"therapy_{key}_frequency"] = freq_enum.value
            self._agent._internal_state[f"therapy_{key}_intensity"] = initial_intensity

        print(f"[Therapy] Started {THERAPY_TARGETS[mod_enum]['name']} (frequency={frequency})")
        return True

    def stop_treatment(self, modality: str) -> bool:
        """停止心理治疗 (进入终止期或直接停止)"""
        mod_enum = self._resolve_modality(modality)
        if mod_enum is None:
            return False

        key = mod_enum.value
        if key not in self.active_treatments:
            return False

        # 进入终止期
        self.active_treatments[key].phase = TherapyPhase.TERMINATION

        if self._agent is not None:
            self._agent._internal_state[f"therapy_{key}_active"] = 0.0

        print(f"[Therapy] Stopped {key}")
        return True

    def step(self, synergy_bonuses: dict[str, float] | None = None) -> dict[str, Any]:
        """每步推进所有活跃治疗

        包含:
        1. 判断是否该做session (基于频率)
        2. 技能衰减 (Ebbinghaus遗忘曲线)
        3. 治疗阶段转换

        Args:
            synergy_bonuses: {modality_name: synergy_factor} 药物协同加成

        Returns:
            各治疗的当前状态
        """
        self._step_count += 1
        if synergy_bonuses is None:
            synergy_bonuses = {}

        summary = {}
        to_remove = []

        for key, progress in self.active_treatments.items():
            # 1. 技能衰减 (Ebbinghaus: R = e^(-λt))
            steps_since_last = self._step_count - progress.last_session_step
            if steps_since_last > 0 and progress.current_skill > 0:
                decay = np.exp(-progress.decay_rate * steps_since_last)
                progress.current_skill *= decay

            # 2. 判断是否该做session
            freq_str = "weekly"
            if self._agent is not None:
                freq_str = self._agent._internal_state.get(
                    f"therapy_{key}_frequency", "weekly"
                )
            freq_enum = SessionFrequency(freq_str)
            interval = self._frequency_intervals.get(freq_enum, 86400 * 7)

            should_session = (self._step_count - progress.last_session_step) >= interval

            # 简化: 每10步做一次session (用于快速模拟)
            # 实际部署时用真实时间间隔
            should_session = (self._step_count - progress.last_session_step) >= 10

            if should_session and progress.phase != TherapyPhase.TERMINATION:
                # 3. 执行session
                intensity = 0.7
                if self._agent is not None:
                    intensity = self._agent._internal_state.get(
                        f"therapy_{key}_intensity", 0.7
                    )
                synergy = synergy_bonuses.get(key, 0.0)

                session = conduct_therapy_session(
                    modality=progress.modality,
                    agent=self._agent,
                    session_intensity=intensity,
                    synergy_bonus=synergy,
                    phase=progress.phase,
                    step=self._step_count,
                    event_bus=self._bus,
                )

                progress.sessions.append(session)
                progress.total_sessions += 1
                progress.last_session_step = self._step_count

                # 累积效应
                progress.total_effect += session.intensity * session.compliance

                # 习得技能增长 (渐进，有上限)
                skill_gain = 0.03 * session.intensity * session.compliance * (1.0 + synergy)
                progress.current_skill = min(1.0, progress.current_skill + skill_gain)

                # 4. 治疗阶段转换
                if progress.phase == TherapyPhase.ENGAGEMENT and progress.total_sessions >= 3:
                    progress.phase = TherapyPhase.ACTIVE
                elif progress.phase == TherapyPhase.ACTIVE and progress.current_skill >= 0.6:
                    progress.phase = TherapyPhase.MAINTENANCE

                # 5. 更新阻抗
                s = self._agent._internal_state if self._agent else {}
                severity = float(np.clip(
                    max(0, -s.get("limbic_valence", 0)) + s.get("limbic_arousal", 0.5) * 0.5,
                    0, 1
                ))
                progress.resistance_level = compute_resistance(
                    severity,
                    s.get("prefrontal_maturity", 0.5),
                    progress.therapeutic_alliance,
                )

            # 终止期: 效果衰减，完成后移除
            if progress.phase == TherapyPhase.TERMINATION:
                progress.current_skill *= 0.95
                if progress.current_skill < 0.01:
                    to_remove.append(key)

            # 记录摘要
            summary[key] = {
                "modality": progress.modality.value,
                "phase": progress.phase.value,
                "total_sessions": progress.total_sessions,
                "current_skill": progress.current_skill,
                "total_effect": progress.total_effect,
                "resistance": progress.resistance_level,
                "alliance": progress.therapeutic_alliance,
            }

        # 清除已完成终止的治疗
        for key in to_remove:
            del self.active_treatments[key]

        # 发布进展事件
        if self._bus is not None and summary:
            self._bus.publish(
                THERAPY_PROGRESS_UPDATE,
                {"treatments": summary},
                source="psychotherapy",
            )

        return summary

    def get_active_treatments(self) -> list[str]:
        """获取当前活跃的治疗列表"""
        return list(self.active_treatments.keys())

    def get_treatment_state(self, modality: str) -> dict | None:
        """获取特定治疗的当前状态"""
        mod_enum = self._resolve_modality(modality)
        if mod_enum is None:
            return None
        key = mod_enum.value
        if key not in self.active_treatments:
            return None
        p = self.active_treatments[key]
        return {
            "modality": p.modality.value,
            "phase": p.phase.value,
            "total_sessions": p.total_sessions,
            "current_skill": p.current_skill,
            "total_effect": p.total_effect,
            "resistance": p.resistance_level,
            "alliance": p.therapeutic_alliance,
        }

    def get_available_modalities(self) -> dict[str, str]:
        """获取所有可用的治疗流派"""
        return {m.value: THERAPY_TARGETS[m]["name"] for m in TherapyModality}

    def get_indicated_modalities(self, condition_id: str) -> list[str]:
        """获取某疾病适应的治疗流派"""
        result = []
        for mod, targets in THERAPY_TARGETS.items():
            if condition_id in targets.get("indicated_for", []):
                result.append(mod.value)
        return result

    @staticmethod
    def _resolve_modality(name: str) -> TherapyModality | None:
        """流派名→枚举"""
        aliases = {
            "cbt": TherapyModality.CBT, "CBT": TherapyModality.CBT,
            "exposure": TherapyModality.EXPOSURE, "exposure_therapy": TherapyModality.EXPOSURE,
            "dbt": TherapyModality.DBT, "DBT": TherapyModality.DBT,
            "emdr": TherapyModality.EMDR, "EMDR": TherapyModality.EMDR,
            "psychodynamic": TherapyModality.PSYCHODYNAMIC,
            "act": TherapyModality.ACT, "ACT": TherapyModality.ACT,
            "interpersonal": TherapyModality.INTERPERSONAL, "ipt": TherapyModality.INTERPERSONAL,
        }
        return aliases.get(name)


# ===== 便捷函数 =====

def create_psychotherapy_system(agent=None, event_bus=None) -> PsychotherapySystem:
    """创建心理治疗系统"""
    return PsychotherapySystem(agent=agent, event_bus=event_bus)


__all__ = [
    "TherapyModality",
    "SessionFrequency",
    "TherapyPhase",
    "TherapySession",
    "TherapyProgress",
    "PsychotherapySystem",
    "conduct_therapy_session",
    "compute_resistance",
    "compute_compliance",
    "THERAPY_TARGETS",
    "create_psychotherapy_system",
]
