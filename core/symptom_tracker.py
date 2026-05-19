"""症状追踪器 — Per-symptom detection, episode management, trajectory tracking.

替代 TherapeuticExperiment._compute_symptom_severity() 的单一标量，
提供6种症状的独立追踪:
  - 失眠 (Insomnia): 持续性, 需sleep系统整合
  - 惊恐发作 (Panic Attack): 阵发性, 快速上升/指数衰减
  - 快感缺失 (Anhedonia): 持续性, DA/BDNF驱动
  - 精神运动性激越 (Psychomotor Agitation): 持续性, NE/ arousal驱动
  - 反刍思维 (Rumination): 持续性, precision/PFC驱动
  - 过度警觉 (Hypervigilance): 持续性, NE/ arousal驱动

每种症状由 _internal_state 键的连续公式驱动，
有独立的检测阈值和缓解阈值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────

@dataclass
class SymptomEpisode:
    """阵发性症状发作 (如惊恐发作)。"""
    symptom_name: str
    onset_step: int
    peak_severity: float
    current_severity: float = 0.0
    duration_steps: int = 0
    is_active: bool = True
    trigger: str = "spontaneous"  # "spontaneous"/"cue_triggered"/"interoceptive"/"pharmacological"


@dataclass
class PersistentSymptom:
    """持续性症状追踪。"""
    symptom_name: str
    current_level: float = 0.0
    trajectory: List[float] = field(default_factory=list)
    detection_threshold: float = 0.4
    resolution_threshold: float = 0.25
    is_detected: bool = False
    first_detected_step: int = -1
    last_detected_step: int = -1


@dataclass
class SymptomSnapshot:
    """一个时间点的完整症状快照。"""
    step: int
    time_h: float
    # 活跃的阵发性发作
    active_episodes: Dict[str, SymptomEpisode] = field(default_factory=dict)
    # 持续性症状水平
    persistent_levels: Dict[str, float] = field(default_factory=dict)
    # 检测状态
    detected_symptoms: Dict[str, bool] = field(default_factory=dict)
    # 复合指标 (临床量表类比)
    insomnia_severity: float = 0.0       # ISI analog (0-1)
    panic_attack_frequency: float = 0.0  # 每周发作次数类比
    anhedonia_severity: float = 0.0      # SHAPS analog
    psychomotor_agitation: float = 0.0   # BPRS item analog
    rumination_level: float = 0.0        # RRS analog
    hypervigilance_level: float = 0.0    # PTSD checklist analog


# ──────────────────────────────────────────────────────
# 症状检测规则
# ──────────────────────────────────────────────────────

# 每种症状由 _internal_state 键的加权公式定义
# formula: {state_key: weight} — 加权求和
SYMPTOM_DETECTION_RULES: Dict[str, Dict] = {
    "insomnia": {
        "description": "失眠 — 入睡/维持困难",
        "formula": {
            "nt_gaba": -0.4,               # GABA低 → 难抑制
            "limbic_arousal": 0.3,          # arousal高 → 难平静
            "scn_melatonin_amplitude_mult": -0.2,  # 褪黑素低 → 难入睡
            "nt_norepinephrine": 0.1,       # NE高 → 过度觉醒
            "orexin_level": 0.15,           # orexin高 → 促进觉醒
        },
        "detection_threshold": 0.40,
        "resolution_threshold": 0.25,
        "requires_sleep": True,
    },
    "panic_attack": {
        "description": "惊恐发作 — 阵发性恐惧/窒息感/濒死感",
        "formula": {
            "nt_norepinephrine": 0.35,      # NE暴发
            "nt_gaba": -0.25,               # GABA不足 → 失抑制
            "limbic_arousal": 0.20,         # arousal飙升
            "prefrontal_inhibition": -0.15, # PFC抑制不足 → 无法下调
            "interoceptive_pe": 0.05,       # 内感受预测误差
        },
        "detection_threshold": 0.65,        # 高阈值 — 惊恐是离散事件
        "resolution_threshold": 0.30,
        "paroxysmal": True,
        "episode_onset_steps": 5,           # 快速上升 (~5步=5分钟)
        "episode_peak_duration": 10,        # 峰值持续10步
        "episode_decay_rate": 0.05,         # 指数衰减
        "episode_min_duration": 20,         # 最短20步
        "episode_max_duration": 60,         # 最长60步
    },
    "anhedonia": {
        "description": "快感缺失 — 无法体验愉悦",
        "formula": {
            "nt_dopamine": -0.50,           # DA低 → 奖赏缺失
            "plasticity_bdnf": -0.20,       # BDNF低 → 神经可塑性差
            "basal_ganglia_td_error_mult": -0.15,  # TD error低 → 预期奖赏缺失
            "hormone_oxytocin": -0.15,       # oxytocin低 → 社交奖赏缺失
        },
        "detection_threshold": 0.35,
        "resolution_threshold": 0.20,
    },
    "psychomotor_agitation": {
        "description": "精神运动性激越 — 坐立不安/踱步",
        "formula": {
            "nt_norepinephrine": 0.40,      # NE驱动
            "limbic_arousal": 0.30,         # arousal驱动
            "nt_gaba": -0.20,               # GABA不足 → 失抑制
            "prefrontal_inhibition": -0.10, # PFC抑制不足
        },
        "detection_threshold": 0.45,
        "resolution_threshold": 0.25,
    },
    "rumination": {
        "description": "反刍思维 — 重复消极思维",
        "formula": {
            "predictive_coding_precision_mult": 0.35,  # 精度高 → 过度关注
            "nt_serotonin": -0.25,          # 5-HT低 → 情绪调节差
            "limbic_arousal": 0.20,         # arousal驱动
            "prefrontal_inhibition": -0.15, # PFC抑制不足 → 无法中断
            "cortisol_level": 0.05,         # cortisol轻度贡献
        },
        "detection_threshold": 0.40,
        "resolution_threshold": 0.25,
    },
    "hypervigilance": {
        "description": "过度警觉 — 对威胁过度敏感",
        "formula": {
            "nt_norepinephrine": 0.40,      # NE驱动警觉
            "nt_gaba": -0.25,               # GABA不足 → 失抑制
            "limbic_arousal": 0.20,         # arousal高
            "brainstem_arousal_setpoint": 0.15,  # 脑干觉醒高
        },
        "detection_threshold": 0.45,
        "resolution_threshold": 0.25,
    },
}


# ──────────────────────────────────────────────────────
# 症状追踪器
# ──────────────────────────────────────────────────────

class SymptomTracker:
    """症状追踪器 — 替代单一标量的多症状追踪系统。"""

    def __init__(self) -> None:
        self.persistent: Dict[str, PersistentSymptom] = {}
        self.episodes: Dict[str, List[SymptomEpisode]] = {}
        self.episode_history: Dict[str, List[SymptomEpisode]] = {}

        # 初始化持续性症状
        for name, rule in SYMPTOM_DETECTION_RULES.items():
            if not rule.get("paroxysmal", False):
                self.persistent[name] = PersistentSymptom(
                    symptom_name=name,
                    detection_threshold=rule["detection_threshold"],
                    resolution_threshold=rule["resolution_threshold"],
                )
            else:
                self.episodes[name] = []
                self.episode_history[name] = []

    def step(
        self,
        state: Dict[str, float],
        current_step: int = 0,
        time_h: float = 0.0,
    ) -> SymptomSnapshot:
        """每步更新所有症状检测。

        Args:
            state: _internal_state 字典
            current_step: 当前模拟步数
            time_h: 当前时间 (小时)

        Returns:
            SymptomSnapshot with all symptom data
        """
        snapshot = SymptomSnapshot(step=current_step, time_h=time_h)

        # ── 持续性症状 ──
        for name, psym in self.persistent.items():
            rule = SYMPTOM_DETECTION_RULES[name]
            level = self._compute_level(rule, state)

            psym.current_level = level
            psym.trajectory.append(level)

            # 检测逻辑: 超过检测阈值 → 检测; 低于缓解阈值 → 缓解
            if level >= psym.detection_threshold:
                if not psym.is_detected:
                    psym.first_detected_step = current_step
                psym.is_detected = True
                psym.last_detected_step = current_step
            elif level <= psym.resolution_threshold:
                psym.is_detected = False

            snapshot.persistent_levels[name] = level
            snapshot.detected_symptoms[name] = psym.is_detected

        # ── 阵发性症状 (惊恐发作) ──
        for name in self.episodes:
            rule = SYMPTOM_DETECTION_RULES[name]
            level = self._compute_level(rule, state)

            # 管理发作
            active_episodes = self._manage_episodes(
                name, level, state, current_step, rule
            )

            snapshot.persistent_levels[name] = level
            snapshot.active_episodes[name] = active_episodes[-1] if active_episodes else None
            snapshot.detected_symptoms[name] = len(active_episodes) > 0

        # ── 复合指标 ──
        snapshot.insomnia_severity = snapshot.persistent_levels.get("insomnia", 0.0)
        snapshot.anhedonia_severity = snapshot.persistent_levels.get("anhedonia", 0.0)
        snapshot.psychomotor_agitation = snapshot.persistent_levels.get("psychomotor_agitation", 0.0)
        snapshot.rumination_level = snapshot.persistent_levels.get("rumination", 0.0)
        snapshot.hypervigilance_level = snapshot.persistent_levels.get("hypervigilance", 0.0)

        # 惊恐发作频率: 过去100步内的发作次数 → 每周类比
        panic_history = self.episode_history.get("panic_attack", [])
        recent = [e for e in panic_history if e.onset_step > current_step - 100]
        # 100步 ≈ 10小时 → 换算为每周频率
        snapshot.panic_attack_frequency = len(recent) * 7 * 24 / 10  # 每周发作次数

        return snapshot

    def _compute_level(
        self,
        rule: Dict,
        state: Dict[str, float],
    ) -> float:
        """计算症状水平 — 加权求和公式。"""
        formula = rule["formula"]
        level = 0.0

        for key, weight in formula.items():
            # 获取状态值
            val = float(state.get(key, 0.5))
            # 负权重: (1-val) × |weight|; 正权重: val × weight
            if weight < 0:
                level += (1.0 - val) * abs(weight)
            else:
                level += val * weight

        return max(0.0, min(1.0, level))

    def _manage_episodes(
        self,
        symptom_name: str,
        trigger_level: float,
        state: Dict[str, float],
        current_step: int,
        rule: Dict,
    ) -> List[SymptomEpisode]:
        """管理阵发性症状发作 (惊恐发作)。"""
        active = self.episodes.get(symptom_name, [])

        # ── 更新活跃发作 ──
        still_active = []
        for ep in active:
            ep.duration_steps += 1
            onset = rule.get("episode_onset_steps", 5)
            peak_dur = rule.get("episode_peak_duration", 10)
            decay = rule.get("episode_decay_rate", 0.05)

            # 发作曲线: 快速上升 → 峰值 → 指数衰减
            if ep.duration_steps < onset:
                ep.current_severity = ep.peak_severity * (ep.duration_steps / onset)
            elif ep.duration_steps < onset + peak_dur:
                ep.current_severity = ep.peak_severity
            else:
                elapsed = ep.duration_steps - onset - peak_dur
                ep.current_severity = ep.peak_severity * max(0.0, 1.0 - decay * elapsed)

            # 发作结束
            min_dur = rule.get("episode_min_duration", 20)
            if ep.current_severity < 0.05 and ep.duration_steps > min_dur:
                ep.is_active = False
                self.episode_history[symptom_name].append(ep)
            else:
                still_active.append(ep)

        self.episodes[symptom_name] = still_active

        # ── 触发新发作 ──
        threshold = rule.get("detection_threshold", 0.65)
        max_dur = rule.get("episode_max_duration", 60)

        if trigger_level >= threshold and len(still_active) == 0:
            # 检测触发类型
            intero_pe = float(state.get("interoceptive_pe", 0.0))
            if intero_pe > 0.5:
                trigger = "interoceptive"
            elif trigger_level > threshold + 0.15:
                trigger = "spontaneous"
            else:
                trigger = "cue_triggered"

            new_episode = SymptomEpisode(
                symptom_name=symptom_name,
                onset_step=current_step,
                peak_severity=trigger_level,
                current_severity=0.0,
                duration_steps=0,
                is_active=True,
                trigger=trigger,
            )
            self.episodes[symptom_name].append(new_episode)

        return self.episodes[symptom_name]

    def get_symptom_trajectory(
        self,
        symptom_name: str,
    ) -> List[Tuple[int, float]]:
        """获取特定症状的历史轨迹。"""
        psym = self.persistent.get(symptom_name)
        if psym:
            return [(i, v) for i, v in enumerate(psym.trajectory)]
        return []

    def get_episode_history(
        self,
        symptom_name: str,
    ) -> List[SymptomEpisode]:
        """获取阵发性症状的所有发作历史。"""
        return self.episode_history.get(symptom_name, [])


__all__ = [
    "SymptomEpisode",
    "PersistentSymptom",
    "SymptomSnapshot",
    "SYMPTOM_DETECTION_RULES",
    "SymptomTracker",
]