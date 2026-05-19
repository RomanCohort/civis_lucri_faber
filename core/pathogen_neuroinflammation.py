"""病原体触发神经炎症 — Pathogen-triggered neuroinflammation.

将感染源引入神经炎症模型:
  旧: microglia只响应stress/damage信号
  新: pathogen → TLR激活 → 细胞因子风暴 → BBB破坏 → 神经毒性 → 认知/情绪/运动/睡眠损害

6种病原体档案:
  - Lyme神经螺旋体 (Borrelia burgdorferi): 复发-缓解, TLR2/4
  - 弓形虫 (Toxoplasma gondii): 慢性, 前额叶/基底节偏好, TLR11/2
  - 神经梅毒 (Treponema pallidum): 慢性进行性, TLR2
  - 病毒性脑炎 (HSV-1): 急性, 颞叶/海马偏好, TLR3/7
  - COVID神经效应 (SARS-CoV-2): 亚急性, TLR7/4
  - 朊病毒病 (Prion): 进行性致死, TLR2/4弱

参考文献:
  - Ransohoff & Brown (2012) Nat Rev Immunol — 小胶质细胞激活
  - Heneka et al. (2015) Lancet Neurol — 神经炎症与神经退行
  - Sweeney et al. (2018) Nat Rev Neurol — BBB破坏
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────
# 病原体档案
# ──────────────────────────────────────────────────────

@dataclass
class PathogenProfile:
    """病原体档案 — 定义感染特征和神经影响。"""
    name: str
    pathogen_type: str              # "bacterial"/"spirochete"/"parasitic"/"viral"/"prion"
    bbb_disruption_rate: float      # 血脑屏障破坏速率 (0-1)
    tlr_activation: Dict[str, float]  # {"TLR2": 0.8, "TLR4": 0.3}
    cytokine_pattern: Dict[str, float]  # {"IL-1beta": 3.0, "TNF-alpha": 2.5}
    neurotoxin_production: float    # 神经毒素产生速率
    chronicity: str                 # "acute"/"subacute"/"chronic"/"relapsing_remitting"/"progressive"
    resolution_rate: float          # 自愈/清除速率
    target_regions: List[str]       # 偏好脑区
    cognitive_impact: float         # 认知损害权重
    mood_impact: float              # 情绪损害权重
    motor_impact: float             # 运动损害权重
    sleep_impact: float             # 睡眠损害权重
    growth_rate: float = 0.01       # 病原体负荷增长速率
    max_load: float = 1.0           # 最大负荷


PATHOGEN_REGISTRY: Dict[str, PathogenProfile] = {
    "lyme_neuroborreliosis": PathogenProfile(
        name="Lyme Neuroborreliosis (B. burgdorferi)",
        pathogen_type="spirochete",
        bbb_disruption_rate=0.03,
        tlr_activation={"TLR2": 0.8, "TLR4": 0.3},
        cytokine_pattern={"IL-1beta": 2.5, "TNF-alpha": 2.0, "IL-6": 1.8, "IFN-gamma": 1.5},
        neurotoxin_production=0.02,
        chronicity="relapsing_remitting",
        resolution_rate=0.005,       # 抗生素可加速
        target_regions=["meninges", "cranial_nerves", "spinal_cord"],
        cognitive_impact=0.3,
        mood_impact=0.4,             # Lyme可致抑郁/焦虑
        motor_impact=0.5,            # 面瘫/神经根痛
        sleep_impact=0.3,
        growth_rate=0.008,
    ),
    "toxoplasma": PathogenProfile(
        name="Toxoplasma gondii",
        pathogen_type="parasitic",
        bbb_disruption_rate=0.02,
        tlr_activation={"TLR11": 0.7, "TLR2": 0.4},
        cytokine_pattern={"IL-12": 2.0, "IFN-gamma": 2.5, "TNF-alpha": 1.5, "IL-1beta": 1.0},
        neurotoxin_production=0.01,
        chronicity="chronic",
        resolution_rate=0.001,       # 慢性潜伏, 极难完全清除
        target_regions=["prefrontal", "basal_ganglia", "amygdala"],
        cognitive_impact=0.2,
        mood_impact=0.5,             # 增加精神分裂症风险
        motor_impact=0.2,
        sleep_impact=0.2,
        growth_rate=0.003,
    ),
    "neurosyphilis": PathogenProfile(
        name="Neurosyphilis (T. pallidum)",
        pathogen_type="bacterial",
        bbb_disruption_rate=0.04,
        tlr_activation={"TLR2": 0.9},
        cytokine_pattern={"IL-1beta": 3.0, "TNF-alpha": 2.5, "IL-6": 2.0},
        neurotoxin_production=0.03,
        chronicity="chronic",
        resolution_rate=0.008,       # 青霉素有效
        target_regions=["brainstem", "spinal_cord", "frontal"],
        cognitive_impact=0.6,        # 痴呆 (general paresis)
        mood_impact=0.5,             # 人格改变
        motor_impact=0.7,            # 脊髓痨
        sleep_impact=0.3,
        growth_rate=0.005,
    ),
    "viral_encephalitis": PathogenProfile(
        name="Viral Encephalitis (HSV-1)",
        pathogen_type="viral",
        bbb_disruption_rate=0.06,
        tlr_activation={"TLR3": 0.9, "TLR7": 0.7},
        cytokine_pattern={"IFN-alpha": 4.0, "IL-6": 3.0, "TNF-alpha": 2.5, "IL-1beta": 2.0},
        neurotoxin_production=0.05,
        chronicity="acute",
        resolution_rate=0.02,        # 抗病毒治疗+免疫清除
        target_regions=["temporal", "hippocampus", "limbic"],
        cognitive_impact=0.7,        # 严重记忆损害
        mood_impact=0.3,
        motor_impact=0.3,
        sleep_impact=0.4,
        growth_rate=0.02,
    ),
    "covid_neuro": PathogenProfile(
        name="COVID-19 Neurological Effects (SARS-CoV-2)",
        pathogen_type="viral",
        bbb_disruption_rate=0.03,
        tlr_activation={"TLR7": 0.8, "TLR4": 0.5},
        cytokine_pattern={"IL-6": 3.5, "TNF-alpha": 2.0, "IL-1beta": 2.0, "IFN-gamma": 1.5},
        neurotoxin_production=0.02,
        chronicity="subacute",
        resolution_rate=0.01,
        target_regions=["olfactory", "brainstem", "prefrontal"],
        cognitive_impact=0.4,        # 脑雾
        mood_impact=0.4,             # 焦虑/抑郁
        motor_impact=0.2,
        sleep_impact=0.5,            # 失眠常见
        growth_rate=0.01,
    ),
    "prion_disease": PathogenProfile(
        name="Prion Disease (CJD)",
        pathogen_type="prion",
        bbb_disruption_rate=0.01,
        tlr_activation={"TLR2": 0.3, "TLR4": 0.2},  # 弱免疫激活
        cytokine_pattern={"IL-1beta": 1.5, "TNF-alpha": 1.0},
        neurotoxin_production=0.08,   # 蛋白错误折叠直接毒性
        chronicity="progressive",
        resolution_rate=0.0,          # 不可逆
        target_regions=["cortex", "thalamus", "cerebellum", "basal_ganglia"],
        cognitive_impact=0.9,         # 快速进展性痴呆
        mood_impact=0.3,
        motor_impact=0.7,             # 肌阵挛
        sleep_impact=0.6,
        growth_rate=0.015,
    ),
}


# ──────────────────────────────────────────────────────
# 病原体状态
# ──────────────────────────────────────────────────────

@dataclass
class PathogenState:
    """运行时病原体状态。"""
    pathogen_name: str
    load: float = 0.1              # 病原体负荷 (0-1)
    bbb_disruption: float = 0.0    # BBB破坏程度 (0-1)
    tlr_signal: float = 0.0        # TLR信号强度
    cytokine_boost: Dict[str, float] = field(default_factory=dict)
    neurotoxin_level: float = 0.0
    is_active: bool = True
    duration_steps: int = 0
    treatment_response: float = 0.0  # 治疗响应 (0-1)
    relapse_count: int = 0


# ──────────────────────────────────────────────────────
# 病原体神经炎症引擎
# ──────────────────────────────────────────────────────

class PathogenTriggeredInflammationEngine:
    """病原体触发的神经炎症引擎。"""

    def __init__(self) -> None:
        self.states: Dict[str, PathogenState] = {}

    def register_pathogen(
        self,
        pathogen_name: str,
        initial_load: float = 0.1,
    ) -> None:
        """注册一个病原体。"""
        profile = PATHOGEN_REGISTRY.get(pathogen_name)
        if profile is None:
            raise ValueError(f"Unknown pathogen: {pathogen_name}")

        self.states[pathogen_name] = PathogenState(
            pathogen_name=pathogen_name,
            load=initial_load,
            cytokine_boost={k: 0.0 for k in profile.cytokine_pattern},
        )

    def step(
        self,
        treatment_efficacy: Dict[str, float] = None,
    ) -> Tuple[
        float,                          # enhanced_damage_signal
        Dict[str, float],               # cytokine_boost
        Dict[str, float],               # state_deltas
    ]:
        """每步更新病原体状态和神经炎症。

        Args:
            treatment_efficacy: {pathogen_name: efficacy (0-1)}

        Returns:
            (enhanced_damage_signal, cytokine_boost, state_deltas)
        """
        treatment = treatment_efficacy or {}
        total_damage = 0.0
        total_cytokines: Dict[str, float] = {}
        total_deltas: Dict[str, float] = {}

        for name, state in self.states.items():
            if not state.is_active:
                continue

            profile = PATHOGEN_REGISTRY[name]
            state.duration_steps += 1

            # ── 1. 病原体负荷动态 ──
            treatment_eff = treatment.get(name, 0.0)
            state.treatment_response = treatment_eff

            # 增长 vs 治疗 vs 自愈
            growth = profile.growth_rate * state.load * (1.0 - state.load / profile.max_load)
            clearance = (profile.resolution_rate + treatment_eff * 0.05) * state.load
            state.load = max(0.0, min(profile.max_load, state.load + growth - clearance))

            # 复发-缓解型: 周期性负荷波动
            if profile.chronicity == "relapsing_remitting":
                cycle = 0.1 * (1.0 + 0.5 * (
                    state.duration_steps % 500 < 100
                ))
                state.load = min(profile.max_load, state.load + cycle * 0.01)

            # 朊病毒: 不可逆增长
            if profile.chronicity == "progressive" and profile.resolution_rate == 0:
                state.load = min(profile.max_load, state.load + growth * 0.5)

            # 负荷归零 → 不再活跃 (朊病毒除外)
            if state.load < 0.01 and profile.resolution_rate > 0:
                state.is_active = False
                state.load = 0.0
                continue

            # ── 2. BBB破坏 ──
            bbd_growth = profile.bbb_disruption_rate * state.load
            bbd_recovery = 0.001 * (1.0 - state.load)  # 无负荷时缓慢修复
            state.bbb_disruption = max(0.0, min(1.0,
                state.bbb_disruption + bbd_growth - bbd_recovery
            ))

            # ── 3. TLR信号 → 细胞因子 ──
            tlr_total = sum(
                weight * state.load for weight in profile.tlr_activation.values()
            )
            state.tlr_signal = min(1.0, tlr_total)

            for cytokine, base_level in profile.cytokine_pattern.items():
                boost = base_level * state.tlr_signal * (1.0 - treatment_eff * 0.3)
                state.cytokine_boost[cytokine] = boost
                total_cytokines[cytokine] = total_cytokines.get(cytokine, 0.0) + boost

            # ── 4. 神经毒素 ──
            state.neurotoxin_level = min(1.0,
                profile.neurotoxin_production * state.load * state.duration_steps * 0.001
            )

            # ── 5. 损害信号 ──
            damage = (
                0.3 * state.load +
                0.3 * state.tlr_signal +
                0.2 * state.neurotoxin_level +
                0.2 * state.bbb_disruption
            )
            total_damage += damage

            # ── 6. 状态delta ──
            # 认知损害
            if "cognitive_impact" in dir(profile):
                total_deltas["cognitive_impairment"] = total_deltas.get(
                    "cognitive_impairment", 0.0
                ) + profile.cognitive_impact * damage * 0.1

            # 情绪影响
            total_deltas["mood_impairment"] = total_deltas.get(
                "mood_impairment", 0.0
            ) + profile.mood_impact * damage * 0.1

            # 运动影响
            total_deltas["motor_impairment"] = total_deltas.get(
                "motor_impairment", 0.0
            ) + profile.motor_impact * damage * 0.1

            # 睡眠影响
            total_deltas["sleep_impairment"] = total_deltas.get(
                "sleep_impairment", 0.0
            ) + profile.sleep_impact * damage * 0.1

            # NT影响 (炎症直接作用于神经递质)
            total_deltas["nt_serotonin"] = total_deltas.get(
                "nt_serotonin", 0.0
            ) - 0.05 * state.tlr_signal  # 炎症↓5-HT (IDO通路)
            total_deltas["nt_dopamine"] = total_deltas.get(
                "nt_dopamine", 0.0
            ) - 0.03 * state.tlr_signal  # 炎症↓DA
            total_deltas["nt_glutamate"] = total_deltas.get(
                "nt_glutamate", 0.0
            ) + 0.04 * state.tlr_signal  # 炎症↑Glu (兴奋性毒性)

        return total_damage, total_cytokines, total_deltas

    def get_state_summary(self) -> Dict[str, Dict]:
        """获取所有病原体状态摘要。"""
        summary = {}
        for name, state in self.states.items():
            summary[name] = {
                "load": round(state.load, 3),
                "bbb_disruption": round(state.bbb_disruption, 3),
                "tlr_signal": round(state.tlr_signal, 3),
                "neurotoxin": round(state.neurotoxin_level, 3),
                "is_active": state.is_active,
                "duration_steps": state.duration_steps,
                "treatment_response": round(state.treatment_response, 3),
            }
        return summary


__all__ = [
    "PathogenProfile",
    "PATHOGEN_REGISTRY",
    "PathogenState",
    "PathogenTriggeredInflammationEngine",
]
