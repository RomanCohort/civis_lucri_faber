"""
HPA轴系统 (Hypothalamic-Pituitary-Adrenal Axis)

模拟完整的应激反应级联：
1. 下丘脑CRH释放 (Hypothalamic CRH)
2. 垂体ACTH释放 (Pituitary ACTH)
3. 肾上腺皮质醇释放 (Adrenal Cortisol)
4. 负反馈环路 (Negative Feedback Loop)
5. 稳态负荷追踪 (Allostatic Load Tracker)

生物参考文献:
- Vale et al. (1981): CRH作为主应激激素
- Sapolsky et al. (2000): 糖皮质激素与应激
- McEwen & Stellar (1993): 稳态负荷
- Jacobson & Sapolsky (1991): HPA负反馈
- Ulrich-Lai & Herman (2009): HPA轴恢复
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque


# ============ 状态定义 ============

@dataclass
class HPAState:
    """HPA轴状态"""
    crh_level: float = 0.2          # 促肾上腺皮质激素释放激素 [0,1]
    acth_level: float = 0.2         # 促肾上腺皮质激素 [0,1]
    cortisol_level: float = 0.3     # 皮质醇 [0,1]
    allostatic_load: float = 0.0    # 稳态负荷累积 [0,1]
    stress_type: str = "none"       # 应激类型: none/acute/chronic
    acute_stress_intensity: float = 0.0   # 连续急性应激强度 [0,1]
    chronic_stress_ratio: float = 0.0     # 连续慢性应激比率 [0,1]
    recovery_state: float = 0.8     # 恢复能力 [0,1]
    cortisol_history: deque = field(default_factory=lambda: deque(maxlen=200))


# ============ 下丘脑CRH ============

class HypothalamicCRH(nn.Module):
    """
    下丘脑促肾上腺皮质激素释放激素 (CRH)

    CRH是HPA轴的主控开关:
    - 应激信号刺激CRH释放
    - 不确定性增加CRH释放
    - 皮质醇负反馈抑制CRH释放

    CRH半衰期约5分钟 (快速起效，快速清除)

    参考: Vale et al. (1981)
    """

    def __init__(self, stress_reactivity: float = 1.0):
        super().__init__()
        self.stress_reactivity = nn.Parameter(torch.tensor(stress_reactivity))
        self.current_crh = 0.2

    def forward(self, stress_signal: float, cortisol_feedback: float,
                uncertainty: float, ne_level: float = 0.3,
                allostatic_load: float = 0.0) -> Dict[str, float]:
        """
        CRH释放计算

        增强: NE 驱动 + 应激敏化
        - 交感激活通过 NE 促进 CRH 释放 (蓝斑→下丘脑)
        - 慢性应激降低 CRH 反应阈值 (应激敏化)
        """
        # 应激敏化: 慢性应激增强 CRH 反应性
        effective_reactivity = self.stress_reactivity.item() * (1.0 + 0.3 * allostatic_load)

        # CRH释放 = 应激驱动 + NE增强 - 皮质醇抑制
        ne_drive = 0.2 * ne_level  # 交感→CRH 耦合
        drive = (effective_reactivity * stress_signal
                 + 0.3 * uncertainty + ne_drive)
        inhibition = cortisol_feedback * 0.8

        crh_release = float(torch.sigmoid(torch.tensor(drive - inhibition)))
        self.current_crh = float(np.clip(crh_release, 0.0, 1.0))

        return {
            'crh_level': self.current_crh,
            'drive': drive,
            'inhibition': inhibition,
        }


# ============ 垂体ACTH ============

class PituitaryACTH(nn.Module):
    """
    垂体促肾上腺皮质激素 (ACTH)

    CRH刺激ACTH释放:
    - CRH水平直接驱动ACTH释放
    - 皮质醇负反馈抑制ACTH释放
    - ACTH释放约延迟1分钟 (模型中直接传递)

    参考: Guillemin & Rosenberg (1955)
    """

    def __init__(self, crh_sensitivity: float = 1.0):
        super().__init__()
        self.crh_sensitivity = nn.Parameter(torch.tensor(crh_sensitivity))
        self.current_acth = 0.2

    def forward(self, crh_level: float, cortisol_feedback: float) -> Dict[str, float]:
        """
        ACTH释放计算

        acth = sigmoid(sensitivity * crh - cortisol_inhibition)
        """
        drive = self.crh_sensitivity.item() * crh_level
        inhibition = cortisol_feedback * 0.7  # 皮质醇对ACTH的抑制略弱于对CRH

        acth_release = float(torch.sigmoid(torch.tensor(drive - inhibition)))
        self.current_acth = float(np.clip(acth_release, 0.0, 1.0))

        return {
            'acth_level': self.current_acth,
        }


# ============ 肾上腺皮质 ============

class AdrenalCortex(nn.Module):
    """
    肾上腺皮质醇释放

    ACTH刺激皮质醇释放:
    - 皮质醇有约60分钟生物半衰期
    - 皮质醇受昼夜节律调节 (8AM峰值)
    - 社会支持可以缓冲皮质醇释放

    参考: Sapolsky et al. (2000)
    """

    def __init__(self, acth_sensitivity: float = 1.0, half_life_steps: int = 60):
        super().__init__()
        self.acth_sensitivity = nn.Parameter(torch.tensor(acth_sensitivity))
        self.half_life_steps = half_life_steps
        self.current_cortisol = 0.3

    def forward(self, acth_level: float, circadian_baseline: float = 0.3,
                social_buffer: float = 0.0) -> Dict[str, float]:
        """
        皮质醇释放计算

        cortisol_new = sigmoid(sensitivity * acth) + circadian_baseline
        皮质醇半衰期衰减: cortisol *= decay_factor^t
        """
        # ACTH驱动的皮质醇释放
        acth_drive = float(torch.sigmoid(
            torch.tensor(self.acth_sensitivity.item() * acth_level)
        ))

        # 昼夜节律基线 (已由外部计算)
        cortisol_target = acth_drive * 0.7 + circadian_baseline * 0.3

        # 社会支持缓冲效应 (降低皮质醇释放)
        cortisol_target *= (1.0 - 0.3 * social_buffer)

        # 皮质醇动态: 半衰期衰减 + 新释放
        decay_factor = 0.5 ** (1.0 / self.half_life_steps)
        self.current_cortisol = float(np.clip(
            self.current_cortisol * decay_factor + cortisol_target * 0.3,
            0.0, 1.0
        ))

        return {
            'cortisol_level': self.current_cortisol,
            'circadian_baseline': circadian_baseline,
        }

    def decay(self):
        """皮质醇自然衰减"""
        decay_factor = 0.5 ** (1.0 / self.half_life_steps)
        self.current_cortisol *= decay_factor
        self.current_cortisol = float(np.clip(self.current_cortisol, 0.05, 1.0))


# ============ 负反馈环路 ============

class NegativeFeedbackLoop(nn.Module):
    """
    HPA轴负反馈环路

    皮质醇通过负反馈抑制自身产生:
    - 皮质醇抑制下丘脑CRH释放
    - 皮质醇抑制垂体ACTH释放
    - 反馈强度随皮质醇水平增加

    这是HPA轴最关键的稳态机制

    参考: Jacobson & Sapolsky (1991)
    """

    def __init__(self, feedback_strength: float = 0.6,
                 genomic_delay: int = 5):
        super().__init__()
        self.feedback_strength = nn.Parameter(torch.tensor(feedback_strength))
        self.genomic_delay = genomic_delay  # 基因组效应延迟 (步数)
        self.cortisol_buffer = deque(maxlen=genomic_delay + 1)

    def forward(self, cortisol: float, gr_sensitivity: float = 1.0) -> Dict[str, float]:
        """
        计算负反馈信号

        增强: GR 受体敏感性调节反馈强度
        - GR 敏感性低时，同样皮质醇产生的反馈减弱
        - 模拟慢性应激导致的受体下调
        """
        self.cortisol_buffer.append(cortisol)

        # 非基因组效应: 快速, 弱 (膜受体, 即时)
        fast_inhibition = 0.2 * self.feedback_strength.item() * cortisol * gr_sensitivity

        # 基因组效应: 延迟, 强 (核受体, 转录调节)
        delayed_cortisol = (list(self.cortisol_buffer)[0]
                            if len(self.cortisol_buffer) > self.genomic_delay
                            else cortisol)
        slow_inhibition = 0.8 * self.feedback_strength.item() * delayed_cortisol * gr_sensitivity

        total_inhibition = float(np.clip(fast_inhibition + slow_inhibition, 0.0, 0.95))

        return {
            'cortisol_inhibition': total_inhibition,
            'fast_inhibition': fast_inhibition,
            'slow_inhibition': slow_inhibition,
        }


# ============ 稳态负荷追踪器 ============

class AllostaticLoadTracker:
    """
    稳态负荷追踪 (Allostatic Load)

    累积应激介质的慢性升高造成的"磨损":
    - 皮质醇慢性升高
    - 去甲肾上腺素慢性升高
    - 炎症标志物升高
    - 能量赤字

    参考: McEwen & Stellar (1993)
    """

    def __init__(self, accumulation_rate: float = 0.002,
                 recovery_rate: float = 0.001,
                 overload_threshold: float = 0.8):
        self.accumulation_rate = accumulation_rate
        self.recovery_rate = recovery_rate
        self.overload_threshold = overload_threshold
        self.load = 0.0
        self.stress_episodes = deque(maxlen=500)

    def update(self, cortisol: float, ne_level: float = 0.3,
               inflammation: float = 0.0, is_recovering: bool = False) -> float:
        """
        更新稳态负荷

        累积: 当介质高于中位点的量超过容差
        恢复: 在睡眠、低应激、社会支持期间
        """
        # 各介质的偏差贡献
        cortisol_deviation = max(0.0, cortisol - 0.4)  # 皮质醇中位点0.4
        ne_deviation = max(0.0, ne_level - 0.4)  # NE中位点0.4
        inflammation_deviation = max(0.0, inflammation - 0.3)

        # 累积负荷 (多介质加权)
        delta_load = self.accumulation_rate * (
            0.4 * cortisol_deviation +
            0.3 * ne_deviation +
            0.3 * inflammation_deviation
        )

        # 恢复
        if is_recovering:
            delta_load -= self.recovery_rate * 3  # 恢复期间加速减少

        self.load = float(np.clip(self.load + delta_load, 0.0, 1.0))
        self.stress_episodes.append({
            'cortisol': cortisol,
            'ne': ne_level,
            'load': self.load,
        })

        return self.load

    def get_load(self) -> float:
        return self.load

    def is_overloaded(self) -> bool:
        return self.load > self.overload_threshold


# ============ HPA轴 (聚合器) ============

class HPAAxis(nn.Module):
    """
    HPA轴 - 完整的应激反应级联系统

    下丘脑 -> 垂体 -> 肾上腺皮质:
    CRH -> ACTH -> Cortisol -> 负反馈

    功能:
    - 急性应激反应 (快速皮质醇释放与恢复)
    - 慢性应激检测 (持续高皮质醇)
    - 稳态负荷追踪 (累积损伤)
    - 昼夜节律交互 (皮质醇昼夜波动)

    参考:
    - Sapolsky et al. (2000): 应激与糖皮质激素
    - McEwen (1993): 稳态负荷理论
    """

    def __init__(self, stress_reactivity: float = 1.0,
                 cortisol_half_life_steps: int = 60,
                 feedback_strength: float = 0.6,
                 load_accumulation_rate: float = 0.002,
                 event_bus=None):
        super().__init__()

        self.crh = HypothalamicCRH(stress_reactivity=stress_reactivity)
        self.acth = PituitaryACTH()
        self.adrenal = AdrenalCortex(half_life_steps=cortisol_half_life_steps)
        self.feedback = NegativeFeedbackLoop(feedback_strength=feedback_strength)
        self.load_tracker = AllostaticLoadTracker(
            accumulation_rate=load_accumulation_rate
        )

        self.state = HPAState()
        self.step_count = 0
        self.chronic_stress_window = 100  # 慢性应激检测窗口
        self.event_bus = event_bus

        # Event-driven registration
        if self.event_bus is not None:
            self.event_bus.subscribe(
                "neural_regulation",
                self.on_neural_regulation,
                priority=1,
                name="hpa",
            )

    def step(self, stress_signal: float = 0.0, uncertainty: float = 0.0,
             circadian_hour: float = 12.0, is_recovering: bool = False,
             social_support: float = 0.5, ne_level: float = 0.3,
             inflammation: float = 0.0, hpa_suppressed: bool = False) -> Dict[str, Any]:
        """
        执行一个HPA轴调节步

        Args:
            stress_signal: 应激信号 [0,1]
            uncertainty: 不确定性 [0,1]
            circadian_hour: 昼夜小时 (0-24)
            is_recovering: 是否处于恢复状态
            social_support: 社会支持水平 [0,1]
            ne_level: 县肾上腺素水平 [0,1]
            inflammation: 炎症水平 [0,1]
            hpa_suppressed: 睡眠阶段HPA抑制标志 (Exp 8改进)
        """
        self.step_count += 1

        # 同步 adrenal.current_cortisol 与 state.cortisol_level (确保一致性)
        self.adrenal.current_cortisol = self.state.cortisol_level

        # ════════════════════════════════════════════════════════════════════
        # Exp 8改进: 睡眠阶段HPA抑制
        # 在睡眠阶段(NREM/REM)时，跳过应激驱动的皮质醇更新
        # 只保留昼夜节律基线和自然衰减，使创伤回放可以独立调节皮质醇
        # ════════════════════════════════════════════════════════════════════
        if hpa_suppressed:
            # 睡眠期: 只保留皮质醇自然衰减和昼夜基线
            current_cortisol = self.state.cortisol_level
            circadian_baseline = 0.15 * np.cos(2 * np.pi * (circadian_hour - 8.0) / 24.0) + 0.35

            # 自然衰减 (调用 adrenal.decay()，它会修改内部 cortisol)
            self.adrenal.current_cortisol = current_cortisol  # 同步状态
            self.adrenal.decay()
            decayed_cortisol = self.adrenal.current_cortisol

            # 向昼夜基线温和回归
            cortisol_target = 0.3 + circadian_baseline * 0.3  # 睡眠期基线更低
            cortisol_new = decayed_cortisol * 0.95 + cortisol_target * 0.05
            self.state.cortisol_level = float(np.clip(cortisol_new, 0.1, 0.7))
            self.adrenal.current_cortisol = self.state.cortisol_level  # 同步回来

            # 睡眠期不更新稳态负荷
            return {
                'cortisol_level': self.state.cortisol_level,
                'hpa_suppressed': True,
                'stress_type': 'sleep',
                'acute_intensity': 0.0,
                'chronic_ratio': float(self.state.cortisol_history[-1]) if self.state.cortisol_history else 0.0,
                'allostatic_load': self.load_tracker.get_load(),
            }

        # 正常清醒期: 完整HPA级联
        # 1. 计算昼夜皮质醇基线 (8AM峰值)
        # 参考: 正常皮质醇昼夜节律
        circadian_baseline = 0.15 * np.cos(2 * np.pi * (circadian_hour - 8.0) / 24.0) + 0.35

        # 2. 负反馈信号 (含 GR 敏感性)
        gr_sensitivity = 1.0  # 默认, 由 HormoneSystem 更新
        feedback_result = self.feedback(self.state.cortisol_level, gr_sensitivity)
        cortisol_inhibition = feedback_result['cortisol_inhibition']

        # 3. CRH释放 (下丘脑) — 含 NE 驱动 + 应激敏化
        crh_result = self.crh(
            stress_signal, cortisol_inhibition, uncertainty,
            ne_level=ne_level, allostatic_load=self.load_tracker.get_load(),
        )

        # 4. ACTH释放 (垂体)
        acth_result = self.acth(crh_result['crh_level'], cortisol_inhibition)

        # 5. 皮质醇释放 (肾上腺皮质)
        social_buffer = max(0.0, social_support - 0.3)
        adrenal_result = self.adrenal(
            acth_result['acth_level'],
            circadian_baseline=circadian_baseline,
            social_buffer=social_buffer,
        )

        # 注意: AdrenalCortex.forward() 内部已应用衰减, 不再重复调 decay()

        # 7. 稳态负荷更新
        allostatic_load = self.load_tracker.update(
            cortisol=adrenal_result['cortisol_level'],
            ne_level=ne_level,
            inflammation=inflammation,
            is_recovering=is_recovering,
        )

        # 8. 慢性应激检测
        cortisol_history = list(self.state.cortisol_history)
        cortisol_history.append(adrenal_result['cortisol_level'])
        self.state.cortisol_history = deque(cortisol_history[-200:], maxlen=200)

        # 连续应激指标 (sigmoid平滑替代硬阈值)
        cortisol = adrenal_result['cortisol_level']
        acute_intensity = float(torch.sigmoid(torch.tensor((cortisol - 0.4) * 10)))

        if len(cortisol_history) >= self.chronic_stress_window:
            recent_mean = np.mean(cortisol_history[-self.chronic_stress_window:])
            chronic_ratio = float(torch.sigmoid(torch.tensor((recent_mean - 0.45) * 12)))
        else:
            chronic_ratio = 0.0

        # 从连续值派生离散类型名 (向后兼容)
        if chronic_ratio > 0.5:
            stress_type = "chronic"
        elif acute_intensity > 0.5:
            stress_type = "acute"
        else:
            stress_type = "none"

        # 9. 恢复能力计算
        recovery_state = float(np.clip(
            0.8 - 0.3 * allostatic_load
            + 0.2 * (1.0 if is_recovering else 0.0)
            + 0.1 * social_support,
            0.1, 1.0
        ))

        # 10. 更新状态
        self.state = HPAState(
            crh_level=crh_result['crh_level'],
            acth_level=acth_result['acth_level'],
            cortisol_level=adrenal_result['cortisol_level'],
            allostatic_load=allostatic_load,
            stress_type=stress_type,
            acute_stress_intensity=acute_intensity,
            chronic_stress_ratio=chronic_ratio,
            recovery_state=recovery_state,
            cortisol_history=self.state.cortisol_history,
        )

        return {
            'crh_level': self.state.crh_level,
            'acth_level': self.state.acth_level,
            'cortisol_level': self.state.cortisol_level,
            'allostatic_load': self.state.allostatic_load,
            'stress_type': self.state.stress_type,
            'acute_stress_intensity': acute_intensity,
            'chronic_stress_ratio': chronic_ratio,
            'recovery_state': self.state.recovery_state,
            'circadian_baseline': circadian_baseline,
            'cortisol_inhibition': cortisol_inhibition,
            'is_overloaded': self.load_tracker.is_overloaded(),
        }

    def on_neural_regulation(self, event) -> Dict[str, Any]:
        """Event handler for NEURAL_REGULATION events (priority=1, depends on ANS)."""
        state = event.data["internal_state"]
        stress_signal = state.get("ans_sympathetic", 0.3)
        uncertainty = 1.0 - state.get("alignment_score", 0.5)
        circadian_hour = state.get("scn_circadian_hour", 12.0)
        thermo_status = event.data.get("thermo_status", "ACTIVE")
        is_recovering = thermo_status == "HIBERNATE"
        ne_level = state.get("nt_norepinephrine", 0.3)  # NE 驱动 CRH

        # Exp 8改进: 从 internal_state 读取睡眠阶段 HPA 抑制标志
        hpa_suppressed = state.get("hpa_suppressed", False)

        result = self.step(
            stress_signal=stress_signal,
            uncertainty=uncertainty,
            circadian_hour=circadian_hour,
            is_recovering=is_recovering,
            social_support=0.5,
            ne_level=ne_level,
            hpa_suppressed=hpa_suppressed,
        )
        state["cortisol"] = result["cortisol_level"]
        state["cortisol_level"] = result["cortisol_level"]  # 修复 key 不匹配
        state["hpa_crh"] = result["crh_level"]
        state["hpa_acth"] = result["acth_level"]
        state["stress_type"] = result["stress_type"]
        state["allostatic_load"] = result["allostatic_load"]
        return result

    def trigger_acute_stress(self, intensity: float):
        """触发急性应激 (外部刺激)"""
        self.crh.current_crh = float(np.clip(
            self.crh.current_crh + intensity * 0.5, 0.0, 1.0
        ))

    def get_summary(self) -> Dict:
        """获取HPA轴摘要"""
        return {
            'crh': self.state.crh_level,
            'acth': self.state.acth_level,
            'cortisol': self.state.cortisol_level,
            'allostatic_load': self.state.allostatic_load,
            'stress_type': self.state.stress_type,
            'recovery_state': self.state.recovery_state,
            'is_overloaded': self.load_tracker.is_overloaded(),
            'step_count': self.step_count,
        }


def create_hpa_axis(**kwargs) -> HPAAxis:
    """工厂函数: 创建HPA轴"""
    return HPAAxis(**kwargs)


__all__ = [
    'HPAState',
    'HypothalamicCRH',
    'PituitaryACTH',
    'AdrenalCortex',
    'NegativeFeedbackLoop',
    'AllostaticLoadTracker',
    'HPAAxis',
    'create_hpa_axis',
]
