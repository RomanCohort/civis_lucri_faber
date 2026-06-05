"""
脑干系统 (Brainstem System)

脑干是大脑最原始的部分，连接脊髓与间脑。包含三大区域：

1. 延髓 (Medulla Oblongata) — 最尾部
   - 呼吸节律发生器（pre-Bötzinger复合体）
   - 心血管中枢（心抑制中枢/血管运动中枢）
   - 化学感受器反射（颈动脉体/主动脉体 → 延髓）
   - 孤束核（NTS）：内脏感觉中继
   - 呕吐/吞咽/咳嗽反射

2. 脑桥 (Pons) — 中部
   - 脑桥呼吸组（PRG）：调节吸/呼比
   - 蓝斑核（Locus Coeruleus）：NE释放，调控唤醒/注意
   - PPT/LDT核：胆碱能神经元，启动REM睡眠
   - 前庭核：平衡/空间定向

3. 中脑 (Midbrain / Mesencephalon) — 最吻侧
   - 网状激活系统（RAS/ARAS）：维持清醒/意识门控
   - 导水管周围灰质（PAG）：痛觉调制，防御行为
   - 上丘/下丘：朝向反射（视听定向）
   - 红核：运动中继（rubrospinal tract）
   - 黑质（Substantia Nigra）：多巴胺，已由basal_ganglia.py覆盖

核心通路：
网状激活系统（RAS）：
  脑干网状结构 → 丘脑板内核 → 广泛皮层投射 → 维持觉醒/意识

呼吸节律：
  pre-Bötzinger → 吸气节律 → 脑桥PRG调制 → 膈神经/肋间神经

痛觉调制：
  PAG → RVM（延髓头端腹内侧） → 脊髓背角 → 痛觉门控

核心类：
1. RespiratoryRhythmGenerator — 呼吸节律
2. ReticularActivatingSystem — 网状激活系统
3. PeriaqueductalGray — 导水管周围灰质
4. Brainstem — 完整脑干
"""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Set

import numpy as np
import torch
import torch.nn as nn

from core.abstract_brain_region import AbstractBrainRegion

# ============ 枚举和数据类 ============

class RespiratoryPhase(Enum):
    """呼吸周期相位"""
    INSPIRATION = "inspiration"    # 吸气
    POST_INSPIRATION = "post_inspiration"  # 吸气后暂停
    EXPIRATION = "expiration"      # 呼气


class ArousalLevel(Enum):
    """觉醒水平"""
    COMA = 0         # 昏迷
    DEEP_SLEEP = 1   # 深睡
    LIGHT_SLEEP = 2  # 浅睡
    DROWSY = 3       # 困倦
    RELAXED = 4      # 放松清醒
    ALERT = 5        # 警觉
    HYPER_VIGILANT = 6  # 高度警戒


class DefensiveBehavior(Enum):
    """PAG防御行为层级"""
    FREEZE = "freeze"        # 冻结（远距离威胁）
    FLIGHT = "flight"        # 逃跑（中等距离）
    FIGHT = "fight"          # 战斗（近距离）
    QUIESCENCE = "quiescence"  # 静止（不可逃避）


@dataclass
class BrainstemState:
    """脑干整体状态"""
    # 呼吸
    respiratory_rate: float = 12.0    # 呼吸频率（次/分钟）
    tidal_volume: float = 0.5         # 潮气量（归一化）
    co2_level: float = 0.4            # CO2水平 [0,1]
    o2_level: float = 0.95            # O2水平 [0,1]
    respiratory_phase: str = "expiration"

    # 心血管（由延髓心血管中枢输出）
    cardiac_output: float = 0.5       # 心输出量 [0,1]
    vascular_resistance: float = 0.5  # 外周血管阻力 [0,1]

    # 觉醒
    arousal_level: float = 0.5        # 觉醒水平 [0,1]
    consciousness_gate: float = 0.5   # 意识门控 [0,1]

    # 痛觉调制
    pain_gating: float = 0.5          # 痛觉门控 [0,1]（越高越抑制痛觉）
    endorphin_release: float = 0.2    # 内啡肽释放 [0,1]


# ============ 延髓：呼吸节律发生器 ============

class RespiratoryRhythmGenerator(nn.Module):
    """
    呼吸节律发生器

    对应延髓pre-Bötzinger复合体：
    - 内源性起搏神经元产生吸气节律（约12-20次/分钟）
    - 脑桥呼吸组（PRG）调制吸气/呼气比
    - 化学感受器反射：CO2↑或O2↓ → 呼吸频率↑
    - 肺牵张反射（Hering-Breuer）：过度膨胀 → 抑制吸气

    简化模型：极限环振荡器
    """

    def __init__(
        self,
        baseline_rate: float = 12.0,     # 基线呼吸频率（次/分钟）
        min_rate: float = 4.0,
        max_rate: float = 40.0,
        chemosensitivity: float = 2.0,   # 化学感受器灵敏度
    ):
        super().__init__()
        self.baseline_rate = baseline_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.chemosensitivity = chemosensitivity

        # 振荡器相位
        self._phase = 0.0
        self._inspiratory_neuron_activity = 0.0

        # 状态
        self.current_rate = baseline_rate
        self.current_phase = RespiratoryPhase.EXPIRATION

    def forward(
        self,
        co2_level: float = 0.4,
        o2_level: float = 0.95,
        metabolic_demand: float = 0.5,
    ) -> dict:
        """
        推进呼吸节律一步

        Args:
            co2_level: 血液CO2水平 [0,1]（正常~0.4）
            o2_level: 血液O2水平 [0,1]（正常~0.95）
            metabolic_demand: 代谢需求 [0,1]

        Returns:
            呼吸状态
        """
        # 化学感受器驱动（延髓中枢化学感受器 + 颈动脉体外周化学感受器）
        # CO2升高 → 强驱动增加呼吸频率
        co2_drive = max(0, (co2_level - 0.35)) * self.chemosensitivity
        # O2下降 → 中等驱动
        o2_drive = max(0, (0.9 - o2_level)) * self.chemosensitivity * 0.5
        # 代谢需求
        metabolic_drive = metabolic_demand * 0.3

        total_drive = co2_drive + o2_drive + metabolic_drive

        # 呼吸频率调节
        target_rate = self.baseline_rate + total_drive * (self.max_rate - self.baseline_rate)
        self.current_rate = np.clip(
            0.9 * self.current_rate + 0.1 * target_rate,
            self.min_rate, self.max_rate
        )

        # 相位推进（一个呼吸周期 = 60/rate 秒，每步1分钟=60秒）
        cycle_duration = 60.0 / self.current_rate  # 一个周期的秒数
        phase_rate = 60.0 / cycle_duration  # 每分钟推进的周期数
        self._phase += phase_rate  # 每步推进

        # 相位 → 呼吸阶段
        phase_fraction = self._phase % 1.0
        if phase_fraction < 0.4:
            self.current_phase = RespiratoryPhase.INSPIRATION
            self._inspiratory_neuron_activity = np.sin(phase_fraction / 0.4 * np.pi)
        elif phase_fraction < 0.5:
            self.current_phase = RespiratoryPhase.POST_INSPIRATION
            self._inspiratory_neuron_activity = max(0, np.cos((phase_fraction - 0.4) / 0.1 * np.pi / 2))
        else:
            self.current_phase = RespiratoryPhase.EXPIRATION
            self._inspiratory_neuron_activity = 0.0

        # 潮气量与频率成反比（高频率时浅呼吸）
        tidal_volume = 0.5 + 0.3 * (1.0 - (self.current_rate - self.min_rate) / (self.max_rate - self.min_rate))

        return {
            'rate': self.current_rate,
            'phase': self.current_phase.value,
            'tidal_volume': tidal_volume,
            'inspiratory_activity': self._inspiratory_neuron_activity,
            'co2_drive': co2_drive,
            'o2_drive': o2_drive,
        }


# ============ 网状激活系统 ============

class ReticularActivatingSystem(nn.Module):
    """
    网状激活系统 (Reticular Activating System, RAS / ARAS)

    对应脑干网状结构的上行激动系统：
    - 网状结构神经元 → 丘脑板内核（intralaminar nuclei）
    - 丘脑 → 广泛皮层投射 → 维持觉醒和意识

    输入：
    - 感觉输入（痛觉、听觉、触觉等多模态）
    - SCN昼夜节律信号
    - 神经递质调制（NE from蓝斑、ACh from PPT/LDT、5-HT from缝际核）

    输出：
    - 觉醒水平 (arousal_level)
    - 意识门控 (consciousness_gate)
    - 皮层激活度 (cortical_activation)
    """

    def __init__(
        self,
        baseline_arousal: float = 0.5,
        sensory_gain: float = 0.3,
        decay_rate: float = 0.02,
    ):
        super().__init__()
        self.baseline_arousal = baseline_arousal
        self.sensory_gain = sensory_gain
        self.decay_rate = decay_rate

        # 觉醒水平
        self.arousal_level = baseline_arousal
        self.consciousness_gate = 0.5

        # 感觉输入整合网络
        self.sensory_integrator = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # 丘脑投射门控
        self.thalamic_gate = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sensory_input: torch.Tensor = None,
        scn_wake_drive: float = 0.5,
        ne_level: float = 0.3,        # 去甲肾上腺素（蓝斑核）
        ach_level: float = 0.3,       # 乙酰胆碱（PPT/LDT）
        pain_signal: float = 0.0,     # 痛觉信号
        novelty: float = 0.0,         # 新奇信号
    ) -> dict:
        """
        RAS步进

        Args:
            sensory_input: 多模态感觉输入 [B, 5]（视觉/听觉/触觉/前庭/内脏）
            scn_wake_drive: SCN觉醒驱动 [0,1]
            ne_level: 去甲肾上腺素水平 [0,1]
            ach_level: 乙酰胆碱水平 [0,1]
            pain_signal: 痛觉信号 [0,1]
            novelty: 新奇信号 [0,1]

        Returns:
            觉醒状态
        """
        # 1. 感觉输入整合（痛觉和听觉对RAS最强）
        if sensory_input is not None:
            sensory_drive = self.sensory_integrator(sensory_input).item()
        else:
            # 用标量输入近似
            sensory_drive = (
                pain_signal * 0.4 +
                novelty * 0.3 +
                scn_wake_drive * 0.3
            )

        # 2. 神经递质调制
        # NE（蓝斑核）→ 增强警觉
        # ACh（PPT/LDT）→ 维持皮层激活（尤其是REM）
        nt_modulation = ne_level * 0.4 + ach_level * 0.3 + 0.3

        # 3. 丘脑门控计算
        gate_input = torch.tensor([[
            self.arousal_level,
            scn_wake_drive,
            ne_level,
        ]], dtype=torch.float32)
        self.consciousness_gate = self.thalamic_gate(gate_input).item()

        # 4. 觉醒水平更新
        # 感觉驱动提升觉醒，自然衰减降低
        target_arousal = (
            self.baseline_arousal * 0.3 +
            sensory_drive * 0.3 +
            scn_wake_drive * 0.2 +
            nt_modulation * 0.2
        )
        self.arousal_level = (
            0.85 * self.arousal_level +
            0.15 * target_arousal
        )
        # 自然衰减（无刺激时逐渐降低）
        self.arousal_level -= self.decay_rate
        self.arousal_level = np.clip(self.arousal_level, 0.05, 1.0)

        # 5. 皮层激活度
        cortical_activation = self.arousal_level * self.consciousness_gate

        # 6. 确定觉醒级别 (加权混合，而非硬阈值)
        # 各级别的激活度基于与arousal_level的距离 (高斯核)
        arousal_centers = {
            'COMA': 0.05, 'DEEP_SLEEP': 0.15, 'LIGHT_SLEEP': 0.25,
            'DROWSY': 0.35, 'RELAXED': 0.5, 'ALERT': 0.7, 'HYPER_VIGILANT': 0.9,
        }
        sigmas = {name: 0.08 for name in arousal_centers}  # 各级别宽度
        activations = {}
        for name, center in arousal_centers.items():
            activations[name] = np.exp(-0.5 * ((self.arousal_level - center) / sigmas[name]) ** 2)
        best_name = max(activations, key=activations.get)
        level = ArousalLevel[best_name]

        return {
            'arousal_level': self.arousal_level,
            'consciousness_gate': self.consciousness_gate,
            'cortical_activation': cortical_activation,
            'arousal_name': level.name,
            'sensory_drive': sensory_drive,
            'nt_modulation': nt_modulation,
        }


# ============ 导水管周围灰质 ============

class PeriaqueductalGray(nn.Module):
    """
    导水管周围灰质 (Periaqueductal Gray, PAG)

    中脑围绕导水管的一圈灰质，是痛觉调制和防御行为的核心。

    功能：
    1. 痛觉调制（下行抑制通路）：
       PAG → RVM（延髓头端腹内侧） → 脊髓背角 → 抑制痛觉传入
       应激/恐惧时激活 → 内啡肽释放 → 痛觉抑制（战场上受伤不觉得痛）

    2. 防御行为层级（从背侧到腹侧）：
       - 背侧PAG：冻结（远距离威胁检测）
       - 外侧PAG：逃跑（中等距离）
       - 腹外侧PAG：战斗（近距离）
       - 腹侧PAG：静止/无助（不可逃避）

    参考：
    - Keay & Bandler (2001): PAG防御行为柱状模型
    - Fields & Basbaum (1999): 下行痛觉调制
    """

    def __init__(self, pain_threshold: float = 0.5):
        super().__init__()
        self.pain_threshold = pain_threshold

        # 防御行为选择网络
        self.defense_net = nn.Sequential(
            nn.Linear(3, 16),   # threat + distance + escape_route
            nn.ReLU(),
            nn.Linear(16, 4),   # freeze/flight/fight/quiescence
            nn.Softmax(dim=-1),
        )

        # 痛觉调制网络
        self.pain_modulation = nn.Sequential(
            nn.Linear(2, 8),    # stress + endorphin
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self.endorphin_level = 0.2
        self.current_defense = DefensiveBehavior.FREEZE

    def forward(
        self,
        pain_input: float = 0.0,
        threat_level: float = 0.0,
        threat_distance: float = 1.0,  # 0=极近, 1=远
        escape_available: float = 1.0,  # 0=无退路, 1=可逃
        stress_level: float = 0.3,
    ) -> dict:
        """
        PAG步进

        Args:
            pain_input: 痛觉输入 [0,1]
            threat_level: 威胁水平 [0,1]
            threat_distance: 威胁距离 [0,1]
            escape_available: 逃跑可用性 [0,1]
            stress_level: 应激水平 [0,1]
        """
        # 1. 防御行为选择
        defense_input = torch.tensor([[threat_level, threat_distance, escape_available]], dtype=torch.float32)
        defense_probs = self.defense_net(defense_input)

        # 确定性选择
        defense_idx = defense_probs.argmax(dim=-1).item()
        defense_behaviors = [
            DefensiveBehavior.FREEZE,
            DefensiveBehavior.FLIGHT,
            DefensiveBehavior.FIGHT,
            DefensiveBehavior.QUIESCENCE,
        ]
        self.current_defense = defense_behaviors[defense_idx]

        # 2. 应激/恐惧驱动的内啡肽释放 (连续调制, 无硬阈值)
        # 对应生物学：PAG → 下行抑制 → 内啡肽/脑啡肽释放
        # 威胁和痛觉连续驱动内啡肽释放，释放速率与刺激强度成正比
        threat_drive = float(np.clip(threat_level, 0.0, 1.0))
        pain_drive = float(np.clip(pain_input, 0.0, 1.0))
        endorphin_release_rate = 0.05 * (threat_drive + pain_drive)  # 连续释放
        # 自然衰减 (内啡肽半衰期约~5步)
        self.endorphin_level = float(np.clip(
            self.endorphin_level * 0.92 + endorphin_release_rate,
            0.05, 1.0
        ))

        # 3. 痛觉门控计算
        mod_input = torch.tensor([[stress_level, self.endorphin_level]], dtype=torch.float32)
        pain_gate = self.pain_modulation(mod_input).item()

        # 4. 调制后的痛觉输出
        modulated_pain = pain_input * (1.0 - pain_gate)

        return {
            'defense_behavior': self.current_defense.value,
            'defense_probs': defense_probs.detach().cpu().numpy()[0].tolist(),
            'pain_gating': pain_gate,
            'modulated_pain': modulated_pain,
            'endorphin_level': self.endorphin_level,
            'original_pain': pain_input,
        }


# ============ 延髓心血管中枢 ============

class MedullaryCardiovascularCenter(nn.Module):
    """
    延髓心血管中枢

    包含：
    - 心迷走中枢（迷走神经背核）：副交感 → 减慢心率
    - 血管运动中枢（延髓头端腹外侧）：交感 → 收缩血管
    - 孤束核（NTS）：接收压力感受器输入，负反馈调节

    参考：Guyton & Hall (2006) 医学生理学
    """

    def __init__(
        self,
        baseline_hr: float = 72.0,     # 基线心率 bpm
        baseline_bp: float = 120.0,     # 基线收缩压 mmHg
    ):
        super().__init__()
        self.baseline_hr = baseline_hr
        self.baseline_bp = baseline_bp

        self.current_hr = baseline_hr
        self.current_bp = baseline_bp

    def forward(
        self,
        sympathetic_tone: float = 0.3,
        parasympathetic_tone: float = 0.5,
        respiratory_phase: str = "expiration",
        metabolic_demand: float = 0.5,
        blood_volume_status: float = 0.5,  # 0=失血, 1=正常
    ) -> dict:
        """
        心血管中枢步进

        呼吸性窦性心律不齐（RSA）：
        吸气时迷走张力降低 → 心率↑
        呼气时迷走张力恢复 → 心率↓
        """
        # 迷走神经对心率的影响
        vagal_effect = -parasympathetic_tone * 30  # 最高减30bpm

        # 交感神经对心率的影响
        sympathetic_effect = sympathetic_tone * 50  # 最高加50bpm

        # 呼吸性窦性心律不齐 (RSA) — 连续正弦波形，非二值
        # 在吸气相位时迷走张力降低 → 心率上升，呼气相位恢复 → 心率下降
        # RSA幅度随迷走张力强度变化 (副交感强时RSA更明显)
        rsa_amplitude = 5.0 * parasympathetic_tone  # 幅度与迷走张力成正比
        # respiratory_phase是字符串，用phase_fraction的正弦近似
        # inspiration → 正半周 (心率↑)，expiration → 负半周 (心率↓)
        # 简化: 用phase字符串的sign
        rsa_sign = 1.0 if respiratory_phase == "inspiration" else -1.0
        rsa_effect = rsa_sign * rsa_amplitude

        # 代谢需求
        metabolic_effect = metabolic_demand * 20

        # 目标心率
        target_hr = self.baseline_hr + vagal_effect + sympathetic_effect + rsa_effect + metabolic_effect
        target_hr = max(40, min(200, target_hr))
        self.current_hr = 0.9 * self.current_hr + 0.1 * target_hr

        # 血压调节（血管运动中枢）
        vascular_tone = 0.5 + sympathetic_tone * 0.4 - parasympathetic_tone * 0.1
        target_bp = self.baseline_bp * vascular_tone * (0.5 + blood_volume_status * 0.5)
        target_bp = max(60, min(200, target_bp))
        self.current_bp = 0.9 * self.current_bp + 0.1 * target_bp

        # 心输出量（HR × 每搏输出量）
        stroke_volume = 0.5 + sympathetic_tone * 0.3
        cardiac_output = (self.current_hr / self.baseline_hr) * stroke_volume

        return {
            'heart_rate': self.current_hr,
            'blood_pressure': self.current_bp,
            'cardiac_output': np.clip(cardiac_output, 0.2, 2.0),
            'vascular_resistance': np.clip(vascular_tone, 0.2, 1.0),
        }


# ============ 完整脑干 ============

class Brainstem(AbstractBrainRegion):
    """
    完整脑干系统

    整合三大区域：
    - 延髓：呼吸节律 + 心血管中枢
    - 脑桥/中脑：RAS觉醒系统 + PAG痛觉调制
    """

    region_name: ClassVar[str] = "brainstem"

    @classmethod
    def required_keys(cls) -> Set[str]:
        """Keys this region reads from the shared state."""
        return set(["pain_input", "threat_level", "novelty",
                    "sensory_input", "threat_distance", "escape_available"])

    @classmethod
    def output_keys(cls) -> Set[str]:
        """Keys this region writes to the shared state."""
        return set(["bsm_arousal", "bsm_consciousness_gate", "bsm_cortical_activation",
                    "bsm_respiratory_rate", "bsm_respiratory_phase", "bsm_heart_rate",
                    "bsm_blood_pressure", "bsm_defense_behavior", "bsm_pain_gating"])

    def __init__(
        self,
        baseline_hr: float = 72.0,
        baseline_resp_rate: float = 12.0,
        baseline_arousal: float = 0.5,
        event_bus=None,
    ):
        super().__init__()

        # 子系统
        self.respiration = RespiratoryRhythmGenerator(
            baseline_rate=baseline_resp_rate,
        )
        self.cardiovascular = MedullaryCardiovascularCenter(
            baseline_hr=baseline_hr,
        )
        self.ras = ReticularActivatingSystem(
            baseline_arousal=baseline_arousal,
        )
        self.pag = PeriaqueductalGray()

        # 整体状态
        self.state = BrainstemState()

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "brain_update",
                self._handle_brain_update,
                priority=3,
                name="brainstem",
            )

    def _handle_brain_update(self, event) -> dict:
        """Event-driven handler for brain_update events."""
        state = event.data.get("internal_state", {})

        result = self.forward(
            pain_input=state.get("pain_input", 0.0),
            threat_level=state.get("threat", 0.0),
            novelty=state.get("novelty", 0.0),
        )

        state["bsm_arousal"] = result["arousal_level"]
        state["bsm_consciousness_gate"] = result["consciousness_gate"]
        state["bsm_cortical_activation"] = result["cortical_activation"]
        state["bsm_respiratory_rate"] = result["respiratory_rate"]
        state["bsm_respiratory_phase"] = result["respiratory_phase"]
        state["bsm_heart_rate"] = result["heart_rate"]
        state["bsm_blood_pressure"] = result["blood_pressure"]
        state["bsm_defense_behavior"] = result["defense_behavior"]
        state["bsm_pain_gating"] = result["pain_gating"]
        state["bsm_arousal_name"] = result["arousal_name"]

        return result

    def forward(
        self,
        # 感觉/环境输入
        sensory_input: torch.Tensor = None,
        pain_input: float = 0.0,
        threat_level: float = 0.0,
        threat_distance: float = 1.0,
        escape_available: float = 1.0,
        novelty: float = 0.0,

        # 系统间信号
        sympathetic_tone: float = 0.3,
        parasympathetic_tone: float = 0.5,
        scn_wake_drive: float = 0.5,
        ne_level: float = 0.3,
        ach_level: float = 0.3,
        metabolic_demand: float = 0.5,

        # 内环境
        co2_level: float = 0.4,
        o2_level: float = 0.95,
        stress_level: float = 0.3,
        blood_volume: float = 0.5,
    ) -> dict:
        """
        脑干完整步进

        Returns:
            所有脑干输出
        """
        # 1. 延髓：呼吸节律
        resp_result = self.respiration(
            co2_level=co2_level,
            o2_level=o2_level,
            metabolic_demand=metabolic_demand,
        )

        # 2. 延髓：心血管中枢
        cardio_result = self.cardiovascular(
            sympathetic_tone=sympathetic_tone,
            parasympathetic_tone=parasympathetic_tone,
            respiratory_phase=resp_result['phase'],
            metabolic_demand=metabolic_demand,
            blood_volume_status=blood_volume,
        )

        # 3. 中脑/脑桥：RAS觉醒系统
        ras_result = self.ras(
            sensory_input=sensory_input,
            scn_wake_drive=scn_wake_drive,
            ne_level=ne_level,
            ach_level=ach_level,
            pain_signal=pain_input,
            novelty=novelty,
        )

        # 4. 中脑：PAG痛觉调制与防御
        pag_result = self.pag(
            pain_input=pain_input,
            threat_level=threat_level,
            threat_distance=threat_distance,
            escape_available=escape_available,
            stress_level=stress_level,
        )

        # 更新整体状态
        self.state.respiratory_rate = resp_result['rate']
        self.state.tidal_volume = resp_result['tidal_volume']
        self.state.respiratory_phase = resp_result['phase']
        self.state.co2_level = co2_level
        self.state.o2_level = o2_level
        self.state.cardiac_output = cardio_result['cardiac_output']
        self.state.vascular_resistance = cardio_result['vascular_resistance']
        self.state.arousal_level = ras_result['arousal_level']
        self.state.consciousness_gate = ras_result['consciousness_gate']
        self.state.pain_gating = pag_result['pain_gating']
        self.state.endorphin_release = pag_result['endorphin_level']

        return {
            # 呼吸
            'respiratory_rate': resp_result['rate'],
            'respiratory_phase': resp_result['phase'],
            'tidal_volume': resp_result['tidal_volume'],
            # 心血管
            'heart_rate': cardio_result['heart_rate'],
            'blood_pressure': cardio_result['blood_pressure'],
            'cardiac_output': cardio_result['cardiac_output'],
            # 觉醒
            'arousal_level': ras_result['arousal_level'],
            'arousal_name': ras_result['arousal_name'],
            'consciousness_gate': ras_result['consciousness_gate'],
            'cortical_activation': ras_result['cortical_activation'],
            # 痛觉/防御
            'pain_gating': pag_result['pain_gating'],
            'modulated_pain': pag_result['modulated_pain'],
            'defense_behavior': pag_result['defense_behavior'],
            'endorphin_level': pag_result['endorphin_level'],
        }

    def get_summary(self) -> dict:
        """获取脑干摘要"""
        return {
            'respiratory_rate': round(self.state.respiratory_rate, 1),
            'respiratory_phase': self.state.respiratory_phase,
            'heart_rate': round(self.cardiovascular.current_hr, 1),
            'blood_pressure': round(self.cardiovascular.current_bp, 1),
            'arousal_level': round(self.state.arousal_level, 3),
            'consciousness_gate': round(self.state.consciousness_gate, 3),
            'pain_gating': round(self.state.pain_gating, 3),
            'endorphin': round(self.state.endorphin_release, 3),
        }


def create_brainstem(**kwargs) -> Brainstem:
    """创建脑干系统"""
    return Brainstem(**kwargs)


__all__ = [
    'RespiratoryPhase',
    'ArousalLevel',
    'DefensiveBehavior',
    'BrainstemState',
    'RespiratoryRhythmGenerator',
    'ReticularActivatingSystem',
    'PeriaqueductalGray',
    'MedullaryCardiovascularCenter',
    'Brainstem',
    'create_brainstem',
]
