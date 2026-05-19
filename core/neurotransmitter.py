"""
神经递质系统 (Neurotransmitter System)

模拟生物神经递质：
1. 多巴胺 (Dopamine) - 动机、奖励、运动
2. 血清素 (Serotonin) - 情绪、睡眠、饱腹感
3. 乙酰胆碱 (Acetylcholine) - 注意力、学习、记忆
4. 谷氨酸 (Glutamate) - 兴奋性、学习
5. GABA - 抑制性、镇静
6. 去甲肾上腺素 (Norepinephrine) - 警觉、应激
7. 内啡肽 (Endorphin) - 镇痛、愉悦

功能：
- 递质动态平衡
- 受体调节
- 递质-行为映射
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from collections import deque


# ============ 神经递质定义 ============

@dataclass
class Neurotransmitter:
    """神经递质"""
    name: str
    level: float          # 当前水平 0-1
    baseline: float      # 基线水平
    release_rate: float  # 释放率
    reuptake_rate: float # 重摄取率
    degradation_rate: float # 降解率
    target_level: float # 目标水平


@dataclass
class Receptor:
    """受体

    新增: 完整受体动力学
    """
    name: str
    neurotransmitter: str  # 对应递质
    affinity: float   # 亲和力 Kd (M)
    density: float   # 密度 (mol/mol)
    state: str       # "active" | "desensitized" | "downregulated" | "upregulated"
    occupancy: float = 0.0  # 受体占有率 (新增)
    desensitization_rate: float = 0.1  # 脱敏速率 (新增)
    upregulation_rate: float = 0.05   # 上调速率 (新增)


class DopamineReceptor:
    """多巴胺受体动力学

    参考:
    - Seeman et al. (2005): D2受体密度与亲和力
    - Laruelle (2000): 受体占有率计算

    D2受体类型:
    - D2_high: 高亲和力态 (占20%)
    - D2_low: 低亲和力态 (占80%)
    """

    def __init__(
        self,
        receptor_type: str = "D2",      # D1, D2, D3, D4, D5
        density: float = 1.0,           # 受体密度 (相对值)
        high_affinity_fraction: float = 0.2,  # 高亲和力态比例
    ):
        self.receptor_type = receptor_type
        self.density = density
        self.high_affinity_fraction = high_affinity_fraction

        # D2受体参数 (参考Seeman 2005)
        if receptor_type == "D2":
            self.Kd_high = 2.0e-9   # 高亲和力 Kd (nM)
            self.Kd_low = 20.0e-9   # 低亲和力 Kd (nM)
        elif receptor_type == "D1":
            self.Kd_high = 5.0e-9
            self.Kd_low = 50.0e-9
        else:
            self.Kd_high = 5.0e-9
            self.Kd_low = 50.0e-9

        # 状态
        self.occupancy = 0.0
        self.desensitization = 0.0
        self.upregulation = 0.0

    def compute_occupancy(
        self,
        dopamine_concentration: float,
    ) -> float:
        """计算受体占有率 - Langmuir吸附方程

        occupancy = [DA] / ([DA] + Kd)

        高亲和力态先饱和，低亲和力态后饱和

        Args:
            dopamine_concentration: DA浓度 (nM, 假设范围0-100)

        Returns:
            occupancy: 受体占有率 [0, 1]
        """
        # 高亲和力态占有率
        occupancy_high = dopamine_concentration / (dopamine_concentration + self.Kd_high * 1e9)

        # 低亲和力态占有率
        occupancy_low = dopamine_concentration / (dopamine_concentration + self.Kd_low * 1e9)

        # 总占有率 = weighted average
        self.occupancy = (
            self.high_affinity_fraction * occupancy_high +
            (1 - self.high_affinity_fraction) * occupancy_low
        ) * self.density

        return self.occupancy

    def apply_desensitization(self, chronic_da_level: float):
        """应用受体脱敏/下调

        高慢性DA → D2下调 (减少密度)
        低慢性DA → D2上调 (增加密度)

        参考: Wilson et al. (1996)
        """
        if chronic_da_level > 0.6:
            # 高DA → 下调
            self.desensitization = min(0.5, self.desensitization + 0.01)
            self.density *= (1 - 0.005)
        elif chronic_da_level < 0.2:
            # 低DA → 上调
            self.upregulation = min(0.5, self.upregulation + 0.01)
            self.density *= (1 + 0.003)

        # 密度限制
        self.density = np.clip(self.density, 0.3, 2.0)

    def get_available_receptors(self) -> float:
        """获取可用受体数量 (考虑脱敏)"""
        return self.density * (1 - self.desensitization)

    def get_summary(self) -> Dict:
        return {
            'receptor_type': self.receptor_type,
            'density': self.density,
            'occupancy': self.occupancy,
            'desensitization': self.desensitization,
            'upregulation': self.upregulation,
            'available': self.get_available_receptors(),
        }


class DATransporter:
    """多巴胺转运体 (DAT)

    重摄取机制:
    - DAT密度决定清除速率
    - 高DAT → 快清除 → 低DA持续时间
    - 低DAT → 慢清除 → 高DA持续时间

    参考: Bannon & Whitty (1997)
    """

    def __init__(
        self,
        density: float = 1.0,    # DAT密度 (相对值)
        Vmax: float = 100.0,     # 最大转运速率
        Km: float = 0.2,         # 半饱和浓度
    ):
        self.density = density
        self.Vmax = Vmax
        self.Km = Km

    def compute_reuptake(
        self,
        extracellular_da: float,
    ) -> float:
        """计算重摄取速率

        Michaelis-Menten动力学: v = Vmax * [DA] / (Km + [DA])

        Args:
            extracellular_da: 细胞外DA浓度 [0, 1]

        Returns:
            reuptake_rate: 重摄取速率
        """
        reuptake = self.Vmax * extracellular_da / (self.Km + extracellular_da)
        return reuptake * self.density

    def apply_blockade(self, blockade_level: float):
        """应用转运体阻断 (如cocaine, methylphenidate)

        Args:
            blockade_level: 阻断程度 [0, 1]
        """
        effective_density = self.density * (1 - blockade_level)
        return effective_density


@dataclass
class SynapticTransmission:
    """突触传递"""
    pre_neuron: str
    post_neuron: str
    neurotransmitter: str
    strength: float  # 突触强度


# ============ 递质系统 ============

class DopamineSystem(nn.Module):
    """
    多巴胺系统

    三个主要通路：
    - Mesolimbic: 奖励、动机 (VTA → NAc)
    - Nigrostriatal: 运动控制 (SNc → Striatum)
    - Mesocortical: 认知、注意力 (VTA → PFC)

    新增: Tonic-Phasic分离
    - Tonic: 慢变化基线水平 (背景DA，调节长期动机)
    - Phasic: 快突发响应 (RPE信号，瞬时奖励)
    """

    def __init__(
        self,
        baseline: float = 0.5,
        tonic_baseline: float = 0.3,    # Tonic基线 (新增)
        phasic_decay_rate: float = 0.3, # Phasic衰减率 (新增)
    ):
        super().__init__()

        self.baseline = baseline
        self.current_level = baseline

        # Tonic-Phasic分离 (新增)
        self.dopamine_tonic = tonic_baseline    # 慢变化基线
        self.dopamine_phasic = 0.0              # 瞬时突发
        self.phasic_decay_rate = phasic_decay_rate
        self.tonic_adaptation_rate = 0.01      # Tonic慢适应率

        # 三个通路
        self.meso_limbic = 0.5  # 奖励
        self.nigro_striatal = 0.5  # 运动
        self.meso_cortical = 0.5   # 认知

        # 释放与重摄取
        self.release_rate = 0.1
        self.reuptake_rate = 0.05

        self.history = deque(maxlen=100)

    def compute_reward_signal(
        self,
        reward: float,
        expectation: float,
    ) -> Tuple[float, float, float]:
        """
        计算奖励信号 (RPE) - Tonic-Phasic分离

        Returns:
            total_level: 总多巴胺水平
            tonic: Tonic基线
            phasic: Phasic突发
        """
        # 预测误差
        rpe = reward - expectation

        # Phasic响应: 快突发/快衰减
        if rpe > 0:
            # 正向RPE → Phasic burst
            self.dopamine_phasic = min(0.7, 0.5 * rpe)
            self.meso_limbic = min(1.0, self.meso_limbic + rpe * self.release_rate)
        else:
            # 负向RPE → Phasic dip
            self.dopamine_phasic = max(-0.3, 0.3 * rpe)

        # Tonic响应: 慢适应
        # RPE累积影响基线 (长期动机状态)
        self.dopamine_tonic = np.clip(
            self.dopamine_tonic + self.tonic_adaptation_rate * rpe,
            0.1, 0.8
        )

        # Phasic衰减 (快速)
        self.dopamine_phasic *= (1 - self.phasic_decay_rate)

        # 总水平 = Tonic + Phasic
        self.current_level = np.clip(self.dopamine_tonic + self.dopamine_phasic, 0.0, 1.0)

        # 更新通路
        self.current_level = (
            self.meso_limbic * 0.4 +
            self.nigro_striatal * 0.3 +
            self.meso_cortical * 0.3
        )

        self.history.append(self.current_level)

        return self.current_level, self.dopamine_tonic, self.dopamine_phasic

    def get_tonic_level(self) -> float:
        """获取Tonic基线水平"""
        return self.dopamine_tonic

    def get_phasic_level(self) -> float:
        """获取Phasic突发水平"""
        return self.dopamine_phasic

    def set_tonic_baseline(self, level: float):
        """设置Tonic基线 (模拟慢性DA状态)"""
        self.dopamine_tonic = np.clip(level, 0.1, 0.8)

    def compute_motor_signal(
        self,
        action_quality: float,
    ) -> float:
        """计算运动信号"""
        # SNc → Striatal 通路
        if action_quality > 0:
            self.nigro_striatal = min(1.0, self.nigro_striatal + action_quality * 0.1)
        else:
            self.nigro_striatal = max(0.0, self.nigro_striatal + action_quality * 0.05)

        return self.nigro_striatal

    def compute_cognitive_signal(
        self,
        novelty: float,
    ) -> float:
        """计算认知信号"""
        self.meso_cortical = min(1.0, self.meso_cortical + novelty * 0.1)
        return self.meso_cortical

    def decay(self):
        """自然衰减 + 重摄取"""
        self.current_level *= (1 - self.reuptake_rate)
        self.meso_limbic *= (1 - self.reuptake_rate)
        self.nigro_striatal *= (1 - self.reuptake_rate)
        self.meso_cortical *= (1 - self.reuptake_rate)

    def get_summary(self) -> Dict:
        return {
            'level': self.current_level,
            'meso_limbic': self.meso_limbic,
            'nigro_striatal': self.nigro_striatal,
            'meso_cortical': self.meso_cortical,
        }


class SerotoninSystem(nn.Module):
    """
    血清素系统

    功能：
    - 情绪调节
    - 睡眠-觉醒
    - 饱腹感
    - 冲动控制
    """

    def __init__(
        self,
        baseline: float = 0.5,
    ):
        super().__init__()

        self.baseline = baseline
        self.current_level = baseline

        # 脑区
        self.raphe = 0.5      # 中缝核
        self.prefrontal = 0.5    # 前额叶

        # 功能调节
        self.mood = 0.5
        self.sleep_wake = 0.5
        self.satiety = 0.5

        self.history = deque(maxlen=100)

    def compute_mood(
        self,
        reward: float,
        punishment: float,
    ) -> float:
        """计算情绪"""
        # 正向事件 → 增加血清素
        # 负向事件 → 减少血清素

        mood_change = (reward * 0.1 - punishment * 0.1)
        self.mood = np.clip(self.mood + mood_change, 0, 1)

        self.raphe = self.mood
        self.prefrontal = self.mood * 0.9

        self.current_level = self.raphe

        self.history.append(self.current_level)

        return self.current_level

    def compute_sleep_signal(
        self,
        circadian_hour: float,
    ) -> float:
        """计算睡眠信号"""
        # 24小时周期
        # 夜间高，白天低
        phase = np.sin(circadian_hour * np.pi / 12)

        if phase < 0:
            # 夜间：增加困意
            self.sleep_wake = 0.5 + abs(phase) * 0.3
        else:
            self.sleep_wake = 0.5 - phase * 0.2

        return self.sleep_wake

    def compute_impulse(
        self,
        temptation: float,
    ) -> float:
        """计算冲动控制 (连续调制)"""
        # 血清素高 → 冲动控制好 (连续, 无硬阈值)
        # impulse_control = serotonergic tone × continuous inhibition function
        serotonergic_inhibition = self.current_level

        # 高诱惑 → 需要更多血清素才能控制
        # 高诱惑 + 低血清素 → 冲动控制渐弱 (而非突然归零)
        temptation_load = float(np.clip(temptation, 0.0, 1.0))
        effective_control = serotonergic_inhibition * (1.0 - 0.5 * temptation_load)
        return float(np.clip(effective_control, 0.0, 1.0))

    def decay(self):
        """自然衰减"""
        self.current_level *= 0.98
        self.mood *= 0.99


class AcetylcholineSystem(nn.Module):
    """
    乙酰胆碱系统

    功能：
    - 注意力聚焦
    - 学习与记忆
    - 肌肉收缩
    """

    def __init__(
        self,
        baseline: float = 0.5,
    ):
        super().__init__()

        self.baseline = baseline
        self.current_level = baseline

        # 脑区
        self.basal_forebrain = 0.5  # 注意力
        self.hippocampus = 0.5      # 记忆
        self.motor_cortex = 0.5        # 运动

        self.history = deque(maxlen=100)

    def compute_attention(
        self,
        novelty: float,
        salience: float,
    ) -> float:
        """计算注意力"""
        # 新奇 + 显著 → 乙酰胆碱释放

        attention_signal = novelty * 0.4 + salience * 0.4 + self.baseline * 0.2

        self.basal_forebrain = np.clip(attention_signal, 0, 1)

        self.current_level = self.basal_forebrain
        self.history.append(self.current_level)

        return self.current_level

    def compute_memory_consolidation(
        self,
        memory_strength: float,
    ) -> float:
        """计算记忆巩固"""
        # 乙酰胆碱促进记忆巩固

        self.hippocampus = np.clip(memory_strength, 0, 1)

        return self.hippocampus

    def compute_motor(
        self,
        motor_command: float,
    ) -> float:
        """计算运动信号"""
        self.motor_cortex = np.clip(motor_command, 0, 1)
        return self.motor_cortex


class GlutamateGABASystem(nn.Module):
    """
    谷氨酸 + GABA 系统

    平衡兴奋/抑制
    """

    def __init__(
        self,
    ):
        super().__init__()

        # 谷氨酸 (兴奋)
        self.glutamate = 0.5
        self.glutamate_release = 0.1

        # GABA (抑制)
        self.gaba = 0.5
        self.gaba_release = 0.1

        # E/I 平衡
        self.e_i_ratio = 1.0

    def compute_excitation(
        self,
        input_strength: float,
        learning: bool = False,
    ) -> float:
        """计算兴奋"""
        if learning:
            # 学习时增加谷氨酸
            self.glutamate = np.clip(input_strength * 1.2, 0, 1)
        else:
            self.glutamate = np.clip(input_strength, 0, 1)

        return self.glutamate

    def compute_inhibition(
        self,
        threat_level: float = 0.0,
    ) -> float:
        """计算抑制 (连续调制, 无硬阈值)"""
        baseline = 0.5
        # 威胁 → GABA增加 (镇静) — 连续响应
        # 低威胁(<0.3): 轻微增加; 中威胁(0.3-0.7): 中等增加; 高威胁(>0.7): 强增加
        gaba_drive = float(np.clip(threat_level, 0.0, 1.0))
        self.gaba = float(np.clip(
            baseline + gaba_drive * 0.3 * gaba_drive,  # quadratic for saturation at high threat
            0.1, 1.0
        ))
        return self.gaba

    def compute_balance(self) -> float:
        """计算E/I平衡"""
        if self.gaba > 0:
            self.e_i_ratio = self.glutamate / self.gaba

        # 平衡 > 1: 兴奋为主
        # 平衡 < 1: 抑制为主
        return self.e_i_ratio


class NorepinephrineSystem(nn.Module):
    """
    去甲肾上腺素系统

    功能：
    - 警觉
    - 应激反应
    - 注意力
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.level = 0.3      # 基线
        self.locus_coeruleus = 0.3

    def compute_alertness(
        self,
        novelty: float,
        urgency: float,
    ) -> float:
        """计算警觉 (连续调制)"""
        # 连续输入 → 连续响应 (无硬阈值)
        alert_drive = 0.4 * novelty + 0.6 * urgency
        # 快速上升，慢衰减 (不对称动力学)
        if alert_drive > self.level:
            self.level = float(np.clip(
                self.level + 0.2 * (alert_drive - self.level),
                0.05, 1.0
            ))
        else:
            self.level = float(np.clip(self.level * 0.95, 0.05, 1.0))

        self.locus_coeruleus = self.level
        return self.level

    def compute_stress(
        self,
        threat: float,
    ) -> float:
        """计算应激 (连续调制, 无硬阈值)"""
        # 连续威胁 → 连续NE上升 (sigmoid-shaped)
        # 低威胁(<0.3): 几乎无影响; 中威胁(0.3-0.7): 线性增长; 高威胁(>0.7): 饱和
        stress_drive = float(np.clip(threat, 0.0, 1.0))
        if stress_drive > self.level:
            # 上升快 (SAM快速激活)
            self.level = float(np.clip(
                self.level + 0.15 * stress_drive,
                0.05, 1.0
            ))
        else:
            # 恢复慢 (NE清除)
            self.level = float(np.clip(self.level * 0.93, 0.05, 1.0))

        return self.level


class EndorphinSystem(nn.Module):
    """
    内啡肽系统

    功能：
    - 镇痛
    - 愉悦
    - 压力缓解
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.level = 0.2
        self.baseline = 0.2

    def compute_reward(
        self,
        pleasure: float,
    ) -> float:
        """计算愉悦"""
        if pleasure > 0:
            self.level = min(1.0, self.level + pleasure * 0.2)
        else:
            self.level *= 0.98

        return self.level

    def compute_painrelief(
        self,
        pain: float,
    ) -> float:
        """镇痛效果"""
        # 内啡肽可以减轻疼痛感知
        pain_reduction = self.level * pain * 0.5
        return max(0, pain - pain_reduction)


# ============ 整合系统 ============

class NeurotransmitterSystem(nn.Module):
    """
    完整神经递质系统
    """

    def __init__(self, event_bus=None):
        super().__init__()

        # 主要递质
        self.dopamine = DopamineSystem()
        self.serotonin = SerotoninSystem()
        self.acetylcholine = AcetylcholineSystem()
        self.glutamate_gaba = GlutamateGABASystem()
        self.norepinephrine = NorepinephrineSystem()
        self.endorphin = EndorphinSystem()

        # 状态
        self.overall_state = "neutral"
        self.motivation = 0.5
        self.arousal = 0.5

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "brain_update",
                self._handle_brain_update,
                priority=1,
                name="neurotransmitter",
            )
            event_bus.subscribe(
                "sensory_neuro_update",
                self._handle_sensory_neuro_update,
                priority=2,
                name="neurotransmitter_sensory",
            )

    def _handle_sensory_neuro_update(self, event) -> Dict:
        """处理感觉-神经递质耦合事件。

        从感觉驱动的神经化学释放更新递质水平。
        """
        sensory_neuro = event.data.get('sensory_neurochemical', {})

        # 感觉驱动的肾上腺素 → 去甲肾上腺素系统
        sensory_adrenaline = sensory_neuro.get('sensory_adrenaline', 0.0)
        if sensory_adrenaline > 0.1:
            # 增加去甲肾上腺素警觉性
            current_ne = self.norepinephrine.alertness_level
            self.norepinephrine.alertness_level = min(1.0, current_ne + sensory_adrenaline * 0.15)

        # 感觉驱动的多巴胺 → 多巴胺系统
        sensory_dopamine = sensory_neuro.get('sensory_dopamine', 0.0)
        if sensory_dopamine > 0.1:
            # 增加多巴胺动机
            current_dop = self.dopamine.current_level
            self.dopamine.current_level = min(1.0, current_dop + sensory_dopamine * 0.10)
            # 触发 phasic burst
            self.dopamine.dopamine_phasic = min(0.5, self.dopamine.dopamine_phasic + sensory_dopamine * 0.15)

        # 感觉驱动的乙酰胆碱 → 注意力系统
        sensory_ach = sensory_neuro.get('sensory_acetylcholine', 0.0)
        if sensory_ach > 0.1:
            # 增加乙酰胆碱注意力
            current_ach = self.acetylcholine.attention_level
            self.acetylcholine.attention_level = min(1.0, current_ach + sensory_ach * 0.10)

        return {}

    def _handle_brain_update(self, event) -> Dict:
        """Event-driven handler for brain_update events."""
        state = event.data.get("internal_state", {})
        info_gain = event.data.get("info_gain_reward", 0.0)

        result = self.step(
            reward=info_gain,
            expectation=state.get("expectation", 0.0),
            novelty=state.get("novelty", 0.0),
            salience=state.get("salience", 0.0),
            pain=0.0,
            threat=state.get("threat", 0.0),
            urgency=state.get("urgency", 0.0),
        )

        state["nt_dopamine"] = result["dopamine"]
        state["nt_serotonin"] = result["serotonin"]
        state["nt_acetylcholine"] = result["acetylcholine"]
        state["nt_norepinephrine"] = result["norepinephrine"]
        state["nt_motivation"] = result["motivation"]
        state["nt_arousal"] = result["arousal"]
        state["nt_state"] = result["state"]
        state["dopamine"] = result["dopamine"]

        return result

    def step(
        self,
        reward: float = 0.0,
        expectation: float = 0.0,
        novelty: float = 0.0,
        salience: float = 0.0,
        pain: float = 0.0,
        threat: float = 0.0,
        urgency: float = 0.0,
    ) -> Dict:
        """
        一步更新

        Returns:
            state: 递质状态
        """
        # 1. 多巴胺
        dop_level = self.dopamine.compute_reward_signal(reward, expectation)
        dop_motor = self.dopamine.compute_motor_signal(0.5)
        dop_cog = self.dopamine.compute_cognitive_signal(novelty)

        # 2. 血清素
        ser_level = self.serotonin.compute_mood(
            reward=max(0, reward),
            punishment=max(0, -reward)
        )

        # 3. 乙酰胆碱
        ach_level = self.acetylcholine.compute_attention(novelty, salience)

        # 4. 谷氨酸/GABA
        glu_level = self.glutamate_gaba.compute_excitation(
            salience,
            learning=novelty > 0.5
        )
        gaba_level = self.glutamate_gaba.compute_inhibition(threat)

        # 5. 去甲肾上腺素
        ne_level = self.norepinephrine.compute_alertness(novelty, urgency)
        stress = self.norepinephrine.compute_stress(threat)

        # 6. 内啡肽
        end_level = self.endorphin.compute_reward(pleasure=reward if reward > 0 else 0)
        pain_relief = self.endorphin.compute_painrelief(pain)

        # 7. 计算整体状态
        self.motivation = dop_level * 0.4 + ser_level * 0.3 + ach_level * 0.3
        self.arousal = ne_level * 0.5 + ach_level * 0.3 + end_level * 0.2

        # 更新整体状态 (连续混合, 无硬阈值if/elif)
        # 使用加权融合: 每种状态的"激活度"基于对应递质水平
        stress_activation = float(np.clip(stress * 1.5 - 0.2, 0.0, 1.0))    # 平滑过渡
        alert_activation = float(np.clip(ne_level * 1.3 - 0.15, 0.0, 1.0))
        motivation_activation = float(np.clip(dop_level * 1.3 - 0.15, 0.0, 1.0))
        sadness_activation = float(np.clip((0.3 - ser_level) * 2, 0.0, 1.0))
        neutral_activation = max(0.0, 1.0 - stress_activation - alert_activation
                                 - motivation_activation - sadness_activation)

        # 归一化激活权重
        activations = {
            "stress": stress_activation,
            "alert": alert_activation,
            "motivated": motivation_activation,
            "sad": sadness_activation,
            "neutral": neutral_activation,
        }
        total_act = sum(activations.values())
        if total_act > 0:
            self.overall_state = max(activations, key=activations.get)
        else:
            self.overall_state = "neutral"

        # 衰减
        self.dopamine.decay()
        self.serotonin.decay()

        return {
            'dopamine': dop_level,
            'serotonin': ser_level,
            'acetylcholine': ach_level,
            'glutamate': glu_level,
            'gaba': gaba_level,
            'norepinephrine': ne_level,
            'endorphin': end_level,
            'pain_relief': pain_relief,
            'motivation': self.motivation,
            'arousal': self.arousal,
            'state': self.overall_state,
        }

    def get_e_i_balance(self) -> float:
        """获取兴奋/抑制平衡"""
        return self.glutamate_gaba.compute_balance()

    def get_attention_modulation(self, novelty: float) -> float:
        """获取注意力调制"""
        # 去甲肾上腺素 + 乙酰胆碱
        ne = self.norepinephrine.level
        ach = self.acetylcholine.current_level

        return ne * 0.3 + ach * 0.5 + novelty * 0.2

    def get_learning_modulation(self) -> float:
        """获取学习调制"""
        # 多巴胺 + 乙酰胆碱 + 谷氨酸
        dop = self.dopamine.current_level
        ach = self.acetylcholine.current_level
        glu = self.glutamate_gaba.glutamate

        return dop * 0.4 + ach * 0.3 + glu * 0.3

    def get_motor_modulation(self) -> float:
        """获取运动调制"""
        return self.dopamine.nigro_striatal

    def get_summary(self) -> Dict:
        """获取摘要"""
        return {
            'overall_state': self.overall_state,
            'motivation': self.motivation,
            'arousal': self.arousal,
            'dopamine': self.dopamine.get_summary(),
            'e_i_balance': self.get_e_i_balance(),
        }


# ============ 便捷函数 ============

def create_neurotransmitter_system() -> NeurotransmitterSystem:
    """创建神经递质系统"""
    return NeurotransmitterSystem()


__all__ = [
    'Neurotransmitter',
    'Receptor',
    'DopamineReceptor',
    'DATransporter',
    'SynapticTransmission',
    'DopamineSystem',
    'SerotoninSystem',
    'AcetylcholineSystem',
    'GlutamateGABASystem',
    'NorepinephrineSystem',
    'EndorphinSystem',
    'NeurotransmitterSystem',
    'create_neurotransmitter_system',
]