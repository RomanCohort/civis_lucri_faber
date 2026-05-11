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
    """受体"""
    name: str
    neurotransmitter: str  # 对应递质
    affinity: float   # 亲和力
    density: float   # 密度
    state: str       # "active" | "desensitized" | "downregulated"


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
    """

    def __init__(
        self,
        baseline: float = 0.5,
    ):
        super().__init__()

        self.baseline = baseline
        self.current_level = baseline

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
    ) -> float:
        """
        计算奖励信号 (RPE)
        """
        # 预测误差
        rpe = reward - expectation

        if rpe > 0:
            # 正向误差 → 释放多巴胺
            release = rpe * self.release_rate
            self.meso_limbic = min(1.0, self.meso_limbic + release)
        else:
            # 负向误差 → 抑制
            self.meso_limbic = max(0.0, self.meso_limbic + rpe * 0.05)

        # 更新总水平
        self.current_level = (
            self.meso_limbic * 0.4 +
            self.nigro_striatal * 0.3 +
            self.meso_cortical * 0.3
        )

        self.history.append(self.current_level)

        return self.current_level

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
        """计算冲动控制"""
        # 血清素高 → 冲动控制好
        impulse_control = self.current_level

        # 高诱惑 → 需要更多血清素
        if temptation > 0.7 and impulse_control < 0.5:
            return 0.0  # 失控

        return impulse_control

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
        """计算抑制"""
        baseline = 0.5
        # 威胁 → GABA增加 (镇静)
        if threat_level > 0.5:
            self.gaba = np.clip(self.gaba + threat_level * 0.2, 0, 1)
        else:
            self.gaba = baseline

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

        self.state = "calm"  # calm | alert | stress

    def compute_alertness(
        self,
        novelty: float,
        urgency: float,
    ) -> float:
        """计算警觉"""
        if novelty > 0.5 or urgency > 0.7:
            self.state = "alert"
            self.level = min(1.0, self.level + 0.2)
        else:
            self.level *= 0.95

        self.locus_coeruleus = self.level

        return self.level

    def compute_stress(
        self,
        threat: float,
    ) -> float:
        """计算应激"""
        if threat > 0.7:
            self.state = "stress"
            # 急性应激
            self.level = min(1.0, self.level + threat * 0.3)
        elif threat > 0.3:
            self.state = "alert"
        else:
            self.state = "calm"

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

    def __init__(self):
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

        # 更新整体状态
        if stress > 0.7:
            self.overall_state = "stress"
        elif ne_level > 0.7:
            self.overall_state = "alert"
        elif dop_level > 0.7:
            self.overall_state = "motivated"
        elif ser_level < 0.3:
            self.overall_state = "sad"
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