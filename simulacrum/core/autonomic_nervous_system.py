"""
自主神经系统 (Autonomic Nervous System)

模拟自主神经系统的交感/副交感平衡：
1. 交感神经 (Sympathetic) - 战斗或逃跑反应
2. 副交感神经 (Parasympathetic) - 休息与消化
3. 压力感受器反射 (Baroreceptor Reflex) - 心血管负反馈
4. 多迷走神经理论 (Polyvagal Theory) - 三层神经状态

生物参考文献:
- Cannon (1932): 战斗或逃跑反应
- Porges (1995, 2001): 多迷走神经理论
- Guyton & Hall (2006): 医学生理学 - 压力感受器反射
- Thayer & Lane (2000): 神经内脏整合模型 (HRV)
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


# ============ 状态定义 ============

@dataclass
class ANSState:
    """自主神经系统状态"""
    sympathetic_tone: float = 0.3       # 交感神经张力 [0,1]
    parasympathetic_tone: float = 0.5   # 副交感神经张力 [0,1]
    heart_rate: float = 0.5             # 心率 (归一化) [0,1]
    blood_pressure: float = 0.5         # 血压 (归一化) [0,1]
    hrv: float = 0.6                    # 心率变异性 [0,1]
    pupil_dilation: float = 0.3         # 瞳孔扩张 [0,1]
    digestion_rate: float = 0.6         # 消化速率 [0,1]
    sweat_response: float = 0.1         # 出汗反应 [0,1]
    glucose_release: float = 0.2        # 肝糖释放 [0,1]
    polyvagal_state: str = "ventral_vagal"  # 多迷走状态
    polyvagal_level: float = 1.0            # 连续多迷走水平 [0,1]: 1.0=ventral, 0.5=sympathetic, 0.0=dorsal
    vagal_withdrawal_steps: int = 0     # 迷走神经撤退持续步数


# ============ 交感神经分支 ============

class SympatheticBranch(nn.Module):
    """
    交感神经分支 - 战斗或逃跑反应

    当检测到威胁、新奇或紧迫性时激活：
    - 心率增加
    - 瞳孔扩张
    - 消化抑制
    - 肝糖释放
    - 出汗增加

    参考: Cannon (1932) "The Wisdom of the Body"
    """

    def __init__(self, reactivity: float = 1.0, decay_rate: float = 0.05):
        super().__init__()
        self.reactivity = nn.Parameter(torch.tensor(reactivity))
        self.decay_rate = decay_rate
        self.current_tone = 0.3

        # 激活网络: threat + novelty + urgency -> sympathetic activation
        self.activation_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, threat: float, novelty: float, urgency: float) -> dict[str, float]:
        x = torch.tensor([threat, novelty, urgency], dtype=torch.float32)
        activation = self.activation_net(x).item()

        # 交感激活受个体反应性调节
        self.current_tone = float(np.clip(
            self.current_tone * (1 - self.decay_rate) + activation * self.reactivity.item(),
            0.0, 1.0
        ))

        # 交感激活的生理效应
        heart_rate_effect = 0.3 * self.current_tone
        pupil_effect = 0.5 * self.current_tone
        digestion_suppression = -0.4 * self.current_tone
        glucose_effect = 0.4 * self.current_tone
        sweat_effect = 0.3 * self.current_tone

        return {
            'sympathetic_tone': self.current_tone,
            'heart_rate_effect': heart_rate_effect,
            'pupil_dilation_effect': pupil_effect,
            'digestion_effect': digestion_suppression,
            'glucose_release_effect': glucose_effect,
            'sweat_effect': sweat_effect,
        }

    def decay(self):
        """自然衰减 (儿茶酚胺清除)"""
        self.current_tone *= (1 - self.decay_rate)
        self.current_tone = float(np.clip(self.current_tone, 0.05, 1.0))


# ============ 副交感神经分支 ============

class ParasympatheticBranch(nn.Module):
    """
    副交感神经分支 - 休息与消化

    当检测到安全信号和社会参与时激活：
    - 心率降低 (通过迷走神经)
    - 促进消化
    - 能量保存
    - 恢复促进

    参考: Porges (1995) "Orienting in a defensive world"
    """

    def __init__(self, baseline_vagal_tone: float = 0.5):
        super().__init__()
        self.vagal_tone = baseline_vagal_tone  # 基础迷走神经张力
        self.current_tone = baseline_vagal_tone

        # 激活网络: safety + social_engagement -> parasympathetic activation
        self.activation_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, safety_signal: float, social_engagement: float) -> dict[str, float]:
        x = torch.tensor([safety_signal, social_engagement], dtype=torch.float32)
        activation = self.activation_net(x).item()

        # 副交感激活: 安全和社会参与促进恢复
        target = 0.3 * self.vagal_tone + 0.7 * activation
        self.current_tone = float(np.clip(
            0.9 * self.current_tone + 0.1 * target,
            0.1, 1.0
        ))

        # 副交感激活的生理效应
        heart_rate_reduction = -0.3 * self.current_tone
        digestion_promotion = 0.4 * self.current_tone
        recovery_boost = 0.3 * self.current_tone

        return {
            'parasympathetic_tone': self.current_tone,
            'heart_rate_effect': heart_rate_reduction,
            'digestion_effect': digestion_promotion,
            'recovery_effect': recovery_boost,
        }

    def compute_vagal_tone(self) -> float:
        """计算当前迷走神经张力"""
        return self.current_tone


# ============ 压力感受器反射 ============

class BaroreceptorReflex(nn.Module):
    """
    压力感受器反射 - 心血管稳态负反馈

    当血压升高时：
    - 增加副交感神经活动 (心率降低)
    - 减少交感神经活动 (血管扩张)
    当血压降低时，反向调节

    参考: Guyton & Hall (2006) "Textbook of Medical Physiology"
    """

    def __init__(self, setpoint: float = 0.5, sensitivity: float = 0.8):
        super().__init__()
        self.setpoint = setpoint
        self.sensitivity = nn.Parameter(torch.tensor(sensitivity))
        self.feedback_gain = nn.Parameter(torch.tensor(0.5))

    def forward(self, blood_pressure: float) -> dict[str, float]:
        """
        计算压力感受器反射修正量

        delta = -sensitivity * (bp - setpoint)
        bp > setpoint -> 增加副交感，减少交感
        bp < setpoint -> 减少副交感，增加交感
        """
        deviation = blood_pressure - self.setpoint
        correction = -self.sensitivity.item() * deviation

        # 副交感修正: bp高时增加
        parasympathetic_correction = max(0.0, correction) * self.feedback_gain.item()
        # 交感修正: bp高时减少
        sympathetic_correction = min(0.0, correction) * self.feedback_gain.item()

        return {
            'parasympathetic_correction': parasympathetic_correction,
            'sympathetic_correction': sympathetic_correction,
            'deviation': deviation,
        }


# ============ 多迷走神经系统 ============

class PolyvagalSystem(nn.Module):
    """
    多迷走神经理论实现

    三层等级状态 (Porges, 2001):
    1. 腹侧迷走 (Ventral Vagal) - 社会参与、安全感
    2. 交感 (Sympathetic) - 战斗或逃跑
    3. 背侧迷走 (Dorsal Vagal) - 冻结/关闭

    等级规则: 不能跳过中间层级
    ventral_vagal <-> sympathetic <-> dorsal_vagal

    参考: Porges (2001) "The Polyvagal Theory"
    """

    def __init__(self):
        super().__init__()
        self.current_state = "ventral_vagal"
        self.polyvagal_level = 1.0  # 连续: 1.0=ventral_vagal, 0.5=sympathetic, 0.0=dorsal_vagal
        self.dorsal_pressure = 0.0  # 连续背侧压力 (替代整数累积计数器)
        self.recovery_pressure = 0.0  # 连续恢复压力 (替代整数累积计数器)
        self.state_history = deque(maxlen=100)

    def forward(self, sympathetic_tone: float, parasympathetic_tone: float,
                social_safety: float, neuroception_safety: float = 0.5) -> dict[str, Any]:
        """
        根据自主神经状态确定多迷走状态 (连续动力学)

        polyvagal_level: 连续值 [0,1]
          1.0 = ventral_vagal (社会参与/安全)
          0.5 = sympathetic (战斗或逃跑)
          0.0 = dorsal_vagal (冻结/关闭)

        使用连续压力累积 + 时间惯性替代离散状态机
        """
        effective_safety = neuroception_safety * social_safety

        # 连续背侧压力: 交感持续高 → 压力渐增; 低 → 衰减
        dorsal_drive = max(0, sympathetic_tone - 0.5) * 0.12
        dorsal_decay = self.dorsal_pressure * 0.08
        self.dorsal_pressure = float(np.clip(
            self.dorsal_pressure + dorsal_drive - dorsal_decay, 0.0, 1.0))

        # 连续恢复压力: 副交感主导 + 交感低 → 恢复渐增
        recovery_cond = max(0, parasympathetic_tone - 0.3) * max(0, 1.0 - sympathetic_tone / 0.6)
        recovery_drive = recovery_cond * 0.10
        recovery_decay = self.recovery_pressure * 0.06
        self.recovery_pressure = float(np.clip(
            self.recovery_pressure + recovery_drive - recovery_decay, 0.0, 1.0))

        # 计算连续目标水平 (三路加权)
        para_drive = parasympathetic_tone * effective_safety
        symp_drive = sympathetic_tone * (1.0 - effective_safety * 0.5)
        dorsal_total = self.dorsal_pressure * max(0, sympathetic_tone - 0.4)

        total_drive = para_drive + symp_drive + dorsal_total + 0.01
        target_level = (para_drive * 1.0 + symp_drive * 0.5 + dorsal_total * 0.0) / total_drive

        # 时间惯性平滑过渡
        transition_rate = 0.08
        self.polyvagal_level += transition_rate * (target_level - self.polyvagal_level)
        self.polyvagal_level = float(np.clip(self.polyvagal_level, 0.0, 1.0))

        # 从连续水平派生离散状态名 (向后兼容)
        if self.polyvagal_level > 0.65:
            self.current_state = "ventral_vagal"
        elif self.polyvagal_level > 0.3:
            self.current_state = "sympathetic"
        else:
            self.current_state = "dorsal_vagal"

        self.state_history.append(self.current_state)

        # 连续计算状态效应
        social_cap = _clamp(
            parasympathetic_tone * (1 - 0.5 * sympathetic_tone) * neuroception_safety
            + 0.1, 0.0, 1.0
        )
        cognitive_cap = _clamp(
            0.3 + 0.5 * parasympathetic_tone * (1 - 0.7 * sympathetic_tone),
            0.05, 1.0
        )
        recovery_rate = _clamp(
            0.1 + 0.5 * parasympathetic_tone * (1 - 0.3 * sympathetic_tone),
            0.05, 1.0
        )

        state_desc = {
            "ventral_vagal": "社会参与/安全状态",
            "sympathetic": "战斗或逃跑状态",
            "dorsal_vagal": "冻结/关闭状态",
        }

        return {
            'polyvagal_state': self.current_state,
            'polyvagal_level': self.polyvagal_level,
            'state_effects': {
                'social_capacity': social_cap,
                'cognitive_capacity': cognitive_cap,
                'recovery_rate': recovery_rate,
                'description': state_desc[self.current_state],
            },
            'dorsal_pressure': self.dorsal_pressure,
            'recovery_pressure': self.recovery_pressure,
        }

    def get_state_name(self) -> str:
        return self.current_state


# ============ 自主神经系统 (聚合器) ============

class AutonomicNervousSystem(nn.Module):
    """
    自主神经系统 - 整合交感/副交感/压力反射/多迷走

    主要功能:
    - 维持心血管稳态
    - 调节应激反应
    - 管理多迷走状态转换
    - 提供HRV作为健康指标

    参考:
    - Thayer & Lane (2000): 神经内脏整合模型
    - Porges (2001): 多迷走神经理论
    """

    def __init__(self, sympathetic_reactivity: float = 1.0,
                 baseline_vagal_tone: float = 0.5,
                 baroreceptor_setpoint: float = 0.5,
                 event_bus=None):
        super().__init__()

        self.sympathetic = SympatheticBranch(reactivity=sympathetic_reactivity)
        self.parasympathetic = ParasympatheticBranch(baseline_vagal_tone=baseline_vagal_tone)
        self.baroreceptor = BaroreceptorReflex(setpoint=baroreceptor_setpoint)
        self.polyvagal = PolyvagalSystem()

        self.state = ANSState()
        self.step_count = 0
        self.event_bus = event_bus

        # RMSSD HRV 计算缓冲区
        self._rr_history = deque(maxlen=60)  # RR 间期历史 (ms)

        # Event-driven registration
        if self.event_bus is not None:
            self.event_bus.subscribe(
                "neural_regulation",
                self.on_neural_regulation,
                priority=0,
                name="ans",
            )

    def step(self, threat: float = 0.0, novelty: float = 0.0,
             urgency: float = 0.0, safety_signal: float = 0.5,
             social_engagement: float = 0.5,
             neuroception_safety: float = 0.5) -> dict[str, Any]:
        """
        执行一个自主神经调节步

        增强: neuroception 输入 + RMSSD HRV
        """
        self.step_count += 1

        # 1. 交感神经激活
        symp_result = self.sympathetic(threat, novelty, urgency)

        # 2. 副交感神经激活
        para_result = self.parasympathetic(safety_signal, social_engagement)

        # 3. 计算当前心率 (交感↑ / 副交感↓)
        heart_rate = float(np.clip(
            0.5 + symp_result['heart_rate_effect'] + para_result['heart_rate_effect'],
            0.3, 1.0
        ))

        # 4. 计算血压
        blood_pressure = float(np.clip(
            0.5 + 0.3 * symp_result['sympathetic_tone']
            - 0.2 * para_result['parasympathetic_tone'],
            0.2, 1.0
        ))

        # 5. 压力感受器反射 (负反馈调节)
        baro_result = self.baroreceptor(blood_pressure)

        # 应用压力反射修正
        adjusted_sympathetic = float(np.clip(
            symp_result['sympathetic_tone'] + baro_result['sympathetic_correction'],
            0.0, 1.0
        ))
        adjusted_parasympathetic = float(np.clip(
            para_result['parasympathetic_tone'] + baro_result['parasympathetic_correction'],
            0.0, 1.0
        ))

        # 6. HRV计算 (RMSSD 模拟)
        # 将心率转换为 RR 间期 (ms) 并追踪历史
        current_rr_ms = 60000.0 / max(1.0, heart_rate * 120.0)  # 心率归一化→实际BPM
        self._rr_history.append(current_rr_ms)

        if len(self._rr_history) >= 10:
            rrs = list(self._rr_history)
            recent = rrs[-30:] if len(rrs) >= 30 else rrs
            diffs = np.diff(recent)
            rmssd = float(np.sqrt(np.mean(diffs**2)))
            # 归一化: 健康人 RMSSD ~20-100ms, 映射到 [0,1]
            hrv = float(np.clip(rmssd / 100.0, 0.0, 1.0))
        else:
            # 冷启动: 用副交感/交感比值估算
            hrv = float(np.clip(
                0.3 + 0.7 * adjusted_parasympathetic * (1 - 0.5 * adjusted_sympathetic),
                0.0, 1.0
            ))

        # 7. 多迷走状态 (含 neuroception)
        polyvagal_result = self.polyvagal(
            adjusted_sympathetic, adjusted_parasympathetic,
            social_engagement, neuroception_safety,
        )

        # 8. 更新状态
        self.state = ANSState(
            sympathetic_tone=adjusted_sympathetic,
            parasympathetic_tone=adjusted_parasympathetic,
            heart_rate=heart_rate,
            blood_pressure=blood_pressure,
            hrv=hrv,
            pupil_dilation=float(np.clip(
                0.3 + symp_result['pupil_dilation_effect'], 0.1, 1.0
            )),
            digestion_rate=float(np.clip(
                0.6 + symp_result['digestion_effect'] + para_result['digestion_effect'],
                0.1, 1.0
            )),
            sweat_response=float(np.clip(
                0.1 + symp_result['sweat_effect'], 0.0, 1.0
            )),
            glucose_release=float(np.clip(
                0.2 + symp_result['glucose_release_effect'], 0.0, 1.0
            )),
            polyvagal_state=polyvagal_result['polyvagal_state'],
            polyvagal_level=polyvagal_result['polyvagal_level'],
            vagal_withdrawal_steps=int(self.polyvagal.dorsal_pressure * 10),
        )

        # 9. 交感神经自然衰减
        self.sympathetic.decay()

        return {
            'sympathetic_tone': self.state.sympathetic_tone,
            'parasympathetic_tone': self.state.parasympathetic_tone,
            'heart_rate': self.state.heart_rate,
            'blood_pressure': self.state.blood_pressure,
            'hrv': self.state.hrv,
            'pupil_dilation': self.state.pupil_dilation,
            'digestion_rate': self.state.digestion_rate,
            'sweat_response': self.state.sweat_response,
            'glucose_release': self.state.glucose_release,
            'polyvagal_state': self.state.polyvagal_state,
            'polyvagal_level': self.state.polyvagal_level,
            'social_capacity': polyvagal_result['state_effects']['social_capacity'],
            'cognitive_capacity': polyvagal_result['state_effects']['cognitive_capacity'],
            'recovery_rate': polyvagal_result['state_effects']['recovery_rate'],
            'baroreceptor_deviation': baro_result['deviation'],
        }

    def on_neural_regulation(self, event) -> dict[str, Any]:
        """Event handler for NEURAL_REGULATION events (priority=0)."""
        state = event.data["internal_state"]
        # neuroception: 基于安全信号和社交参与的无意识安全检测
        emotion_crit = state.get("emotion_criticality", 0.0)
        social_eng = state.get("social_engagement", 0.5)
        neuroception = (1.0 - emotion_crit) * 0.5 + social_eng * 0.3 + 0.2

        result = self.step(
            threat=emotion_crit,
            novelty=event.data.get("info_gain_reward", 0.0),
            urgency=event.data.get("urgency", 0.0),
            safety_signal=1.0 - emotion_crit,
            social_engagement=social_eng,
            neuroception_safety=neuroception,
        )
        state["ans_sympathetic"] = result["sympathetic_tone"]
        state["ans_parasympathetic"] = result["parasympathetic_tone"]
        state["ans_hrv"] = result["hrv"]
        state["ans_polyvagal_state"] = result["polyvagal_state"]
        state["ans_polyvagal_level"] = result["polyvagal_level"]
        state["heart_rate"] = result["heart_rate"]
        state["blood_pressure"] = result["blood_pressure"]
        return result

    def get_summary(self) -> dict:
        """获取自主神经系统摘要"""
        return {
            'sympathetic_tone': self.state.sympathetic_tone,
            'parasympathetic_tone': self.state.parasympathetic_tone,
            'hrv': self.state.hrv,
            'heart_rate': self.state.heart_rate,
            'blood_pressure': self.state.blood_pressure,
            'polyvagal_state': self.state.polyvagal_state,
            'autonomic_balance': self.state.parasympathetic_tone - self.state.sympathetic_tone,
            'step_count': self.step_count,
        }


def create_autonomic_nervous_system(**kwargs) -> AutonomicNervousSystem:
    """工厂函数: 创建自主神经系统"""
    return AutonomicNervousSystem(**kwargs)


__all__ = [
    'ANSState',
    'SympatheticBranch',
    'ParasympatheticBranch',
    'BaroreceptorReflex',
    'PolyvagalSystem',
    'AutonomicNervousSystem',
    'create_autonomic_nervous_system',
]
