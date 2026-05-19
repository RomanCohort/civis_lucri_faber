"""
稳态调节系统 (Allostatic Regulation)

模拟预测性稳态调节：
1. 预测性调节器 (Predictive Regulator) - 基于趋势的预判调节
2. 负荷累积器 (Load Accumulator) - 稳态负荷追踪
3. 状态选择器 (Regime Selector) - 操作模式分类
4. 稳态调节器 (Allostatic Regulation) - 元调节层

与稳态(Homeostasis)的区别:
- 稳态: 反应式, 固定设定点
- 异稳态(Allostasis): 预测性, 动态设定点, 基于预期需求提前调节

生物参考文献:
- Sterling & Eyer (1988): Allostasis概念
- McEwen & Wingfield (2003): 稳态超载
- Seeman et al. (1997): 稳态负荷量化
- Sterling (2012): 预测性调节模型
- Ulrich-Lai & Herman (2009): HPA轴恢复动力学
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
class AllostaticRegime:
    """稳态操作模式"""
    name: str                           # 模式名
    metabolic_setpoint: float           # 代谢率设定点
    sympathetic_setpoint: float         # 交感张力设定点
    cortisol_setpoint: float            # 皮质醇设定点
    cognitive_load_setpoint: float      # 认知负荷设定点


@dataclass
class AllostaticState:
    """稳态调节状态"""
    current_regime: str = "rest"            # 当前模式
    predictive_adjustment: float = 0.0      # 预测性调节幅度 [0,1]
    load: float = 0.0                       # 累积稳态负荷 [0,1]
    regulatory_capacity: float = 0.8        # 调节能力 [0,1]
    prediction_error: float = 0.0           # 预测误差
    recovery_rate: float = 0.5              # 当前恢复速率 [0,1]
    is_overloaded: bool = False             # 是否稳态超载


# ============ 操作模式定义 ============

REGIMES = {
    "rest": AllostaticRegime(
        name="rest",
        metabolic_setpoint=0.2,
        sympathetic_setpoint=0.2,
        cortisol_setpoint=0.2,
        cognitive_load_setpoint=0.2,
    ),
    "active": AllostaticRegime(
        name="active",
        metabolic_setpoint=0.5,
        sympathetic_setpoint=0.4,
        cortisol_setpoint=0.3,
        cognitive_load_setpoint=0.5,
    ),
    "stress": AllostaticRegime(
        name="stress",
        metabolic_setpoint=0.7,
        sympathetic_setpoint=0.7,
        cortisol_setpoint=0.6,
        cognitive_load_setpoint=0.7,
    ),
    "recovery": AllostaticRegime(
        name="recovery",
        metabolic_setpoint=0.3,
        sympathetic_setpoint=0.3,
        cortisol_setpoint=0.3,
        cognitive_load_setpoint=0.3,
    ),
}


# ============ 预测性调节器 ============

class PredictiveRegulator(nn.Module):
    """
    预测性调节器

    基于近期状态趋势预测下一步, 并提前调节:
    - 如果代谢支出趋势上升 -> 提前增加能量分配
    - 如果应激指标上升 -> 提前增加副交感神经张力
    - 如果疲劳累积 -> 提前减少认知负荷

    参考: Sterling (2012) "Allostasis: A model of predictive regulation"
    """

    def __init__(self, prediction_alpha: float = 0.3):
        super().__init__()
        self.alpha = prediction_alpha  # EMA平滑系数
        self.history = deque(maxlen=20)  # 最近20步历史
        self.predictions = {}  # 各变量的预测值

    def forward(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """
        基于历史趋势预测下一步状态

        使用EMA预测: predicted = alpha * current + (1-alpha) * previous_prediction
        """
        self.history.append(dict(current_state))

        predictions = {}
        for key, value in current_state.items():
            if not isinstance(value, (int, float)):
                continue
            prev = self.predictions.get(key, value)
            predicted = self.alpha * value + (1 - self.alpha) * prev
            self.predictions[key] = predicted
            predictions[key] = predicted

        # 计算预测性调节量
        adjustments = self._compute_adjustments(current_state, predictions)
        return {
            'predictions': predictions,
            'adjustments': adjustments,
        }

    def compute_prediction_error(self, predicted: Dict[str, float],
                                  actual: Dict[str, float]) -> float:
        """计算预测误差"""
        errors = []
        for key in predicted:
            if key in actual and isinstance(actual[key], (int, float)):
                errors.append(abs(predicted[key] - actual[key]))
        return float(np.mean(errors)) if errors else 0.0

    def _compute_adjustments(self, current: Dict[str, float],
                              predicted: Dict[str, float]) -> Dict[str, float]:
        """
        基于趋势预测计算提前调节量

        核心异稳态逻辑: 不等偏差出现, 而是预测并提前修正
        """
        adjustments = {}

        # 如果代谢趋势上升 -> 提前增加能量
        metabolic_trend = predicted.get('energy_demand', 0.3) - current.get('energy_demand', 0.3)
        if metabolic_trend > 0.05:
            adjustments['energy_allocation'] = min(0.2, metabolic_trend * 2)

        # 如果应激趋势上升 -> 提前增强副交感
        stress_trend = predicted.get('cortisol', 0.3) - current.get('cortisol', 0.3)
        if stress_trend > 0.05:
            adjustments['parasympathetic_boost'] = min(0.15, stress_trend * 1.5)

        # 如果疲劳趋势上升 -> 提前减少认知负荷
        fatigue_trend = predicted.get('fatigue', 0.3) - current.get('fatigue', 0.3)
        if fatigue_trend > 0.05:
            adjustments['cognitive_reduction'] = min(0.2, fatigue_trend * 2)

        # 如果炎症趋势上升 -> 提前激活修复
        inflammation_trend = (predicted.get('neuroinflammation', 0.0)
                              - current.get('neuroinflammation', 0.0))
        if inflammation_trend > 0.03:
            adjustments['repair_boost'] = min(0.1, inflammation_trend * 2)

        return adjustments


# ============ 负荷累积器 ============

class LoadAccumulator(nn.Module):
    """
    稳态负荷累积器

    追踪各应激介质偏离最优范围的累积量:
    - 皮质醇偏离
    - 交感张力偏离
    - 炎症偏离
    - 能量赤字

    参考: Seeman et al. (1997) - 稳态负荷量化方法
    """

    def __init__(self, base_recovery_rate: float = 0.005):
        super().__init__()
        # 各介质的最优范围 (中点, 容差)
        self.optimal_ranges = {
            'cortisol': (0.3, 0.15),       # 皮质醇中位点0.3, 容差±0.15
            'sympathetic_tone': (0.3, 0.15), # 交感中位点0.3
            'neuroinflammation': (0.1, 0.1), # 炎症中位点0.1
            'energy_deficit': (0.0, 0.2),    # 能量赤字理想为0
        }
        # 各介质的权重
        self.mediator_weights = nn.Parameter(torch.tensor([0.35, 0.25, 0.25, 0.15]))
        self.base_recovery_rate = base_recovery_rate
        self.load = 0.0

    def forward(self, mediators: Dict[str, float]) -> float:
        """
        计算稳态负荷变化

        load_delta = sum(w_i * max(0, |mediator_i - midpoint_i| - tolerance_i))
        """
        deviations = []
        for i, (key, (midpoint, tolerance)) in enumerate(self.optimal_ranges.items()):
            value = mediators.get(key, midpoint)
            deviation = max(0.0, abs(value - midpoint) - tolerance)
            deviations.append(deviation)

        weights = F.softmax(self.mediator_weights, dim=0)
        delta_load = float(sum(w * d for w, d in zip(weights.tolist(), deviations)))
        delta_load *= 0.01  # 缩放到合理范围

        self.load = float(np.clip(self.load + delta_load, 0.0, 1.0))
        return self.load

    def recover(self, recovery_signal: float):
        """
        恢复: 稳态负荷在真正休息时减少

        effective_recovery = base_rate * sleep_factor * social_factor * (1 - 0.5*load)
        高负荷会减缓恢复 (正反馈环路)

        参考: Ulrich-Lai & Herman (2009)
        """
        effective_rate = self.base_recovery_rate * recovery_signal * (1 - 0.5 * self.load)
        self.load = float(np.clip(self.load - effective_rate, 0.0, 1.0))


# ============ 模式选择器 ============

class RegimeSelector(nn.Module):
    """
    操作模式选择器

    根据当前状态将系统分类为4种操作模式:
    - rest: 低活动, 低应激, 高能量
    - active: 中高活动, 低中应激
    - stress: 高威胁, 高皮质醇, 高交感
    - recovery: 应激后恢复, 介质逐渐下降

    增强: setpoints根据慢性负荷漂移 (allostatic drift)
    """

    def __init__(self, transition_inertia: float = 0.3):
        super().__init__()
        self.transition_inertia = transition_inertia
        self.current_regime = "rest"
        self.regime_scores = {"rest": 0.0, "active": 0.0, "stress": 0.0, "recovery": 0.0}
        self.transition_net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1),
        )
        # 慢性负荷驱动的setpoint漂移 (accumulated over many steps)
        self._allostatic_drift = {
            'cortisol': 0.0,       # 正=皮质醇设定点上移 (慢性应激)
            'sympathetic': 0.0,    # 正=交感设定点上移
            'metabolic': 0.0,      # 正=代谢设定点上移
            'cognitive': 0.0,      # 正=认知负荷设定点上移
        }

    def forward(self, state: Dict[str, float]) -> Tuple[str, AllostaticRegime]:
        """
        分类当前状态到操作模式 (含setpoint漂移)

        慢性负荷 → 设定点漂移: 持续高应激 → cortisol设定点逐渐升高
        (模拟慢性应激下的HPA轴重设)
        """
        neural_activity = state.get('neural_activity', 0.3)
        cortisol = state.get('cortisol', 0.3)
        sympathetic = state.get('sympathetic_tone', 0.3)
        fatigue = state.get('fatigue', 0.3)
        energy_deficit = state.get('energy_deficit', 0.0)

        # 更新setpoint漂移 (缓慢累积)
        drift_rate = 0.001
        recovery_rate = 0.0005
        # 慢性应激 → cortisol/sympathetic设定点上移
        if cortisol > 0.5:
            self._allostatic_drift['cortisol'] += drift_rate * (cortisol - 0.5)
        else:
            self._allostatic_drift['cortisol'] -= recovery_rate
        if sympathetic > 0.4:
            self._allostatic_drift['sympathetic'] += drift_rate * (sympathetic - 0.4)
        else:
            self._allostatic_drift['sympathetic'] -= recovery_rate
        # 限制漂移范围
        for key in self._allostatic_drift:
            self._allostatic_drift[key] = float(np.clip(
                self._allostatic_drift[key], -0.15, 0.15
            ))

        x = torch.tensor([
            neural_activity, cortisol, sympathetic, fatigue, energy_deficit
        ], dtype=torch.float32)

        scores = self.transition_net(x)
        regime_names = ["rest", "active", "stress", "recovery"]

        # 惯性平滑: 不立即切换, 而是加权平均
        for i, name in enumerate(regime_names):
            self.regime_scores[name] = (
                self.transition_inertia * self.regime_scores[name]
                + (1 - self.transition_inertia) * scores[i].item()
            )

        # 选择得分最高的模式
        best_regime = max(self.regime_scores, key=self.regime_scores.get)
        self.current_regime = best_regime

        # 应用setpoint漂移到regime定义
        base_regime = REGIMES[best_regime]
        drifted_regime = AllostaticRegime(
            name=base_regime.name,
            metabolic_setpoint=base_regime.metabolic_setpoint + self._allostatic_drift['metabolic'],
            sympathetic_setpoint=base_regime.sympathetic_setpoint + self._allostatic_drift['sympathetic'],
            cortisol_setpoint=base_regime.cortisol_setpoint + self._allostatic_drift['cortisol'],
            cognitive_load_setpoint=base_regime.cognitive_load_setpoint + self._allostatic_drift['cognitive'],
        )

        return best_regime, drifted_regime


# ============ 稳态调节系统 (聚合器) ============

class AllostaticRegulation(nn.Module):
    """
    稳态调节系统 - 元调节层

    位于所有其他自调节系统之上:
    读取所有内部状态, 产生预测性调节建议

    功能:
    - 预测下一步状态趋势
    - 追踪累积稳态负荷
    - 选择当前操作模式
    - 在超载时触发保护模式

    参考:
    - Sterling & Eyer (1988): Allostasis
    - McEwen & Wingfield (2003): 稳态超载理论
    """

    def __init__(self, overload_threshold: float = 0.8,
                 load_recovery_rate: float = 0.005,
                 event_bus=None):
        super().__init__()
        self.predictive = PredictiveRegulator()
        self.load_accumulator = LoadAccumulator(base_recovery_rate=load_recovery_rate)
        self.regime_selector = RegimeSelector()
        self.state = AllostaticState()
        self.overload_threshold = overload_threshold
        self.step_count = 0
        self.prev_predictions = {}
        self.event_bus = event_bus

        # Event-driven registration
        if self.event_bus is not None:
            self.event_bus.subscribe(
                "neural_regulation",
                self.on_neural_regulation,
                priority=3,
                name="allostatic",
            )

    def step(self, internal_state: Dict[str, float]) -> Dict[str, Any]:
        """
        执行一个稳态调节步

        读取全部内部状态, 输出调节建议

        Args:
            internal_state: agent的完整内部状态字典
        """
        self.step_count += 1

        # 1. 提取关键变量
        key_state = {
            'cortisol': internal_state.get('cortisol', 0.3),
            'sympathetic_tone': internal_state.get('ans_sympathetic', 0.3),
            'neuroinflammation': internal_state.get('neuroinflammation', 0.0),
            'energy_deficit': max(0.0, 1.0 - internal_state.get('balance', 100.0) / 100.0),
            'neural_activity': internal_state.get('neural_activity', 0.3),
            'fatigue': internal_state.get('fatigue', 0.3),
            'energy_demand': internal_state.get('energy_demand', 0.3),
        }

        # 2. 预测性调节
        pred_result = self.predictive.forward(key_state)

        # 计算上一步的预测误差
        prediction_error = self.predictive.compute_prediction_error(
            self.prev_predictions, key_state
        ) if self.prev_predictions else 0.0
        self.prev_predictions = pred_result['predictions']

        # 3. 稳态负荷累积
        mediator_values = {
            'cortisol': key_state['cortisol'],
            'sympathetic_tone': key_state['sympathetic_tone'],
            'neuroinflammation': key_state['neuroinflammation'],
            'energy_deficit': key_state['energy_deficit'],
        }
        load = self.load_accumulator.forward(mediator_values)

        # 4. 恢复 (在睡眠或低应激时)
        is_recovering = (key_state.get('fatigue', 0.3) < 0.2
                         or internal_state.get('ans_polyvagal_state') == 'ventral_vagal')
        if is_recovering:
            self.load_accumulator.recover(recovery_signal=0.8)

        # 5. 模式选择
        regime_name, regime = self.regime_selector.forward(key_state)

        # 6. 调节能力计算
        regulatory_capacity = float(np.clip(
            0.8 - 0.4 * load + 0.2 * (1.0 if is_recovering else 0.0),
            0.1, 1.0
        ))

        # 7. 超载检测
        is_overloaded = load > self.overload_threshold

        # 8. 恢复速率
        recovery_rate = float(np.clip(
            0.5 * regulatory_capacity * (1.5 if is_recovering else 0.5),
            0.1, 1.0
        ))

        # 9. 更新状态
        self.state = AllostaticState(
            current_regime=regime_name,
            predictive_adjustment=max(pred_result['adjustments'].values())
            if pred_result['adjustments'] else 0.0,
            load=load,
            regulatory_capacity=regulatory_capacity,
            prediction_error=prediction_error,
            recovery_rate=recovery_rate,
            is_overloaded=is_overloaded,
        )

        return {
            'current_regime': regime_name,
            'regime_setpoints': {
                'metabolic': regime.metabolic_setpoint,
                'sympathetic': regime.sympathetic_setpoint,
                'cortisol': regime.cortisol_setpoint,
                'cognitive_load': regime.cognitive_load_setpoint,
            },
            'predictive_adjustments': pred_result['adjustments'],
            'load': load,
            'regulatory_capacity': regulatory_capacity,
            'prediction_error': prediction_error,
            'recovery_rate': recovery_rate,
            'is_overloaded': is_overloaded,
        }

    def on_neural_regulation(self, event) -> Dict[str, Any]:
        """Event handler for NEURAL_REGULATION events (priority=3, depends on all above)."""
        state = event.data["internal_state"]
        result = self.step(state)
        state["allostatic_regime"] = result["current_regime"]
        state["allostatic_load"] = result["load"]
        state["regulatory_capacity"] = result["regulatory_capacity"]
        state["predictive_adjustments"] = result["predictive_adjustments"]
        if result["is_overloaded"]:
            state["defensive_mode"] = True
        return result

    def is_overloaded(self) -> bool:
        return self.state.is_overloaded

    def force_recovery(self) -> Dict[str, Any]:
        """强制恢复模式 (外部触发)"""
        self.load_accumulator.recover(recovery_signal=1.0)
        return {
            'action': 'force_recovery',
            'load_after': self.state.load,
            'regime': self.state.current_regime,
        }

    def get_summary(self) -> Dict:
        return {
            'regime': self.state.current_regime,
            'load': self.state.load,
            'regulatory_capacity': self.state.regulatory_capacity,
            'prediction_error': self.state.prediction_error,
            'is_overloaded': self.state.is_overloaded,
            'recovery_rate': self.state.recovery_rate,
            'step_count': self.step_count,
        }


def create_allostatic_regulation(**kwargs) -> AllostaticRegulation:
    """工厂函数: 创建稳态调节系统"""
    return AllostaticRegulation(**kwargs)


__all__ = [
    'AllostaticRegime',
    'AllostaticState',
    'REGIMES',
    'PredictiveRegulator',
    'LoadAccumulator',
    'RegimeSelector',
    'AllostaticRegulation',
    'create_allostatic_regulation',
]
