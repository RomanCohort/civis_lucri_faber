"""
胶质细胞系统 (Glial System)

模拟三种胶质细胞的大脑维护功能：
1. 星形胶质细胞 (Astrocytes) - 三方突触、K+缓冲、乳酸穿梭、类淋巴清除
2. 小胶质细胞 (Microglia) - 免疫监视、突触修剪、炎症响应
3. 少突胶质细胞 (Oligodendrocytes) - 髓鞘化、传导速度优化

生物参考文献:
- Araque et al. (1999): 三方突触
- Kofuji & Newman (2004): 星形胶质细胞K+缓冲
- Pellerin & Magistretti (1994): 星形胶质细胞-神经元乳酸穿梭 (ANLS)
- Iliff et al. (2012): 类淋巴系统
- Schafer et al. (2012): 补体介导的突触修剪
- Gibson et al. (2014): 活动依赖性髓鞘化
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
class AstrocyteState:
    """星形胶质细胞状态"""
    calcium_wave_amplitude: float = 0.1   # 钙波振幅 [0,1]
    potassium_buffer_level: float = 0.8   # K+缓冲能力 [0,1]
    lactate_shuttle_rate: float = 0.3     # 乳酸穿梭速率 [0,1]
    glymphatic_clearance_rate: float = 0.1  # 类淋巴清除率 [0,1]
    gliotransmitter_release: float = 0.1  # 胶质递质释放 [0,1]
    waste_level: float = 0.0              # 代谢废物水平 [0,1]


@dataclass
class MicrogliaState:
    """小胶质细胞状态"""
    activation_state: str = "resting"       # resting / M1 / M2
    surveillance_rate: float = 0.7          # 巡视速率 [0,1]
    pruning_rate: float = 0.05              # 修剪速率 [0,1]
    cytokine_il1b: float = 0.1              # IL-1β水平 [0,1]
    cytokine_tnfa: float = 0.1              # TNF-α水平 [0,1]
    neuroinflammation: float = 0.0          # 神经炎症水平 [0,1]
    synapses_pruned: int = 0                # 本轮修剪突触数


@dataclass
class OligodendrocyteState:
    """少突胶质细胞状态"""
    myelination_level: float = 0.3          # 总体髓鞘化水平 [0,1]
    conduction_speed_boost: float = 0.3     # 传导速度提升 [0,1]
    energy_cost: float = 0.05               # 髓鞘化能耗 [0,1]
    plasticity_rate: float = 0.01           # 髓鞘可塑性速率 [0,1]


@dataclass
class GlialState:
    """胶质系统总状态"""
    astrocyte: AstrocyteState = field(default_factory=AstrocyteState)
    microglia: MicrogliaState = field(default_factory=MicrogliaState)
    oligodendrocyte: OligodendrocyteState = field(default_factory=OligodendrocyteState)
    overall_brain_health: float = 0.8       # 大脑总体健康 [0,1]


# ============ 星形胶质细胞系统 ============

class AstrocyteSystem(nn.Module):
    """
    星形胶质细胞系统

    功能:
    1. 三方突触: 检测突触活动，释放胶质递质(D-丝氨酸)调节NMDA受体
    2. K+缓冲: 清除胞外K+离子，防止过度兴奋
    3. 乳酸穿梭: 为活跃神经元提供能量 (ANLS模型)
    4. 类淋巴清除: 睡眠期间清除代谢废物

    参考:
    - Araque et al. (1999): 三方突触概念
    - Pellerin & Magistretti (1994): ANLS
    - Iliff et al. (2012): 类淋巴系统
    """

    def __init__(self):
        super().__init__()
        self.state = AstrocyteState()
        self.waste_accumulation_rate = 0.01
        self.pathology_threshold = 0.8  # K+病理阈值

    def forward(self, neural_activity: float, extracellular_k: float,
                energy_demand: float, is_sleeping: bool,
                sleep_stage: str = "awake") -> Dict[str, Any]:
        """星形胶质细胞综合更新"""

        # 1. 三方突触: 钙波检测突触活动
        calcium_result = self._process_calcium_wave(neural_activity)

        # 2. K+缓冲
        k_result = self._buffer_potassium(extracellular_k)

        # 3. 乳酸穿梭 (ANLS)
        lactate_result = self._shuttle_lactate(energy_demand)

        # 4. 类淋巴清除
        glymphatic_result = self._glymphatic_clearance(is_sleeping, sleep_stage)

        # 5. 废物积累
        self.state.waste_level = float(np.clip(
            self.state.waste_level + self.waste_accumulation_rate * neural_activity
            - glymphatic_result['clearance'],
            0.0, 1.0
        ))

        # 6. 胶质递质释放 (D-丝氨酸增强NMDA功能)
        gliotransmitter = 0.3 * self.state.calcium_wave_amplitude
        self.state.gliotransmitter_release = float(np.clip(gliotransmitter, 0.0, 1.0))

        # 更新状态
        self.state.potassium_buffer_level = k_result['buffer_capacity']
        self.state.lactate_shuttle_rate = lactate_result['rate']
        self.state.glymphatic_clearance_rate = glymphatic_result['rate']

        return {
            'calcium_wave_amplitude': self.state.calcium_wave_amplitude,
            'potassium_buffer_level': self.state.potassium_buffer_level,
            'lactate_shuttle_rate': self.state.lactate_shuttle_rate,
            'glymphatic_clearance_rate': self.state.glymphatic_clearance_rate,
            'gliotransmitter_release': self.state.gliotransmitter_release,
            'waste_level': self.state.waste_level,
            'k_danger': k_result['is_danger'],
            'effective_energy_boost': lactate_result['energy_boost'],
        }

    def _process_calcium_wave(self, activity: float) -> Dict[str, float]:
        """
        钙波处理

        高活动 -> 钙波振幅增加 -> 胶质递质释放
        参考: Araque et al. (1999)
        """
        self.state.calcium_wave_amplitude = float(np.clip(
            self.state.calcium_wave_amplitude * 0.9 + 0.1 * activity,
            0.0, 1.0
        ))
        return {'amplitude': self.state.calcium_wave_amplitude}

    def _buffer_potassium(self, k_level: float) -> Dict[str, Any]:
        """
        K+缓冲

        神经元放电释放K+到胞外, 星形胶质细胞清除
        K+ > 0.8 = 过度兴奋危险 (癫痫样活动)

        参考: Kofuji & Newman (2004)
        """
        buffer_capacity = self.state.potassium_buffer_level
        k_cleared = buffer_capacity * k_level * 0.5
        k_remaining = max(0.0, k_level - k_cleared)

        # 缓冲能力恢复 (缓慢)
        self.state.potassium_buffer_level = float(np.clip(
            self.state.potassium_buffer_level * 0.98 + 0.02 * 0.8,
            0.3, 1.0
        ))

        return {
            'k_cleared': k_cleared,
            'k_remaining': k_remaining,
            'buffer_capacity': self.state.potassium_buffer_level,
            'is_danger': k_remaining > self.pathology_threshold,
        }

    def _shuttle_lactate(self, energy_demand: float) -> Dict[str, float]:
        """
        星形胶质细胞-神经元乳酸穿梭 (ANLS)

        活跃神经元需要能量, 星形胶质细胞摄取葡萄糖,
        转化为乳酸, 输送给神经元

        参考: Pellerin & Magistretti (1994)
        """
        rate = float(np.clip(energy_demand * 1.2, 0.0, 1.0))
        self.state.lactate_shuttle_rate = rate

        # 有效能量提升: 乳酸穿梭增加代谢预算
        energy_boost = 0.3 * rate
        return {
            'rate': rate,
            'energy_boost': energy_boost,
        }

    def _glymphatic_clearance(self, is_sleeping: bool,
                               sleep_stage: str) -> Dict[str, float]:
        """
        类淋巴系统清除

        睡眠期间(尤其NREM3深睡), 星形胶质细胞缩小~60%,
        允许脑脊液冲洗废物 (β-淀粉样蛋白等)

        参考: Iliff et al. (2012)
        """
        # 睡眠阶段因子
        stage_factors = {
            "awake": 0.1,
            "NREM1": 0.3,
            "NREM2": 0.5,
            "NREM3": 2.0,   # 深睡清除效率最高
            "REM": 0.4,
        }
        factor = stage_factors.get(sleep_stage, 0.1)

        rate = 0.05 * factor
        self.state.glymphatic_clearance_rate = float(np.clip(rate, 0.0, 1.0))

        clearance = rate * 0.5  # 每步清除量
        return {
            'rate': self.state.glymphatic_clearance_rate,
            'clearance': clearance,
        }


# ============ 小胶质细胞系统 ============

class MicrogliaSystem(nn.Module):
    """
    小胶质细胞系统

    功能:
    1. 免疫监视: 持续扫描受损神经元、病原体、碎片
    2. 突触修剪: 补体介导的弱突触清除 (C1q标记 -> 小胶质细胞吞噬)
    3. 炎症响应: M1/M2极化, 释放促炎/抗炎细胞因子
    4. 神经炎症: 慢性激活导致附带突触损伤

    参考:
    - Schafer et al. (2012): 补体介导突触修剪
    - Cherry et al. (2014): 小胶质细胞激活状态
    """

    def __init__(self, base_pruning_rate: float = 0.05,
                 max_pruning_per_step: float = 0.05):
        super().__init__()
        self.state = MicrogliaState()
        self.base_pruning_rate = base_pruning_rate
        self.max_pruning_per_step = max_pruning_per_step
        self.inflammation_history = deque(maxlen=100)

    def forward(self, damage_signal: float, stress_level: float,
                synaptic_health: float,
                pathogen_signal: float = 0.0,
                cytokine_boost: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """小胶质细胞综合更新

        Args:
            damage_signal: 神经损伤信号
            stress_level: 应激水平
            synaptic_health: 突触健康度
            pathogen_signal: 病原体TLR信号 (来自PathogenTriggeredInflammationEngine)
            cytokine_boost: 细胞因子增强 {IL-1beta: val, TNF-alpha: val} (来自病原体引擎)
        """

        # 增强damage_signal: 病原体TLR信号叠加
        enhanced_damage = damage_signal + pathogen_signal * 0.3

        # 1. 激活状态转换
        prev_state = self.state.activation_state
        self._transition_state(enhanced_damage, stress_level)

        # 2. 巡视速率 (静息态最高, 激活态降低)
        if self.state.activation_state == "resting":
            self.state.surveillance_rate = float(np.clip(
                self.state.surveillance_rate * 0.95 + 0.05 * 0.8, 0.3, 1.0
            ))
        else:
            self.state.surveillance_rate = float(np.clip(
                self.state.surveillance_rate * 0.95 + 0.05 * 0.4, 0.1, 0.6
            ))

        # 3. 释放细胞因子
        cytokine_result = self._release_cytokines(stress_level, enhanced_damage)

        # 3b. 叠加病原体引擎的细胞因子增强
        if cytokine_boost:
            il1b_boost = float(cytokine_boost.get("IL-1beta", 0.0))
            tnfa_boost = float(cytokine_boost.get("TNF-alpha", 0.0))
            self.state.cytokine_il1b = float(np.clip(
                self.state.cytokine_il1b + il1b_boost * 0.1, 0.0, 1.0
            ))
            self.state.cytokine_tnfa = float(np.clip(
                self.state.cytokine_tnfa + tnfa_boost * 0.1, 0.0, 1.0
            ))

        # 4. 计算神经炎症
        self.state.neuroinflammation = float(np.clip(
            0.7 * self.state.cytokine_il1b + 0.3 * self.state.cytokine_tnfa,
            0.0, 1.0
        ))
        self.inflammation_history.append(self.state.neuroinflammation)

        # 5. 突触修剪
        pruning_rate = self.base_pruning_rate * (1.0 + self.state.neuroinflammation)
        pruning_rate = min(pruning_rate, self.max_pruning_per_step)
        self.state.pruning_rate = pruning_rate

        return {
            'activation_state': self.state.activation_state,
            'surveillance_rate': self.state.surveillance_rate,
            'pruning_rate': self.state.pruning_rate,
            'cytokine_il1b': self.state.cytokine_il1b,
            'cytokine_tnfa': self.state.cytokine_tnfa,
            'neuroinflammation': self.state.neuroinflammation,
            'state_changed': prev_state != self.state.activation_state,
        }

    def prune_synapses(self, synapse_weights: List[float],
                       threshold: float = 0.1) -> Tuple[List[float], int]:
        """
        补体介导突触修剪

        C1q标记弱突触 (weight < threshold), 小胶质细胞吞噬
        炎症加速修剪 (可能变为病理性)

        参考: Schafer et al. (2012)
        """
        pruned_count = 0
        effective_threshold = threshold * (1.0 + 0.5 * self.state.neuroinflammation)
        effective_threshold = min(effective_threshold, 0.3)  # 防止过度修剪

        pruned_weights = []
        for w in synapse_weights:
            if abs(w) < effective_threshold:
                # 修剪: 将弱突触权重置零
                pruned_weights.append(0.0)
                pruned_count += 1
            else:
                pruned_weights.append(w)

        self.state.synapses_pruned = pruned_count
        return pruned_weights, pruned_count

    def _release_cytokines(self, stress: float, damage: float) -> Dict[str, float]:
        """
        细胞因子释放

        M1状态: 释放促炎因子 IL-1β, TNF-α
        M2状态: 释放抗炎因子, 促进修复
        静息态: 基础低水平
        """
        if self.state.activation_state == "M1":
            il1b_target = 0.3 + 0.5 * max(stress, damage)
            tnfa_target = 0.2 + 0.4 * damage
        elif self.state.activation_state == "M2":
            il1b_target = 0.05
            tnfa_target = 0.05
        else:  # resting
            il1b_target = 0.1
            tnfa_target = 0.1

        # EMA更新
        self.state.cytokine_il1b = float(np.clip(
            0.9 * self.state.cytokine_il1b + 0.1 * il1b_target, 0.0, 1.0
        ))
        self.state.cytokine_tnfa = float(np.clip(
            0.9 * self.state.cytokine_tnfa + 0.1 * tnfa_target, 0.0, 1.0
        ))

        return {
            'il1b': self.state.cytokine_il1b,
            'tnfa': self.state.cytokine_tnfa,
        }

    def _transition_state(self, damage_signal: float, stress_level: float):
        """
        激活状态转换 (渐进式，带累积和滞后)

        使用累积变量而非硬阈值即时切换：
        - Resting→M1: 损伤信号持续高于阈值时逐渐累积
        - M1→M2: 损伤下降后累积恢复，而非立即切换
        - M2→Resting: 低损伤持续一段时间后逐渐回归

        参考: Cherry et al. (2014)
        """
        # 累积驱动变量 (EMA平滑, 防止状态闪烁)
        if not hasattr(self, '_activation_drive'):
            self._activation_drive = 0.0     # 正=趋向M1, 负=趋向resting
            self._resolution_drive = 0.0     # M1→M2的解析驱动

        # M1激活驱动: 损伤/应激的持续累积
        m1_signal = max(0.0, damage_signal - 0.4) + max(0.0, stress_level - 0.5) * 0.5
        self._activation_drive = 0.85 * self._activation_drive + 0.15 * m1_signal

        # M2解析驱动: 损伤下降后的恢复
        resolution_signal = max(0.0, 0.5 - damage_signal) * (1.0 - stress_level)
        self._resolution_drive = 0.88 * self._resolution_drive + 0.12 * resolution_signal

        # 状态转换 (累积超过阈值才切换, 带滞后)
        if self.state.activation_state == "resting":
            if self._activation_drive > 0.3:
                self.state.activation_state = "M1"
                self._resolution_drive = 0.0  # 重置解析驱动
        elif self.state.activation_state == "M1":
            # M1→M2: 需要损伤下降 + 解析驱动足够
            if self._activation_drive < 0.15 and self._resolution_drive > 0.25:
                self.state.activation_state = "M2"
        elif self.state.activation_state == "M2":
            # M2→resting: 解析驱动充分 + 损伤极低
            if self._resolution_drive > 0.3 and damage_signal < 0.15:
                self.state.activation_state = "resting"
                self._activation_drive = 0.0  # 重置
            # M2→M1: 损伤重新升高
            if self._activation_drive > 0.35:
                self.state.activation_state = "M1"


# ============ 少突胶质细胞系统 ============

class OligodendrocyteSystem(nn.Module):
    """
    少突胶质细胞系统

    功能:
    1. 髓鞘化: 包裹轴突, 提高传导速度 (跳跃传导)
    2. 自适应髓鞘化: 高频使用的通路获得更多髓鞘 (更快)
    3. 髓鞘可塑性: 学习新技能触发相关回路的髓鞘化
    4. 能量代价: 髓鞘化昂贵, 仅在有能量预算时执行

    参考: Gibson et al. (2014) - 活动依赖性髓鞘化
    """

    def __init__(self, base_myelination_rate: float = 0.01,
                 max_myelination: float = 0.9):
        super().__init__()
        self.state = OligodendrocyteState()
        self.base_myelination_rate = base_myelination_rate
        self.max_myelination = max_myelination
        self.pathway_usage = {}  # 通路使用频率追踪
        self.pathway_myelin = {}  # 通路髓鞘水平

    def forward(self, pathway_activity: Dict[str, float],
                energy_budget: float) -> Dict[str, Any]:
        """少突胶质细胞综合更新"""

        # 1. 更新通路使用频率
        for pathway_id, activity in pathway_activity.items():
            if pathway_id not in self.pathway_usage:
                self.pathway_usage[pathway_id] = 0.0
            self.pathway_usage[pathway_id] = (
                0.95 * self.pathway_usage[pathway_id] + 0.05 * activity
            )

        # 2. 自适应髓鞘化 (仅在有能量时)
        myelination_updates = {}
        total_energy_cost = 0.0

        if energy_budget > 0.15:  # 最低能量阈值
            for pathway_id, usage in self.pathway_usage.items():
                if pathway_id not in self.pathway_myelin:
                    self.pathway_myelin[pathway_id] = 0.1

                # 高频使用的通路增加髓鞘
                if usage > 0.3:  # 使用频率超过阈值
                    delta = self.base_myelination_rate * usage * energy_budget
                    delta = min(delta, 0.01)  # 每步最大增量
                    self.pathway_myelin[pathway_id] = min(
                        self.max_myelination,
                        self.pathway_myelin[pathway_id] + delta
                    )
                    total_energy_cost += delta * 0.5  # 髓鞘化能耗
                    myelination_updates[pathway_id] = delta

        # 3. 总体髓鞘化水平
        if self.pathway_myelin:
            self.state.myelination_level = float(np.clip(
                np.mean(list(self.pathway_myelin.values())), 0.0, 1.0
            ))

        # 4. 传导速度提升: speed = base_speed * (1 + myelination)
        self.state.conduction_speed_boost = float(np.clip(
            self.state.myelination_level, 0.0, 1.0
        ))

        # 5. 能量消耗
        self.state.energy_cost = float(np.clip(total_energy_cost, 0.0, 0.1))

        # 6. 可塑性速率
        self.state.plasticity_rate = float(np.clip(
            self.base_myelination_rate * energy_budget * 2, 0.0, 0.05
        ))

        return {
            'myelination_level': self.state.myelination_level,
            'conduction_speed_boost': self.state.conduction_speed_boost,
            'energy_cost': self.state.energy_cost,
            'plasticity_rate': self.state.plasticity_rate,
            'pathways_myelinated': len(self.pathway_myelin),
            'myelination_updates': myelination_updates,
        }

    def get_conduction_speed(self, pathway_id: str) -> float:
        """获取特定通路的传导速度"""
        myelin = self.pathway_myelin.get(pathway_id, 0.1)
        return 1.0 + myelin  # base_speed * (1 + myelination_level)


# ============ 胶质系统 (聚合器) ============

class GlialSystem(nn.Module):
    """
    胶质系统 - 整合星形胶质细胞、小胶质细胞、少突胶质细胞

    功能:
    - 大脑维护与清洁 (类淋巴清除)
    - 突触修剪与优化 (小胶质细胞)
    - 能量供应增强 (乳酸穿梭)
    - 传导速度优化 (髓鞘化)
    - 神经炎症管理

    参考:
    - Araque et al. (1999): 三方突触
    - Iliff et al. (2012): 类淋巴系统
    - Schafer et al. (2012): 突触修剪
    """

    def __init__(self, pruning_rate: float = 0.05, event_bus=None):
        super().__init__()
        self.astrocyte = AstrocyteSystem()
        self.microglia = MicrogliaSystem(base_pruning_rate=pruning_rate)
        self.oligodendrocyte = OligodendrocyteSystem()
        self.state = GlialState()
        self.step_count = 0
        self.event_bus = event_bus

        # Event-driven registration
        if self.event_bus is not None:
            self.event_bus.subscribe(
                "neural_regulation",
                self.on_neural_regulation,
                priority=2,
                name="glial",
            )

    def step(self, neural_activity: float = 0.5,
             extracellular_k: float = 0.3,
             energy_demand: float = 0.3,
             is_sleeping: bool = False,
             sleep_stage: str = "awake",
             damage_signal: float = 0.0,
             stress_level: float = 0.0,
             pathway_activities: Dict[str, float] = None,
             energy_budget: float = 0.3,
             pathogen_signal: float = 0.0,
             cytokine_boost: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        执行一个胶质系统调节步

        Args:
            neural_activity: 神经活动水平 [0,1]
            extracellular_k: 胞外K+浓度 [0,1]
            energy_demand: 能量需求 [0,1]
            is_sleeping: 是否睡眠中
            sleep_stage: 睡眠阶段
            damage_signal: 损伤/稳态负荷信号 [0,1]
            stress_level: 应激水平 [0,1]
            pathway_activities: 各通路活动频率
            energy_budget: 能量预算 [0,1]
        """
        self.step_count += 1
        if pathway_activities is None:
            pathway_activities = {"default": neural_activity}

        # 1. 星形胶质细胞更新
        astro_result = self.astrocyte(
            neural_activity, extracellular_k,
            energy_demand, is_sleeping, sleep_stage
        )

        # 2. 小胶质细胞更新
        micro_result = self.microglia(
            damage_signal, stress_level,
            1.0 - self.state.overall_brain_health,
            pathogen_signal=pathogen_signal,
            cytokine_boost=cytokine_boost,
        )

        # 3. 少突胶质细胞更新
        oligo_result = self.oligodendrocyte(
            pathway_activities, energy_budget
        )

        # 4. 计算大脑总体健康
        waste_penalty = self.state.astrocyte.waste_level * 0.3
        inflammation_penalty = micro_result['neuroinflammation'] * 0.4
        myelination_bonus = oligo_result['myelination_level'] * 0.2
        glymphatic_bonus = astro_result['glymphatic_clearance_rate'] * 0.1

        self.state.overall_brain_health = float(np.clip(
            0.8 - waste_penalty - inflammation_penalty
            + myelination_bonus + glymphatic_bonus,
            0.1, 1.0
        ))

        # 更新子系统状态
        self.state.astrocyte = self.astrocyte.state
        self.state.microglia = self.microglia.state
        self.state.oligodendrocyte = self.oligodendrocyte.state

        return {
            'waste_level': self.state.astrocyte.waste_level,
            'glymphatic_clearance': astro_result['glymphatic_clearance_rate'],
            'neuroinflammation': micro_result['neuroinflammation'],
            'pruning_rate': micro_result['pruning_rate'],
            'myelination_level': oligo_result['myelination_level'],
            'gliotransmitter_release': astro_result['gliotransmitter_release'],
            'brain_health': self.state.overall_brain_health,
            'effective_energy_boost': astro_result['effective_energy_boost'],
            'k_danger': astro_result['k_danger'],
            'astrocyte': {
                'calcium_wave': astro_result['calcium_wave_amplitude'],
                'lactate_shuttle': astro_result['lactate_shuttle_rate'],
                'waste': self.state.astrocyte.waste_level,
            },
            'microglia': {
                'activation': micro_result['activation_state'],
                'neuroinflammation': micro_result['neuroinflammation'],
                'pruning_rate': micro_result['pruning_rate'],
                'il1b': micro_result['cytokine_il1b'],
                'tnfa': micro_result['cytokine_tnfa'],
            },
            'oligodendrocyte': {
                'myelination': oligo_result['myelination_level'],
                'speed_boost': oligo_result['conduction_speed_boost'],
                'energy_cost': oligo_result['energy_cost'],
            },
        }

    def on_neural_regulation(self, event) -> Dict[str, Any]:
        """Event handler for NEURAL_REGULATION events (priority=2, depends on HPA)."""
        state = event.data["internal_state"]
        info_gain_reward = event.data.get("info_gain_reward", 0.0)
        thermo_status = event.data.get("thermo_status", "ACTIVE")

        neural_act = abs(info_gain_reward)
        is_sleeping = thermo_status == "HIBERNATE"
        damage_signal = 1.0 if state.get("stress_type") == "chronic" else 0.0

        result = self.step(
            neural_activity=neural_act,
            extracellular_k=state.get("extracellular_k", 0.3),
            energy_demand=state.get("energy_demand", 0.3),
            is_sleeping=is_sleeping,
            sleep_stage="NREM3" if is_sleeping else "awake",
            damage_signal=damage_signal,
            stress_level=state.get("cortisol", 0.3),
            energy_budget=state.get("resource_budget", 0.3),
            pathogen_signal=float(event.data.get("pathogen_signal", 0.0)),
            cytokine_boost=event.data.get("cytokine_boost"),
        )
        state["brain_waste"] = result["waste_level"]
        state["glymphatic_clearance"] = result["glymphatic_clearance"]
        state["neuroinflammation"] = result["neuroinflammation"]
        state["myelination_level"] = result["myelination_level"]
        state["glial_gliotransmitter"] = result["gliotransmitter_release"]
        state["brain_health"] = result["brain_health"]
        return result

    def get_summary(self) -> Dict:
        """获取胶质系统摘要"""
        return {
            'brain_health': self.state.overall_brain_health,
            'waste_level': self.state.astrocyte.waste_level,
            'neuroinflammation': self.state.microglia.neuroinflammation,
            'microglia_state': self.state.microglia.activation_state,
            'myelination': self.state.oligodendrocyte.myelination_level,
            'glymphatic_rate': self.state.astrocyte.glymphatic_clearance_rate,
            'step_count': self.step_count,
        }


def create_glial_system(**kwargs) -> GlialSystem:
    """工厂函数: 创建胶质系统"""
    return GlialSystem(**kwargs)


__all__ = [
    'AstrocyteState',
    'MicrogliaState',
    'OligodendrocyteState',
    'GlialState',
    'AstrocyteSystem',
    'MicrogliaSystem',
    'OligodendrocyteSystem',
    'GlialSystem',
    'create_glial_system',
]
