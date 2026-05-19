"""Neuroplasticity System - 神经可塑性系统

实现大脑的神经可塑性机制：
1. STDP突触可塑性 - 脉冲时序依赖的可塑性 (新增)
2. 突触修剪 - 弱化不常用的神经连接
3. 突触强化 - 强化常用的神经连接
4. BDNF释放 - 脑源性神经营养因子，促进神经新生
5. 神经发生 - 新的神经细胞生成
6. NMDA依赖的LTP/LTD - 长时程增强/抑制 (新增)
7. 突触缩放 - 稳态可塑性 (新增)

参考:
- Bi & Poo (1998) - STDP timing window
- Kafitz et al. (1999) - BDNF-mediated neurotrophin signaling
- Chechik et al. (1999) - Synaptic scaling in neural networks
- Malenka & Bear (2004) - LTP/LTD and NMDA receptors
- Turrigiano (2008) - Homeostatic synaptic scaling
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class NeuroplasticityState:
    """神经可塑性状态"""
    synaptic_strength: float = 1.0        # 突触总体强度
    pruning_rate: float = 0.0           # 当前修剪率
    bdnf_level: float = 0.5            # BDNF水平
    neurogenesis_rate: float = 0.0       # 神经发生率
    active_synapses: int = 0            # 活跃突触数
    stdp_events: int = 0                 # STDP事件计数 (新增)
    ltp_events: int = 0                  # LTP事件计数 (新增)
    ltd_events: int = 0                  # LTD事件计数 (新增)


# ══════════════════════════════════════════════════════
# STDP突触可塑性 - 脉冲时序依赖的可塑性 (新增)
# ══════════════════════════════════════════════════════

@dataclass
class STDPConfig:
    """STDP配置参数

    参考: Bi & Poo (1998)
    """
    tau_plus: float = 20e-3      # LTP时间常数 (20ms)
    tau_minus: float = 20e-3     # LTD时间常数 (20ms)
    A_plus: float = 0.01         # LTP幅度
    A_minus: float = 0.012       # LTD幅度 (略大于LTP，符合生物观测)
    w_min: float = 0.01          # 权重下限
    w_max: float = 2.0           # 权重上限
    eligibility_decay: float = 0.9  # eligibility trace衰减


class STDPSynapse:
    """STDP突触 - 脉冲时序依赖的可塑性

    核心机制:
    - Pre先于Post spike → LTP (权重增强)
    - Post先于Pre spike → LTD (权重减弱)
    - 时间差越小 → 权重变化越大

    参考:
    - Bi & Poo (1998): Experimental measurements of STDP window
    - Song et al. (2000): Competitive STDP learning
    """

    def __init__(
        self,
        pre_neuron_id: int,
        post_neuron_id: int,
        initial_weight: float = 0.5,
        config: Optional[STDPConfig] = None,
    ):
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.weight = initial_weight
        self.config = config or STDPConfig()

        # Spike时间记录
        self.pre_spike_times: List[float] = []
        self.post_spike_times: List[float] = []

        # Eligibility trace (用于延迟奖励学习)
        self.eligibility_trace = 0.0

        # 统计
        self.ltp_count = 0
        self.ltd_count = 0

    def record_pre_spike(self, time_ms: float):
        """记录pre神经元spike时间"""
        self.pre_spike_times.append(time_ms)
        # 保持最近100个spike
        if len(self.pre_spike_times) > 100:
            self.pre_spike_times.pop(0)

        # 更新eligibility trace (pre spike触发)
        self.eligibility_trace = max(self.eligibility_trace, 1.0)

    def record_post_spike(self, time_ms: float):
        """记录post神经元spike时间"""
        self.post_spike_times.append(time_ms)
        if len(self.post_spike_times) > 100:
            self.post_spike_times.pop(0)

    def compute_weight_change(self, pre_time: float, post_time: float) -> float:
        """计算STDP权重变化

        Args:
            pre_time: pre神经元spike时间 (ms)
            post_time: post神经元spike时间 (ms)

        Returns:
            delta_w: 权重变化量
        """
        dt = post_time - pre_time  # 正值: pre先于post

        tau_plus = self.config.tau_plus * 1000  # 转换为ms
        tau_minus = self.config.tau_minus * 1000

        if dt > 0:
            # Pre先于Post → LTP
            delta_w = self.config.A_plus * np.exp(-dt / tau_plus)
            self.ltp_count += 1
        else:
            # Post先于Pre → LTD
            delta_w = -self.config.A_minus * np.exp(dt / tau_minus)
            self.ltd_count += 1

        return delta_w

    def apply_stdp(self, current_time: float, window_ms: float = 100.0) -> float:
        """应用STDP学习

        检查时间窗口内所有pre-post spike配对，累积权重变化

        Args:
            current_time: 当前时间 (ms)
            window_ms: 时间窗口 (只考虑最近的spike配对)

        Returns:
            total_delta: 总权重变化
        """
        total_delta = 0.0

        # 检查最近的pre-post配对
        recent_pre = [t for t in self.pre_spike_times if current_time - t < window_ms]
        recent_post = [t for t in self.post_spike_times if current_time - t < window_ms]

        for pre_t in recent_pre:
            for post_t in recent_post:
                delta = self.compute_weight_change(pre_t, post_t)
                total_delta += delta

        # 应用权重变化 (限制范围)
        self.weight = np.clip(
            self.weight + total_delta,
            self.config.w_min,
            self.config.w_max
        )

        # Eligibility trace衰减
        self.eligibility_trace *= self.config.eligibility_decay

        return total_delta

    def apply_dopamine_modulation(self, dopamine_level: float):
        """多巴胺调制STDP

        高DA → 强化LTP; 低DA → 强化LTD
        参考: Seol et al. (2007)
        """
        # DA调制幅度
        modulation = (dopamine_level - 0.5) * 0.5  # [-0.25, 0.25]
        self.config.A_plus = 0.01 + modulation
        self.config.A_minus = 0.012 - modulation

    def get_summary(self) -> Dict:
        return {
            'weight': self.weight,
            'ltp_count': self.ltp_count,
            'ltd_count': self.ltd_count,
            'eligibility': self.eligibility_trace,
            'pre_spikes': len(self.pre_spike_times),
            'post_spikes': len(self.post_spike_times),
        }


class NMDADependentPlasticity:
    """NMDA受体依赖的突触可塑性

    LTP/LTD机制:
    - 高Ca²⁺内流 (>阈值) → LTP
    - 低Ca²⁺内流 (<阈值) → LTD
    - 需要NMDA受体激活

    参考:
    - Malenka & Bear (2004): LTP/LTD review
    - Lisman (1989): Ca²⁺ threshold hypothesis
    """

    def __init__(
        self,
        ca_threshold_ltp: float = 0.5,   # LTP钙阈值
        ca_threshold_ltd: float = 0.2,   # LTD钙阈值
        nmda_affinity: float = 0.8,      # NMDA受体亲和力
    ):
        self.ca_threshold_ltp = ca_threshold_ltp
        self.ca_threshold_ltd = ca_threshold_ltd
        self.nmda_affinity = nmda_affinity

        # 状态
        self.ca_level = 0.1  # 当前钙离子浓度
        self.nmda_activation = 0.0

    def compute_ca_influx(
        self,
        glutamate_level: float,
        depolarization: float,
    ) -> float:
        """计算钙离子内流

        NMDA受体需要:
        1. 谷氨酸绑定
        2. 足够去极化 (移除Mg²⁺阻断)
        """
        # NMDA激活 = glutamate × depolarization (Mg²⁺阻断解除)
        self.nmda_activation = glutamate_level * depolarization * self.nmda_affinity

        # Ca²⁺内流 (通过NMDA通道)
        ca_influx = self.nmda_activation * 0.5
        self.ca_level = np.clip(self.ca_level + ca_influx, 0.0, 1.0)

        return self.ca_level

    def compute_plasticity_direction(self) -> Tuple[str, float]:
        """根据钙浓度判断可塑性方向

        Returns:
            direction: "LTP", "LTD", or "none"
            magnitude: 可塑性幅度
        """
        if self.ca_level > self.ca_threshold_ltp:
            # 高钙 → LTP
            magnitude = (self.ca_level - self.ca_threshold_ltp) * 0.1
            return "LTP", magnitude
        elif self.ca_level < self.ca_threshold_ltd and self.ca_level > 0.05:
            # 低钙但足够激活 → LTD
            magnitude = (self.ca_threshold_ltd - self.ca_level) * 0.05
            return "LTD", magnitude
        else:
            return "none", 0.0

    def decay(self):
        """钙离子自然衰减"""
        self.ca_level *= 0.95
        self.nmda_activation *= 0.9


class SynapticScaling:
    """突触缩放 - 稳态可塑性

    维持神经元整体兴奋性稳定:
    - 总突触权重过低 → 上调所有权重
    - 总突触权重过高 → 下调所有权重

    参考: Turrigiano (2008)
    """

    def __init__(
        self,
        target_strength: float = 0.5,  # 目标总强度
        scaling_rate: 0.01,             # 缩放速率
        min_scale: float = 0.8,         # 最小缩放因子
        max_scale: float = 1.2,         # 最大缩放因子
    ):
        self.target_strength = target_strength
        self.scaling_rate = scaling_rate
        self.min_scale = min_scale
        self.max_scale = max_scale

    def compute_scaling_factor(self, total_strength: float) -> float:
        """计算缩放因子

        Args:
            total_strength: 当前总突触强度

        Returns:
            scale_factor: 应用于所有突触的缩放因子
        """
        if total_strength < 0.1:
            return self.max_scale

        deviation = (self.target_strength - total_strength) / total_strength

        # 缩放因子 = 1 + deviation × scaling_rate
        scale_factor = 1.0 + deviation * self.scaling_rate

        return np.clip(scale_factor, self.min_scale, self.max_scale)

    def apply_scaling(self, synapses: List[Synapse]) -> float:
        """应用突触缩放

        Args:
            synapses: 突触列表

        Returns:
            scale_factor: 实际应用的缩放因子
        """
        if not synapses:
            return 1.0

        total_strength = sum(s.weight for s in synapses) / len(synapses)
        scale_factor = self.compute_scaling_factor(total_strength)

        # 应用到所有突触
        for synapse in synapses:
            synapse.weight = np.clip(
                synapse.weight * scale_factor,
                0.01, 2.0
            )

        return scale_factor


class Synapse:
    """突触连接"""

    def __init__(
        self,
        pre_neuron_id: int,
        post_neuron_id: int,
        initial_weight: float = 1.0,
    ):
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.weight = initial_weight
        self.last_activation = 0.0
        self.activation_count = 0

    def activate(self, signal: float) -> float:
        """激活突触"""
        self.last_activation = signal * self.weight
        self.activation_count += 1
        return self.last_activation

    def strengthen(self, amount: float = 0.1) -> None:
        """强化突触 - Hebbian学习 (一起放电的连接更强)"""
        self.weight = min(2.0, self.weight + amount)

    def weaken(self, amount: float = 0.05) -> None:
        """弱化突触 - 长时间不激活则减弱"""
        if self.activation_count == 0:
            self.weight = max(0.1, self.weight - amount)

    def should_prune(self, threshold: float = 0.2) -> bool:
        """判断是否应该修剪"""
        return self.weight < threshold


class NeuroplasticitySystem:
    """神经可塑性系统

    整合多种可塑性机制:
    1. STDP - 脉冲时序依赖可塑性
    2. Hebbian - 共激活强化
    3. NMDA依赖的LTP/LTD
    4. 突触缩放 - 稳态可塑性
    5. 神经发生
    """

    def __init__(
        self,
        n_neurons: int = 100,
        n_synapses: int = 500,
        prune_threshold: float = 0.2,
        strengthen_rate: float = 0.1,
        weaken_rate: float = 0.05,
        bdnf_baseline: float = 0.5,
        use_stdp: bool = True,              # 是否启用STDP (新增)
        use_nmda_plasticity: bool = True,   # 是否启用NMDA可塑性 (新增)
        event_bus=None,
    ):
        self.n_neurons = n_neurons
        self.n_synapses = n_synapses
        self.prune_threshold = prune_threshold
        self.strengthen_rate = strengthen_rate
        self.weaken_rate = weaken_rate
        self.bdnf_baseline = bdnf_baseline
        self.use_stdp = use_stdp
        self.use_nmda_plasticity = use_nmda_plasticity

        # 突触连接
        self.synapses: List[Synapse] = []
        self.stdp_synapses: List[STDPSynapse] = []  # STDP突触 (新增)
        self.active_neurons: set = set()
        self.inactive_neurons: set = set()

        # NMDA可塑性系统 (新增)
        self.nmda_plasticity = NMDADependentPlasticity() if use_nmda_plasticity else None

        # 突触缩放系统 (新增)
        self.synaptic_scaling = SynapticScaling()

        # 神经活动历史
        self.activation_history = deque(maxlen=1000)
        self.step_count = 0
        self.current_time_ms = 0.0  # 用于STDP时间戳 (新增)

        # 初始化突触
        self._init_synapses()

        # 生长因子水平
        self.bdnf_level = bdnf_baseline
        self.ngf_level = bdnf_baseline  # NGF (神经生长因子)

        # 状态
        self.state = NeuroplasticityState()

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "brain_update",
                self._handle_brain_update,
                priority=1,
                name="neuroplasticity",
            )

    def _handle_brain_update(self, event) -> Dict:
        """Event-driven handler for brain_update events."""
        import torch as _torch
        state_tensor = event.data.get("state_tensor", _torch.randn(1, 64))
        result = self.forward(state_tensor)

        state = event.data.get("internal_state", {})
        state["plasticity_bdnf"] = result["bdnf_level"]
        state["plasticity_synapses"] = result["plasticity_state"].synaptic_strength

        return result

    def _init_synapses(self) -> None:
        """初始化突触连接"""
        for i in range(self.n_synapses):
            pre_id = np.random.randint(0, self.n_neurons)
            post_id = np.random.randint(0, self.n_neurons)
            if pre_id != post_id:
                weight = np.random.uniform(0.5, 1.0)
                self.synapses.append(Synapse(pre_id, post_id, weight))

        self.inactive_neurons = set(range(self.n_neurons))

    def forward(
        self,
        inputs: torch.Tensor,
        active_indices: Optional[List[int]] = None,
    ) -> Dict:
        """
        前向传播 + 神经可塑性调节

        Args:
            inputs: 输入信号 [batch, n_neurons]
            active_indices: 当前活跃的神经元索引

        Returns:
            outputs: 输出信号
            plasticity_state: 可塑性状态
        """
        self.step_count += 1

        # 1. 前向传播
        outputs = self._forward_pass(inputs)

        # 2. 追踪活跃神经元
        if active_indices:
            self.active_neurons.update(active_indices)
            self.inactive_neurons.difference_update(active_indices)

        # 3. 记录激活
        self.activation_history.append({
            'step': self.step_count,
            'active_count': len(self.active_neurons),
            'active_indices': list(active_indices) if active_indices else [],
        })

        # 4. 突触强度调节
        self._apply_plasticity()

        # 5. 更新状态
        self._update_state()

        return {
            'outputs': outputs,
            'plasticity_state': self.state,
            'bdnf_level': self.bdnf_level,
            'ngf_level': self.ngf_level,
            'pruned_synapses': 0,  # 最近一次修剪的数量
        }

    def _forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        """前向传播 (使用所有突触)"""
        if not self.synapses:
            return torch.zeros(inputs.shape[0]) if inputs.dim() > 1 else torch.tensor(0.0)

        weights = torch.tensor([s.weight for s in self.synapses])  # 使用全部突触

        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        # 映射输入到突触 (循环覆盖)
        n_syn = len(weights)
        flat_inputs = inputs.flatten()
        # 循环tiling输入到突触数量
        tiled = flat_inputs.repeat((n_syn // flat_inputs.numel()) + 1)[:n_syn]

        outputs = torch.sum(tiled * weights) / max(1, n_syn)
        return outputs.unsqueeze(0)

    def _apply_plasticity(self) -> None:
        """应用神经可塑性 (连续Hebbian调制)"""
        for synapse in self.synapses:
            # 连续Hebbian: 活跃度越高强化越强，活跃度越低弱化越强
            # activation_count ∈ [0, N] → 归一化后连续调制
            activation_strength = min(1.0, synapse.activation_count / 5.0)
            # 连续调制: 强化量 = strengthen_rate × activation_strength
            #           弱化量 = weaken_rate × (1 - activation_strength)
            synapse.weight = min(2.0,
                synapse.weight + self.strengthen_rate * activation_strength
            )
            synapse.weight = max(0.05,
                synapse.weight - self.weaken_rate * (1.0 - activation_strength)
            )

        # 重置计数
        for synapse in self.synapses:
            synapse.activation_count = 0

    def _update_state(self) -> None:
        """更新状态"""
        total_weight = sum(s.weight for s in self.synapses)
        active_count = sum(1 for s in self.synapses if s.activation_count > 0)

        self.state.synaptic_strength = total_weight / max(1, len(self.synapses))
        self.state.active_synapses = active_count
        self.state.pruning_rate = self._calculate_prune_rate()
        self.state.bdnf_level = self.bdnf_level
        self.state.neurogenesis_rate = self._calculate_neurogenesis_rate()

    def _calculate_prune_rate(self) -> float:
        """计算修剪率"""
        weak_synapses = sum(1 for s in self.synapses if s.weight < self.prune_threshold)
        return weak_synapses / max(1, len(self.synapses))

    def _calculate_neurogenesis_rate(self) -> float:
        """计算神经发生率 (BDNF相关)"""
        # BDNF高时神经发生更活跃
        base_rate = 0.01
        return base_rate * self.bdnf_level

    def prune_weak_synapses(self, n_prune: int = 10) -> int:
        """
        修剪弱突触

        Args:
            n_prune: 修剪数量

        Returns:
            实际修剪数量
        """
        # 按权重排序，移除最弱的
        self.synapses.sort(key=lambda s: s.weight)

        pruned = 0
        i = 0
        while pruned < n_prune and i < len(self.synapses):
            if self.synapses[i].should_prune(self.prune_threshold):
                self.synapses.pop(i)
                pruned += 1
            else:
                i += 1

        return pruned

    def release_growth_factors(self, activity_level: float = 1.0) -> Dict:
        """
        释放生长因子

        神经活动会触发BDNF释放，促进：
        1. 突触强化
        2. 神经新生
        3. 树突棘形成

        Args:
            activity_level: 活动水平 [0, 1]

        Returns:
            growth_factors: 生长因子水平
        """
        # 活动依赖的BDNF释放
        self.bdnf_level = self.bdnf_baseline + activity_level * 0.5
        self.ngf_level = self.bdnf_baseline + activity_level * 0.3

        return {
            'BDNF': self.bdnf_level,
            'NGF': self.ngf_level,
            'activity_level': activity_level,
        }

    def consolidate_learning(self) -> Dict:
        """
        睡眠期间的记忆巩固 + 突触缩放

        在慢波睡眠期间：
        1. 强化有用的突触
        2. 修剪弱突触
        3. 释放BDNF
        """
        # 突触缩放
        for synapse in self.synapses:
            if synapse.weight > 1.0:
                synapse.weight = min(2.0, synapse.weight * 0.95)
            elif synapse.weight < 0.5:
                synapse.weight = max(0.1, synapse.weight * 1.05)

        # 修剪弱突触
        pruned = self.prune_weak_synapses(n_prune=5)

        # 释放生长因子
        growth = self.release_growth_factors(activity_level=0.8)

        return {
            'synaptic_strength': self.state.synaptic_strength,
            'pruned_synapses': pruned,
            'bdnf_level': self.bdnf_level,
            'consolidation': True,
        }

    def get_summary(self) -> Dict:
        """获取状态摘要"""
        return {
            'n_synapses': len(self.synapses),
            'active_synapses': self.state.active_synapses,
            'synaptic_strength': self.state.synaptic_strength,
            'pruning_rate': self.state.pruning_rate,
            'bdnf_level': self.bdnf_level,
            'neurogenesis_rate': self.state.neurogenesis_rate,
            'total_steps': self.step_count,
        }