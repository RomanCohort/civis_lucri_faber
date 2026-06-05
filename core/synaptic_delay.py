"""
突触延迟机制 (Synaptic Delay Mechanism)

实现真实的突触传递延迟 (0.5-4ms)，替代即时传递。

生物学基础:
  - Shepherd (2004): 突触传递时间 ~0.5-4ms
  - Rall (1967): 轴突传导延迟随距离增加
  - Izhikevich (2006): 延迟对网络同步的影响

关键参数:
  - 化学突触延迟: 0.5-1ms (递质释放+受体结合)
  - 电突触延迟: <0.1ms (直接传导)
  - 轴突传导: 0.1-10ms (取决于距离和直径)
  - 总延迟: 通常1-20ms

用途:
  - 增加网络真实性
  - 影响同步振荡
  - 实现时序编码
"""

from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, Set, Any, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DelayedSignal:
    """延迟信号包"""
    signal: torch.Tensor
    arrival_time: float  # 预计到达时间
    source: str = ""
    metadata: dict = field(default_factory=dict)


class DelayQueue:
    """
    延迟队列 - 存储待传递信号

    使用时间戳队列实现延迟传递
    """

    def __init__(self, max_queue_size: int = 100):
        self.queue: deque = deque(maxlen=max_queue_size)
        self.current_time: float = 0.0

    def push(
        self,
        signal: torch.Tensor,
        delay_ms: float,
        source: str = "",
        metadata: dict = None,
    ):
        """推入延迟信号"""
        arrival_time = self.current_time + delay_ms
        delayed_signal = DelayedSignal(
            signal=signal.clone() if isinstance(signal, torch.Tensor) else signal,
            arrival_time=arrival_time,
            source=source,
            metadata=metadata or {},
        )
        self.queue.append(delayed_signal)

    def advance_time(self, dt_ms: float):
        """推进时间"""
        self.current_time += dt_ms

    def pop_ready(self) -> list[DelayedSignal]:
        """取出已到达的信号"""
        ready = []
        while self.queue and self.queue[0].arrival_time <= self.current_time:
            ready.append(self.queue.popleft())
        return ready

    def peek_next_arrival(self) -> Optional[float]:
        """查看下一个信号到达时间"""
        if self.queue:
            return self.queue[0].arrival_time
        return None

    def clear(self):
        """清空队列"""
        self.queue.clear()

    def size(self) -> int:
        """队列大小"""
        return len(self.queue)


class DelayedSynapse(nn.Module):
    """
    延迟突触 - 实现真实突触延迟

    突触类型:
    1. 化学突触 (Chemical): 0.5-4ms延迟
    2. 电突触 (Electrical): <0.1ms延迟
    3. 混合突触 (Mixed): 两种机制并存

    参考: Shepherd (2004) - Synaptic transmission
    """

    def __init__(
        self,
        synaptic_type: str = "chemical",
        delay_ms: float = 1.0,
        jitter_ms: float = 0.1,  # 延迟抖动
        weight: float = 1.0,
        max_queue_size: int = 100,
    ):
        super().__init__()

        self.synaptic_type = synaptic_type
        self.base_delay = delay_ms
        self.jitter = jitter_ms
        self.weight = nn.Parameter(torch.tensor(weight))

        # 延迟队列
        self.delay_queue = DelayQueue(max_queue_size)

        # 突触类型参数
        if synaptic_type == "chemical":
            self.min_delay = 0.5
            self.max_delay = 4.0
        elif synaptic_type == "electrical":
            self.min_delay = 0.05
            self.max_delay = 0.2
        else:  # mixed
            self.min_delay = 0.1
            self.max_delay = 3.0

        # 确保延迟在合理范围
        self.base_delay = np.clip(delay_ms, self.min_delay, self.max_delay)

    def forward(
        self,
        signal: torch.Tensor,
        time_ms: float = None,
        add_jitter: bool = True,
    ) -> dict[str, Any]:
        """
        突触传递

        Args:
            signal: 输入信号 [B, dim]
            time_ms: 当前时间 (None则使用队列内部时间)
            add_jitter: 是否添加延迟抖动

        Returns:
            pushed: 是否成功推入队列
            effective_delay: 实际延迟时间
            queue_size: 队列当前大小
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        # 更新时间
        if time_ms is not None:
            self.delay_queue.current_time = time_ms

        # 计算有效延迟
        effective_delay = self.base_delay
        if add_jitter:
            jitter = np.random.uniform(-self.jitter, self.jitter)
            effective_delay = np.clip(self.base_delay + jitter, self.min_delay, self.max_delay)

        # 应用权重
        weighted_signal = signal * self.weight

        # 推入延迟队列
        self.delay_queue.push(
            signal=weighted_signal,
            delay_ms=effective_delay,
            metadata={'synaptic_type': self.synaptic_type},
        )

        return {
            'pushed': True,
            'effective_delay': effective_delay,
            'queue_size': self.delay_queue.size(),
            'next_arrival': self.delay_queue.peek_next_arrival(),
        }

    def receive(self, advance_time_ms: float = 0.0) -> list[DelayedSignal]:
        """
        接收已到达信号

        Args:
            advance_time_ms: 时间推进量

        Returns:
            ready_signals: 已到达的信号列表
        """
        self.delay_queue.advance_time(advance_time_ms)
        return self.delay_queue.pop_ready()

    def get_current_output(self, advance_time_ms: float = 1.0) -> torch.Tensor:
        """
        获取当前时刻的输出信号 (聚合所有到达信号)

        Args:
            advance_time_ms: 时间推进量

        Returns:
            aggregated_output: 聚合输出信号
        """
        ready_signals = self.receive(advance_time_ms)

        if not ready_signals:
            return None

        # 聚合所有到达信号
        outputs = [s.signal for s in ready_signals]
        aggregated = torch.stack(outputs).sum(dim=0)

        return aggregated

    def reset(self):
        """重置突触"""
        self.delay_queue.clear()

    def get_delay_stats(self) -> dict[str, Any]:
        """获取延迟统计"""
        return {
            'synaptic_type': self.synaptic_type,
            'base_delay': self.base_delay,
            'min_delay': self.min_delay,
            'max_delay': self.max_delay,
            'jitter': self.jitter,
            'weight': float(self.weight),
            'queue_size': self.delay_queue.size(),
            'current_time': self.delay_queue.current_time,
        }


class AxonalConduction(nn.Module):
    """
    轴突传导延迟

    轴突直径和长度影响传导速度:
    - 直径越大 → 传导越快
    - 长度越长 → 延迟越大
    - 有髓鞘 → 传导更快 (跳跃传导)

    参考: Rall (1967) - Cable theory
    """

    def __init__(
        self,
        fiber_length_mm: float = 10.0,
        fiber_diameter_um: float = 5.0,
        myelinated: bool = True,
    ):
        super().__init__()

        self.length = fiber_length_mm
        self.diameter = fiber_diameter_um
        self.myelinated = myelinated

        # 传导速度计算 (m/s)
        # 有髓鞘: v ≈ 6 * diameter (um)
        # 无髓鞘: v ≈ 0.5 * sqrt(diameter) (um)
        if myelinated:
            self.conduction_velocity = 6.0 * fiber_diameter_um
        else:
            self.conduction_velocity = 0.5 * np.sqrt(fiber_diameter_um)

        # 传导延迟 (ms) = length(mm) / velocity(m/s) * 1000
        self.conduction_delay = (fiber_length_mm / self.conduction_velocity) if self.conduction_velocity > 0 else 10.0

    def forward(self, signal: torch.Tensor) -> dict[str, float]:
        """
        轴突传导

        Args:
            signal: 输入信号

        Returns:
            conduction_delay: 传导延迟 (ms)
            velocity: 传导速度 (m/s)
            arrived: 信号已到达 (对于延迟模型，返回False)
        """
        return {
            'conduction_delay': self.conduction_delay,
            'conduction_velocity': self.conduction_velocity,
            'fiber_length': self.length,
            'fiber_diameter': self.diameter,
            'myelinated': self.myelinated,
        }


class DelayedSynapseManager(nn.Module):
    """
    延迟突触管理器

    管理多个延迟突触，统一时间推进和信号接收
    """

    def __init__(self, n_synapses: int = 10, default_delay_ms: float = 1.0):
        super().__init__()

        self.global_time: float = 0.0
        self.time_step_ms: float = 1.0  # 每步时间推进

        # 创建延迟突触列表
        self.synapses = nn.ModuleList([
            DelayedSynapse(delay_ms=default_delay_ms, jitter_ms=0.1)
            for _ in range(n_synapses)
        ])

        # 突触命名映射
        self.synapse_names: dict[str, int] = {}

    def register_synapse(self, name: str, index: int):
        """注册突触名称"""
        self.synapse_names[name] = index

    def send_to(
        self,
        name_or_index: str | int,
        signal: torch.Tensor,
        add_jitter: bool = True,
    ) -> dict[str, Any]:
        """发送信号到指定突触"""
        if isinstance(name_or_index, str):
            index = self.synapse_names.get(name_or_index, 0)
        else:
            index = name_or_index

        if index >= len(self.synapses):
            return {'error': 'synapse index out of range'}

        return self.synapses[index].forward(
            signal=signal,
            time_ms=self.global_time,
            add_jitter=add_jitter,
        )

    def advance_time(self, dt_ms: float = None):
        """推进全局时间"""
        if dt_ms is None:
            dt_ms = self.time_step_ms
        self.global_time += dt_ms

    def receive_all(self) -> dict[str, list[DelayedSignal]]:
        """从所有突触接收信号"""
        results = {}
        for name, index in self.synapse_names.items():
            self.synapses[index].delay_queue.advance_time(0)  # 同步时间
            ready = self.synapses[index].delay_queue.pop_ready()
            if ready:
                results[name] = ready
        return results

    def get_aggregated_output(self, synapse_name: str) -> torch.Tensor:
        """获取指定突触的聚合输出"""
        if synapse_name not in self.synapse_names:
            return None

        index = self.synapse_names[synapse_name]
        synapse = self.synapses[index]

        ready_signals = synapse.receive(self.time_step_ms)
        if not ready_signals:
            return None

        outputs = [s.signal for s in ready_signals]
        return torch.stack(outputs).sum(dim=0)

    def reset_all(self):
        """重置所有突触"""
        self.global_time = 0.0
        for synapse in self.synapses:
            synapse.reset()

    def get_manager_stats(self) -> dict[str, Any]:
        """获取管理器统计"""
        synapse_stats = []
        for i, synapse in enumerate(self.synapses):
            synapse_stats.append(synapse.get_delay_stats())

        return {
            'global_time': self.global_time,
            'time_step_ms': self.time_step_ms,
            'n_synapses': len(self.synapses),
            'synapse_stats': synapse_stats,
        }


# 典型延迟参数配置
SYNAPSE_DELAY_CONFIGS = {
    'fast_excitatory': {
        'type': 'chemical',
        'delay_ms': 0.5,
        'jitter_ms': 0.05,
    },
    'standard_excitatory': {
        'type': 'chemical',
        'delay_ms': 1.0,
        'jitter_ms': 0.1,
    },
    'slow_inhibitory': {
        'type': 'chemical',
        'delay_ms': 2.0,
        'jitter_ms': 0.2,
    },
    'gap_junction': {
        'type': 'electrical',
        'delay_ms': 0.1,
        'jitter_ms': 0.02,
    },
    'cortical_local': {
        'type': 'chemical',
        'delay_ms': 1.0,  # 皮层内局部连接
        'jitter_ms': 0.1,
    },
    'cortical_long_range': {
        'type': 'chemical',
        'delay_ms': 5.0,  # 皮层长距离连接
        'jitter_ms': 0.5,
    },
    'thalamic_relay': {
        'type': 'chemical',
        'delay_ms': 3.0,  # 丘脑中继
        'jitter_ms': 0.3,
    },
    'hippocampal_loop': {
        'type': 'chemical',
        'delay_ms': 10.0,  # 海马环路
        'jitter_ms': 1.0,
    },
}


def create_delayed_synapse(config_name: str = 'standard_excitatory') -> DelayedSynapse:
    """根据配置创建延迟突触"""
    config = SYNAPSE_DELAY_CONFIGS.get(config_name, SYNAPSE_DELAY_CONFIGS['standard_excitatory'])
    return DelayedSynapse(
        synaptic_type=config['type'],
        delay_ms=config['delay_ms'],
        jitter_ms=config['jitter_ms'],
    )


__all__ = [
    'DelayedSynapse',
    'DelayQueue',
    'DelayedSignal',
    'AxonalConduction',
    'DelayedSynapseManager',
    'SYNAPSE_DELAY_CONFIGS',
    'create_delayed_synapse',
]