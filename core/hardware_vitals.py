"""
硬件生命体征桥接器 (Hardware Vitals Bridge)

将真实计算资源指标映射为生物学隐喻参数，替代硬编码常量和随机噪声。

映射表：
    CPU利用率    → 心率 (Heart Rate, bpm)
    RAM使用率    → 血压 (Blood Pressure, mmHg)
    可用计算余量  → 血氧 (O2 Saturation)
    事件队列积压  → CO2水平 (Blood Gas)
    CPU+RAM均值  → 代谢需求 (Metabolic Demand)
    系统负载峰值  → 交感神经张力 (Sympathetic Tone)
    系统余量      → 副交感神经张力 (Parasympathetic Tone)
    进程内存增长  → 疲劳 (Fatigue)
    错误率        → 皮质醇驱动 (Cortisol/Stress)
    进程内存/总RAM → 神经废物 (Brain Waste)
    磁盘满度      → 痛觉信号 (Pain/Nociception)
    可用RAM比例   → 血容量 (Blood Volume)
    GC频率        → 肠道血清素 (Gut Serotonin)
    线程密度      → 肠道GABA (Gut GABA)

核心类：
1. HardwareState — 硬件状态快照
2. HardwareVitals — 读取+映射+双语输出
"""

import gc
import os
import time
import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class HardwareState:
    """硬件状态快照"""
    # CPU
    cpu_percent: float = 0.0           # [0, 1]
    cpu_count: int = 1

    # RAM
    ram_percent: float = 0.0           # [0, 1]
    ram_used_mb: float = 0.0
    ram_total_mb: float = 1.0
    ram_available_mb: float = 1.0

    # Disk
    disk_percent: float = 0.0          # [0, 1]

    # GPU (optional)
    gpu_available: bool = False
    gpu_memory_percent: float = 0.0    # [0, 1]

    # Process
    process_rss_mb: float = 0.0        # 进程内存 RSS
    thread_count: int = 1

    # Derived
    event_queue_size: int = 0
    error_rate: float = 0.0            # [0, 1]
    gc_gen0_count: int = 0
    wall_clock_hour: float = 0.0       # 0-24


class HardwareVitals:
    """
    硬件生命体征桥接

    每步读取硬件指标，转换为 [0,1] 范围的生物学参数。
    所有 to_xxx() 方法返回 float，范围 [0, 1]（除特别标注外）。
    """

    def __init__(self):
        self.state = HardwareState()

        # 内部追踪
        self._step_count = 0
        self._last_gc_count = 0
        self._error_count = 0
        self._last_read_time = time.time()

        # 平滑缓存（避免指标剧烈跳变）
        self._smooth_cpu = 0.0
        self._smooth_ram = 0.0
        self._smooth_disk = 0.0
        self._smoothing = 0.7  # EMA系数

        # 进程对象缓存
        self._process = psutil.Process(os.getpid()) if HAS_PSUTIL else None

    def read(self, agent=None) -> HardwareState:
        """
        读取硬件指标

        Args:
            agent: 可选，用于读取event_bus队列大小等agent内部指标
        """
        if not HAS_PSUTIL:
            return self.state

        s = self.state

        # CPU
        raw_cpu = psutil.cpu_percent(interval=0) / 100.0
        self._smooth_cpu = self._smoothing * raw_cpu + (1 - self._smoothing) * self._smooth_cpu
        s.cpu_percent = self._smooth_cpu
        s.cpu_count = psutil.cpu_count(logical=True) or 1

        # RAM
        mem = psutil.virtual_memory()
        raw_ram = mem.percent / 100.0
        self._smooth_ram = self._smoothing * raw_ram + (1 - self._smoothing) * self._smooth_ram
        s.ram_percent = self._smooth_ram
        s.ram_used_mb = mem.used / (1024 * 1024)
        s.ram_total_mb = mem.total / (1024 * 1024)
        s.ram_available_mb = mem.available / (1024 * 1024)

        # Disk
        try:
            disk_path = 'C:\\' if os.name == 'nt' else '/'
            disk = psutil.disk_usage(disk_path)
            raw_disk = disk.percent / 100.0
            self._smooth_disk = self._smoothing * raw_disk + (1 - self._smoothing) * self._smooth_disk
            s.disk_percent = self._smooth_disk
        except Exception:
            pass

        # GPU
        s.gpu_available = torch.cuda.is_available()
        if s.gpu_available:
            try:
                gpu_alloc = torch.cuda.memory_allocated()
                gpu_reserved = torch.cuda.memory_reserved() or 1
                s.gpu_memory_percent = min(1.0, gpu_alloc / gpu_reserved)
            except Exception:
                pass

        # Process
        if self._process is not None:
            try:
                s.process_rss_mb = self._process.memory_info().rss / (1024 * 1024)
                s.thread_count = self._process.num_threads()
            except Exception:
                pass

        # GC频率
        gc_counts = gc.get_count()
        s.gc_gen0_count = gc_counts[0]
        gc_delta = max(0, gc_counts[0] - self._last_gc_count)
        self._last_gc_count = gc_counts[0]

        # Event queue (从agent读取)
        if agent is not None and hasattr(agent, 'bus'):
            bus = agent.bus
            s.event_queue_size = len(getattr(bus, '_log', []))

        # 墙钟时间
        now = time.localtime()
        s.wall_clock_hour = now.tm_hour + now.tm_min / 60.0

        # 错误率衰减
        s.error_rate *= 0.95

        self._step_count += 1
        self._last_read_time = time.time()
        return self.state

    def report_error(self):
        """报告一个错误（提升错误率）"""
        self.state.error_rate = min(1.0, self.state.error_rate + 0.2)

    # ================================================================
    # 生物学隐喻映射
    # ================================================================

    def to_heart_rate(self) -> float:
        """CPU利用率 → 心率 (bpm)"""
        # 0% → 60bpm (静息), 100% → 160bpm (极量运动)
        return 60.0 + self.state.cpu_percent * 100.0

    def to_blood_pressure(self) -> float:
        """RAM使用率 → 收缩压 (mmHg)"""
        # 30% → 90 mmHg (正常), 95% → 180 mmHg (高血压危象)
        return 70.0 + self.state.ram_percent * 120.0

    def to_co2_level(self) -> float:
        """事件队列积压 + CPU负载 → CO2水平 [0, 1]"""
        # 正常 ~0.35, 积压时上升
        queue_pressure = min(0.3, self.state.event_queue_size * 0.005)
        cpu_pressure = self.state.cpu_percent * 0.15
        return np.clip(0.35 + queue_pressure + cpu_pressure, 0.2, 0.9)

    def to_o2_level(self) -> float:
        """可用计算余量 → 血氧 [0, 1]"""
        headroom = 1.0 - max(self.state.cpu_percent, self.state.ram_percent)
        return 0.7 + 0.3 * headroom  # 0.7 ~ 1.0

    def to_metabolic_demand(self) -> float:
        """CPU+RAM平均负载 → 代谢需求 [0, 1]"""
        return (self.state.cpu_percent + self.state.ram_percent) / 2.0

    def to_sympathetic(self) -> float:
        """系统负载峰值 → 交感神经张力 [0, 1]"""
        load = max(self.state.cpu_percent, self.state.ram_percent)
        return min(1.0, load * 1.1)

    def to_parasympathetic(self) -> float:
        """系统余量 → 副交感神经张力 [0, 1]"""
        headroom = 1.0 - max(self.state.cpu_percent, self.state.ram_percent)
        return 0.3 + 0.7 * headroom

    def to_fatigue(self) -> float:
        """进程内存增长 + 运行步数 → 疲劳 [0, 1]"""
        if self.state.ram_total_mb > 0:
            mem_ratio = self.state.process_rss_mb / self.state.ram_total_mb
        else:
            mem_ratio = 0.1
        step_factor = min(0.3, self._step_count / 50000)
        return np.clip(mem_ratio * 0.7 + step_factor, 0.05, 1.0)

    def to_cortisol_drive(self) -> float:
        """资源紧张 + 错误率 → 应激驱动 [0, 1]"""
        resource_stress = max(self.state.cpu_percent, self.state.ram_percent)
        return min(1.0, resource_stress * 0.6 + self.state.error_rate * 0.4)

    def to_waste_level(self) -> float:
        """进程内存/总RAM → 神经废物 [0, 1]"""
        if self.state.ram_total_mb > 0:
            ratio = self.state.process_rss_mb / self.state.ram_total_mb
            return min(1.0, ratio * 3.0)  # 放大灵敏度
        return 0.2

    def to_pain_signal(self) -> float:
        """磁盘满 + 内存临界 → 痛觉信号 [0, 1]"""
        if self.state.ram_percent > 0.92 or self.state.disk_percent > 0.95:
            return 0.9
        elif self.state.ram_percent > 0.85:
            return 0.5
        elif self.state.disk_percent > 0.90:
            return 0.3
        return 0.0

    def to_blood_volume(self) -> float:
        """可用RAM比例 → 血容量 [0, 1]"""
        return 1.0 - self.state.ram_percent

    def to_gut_serotonin(self) -> float:
        """GC频率 → 肠道血清素 [0, 1]

        生物学类比：肠道微生物产生90%的血清素前体
        计算类比：垃圾回收频率反映系统"消化"负荷
        高GC → 高处理负荷 → 血清素波动
        """
        gc_gen0 = self.state.gc_gen0_count
        # 正常范围：GC gen0 约 200-700 → 映射到 0.3-0.7
        normalized = np.clip(gc_gen0 / 1000.0, 0.1, 1.0)
        return 0.3 + 0.4 * normalized

    def to_gut_gaba(self) -> float:
        """线程密度 → 肠道GABA [0, 1]

        生物学类比：肠道神经系统的抑制性递质
        计算类比：线程/CPU核数比反映并发负荷
        过多线程 → 系统紧张 → GABA下降
        """
        if self.state.cpu_count > 0:
            thread_density = self.state.thread_count / self.state.cpu_count
            # 1线程/核 → 0.7 (松弛), 10线程/核 → 0.3 (紧张)
            return max(0.2, 0.8 - thread_density * 0.05)
        return 0.5

    def to_adrenaline(self) -> float:
        """GPU负载 / CPU突发 → 肾上腺素 [0, 1]"""
        if self.state.gpu_available:
            return max(self.state.gpu_memory_percent, self.state.cpu_percent * 0.8)
        return self.state.cpu_percent * 0.8

    def to_stress_level(self) -> float:
        """综合压力 → 应激水平 [0, 1]"""
        return (self.state.cpu_percent + self.state.ram_percent + self.state.error_rate) / 3.0

    # ================================================================
    # 状态向量生成（替代 torch.randn）
    # ================================================================

    def make_state_vector(self, dim: int = 64) -> torch.Tensor:
        """
        将硬件指标编码为固定维度向量

        前16维：硬件指标编码（每个重复以填满）
        后48维：可控噪声（幅度由系统负载调制）

        高负载 → 更多噪声（模拟不确定性增加）
        低负载 → 较少噪声（稳定状态）
        """
        # 核心指标归一化
        metrics = [
            self.state.cpu_percent,
            self.state.ram_percent,
            self.state.disk_percent,
            self.state.gpu_memory_percent if self.state.gpu_available else 0.0,
            self.state.process_rss_mb / max(self.state.ram_total_mb, 1.0),
            self.state.error_rate,
            self.to_fatigue(),
            self.to_o2_level(),
            self.to_sympathetic(),
            self.to_parasympathetic(),
            self.to_co2_level(),
            self.to_metabolic_demand(),
            self.to_waste_level(),
            self.to_gut_serotonin(),
            self.to_gut_gaba(),
            self.to_pain_signal(),
        ]

        # 编码前16维：重复核心指标
        encoded = []
        for i in range(16):
            encoded.append(metrics[i % len(metrics)])

        # 后48维：负载调制的噪声
        load = max(self.state.cpu_percent, self.state.ram_percent)
        noise_scale = 0.1 + 0.9 * load  # 低负载0.1噪声，高负载1.0噪声
        noise = torch.randn(48) * noise_scale
        noise_vec = noise.tolist()

        full_vector = encoded + noise_vec
        return torch.tensor(full_vector[:dim], dtype=torch.float32).unsqueeze(0)

    # ================================================================
    # 双语摘要
    # ================================================================

    def get_bilingual_summary(self) -> Dict:
        """
        返回双语摘要：生物学名称 ↔ 硬件指标

        格式: {metric_name: {'bio': '72 bpm', 'hw': 'CPU 12.3%'}}
        """
        s = self.state
        return {
            'heart_rate': {
                'bio': f'{self.to_heart_rate():.0f} bpm',
                'hw': f'CPU {s.cpu_percent * 100:.1f}%',
                'metaphor': 'CPU利用率 → 心率',
            },
            'blood_pressure': {
                'bio': f'{self.to_blood_pressure():.0f} mmHg',
                'hw': f'RAM {s.ram_percent * 100:.1f}%',
                'metaphor': '内存使用率 → 血压',
            },
            'o2_saturation': {
                'bio': f'{self.to_o2_level():.2f}',
                'hw': f'余量 {(1 - max(s.cpu_percent, s.ram_percent)) * 100:.1f}%',
                'metaphor': '计算余量 → 血氧',
            },
            'co2_level': {
                'bio': f'{self.to_co2_level():.3f}',
                'hw': f'队列 {s.event_queue_size} + CPU {s.cpu_percent * 100:.1f}%',
                'metaphor': '积压负载 → CO2',
            },
            'metabolic_demand': {
                'bio': f'{self.to_metabolic_demand():.2f}',
                'hw': f'平均负载 {self.to_metabolic_demand() * 100:.1f}%',
                'metaphor': 'CPU+RAM均值 → 代谢需求',
            },
            'sympathetic_tone': {
                'bio': f'{self.to_sympathetic():.2f}',
                'hw': f'负载峰值 {max(s.cpu_percent, s.ram_percent) * 100:.1f}%',
                'metaphor': '负载峰值 → 交感张力',
            },
            'fatigue': {
                'bio': f'{self.to_fatigue():.2f}',
                'hw': f'进程 {s.process_rss_mb:.0f}MB / {s.ram_total_mb:.0f}MB, step {self._step_count}',
                'metaphor': '进程内存增长 → 疲劳',
            },
            'waste_level': {
                'bio': f'{self.to_waste_level():.2f}',
                'hw': f'RSS占比 {s.process_rss_mb / max(s.ram_total_mb, 1) * 100:.1f}%',
                'metaphor': '进程内存/总RAM → 神经废物',
            },
            'pain_signal': {
                'bio': f'{self.to_pain_signal():.2f}',
                'hw': f'RAM {s.ram_percent * 100:.0f}% + Disk {s.disk_percent * 100:.0f}%',
                'metaphor': '磁盘满+内存临界 → 痛觉',
            },
            'blood_volume': {
                'bio': f'{self.to_blood_volume():.2f}',
                'hw': f'可用RAM {(1 - s.ram_percent) * 100:.1f}%',
                'metaphor': '可用RAM → 血容量',
            },
            'gut_serotonin': {
                'bio': f'{self.to_gut_serotonin():.2f}',
                'hw': f'GC gen0={s.gc_gen0_count}',
                'metaphor': '垃圾回收频率 → 肠道血清素',
            },
            'gut_gaba': {
                'bio': f'{self.to_gut_gaba():.2f}',
                'hw': f'线程 {s.thread_count} / {s.cpu_count}核',
                'metaphor': '线程密度 → 肠道GABA',
            },
        }


def create_hardware_vitals() -> HardwareVitals:
    """创建硬件生命体征桥接"""
    return HardwareVitals()


__all__ = [
    'HardwareState',
    'HardwareVitals',
    'create_hardware_vitals',
]
