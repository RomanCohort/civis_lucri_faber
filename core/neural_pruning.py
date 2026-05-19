"""
神经修剪系统 - Neural Pruning System

核心思想: 模拟生物突触修剪的三阶段过程
  阶段1 (衰减): 不常用的神经元权重逐渐降低，越久不用衰减越快
  阶段2 (硬剪): 超过容忍期限后权重归零，真正减少参数量
  阶段3 (恢复): 再次激活时大量释放生长因子，从快照恢复权重

生物参考:
  - 突触修剪: 发育期大量突触被消除，但不是瞬间完成的
  - BDNF 释放: 活跃神经元释放脑源性神经营养因子，促进突触存活和再生
  - 用进废退: Hebb 理论的核心——一起放电的连接存活，不用的被淘汰
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass
from collections import deque


# ============ 配置 ============

@dataclass
class PruningConfig:
    """神经修剪配置"""
    # --- 衰减阶段 ---
    decay_base: float = 0.002          # 初始衰减率（每步）
    decay_max: float = 0.08            # 最终衰减率（久不活动后）
    decay_warmup_steps: int = 30       # 开始衰减前的观察期（新神经元宽限期）
    decay_ramp_steps: int = 80         # 从 base 爬升到 max 的步数

    # --- 硬剪阶段 ---
    hibernation_steps: int = 200       # 连续不活动多少步后触发硬剪（零化）
    weight_zero_threshold: float = 0.01  # 权重绝对值低于此视为已零化

    # --- 生长因子恢复阶段 ---
    gf_surge_scale: float = 3.0        # 恢复时生长因子释放倍数
    restore_ratio: float = 0.8         # 从快照恢复的比例（1.0=完全恢复）

    # --- 活动追踪 ---
    activation_threshold: float = 1e-4  # 绝对阈值（下限）
    activation_relative_ratio: float = 0.3  # 相对阈值: 输出 > mean * ratio 才算活跃
    ema_alpha: float = 0.1             # 活动率 EMA 更新速率

    # --- 生长因子扩散 ---
    diffusion_rate: float = 0.05       # 生长因子向邻居扩散的速率


# ============ 单个神经元状态 ============

@dataclass
class NeuronState:
    """单个神经元的修剪状态"""
    consecutive_inactive: int = 0      # 连续不活动步数
    recent_activity: float = 1.0       # 活动率 EMA
    is_hibernated: bool = False        # 是否已被硬剪（冬眠中）
    activated_this_step: bool = False  # 本步是否激活


# ============ 活动追踪器 ============

class ActivityTracker:
    """通过前向钩子自动追踪每个输出神经元的活动状态"""

    def __init__(self, activation_threshold: float = 1e-4, ema_alpha: float = 0.1,
                 relative_ratio: float = 0.3):
        self.activation_threshold = activation_threshold
        self.relative_ratio = relative_ratio
        self.ema_alpha = ema_alpha
        self.records: Dict[str, Dict[int, NeuronState]] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._step_activations: Dict[str, set] = {}

    def register(self, module: nn.Module, name: str, n_neurons: int):
        hook = module.register_forward_hook(self._make_hook(name))
        self._hooks.append(hook)
        # 预创建所有神经元的状态记录
        self.records[name] = {j: NeuronState() for j in range(n_neurons)}
        self._step_activations[name] = set()

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                return
            with torch.no_grad():
                if tensor.dim() >= 3:
                    flat = tensor.mean(dim=1)  # [batch, features]
                elif tensor.dim() == 2:
                    flat = tensor
                elif tensor.dim() == 1:
                    flat = tensor.unsqueeze(0)
                else:
                    return

                # 相对阈值: 输出绝对值 > mean * ratio 才算活跃
                abs_out = flat.abs()            # [batch, features]
                mean_abs = abs_out.mean() + 1e-8
                threshold = max(mean_abs * self.relative_ratio, self.activation_threshold)
                active = (abs_out > threshold).any(dim=0)  # [features]

                for j in range(active.shape[0]):
                    if active[j]:
                        self._step_activations[name].add(j)
        return hook_fn

    def advance(self):
        """推进一个时间步，更新所有神经元状态"""
        for name, activated in self._step_activations.items():
            if name not in self.records:
                continue
            n = len(self.records[name])

            for j in range(n):
                if j not in self.records[name]:
                    self.records[name][j] = NeuronState()
                s = self.records[name][j]
                is_active = j in activated

                if is_active:
                    s.consecutive_inactive = 0
                    s.activated_this_step = True
                    s.recent_activity += self.ema_alpha * (1.0 - s.recent_activity)
                    if s.is_hibernated:
                        s.is_hibernated = False
                else:
                    s.consecutive_inactive += 1
                    s.activated_this_step = False
                    s.recent_activity += self.ema_alpha * (0.0 - s.recent_activity)

        for name in self._step_activations:
            self._step_activations[name] = set()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============ 主系统 ============

class NeuralPruningSystem(nn.Module):
    """
    三阶段神经修剪系统

    阶段1 - 渐进衰减:
      不活动神经元的权重逐渐缩小。衰减率从 decay_base 线性爬升到 decay_max，
      爬升持续 decay_ramp_steps 步。前 decay_warmup_steps 步为宽限期，不衰减。

    阶段2 - 硬剪零化:
      连续不活动超过 hibernation_steps 步后，整行权重归零，并保存快照。
      此后该行不参与前向传播，真正减少计算量。

    阶段3 - 生长因子恢复:
      当冬眠神经元重新被激活时，释放大量 BDNF（gf_surge_scale 倍），
      从保存的快照中恢复 restore_ratio 比例的权重，快速恢复功能。

    Usage:
        pruning = NeuralPruningSystem()
        pruning.attach(model.fc1, "fc1")

        for x, y in datalooder:
            out = model(x)        # 钩子自动追踪活动
            pruning.step()         # 衰减 → 硬剪 → 恢复
    """

    def __init__(self, config: PruningConfig = None, event_bus=None):
        super().__init__()
        self.config = config or PruningConfig()
        self.tracker = ActivityTracker(
            activation_threshold=self.config.activation_threshold,
            ema_alpha=self.config.ema_alpha,
            relative_ratio=self.config.activation_relative_ratio,
        )
        self._modules_map: Dict[str, nn.Module] = {}
        self._output_sizes: Dict[str, int] = {}

        # 冬眠快照: {module_name: {neuron_id: saved_weight_row}}
        self._snapshots: Dict[str, Dict[int, torch.Tensor]] = {}

        # 生长因子浓度 (每个神经元)
        self._gf: Dict[str, torch.Tensor] = {}

        # 统计
        self.step_count = 0
        self.stats = {
            'total_decayed': 0,
            'total_hibernated': 0,
            'total_revived': 0,
            'active_params': 0,
            'total_params': 0,
        }

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "pruning_update",
                self._handle_pruning_update,
                priority=0,
                name="neural_pruning",
            )

    def _handle_pruning_update(self, event) -> Dict:
        """Event-driven handler for pruning_update events."""
        result = self.step()
        return result

    def attach(self, module: nn.Module, name: str):
        """注册模块，安装活动追踪钩子"""
        self._modules_map[name] = module

        # 推断输出维度
        if isinstance(module, nn.Linear):
            n = module.out_features
        elif isinstance(module, (nn.GRU, nn.LSTM)):
            n = module.hidden_size
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            n = module.out_channels
        else:
            n = 64
        self._output_sizes[name] = n
        self._gf[name] = torch.ones(n) * 0.5
        self._snapshots[name] = {}

        # 注册钩子（在知道 n 之后）
        self.tracker.register(module, name, n_neurons=n)

    def step(self) -> Dict:
        """
        每步调用：追踪活动 → 衰减 → 硬剪 → 恢复
        """
        self.step_count += 1
        self.tracker.advance()

        decayed = 0
        hibernated = 0
        revived = 0

        for name, module in self._modules_map.items():
            weight = self._get_weight(module)
            if weight is None or weight.dim() < 2:
                continue

            records = self.tracker.records.get(name, {})
            n_out = self._output_sizes[name]
            cfg = self.config

            with torch.no_grad():
                for j in range(min(n_out, weight.shape[0])):
                    s = records.get(j)
                    if s is None:
                        continue

                    # ========== 阶段3: 恢复（冬眠中且被激活）==========
                    if s.is_hibernated and s.activated_this_step:
                        hibernated_neuron_revived = self._revive(name, j, weight)
                        if hibernated_neuron_revived:
                            revived += 1
                            s.is_hibernated = False
                        continue

                    # ========== 阶段2: 硬剪（超时零化）==========
                    if not s.is_hibernated and s.consecutive_inactive >= cfg.hibernation_steps:
                        self._hibernate(name, j, weight)
                        hibernated += 1
                        s.is_hibernated = True
                        continue

                    # ========== 阶段1: 渐进衰减 ==========
                    if (not s.is_hibernated
                            and s.consecutive_inactive > cfg.decay_warmup_steps
                            and weight.data[j].abs().sum() > 0):
                        # 爬升: 从 warmup 结束开始，在 ramp_steps 内线性爬升
                        progress = min(1.0, (s.consecutive_inactive - cfg.decay_warmup_steps) / cfg.decay_ramp_steps)
                        rate = cfg.decay_base + progress * (cfg.decay_max - cfg.decay_base)
                        weight.data[j] *= (1.0 - rate)
                        decayed += 1

                    # ========== 活跃神经元: 释放生长因子 ==========
                    if s.activated_this_step:
                        self._gf[name][j] += cfg.gf_surge_scale * s.recent_activity

                # 生长因子扩散 + 衰减
                self._diffuse_and_decay_gf(name)

            # 更新参数统计
            self._count_params(name, weight)

        self.stats['total_decayed'] += decayed
        self.stats['total_hibernated'] += hibernated
        self.stats['total_revived'] += revived

        return {
            'step': self.step_count,
            'decayed': decayed,
            'hibernated': hibernated,
            'revived': revived,
            'active_ratio': self.stats['active_params'] / max(1, self.stats['total_params']),
        }

    # ---------- 阶段2: 硬剪 ----------

    def _hibernate(self, name: str, neuron_id: int, weight: nn.Parameter):
        """零化权重行，保存快照"""
        row = weight.data[neuron_id].clone()
        # 只有非零行才值得保存
        if row.abs().sum() > self.config.weight_zero_threshold:
            self._snapshots[name][neuron_id] = row
        weight.data[neuron_id].zero_()

    # ---------- 阶段3: 恢复 ----------

    def _revive(self, name: str, neuron_id: int, weight: nn.Parameter) -> bool:
        """从快照恢复权重，模拟生长因子大量释放后的突触再生"""
        snapshot = self._snapshots.get(name, {}).get(neuron_id)
        if snapshot is not None:
            # 恢复 restore_ratio 比例的原始权重
            weight.data[neuron_id] = snapshot * self.config.restore_ratio
            # 大量释放 BDNF → 强化周围连接
            n = self._output_sizes[name]
            surge = self.config.gf_surge_scale
            j = neuron_id
            # 自己和邻居都受益
            for k in range(max(0, j - 2), min(n, j + 3)):
                dist = abs(k - j)
                self._gf[name][k] += surge / (1.0 + dist)
            del self._snapshots[name][neuron_id]
            return True
        else:
            # 没有快照: 用小随机值重新初始化
            weight.data[neuron_id] = torch.randn_like(weight.data[neuron_id]) * 0.01
            return True

    # ---------- 生长因子扩散与衰减 ----------

    def _diffuse_and_decay_gf(self, name: str):
        """生长因子向邻居扩散，然后自然衰减"""
        gf = self._gf[name]
        n = gf.shape[0]
        if n < 3:
            gf *= 0.95  # 简单衰减
            return

        # 扩散: 向邻居均值移动
        padded = torch.nn.functional.pad(gf.unsqueeze(0).unsqueeze(0), (1, 1), mode='replicate')
        neighbor_mean = (padded[0, 0, :-2] + padded[0, 0, 2:]) / 2.0
        gf += self.config.diffusion_rate * (neighbor_mean - gf)

        # 衰减
        gf *= 0.95
        # 防止溢出
        gf.clamp_(0.0, 10.0)

    # ---------- 辅助 ----------

    def _get_weight(self, module: nn.Module) -> Optional[nn.Parameter]:
        for pname, param in module.named_parameters():
            if 'weight' in pname:
                return param
        return None

    def _count_params(self, name: str, weight: nn.Parameter):
        total = weight.numel()
        active = (weight.data.abs() > self.config.weight_zero_threshold).sum().item()
        self.stats['total_params'] = total
        self.stats['active_params'] = active

    def get_summary(self) -> Dict:
        """获取系统状态摘要"""
        total_snapshots = sum(len(v) for v in self._snapshots.values())
        ratio = self.stats['active_params'] / max(1, self.stats['total_params'])
        return {
            'step_count': self.step_count,
            'n_modules': len(self._modules_map),
            'modules': list(self._modules_map.keys()),
            'active_param_ratio': ratio,
            'active_params': self.stats['active_params'],
            'total_params': self.stats['total_params'],
            'hibernated_neurons': total_snapshots,
            'stats': dict(self.stats),
            'avg_gf': {k: v.mean().item() for k, v in self._gf.items()} if self._gf else {},
        }

    def cleanup(self):
        self.tracker.remove_hooks()


# ============ 便捷函数 ============

def create_neural_pruning_system(**kwargs) -> NeuralPruningSystem:
    """创建神经修剪系统"""
    config = PruningConfig(**kwargs)
    return NeuralPruningSystem(config)


__all__ = [
    'PruningConfig',
    'NeuronState',
    'ActivityTracker',
    'NeuralPruningSystem',
    'create_neural_pruning_system',
]
