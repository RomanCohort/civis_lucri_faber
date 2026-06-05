"""
代谢预算模块 - Metable Budget Module

强制模型在解决问题时只能使用有限计算资源，模拟生物体代谢限制。
引入周期性饥饿机制，逼迫模型发掘更鲁棒的特征。

核心思想：
1. 代谢成本项 - 限制活跃神经元比例
2. 周期性饥饿 - 随机屏蔽最佳通路
3. 资源感知学习 - 让模型自己学会节能
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MetabolicState:
    """代谢状态"""
    active_ratio: float = 0.0      # 当前激活率
    budget: float = 0.3           # 预算上限
    starvation_count: int = 0      # 饥饿次数
    blocked_importance: float = 0.0  # 被屏蔽通路的重要性
    met_cost: float = 0.0          # 当前代谢成本


class MetabolicCostCalculator(nn.Module):
    """
    代谢成本计算器

    模拟生物体的能量代谢限制：
    - 强制模型只能激活X%的神经元
    - L1稀疏性惩罚促进紧凑表示
    - 超预算惩罚防止资源浪费
    """

    def __init__(
        self,
        resource_budget: float = 0.3,  # 30% 预算
        sparse_coef: float = 0.01,        # 稀疏系数
        overuse_penalty: float = 10.0,     # 超预算惩罚倍数
        warmup_steps: int = 1000,          # 预热步数
    ):
        super().__init__()
        self.budget = resource_budget
        self.sparse_coef = sparse_coef
        self.overuse_penalty = overuse_penalty
        self.warmup_steps = warmup_steps
        self.global_step = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_detail: bool = False
    ) -> tuple[torch.Tensor, dict | None]:
        """
        计算代谢成本

        Args:
            hidden_states: [batch, seq, hidden_dim] or [batch, hidden_dim]
            return_detail: 是否返回详细信息

        Returns:
            loss: 代谢成本
            detail: (optional) 详细信息字典
        """
        self.global_step += 1

        # 计算激活率（非零值比例）
        activation_rate = (hidden_states.abs() > 1e-6).float().mean()

        # L1稀疏性惩罚 - 促进紧凑表示
        sparse_penalty = hidden_states.abs().mean()

        # 对于超出预算的惩罚
        overuse_penalty = F.relu(activation_rate - self.budget)

        # 预热期：逐渐增加预算约束
        if self.global_step < self.warmup_steps:
            effective_budget = min(1.0, self.global_step / self.warmup_steps)
            overuse_penalty = F.relu(activation_rate - self.budget * effective_budget)
            budget_used = effective_budget
        else:
            budget_used = self.budget

        # 总成本
        met_cost = self.sparse_coef * sparse_penalty + self.overuse_penalty * overuse_penalty

        if return_detail:
            detail = {
                'activation_rate': activation_rate.item(),
                'budget': budget_used,
                'sparse_penalty': sparse_penalty.item(),
                'overuse_penalty': overuse_penalty.item(),
                'total_cost': met_cost.item(),
            }
            return met_cost, detail
        return met_cost, None

    def get_metabolic_state(self, hidden_states: torch.Tensor) -> MetabolicState:
        """获取当前代谢状态"""
        activation_rate = (hidden_states.abs() > 1e-6).float().mean()
        return MetabolicState(
            active_ratio=activation_rate.item(),
            budget=self.budget,
        )


class PeriodicStarvation:
    """
    周期性饥饿机制

    生物体在能量不足时被迫使用备用代谢通路。
    类似地，随机屏蔽当前表现最好的通路，
    逼迫模型发掘数据中更底层、更鲁棒的特征。

    这治疗AI的"捷径思维"有奇效。
    """

    def __init__(
        self,
        starvation_prob: float = 0.15,   # 饥饿概率
        min_block_ratio: float = 0.05,    # 最小屏蔽比例
        max_block_ratio: float = 0.30,      # 最大屏蔽比例
        cycle_steps: int = 500,           # 饥饿周期
        recovery_steps: int = 200,         # 恢复周期
        importance_decay: float = 0.95,   # 重要性衰减
    ):
        self.starvation_prob = starvation_prob
        self.min_block = min_block_ratio
        self.max_block = max_block_ratio
        self.cycle_steps = cycle_steps
        self.recovery_steps = recovery_steps
        self.decay = importance_decay

        self.global_step = 0
        self.last_starvation = 0
        self.importance_history = []  # 重要性分数历史
        self.current_mask = None  # 当前门控mask

    def compute_importance_scores(
        self,
        model: nn.Module,
        batch: dict,
        loss_fn: callable,
    ) -> dict[str, torch.Tensor]:
        """
        计算每个参数的重要性分数（基于梯度）

        Args:
            model: 模型
            batch: 输入数据
            loss_fn: 损失函数

        Returns:
            importance: 每个参数的重要性分数
        """
        model.zero_grad()

        # 前向传播
        output = model(**batch) if isinstance(batch, dict) else model(batch)

        # 计算损失
        if isinstance(output, dict):
            loss = output.get('loss', output.get('reward', 0))
        else:
            loss = output if isinstance(output, torch.Tensor) else 0

        # 反向传播
        if isinstance(loss, torch.Tensor):
            loss.backward(retain_graph=True)

        # 收集梯度作为重要性指标
        importance = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                importance[name] = param.grad.abs().detach()

        return importance

    def update_importance(self, new_scores: dict[str, torch.Tensor]):
        """更新重要性历史"""
        if not self.importance_history:
            self.importance_history = [
                {k: v.clone() for k, v in new_scores.items()}
            ]
        else:
            # 指数移动平均
            last = self.importance_history[-1]
            new_history = {}
            for k in new_scores:
                if k in last:
                    new_history[k] = self.decay * last[k] + (1 - self.decay) * new_scores[k]
                else:
                    new_history[k] = new_scores[k].clone()
            self.importance_history.append(new_history)

            # 限制历史长度
            if len(self.importance_history) > 10:
                self.importance_history.pop(0)

    def should_starve(self) -> bool:
        """判断是否应该进入饥饿状态"""
        if self.global_step - self.last_starvation < self.cycle_steps:
            return False
        return np.random.random() < self.starvation_prob

    def get_gate_mask(
        self,
        importance: dict[str, torch.Tensor],
        layer_name: str = None,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """
        获取门控mask

        Returns:
            masks: 每层的门���字��
            is_starving: 是否处于饥饿状态
        """
        self.global_step += 1
        is_starving = False
        masks = {}

        # 判断是否触发饥饿
        if self.should_starve():
            is_starving = True
            self.last_starvation = self.global_step

            # 随机决定屏蔽比例
            block_ratio = np.random.uniform(self.min_block, self.max_block)

            # 对每一层计算mask
            for name, scores in importance.items():
                if layer_name and name != layer_name:
                    masks[name] = torch.ones_like(scores)
                    continue

                # 屏蔽重要性最高的部分（强制模型用备用通路）
                flat_scores = scores.flatten()
                n_block = int(flat_scores.numel() * block_ratio)

                if n_block > 0:
                    _, indices = flat_scores.topk(n_block, largest=False)
                    mask = torch.ones_like(flat_scores)
                    mask[indices] = 0
                    masks[name] = mask.view_as(scores)
                else:
                    masks[name] = torch.ones_like(scores)

        else:
            # 非饥饿状态：全通过
            for name in importance:
                masks[name] = torch.ones_like(importance[name])

        self.current_mask = masks
        return masks, is_starving

    def get_starvation_stats(self) -> dict:
        """获取饥饿统计数据"""
        return {
            'global_step': self.global_step,
            'cycles_since_last': self.global_step - self.last_starvation,
            'is_starving': self.global_step - self.last_starvation < self.recovery_steps,
            'importance_history_len': len(self.importance_history),
        }


class ResourceAwareOptimizer:
    """
    资源感知优化器

    根据代谢状态动态调整学习率或梯度。
    当资源紧张时，降低学习率以节省"能量"。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 0.001,
        min_lr_ratio: float = 0.1,     # 资源紧张时的最小学习率比例
        resource_threshold: float = 0.8,  # 资源紧张阈值（激活率超过此值时）
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_ratio = min_lr_ratio
        self.threshold = resource_threshold

    def step(self, activation_rate: float):
        """根据激活率调整学习率并执行优化器步骤"""
        # 激活率越高，学习率越低（模拟能量紧张）
        if activation_rate > self.threshold:
            # 资源紧张，降低学习率
            lr_ratio = max(self.min_ratio, 1 - activation_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * lr_ratio
        else:
            # 资源充足，正常学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr

        self.optimizer.step()


class MetabolicBudget:
    """
    完整的代谢预算系统

    整合代谢成本、周期性饥饿和资源感知学习。
    """

    def __init__(
        self,
        resource_budget: float = 0.3,
        starvation_prob: float = 0.15,
        lambda_met: float = 0.01,         # 代谢成本权重
        warmup_steps: int = 1000,
    ):
        self.cost_calculator = MetabolicCostCalculator(
            resource_budget=resource_budget,
            warmup_steps=warmup_steps,
        )
        self.starvation = PeriodicStarvation(
            starvation_prob=starvation_prob,
            cycle_steps=warmup_steps // 2,
        )
        self.lambda_met = lambda_met
        self.state = MetabolicState()

    def compute_loss(
        self,
        task_loss: torch.Tensor,
        hidden_states: torch.Tensor,
        return_detail: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        计算带代谢成本的总体损失

        Args:
            task_loss: 任务损失
            hidden_states: 隐藏状态
            return_detail: 是否返回详情

        Returns:
            total_loss: 总损失
            detail: 详细信息
        """
        met_cost, detail = self.cost_calculator(hidden_states, return_detail=True)

        # 更新状态
        self.state.active_ratio = detail.get('activation_rate', 0)
        self.state.met_cost = met_cost.item()

        # 总损失 = 任务损失 + λ * 代谢成本
        total_loss = task_loss + self.lambda_met * met_cost

        if return_detail:
            detail['total_loss'] = total_loss.item()
            detail['task_loss'] = task_loss.item()
            detail['added_met_cost'] = (self.lambda_met * met_cost).item()
            return total_loss, detail
        return total_loss, None

    def update_starvation(
        self,
        model: nn.Module,
        batch: dict,
        loss_fn: callable,
    ):
        """更新饥饿机制的重要性分数"""
        importance = self.starvation.compute_importance_scores(model, batch, loss_fn)
        self.starvation.update_importance(importance)

    def get_gate_mask(self, layer_name: str = None) -> tuple[dict, bool]:
        """获取当前门控mask"""
        if not self.starvation.importance_history:
            return {}, False
        return self.starvation.get_gate_mask(
            self.starvation.importance_history[-1],
            layer_name
        )

    def get_state(self) -> MetabolicState:
        """获取当前代谢状态"""
        self.state.starvation_count = self.starvation.global_step - self.starvation.last_starvation
        return self.state

    def get_summary(self) -> dict:
        """获取摘要"""
        return {
            'active_ratio': self.state.active_ratio,
            'budget': self.state.budget,
            'met_cost': self.state.met_cost,
            'starvation_count': self.state.starvation_count,
        }


# ============ 便捷函数 ============

def create_metabolic_budget(
    resource_budget: float = 0.3,
    starvation_prob: float = 0.15,
    lambda_met: float = 0.01,
) -> MetabolicBudget:
    """创建代谢预算系统"""
    return MetabolicBudget(
        resource_budget=resource_budget,
        starvation_prob=starvation_prob,
        lambda_met=lambda_met,
    )


def compute_metabolic_loss(
    task_loss: torch.Tensor,
    hidden_states: torch.Tensor,
    budget: float = 0.3,
    lambda_met: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """便捷函数：计算带代谢成本的损失"""
    calc = MetabolicCostCalculator(resource_budget=budget)
    return calc(hidden_states, return_detail=True)


# ============ 注册到模块导出 ============

__all__ = [
    'MetabolicCostCalculator',
    'PeriodicStarvation',
    'ResourceAwareOptimizer',
    'MetabolicBudget',
    'MetabolicState',
    'create_metabolic_budget',
    'compute_metabolic_loss',
]
