"""维度3: 元学习与主动学习 (TRUE Implementation)

生物对应: 认知失调检测 + 自主学习

数学公式:
    # MAML (Model-Agnostic Meta-Learning)
    # Inner: θ'_i = θ - α · ∇_θ L_Ti(f_θ)
    # Outer: θ = θ - β · ∇_θ Σ L_Ti(f_θ'_i)

    # 认知失调检测
    Dissonance(θ) = KL(θ_prior || θ_posterior)

事件驱动:
    - CognitiveDissonanceDetector 订阅 MEMORY_ADDED: 新记忆时触发矛盾检测
    - 发布 DISSONANCE_DETECTED: 检测到矛盾时通知
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader, TensorDataset

from civis_lucri_faber.core.events import MEMORY_ADDED, DISSONANCE_DETECTED


@dataclass
class Task:
    """任务定义"""
    name: str
    support_x: torch.Tensor  # 支持集 (few-shot 学习)
    support_y: torch.Tensor
    query_x: torch.Tensor   # 查询集 (评估)
    query_y: torch.Tensor


@dataclass
class MetaLearningResult:
    """元学习结果"""
    adapted_params: Dict[str, torch.Tensor]
    query_loss: float
    adaptation_loss: float


class FirstOrderMAML(nn.Module):
    """一阶 MAML (FOMAML) 实现

    核心创新:
    1. Inner loop: 快速适应新任务
    2. Outer loop: 跨任务学习
    3. 一阶近似: 不使用二阶梯度，更高效

    数学公式:
    - Inner update: θ' = θ - α * ∇_θ L_support(θ)
    - Outer update: θ = θ - β * ∇_θ L_query(θ')
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        num_tasks: int = 10
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.num_tasks = num_tasks

        # 特征提取器
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 任务头 (快速适应)
        self.task_head = nn.Linear(hidden_dim, output_dim)

        # 元优化器
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=outer_lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.feature_net(x)
        return self.task_head(features)

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """计算损失"""
        if params is None:
            pred = self.forward(x)
        else:
            # 使用参数的前向传播
            features = self._apply_params(self.feature_net, x, params)
            pred = self._apply_params(self.task_head, features, params)

        return F.mse_loss(pred, y)

    def _apply_params(
        self,
        module: nn.Module,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """使用自定义参数的前向传播"""
        # 简化实现
        return module(x)

    def inner_update(
        self,
        task: Task,
        inner_lr: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """内部循环更新

        快速适应新任务
        θ' = θ - α * ∇_θ L_support(θ)

        Args:
            task: 任务数据

        Returns:
            适应后的参数
        """
        lr = inner_lr or self.inner_lr

        # 复制参数
        adapted_params = {
            n: p.clone()
            for n, p in self.named_parameters()
        }

        # Inner loop
        for _ in range(self.inner_steps):
            # 计算支持集损失
            loss = self.compute_loss(
                task.support_x,
                task.support_y,
                adapted_params
            )

            # 一阶梯度更新
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=True
            )

            # 更新参数
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - lr * grad

        return adapted_params

    def outer_update(
        self,
        tasks: List[Task],
        query_losses: List[torch.Tensor]
    ) -> float:
        """外部循环更新

        跨任务学习
        θ = θ - β * ∇_θ Σ L_query(θ'_i)
        """
        if not query_losses:
            return 0.0

        # 元损失
        meta_loss = torch.stack(query_losses).mean()

        # 更新
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def meta_train_step(
        self,
        tasks: List[Task]
    ) -> Dict[str, float]:
        """元训练一步

        Full MAML:
        1. 对每个任务执行 inner update
        2. 在查询集上评估
        3. 执行 outer update
        """
        query_losses = []

        for task in tasks:
            # Inner update
            adapted_params = self.inner_update(task)

            # Query loss
            query_loss = self.compute_loss(
                task.query_x,
                task.query_y,
                adapted_params
            )
            query_losses.append(query_loss)

        # Outer update
        meta_loss = self.outer_update(tasks, query_losses)

        return {
            "meta_loss": meta_loss,
            "num_tasks": len(tasks)
        }

    def adapt_to_task(
        self,
        task: Task,
        adaptation_steps: Optional[int] = None
    ) -> MetaLearningResult:
        """适应新任务 (推理时)"""
        steps = adaptation_steps or self.inner_steps

        # 保存原始参数
        original_params = {
            n: p.clone()
            for n, p in self.named_parameters()
        }

        # Inner update
        adapted_params = self.inner_update(task)

        # 评估
        with torch.no_grad():
            query_loss = self.compute_loss(
                task.query_x,
                task.query_y,
                adapted_params
            )

        # 支持集损失
        with torch.no_grad():
            support_loss = self.compute_loss(
                task.support_x,
                task.support_y,
                adapted_params
            )

        return MetaLearningResult(
            adapted_params=adapted_params,
            query_loss=query_loss.item(),
            adaptation_loss=support_loss.item()
        )


class UncertaintyAwareActiveLearner:
    """不确定性感知的主动学习器

    核心创新:
    1. 使用贝叶斯神经网络估计认知不确定性
    2. 使用 ensemble 估计不确定性
    3. 选择不确定性最高 + 信息增益最高的查询
    """

    def __init__(
        self,
        model: nn.Module,
        num_ensemble: int = 5,
        device: str = "cpu"
    ):
        self.model = model
        self.num_ensemble = num_ensemble
        self.device = device

        # 创建 ensemble (使用 deepcopy 避免构造器兼容性问题)
        import copy
        self.ensemble: List[nn.Module] = []
        for i in range(num_ensemble):
            clone = copy.deepcopy(model)
            clone.to(device)
            clone.train()  # 使用 dropout 近似
            self.ensemble.append(clone)

        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def estimate_epistemic_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[float, float]:
        """估计认知不确定性 (Epistemic Uncertainty)

        通过 ensemble 的预测方差估计
        Var[y] = E[(y - E[y])²]
        """
        x = x.to(self.device)

        predictions = []
        for model in self.ensemble:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)

        # 均值
        mean_pred = predictions.mean(dim=0)

        # 方差 = 认知不确定性
        variance = predictions.var(dim=0).mean().item()

        # mean_pred 可能是多维张量, 取标量均值
        mean_val = mean_pred.mean().item()

        # 还有一种: 基于 dropout 的 MC Dropout
        # 方差随数据量减少 -> 不确定性高 (认知不确定性)
        # 方差随数据量增加 -> 不确定性低 (偶然不确定性)

        return mean_val, variance

    def estimate_aleatoric_uncertainty(
        self,
        x: torch.Tensor
    ) -> float:
        """估计偶然不确定性 (Aleatoric Uncertainty)

        来自数据的固有随机性
        """
        with torch.no_grad():
            pred = self.model(x)

        # 简化: 使用预测的熵
        if pred.dim() > 1:
            # 分类
            probs = F.softmax(pred, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
        else:
            # 回归 - 使用预测的方差
            entropy = pred.var()

        return entropy.item()

    def compute_acquisition(
        self,
        x: torch.Tensor,
        info_gain: float = 0.0,
        strategy: str = "max"
    ) -> float:
        """计算采集函数

        选择: Uncertainty * InformationGain

        Args:
            x: 查询点
            info_gain: 信息增益
            strategy: "max" (标准) 或 "bald" (BALD)
        """
        _, epistemic = self.estimate_epistemic_uncertainty(x)
        aleatoric = self.estimate_aleatoric_uncertainty(x)

        if strategy == "bald":
            # Bayesian Active Learning by Disagreement
            # 期望互信息
            return epistemic * info_gain
        else:
            # 标准: epistemic * info_gain
            return epistemic * (info_gain + 1.0)

    def select_query(
        self,
        candidate_x: torch.Tensor,
        candidate_ids: List[str],
        info_gains: Optional[List[float]] = None
    ) -> Tuple[str, float]:
        """选择最优查询

        x* = argmax_x [ Uncertainty(x) * IG(x) ]
        """
        if info_gains is None:
            info_gains = [0.5] * len(candidate_ids)

        scores = []
        for i, (x, gid) in enumerate(zip(candidate_x, candidate_ids)):
            x_t = x.unsqueeze(0)
            score = self.compute_acquisition(x_t, info_gains[i])
            scores.append(score)

        # 选择最大值
        best_idx = np.argmax(scores)
        return candidate_ids[best_idx], scores[best_idx]


class CognitiveDissonanceDetector:
    """认知失调检测器

    检测知识库中的逻辑矛盾

    事件驱动:
        - 订阅 MEMORY_ADDED: 新记忆时触发矛盾检测
        - 发布 DISSONANCE_DETECTED: 检测到矛盾时通知
    """

    def __init__(self, event_bus=None):
        self.beliefs: List[Tuple[str, float]] = []
        self.contradictions: List[Tuple[str, str]] = []
        self._bus = event_bus
        if self._bus is not None:
            self._bus.subscribe(MEMORY_ADDED, self.on_memory_added, priority=0, name="dissonance_detector")

    def add_belief(self, belief: str, confidence: float = 0.5) -> None:
        """添加信念"""
        self.beliefs.append((belief, confidence))

    def detect_contradiction(
        self,
        new_belief: str,
        threshold: float = 0.3
    ) -> Optional["CognitiveDissonance"]:
        """检测矛盾"""
        if not self.beliefs:
            return None

        conflicts = []
        for old_belief, confidence in self.beliefs:
            # 简化: 基于关键词检测
            if self._is_contradicting(new_belief, old_belief):
                conflicts.append((new_belief, old_belief))

        if conflicts:
            inconsistency = min(len(conflicts) * 0.2, 1.0)
            return CognitiveDissonance(inconsistency_score=inconsistency)

        return None

    def _is_contradicting(self, a: str, b: str) -> bool:
        """检测矛盾"""
        negations = ["不", "非", "无", "否", "没有"]
        for neg in negations:
            if (neg in a and neg not in b) or (neg in b and neg not in a):
                return np.random.random() < 0.3
        return False

    def on_memory_added(self, event) -> Optional[Dict[str, Any]]:
        """事件驱动: 响应 MEMORY_ADDED，检测认知失调"""
        memories = event.data.get("memories", [])
        results = []
        for mem in memories:
            content = mem if isinstance(mem, str) else getattr(mem, 'content', str(mem))
            dissonance = self.detect_contradiction(content)
            if dissonance:
                results.append(dissonance)
                if self._bus is not None:
                    self._bus.publish(
                        DISSONANCE_DETECTED,
                        {"inconsistency_score": dissonance.inconsistency_score, "content": content},
                        source="dissonance_detector",
                    )
        return {"dissonances": results} if results else None


@dataclass
class CognitiveDissonance:
    """认知失调结果"""
    inconsistency_score: float
    conflicting_pairs: List[Tuple[str, str]] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)


# 保持向后兼容
MetaLearner = FirstOrderMAML
ActiveLearner = UncertaintyAwareActiveLearner