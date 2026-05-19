"""
预测编码系统 (Predictive Coding / Free Energy Principle)

模拟大脑作为预测机器的运作方式：
1. 生成层 (Generative Layer) - 自上而下预测
2. 层级生成模型 (Hierarchical Generative Model) - 三层感知/特征/概念
3. 精度调制器 (Precision Modulator) - 注意力=预测误差精度加权
4. 主动推理控制器 (Active Inference Controller) - 通过行动减少预测误差

核心原理:
- 大脑持续生成对感官输入的预测
- 预测误差 = 实际输入 - 预测输入
- 自由能 F = 复杂度 + 不准确性
- 系统通过更新模型和行动来最小化自由能

生物参考文献:
- Friston (2010): 自由能原理
- Friston (2005): 大脑中的层级模型
- Feldman & Friston (2010): 注意力、不确定性与自由能
- Clark (2013): 预测大脑
- Friston et al. (2012): 主动推理
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
class PredictiveCodingState:
    """预测编码状态"""
    free_energy: float = 0.5              # 当前自由能
    prediction_error_total: float = 0.0   # 总加权预测误差
    precision: float = 0.5                # 总体精度加权
    learning_rate_predictive: float = 0.01  # 生成模型更新率
    active_inference_drive: float = 0.0   # 主动推理驱力 [0,1]
    surprise: float = 0.0                 # 惊讶度 [0,1]
    attention_weights: Optional[List[float]] = None  # 注意力权重


# ============ 生成层 ============

class GenerativeLayer(nn.Module):
    """
    生成层 - 单层预测/误差计算

    每一层:
    - 接收上层预测 (top-down)
    - 生成对下层的预测
    - 计算预测误差 (bottom-up input - prediction)

    参考: Friston (2005) "A theory of cortical responses"
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # 自上而下预测网络
        self.prediction_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        # 误差计算网络
        self.error_net = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        # 层级状态
        self.state = nn.Parameter(torch.zeros(hidden_dim), requires_grad=False)
        self.last_prediction = None
        self.last_error = None

    def forward(self, bottom_up_input: torch.Tensor,
                top_down_prediction: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        生成预测并计算预测误差

        Args:
            bottom_up_input: 来自下层的输入
            top_down_prediction: 来自上层的预测
        """
        # 如果有上层预测, 更新本层状态
        if top_down_prediction is not None:
            state_update = F.adaptive_avg_pool1d(
                top_down_prediction.unsqueeze(0).unsqueeze(0),
                self.state.shape[0]
            ).squeeze()
            self.state.data = 0.9 * self.state.data + 0.1 * state_update

        # 生成本层预测 (传给下层)
        prediction = self.prediction_net(self.state.unsqueeze(0)).squeeze(0)

        # 计算预测误差
        if bottom_up_input.shape[0] != prediction.shape[0]:
            # 维度不匹配时用池化适配
            target_dim = min(bottom_up_input.shape[0], prediction.shape[0])
            bu_pooled = F.adaptive_avg_pool1d(
                bottom_up_input.unsqueeze(0).unsqueeze(0), target_dim
            ).squeeze()
            pred_pooled = F.adaptive_avg_pool1d(
                prediction.unsqueeze(0).unsqueeze(0), target_dim
            ).squeeze()
            error_input = torch.cat([bu_pooled, pred_pooled])
        else:
            error_input = torch.cat([bottom_up_input, prediction])

        error = self.error_net(error_input)

        self.last_prediction = prediction.detach()
        self.last_error = error.detach()

        return {
            'prediction': prediction,
            'error': error,
            'state': self.state.clone(),
        }


# ============ 层级生成模型 ============

class HierarchicalGenerativeModel(nn.Module):
    """
    层级生成模型

    三层结构:
    - Layer 0 (Sensory): 感觉层 - 直接处理感官输入
    - Layer 1 (Feature): 特征层 - 提取和预测特征
    - Layer 2 (Conceptual): 概念层 - 高层抽象预测

    每层预测下一层, 误差向上传播

    参考: Friston (2005) - 大脑中的层级模型
    """

    def __init__(self, sensory_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_layers = n_layers
        dims = [sensory_dim]
        for i in range(1, n_layers):
            dims.append(max(16, sensory_dim // (2 ** i)))

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            input_dim = dims[i] if i == 0 else dims[i]
            hidden_dim = dims[i]
            output_dim = dims[i - 1] if i > 0 else dims[i]
            self.layers.append(GenerativeLayer(input_dim, hidden_dim, output_dim))

        # 每层精度权重
        self.precision_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(n_layers)
        ])

    def forward(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        """
        层级前向传播

        自下而上: 误差传播
        自上而下: 预测传播
        """
        all_errors = []
        all_predictions = []
        all_precisions = []

        # 自下而上传播
        current_input = sensory_input
        for i, layer in enumerate(self.layers):
            top_down = None
            if i > 0 and self.layers[i - 1].last_prediction is not None:
                top_down = self.layers[i - 1].last_prediction

            result = layer.forward(current_input, top_down)

            precision = torch.sigmoid(self.precision_weights[i])
            weighted_error = precision * result['error']

            all_errors.append(result['error'])
            all_predictions.append(result['prediction'])
            all_precisions.append(precision.item())

            # 误差向上传递
            current_input = result['error']

        return {
            'errors': all_errors,
            'predictions': all_predictions,
            'precisions': all_precisions,
            'layer_states': [layer.state.clone() for layer in self.layers],
        }

    def update_from_errors(self, errors: List[torch.Tensor], lr: float = 0.01):
        """
        从预测误差更新生成模型 (感知/学习)

        最小化预测误差损失: loss = sum(precision_i * MSE(input_i, prediction_i))
        """
        total_loss = torch.tensor(0.0)
        for i, (error, layer) in enumerate(zip(errors, self.layers)):
            precision = torch.sigmoid(self.precision_weights[i])
            total_loss = total_loss + precision * (error ** 2).mean()

        if total_loss.requires_grad:
            total_loss.backward(retain_graph=False)

        # 手动梯度下降更新参数
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad
                    param.grad.zero_()


# ============ 精度调制器 ============

class PrecisionModulator(nn.Module):
    """
    精度调制器

    精度 = 预测误差的预期可靠性 (类似卡尔曼增益)
    精度由神经调质调节:
    - 多巴胺: 增加精度 (更信任预测误差 = 更注意)
    - 乙酰胆碱: 增加精度 (增强注意力聚焦)
    - 不确定性: 降低精度 (不确定时不信任误差)

    注意力 = 精度加权预测误差

    参考: Feldman & Friston (2010)
    """

    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.n_layers = n_layers

        # 神经调质对精度的影响权重
        self.dopamine_weight = nn.Parameter(torch.tensor(0.4))
        self.ach_weight = nn.Parameter(torch.tensor(0.3))
        self.uncertainty_weight = nn.Parameter(torch.tensor(-0.3))

        # 精度历史追踪
        self.precision_history = deque(maxlen=100)

    def forward(self, dopamine_level: float, ach_level: float,
                uncertainty: float) -> Dict[str, Any]:
        """
        计算精度加权

        precision = sigmoid(w_da * dopamine + w_ach * ach + w_unc * uncertainty)
        """
        precision_signal = (
            self.dopamine_weight.item() * dopamine_level
            + self.ach_weight.item() * ach_level
            + self.uncertainty_weight.item() * uncertainty
        )
        precision = float(torch.sigmoid(torch.tensor(precision_signal)))

        # 每层精度 (可略有差异)
        layer_precisions = []
        for i in range(self.n_layers):
            # 低层(感官)精度更依赖ACh, 高层(概念)更依赖DA
            layer_signal = (
                (self.dopamine_weight.item() * (0.5 + 0.1 * i)) * dopamine_level
                + (self.ach_weight.item() * (1.0 - 0.1 * i)) * ach_level
                + self.uncertainty_weight.item() * uncertainty
            )
            layer_precisions.append(float(torch.sigmoid(torch.tensor(layer_signal))))

        self.precision_history.append(precision)

        return {
            'precision': precision,
            'layer_precisions': layer_precisions,
            'dopamine_contribution': self.dopamine_weight.item() * dopamine_level,
            'ach_contribution': self.ach_weight.item() * ach_level,
        }


# ============ 主动推理控制器 ============

class ActiveInferenceController(nn.Module):
    """
    主动推理控制器

    当预测误差无法通过更新信念解决时,
    系统通过行动使预测成真

    主动推理 = 通过行动减少预测误差
    - 认识驱动 (Epistemic): 寻求信息以减少不确定性
    - 实用驱动 (Pragmatic): 通过行动实现预期状态

    好奇心 = 认识驱动的一种形式

    参考: Friston et al. (2012) "Active inference and agency"
    """

    def __init__(self, n_actions: int = 4, state_dim: int = 64):
        super().__init__()
        self.action_selection_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=-1),
        )
        # 认识驱动 (好奇心) — 自适应: 自由能高时增强，低时减退
        self.epistemic_drive = 0.5
        self.drive_history = deque(maxlen=100)
        self._adaptive_rate = 0.01  # 认识驱动的自适应速率

    def forward(self, prediction_errors: List[torch.Tensor],
                precision: float) -> Dict[str, Any]:
        """
        计算主动推理驱力和行动偏好

        active_inference_drive = sigmoid(sum(weighted_errors) - threshold)
        """
        # 总加权预测误差
        total_error = sum(pe.abs().mean().item() for pe in prediction_errors)
        weighted_total = total_error * precision

        # 主动推理驱力: 当加权误差超过阈值时激活
        threshold = 0.3
        drive = float(torch.sigmoid(torch.tensor(weighted_total - threshold)))
        drive = drive * (1.0 + self.epistemic_drive)  # 认识驱动增强
        drive = min(drive, 1.0)

        self.drive_history.append(drive)

        # 行动偏好 (基于误差模式)
        if prediction_errors and len(prediction_errors) > 0:
            error_flat = prediction_errors[0].flatten()
            target_dim = min(64, error_flat.shape[0])
            if error_flat.shape[0] >= target_dim:
                error_pooled = error_flat[:target_dim]
            else:
                error_pooled = F.pad(error_flat, (0, target_dim - error_flat.shape[0]))

            action_preferences = self.action_selection_net(error_pooled.unsqueeze(0)).squeeze(0)
        else:
            action_preferences = torch.ones(4) / 4

        return {
            'active_inference_drive': drive,
            'action_preferences': action_preferences,
            'epistemic_drive': self.epistemic_drive,
            'total_weighted_error': weighted_total,
        }


# ============ 预测编码系统 (聚合器) ============

class PredictiveCodingSystem(nn.Module):
    """
    预测编码系统 - 自由能原理实现

    大脑作为预测机器:
    - 持续生成对世界的预测
    - 比较预测与实际输入
    - 通过两种方式最小化自由能:
      1. 更新内部模型 (感知/学习)
      2. 行动改变输入 (主动推理)

    参考:
    - Friston (2010): 自由能原理
    - Clark (2013): "Whatever next? Predictive brains..."
    """

    def __init__(self, sensory_dim: int = 64, n_layers: int = 3,
                 learning_rate: float = 0.01, event_bus=None):
        super().__init__()
        self.model = HierarchicalGenerativeModel(
            sensory_dim=sensory_dim, n_layers=n_layers
        )
        self.precision_modulator = PrecisionModulator(n_layers=n_layers)
        self.active_inference = ActiveInferenceController(
            n_actions=4, state_dim=sensory_dim
        )
        self.state = PredictiveCodingState()
        self.learning_rate = learning_rate
        self.step_count = 0
        self.free_energy_history = deque(maxlen=200)

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "brain_update",
                self._handle_brain_update,
                priority=0,
                name="predictive_coding",
            )

    def _handle_brain_update(self, event) -> Dict:
        """Event-driven handler for brain_update events."""
        import torch as _torch
        state = event.data.get("internal_state", {})
        dopamine_level = state.get("dopamine_level", 0.5)
        acetylcholine = state.get("acetylcholine", 0.5)
        uncertainty = 1.0 - state.get("alignment_score", 0.5)
        state_tensor = event.data.get("state_tensor", _torch.randn(1, 64))

        # 使用真实状态向量作为感官输入
        sensory_input = state_tensor.squeeze(0) if state_tensor.dim() == 2 else state_tensor

        result = self.step(
            sensory_input=sensory_input,
            dopamine_level=dopamine_level,
            ach_level=acetylcholine,
            uncertainty=uncertainty,
        )

        state["free_energy"] = result["free_energy"]
        state["prediction_error"] = result["prediction_error"]
        state["precision"] = result["precision"]
        state["active_inference_drive"] = result["active_inference_drive"]
        state["attention_weights"] = result["attention_weights"]

        return result

    def step(self, sensory_input: torch.Tensor,
             dopamine_level: float = 0.5, ach_level: float = 0.5,
             uncertainty: float = 0.5) -> Dict[str, Any]:
        """
        执行一个预测编码步

        Args:
            sensory_input: 感官输入张量
            dopamine_level: 多巴胺水平 [0,1]
            ach_level: 乙酰胆碱水平 [0,1]
            uncertainty: 不确定性水平 [0,1]
        """
        self.step_count += 1

        # 1. 层级生成模型前向传播
        model_result = self.model.forward(sensory_input)

        # 2. 精度调制
        precision_result = self.precision_modulator.forward(
            dopamine_level, ach_level, uncertainty
        )

        # 3. 更新模型精度权重
        for i, layer_precision in enumerate(precision_result['layer_precisions']):
            with torch.no_grad():
                self.model.precision_weights[i].fill_(
                    float(torch.logit(torch.tensor(max(0.01, min(0.99, layer_precision)))))
                )

        # 4. 计算自由能
        free_energy = self._compute_free_energy(
            model_result['errors'],
            precision_result['layer_precisions']
        )

        # 5. 主动推理
        inference_result = self.active_inference.forward(
            model_result['errors'],
            precision_result['precision']
        )

        # 6. 计算惊讶度 (自由能的上界)
        surprise = float(np.clip(free_energy * 1.2, 0.0, 1.0))

        # 7. 从误差更新生成模型 (连续学习, 速率与自由能成正比)
        # 当前自由能高 → 更大更新步长 (更急切地学习以减少误差)
        # 自由能低 → 微调 (已接近最优)
        effective_lr = self.learning_rate * float(np.clip(free_energy * 2.0, 0.1, 3.0))
        self.model.update_from_errors(model_result['errors'], lr=effective_lr)

        # 更新认识驱动 (自适应)
        # 自由能持续高 → 好奇心增强 (寻求更多信息)
        # 自由能持续低 → 好奇心减退 (已充分理解)
        fe_trend = free_energy - (float(np.mean(list(self.free_energy_history)[-20:]))) if len(self.free_energy_history) >= 20 else free_energy - 0.5
        self.active_inference.epistemic_drive = float(np.clip(
            self.active_inference.epistemic_drive + self.active_inference._adaptive_rate * fe_trend,
            0.1, 1.0
        ))

        # 8. 计算注意力权重
        attention_weights = precision_result['layer_precisions']

        # 9. 更新状态
        self.state = PredictiveCodingState(
            free_energy=free_energy,
            prediction_error_total=inference_result['total_weighted_error'],
            precision=precision_result['precision'],
            learning_rate_predictive=self.learning_rate,
            active_inference_drive=inference_result['active_inference_drive'],
            surprise=surprise,
            attention_weights=attention_weights,
        )

        self.free_energy_history.append(free_energy)

        return {
            'free_energy': free_energy,
            'prediction_error': inference_result['total_weighted_error'],
            'precision': precision_result['precision'],
            'active_inference_drive': inference_result['active_inference_drive'],
            'attention_weights': attention_weights,
            'surprise': surprise,
            'layer_precisions': precision_result['layer_precisions'],
            'action_preferences': inference_result['action_preferences'].tolist(),
        }

    def _compute_free_energy(self, errors: List[torch.Tensor],
                              precisions: List[float]) -> float:
        """
        计算自由能

        F = Complexity + Inaccuracy
        简化实现: F = sum(precision_i * MSE_i)
        加上复杂度正则项

        参考: Friston (2010)
        """
        accuracy = 0.0
        for i, error in enumerate(errors):
            precision = precisions[i] if i < len(precisions) else 0.5
            accuracy += precision * (error ** 2).mean().item()

        # 复杂度正则 (防止模型过于复杂)
        complexity = 0.0
        for param in self.model.parameters():
            complexity += param.data.norm(2).item() * 0.001

        free_energy = accuracy + complexity
        return float(np.clip(free_energy, 0.0, 1.0))

    def compute_free_energy(self) -> float:
        """获取当前自由能"""
        return self.state.free_energy

    def get_attention_weights(self) -> List[float]:
        """获取当前注意力权重"""
        return self.state.attention_weights or [0.5] * self.model.n_layers

    def get_summary(self) -> Dict:
        """获取预测编码系统摘要"""
        fe_history = list(self.free_energy_history)
        return {
            'free_energy': self.state.free_energy,
            'prediction_error': self.state.prediction_error_total,
            'precision': self.state.precision,
            'surprise': self.state.surprise,
            'active_inference_drive': self.state.active_inference_drive,
            'avg_free_energy': float(np.mean(fe_history)) if fe_history else 0.5,
            'step_count': self.step_count,
        }


def create_predictive_coding_system(**kwargs) -> PredictiveCodingSystem:
    """工厂函数: 创建预测编码系统"""
    return PredictiveCodingSystem(**kwargs)


__all__ = [
    'PredictiveCodingState',
    'GenerativeLayer',
    'HierarchicalGenerativeModel',
    'PrecisionModulator',
    'ActiveInferenceController',
    'PredictiveCodingSystem',
    'create_predictive_coding_system',
]
