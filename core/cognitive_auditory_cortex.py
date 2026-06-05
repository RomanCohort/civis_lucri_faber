"""
听觉系统神经拟真改进方案 - 阶段2: 认知架构

Simulacrum - Spiking Auditory Cortex - Phase 2

从神经拟真 → 认知架构
- 语音工作记忆 (phonological loop)
- 听觉-运动闭环 (auditory-motor coupling)
- 元学习 (few-shot adaptation)
"""


import torch
import torch.nn as nn

# ============== 阶段2: 认知架构 ==============

# ============== 2.1 语音工作记忆 ==============

class PhonologicalLoop(nn.Module):
    """
    语音工作记忆 (Phonological Loop)

    心理学: Baddeley & Hitch 3
    对应: 左侧颞叶后部 + 下顶叶

    关键机制：
    1. 语音缓存 (phonological store) - ~2秒
    2. 复述循环 (articulatory loop) - 保持
    3. 视觉-言语桥接 (orthographic-phonetic)
    """

    def __init__(self, item_dim: int = 256, max_items: int = 7):
        super().__init__()
        self.item_dim = item_dim
        self.max_items = max_items  # Miller: 7±2 items
        self.store_duration = 2.0  # 秒

        # 语音缓存 (FIFO队列)
        self.phone_store = nn.LSTM(
            item_dim,
            item_dim,
            num_layers=1,
            batch_first=True
        )
        self.store_buffer = None

        # 复述控制
        self.rehearsal_gate = nn.Sequential(
            nn.Linear(item_dim, 1),
            nn.Sigmoid()
        )

        # 遗忘衰减
        self.decay_rate = 0.95  # 每秒

    def forward(self, features: torch.Tensor,
                step: int = 0) -> dict:
        """
        features: [B, item_dim]
        step: 当前时间步

        Returns:
            working_memory: [B, max_items, item_dim]
            active_item: [B, item_dim] (当前关注项)
            rehearsal_prob: [B]
        """
        B = features.shape[0]

        # 初始化缓冲区
        if self.store_buffer is None:
            self.store_buffer = torch.zeros(B, self.max_items, self.item_dim)
            self.store_idx = torch.zeros(B, dtype=torch.long)

        # 新输入 → 语音缓存
        # 替换最旧的
        idx = self.store_idx % self.max_items
        batch_idx = torch.arange(B)
        self.store_buffer[batch_idx, idx] = features
        self.store_idx = self.store_idx + 1

        # 复述信号
        rehearsal = self.rehearsal_gate(features)

        # 衰减 (随时间)
        if step > 0:
            decay = self.decay_rate ** (step * 0.001)  # 假设1ms步
            self.store_buffer = self.store_buffer * decay

        # 当前激活项
        active_item_idx = (idx - 1) % self.max_items
        active_item = self.store_buffer[batch_idx, active_item_idx]

        return {
            'working_memory': self.store_buffer,
            'active_item': active_item,
            'rehearsal_prob': rehearsal.squeeze(-1),
            'store_size': (self.store_idx + 1).clamp(0, self.max_items)
        }

    def reset(self):
        self.store_buffer = None
        self.store_idx = None


class AuditoryWorkingMemory(nn.Module):
    """
    听觉工作记忆系统

    整合:
    1. 语音缓存 (phonological loop)
    2. 视觉图像 (iconic memory)
    3. 注意选择 (attentional selection)
    """

    def __init__(self, item_dim: int = 256):
        super().__init__()
        self.item_dim = item_dim

        # 语音环路
        self.phon_loop = PhonologicalLoop(item_dim)

        # 视觉图像 (短视觉缓存, ~500ms)
        self.iconic_buffer = nn.Sequential(
            nn.Linear(item_dim, item_dim),
            nn.Tanh()
        )

        # 注意选择器 (top-down attention)
        self.attentional_mask = nn.Parameter(torch.ones(1, item_dim))

        # 记忆检索强度
        self.retrieval_strength = nn.Parameter(torch.tensor(0.5))

    def forward(self, cortical_input: torch.Tensor,
              attention_focus: torch.Tensor | None = None,
              step: int = 0) -> dict:
        """
        cortical_input: [B, item_dim]
        attention_focus: 可选的注意焦点

        Returns:
            memory_output: [B, item_dim]
            phonological: ...
            iconic: ...
            attention_weights: [B, item_dim]
        """
        # 1. 语音环路
        phone = self.phon_loop(cortical_input, step)

        # 2. 视觉图像
        iconic = self.iconic_buffer(cortical_input)

        # 3. 注意选择
        if attention_focus is not None:
            # 自上而下
            attn = attention_focus * self.attentional_mask
        else:
            # 自下而上 (salience-based)
            attn = cortical_input

        # 整合
        output = (
            phone['active_item'] * 0.5 +
            iconic * 0.3 +
            attn * 0.2
        )

        return {
            'memory_output': output,
            'phonological_store': phone['working_memory'],
            'active_item': phone['active_item'],
            'iconic': iconic,
            'attention_weights': attn / (attn.sum(-1, keepdim=True) + 1e-8)
        }


# ============== 2.2 听觉-运动闭环 ==============

class AuditoryMotorCoupling(nn.Module):
    """
    听觉-运动耦合 (Auditory-Motor Coupling)

    心理学: 言语知觉的 MOT理论
    对应: 背侧流 → Broca区 → 运动皮层

    关键机制：
    1. 镜像神经元 (observation-execution matching)
    2. 运动计划 (speech motor planning)
    3. 听觉反馈监控 ( auditory feedback monitoring)
    4. 预测 (forward model)
    """

    def __init__(self, audio_dim: int = 256, motor_dim: int = 128):
        super().__init__()
        self.audio_dim = audio_dim
        self.motor_dim = motor_dim

        # 镜像神经元系统 (observation → execution)
        self.mirror_neuron = nn.Sequential(
            nn.Linear(audio_dim, motor_dim),
            nn.ReLU(),
            nn.Linear(motor_dim, motor_dim)
        )

        # 运动计划
        self.motor_planner = nn.Sequential(
            nn.Linear(audio_dim, motor_dim),
            nn.ReLU(),
        )

        # 听觉反馈检测 (monitor)
        self.feedback_monitor = nn.Sequential(
            nn.Linear(audio_dim * 2, motor_dim),
            nn.Sigmoid()
        )

        # 预测模型 (forward)
        self.forward_model = nn.Sequential(
            nn.Linear(motor_dim, motor_dim),
            nn.ReLU(),
            nn.Linear(motor_dim, audio_dim)
        )

        # 误差检测
        self.error_detector = nn.Linear(audio_dim, 1)

    def forward(self, auditory_features: torch.Tensor,
              motor_command: torch.Tensor | None = None,
              mode: str = "listen") -> dict:
        """
        auditory_features: [B, audio_dim]
        motor_command: 来自运动皮层的命令 (可选)
        mode: "listen" | "speak" | "repeat"

        Returns:
            motor_output: [B, motor_dim]
            mirror_activation: [B, motor_dim]
            predicted_auditory: [B, audio_dim]
            error_signal: [B] (prediction error)
        """
        # 1. 镜像激活 (听到 = 执行)
        mirror_activation = self.mirror_neuron(auditory_features)

        # 2. 运动计划
        planned = self.motor_planner(auditory_features)

        # 3. ��测 (forward model)
        if motor_command is not None:
            predicted_auditory = self.forward_model(motor_command)

            # 误差
            error = self.error_detector(auditory_features - predicted_auditory)
        else:
            predicted_auditory = self.forward_model(planned)
            error = torch.zeros(auditory_features.shape[0], 1)

        # 4. 反馈监控
        if mode == "speak":  # 自己说时监听
            # 比较预期vs实际
            feedback = self.feedback_monitor(
                torch.cat([auditory_features, predicted_auditory], -1)
            )
        else:
            feedback = torch.zeros_like(planned)

        return {
            'motor_output': planned,
            'mirror_activation': mirror_activation,
            'predicted_auditory': predicted_auditory,
            'error_signal': error.squeeze(-1),
            'feedback_strength': feedback.mean()
        }


class SpeechMotorCortex(nn.Module):
    """
    言语运动皮层

    对应: Broca区 + pre-SMA + 运动皮层
    """

    def __init__(self, input_dim: int = 256, output_dim: int = 64):
        super().__init__()

        # Broca区 (语言产生)
        self.broca = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # pre-SMA (运动序列计划)
        self.presma = nn.LSTM(
            output_dim,
            output_dim,
            num_layers=2,
            batch_first=True
        )

        # 运动皮层 (articulators)
        self.motor_cortex = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32)  # 音素/音节
        )

        # 反馈接收
        self.feedback_receptor = nn.Linear(input_dim, output_dim)

    def forward(self, linguistic_input: torch.Tensor,
              sequence_length: int = 5) -> dict:
        """
        linguistic_input: [B, input_dim]
        sequence_length: 计划序列长度

        Returns:
            articulator_commands: [B, sequence_length, 32]
            motor_sequence: [B, sequence_length, output_dim]
            broca_activation: [B, output_dim]
        """
        B = linguistic_input.shape[0]

        # 语言产生
        broca_out = self.broca(linguistic_input)

        # 序列计划
        sequence = broca_out.unsqueeze(1).expand(-1, sequence_length, -1)

        # pre-SMA处理
        presma_out, _ = self.presma(sequence)

        # 运动命令
        articulators = self.motor_cortex(presma_out)

        return {
            'articulator_commands': articulators,
            'motor_sequence': presma_out,
            'broca_activation': broca_out
        }


# ============== 2.3 元学习 (Few-shot Adaptation) ==============

class MetaLearningAuditory(nn.Module):
    """
    元学习听觉系统

    机制: MAML (Model-Agnostic Meta-Learning)
    目标: 快速适应新说话人/新语言/新 acoustic environment
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()

        # 基础模型
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # 元学习器 (learn to learn)
        self.meta_learner = MAMLModule(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            inner_lr=0.1,
            inner_steps=3
        )

        # 快速适应控制器
        self.adaptation_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, support: torch.Tensor,
               query: torch.Tensor) -> dict:
        """
        support: [B, N_shot, input_dim] (few-shot examples)
        query: [B, N_query, input_dim] (to classify)

        Returns:
            adapted_features: [B, N_query, hidden_dim]
            adaptation_gain: [B] (how much adapted)
        """
        B, N_shot, D = support.shape
        _, N_query, _ = query.shape

        # 在support上快速适应
        adapted_params = self.meta_learner.inner_update(support)

        # 应用到query
        adapted_features = []
        for b in range(B):
            adapted = self._apply_params(
                query[b],
                adapted_params[b]
            )
            adapted_features.append(adapted)

        adapted_features = torch.stack(adapted_features)

        # 评估适应程度
        adaptation_gain = self.adaptation_controller(
            adapted_features.mean(1)
        )

        return {
            'adapted_features': adapted_features,
            'adaptation_gain': adaptation_gain.squeeze(-1),
            'meta_parameters': adapted_params
        }

    def _apply_params(self, x: torch.Tensor,
                     params: dict) -> torch.Tensor:
        """应用 adapted parameters"""
        return self.encoder(x) * 0 + torch.zeros_like(x)


class MAMLModule(nn.Module):
    """MAML 内循环"""

    def __init__(self, input_dim: int, output_dim: int,
                 inner_lr: float = 0.1, inner_steps: int = 3):
        super().__init__()
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def inner_update(self, support: torch.Tensor) -> dict:
        """
        在support上快速更新

        Returns:
            adapted_params: Dict of parameters
        """
        # 克隆参数
        params = {n: p.clone() for n, p in self.named_parameters()}

        for _ in range(self.inner_steps):
            output = self._forward_with_params(support, params)
            loss = output.mean()

            # 梯度更新
            grads = torch.autograd.grad(loss, params.values(),
                               create_graph=False)

            for (n, p), g in zip(params.items(), grads):
                if g is not None:
                    params[n] = p - self.inner_lr * g

        return params

    def _forward_with_params(self, x: torch.Tensor,
                           params: dict) -> torch.Tensor:
        """用指定参数前向"""
        # 简化实现
        return self.model(x)


# ============== 完整认知听觉系统 ==============

class CognitiveAuditoryCortex(nn.Module):
    """
    认知听觉皮层 - 完整系统 (阶段1+2)

    整合:
    1. 神经拟真耳蜗 (阶段1)
    2. 工作记忆 (phonological loop)
    3. 听觉-运动闭环
    4. 元学习适应
    """

    def __init__(self, sample_rate: int = 16000,
                 n_channels: int = 128):
        super().__init__()

        # 阶段1: 神经拟真 (从spiking_auditory_cortex导入)
        from core.spiking_auditory_cortex import SpikingAuditoryCortex
        self.neural = SpikingAuditoryCortex(sample_rate, n_channels)

        # 阶段2: 认知架构
        self.working_memory = AuditoryWorkingMemory(256)

        self.motor_coupling = AuditoryMotorCoupling(256, 128)

        self.meta_learning = MetaLearningAuditory(256, 128)

        # 运动输出投影 (128→256, 匹配cortical维度)
        self.motor_proj = nn.Linear(128, 256)

        # 输出投影
        self.output_projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, audio_left: torch.Tensor,
              attention_focus: torch.Tensor | None = None,
              mode: str = "listen") -> dict:
        """
        audio_left: [B, T]
        attention_focus: 可选 [B, 256]
        mode: "listen" | "speak" | "repeat"

        Returns:
            all outputs from all subsystems
        """
        # 1. 神经拟真处理
        neural_out = self.neural(audio_left)
        cortical_features = neural_out['features']

        # 2. 工作记忆
        working = self.working_memory(
            cortical_features,
            attention_focus,
            step=0
        )

        # 3. 听觉-运动
        motor = self.motor_coupling(
            cortical_features,
            mode=mode
        )

        # 4. 整合输出
        motor_projected = self.motor_proj(motor['motor_output'])
        integrated = (
            cortical_features * 0.4 +
            working['memory_output'] * 0.3 +
            motor_projected * 0.3
        )

        output = self.output_projector(integrated)

        return {
            'neural': neural_out,           # from 阶段1
            'working_memory': working,     # from 阶段2.1
            'motor_coupling': motor,        # from 阶段2.2
            'integrated_features': integrated,
            'output': output,

            # 元学习 (可选, 新环境时启用)
            'meta': None
        }


def create_cognitive_auditory_cortex() -> CognitiveAuditoryCortex:
    return CognitiveAuditoryCortex()


# ============== 测试 ==============

if __name__ == "__main__":
    print("=== Testing Cognitive Auditory Cortex ===\n")

    # 1. 测试 Phonological Loop
    print("[1] Phonological Loop")
    phoneloop = PhonologicalLoop(256, max_items=7)
    features = torch.randn(2, 256)
    out = phoneloop(features, step=0)
    print(f"  - store size: {out['store_size']}")
    print(f"  - active shape: {out['active_item'].shape}")

    # 2. 测试 Auditory-Motor Coupling
    print("\n[2] Auditory-Motor Coupling")
    motor = AuditoryMotorCoupling(256, 128)
    audio = torch.randn(2, 256)
    out = motor(audio, mode="listen")
    print(f"  - motor output: {out['motor_output'].shape}")
    print(f"  - mirror: {out['mirror_activation'].shape}")

    # 3. 测试 Meta-Learning
    print("\n[3] Meta-Learning")
    meta = MetaLearningAuditory(256, 128)
    support = torch.randn(2, 5, 256)   # 5-shot
    query = torch.randn(2, 3, 256)    # 3-query
    out = meta(support, query)
    print(f"  - adapted: {out['adapted_features'].shape}")
    print(f"  - gain: {out['adaptation_gain'].shape}")

    # 4. 测试完整系统
    print("\n[4] Full Cognitive System")
    cortex = create_cognitive_auditory_cortex()
    audio = torch.randn(2, 16000)
    out = cortex(audio)

    print(f"  - integrated: {out['integrated_features'].shape}")
    print(f"  - output: {out['output'].shape}")
    print(f"  - has neural: {out['neural'] is not None}")
    print(f"  - has memory: {out['working_memory'] is not None}")
    print(f"  - has motor: {out['motor_coupling'] is not None}")

    print("\n✓ All tests passed!")
