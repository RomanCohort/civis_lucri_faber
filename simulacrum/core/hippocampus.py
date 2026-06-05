"""
海马体系统 (Hippocampus)

对应生物学的情景记忆与空间导航：
1. CA3 - 模式分离/完成
2. CA1 - 模式分离/时间细胞
3. Dentate Gyrus (DG) - 模式分离
4. Entorhinal Cortex - 內嗅皮层中继

核心功能：
1. 情景记忆编码/检索
2. 空间表征 (place cells)
3. 情景想象 (forward/backward replay)
4. 联想学习
5. Sharp-wave ripple回放 (~200Hz) - 新增
6. 海马-前额叶双向连接 - 新增

参考:
- Buzsáki (2015) - Hippocampal sharp-wave ripples
- Eichenbaum (2017) - Hippocampus as cognitive map
- Preston & Eichenbaum (2013) - HC-PFC interactions
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# 延迟导入，避免循环引用
try:
    from core.interference_forgetting import InterferenceEngine
except ImportError:
    from interference_forgetting import InterferenceEngine


# ============ 海马体核心 ============

@dataclass
class EpisodeMemory:
    """情景记忆"""
    state: np.ndarray
    action: str
    reward: float
    encoding: np.ndarray  # 海马编码
    timestamp: int
    importance: float = 1.0


@dataclass
class PlaceCell:
    """位置细胞"""
    position: np.ndarray  # 2D或3D位置
    firing_rate: float
    field_size: float


class DentateGyrus(nn.Module):
    """
    齿状回 (DG)

    模式分离：相似记忆分开存储
    """

    def __init__(
        self,
        input_dim: int = 64,
        encoding_dim: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        # 模式分离网络
        self.separator = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim),
            nn.Tanh()
        )

        # 稀疏度约束
        self.sparsity = 0.1

    def forward(
        self,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        模式分离编码
        """
        encoding = self.separator(memory)

        # 稀疏化
        if self.sparsity < 1.0:
            k = int(encoding.numel() * self.sparsity)
            values, _ = encoding.abs().view(-1).topk(k)
            threshold = values.min() if k > 0 else 0
            mask = (encoding.abs() >= threshold).float()
            encoding = encoding * mask

        return encoding


class NeurogenicDG(DentateGyrus):
    """
    具有神经发生能力的齿状回

    对应生物学：成人海马神经发生（Adult Hippocampal Neurogenesis）
    - DG是大脑中少数持续产生新神经元的区域之一
    - 每天约产生700个新神经元（人类）
    - 新神经元具有高可塑性（宽学习窗口）
    - 通过竞争存活：找到功能角色的存活，否则死亡
    - 新神经元增强模式分离能力

    参数：
        neurogenesis_rate: 每步产生新神经元的概率
        max_encoding_dim: 最大编码维度上限
        survival_window: 新神经元竞争存活的窗口步数
    """

    def __init__(
        self,
        input_dim: int = 64,
        encoding_dim: int = 128,
        neurogenesis_rate: float = 0.01,
        max_encoding_dim: int = 200,
        survival_window: int = 100,
    ):
        super().__init__(input_dim, encoding_dim)
        self.neurogenesis_rate = neurogenesis_rate
        self.max_encoding_dim = max_encoding_dim
        self.survival_window = survival_window
        self.current_encoding_dim = encoding_dim
        self.step_count = 0

        # 新神经元追踪
        self._neuron_birth_dates: dict[int, int] = {}  # dim_index -> step_born
        self._neuron_survival_scores: dict[int, float] = {}  # dim_index -> score
        self._neurogenesis_accumulator: float = 0.0  # 确定性神经发生累积器

    def _birth_new_neurons(self, n: int = 1):
        """
        诞生新神经元

        通过扩展输出维度实现：
        1. 在separator网络的最后一层添加新输出神经元
        2. 新神经元初始连接随机（高可塑性）
        3. 记录出生日期用于存活竞争
        """
        if self.current_encoding_dim >= self.max_encoding_dim:
            return

        for _ in range(n):
            new_dim = self.current_encoding_dim + 1
            # 扩展最后一层Linear
            last_layer = self.separator[-2]  # Tanh之前的Linear
            if isinstance(last_layer, nn.Linear):
                old_weight = last_layer.weight.data
                old_bias = last_layer.bias.data
                # 新神经元：随机初始连接
                new_weight_row = torch.randn(1, old_weight.shape[1]) * 0.1
                new_bias_val = torch.tensor([0.0])
                last_layer.weight = nn.Parameter(torch.cat([old_weight, new_weight_row], dim=0))
                last_layer.bias = nn.Parameter(torch.cat([old_bias, new_bias_val], dim=0))

            self.current_encoding_dim = new_dim
            neuron_idx = new_dim - 1
            self._neuron_birth_dates[neuron_idx] = self.step_count
            self._neuron_survival_scores[neuron_idx] = 1.0  # 高初始可塑性

    def _update_survival_scores(self, encoding: torch.Tensor):
        """根据激活更新新神经元的存活分数"""
        if encoding.dim() == 2:
            flat = encoding[0]
        else:
            flat = encoding

        for idx in list(self._neuron_survival_scores.keys()):
            if idx < flat.shape[0]:
                activation = abs(flat[idx].item())
                if activation > 0.1:
                    # 被激活 → 增强存活概率
                    self._neuron_survival_scores[idx] = min(
                        1.0, self._neuron_survival_scores[idx] + 0.05
                    )
                else:
                    # 未激活 → 存活分数衰减
                    self._neuron_survival_scores[idx] *= 0.98

    def _apply_survival_competition(self):
        """存活竞争：超过窗口期且得分低的新神经元死亡"""
        dead_neurons = []
        for idx, birth_step in list(self._neuron_birth_dates.items()):
            age = self.step_count - birth_step
            if age > self.survival_window:
                score = self._neuron_survival_scores.get(idx, 0.0)
                if score < 0.1:
                    dead_neurons.append(idx)

        # 清理死亡神经元记录（实际维度不缩减以保持张量一致性）
        for idx in dead_neurons:
            del self._neuron_birth_dates[idx]
            del self._neuron_survival_scores[idx]

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """带神经发生的模式分离"""
        encoding = super().forward(memory)

        self.step_count += 1

        # 神经发生: 确定性累积替代随机门控
        # neurogenesis_rate 累积到 1.0 时诞生新神经元
        self._neurogenesis_accumulator += self.neurogenesis_rate
        if self._neurogenesis_accumulator >= 1.0:
            self._birth_new_neurons()
            self._neurogenesis_accumulator -= 1.0

        # 更新存活分数
        self._update_survival_scores(encoding)

        # 定期存活竞争（每100步）
        if self.step_count % 100 == 0:
            self._apply_survival_competition()

        return encoding

    def get_neurogenesis_summary(self) -> dict:
        """获取神经发生统计"""
        young_neurons = sum(
            1 for step in self._neuron_birth_dates.values()
            if self.step_count - step < self.survival_window
        )
        return {
            'current_encoding_dim': self.current_encoding_dim,
            'young_neurons': young_neurons,
            'total_new_neurons': len(self._neuron_birth_dates),
            'step_count': self.step_count,
            'avg_survival_score': (
                sum(self._neuron_survival_scores.values()) / len(self._neuron_survival_scores)
                if self._neuron_survival_scores else 0.0
            ),
        }


class CA3Region(nn.Module):
    """
    CA3区域

    模式完成 + 联想学习
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 联想网络
        self.associative_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # 复发网络
        self.recurrent = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # 记忆关联
        self.memory_links = {}  # key -> [associated_keys]

    def forward(
        self,
        encoding: torch.Tensor,
        retrieve_hint: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        模式完成 + 循环处理
        """
        if retrieve_hint is not None:
            # 部分线索 → 完整回忆
            combined = encoding * 0.7 + retrieve_hint * 0.3
            output = self.associative_net(combined)
        else:
            # 完整编码
            output = self.associative_net(encoding)

        # GRU循环处理（残差连接）
        if output.dim() == 2:
            gru_input = output.unsqueeze(1)  # [B, 1, dim]
        else:
            gru_input = output
        gru_out, _ = self.recurrent(gru_input)
        gru_out = gru_out.squeeze(1)  # [B, hidden_dim]

        # 将GRU输出投影回encoding_dim并与关联输出残差连接
        if not hasattr(self, '_gru_proj'):
            self._gru_proj = nn.Linear(gru_out.shape[-1], encoding.shape[-1]).to(encoding.device)
        projected = self._gru_proj(gru_out)
        output = output + projected * 0.3  # 残差：30%来自GRU循环

        return output

    def link_memories(
        self,
        key1: str,
        key2: str,
        strength: float = 1.0,
    ):
        """关联两个记忆"""
        if key1 not in self.memory_links:
            self.memory_links[key1] = []
        if key2 not in self.memory_links[key1]:
            self.memory_links[key1].append((key2, strength))

    def retrieve_associated(
        self,
        key: str,
    ) -> list[tuple[str, float]]:
        """检索关联记忆"""
        return self.memory_links.get(key, [])


class CA1Region(nn.Module):
    """
    CA1区域

    时间序列 + 预测
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 时间序列学习
        self.temporal_net = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # 预测网络
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def encode_sequence(
        self,
        sequence: list[torch.Tensor],
    ) -> dict:
        """
        编码序列
        """
        if not sequence:
            return {'encoding': None, 'prediction': None}

        # 确保是2D [seq, dim]
        seq_list = []
        for s in sequence:
            if s.dim() == 4:
                seq_list.append(s.squeeze(0).squeeze(0))
            elif s.dim() == 3:
                seq_list.append(s.squeeze(0))
            elif s.dim() == 2 and s.shape[0] != 1:
                seq_list.append(s)
            else:
                seq_list.append(s.squeeze(0))

        seq_tensor = torch.stack(seq_list).unsqueeze(0)  # [1, seq, dim]

        output, (h, c) = self.temporal_net(seq_tensor)

        # 最后状态编码
        encoding = output[:, -1]

        # 预测下一个
        prediction = self.predictor(encoding)

        return {
            'encoding': encoding,
            'prediction': prediction,
        }

    def predict_next(
        self,
        current_state: torch.Tensor,
        sequence_so_far: list[torch.Tensor],
    ) -> torch.Tensor:
        """预测下一状态"""
        if not sequence_so_far:
            return current_state

        seq_tensor = torch.stack(sequence_so_far).unsqueeze(0)
        output, _ = self.temporal_net(seq_tensor)
        encoding = output[:, -1]

        return self.predictor(encoding)


class EntorhinalCortex(nn.Module):
    """
    內嗅皮层 (EC)

    网格细胞 + 路径整合
    """

    def __init__(
        self,
        input_dim: int = 64,
        grid_dim: int = 8,
    ):
        super().__init__()

        self.grid_dim = grid_dim

        # 网格细胞模式
        self.grid_patterns = self._init_grid_patterns(input_dim, grid_dim)

        # 路径整合网络
        self.path_integrator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, grid_dim * 2),  # position + velocity
        )

    def _init_grid_patterns(
        self,
        dim: int,
        grid_dim: int,
    ) -> torch.Tensor:
        """初始化网格模式"""
        patterns = []
        for i in range(grid_dim):
            freq = 0.5 + i * 0.1
            phase = np.random.rand() * 2 * np.pi
            pattern = torch.tensor([
                np.sin(freq * j + phase) for j in range(dim)
            ], dtype=torch.float32)
            patterns.append(pattern)

        return torch.stack(patterns)

    def compute_position(
        self,
        sensory: torch.Tensor,
        velocity: torch.Tensor = None,
    ) -> np.ndarray:
        """
        计算当前位置
        """
        pos_vel = self.path_integrator(sensory)

        if velocity is not None:
            pos_vel = pos_vel + velocity * 0.1

        return pos_vel.detach().cpu().numpy()

    def grid_cell_response(
        self,
        position: np.ndarray,
    ) -> np.ndarray:
        """位置细胞响应"""
        pos_t = torch.tensor(position, dtype=torch.float32)
        responses = []
        for pattern in self.grid_patterns:
            response = torch.cos(pos_t @ pattern.float())
            responses.append(response)

        return torch.stack(responses).numpy()


# ══════════════════════════════════════════════════════
# Sharp-wave Ripple 机制 (新增)
# 参考: Buzsáki (2015) - Hippocampal sharp-wave ripples
# ══════════════════════════════════════════════════════

@dataclass
class SharpWaveRippleConfig:
    """Sharp-wave ripple配置参数

    参考: Buzsáki (2015)
    - Ripple频率: ~200Hz (100-250Hz)
    - Ripple持续时间: 50-100ms
    - 触发条件: 慢波睡眠、静息状态、记忆巩固
    """
    ripple_frequency: float = 200.0     # Hz
    ripple_duration_ms: float = 80.0    # ms
    ripple_amplitude: float = 1.0       # 基础振幅
    propagation_rate: float = 0.8       # ripple传播速度
    min_interval_ms: float = 500.0      # ripple最小间隔


class SharpWaveRipple(nn.Module):
    """Sharp-wave ripple事件

    高频(~200Hz)局部场电位震荡，发生在:
    1. 慢波睡眠(SWS)期间 - 记忆巩固
    2. 静息状态 - 最近经历的回放
    3. 决策前的计划模拟

    功能:
    - 快速压缩回放最近经历序列
    - 向PFC传递整合后的记忆
    - 支持离线学习(无外部输入时的学习)

    参考: Buzsáki (2015), Jadhav et al. (2012)
    """

    def __init__(
        self,
        n_neurons: int = 128,
        config: SharpWaveRippleConfig | None = None,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config or SharpWaveRippleConfig()

        # Ripple状态
        self.is_active = False
        self.current_phase = 0.0  # ripple相位 [0, 2π]
        self.step_count = 0

        # Ripple历史 (用于统计)
        self.ripple_events: list[dict] = []
        self.last_ripple_step = 0

        # 神经元相位偏移 (模拟空间传播)
        phase_offsets = torch.linspace(0, 2*np.pi, n_neurons)
        self.register_buffer('phase_offsets', phase_offsets)

    def check_ripple_trigger(
        self,
        sleep_stage: str = "awake",
        arousal_level: float = 0.5,
        recent_activity: float = 0.3,
    ) -> bool:
        """检查是否触发ripple

        触发条件:
        1. 慢波睡眠(SWS/REM)期间 - 高概率
        2. 低唤醒状态(arousal < 0.3) - 中概率
        3. 近期高活动后静息 - 高概率

        Args:
            sleep_stage: 睡眠阶段 ("awake", "NREM", "REM")
            arousal_level: 唤醒水平 [0, 1]
            recent_activity: 近期活动强度 [0, 1]

        Returns:
            should_trigger: 是否触发ripple
        """
        # 基础触发概率
        base_prob = 0.02  # 每步2%基础概率

        # 睡眠阶段调制
        stage_factor = {
            "awake": 0.3,      # 清醒期低概率
            "NREM": 3.0,       # 慢波睡眠高概率
            "REM": 1.5,        # REM中等概率
        }.get(sleep_stage, 0.3)

        # 低唤醒增强概率
        arousal_factor = 1.0 + 0.5 * (1.0 - arousal_level)

        # 近期活动后静息增强概率 (经验回放需要)
        activity_factor = 1.0 + recent_activity

        # 最终触发概率
        trigger_prob = base_prob * stage_factor * arousal_factor * activity_factor
        trigger_prob = min(0.5, trigger_prob)  # 上限50%

        return np.random.random() < trigger_prob

    def start_ripple(self, sequence: list[np.ndarray]) -> dict:
        """启动ripple事件

        Args:
            sequence: 要回放的记忆序列编码

        Returns:
            ripple_info: ripple启动信息
        """
        self.is_active = True
        self.current_phase = 0.0

        ripple_info = {
            'start_step': self.step_count,
            'sequence_length': len(sequence),
            'frequency': self.config.ripple_frequency,
            'duration_ms': self.config.ripple_duration_ms,
        }
        self.ripple_events.append(ripple_info)
        self.last_ripple_step = self.step_count

        return ripple_info

    def propagate_ripple(
        self,
        encoding: torch.Tensor,
        phase: float,
    ) -> torch.Tensor:
        """传播ripple信号

        高频振荡调制神经元输出:
        - 每个神经元有不同的相位偏移(模拟空间传播)
        - ripple相位调制激活强度

        Args:
            encoding: 记忆编码 [n_neurons]
            phase: 当前ripple相位 [0, 2π]

        Returns:
            modulated: ripple调制后的输出
        """
        # 各神经元相位 = 全局相位 + 偏移
        neuron_phases = phase + self.phase_offsets

        # 调制因子 = sin(phase)的高频震荡
        # ~200Hz意味着每~5ms完成一个周期
        modulation = torch.sin(neuron_phases) * self.config.ripple_amplitude

        # 调制激活
        modulated = encoding * (1.0 + 0.3 * modulation)

        return modulated

    def step(
        self,
        sleep_stage: str = "awake",
        arousal_level: float = 0.5,
        recent_sequence: list[np.ndarray] | None = None,
    ) -> dict:
        """执行一步ripple

        Args:
            sleep_stage: 睡眠阶段
            arousal_level: 唤醒水平
            recent_sequence: 待回放的记忆序列

        Returns:
            result: ripple执行结果
        """
        self.step_count += 1

        result = {
            'is_active': self.is_active,
            'phase': self.current_phase,
            'ripple_triggered': False,
            'replay_output': None,
        }

        # 检查是否触发新ripple
        if not self.is_active and self.step_count - self.last_ripple_step > 50:
            recent_activity = len(recent_sequence) / 20.0 if recent_sequence else 0.0
            if self.check_ripple_trigger(sleep_stage, arousal_level, recent_activity):
                if recent_sequence:
                    ripple_info = self.start_ripple(recent_sequence)
                    result['ripple_triggered'] = True
                    result['ripple_info'] = ripple_info

        # 活跃ripple的相位推进
        if self.is_active:
            # 每步推进相位 (模拟~200Hz)
            phase_increment = 2 * np.pi * 0.2  # 每步0.2个周期
            self.current_phase += phase_increment

            # 检查ripple结束
            ripple_steps = self.config.ripple_duration_ms / 5.0  # ~16步
            if self.current_phase > 2 * np.pi * ripple_steps / 5.0:
                self.is_active = False
                self.current_phase = 0.0
                result['ripple_ended'] = True

            # 回放输出
            if recent_sequence and len(recent_sequence) > 0:
                # 选择当前相位对应的记忆
                seq_idx = int(self.current_phase / (2 * np.pi) * len(recent_sequence)) % len(recent_sequence)
                current_encoding = torch.tensor(recent_sequence[seq_idx], dtype=torch.float32)

                # ripple调制
                modulated = self.propagate_ripple(current_encoding, self.current_phase)
                result['replay_output'] = modulated.detach().cpu().numpy()

        return result

    def get_ripple_stats(self) -> dict:
        """获取ripple统计"""
        return {
            'total_ripples': len(self.ripple_events),
            'is_active': self.is_active,
            'current_phase': self.current_phase,
            'last_ripple_step': self.last_ripple_step,
        }


# ══════════════════════════════════════════════════════
# 海马-前额叶双向连接 (新增)
# 参考: Preston & Eichenbaum (2013)
# ══════════════════════════════════════════════════════

class HCPFCConnection(nn.Module):
    """海马-前额叶双向连接

    功能:
    1. HC→PFC: 传递情景记忆、预测、空间信息
    2. PFC→HC: 执行目标调制、注意门控、策略指导

    参考:
    - Preston & Eichenbaum (2013): HC-PFC交互在记忆中的作用
    - Eichenbaum (2017): 海马作为认知地图
    - Bayer et al. (2017): HC-PFC在决策中的协同
    """

    def __init__(
        self,
        hc_dim: int = 128,
        pfc_dim: int = 64,
        connection_strength: float = 0.5,
    ):
        super().__init__()
        self.hc_dim = hc_dim
        self.pfc_dim = pfc_dim

        # HC→PFC 传递通路
        self.hc_to_pfc = nn.Sequential(
            nn.Linear(hc_dim, pfc_dim),
            nn.ReLU(),
            nn.Linear(pfc_dim, pfc_dim),
        )

        # PFC→HC 调制通路 (目标、注意)
        self.pfc_to_hc = nn.Sequential(
            nn.Linear(pfc_dim, hc_dim),
            nn.Tanh(),  # 调制信号用Tanh [-1, 1]
        )

        # 连接强度 (可调节)
        self.connection_strength = nn.Parameter(torch.tensor(connection_strength))

        # 最近传输记录
        self.transfer_history: deque = deque(maxlen=50)

    def transfer_to_pfc(
        self,
        hc_encoding: torch.Tensor,
        transfer_type: str = "episodic",
    ) -> torch.Tensor:
        """HC→PFC传输

        传输类型:
        - episodic: 情景记忆编码
        - prediction: 未来预测
        - spatial: 空间位置信息

        Args:
            hc_encoding: 海马编码 [hc_dim]
            transfer_type: 传输类型

        Returns:
            pfc_input: 前额叶输入信号 [pfc_dim]
        """
        # 确保维度
        if hc_encoding.dim() == 1:
            hc_encoding = hc_encoding.unsqueeze(0)

        # 传输
        pfc_input = self.hc_to_pfc(hc_encoding)

        # 类型调制
        type_factors = {
            "episodic": 1.0,
            "prediction": 0.7,  # 预测信号稍弱
            "spatial": 0.5,     # 空间信息更弱
        }
        factor = type_factors.get(transfer_type, 0.8)
        pfc_input = pfc_input * factor * self.connection_strength

        # 记录传输
        self.transfer_history.append({
            'direction': 'hc_to_pfc',
            'type': transfer_type,
            'strength': self.connection_strength.item(),
        })

        return pfc_input

    def modulate_from_pfc(
        self,
        pfc_signal: torch.Tensor,
        modulation_type: str = "goal",
    ) -> torch.Tensor:
        """PFC→HC调制

        调制类型:
        - goal: 目标导向的检索调制
        - attention: 注意门控
        - strategy: 策略指导

        Args:
            pfc_signal: 前额叶信号 [pfc_dim]
            modulation_type: 调制类型

        Returns:
            hc_modulation: 海马调制信号 [hc_dim]
        """
        # 确保维度
        if pfc_signal.dim() == 1:
            pfc_signal = pfc_signal.unsqueeze(0)

        # 调制信号
        hc_modulation = self.pfc_to_hc(pfc_signal)

        # 类型调制
        type_factors = {
            "goal": 1.0,        # 目标调制最强
            "attention": 0.6,   # 注意门控中等
            "strategy": 0.4,    # 策略指导较弱
        }
        factor = type_factors.get(modulation_type, 0.5)
        hc_modulation = hc_modulation * factor * self.connection_strength

        # 记录传输
        self.transfer_history.append({
            'direction': 'pfc_to_hc',
            'type': modulation_type,
            'strength': self.connection_strength.item(),
        })

        return hc_modulation

    def get_connection_stats(self) -> dict:
        """获取连接统计"""
        hc_to_pfc_count = sum(1 for h in self.transfer_history if h['direction'] == 'hc_to_pfc')
        pfc_to_hc_count = sum(1 for h in self.transfer_history if h['direction'] == 'pfc_to_hc')

        return {
            'connection_strength': self.connection_strength.item(),
            'hc_to_pfc_transfers': hc_to_pfc_count,
            'pfc_to_hc_transfers': pfc_to_hc_count,
            'total_transfers': len(self.transfer_history),
        }


@dataclass
class HippocampusState:
    """海马状态"""
    current_position: np.ndarray = None
    memory_count: int = 0
    replay_mode: str = "none"  # "forward" | "backward" | "none"


class Hippocampus(nn.Module):
    """
    完整海马体系统

    整合DG + CA3 + CA1 + EC + Sharp-wave Ripple + HC-PFC连接
    """

    def __init__(
        self,
        input_dim: int = 64,
        encoding_dim: int = 128,
        pfc_dim: int = 64,
        event_bus=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.pfc_dim = pfc_dim

        # 各子区
        self.dg = DentateGyrus(input_dim, encoding_dim)
        self.ca3 = CA3Region(encoding_dim, 64)
        self.ca1 = CA1Region(encoding_dim, 64)
        self.ec = EntorhinalCortex(input_dim)

        # Sharp-wave ripple机制 (新增)
        self.sharp_wave_ripple = SharpWaveRipple(n_neurons=encoding_dim)

        # 海马-前额叶双向连接 (新增)
        self.hc_pfc_connection = HCPFCConnection(
            hc_dim=encoding_dim,
            pfc_dim=pfc_dim,
        )

        # 记忆存储
        self.episodic_memory: list[EpisodeMemory] = []
        self.memory_traces = deque(maxlen=1000)

        # 干扰性遗忘引擎（替代FIFO淘汰）
        self.interference_engine = InterferenceEngine(
            decay_rate=0.01,
            proactive_strength=0.3,
            retroactive_strength=0.01,
            min_importance=0.05,
        )
        self.use_interference_forgetting = True

        # 状态
        self.state = HippocampusState()

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "memory_encode",
                self._handle_memory_encode,
                priority=0,
                name="hippocampus",
            )

    def step(self, *args, **kwargs) -> dict:
        """One simulation step — delegates to encode_memory or retrieve."""
        return self.forward(*args, **kwargs)

    @staticmethod
    def required_keys() -> list[str]:
        """Keys this region reads from the shared state."""
        return ["state_tensor", "emotion_signal"]

    @staticmethod
    def output_keys() -> list[str]:
        """Keys this region writes to the shared state."""
        return ["hippo_episode", "hippo_retrieval", "hippo_consolidation",
                "hippo_place_code", "hippo_novelty"]

    def _handle_memory_encode(self, event) -> dict:
        """Event-driven handler for memory_encode events."""
        import numpy as _np
        state = event.data.get("internal_state", {})

        state_np = event.data.get("state_np")
        if state_np is None:
            state_np = _np.random.randn(self.input_dim).astype(_np.float32)

        action_str = event.data.get("action_str", "default")
        reward_val = event.data.get("reward_val", 0.0)

        encoding = self.encode_memory(
            state=state_np,
            action=action_str,
            reward=reward_val,
        )

        state["hc_memory_count"] = self.state.memory_count
        state["hc_last_encoding_norm"] = float(_np.linalg.norm(encoding))

        # Retrieve if enough memories
        retrieved_avg_reward = 0.0
        if self.state.memory_count > 5:
            retrieved = self.retrieve(state_np, top_k=5)
            if retrieved:
                retrieved_avg_reward = float(_np.mean([m.reward for m in retrieved]))
        state["hc_retrieved_avg_reward"] = retrieved_avg_reward

        # Defensive mode: high negative reward triggers defensive behavior
        state["defensive_mode"] = reward_val < -0.5

        return {
            "encoding_norm": state["hc_last_encoding_norm"],
            "memory_count": self.state.memory_count,
            "retrieved_avg_reward": retrieved_avg_reward,
        }

    def encode_memory(
        self,
        state: np.ndarray,
        action: str,
        reward: float,
    ) -> np.ndarray:
        """
        编码情景记忆
        """
        # 转换为tensor
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # DG模式分离
        encoding = self.dg(state_t)

        # CA3关联
        encoding = self.ca3(encoding)

        # CA1时间编码 - encoding已经是tensor
        encoding_t = encoding.detach()
        sequence_result = self.ca1.encode_sequence([encoding_t])

        encoding_np = encoding.detach().cpu().numpy()[0]

        # 存储
        memory = EpisodeMemory(
            state=state,
            action=action,
            reward=reward,
            encoding=encoding_np,
            timestamp=self.state.memory_count,
        )
        self.episodic_memory.append(memory)
        self.memory_traces.append(encoding_np)
        self.state.memory_count += 1

        # 遗忘机制
        if self.use_interference_forgetting:
            # 前摄干扰：旧记忆降低新记忆的重要性
            proactive = self.interference_engine.compute_proactive_interference(
                self.episodic_memory[:-1], encoding_np
            )
            memory.importance *= (1.0 - proactive * 0.3)

            # 倒摄干扰 + 淘汰低于阈值的记忆
            self.episodic_memory = self.interference_engine.apply_forgetting(
                self.episodic_memory, encoding_np
            )
        else:
            # 原始FIFO后备
            if len(self.episodic_memory) > 1000:
                self.episodic_memory.pop(0)

        return encoding_np

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> list[EpisodeMemory]:
        """
        检索记忆

        基于相似度检索
        """
        if not self.episodic_memory:
            return []

        query_t = torch.tensor(query, dtype=torch.float32).unsqueeze(0)
        query_enc = self.dg(query_t).detach().cpu().numpy()[0]

        # 计算相似度
        similarities = []
        for mem in self.episodic_memory:
            sim = -np.linalg.norm(query_enc - mem.encoding)
            similarities.append((mem, sim))

        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [mem for mem, _ in similarities[:top_k]]

    def replay_forward(
        self,
        start_idx: int = None,
    ) -> list[EpisodeMemory]:
        """
        正向回放 (经验学习)
        """
        if not self.episodic_memory:
            return []

        self.state.replay_mode = "forward"

        if start_idx is None:
            start_idx = max(0, len(self.episodic_memory) - 10)

        return self.episodic_memory[start_idx:start_idx + 10]

    def replay_backward(
        self,
        start_idx: int = None,
    ) -> list[EpisodeMemory]:
        """
        反向回放 (错误学习)
        """
        if not self.episodic_memory:
            return []

        self.state.replay_mode = "backward"

        if start_idx is None:
            start_idx = len(self.episodic_memory) - 1

        memories = list(reversed(self.episodic_memory[max(0, start_idx - 10):start_idx + 1]))
        return memories

    def imagine_future(
        self,
        current_state: torch.Tensor,
        n_steps: int = 5,
    ) -> list[torch.Tensor]:
        """
        情景想象 (预测未来)
        """
        # 调整维度
        if current_state.shape[-1] != self.encoding_dim:
            if current_state.dim() == 2:
                current_state = current_state.squeeze(0)
            # 简单重复或填充
            current_state = current_state[:self.encoding_dim] if current_state.shape[0] > self.encoding_dim else \
                torch.nn.functional.pad(current_state, (0, self.encoding_dim - current_state.shape[0]))

        sequence = [current_state.unsqueeze(0)]
        for _ in range(n_steps):
            last = sequence[-1]
            try:
                result = self.ca1.encode_sequence([last.squeeze(0)])
                next_state = result.get('prediction', last)
            except RuntimeError:
                # 简单预测
                noise = torch.randn_like(last) * 0.1
                next_state = last + noise
            sequence.append(next_state)

        return sequence[1:]  # 不包含当前状态

    def imagine_past(
        self,
        current_state: torch.Tensor,
        n_steps: int = 5,
    ) -> list[torch.Tensor]:
        """
        情景回溯 (推理过去)

        复用已有CA1的temporal_net进行反向预测，
        通过反转预测方向实现时间回溯。
        """
        # 确保维度正确
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)

        sequence = [current_state]
        for _ in range(n_steps):
            last = sequence[-1]
            # 使用CA1预测，但反转方向
            try:
                result = self.ca1.encode_sequence([last.squeeze(0)])
                if result['prediction'] is not None:
                    # 反向：prev ≈ current - (predicted_next - current)
                    delta = result['prediction'] - last
                    prev = last - delta * 0.5  # 半步回溯
                else:
                    prev = last
            except Exception:
                prev = last
            sequence.append(prev)

        return list(reversed(sequence[:-1]))

    def link_episodes(
        self,
        episode1_idx: int,
        episode2_idx: int,
    ):
        """关联两个情景"""
        if episode1_idx >= len(self.episodic_memory) or episode2_idx >= len(self.episodic_memory):
            return

        e1 = self.episodic_memory[episode1_idx]
        e2 = self.episodic_memory[episode2_idx]

        # CA3关联
        key1 = f"ep{episode1_idx}"
        key2 = f"ep{episode2_idx}"
        self.ca3.link_memories(key1, key2)

    def consolidate_recent(self):
        """
        情景巩固 (sleep-like)
        """
        if len(self.episodic_memory) < 2:
            return

        # 重新编码最近记忆
        recent = self.episodic_memory[-10:]

        for mem in recent:
            # 加强
            mem.importance *= 1.1

    def get_spatial_representation(
        self,
        position: np.ndarray,
    ) -> np.ndarray:
        """获取空间表征"""
        return self.ec.grid_cell_response(position)

    def transfer_to_pfc(
        self,
        encoding: np.ndarray,
        transfer_type: str = "episodic",
    ) -> np.ndarray:
        """向PFC传输信息 (新增)

        Args:
            encoding: 海马编码
            transfer_type: 传输类型 (episodic/prediction/spatial)

        Returns:
            pfc_input: 前额叶输入
        """
        encoding_t = torch.tensor(encoding, dtype=torch.float32)
        pfc_input = self.hc_pfc_connection.transfer_to_pfc(encoding_t, transfer_type)
        return pfc_input.squeeze(0).detach().cpu().numpy()

    def receive_pfc_modulation(
        self,
        pfc_signal: np.ndarray,
        modulation_type: str = "goal",
    ) -> np.ndarray:
        """接收PFC调制 (新增)

        Args:
            pfc_signal: 前额叶信号
            modulation_type: 调制类型 (goal/attention/strategy)

        Returns:
            hc_modulation: 海马调制信号
        """
        pfc_t = torch.tensor(pfc_signal, dtype=torch.float32)
        hc_modulation = self.hc_pfc_connection.modulate_from_pfc(pfc_t, modulation_type)
        return hc_modulation.squeeze(0).detach().cpu().numpy()

    def trigger_ripple_replay(
        self,
        sleep_stage: str = "awake",
        arousal_level: float = 0.5,
    ) -> dict:
        """触发sharp-wave ripple回放 (新增)

        在睡眠或低唤醒状态下，高频压缩回放近期经历

        Args:
            sleep_stage: 睡眠阶段 (awake/NREM/REM)
            arousal_level: 唤醒水平 [0, 1]

        Returns:
            replay_result: 回放结果
        """
        # 获取近期记忆编码
        recent_encodings = [m.encoding for m in self.episodic_memory[-20:]]

        ripple_result = self.sharp_wave_ripple.step(
            sleep_stage=sleep_stage,
            arousal_level=arousal_level,
            recent_sequence=recent_encodings,
        )

        # 如果有回放输出，传递给PFC
        if ripple_result.get('replay_output') is not None:
            replay_encoding = ripple_result['replay_output']
            pfc_input = self.transfer_to_pfc(replay_encoding, transfer_type="episodic")
            ripple_result['pfc_transfer'] = pfc_input

        return ripple_result

    def get_summary(self) -> dict:
        """获取摘要"""
        return {
            'memory_count': len(self.episodic_memory),
            'replay_mode': self.state.replay_mode,
            'current_position': self.state.current_position,
            'avg_reward': np.mean([m.reward for m in self.episodic_memory[-10:]]) if self.episodic_memory else 0,
            'ripple_stats': self.sharp_wave_ripple.get_ripple_stats(),
            'hc_pfc_stats': self.hc_pfc_connection.get_connection_stats(),
        }


# ============ 便捷函数 ============

def create_hippocampus(
    input_dim: int = 64,
    encoding_dim: int = 128,
) -> Hippocampus:
    """创建海马体系统"""
    return Hippocampus(input_dim, encoding_dim)


__all__ = [
    'EpisodeMemory',
    'PlaceCell',
    'DentateGyrus',
    'NeurogenicDG',
    'CA3Region',
    'CA1Region',
    'EntorhinalCortex',
    'SharpWaveRippleConfig',
    'SharpWaveRipple',
    'HCPFCConnection',
    'HippocampusState',
    'Hippocampus',
    'create_hippocampus',
]
