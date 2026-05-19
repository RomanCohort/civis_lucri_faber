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
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from collections import deque

# 延迟导入，避免循环引用
try:
    from civis_lucri_faber.core.interference_forgetting import InterferenceEngine
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
        self._neuron_birth_dates: Dict[int, int] = {}  # dim_index -> step_born
        self._neuron_survival_scores: Dict[int, float] = {}  # dim_index -> score
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

    def get_neurogenesis_summary(self) -> Dict:
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
    ) -> List[Tuple[str, float]]:
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
        sequence: List[torch.Tensor],
    ) -> Dict:
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
        sequence_so_far: List[torch.Tensor],
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

        return pos_vel.detach().numpy()

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


@dataclass
class HippocampusState:
    """海马状态"""
    current_position: np.ndarray = None
    memory_count: int = 0
    replay_mode: str = "none"  # "forward" | "backward" | "none"


class Hippocampus(nn.Module):
    """
    完整海马体系统

    整合DG + CA3 + CA1 + EC
    """

    def __init__(
        self,
        input_dim: int = 64,
        encoding_dim: int = 128,
        event_bus=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        # 各子区
        self.dg = DentateGyrus(input_dim, encoding_dim)
        self.ca3 = CA3Region(encoding_dim, 64)
        self.ca1 = CA1Region(encoding_dim, 64)
        self.ec = EntorhinalCortex(input_dim)

        # 记忆存储
        self.episodic_memory: List[EpisodeMemory] = []
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

    def _handle_memory_encode(self, event) -> Dict:
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

        encoding_np = encoding.detach().numpy()[0]

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
    ) -> List[EpisodeMemory]:
        """
        检索记忆

        基于相似度检索
        """
        if not self.episodic_memory:
            return []

        query_t = torch.tensor(query, dtype=torch.float32).unsqueeze(0)
        query_enc = self.dg(query_t).detach().numpy()[0]

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
    ) -> List[EpisodeMemory]:
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
    ) -> List[EpisodeMemory]:
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
    ) -> List[torch.Tensor]:
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
            except:
                # 简单预测
                noise = torch.randn_like(last) * 0.1
                next_state = last + noise
            sequence.append(next_state)

        return sequence[1:]  # 不包含当前状态

    def imagine_past(
        self,
        current_state: torch.Tensor,
        n_steps: int = 5,
    ) -> List[torch.Tensor]:
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

    def get_summary(self) -> Dict:
        """获取摘要"""
        return {
            'memory_count': len(self.episodic_memory),
            'replay_mode': self.state.replay_mode,
            'current_position': self.state.current_position,
            'avg_reward': np.mean([m.reward for m in self.episodic_memory[-10:]]) if self.episodic_memory else 0,
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
    'HippocampusState',
    'Hippocampus',
    'create_hippocampus',
]