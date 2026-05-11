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
        模式完成
        """
        if retrieve_hint is not None:
            # 部分线索 → 完整回忆
            combined = encoding * 0.7 + retrieve_hint * 0.3
            output = self.associative_net(combined)
        else:
            # 完整编码
            output = self.associative_net(encoding)

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

        # 状态
        self.state = HippocampusState()

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

        # 限制
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
        """
        # 反向序列学习
        reversed_ca1 = nn.LSTM(
            input_size=self.encoding_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        sequence = [current_state]
        for _ in range(n_steps):
            # 简化的反向预测
            last = sequence[-1]
            noise = torch.randn_like(last) * 0.1
            prev = last + noise
            sequence.append(prev)

        return reversed(sequence[:-1])

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
    'CA3Region',
    'CA1Region',
    'EntorhinalCortex',
    'HippocampusState',
    'Hippocampus',
    'create_hippocampus',
]