"""
基底神经节系统 (Basal Ganglia)

对应生物学的动作选择与习惯形成：
1. Striatum (纹状体) - 输入整合
2. GPi/SNr (输出) - 动作抑制
3. GPe (间接通路) - 动作选择
4. VTA/SNc (多巴胺来源) - 强化学习

核心功能：
1. 动作选择 (action selection)
2. 习惯形成 (habit formation)
3. TD learning (真实RPG近似)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from collections import deque


# ============ 基底神经节核心 ============

@dataclass
class ActionSelection:
    """动作选择结果"""
    action: int
    q_values: np.ndarray
    selected_q: float
    action_probs: np.ndarray


@dataclass
class HabitMemory:
    """习惯记忆"""
    context: np.ndarray
    action: int
    frequency: int
    last_used: int


class StriatumInput(nn.Module):
    """
    纹状体输入层

    整合多巴胺信号 + 皮层输入 → 动作价值
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.n_actions = n_actions

        # 动作价值网络 (每个动作一个)
        self.action_values = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_actions)
        ])

        # 多巴胺调制
        self.dopamine_modulation = 1.0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """计算各动作Q值"""
        q_values = []
        for av in self.action_values:
            q = av(state)
            q_values.append(q)

        q_tensor = torch.cat(q_values, dim=-1)
        return q_tensor

    def get_q_value(self, state: torch.Tensor, action: int) -> float:
        """获取特定动作Q值"""
        q_tensor = self.forward(state)
        return q_tensor[0, action].item()


class IndirectPathway(nn.Module):
    """
    间接通路 (GPe → STN → GPi)

    动作选择，抑制不想要的动作
    """

    def __init__(
        self,
        n_actions: int = 4,
        inhibition_strength: float = 0.5,
    ):
        super().__init__()

        self.n_actions = n_actions
        self.inhibition = inhibition_strength

        # 动作抑制网络
        self.inhibition_net = nn.Sequential(
            nn.Linear(n_actions, n_actions),
            nn.Sigmoid()
        )

    def compute_selection(
        self,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算动作选择

        竞争性 inhibiton
        """
        # 归一化Q值
        q_norm = F.softmax(q_values, dim=-1)

        # 执行抑制
        inhibition = self.inhibition_net(q_norm)

        # 动作间的相对抑制
        selected = q_norm * (1 - inhibition * self.inhibition)

        return selected


class DirectPathway(nn.Module):
    """
    直接通路 (D1)

    促进动作，经典RPG的"GO"信号
    """

    def __init__(
        self,
        n_actions: int = 4,
    ):
        super().__init__()

        self.n_actions = n_actions

        # 动作促进网络
        self.go_net = nn.Sequential(
            nn.Linear(n_actions, n_actions),
            nn.Sigmoid()
        )

    def compute_go_signal(
        self,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        """计算GO信号"""
        return self.go_net(q_values)


class HyperdirectPathway(nn.Module):
    """
    超通路 (Hyperdirect)

    快速抑制，突发动作控制
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
    ):
        super().__init__()

        # 快速评估
        self.fast_eval = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def compute_fast_inhibit(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """快速抑制"""
        return self.fast_eval(state)


@dataclass
class BGState:
    """基底神经节状态"""
    current_action: int = 0
    exploration_rate: float = 0.1
    habit_strength: float = 0.0
    last_reward: float = 0.0
    action_history: List = field(default_factory=list)


class BasalGanglia(nn.Module):
    """
    完整基底神经节系统

    整合三条通路：
    1. 直接通路 (D1) - GO
    2. 间接通路 (GPe) - NO-GO
    3. 超通路 (STN) - 急停
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma

        # 三条通路
        self.striatum = StriatumInput(state_dim, n_actions)
        self.direct = DirectPathway(n_actions)
        self.indirect = IndirectPathway(n_actions)
        self.hyper = HyperdirectPathway(state_dim, n_actions)

        # 多巴胺调节
        self.dopamine_level = 0.5  # 中性
        self.eligibility_trace = {}
        self.trace_decay = 0.9

        # 习惯系统
        self.habit_memory: List[HabitMemory] = []
        self.habit_threshold = 5  # 重复次数

        # 状态
        self.state = BGState()
        self.training = True

    def forward(
        self,
        state: torch.Tensor,
        epsilon: float = 0.1,
    ) -> ActionSelection:
        """
        动作选择

        Args:
            state: 状态输入
            epsilon: 探索率

        Returns:
            selection: 动作选择结果
        """
        # 检查习惯
        habit_action = self._check_habit(state)
        if habit_action is not None and np.random.random() < self.state.habit_strength:
            # 使用习惯
            q_values = torch.zeros(1, self.n_actions)
            q_values[0, habit_action] = 1.0
            return ActionSelection(
                action=habit_action,
                q_values=q_values.numpy()[0],
                selected_q=1.0,
                action_probs=q_values.numpy()[0],
            )

        # Q值计算
        q_values = self.striatum(state)

        # 多巴胺调制
        if self.dopamine_level > 0.7:
            # 高多巴胺 → exploitation
            action_probs = F.softmax(q_values, dim=-1)
        elif self.dopamine_level < 0.3:
            # 低多巴胺 → 抑制行为
            action_probs = F.softmax(-q_values, dim=-1)
        else:
            # 中性 → ε-greedy
            action_probs = self._epsilon_greedy(q_values, epsilon)

        # 选择动作
        if self.training or np.random.random() < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = action_probs.argmax().item()

        selected_q = q_values[0, action].item()

        self.state.current_action = action
        self.state.action_history.append(action)

        return ActionSelection(
            action=action,
            q_values=q_values.detach().numpy()[0],
            selected_q=selected_q,
            action_probs=action_probs.detach().numpy()[0],
        )

    def _epsilon_greedy(
        self,
        q_values: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """ε-greedy选择"""
        if np.random.random() < epsilon:
            return torch.rand_like(q_values[0])
        else:
            return F.softmax(q_values, dim=-1)[0]

    def _check_habit(self, state: torch.Tensor) -> Optional[int]:
        """检查是否形成习惯"""
        state_np = state.detach().numpy()[0]

        for habit in self.habit_memory:
            if np.allclose(habit.context, state_np, atol=0.3):
                if habit.frequency >= self.habit_threshold:
                    return habit.action

        return None

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        dopamine: float = None,
    ) -> Dict:
        """
        更新价值函数 (TD学习 + 习惯形成)

        Args:
            state: 当前状态
            action: 执行动作
            reward: 奖励
            next_state: 下一状态
            dopamine: 多巴胺信号 (可选)

        Returns:
            update_result: 更新结果
        """
        # 多巴胺信号
        if dopamine is None:
            dopamine = self._compute_dopamine(reward)
        self.dopamine_level = dopamine

        # TD误差
        current_q = self.striatum.get_q_value(state, action)

        # 下一状态最大Q
        next_q_values = self.striatum(next_state)
        max_next_q = next_q_values.max().item() if next_state.numel() > 0 else 0

        # TD误差
        td_error = reward + self.gamma * max_next_q - current_q

        # 更新 eligibility trace
        if action not in self.eligibility_trace:
            self.eligibility_trace[action] = 0
        self.eligibility_trace[action] = (
            self.trace_decay * self.eligibility_trace.get(action, 0) + td_error
        )

        # 习惯形成检查
        self._update_habit_memory(state, action)

        self.state.last_reward = reward

        return {
            'td_error': td_error,
            'dopamine': dopamine,
            'current_q': current_q,
            'next_q': max_next_q,
        }

    def _compute_dopamine(self, reward: float) -> float:
        """计算多巴胺信号"""
        if reward > 0:
            return 0.5 + 0.5 * min(reward, 1.0)
        elif reward < 0:
            return 0.5 + 0.5 * max(reward, -1.0)
        else:
            return 0.3  # 中性

    def _update_habit_memory(self, state: torch.Tensor, action: int):
        """更新习惯记忆"""
        state_np = state.detach().numpy()[0]

        # 查找现有
        for habit in self.habit_memory:
            if habit.action == action:
                if np.allclose(habit.context, state_np, atol=0.3):
                    habit.frequency += 1
                    habit.last_used = len(self.state.action_history)
                    return

        # 新习惯
        self.habit_memory.append(HabitMemory(
            context=state_np,
            action=action,
            frequency=1,
            last_used=len(self.state.action_history),
        ))

        # 限制记忆数量
        if len(self.habit_memory) > 100:
            self.habit_memory.pop(0)

    def set_dopamine(self, level: float):
        """设置多巴胺水平"""
        self.dopamine_level = np.clip(level, 0, 1)

    def get_habit_strength(self) -> float:
        """获取习惯强度"""
        if not self.habit_memory:
            return 0.0

        max_freq = max(h.frequency for h in self.habit_memory)
        return min(1.0, max_freq / self.habit_threshold)

    def get_summary(self) -> Dict:
        """获取摘要"""
        return {
            'current_action': self.state.current_action,
            'dopamine_level': self.dopamine_level,
            'habit_strength': self.get_habit_strength(),
            'habit_count': len(self.habit_memory),
            'last_reward': self.state.last_reward,
            'action_history_len': len(self.state.action_history),
        }


# ============ VTA/SNc (多巴胺来源) ============

class VentralTegmentalArea(nn.Module):
    """
    腹侧被盖区 (VTA)

    内源性多巴胺来源
    """

    def __init__(
        self,
        state_dim: int = 64,
    ):
        super().__init__()

        # 奖励预测网络
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        # 价值网络
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.baseline = 0.0

    def compute_dopamine_signal(
        self,
        state: torch.Tensor,
        reward: float,
    ) -> float:
        """
        计算多巴胺信号

        预期奖励 vs 实际奖励
        """
        predicted = self.reward_predictor(state).item()

        # 实际奖励与预期差异
        reward_error = reward - predicted

        # 更新baseline
        self.baseline = self.baseline * 0.99 + reward * 0.01

        return reward_error

    def predict_reward(self, state: torch.Tensor) -> float:
        """预测奖励"""
        return self.reward_predictor(state).item()


class SubstantiaNigra(nn.Module):
    """
    黑质 (SNc)

    运动多巴胺来源，习惯形成
    """

    def __init__(self):
        super().__init__()

        self.synaptic_weights = {}
        self.plasticity = 0.1

    def compute_motor_dopamine(
        self,
        action_quality: float,
    ) -> float:
        """计算运动多巴胺"""
        return (action_quality + 1) / 2  # 归一化到0-1


# ============ 整合系统 ============

class BasalGangliaSystem(nn.Module):
    """
    完整基底神经节系统

    整合BG + VTA/SNc
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
    ):
        super().__init__()

        self.bg = BasalGanglia(state_dim, n_actions)
        self.vta = VentralTegmentalArea(state_dim)
        self.snc = SubstantiaNigra()

    def forward(
        self,
        state: torch.Tensor,
        epsilon: float = 0.1,
        reward: float = None,
    ) -> Dict:
        """
        执行动作并更新
        """
        # 动作选择
        selection = self.bg(state, epsilon)
        action = selection.action

        # 计算多巴胺
        if reward is not None:
            dopamine = self.vta.compute_dopamine_signal(state, reward)
            self.bg.set_dopamine(dopamine)
            motor_dop = self.snc.compute_motor_dopamine(selection.selected_q)
        else:
            dopamine = 0.5
            motor_dop = 0.5

        return {
            'action': action,
            'q_values': selection.q_values,
            'selected_q': selection.selected_q,
            'dopamine': dopamine,
            'motor_dopamine': motor_dop,
            'habit_strength': self.bg.get_habit_strength(),
        }

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
    ) -> Dict:
        """更新"""
        return self.bg.update(state, action, reward, next_state)

    def learn_habit(
        self,
        context: torch.Tensor,
        action: int,
    ):
        """强制形成习惯"""
        self.bg._update_habit_memory(context, action)

    def get_summary(self) -> Dict:
        """获取摘要"""
        return {
            'bg': self.bg.get_summary(),
            'habit_strength': self.bg.get_habit_strength(),
        }


# ============ 便捷函数 ============

def create_basal_ganglia(
    state_dim: int = 64,
    n_actions: int = 4,
) -> BasalGangliaSystem:
    """创建基底神经节系统"""
    return BasalGangliaSystem(state_dim, n_actions)


__all__ = [
    'BGState',
    'ActionSelection',
    'HabitMemory',
    'StriatumInput',
    'IndirectPathway',
    'DirectPathway',
    'HyperdirectPathway',
    'BasalGanglia',
    'VentralTegmentalArea',
    'SubstantiaNigra',
    'BasalGangliaSystem',
    'create_basal_ganglia',
]