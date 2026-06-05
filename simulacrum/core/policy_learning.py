"""简单策略学习模块 - Q-Learning 实现"""

import numpy as np


class SimpleQLearning:
    """简化版 Q-Learning"""

    def __init__(
        self,
        state_dim: int = 4,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Q表 - 使用字典避免维度问题
        self.q_table: dict[tuple, np.ndarray] = {}

    def _discretize_state(self, state: np.ndarray, n_bins: int = 10) -> tuple:
        """将连续状态离散化"""
        # 将状态映射到n_bins个桶
        bins = [np.linspace(0, 1, n_bins) for _ in range(len(state))]
        discretized = []
        for i, s in enumerate(state):
            idx = np.digitize(s, bins[i]) - 1
            idx = max(0, min(n_bins - 1, idx))
            discretized.append(idx)
        return tuple(discretized)

    def get_q(self, state: np.ndarray) -> np.ndarray:
        """获取Q值"""
        state_key = self._discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作"""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        q_values = self.get_q(state)
        return int(np.argmax(q_values))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """更新Q值"""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_actions)

        # TD目标
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])

        # TD更新
        self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])

    def save(self, path: str) -> None:
        """保存Q表"""
        np.save(path, self.q_table, allow_pickle=True)

    def load(self, path: str) -> None:
        """加载Q表"""
        self.q_table = np.load(path, allow_pickle=True).item()


class EpsilonGreedyBaseline:
    """ε-greedy基线"""

    def __init__(self, n_actions: int = 4, epsilon: float = 0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.action_counts = np.zeros(n_actions)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy选择"""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        # 偏向访问次数少的动作
        counts = self.action_counts + 1e-8
        probs = 1.0 / counts
        probs = probs / probs.sum()
        return np.random.choice(self.n_actions, p=probs)


class UCBAction:
    """UCB动作选择基线"""

    def __init__(self, n_actions: int = 4, c: float = 1.0):
        self.n_actions = n_actions
        self.c = c
        self.action_counts = np.zeros(n_actions)
        self.total_counts = 0
        self.action_rewards = np.zeros(n_actions)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """UCB选择"""
        self.total_counts += 1

        if self.total_counts < self.n_actions:
            return self.total_counts % self.n_actions

        # UCB分数
        means = self.action_rewards / (self.action_counts + 1e-8)
        ucb_scores = means + self.c * np.sqrt(
            np.log(self.total_counts) / (self.action_counts + 1e-8)
        )

        action = int(np.argmax(ucb_scores))
        self.action_counts[action] += 1
        return action

    def update(self, action: int, reward: float) -> None:
        """更新统计"""
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.action_rewards[action] = (
            (n - 1) / n * self.action_rewards[action] + reward / n
        )
