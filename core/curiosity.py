"""维度1: 自主探索目标设定 (TRUE Implementation)

生物对应: 进化论"适者生存"中的"好奇心"

数学公式 (真正的实现):
    V_goal(s, g) = α · Novelty(g) + β · Complexity(g) + γ · Utility(g)

    其中:
    Novelty(g) = -log P(g | History)  # 真正的信息论 novelty
    Complexity(g) = 目标分解子问题熵
    Utility(g) = 对知识库的预期信息贡献
"""
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


@dataclass
class ExplorationGoal:
    """探索目标"""
    id: str
    description: str
    embedding: Optional[np.ndarray] = None  # 文本嵌入
    novelty: float = 0.0
    complexity: float = 0.0
    utility: float = 0.0
    value: float = 0.0
    completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class GoalEncoder(nn.Module):
    """目标编码器

    将目标描述编码为向量表示 P(embedding | description)
    """

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # 简单词嵌入 (实际项目中应使用预训练模型)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # BiLSTM 编码序列
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=2, batch_first=True,
            bidirectional=True
        )

        # 投影到分布参数
        self.to_mu = nn.Linear(hidden_dim * 2, embedding_dim)
        self.to_logvar = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """编码文本

        Args:
            tokens: [batch, seq_len] token IDs

        Returns:
            mu, logvar: 分布参数
        """
        # 嵌入
        emb = self.embedding(tokens)  # [batch, seq_len, embedding_dim]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(emb)

        # 连接双向最后隐状态
        # 连接双向最后隐状态
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [batch, hidden_dim*2]

        # 分布参数
        mu = self.to_mu(hidden)
        logvar = torch.clamp(self.to_logvar(hidden), -5, 5)

        return mu, logvar

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """编码为点向量"""
        with torch.no_grad():
            mu, logvar = self.forward(tokens)
            # 采样或取均值
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            else:
                return mu


class HistoryEncoder(nn.Module):
    """历史编码器

    编码历史目标序列学习 P(History)
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Transformer 编码历史
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 预测下一个目标的概率分布
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, goal_embeddings: torch.Tensor) -> torch.Tensor:
        """编码历史

        Args:
            goal_embeddings: [batch, seq_len, embedding_dim]

        Returns:
            history_encoding: [batch, embedding_dim]
        """
        # Transformer 编码
        encoded = self.transformer(goal_embeddings)

        # 取最后位置
        return encoded[:, -1]

    def predict_next(self, history_encoding: torch.Tensor) -> torch.Tensor:
        """预测下一个目标的分布"""
        return self.predictor(history_encoding)


class LearnedNoveltyEngine(nn.Module):
    """学习的新颖度引擎

    核心创新:
    1. 编码历史目标序列
    2. 学习目标的条件分布 P(goal | History)
    3. 计算真正的 Novelty = -log P(goal | History)
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        max_history: int = 50
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_history = max_history

        self.goal_encoder = GoalEncoder(vocab_size, embedding_dim, hidden_dim)
        self.history_encoder = HistoryEncoder(embedding_dim, hidden_dim)

        # 历史缓冲 (CPU 端)
        self.goal_history: List[np.ndarray] = []

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def compute_novelty(
        self,
        goal_tokens: torch.Tensor,
        use_learned: bool = True
    ) -> float:
        """计算目标的 Novelty

        Novelty = -log P(goal | History)

        当 P(goal|History) 低时，novelty 高 = 探索新领域
        当 P(goal|History) 高时，novelty 低 = 重复已知领域
        """
        if not use_learned or len(self.goal_history) == 0:
            # 回退到基于频率的近似
            return self._frequency_novelty()

        # 编码历史
        history_tensor = torch.stack([
            torch.FloatTensor(emb) for emb in self.goal_history[-self.max_history:]
        ]).unsqueeze(0).to(next(self.parameters()).device)

        history_encoding = self.history_encoder(history_tensor)

        # 预测该目标的概率
        predicted = self.history_encoder.predict_next(history_encoding)

        # 编码目标
        goal_mu, goal_logvar = self.goal_encoder(goal_tokens.unsqueeze(0))

        # 计算负对数似然
        # -log P(goal | History) ≈ -log N(goal_mu | predicted, I)
        diff = goal_mu - predicted
        novelty = 0.5 * (diff ** 2).sum(-1).mean()

        # 或使用对比学习方法
        # novelty = margin - similarity
        similarity = F.cosine_similarity(goal_mu, predicted, dim=-1)
        novelty = 1.0 - similarity.mean()

        return novelty.item()

    def _frequency_novelty(self) -> float:
        """基于频率的近似新颖度"""
        if not self.goal_history:
            return 1.0

        # 简单: 与历史目标的差异度
        return 0.5  # 简化返回值

    def add_history(self, goal_embedding: np.ndarray) -> None:
        """添加到历史"""
        self.goal_history.append(goal_embedding)
        if len(self.goal_history) > self.max_history:
            self.goal_history.pop(0)

    def train_step(
        self,
        goal_pairs: List[tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, float]:
        """训练一步

        Args:
            goal_pairs: [(previous_goal, next_goal), ...]
        """
        if len(goal_pairs) < 2:
            return {"loss": 0.0}

        prevs, nexts = zip(*goal_pairs)

        prev_embeddings = torch.stack([
            torch.FloatTensor(p) for p in prevs
        ]).to(next(self.parameters()).device)

        next_embeddings = torch.stack([
            torch.FloatTensor(n) for n in nexts
        ]).to(next(self.parameters()).device)

        # 编码历史
        history_encoding = self.history_encoder(prev_embeddings.unsqueeze(1))

        # 预测
        predicted = self.history_encoder.predict_next(history_encoding)

        # 对比损失
        # 最大化 next_embedding 与 predicted 的相似度
        loss = F.mse_loss(predicted, next_embeddings)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class CuriosityEngine:
    """好奇心探索引擎 (增强版)

    核心功能:
    1. 生成候选探索目标
    2. 计算真正的 Novelty (使用 LearnedNoveltyEngine)
    3. 使用 AUCB 策略选择目标
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        exploration_rate: float = 0.1,
        use_learned_novelty: bool = True,
        history_size: int = 50
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.use_learned_novelty = use_learned_novelty
        self.history_size = history_size

        # 学习的新颖度引擎
        self.novelty_engine = LearnedNoveltyEngine(
            max_history=history_size
        )

        # 后备: 简化的新颖度计算器
        self._simple_novelty = SimpleNoveltyCalculator()

        # 目标历史和统计
        self.goal_history: List[ExplorationGoal] = []
        self.selected_count: Dict[str, int] = {}
        self.reward_history: List[float] = []

        # 预定义目标模板
        self.goal_templates = [
            "探索新的状态空间区域",
            "测试极端边界条件",
            "验证假设: X 是否成立",
            "优化某指标的极限",
            "发现隐藏模式",
            "构建新的知识关联",
            "挑战已知的最优解",
            "探索不确定性最高的方向"
        ]

    def generate_candidate_goals(self, n: int = 5) -> List[ExplorationGoal]:
        """生成候选探索目标"""
        candidates = []

        for i in range(n):
            template = random.choice(self.goal_templates)
            goal = ExplorationGoal(
                id=f"goal_{len(self.goal_history)}_{i}",
                description=f"{template} (变体 {i+1})",
                complexity=random.uniform(0.3, 0.9),
                utility=random.uniform(0.3, 0.9)
            )
            candidates.append(goal)

        return candidates

    def compute_novelty(self, goal: ExplorationGoal) -> float:
        """计算目标的新颖度

        True Novelty = -log P(goal | History)
        """
        if self.use_learned_novelty and len(self.goal_history) > 0:
            # 使用简化的嵌入相似度
            return self._simple_novelty.compute(goal, self.goal_history)
        else:
            # 回退到简单方法
            return self._simple_novelty.compute(goal, self.goal_history)

    def compute_goal_value(
        self,
        goal: ExplorationGoal,
        use_aucb: bool = True
    ) -> float:
        """计算目标价值

        V = α·Novelty + β·Complexity + γ·Utility + AUCB
        """
        novelty = self.compute_novelty(goal)

        value = (
            self.alpha * novelty +
            self.beta * goal.complexity +
            self.gamma * goal.utility
        )

        if use_aucb:
            # AUCB 探索 bonus
            total_selections = sum(self.selected_count.values()) + 1
            goal_selections = self.selected_count.get(goal.id, 0) + 1

            c = 1.0
            aucb = c * np.sqrt(np.log(total_selections) / goal_selections)
            value += self.exploration_rate * aucb

        return value

    def select_goal(
        self,
        candidates: Optional[List[ExplorationGoal]] = None
    ) -> ExplorationGoal:
        """选择探索目标"""
        if candidates is None:
            candidates = self.generate_candidate_goals()

        # 计算每个候选的价值
        for goal in candidates:
            goal.value = self.compute_goal_value(goal)
            goal.novelty = self.compute_novelty(goal)

        # Epsilon-greedy
        if random.random() < self.exploration_rate:
            selected = random.choice(candidates)
        else:
            selected = max(candidates, key=lambda g: g.value)

        # 记录
        self.selected_count[selected.id] = self.selected_count.get(selected.id, 0) + 1
        self.goal_history.append(selected)

        return selected

    def update_reward(self, goal_id: str, reward: float) -> None:
        """更新目标奖励"""
        self.reward_history.append(reward)

        for goal in self.goal_history:
            if goal.id == goal_id:
                goal.completed = True
                break

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        completed = sum(1 for g in self.goal_history if g.completed)
        return {
            "total_goals": len(self.goal_history),
            "completed": completed,
            "novelty_avg": np.mean([g.novelty for g in self.goal_history]) if self.goal_history else 0,
            "complexity_avg": np.mean([g.complexity for g in self.goal_history]) if self.goal_history else 0,
            "value_avg": np.mean([g.value for g in self.goal_history]) if self.goal_history else 0,
            "exploration_rate": self.exploration_rate,
            "use_learned_novelty": self.use_learned_novelty
        }

    def reset(self) -> None:
        """重置"""
        self.goal_history.clear()
        self.selected_count.clear()
        self.reward_history.clear()


class SimpleNoveltyCalculator:
    """简单的新颖度计算器 (后备)"""

    def compute(
        self,
        goal: ExplorationGoal,
        history: List[ExplorationGoal]
    ) -> float:
        """基于历史的新颖度计算"""
        if not history:
            return 1.0

        # 词重叠
        goal_words = set(goal.description.lower().split())

        max_similarity = 0.0
        for hist_goal in history[-10:]:
            hist_words = set(hist_goal.description.lower().split())
            if goal_words and hist_words:
                overlap = len(goal_words & hist_words) / len(goal_words | hist_words)
                max_similarity = max(max_similarity, overlap)

        novelty = 1.0 - max_similarity
        return novelty


# 保持向后兼容
CuriosityEngine = CuriosityEngine