"""维度1: 自主探索目标设定 (TRUE Implementation)

生物对应: 进化论"适者生存"中的"好奇心"

数学公式 (真正的实现):
    V_goal(s, g) = α · Novelty(g) + β · Complexity(g) + γ · Utility(g)

    其中:
    Novelty(g) = -log P(g | History)  # 真正的信息论 novelty
    Complexity(g) = 目标分解子问题熵
    Utility(g) = 对知识库的预期信息贡献

事件驱动:
    - 订阅 GOAL_NEEDED: 收到请求时生成并选择目标
    - 发布 GOAL_SELECTED: 目标选定后通知下游
"""
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.events import GOAL_NEEDED, GOAL_SELECTED


@dataclass
class ExplorationGoal:
    """探索目标"""
    id: str
    description: str
    embedding: np.ndarray | None = None  # 文本嵌入
    novelty: float = 0.0
    complexity: float = 0.0
    utility: float = 0.0
    value: float = 0.0
    completed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


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
        self.goal_history: list[np.ndarray] = []

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
        goal_pairs: list[tuple[np.ndarray, np.ndarray]]
    ) -> dict[str, float]:
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
        history_size: int = 50,
        event_bus=None,
        world_model=None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.use_learned_novelty = use_learned_novelty
        self.history_size = history_size

        # 世界模型引用 (用于不确定性驱动目标生成)
        self._world_model = world_model

        # 事件总线
        self._bus = event_bus
        if self._bus is not None:
            self._bus.subscribe(GOAL_NEEDED, self.on_goal_needed, priority=0, name="curiosity")

        # 学习的新颖度引擎
        self.novelty_engine = LearnedNoveltyEngine(
            max_history=history_size
        )

        # 后备: 简化的新颖度计算器
        self._simple_novelty = SimpleNoveltyCalculator()

        # 目标历史和统计
        self.goal_history: list[ExplorationGoal] = []
        self.selected_count: dict[str, int] = {}
        self.reward_history: list[float] = []

        # 探索反馈记录 (Phase 5 闭环)
        self._ig_feedback: dict[str, dict[str, float]] = {}  # goal_id -> {ig, lp}

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

    def _description_to_tokens(self, description: str) -> torch.Tensor | None:
        """将目标描述转换为 token tensor

        使用字符级 hash tokenizer:
        - 每个字符 → hash(char) % vocab_size → token ID
        - 截断/填充到固定长度
        """
        if not description:
            return None

        vocab_size = self.novelty_engine.goal_encoder.vocab_size
        max_len = 32  # 最大序列长度

        # 字符级 hash tokenize
        token_ids = []
        for ch in description[:max_len]:
            token_id = hash(ch) % vocab_size
            token_ids.append(token_id)

        # 填充到固定长度
        while len(token_ids) < max_len:
            token_ids.append(0)  # padding token

        return torch.LongTensor(token_ids)

    def _encode_goal(self, goal: ExplorationGoal) -> np.ndarray | None:
        """将目标编码为向量嵌入"""
        tokens = self._description_to_tokens(goal.description)
        if tokens is None:
            return None
        with torch.no_grad():
            embedding = self.novelty_engine.goal_encoder.encode(tokens.unsqueeze(0))
            return embedding.squeeze(0).cpu().numpy()

    def generate_candidate_goals(self, n: int = 5, state_vector: np.ndarray = None) -> list[ExplorationGoal]:
        """生成候选探索目标 (不确定性驱动 + 模板后备)

        Phase 4: 当世界模型可用且有真实状态向量时，
        在潜在空间中采样多个方向，选不确定性最高的方向作为目标。
        否则回退到模板生成。
        """
        candidates = []

        # ---- 不确定性驱动生成 ----
        if (self._world_model is not None
                and state_vector is not None
                and len(self.goal_history) >= 3):
            try:
                uncertainty_goals = self._generate_uncertainty_goals(state_vector, n)
                if uncertainty_goals:
                    candidates.extend(uncertainty_goals)
            except RuntimeError:
                pass  # World model uncertainty goal generation failed

        # ---- 模板后备 (填充不足的部分) ----
        remaining = n - len(candidates)
        for i in range(remaining):
            template = random.choice(self.goal_templates)
            # 利用IG反馈调整模板权重
            direction = random.choice(["正向", "逆向", "边界", "极值", "对比", "迁移"])
            description = f"{template} [{direction}路径 v{i+1}]"

            # 从历史反馈推断 complexity/utility
            if self._ig_feedback:
                avg_ig = np.mean([f["ig"] for f in self._ig_feedback.values()])
                avg_lp = np.mean([f.get("lp", 0) for f in self._ig_feedback.values()])
                complexity = 0.3 + 0.4 * min(1.0, avg_ig + avg_lp)
                utility = 0.3 + 0.4 * min(1.0, avg_lp)
            else:
                complexity = random.uniform(0.3, 0.9)
                utility = random.uniform(0.3, 0.9)

            goal = ExplorationGoal(
                id=f"goal_{len(self.goal_history)}_{len(candidates)}",
                description=description,
                complexity=complexity,
                utility=utility,
            )

            # 编码目标为embedding
            embedding = self._encode_goal(goal)
            if embedding is not None:
                goal.embedding = embedding
                self.novelty_engine.add_history(embedding)

            candidates.append(goal)

        return candidates

    def _generate_uncertainty_goals(
        self, state_vector: np.ndarray, n: int
    ) -> list[ExplorationGoal]:
        """从世界模型预测不确定性中生成目标 (Phase 4)

        在潜在空间中采样多个方向，计算每个方向的世界模型预测方差，
        选不确定性最高的方向。
        """
        import torch
        goals = []
        wm = self._world_model
        device = next(wm.parameters()).device

        state_t = torch.FloatTensor(state_vector).unsqueeze(0).to(device)

        # 采样多个action方向
        n_samples = min(n * 3, 30)
        uncertainties = []

        for _ in range(n_samples):
            # 随机action (one-hot)
            action_idx = random.randint(0, wm.n_actions - 1)
            action_t = torch.zeros(1, wm.n_actions, device=device)
            action_t[0, action_idx] = 1.0

            with torch.no_grad():
                pred_mu, pred_std, kl, _ = wm(state_t, action_t)
                # 不确定性 = 预测标准差的均值
                uncertainty = pred_std.mean().item()

            uncertainties.append((action_idx, uncertainty, pred_std))

        # 按不确定性降序排序
        uncertainties.sort(key=lambda x: x[1], reverse=True)

        # 取前n个最不确定的方向
        for rank, (action_idx, uncertainty, pred_std) in enumerate(uncertainties[:n]):
            # 找到最不确定的维度
            top_dim = pred_std.argmax().item()
            direction_desc = f"维度{top_dim}(σ={uncertainty:.3f})"

            description = (
                f"探索不确定性热点: {direction_desc} "
                f"[action={action_idx}, rank={rank+1}]"
            )

            goal = ExplorationGoal(
                id=f"goal_{len(self.goal_history)}_u{rank}",
                description=description,
                complexity=min(0.95, 0.4 + uncertainty * 2.0),
                utility=min(0.95, 0.3 + uncertainty * 1.5),
                metadata={"uncertainty": uncertainty, "action": action_idx},
            )

            embedding = self._encode_goal(goal)
            if embedding is not None:
                goal.embedding = embedding
                self.novelty_engine.add_history(embedding)

            goals.append(goal)

        return goals

    def compute_novelty(self, goal: ExplorationGoal) -> float:
        """计算目标的新颖度

        True Novelty = -log P(goal | History)
        """
        if self.use_learned_novelty and len(self.goal_history) >= 2:
            try:
                # 使用学习的novelty引擎（BiLSTM + Transformer）
                goal_tokens = self._description_to_tokens(goal.description)
                if goal_tokens is not None:
                    return self.novelty_engine.compute_novelty(goal_tokens)
            except RuntimeError:
                pass  # World model uncertainty goal generation failed
            # fallback to simple novelty
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
        candidates: list[ExplorationGoal] | None = None
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

        # 训练novelty引擎：如果前一个目标存在，用(前一个, 当前)对训练
        if selected.embedding is not None and len(self.goal_history) >= 2:
            prev = self.goal_history[-2]
            if prev.embedding is not None:
                self.novelty_engine.train_step([(prev.embedding, selected.embedding)])

        return selected

    def update_reward(self, goal_id: str, reward: float) -> None:
        """更新目标奖励"""
        self.reward_history.append(reward)

        for goal in self.goal_history:
            if goal.id == goal_id:
                goal.completed = True
                break

    def update_exploration_result(
        self, goal_id: str, ig_reward: float, learning_progress: float = 0.0
    ) -> None:
        """探索结果反馈闭环 (Phase 5)

        高IG + 高LP → 方向值得深入
        高IG + 低LP → 方向已学过，换方向
        低IG → 方向无趣
        """
        self._ig_feedback[goal_id] = {
            "ig": ig_reward,
            "lp": learning_progress,
        }

        # 保留最近50条反馈
        if len(self._ig_feedback) > 50:
            oldest = list(self._ig_feedback.keys())[0]
            del self._ig_feedback[oldest]

        # 调整探索策略
        if ig_reward > 0.5 and learning_progress > 0.1:
            # 高进步 → 可略微降低该方向的探索率（已找到有价值的方向）
            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
        elif ig_reward > 0.5 and learning_progress < 0.05:
            # 高IG但没进步 → 可能是噪声，增加探索多样性
            self.exploration_rate = min(0.4, self.exploration_rate * 1.1)
        elif ig_reward < 0.1:
            # 无趣方向 → 维持当前探索率
            pass

    def get_statistics(self) -> dict[str, Any]:
        """获取统计"""
        completed = sum(1 for g in self.goal_history if g.completed)
        ig_values = [f["ig"] for f in self._ig_feedback.values()] if self._ig_feedback else []
        lp_values = [f.get("lp", 0) for f in self._ig_feedback.values()] if self._ig_feedback else []
        return {
            "total_goals": len(self.goal_history),
            "completed": completed,
            "novelty_avg": np.mean([g.novelty for g in self.goal_history]) if self.goal_history else 0,
            "complexity_avg": np.mean([g.complexity for g in self.goal_history]) if self.goal_history else 0,
            "value_avg": np.mean([g.value for g in self.goal_history]) if self.goal_history else 0,
            "exploration_rate": self.exploration_rate,
            "use_learned_novelty": self.use_learned_novelty,
            "ig_feedback_count": len(self._ig_feedback),
            "avg_ig": np.mean(ig_values) if ig_values else 0,
            "avg_lp": np.mean(lp_values) if lp_values else 0,
            "uncertainty_driven": self._world_model is not None,
        }

    def reset(self) -> None:
        """重置"""
        self.goal_history.clear()
        self.selected_count.clear()
        self.reward_history.clear()

    def on_goal_needed(self, event) -> dict[str, Any]:
        """事件驱动: 响应 GOAL_NEEDED，生成并选择目标"""
        emotion_state = event.data.get("emotion_state", {})
        state_vector = event.data.get("state_vector")
        emotion_bonus = 0.0

        # 情绪影响因子
        if emotion_state:
            mood_valence = emotion_state.get('mood_valence', 0.0)
            mood_arousal = emotion_state.get('mood_arousal', 0.5)
            if mood_valence < -0.3:
                emotion_bonus = -0.2
            if mood_arousal > 0.7:
                emotion_bonus = 0.15

        candidates = self.generate_candidate_goals(n=5, state_vector=state_vector)

        if emotion_bonus != 0.0:
            for goal in candidates:
                goal.value = goal.value * (1.0 + emotion_bonus)

        selected = self.select_goal(candidates)

        # 发布目标选定事件
        if self._bus is not None:
            self._bus.publish(
                GOAL_SELECTED,
                {"goal": selected, "emotion_bonus": emotion_bonus},
                source="curiosity",
            )

        return {"goal": selected, "emotion_bonus": emotion_bonus}


class SimpleNoveltyCalculator:
    """简单的新颖度计算器 (后备)"""

    def compute(
        self,
        goal: ExplorationGoal,
        history: list[ExplorationGoal]
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
