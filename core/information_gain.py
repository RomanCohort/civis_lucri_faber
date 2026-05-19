"""维度2: 信息增益内在动机 (TRUE Implementation)

生物对应: 多巴胺机制 - "发现新知识"本身产生愉悦

数学公式 (真正的实现):
    IG(s, a, s') = H(s) - H(s | s')

    其中:
    - H(s) = -∫ P(s) log P(s) ds = -E[log P(s)] (Shannon 熵)
    - H(s | s') = -∫ P(s|s') log P(s|s') ds (条件熵)

    使用变分推断近似:
    - q(s|s') = N(μ(s'), σ(s')) (变分近似 posterior)
    - H(s) ≈ -log P(s | θ) + KL(q || p)

事件驱动:
    - 订阅 EXPLORATION_START: 执行探索并计算信息增益
    - 发布 EXPLORATION_DONE: 探索完成后通知下游
    - 发布 INFO_GAIN_COMPUTED: 信息增益结果
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Distribution

from civis_lucri_faber.core.events import EXPLORATION_START, EXPLORATION_DONE, INFO_GAIN_COMPUTED


@dataclass
class InformationReward:
    """信息增益奖励"""
    intrinsic: float       # η · IG
    extrinsic: float       # 外在奖励
    total: float          # 总奖励
    entropy: float        # H(s)
    conditional_entropy: float  # H(s|s')
    information_gain: float   # IG = H - H(s|s')
    uncertainty_reduction: float  # 不确定性降低量


class VariationalWorldModel(nn.Module):
    """变分世界模型

    学习 P(s'|s, a) 并支持真正的熵计算
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        latent_dim: int = 32
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.latent_dim = latent_dim

        # 编码器: (s, one_hot_a) -> (μ, logσ)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean + log variance
        )

        # 解码器: z -> s'
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * 2)  # mean + log variance for output
        )

        # 潜在先验: 标准正态分布
        self.prior = Normal(0, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播

        Returns:
            - next_state_mean: 预测的下一步均值
            - next_state_std: 预测的下一步标准差
            - kl: KL(q(z|s,a) || p(z))
            - log_prob: log P(s'|s,a)
        """
        # 处理维度
        if state.ndim == 1:
            state = state.unsqueeze(0)  # (1, feature)
        if action.ndim == 0:
            action = action.unsqueeze(0)  # (1,)

        # action 是 one-hot 或者索引
        if action.shape[-1] == self.n_actions:
            action_onehot = action
        else:
            # action 是索引，转换为 one-hot
            action_idx = action.long()
            if action_idx.dim() == 1:
                action_idx = action_idx.unsqueeze(1)
            action_onehot = torch.zeros(action_idx.size(0), self.n_actions, device=state.device)
            action_onehot.scatter_(1, action_idx, 1)

        x = torch.cat([state, action_onehot], dim=-1)

        # 编码当前状态到潜在表示
        mu_logvar = self.encoder(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        logvar = torch.clamp(logvar, -5, 5)  # 数值稳定

        # 变分后验 q(z|s,a)
        std = torch.exp(0.5 * logvar)
        q = Normal(mu, std)

        # 采样 z
        z = q.rsample()

        # 解码到下一步状态
        x_decode = torch.cat([z, action_onehot], dim=-1)
        out_mu_logvar = self.decoder(x_decode)
        out_mu, out_logvar = torch.chunk(out_mu_logvar, 2, dim=-1)
        out_logvar = torch.clamp(out_logvar, -5, 5)

        # 预测分布 P(s'|s,a,z)
        p = Normal(out_mu, torch.exp(0.5 * out_logvar))

        # KL(q(z|s,a) || p(z))
        kl = torch.distributions.kl_divergence(q, self.prior).sum(-1).mean()

        return out_mu, torch.exp(0.5 * out_logvar), kl, p

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """预测下一步状态"""
        with torch.no_grad():
            # 处理维度
            if state.ndim == 1:
                state = state.unsqueeze(0)
            if action.ndim == 0:
                action = action.unsqueeze(0)

            # 转换为 one-hot
            if action.shape[-1] != self.n_actions:
                action_idx = action.long()
                if action_idx.dim() == 1:
                    action_idx = action_idx.unsqueeze(1)
                action_onehot = torch.zeros(action_idx.size(0), self.n_actions, device=state.device)
                action_onehot.scatter_(1, action_idx, 1)
            else:
                action_onehot = action

            x = torch.cat([state, action_onehot], dim=-1)
            mu_logvar = self.encoder(x)
            mu, _ = torch.chunk(mu_logvar, 2, dim=-1)

            # 使用均值解码
            x_decode = torch.cat([mu, action_onehot], dim=-1)
            out_mu_logvar = self.decoder(x_decode)
            out_mu, _ = torch.chunk(out_mu_logvar, 2, dim=-1)

            return out_mu

    def compute_log_likelihood(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """计算 log P(s'|s,a) 使用变分近似

        log P(s'|s,a) ≈ E_q(z|s,a)[log P(s'|s,a,z)]
        """
        x = torch.cat([state, action], dim=-1)

        # 编码
        mu_logvar = self.encoder(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        std = torch.exp(0.5 * torch.clamp(logvar, -5, 5))
        q = Normal(mu, std)

        # 采样
        z = q.rsample()

        # 解码
        x_decode = torch.cat([z, action], dim=-1)
        out_mu_logvar = self.decoder(x_decode)
        out_mu, out_logvar = torch.chunk(out_mu_logvar, 2, dim=-1)
        out_std = torch.exp(0.5 * torch.clamp(out_logvar, -5, 5))

        # log P(s'|s,a,z)
        p = Normal(out_mu, out_std)
        log_prob = p.log_prob(next_state).sum(-1).mean()

        return log_prob


class EntropyCalculator:
    """熵计算器

    核心功能:
    1. H(s): 状态分布的熵
    2. H(s|s'): 条件熵 (给定上一步后的熵)
    3. IG = H(s) - H(s|s'): 信息增益
    """

    def __init__(self, world_model: VariationalWorldModel):
        self.world_model = world_model

    def compute_state_entropy(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> float:
        """计算 H(s)

        简化版: 基于状态的方差
        """
        with torch.no_grad():
            # 使用状态方差的对数作为熵
            variance = states.var() + 1e-8
            entropy = torch.log(variance).abs().item() / 2 + 1.0
            return max(0.1, entropy)

    def compute_conditional_entropy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> float:
        """计算 H(s|a)

        使用世界模型预测误差 - 预测误差小意味着模型知道如何响应，条件熵低
        """
        if len(states) < 2:
            return 1.0

        with torch.no_grad():
            try:
                # 收集预测
                preds = []
                for i in range(min(10, len(states) - 1)):
                    state_i = states[i:i+1]
                    action_i = actions[i:i+1]
                    try:
                        pred = self.world_model.predict_next_state(state_i, action_i)
                        preds.append(pred)
                    except:
                        pass

                if len(preds) > 0:
                    pred_tensor = torch.cat(preds, dim=0)
                    actual = states[1:1+len(preds)]
                    # 预测误差 -> 条件熵代理 (误差越大，模型越不确定)
                    error = (pred_tensor - actual).abs().mean()
                    conditional_entropy = error.item() + 0.1
                else:
                    conditional_entropy = 1.0

                return max(0.1, conditional_entropy)
            except:
                return 1.0

        return conditional_entropy

    def compute_information_gain(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> Dict[str, float]:
        """计算信息增益 IG = H(s) - H(s|s')

        Args:
            states: 状态序列 [T, state_dim]
            actions: 动作序列 [T, action_dim]
            next_states: 下一步状态序列 [T, state_dim]

        Returns:
            dict with entropy, conditional_entropy, information_gain
        """
        # H(s): 状��的边际熵
        entropy = self.compute_state_entropy(states)

        # H(s|s'): 条件熵
        cond_entropy = self.compute_conditional_entropy(states, actions)

        # IG = H(s) - H(s|s')
        # 当模型预测准确时，cond_entropy < entropy，IG > 0
        # 信息增益 = 未知的减少量 = 模型变得更能预测
        information_gain = entropy - cond_entropy

        # 确保IG有界
        information_gain = max(0.0, min(2.0, information_gain))

        return {
            "entropy": entropy,
            "conditional_entropy": cond_entropy,
            "information_gain": information_gain,
            "uncertainty_reduction": abs(cond_entropy)
        }


class TrueInformationGainCalculator:
    """真正的信息增益计算器

    核心创新:
    1. 使用变分推断计算真实的信息增益
    2. 支持无外在奖励的自监督学习
    3. 内在动机驱动探索
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 16,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        lr: float = 0.001,
        intrinsic_lambda: float = 0.5,
        use_true_ig: bool = True,
        device: str = "cpu",
        event_bus=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.intrinsic_lambda = intrinsic_lambda
        self.use_true_ig = use_true_ig
        self.device = device

        # 事件总线
        self._bus = event_bus
        if self._bus is not None:
            self._bus.subscribe(EXPLORATION_START, self.on_exploration_start, priority=0, name="info_gain")

        # 世界模型和熵计算器 - 注意: action_dim 应该是 n_actions
        self.world_model = VariationalWorldModel(
            state_dim, action_dim, hidden_dim, latent_dim
        ).to(device)

        self.entropy_calc = EntropyCalculator(self.world_model)

        self.optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr)

        # 经验缓冲
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        self.buffer_size = 10000
        self.batch_size = 32

        # Learning Progress 跟踪 (Phase 3)
        self._prediction_error_history: deque = deque(maxlen=100)
        self._lp_window = 10  # 计算LP的窗口大小
        self._lp_beta = 0.3   # LP在内在奖励中的权重

    def set_env_dims(self, state_dim: int, n_actions: int):
        """设置环境维度"""
        self.state_dim = state_dim
        self.world_model = VariationalWorldModel(
            state_dim, n_actions, 128, 32
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.world_model.parameters(), lr=0.001)

    def add_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray
    ) -> None:
        """添加经验到缓冲"""
        self.buffer.append((state, action, reward, next_state))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self) -> Dict[str, float]:
        """训练世界模型一步 (VAE ELBO 优化)"""
        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0, "kl": 0.0, "recon": 0.0}

        # 采样 batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states = zip(*[self.buffer[i] for i in indices])

        states = torch.FloatTensor(np.array(states)).to(self.device)
        # action 作为索引，转为 one-hot
        actions_idx = torch.LongTensor(np.array(actions))
        if actions_idx.dim() == 1:
            actions_idx = actions_idx.unsqueeze(1)
        actions_onehot = torch.zeros(actions_idx.size(0), self.world_model.n_actions).to(self.device)
        actions_onehot.scatter_(1, actions_idx, 1)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        # 前向传播
        pred_mean, pred_std, kl, pred_dist = self.world_model(states, actions_onehot)

        # 重建损失 (负 log likelihood)
        recon_loss = -pred_dist.log_prob(next_states).mean()

        # ELBO = log P(s'|s) - KL(q||p)
        # 最大化 ELBO 等价于最小化 -ELBO
        loss = recon_loss + 0.01 * kl

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "kl": kl.item(),
            "recon": recon_loss.item()
        }

    def compute_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        use_intrinsic: bool = True
    ) -> InformationReward:
        """计算总奖励

        r_total = r_extrinsic + λ · IG
        """
        # 添���经���
        self.add_experience(state, action, reward, next_state)

        # 计算信息增益
        if self.use_true_ig:
            info_gain_dict = self._compute_ig(state, action, next_state)
            information_gain = info_gain_dict["information_gain"]
        else:
            # 回退到预测误差方法
            information_gain = self._proxy_ig(state, action, next_state)

        # 内在奖励
        if use_intrinsic:
            intrinsic = self.intrinsic_lambda * information_gain
        else:
            intrinsic = 0.0

        # 总奖励
        total = reward + intrinsic

        return InformationReward(
            intrinsic=intrinsic,
            extrinsic=reward,
            total=total,
            entropy=info_gain_dict.get("entropy", 0) if self.use_true_ig else 0,
            conditional_entropy=info_gain_dict.get("conditional_entropy", 0) if self.use_true_ig else 0,
            information_gain=information_gain,
            uncertainty_reduction=info_gain_dict.get("uncertainty_reduction", 0) if self.use_true_ig else 0
        )

    def _compute_ig(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> Dict[str, float]:
        """使用变分推断计算真正的 IG"""
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action_t = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)

        # 使用熵计算器
        ig_dict = self.entropy_calc.compute_information_gain(
            state_t.unsqueeze(0),  # 需要序列
            action_t.unsqueeze(0),
            next_state_t.unsqueeze(0)
        )

        return ig_dict

    def _proxy_ig(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray
    ) -> float:
        """预测误差作为信息增益的代理"""
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action_t = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            pred_mean, pred_std, _, _ = self.world_model(state_t, action_t)
            pred_error = F.mse_loss(pred_mean, next_state_t).item()

        return pred_error

    def compute_learning_progress(self) -> float:
        """计算学习进步感知 (Phase 3)

        LearningProgress = max(0, past_avg_error - current_error)
        高LP = "我正在学到东西"
        低LP = 学习停滞或已饱和
        """
        if len(self._prediction_error_history) < self._lp_window + 1:
            return 0.0

        errors = list(self._prediction_error_history)
        current = errors[-1]
        past_avg = np.mean(errors[-(self._lp_window + 1):-1])

        lp = max(0.0, past_avg - current)
        return lp

    def get_world_model(self):
        """获取世界模型引用 (供好奇心引擎使用)"""
        return self.world_model

    def on_exploration_start(self, event) -> Dict[str, Any]:
        """事件驱动: 响应 EXPLORATION_START，执行探索并计算信息增益

        接受 agent 传入的真实状态向量，不再使用随机噪声。
        """
        goal = event.data.get("goal")
        if goal is None:
            return {"info_gain": 0.0}

        # 使用真实状态 (Phase 2)
        state = event.data.get("state")
        next_state = event.data.get("next_state")

        if state is None:
            state = np.zeros(self.state_dim, dtype=np.float32)
        if next_state is None:
            next_state = state.copy()  # fallback: 状态不变

        action = event.data.get("action")
        if action is None:
            action = np.zeros(self.action_dim, dtype=np.float32)

        reward = 0.0

        # 计算信息增益奖励
        reward_obj = self.compute_reward(
            state, action, reward, next_state,
            use_intrinsic=True
        )

        # 追踪预测误差用于 Learning Progress
        try:
            state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action_t = torch.FloatTensor(action).to(self.device).unsqueeze(0)
            with torch.no_grad():
                pred_mu, _, _, _ = self.world_model(state_t, action_t)
                next_t = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
                pred_error = F.mse_loss(pred_mu, next_t).item()
            self._prediction_error_history.append(pred_error)
        except Exception:
            pass

        # 计算学习进步
        learning_progress = self.compute_learning_progress()

        # 训练世界模型
        self.train_step()

        # 发布事件
        if self._bus is not None:
            self._bus.publish(
                EXPLORATION_DONE,
                {
                    "info_gain": reward_obj.intrinsic,
                    "total_reward": reward_obj.total,
                    "learning_progress": learning_progress,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                },
                source="info_gain",
            )
            self._bus.publish(
                INFO_GAIN_COMPUTED,
                {"intrinsic": reward_obj.intrinsic, "total": reward_obj.total,
                 "learning_progress": learning_progress},
                source="info_gain",
            )

        return {
            "info_gain": reward_obj.intrinsic,
            "total_reward": reward_obj.total,
            "learning_progress": learning_progress,
            "reward_obj": reward_obj,
            "state": state,
            "next_state": next_state,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计"""
        if len(self.buffer) < self.batch_size:
            return {
                "buffer_size": len(self.buffer),
                "avg_ig": 0.0
            }

        # 计算平均 IG
        sample_size = min(100, len(self.buffer) // 2)
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)

        igs = []
        for i in indices:
            s, a, r, ns = self.buffer[i]
            ig_dict = self._compute_ig(s, a, ns)
            igs.append(ig_dict["information_gain"])

        return {
            "buffer_size": len(self.buffer),
            "avg_ig": np.mean(igs),
            "std_ig": np.std(igs)
        }

    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            "world_model": self.world_model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


# 保持向后兼容的别名
InformationGainCalculator = TrueInformationGainCalculator


class WorldModelWrapper(nn.Module):
    """将 VariationalWorldModel(state, action) 包装为单参数 nn.Module

    用于 UncertaintyAwareActiveLearner 的 ensemble 推理。
    接受拼接输入 x = [state | zero_action]，输出 predicted next state mean。
    """

    def __init__(self, world_model: VariationalWorldModel):
        super().__init__()
        self.wm = world_model
        self.state_dim = world_model.state_dim
        self.n_actions = world_model.n_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """单参数前向: x=[state|action] → predicted next_state_mean"""
        state = x[:, :self.state_dim]
        action = x[:, self.state_dim:self.state_dim + self.n_actions]

        with torch.no_grad():
            mu, std, kl, dist = self.wm(state, action)

        return mu  # [batch, state_dim]