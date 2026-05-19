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


@dataclass
class ArchivedSkill:
    """已存档技能 (从BG转移到小脑的肌肉记忆)"""
    context_hash: int          # 上下文哈希 (快速匹配)
    action: int                # 动作
    skill_level: float         # 技能熟练度 0-1
    motor_pattern: np.ndarray  # 执行该动作的运动模式
    archived_at_step: int      # 存档时的步数
    execution_count: int = 0   # 存档后执行次数


class SkillArchive:
    """
    技能存档系统

    对应"熟能生巧"：BG将重复达标的动作存档到小脑，
    释放意识资源 (conscious_load 降低)。

    生物对应：
    - BG从goal-directed控制 → 转为habit控制
    - 小脑接管自动执行 → 释放前额叶资源
    - 类似人类学骑车：先全神贯注(BG+PFC)，后无需思考(小脑自动)
    """

    def __init__(self, max_skills: int = 50):
        self.skills: List[ArchivedSkill] = []
        self.max_skills = max_skills
        self.total_archived = 0

    def archive(
        self,
        context: np.ndarray,
        action: int,
        frequency: int,
        motor_pattern: np.ndarray,
        step: int,
    ) -> ArchivedSkill:
        """存档一个技能"""
        context_hash = self._hash_context(context)
        skill_level = min(1.0, frequency / 10.0)

        skill = ArchivedSkill(
            context_hash=context_hash,
            action=action,
            skill_level=skill_level,
            motor_pattern=motor_pattern.copy(),
            archived_at_step=step,
        )
        self.skills.append(skill)
        self.total_archived += 1

        # 限制数量
        if len(self.skills) > self.max_skills:
            self.skills.pop(0)

        return skill

    def find_skill(self, context: np.ndarray, action: int) -> Optional[ArchivedSkill]:
        """查找匹配的已存档技能"""
        context_hash = self._hash_context(context)
        for skill in self.skills:
            if skill.action == action and skill.context_hash == context_hash:
                return skill
        return None

    def has_skill(self, context: np.ndarray) -> Optional[ArchivedSkill]:
        """检查该上下文是否有任何已存档技能"""
        context_hash = self._hash_context(context)
        best = None
        for skill in self.skills:
            if skill.context_hash == context_hash:
                if best is None or skill.skill_level > best.skill_level:
                    best = skill
        return best

    def record_execution(self, context: np.ndarray, action: int):
        """记录自动执行次数"""
        skill = self.find_skill(context, action)
        if skill is not None:
            skill.execution_count += 1

    def get_summary(self) -> Dict:
        return {
            'total_archived': self.total_archived,
            'active_skills': len(self.skills),
            'total_auto_executions': sum(s.execution_count for s in self.skills),
        }

    @staticmethod
    def _hash_context(context: np.ndarray) -> int:
        """将连续上下文离散化为哈希 (用于快速匹配)"""
        # 将每个维度分为3档: < -0.3, [-0.3, 0.3], > 0.3
        discretized = np.sign(context + 0.3) + np.sign(context - 0.3)
        return hash(tuple(discretized.astype(int).tolist()))


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

        # 技能存档系统 (BG → 小脑转移)
        self.skill_archive = SkillArchive()
        self.conscious_load = 1.0  # 1.0=全部需意识控制, 0.0=全部自动化
        self._cerebellum = None   # 耦合的小脑引用
        self.archive_threshold = 8  # 频率超过此值触发存档

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
        # 检查习惯 (概率性混合，非完全覆盖)
        habit_action = self._check_habit(state)
        q_values = self.striatum(state)

        if habit_action is not None and self.state.habit_strength > 0.1:
            # 习惯性动作的概率性混合 (而非完全覆盖)
            # habit_strength控制习惯Q值的权重，剩余权重由网络Q值承担
            habit_weight = self.state.habit_strength
            goal_directed_weight = 1.0 - habit_weight

            # 构造混合Q值: 习惯动作获得habit_weight，其余按网络Q值缩放
            mixed_q_values = q_values.detach().clone() * goal_directed_weight
            mixed_q_values[0, habit_action] += habit_weight  # 习惯动作额外加权
            q_values = mixed_q_values

        # 多巴胺调制 (连续调制，无硬阈值切换)
        # DA高 → exploitation倾向 (softmax更尖锐); DA低 → exploration (更平坦/反向)
        # 使用温度参数连续控制softmax锐度
        da_temperature = float(np.clip(0.5 + self.dopamine_level, 0.3, 2.0))
        action_probs = F.softmax(q_values / da_temperature, dim=-1)

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
        """检查是否形成习惯 (含技能存档检查)"""
        state_np = state.detach().numpy()[0]

        # 优先检查已存档到小脑的技能 (完全自动化)
        archived = self.skill_archive.has_skill(state_np)
        if archived is not None:
            self.skill_archive.record_execution(state_np, archived.action)
            return archived.action

        # 然后检查BG内部习惯
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
        """计算多巴胺信号 (连续RPE, 无硬阈值)"""
        # 连续的奖励预测误差 (RPE) 映射
        # reward ∈ [-1, 1] → dopamine ∈ [0, 1]
        # 正奖励 → DA > 0.5 (phasic burst)
        # 零奖励 → DA ≈ 0.3 (baseline dips slightly)
        # 负奖励 → DA < 0.3 (phasic dip)
        rpe_dopamine = float(np.clip(0.5 + 0.4 * np.tanh(reward * 2), 0.1, 0.9))
        return rpe_dopamine

    def _update_habit_memory(self, state: torch.Tensor, action: int):
        """更新习惯记忆 (含技能存档触发)"""
        state_np = state.detach().numpy()[0]

        # 查找现有
        for habit in self.habit_memory:
            if habit.action == action:
                if np.allclose(habit.context, state_np, atol=0.3):
                    habit.frequency += 1
                    habit.last_used = len(self.state.action_history)

                    # 检查是否触发技能存档
                    if (habit.frequency >= self.archive_threshold
                            and self.skill_archive.find_skill(state_np, action) is None):
                        self._archive_skill_to_cerebellum(state_np, action, habit.frequency)

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

    def set_cerebellum(self, cerebellum) -> None:
        """耦合小脑 (用于技能转移)"""
        self._cerebellum = cerebellum

    def _archive_skill_to_cerebellum(
        self,
        context: np.ndarray,
        action: int,
        frequency: int,
    ) -> None:
        """将重复达标的技能存档到小脑

        对应"熟能生巧"：意识控制 → 肌肉记忆
        BG释放该动作的意识资源，小脑接管自动执行
        """
        # 生成运动模式 (当前BG Q值作为模式)
        state_t = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.striatum(state_t)
            motor_pattern = q_values.numpy()[0]

        # 存档到BG的skill_archive
        skill = self.skill_archive.archive(
            context=context,
            action=action,
            frequency=frequency,
            motor_pattern=motor_pattern,
            step=len(self.state.action_history),
        )

        # 降低意识负荷
        n_skills = len(self.skill_archive.skills)
        self.conscious_load = max(0.1, 1.0 - n_skills * 0.1)

        # 转移到小脑
        if self._cerebellum is not None:
            self._cerebellum.receive_archived_skill(
                context=context,
                action=action,
                motor_pattern=motor_pattern,
                skill_level=skill.skill_level,
            )

        print(f"[ARCHIVE] Skill archived: action={action}, freq={frequency}, "
              f"skill_level={skill.skill_level:.2f}, conscious_load={self.conscious_load:.2f}")

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
            'conscious_load': self.conscious_load,
            'skill_archive': self.skill_archive.get_summary(),
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
        event_bus=None,
    ):
        super().__init__()

        self.bg = BasalGanglia(state_dim, n_actions)
        self.vta = VentralTegmentalArea(state_dim)
        self.snc = SubstantiaNigra()

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "motor_control",
                self._handle_motor_control,
                priority=0,
                name="basal_ganglia",
            )

    def _handle_motor_control(self, event) -> Dict:
        """Event-driven handler for motor_control events."""
        import torch as _torch
        state_tensor = event.data.get("state_tensor")
        next_state_tensor = event.data.get("next_state_tensor")
        if state_tensor is None:
            state_tensor = _torch.randn(1, 64)

        result = self(state_tensor)

        if next_state_tensor is not None:
            update_result = self.update(state_tensor, result["action"], 0.0, next_state_tensor)
            state = event.data.get("internal_state", {})
            state["bg_td_error"] = update_result.get("td_error", 0.0)
        else:
            state = event.data.get("internal_state", {})

        state["bg_habit_strength"] = result.get("habit_strength", 0.0)
        state["conscious_load"] = self.bg.conscious_load

        return result

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
    'ArchivedSkill',
    'SkillArchive',
    'StriatumInput',
    'IndirectPathway',
    'DirectPathway',
    'HyperdirectPathway',
    'BasalGanglia',
    'VentralTegmentalArea',
    'SubstantiaNigra',
    'BasalGangliaSystem',
    'create_basal_ganglia',
    # NAc Core
    'NAcState',
    'NAcCore',
]


# ══════════════════════════════════════════════════════
# 伏隔核核心 — Wanting/Liking分离 (Berridge & Robinson, 1998)
# ══════════════════════════════════════════════════════

@dataclass
class NAcState:
    """伏隔核(NAc)状态 — wanting vs liking分离。

    参考:
    - Berridge & Robinson (1998): incentive salience
    - 动机显著性(wanting) ≠ 享乐影响(liking)
    - 慢性药物使用: wanting↑↑, liking↓ → 成瘾核心
    """
    wanting: float = 0.5          # 动机显著性 (DA驱动)
    liking: float = 0.5           # 享乐影响 ( opioid/PFC编码)
    cue_reactivity: float = 0.0   # 线索反应性
    wanting_liking_separation: float = 0.0  # wanting - liking


class NAcCore:
    """伏隔核核心 — wanting/liking分离计算。

    生理机制:
    - NAc shell: opioid编码liking (享乐热点)
    - NAc core: DA编码wanting (动机显著性)
    - 敏感化: DA系统超敏 → wanting放大; 同时liking钝化
    """

    def __init__(
        self,
        baseline_wanting: float = 0.5,
        baseline_liking: float = 0.5,
    ):
        self.state = NAcState(
            wanting=baseline_wanting,
            liking=baseline_liking,
        )

    def step(
        self,
        dopamine_signal: float = 0.5,
        opioid_signal: float = 0.5,
        cue_strength: float = 0.0,
        sensitization_factor: float = 1.0,
        liking_decay: float = 1.0,
    ) -> NAcState:
        """更新NAc状态。

        Args:
            dopamine_signal: DA信号 (VTA→NAc)
            opioid_signal: opioid信号 (VP→NAc shell)
            cue_strength: 线索强度 (环境/条件刺激)
            sensitization_factor: 致敏化因子 (≥1.0, 慢性药物递增)
            liking_decay: 享乐衰减因子 (≤1.0, 慢性药物递减)

        Returns:
            NAcState with wanting/liking/separation
        """
        # Wanting: DA驱动 × 致敏化 × (1 + 线索放大)
        cue_amplification = 1.0 + cue_strength * 0.5
        self.state.wanting = float(np.clip(
            dopamine_signal * sensitization_factor * cue_amplification, 0.0, 1.0
        ))

        # Liking: opioid驱动 × 享乐衰减
        self.state.liking = float(np.clip(
            opioid_signal * liking_decay, 0.0, 1.0
        ))

        # 线索反应性
        self.state.cue_reactivity = float(np.clip(
            cue_strength * sensitization_factor * 0.3, 0.0, 1.0
        ))

        # Wanting-liking分离 (成瘾指标)
        self.state.wanting_liking_separation = self.state.wanting - self.state.liking

        return self.state