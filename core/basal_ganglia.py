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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Set, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Delayed import to avoid circular dependency
try:
    from core.abstract_brain_region import AbstractBrainRegion
except ImportError:
    from abstract_brain_region import AbstractBrainRegion

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
        self.skills: list[ArchivedSkill] = []
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

    def find_skill(self, context: np.ndarray, action: int) -> ArchivedSkill | None:
        """查找匹配的已存档技能"""
        context_hash = self._hash_context(context)
        for skill in self.skills:
            if skill.action == action and skill.context_hash == context_hash:
                return skill
        return None

    def has_skill(self, context: np.ndarray) -> ArchivedSkill | None:
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

    def get_summary(self) -> dict:
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


# ══════════════════════════════════════════════════════
# D1/D2 纹状体神经元群体 — DA调制通路分离
# ══════════════════════════════════════════════════════

class StriatalD1Population(nn.Module):
    """
    纹状体D1中型多棘神经元 (Direct/Go pathway)

    生物学机制:
    - D1受体: DA兴奋D1 MSNs → 增强放电
    - 输出: 抑制GPi/SNr → 解除对丘脑抑制 → 促进动作

    参考: Gerfen & Surmeier (2011)
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.n_actions = n_actions

        # D1 MSN权重矩阵 (皮层→D1纹状体)
        self.cortex_to_d1 = nn.Linear(state_dim, hidden_dim)
        self.d1_to_gpi = nn.Linear(hidden_dim, n_actions, bias=False)

        # 初始化权重 (GABAergic输出为抑制性, 但我们用正值表示抑制强度)
        nn.init.xavier_uniform_(self.cortex_to_d1.weight)
        nn.init.xavier_uniform_(self.d1_to_gpi.weight)

        # 基线放电率
        self.baseline_firing = 0.1

    def forward(
        self,
        cortical_input: torch.Tensor,
        dopamine_level: float = 0.5,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            cortical_input: 皮层输入 [batch, state_dim]
            dopamine_level: 多巴胺水平 [0, 1]

        Returns:
            d1_output: D1输出到GPi [batch, n_actions]
        """
        # D1受体: DA兴奋 (D1+DA → cAMP↑ → 兴奋性增强)
        # 调制因子: 低DA时基线活动, 高DA时增强
        da_excitation = 0.5 + dopamine_level  # DA=0.5时为1.0 (基线)

        # 皮层输入激活D1 MSNs
        d1_hidden = F.relu(self.cortex_to_d1(cortical_input))

        # D1输出 (GABAergic抑制信号)
        d1_output = torch.sigmoid(self.d1_to_gpi(d1_hidden)) * da_excitation

        return d1_output


class StriatalD2Population(nn.Module):
    """
    纹状体D2中型多棘神经元 (Indirect/NoGo pathway)

    生物学机制:
    - D2受体: DA抑制D2 MSNs → 降低放电
    - 输出: 抑制GPe → GPe去抑制 → STN兴奋 → GPi兴奋 → 抑制动作

    参考: Gerfen & Surmeier (2011)
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.n_actions = n_actions

        # D2 MSN权重矩阵 (皮层→D2纹状体)
        self.cortex_to_d2 = nn.Linear(state_dim, hidden_dim)
        self.d2_to_gpe = nn.Linear(hidden_dim, n_actions, bias=False)

        # 初始化权重
        nn.init.xavier_uniform_(self.cortex_to_d2.weight)
        nn.init.xavier_uniform_(self.d2_to_gpe.weight)

        # 基线放电率
        self.baseline_firing = 0.1

    def forward(
        self,
        cortical_input: torch.Tensor,
        dopamine_level: float = 0.5,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            cortical_input: 皮层输入 [batch, state_dim]
            dopamine_level: 多巴胺水平 [0, 1]

        Returns:
            d2_output: D2输出到GPe [batch, n_actions]
        """
        # D2受体: DA抑制 (D2+DA → cAMP↓ → 抑制性增强)
        # 调制因子: 高DA时降低D2活动, 低DA时增强
        da_inhibition = 1.5 - dopamine_level  # DA=0.5时为1.0 (基线)

        # 皮层输入激活D2 MSNs
        d2_hidden = F.relu(self.cortex_to_d2(cortical_input))

        # D2输出 (GABAergic抑制信号到GPe)
        d2_output = torch.sigmoid(self.d2_to_gpe(d2_hidden)) * da_inhibition

        return d2_output


class GlobusPallidusExternal(nn.Module):
    """
    外侧苍白球 (GPe) — 间接通路中继

    生物学机制:
    - 接收D2纹状体的GABAergic抑制
    - 输出GABAergic抑制到STN
    - D2→GPe抑制 → GPe去抑制 → STN兴奋

    关键: GPe是"抑制的抑制 = 兴奋"
    """

    def __init__(
        self,
        n_actions: int = 4,
        hidden_dim: int = 16,
    ):
        super().__init__()

        self.n_actions = n_actions

        # GPe内部处理 (来自D2的抑制信号)
        self.d2_integration = nn.Linear(n_actions, hidden_dim)
        self.gpe_to_stn = nn.Linear(hidden_dim, n_actions, bias=False)

        # GPe基线放电率 (高自发活动)
        self.baseline_firing = 0.8

        # 初始化
        nn.init.xavier_uniform_(self.d2_integration.weight)
        nn.init.xavier_uniform_(self.gpe_to_stn.weight)

    def forward(self, d2_input: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            d2_input: D2纹状体的GABAergic输出 [batch, n_actions]

        Returns:
            gpe_output: GPe输出到STN [batch, n_actions]
        """
        # D2抑制GPe → GPe活动降低
        # (d2_input是抑制强度, 所以GPe输出 = baseline - inhibition)
        gpe_hidden = F.relu(self.d2_integration(self.baseline_firing - d2_input))

        # GPe输出 (抑制STN)
        gpe_output = torch.sigmoid(self.gpe_to_stn(gpe_hidden))

        return gpe_output


class SubthalamicNucleus(nn.Module):
    """
    丘脑底核 (STN) — 间接/超通路的汇聚点

    生物学机制:
    - 接收GPe的GABAergic抑制 (间接通路)
    - 接收皮层直接兴奋 (超通路)
    - 输出谷氨酸能兴奋到GPi/SNr

    关键角色:
    - 间接通路: D2→GPe→STN→GPi (抑制动作)
    - 超通路: 皮层→STN→GPi (全局STOP)
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
        hidden_dim: int = 16,
    ):
        super().__init__()

        self.n_actions = n_actions
        self.state_dim = state_dim

        # 超通路: 皮层→STN (直接兴奋性输入)
        self.cortex_to_stn = nn.Linear(state_dim, hidden_dim)

        # 间接通路: GPe→STN (抑制性输入整合)
        self.gpe_to_stn = nn.Linear(n_actions, hidden_dim)

        # STN输出到GPi
        self.stn_to_gpi = nn.Linear(hidden_dim, n_actions, bias=False)

        # STN基线放电率
        self.baseline_firing = 0.3

        # 初始化
        nn.init.xavier_uniform_(self.cortex_to_stn.weight)
        nn.init.xavier_uniform_(self.gpe_to_stn.weight)
        nn.init.xavier_uniform_(self.stn_to_gpi.weight)

    def forward(
        self,
        cortical_input: torch.Tensor,
        gpe_input: torch.Tensor,
        hyperdirect_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            cortical_input: 皮层输入 (超通路) [batch, state_dim]
            gpe_input: GPe输入 (间接通路) [batch, n_actions]
            hyperdirect_strength: 超通路强度权重

        Returns:
            stn_output: STN输出到GPi [batch, n_actions]
        """
        # 超通路: 皮层直接兴奋STN
        hyperdirect_activation = F.relu(self.cortex_to_stn(cortical_input)) * hyperdirect_strength

        # 间接通路: GPe抑制STN (gpe_input是抑制强度)
        # GPe高活动 → STN被抑制; GPe低活动 → STN去抑制
        indirect_modulation = self.baseline_firing - gpe_input

        # 综合STN活动: 超通路兴奋 + 间接通路调制
        stn_hidden = hyperdirect_activation + F.relu(self.gpe_to_stn(indirect_modulation))

        # STN输出 (谷氨酸能兴奋信号到GPi)
        stn_output = torch.sigmoid(self.stn_to_gpi(stn_hidden))

        return stn_output


class GlobusPallidusInternal(nn.Module):
    """
    内侧苍白球/黑质网状部 (GPi/SNr) — 输出核

    生物学机制:
    - 接收D1纹状体的GABAergic抑制 (直接通路 → 解除丘脑抑制)
    - 接收STN的谷氨酸能兴奋 (间接/超通路 → 增强丘脑抑制)
    - 输出GABAergic抑制到丘脑

    关键:
    - GPi高活动 → 抑制丘脑 → 抑制动作
    - GPi低活动 → 解除丘脑抑制 → 允许动作
    """

    def __init__(
        self,
        n_actions: int = 4,
    ):
        super().__init__()

        self.n_actions = n_actions

        # GPi/SNr基线放电率 (高自发活动, 持续抑制丘脑)
        self.baseline_firing = 0.9

        # 直接通路权重 (D1→GPi抑制)
        self.direct_weight = nn.Parameter(torch.ones(n_actions))

        # 间接通路权重 (STN→GPi兴奋)
        self.indirect_weight = nn.Parameter(torch.ones(n_actions))

        # 超通路权重 (STN→GPi全局STOP)
        self.hyperdirect_weight = nn.Parameter(torch.ones(n_actions) * 1.5)

    def forward(
        self,
        d1_input: torch.Tensor,
        stn_input: torch.Tensor,
        hyperdirect_input: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            d1_input: D1纹状体抑制信号 [batch, n_actions]
            stn_input: STN兴奋信号 (间接通路) [batch, n_actions]
            hyperdirect_input: STN兴奋信号 (超通路) [batch, n_actions]

        Returns:
            gpi_output: GPi输出 (丘脑抑制强度) [batch, n_actions]
        """
        # 直接通路: D1抑制GPi → 降低GPi活动 → 允许动作
        # (D1高 → GPi低 → 丘脑去抑制 → 动作促进)
        direct_inhibition = d1_input * torch.sigmoid(self.direct_weight)

        # 间接通路: STN兴奋GPi → 增强GPi活动 → 抑制动作
        # (STN高 → GPi高 → 丘脑抑制 → 动作抑制)
        indirect_excitation = stn_input * torch.sigmoid(self.indirect_weight)

        # 超通路: 皮层→STN→GPi 全局STOP
        if hyperdirect_input is None:
            hyperdirect_input = stn_input  # 默认使用STN输入
        hyperdirect_excitation = hyperdirect_input * torch.sigmoid(self.hyperdirect_weight)

        # GPi输出 = 基线 - 直接通路抑制 + 间接/超通路兴奋
        # (竞争性整合: Go vs NoGo)
        gpi_output = torch.clamp(
            self.baseline_firing - direct_inhibition + indirect_excitation + hyperdirect_excitation,
            min=0.0,
            max=1.0,
        )

        return gpi_output


class IndirectPathway(nn.Module):
    """
    间接通路 (Striatum-D2 → GPe → STN → GPi)

    生物学机制:
    1. 皮层兴奋D2纹状体MSNs
    2. D2 MSNs抑制GPe (GABA)
    3. GPe抑制STN (GABA) — D2高→GPe低→STN高
    4. STN兴奋GPi (Glu) — STN高→GPi高
    5. GPi抑制丘脑 → 抑制动作

    结果: D2活动 → 动作抑制 (NoGo)

    多巴胺调制: DA抑制D2受体 → 降低间接通路 → 减少动作抑制
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.n_actions = n_actions
        self.state_dim = state_dim

        # D2纹状体群体
        self.striatum_d2 = StriatalD2Population(state_dim, n_actions, hidden_dim)

        # GPe (外侧苍白球)
        self.gpe = GlobusPallidusExternal(n_actions, hidden_dim // 2)

        # STN整合 (部分, 完整STN在超通路中)
        self.stn_integration = nn.Linear(n_actions, n_actions, bias=False)
        nn.init.xavier_uniform_(self.stn_integration.weight)

    def forward(
        self,
        cortical_input: torch.Tensor,
        dopamine_level: float = 0.5,
    ) -> torch.Tensor:
        """
        计算间接通路输出

        Args:
            cortical_input: 皮层输入 [batch, state_dim]
            dopamine_level: 多巴胺水平 [0, 1]

        Returns:
            indirect_output: 间接通路到GPi的兴奋信号 [batch, n_actions]
        """
        # D2纹状体 (DA抑制D2 → 降低活动)
        d2_output = self.striatum_d2(cortical_input, dopamine_level)

        # GPe (D2抑制GPe → GPe活动降低)
        gpe_output = self.gpe(d2_output)

        # STN被GPe抑制 → GPe低时STN去抑制兴奋
        # (STN输出 = 1 - GPe抑制)
        stn_output = torch.sigmoid(self.stn_integration(1.0 - gpe_output))

        return stn_output

    def compute_selection(
        self,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        兼容旧API: 计算动作选择

        竞争性 inhibition
        """
        # 归一化Q值
        q_norm = F.softmax(q_values, dim=-1)

        # 动作抑制网络
        inhibition = torch.sigmoid(self.stn_integration(q_norm))

        # 动作间的相对抑制
        selected = q_norm * (1 - inhibition * 0.5)

        return selected


class DirectPathway(nn.Module):
    """
    直接通路 (Striatum-D1 → GPi/SNr → Thalamus)

    生物学机制:
    1. 皮层兴奋D1纹状体MSNs
    2. D1 MSNs直接抑制GPi/SNr (GABA)
    3. GPi/SNr抑制丘脑 → D1高→GPi低→丘脑去抑制
    4. 丘脑兴奋 → 促进动作

    结果: D1活动 → 动作促进 (Go)

    多巴胺调制: DA兴奋D1受体 → 增强直接通路 → 促进动作
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.n_actions = n_actions
        self.state_dim = state_dim

        # D1纹状体群体
        self.striatum_d1 = StriatalD1Population(state_dim, n_actions, hidden_dim)

        # GPi输出整合
        self.gpi_integration = nn.Linear(n_actions, n_actions, bias=False)
        nn.init.xavier_uniform_(self.gpi_integration.weight)

        # 抑制强度 (默认)
        self.inhibition_strength = 0.8

    def forward(
        self,
        cortical_input: torch.Tensor,
        dopamine_level: float = 0.5,
    ) -> torch.Tensor:
        """
        计算直接通路输出

        Args:
            cortical_input: 皮层输入 [batch, state_dim]
            dopamine_level: 多巴胺水平 [0, 1]

        Returns:
            direct_output: 直接通路对GPi的抑制信号 [batch, n_actions]
        """
        # D1纹状体 (DA兴奋D1 → 增强活动)
        d1_output = self.striatum_d1(cortical_input, dopamine_level)

        # D1输出抑制GPi
        direct_output = torch.sigmoid(self.gpi_integration(d1_output))

        return direct_output

    def compute_go_signal(
        self,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        """兼容旧API: 计算GO信号"""
        return torch.sigmoid(self.gpi_integration(q_values))


class HyperdirectPathway(nn.Module):
    """
    超通路 (Hyperdirect: Cortex → STN → GPi/SNr)

    生物学机制:
    1. 皮层直接兴奋STN (绕过纹状体)
    2. STN兴奋GPi/SNr (谷氨酸)
    3. GPi/SNr全局抑制丘脑
    4. 丘脑活动下降 → 全局动作抑制

    结果: 超通路激活 → 全局STOP信号

    作用:
    - 快速停止所有动作 (紧急情况)
    - 动作切换时的抑制
    - 冲动控制

    参考: Nambu et al. (2002)
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
        hidden_dim: int = 16,
    ):
        super().__init__()

        self.n_actions = n_actions
        self.state_dim = state_dim

        # STN核 (皮层直接兴奋)
        self.stn = SubthalamicNucleus(state_dim, n_actions, hidden_dim)

        # 全局STOP信号强度
        self.stop_strength = nn.Parameter(torch.tensor(1.5))

        # 快速评估 (紧急情况检测)
        self.fast_eval = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(
        self,
        cortical_input: torch.Tensor,
        gpe_input: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        计算超通路输出

        Args:
            cortical_input: 皮层输入 [batch, state_dim]
            gpe_input: GPe输入 (可选, 来自间接通路) [batch, n_actions]

        Returns:
            hyperdirect_output: 超通路到GPi的全局STOP信号 [batch, n_actions]
        """
        # GPe输入默认为零 (超通路绕过间接通路)
        if gpe_input is None:
            gpe_input = torch.zeros(cortical_input.shape[0], self.n_actions)

        # STN处理皮层直接输入 (全局兴奋)
        stn_output = self.stn(
            cortical_input,
            gpe_input,
            hyperdirect_strength=torch.sigmoid(self.stop_strength)
        )

        # 超通路输出 = STN全局兴奋信号
        hyperdirect_output = stn_output * torch.sigmoid(self.stop_strength)

        return hyperdirect_output

    def compute_fast_inhibit(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """兼容旧API: 快速抑制"""
        return self.fast_eval(state)

    def compute_global_stop(
        self,
        cortical_input: torch.Tensor,
        urgency: float = 0.0,
    ) -> torch.Tensor:
        """
        计算全局STOP信号

        Args:
            cortical_input: 皮层输入
            urgency: 紧急程度 [0, 1]

        Returns:
            stop_signal: 全局STOP强度
        """
        # 紧急情况增强STOP信号
        stop_factor = 1.0 + urgency * 0.5

        stn_output = self.forward(cortical_input)

        return stn_output * stop_factor


@dataclass
class BGState:
    """基底神经节状态"""
    current_action: int = 0
    exploration_rate: float = 0.1
    habit_strength: float = 0.0
    last_reward: float = 0.0
    action_history: list = field(default_factory=list)


class BasalGanglia(AbstractBrainRegion):
    """
    完整基底神经节系统

    整合三条通路:
    1. 直接通路 (D1 MSN → GPi) - GO - 促进动作
    2. 间接通路 (D2 MSN → GPe → STN → GPi) - NOGO - 抑制动作
    3. 超通路 (Cortex → STN → GPi) - 全局STOP

    多巴胺调制:
    - DA兴奋D1受体 → 增强直接通路 → 促进动作
    - DA抑制D2受体 → 降低间接通路 → 减少动作抑制

    GPi/SNr输出:
    - 竞争性整合直接通路(解抑制)和间接/超通路(抑制)
    - 输出到丘脑, 控制动作执行
    """

    region_name: ClassVar[str] = "basal_ganglia"

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.hidden_dim = hidden_dim

        # ===== 核心通路组件 =====

        # D1/D2纹状体群体 (DA调制输入)
        self.striatum_d1 = StriatalD1Population(state_dim, n_actions, hidden_dim)
        self.striatum_d2 = StriatalD2Population(state_dim, n_actions, hidden_dim)

        # GPe (间接通路中继)
        self.gpe = GlobusPallidusExternal(n_actions, hidden_dim // 2)

        # STN (间接/超通路汇聚点)
        self.stn = SubthalamicNucleus(state_dim, n_actions, hidden_dim // 2)

        # GPi/SNr (输出核)
        self.gpi = GlobusPallidusInternal(n_actions)

        # ===== 三条通路 =====
        # (保留兼容旧API, 同时使用新组件)
        self.striatum = StriatumInput(state_dim, n_actions)  # Q值网络
        self.direct = DirectPathway(state_dim, n_actions, hidden_dim)
        self.indirect = IndirectPathway(state_dim, n_actions, hidden_dim)
        self.hyper = HyperdirectPathway(state_dim, n_actions, hidden_dim // 2)

        # ===== 多巴胺调节 =====
        self.dopamine_level = 0.5  # 中性
        self.eligibility_trace = {}
        self.trace_decay = 0.9

        # ===== 通路通路权重 (可学习) =====
        self.direct_weight = nn.Parameter(torch.tensor(0.8))
        self.indirect_weight = nn.Parameter(torch.tensor(0.6))
        self.hyperdirect_weight = nn.Parameter(torch.tensor(0.4))

        # ===== 习惯系统 =====
        self.habit_memory: list[HabitMemory] = []
        self.habit_threshold = 5  # 重复次数

        # 技能存档系统 (BG → 小脑转移)
        self.skill_archive = SkillArchive()
        self.conscious_load = 1.0  # 1.0=全部需意识控制, 0.0=全部自动化
        self._cerebellum = None   # 耦合的小脑引用
        self.archive_threshold = 8  # 频率超过此值触发存档

        # 状态
        self.state = BGState()
        self.training = True

        # 通路活动记录 (用于分析)
        self.pathway_activity = {
            'd1_activity': None,
            'd2_activity': None,
            'gpe_activity': None,
            'stn_activity': None,
            'gpi_activity': None,
            'direct_go': None,
            'indirect_nogo': None,
            'hyperdirect_stop': None,
        }

    def step(self, internal_state: dict[str, Any] = None, **kwargs) -> dict[str, Any]:
        """One simulation step — delegates to forward()."""
        if internal_state is None:
            internal_state = {}
        result = self.forward(**kwargs)
        return {
            "action": result.action,
            "q_values": result.q_values,
            "selected_q": result.selected_q,
            "action_probs": result.action_probs,
        }

    @classmethod
    def required_keys(cls) -> Set[str]:
        """Keys this region reads from the shared state."""
        return set(["state_tensor", "dopamine_level"])

    @classmethod
    def output_keys(cls) -> Set[str]:
        """Keys this region writes to the shared state."""
        return set(["bg_action", "bg_q_values", "bg_dopamine", "bg_td_error",
                    "bg_habit_strength", "bg_conscious_load"])

    def forward(
        self,
        state: torch.Tensor,
        epsilon: float = 0.1,
    ) -> ActionSelection:
        """
        动作选择 — 三通路竞争性整合

        生物学流程:
        1. 皮层输入同时激活D1/D2纹状体
        2. 多巴胺调制D1(兴奋)和D2(抑制)
        3. 直接通路(D1→GPi抑制)→促进动作
        4. 间接通路(D2→GPe→STN→GPi兴奋)→抑制动作
        5. 超通路(皮层→STN→GPi)→全局STOP
        6. GPi/SNr输出竞争性整合 → 丘脑去抑制程度
        7. 动作选择基于通路竞争结果

        Args:
            state: 状态输入 (皮层信号)
            epsilon: 探索率

        Returns:
            selection: 动作选择结果
        """
        # ===== Step 1: D1/D2纹状体激活 =====
        # 多巴胺调制: DA兴奋D1, 抑制D2
        d1_output = self.striatum_d1(state, self.dopamine_level)
        d2_output = self.striatum_d2(state, self.dopamine_level)

        self.pathway_activity['d1_activity'] = d1_output.detach()
        self.pathway_activity['d2_activity'] = d2_output.detach()

        # ===== Step 2: GPe处理 (间接通路中继) =====
        # D2抑制GPe → GPe去抑制
        gpe_output = self.gpe(d2_output)

        self.pathway_activity['gpe_activity'] = gpe_output.detach()

        # ===== Step 3: STN处理 (间接/超通路汇聚) =====
        # 超通路: 皮层直接兴奋STN
        # 间接通路: GPe抑制STN
        stn_indirect_output = self.stn(state, gpe_output, hyperdirect_strength=0.0)

        # 超通路单独计算 (全局STOP)
        stn_hyperdirect_output = self.stn(state, torch.zeros_like(gpe_output), hyperdirect_strength=1.0)

        self.pathway_activity['stn_activity'] = stn_indirect_output.detach()

        # ===== Step 4: GPi/SNr输出竞争 =====
        # 直接通路: D1→GPi抑制 → 降低GPi活动 → 允许动作
        direct_go = torch.sigmoid(self.direct_weight) * d1_output

        # 间接通路: STN→GPi兴奋 → 增强GPi活动 → 抑制动作
        indirect_nogo = torch.sigmoid(self.indirect_weight) * stn_indirect_output

        # 超通路: 皮层→STN→GPi全局STOP
        hyperdirect_stop = torch.sigmoid(self.hyperdirect_weight) * stn_hyperdirect_output

        self.pathway_activity['direct_go'] = direct_go.detach()
        self.pathway_activity['indirect_nogo'] = indirect_nogo.detach()
        self.pathway_activity['hyperdirect_stop'] = hyperdirect_stop.detach()

        # GPi/SNr输出 = 基线 - 直接通路抑制 + 间接/超通路兴奋
        # 高GPi → 抑制丘脑 → 抑制动作
        # 低GPi → 丘脑去抑制 → 促进动作
        gpi_output = self.gpi(direct_go, indirect_nogo, hyperdirect_stop)

        self.pathway_activity['gpi_activity'] = gpi_output.detach()

        # ===== Step 5: 动作选择 =====
        # GPi抑制丘脑 → 丘脑活动 = 1 - GPi抑制
        thalamus_activity = 1.0 - gpi_output

        # 转换为动作概率 (丘脑活动越高 → 动作越可能)
        action_probs = F.softmax(thalamus_activity, dim=-1)

        # 检查习惯 (概率性混合)
        habit_action = self._check_habit(state)
        q_values = self.striatum(state)  # 保留Q值计算 (兼容)

        if habit_action is not None and self.state.habit_strength > 0.1:
            habit_weight = self.state.habit_strength
            action_probs = action_probs * (1.0 - habit_weight)
            action_probs[0, habit_action] += habit_weight
            action_probs = F.softmax(action_probs, dim=-1)  # renormalize

        # 多巴胺调制探索温度
        da_temperature = float(np.clip(0.5 + self.dopamine_level, 0.3, 2.0))
        action_probs = F.softmax(action_probs / da_temperature, dim=-1)

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
            q_values=q_values.detach().cpu().numpy()[0],
            selected_q=selected_q,
            action_probs=action_probs.detach().cpu().numpy()[0],
        )

    def compute_pathway_output(
        self,
        state: torch.Tensor,
        dopamine_level: float = None,
    ) -> dict:
        """
        计算各通路输出 — 用于分析和调试

        Returns:
            pathway_output: 各通路活动详情
        """
        if dopamine_level is not None:
            self.dopamine_level = dopamine_level

        # 触发forward计算通路活动
        _ = self.forward(state)

        return {
            'd1_striatal_output': self.pathway_activity['d1_activity'],
            'd2_striatal_output': self.pathway_activity['d2_activity'],
            'gpe_output': self.pathway_activity['gpe_activity'],
            'stn_output': self.pathway_activity['stn_activity'],
            'gpi_output': self.pathway_activity['gpi_activity'],
            'direct_go_signal': self.pathway_activity['direct_go'],
            'indirect_nogo_signal': self.pathway_activity['indirect_nogo'],
            'hyperdirect_stop_signal': self.pathway_activity['hyperdirect_stop'],
            'dopamine_level': self.dopamine_level,
            'pathway_weights': {
                'direct': torch.sigmoid(self.direct_weight).item(),
                'indirect': torch.sigmoid(self.indirect_weight).item(),
                'hyperdirect': torch.sigmoid(self.hyperdirect_weight).item(),
            },
        }

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

    def _check_habit(self, state: torch.Tensor) -> int | None:
        """检查是否形成习惯 (含技能存档检查)"""
        state_np = state.detach().cpu().numpy()[0]

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
    ) -> dict:
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

        # 更新 eligibility trace (Sutton & Barto §12.2 replacing trace)
        # 1. Decay all traces by gamma * trace_decay
        for a in list(self.eligibility_trace.keys()):
            self.eligibility_trace[a] *= self.gamma * self.trace_decay
        # 2. Set selected action trace to 1.0 (replacing, not accumulating)
        self.eligibility_trace[action] = 1.0

        # 3. Trace-weighted Q-value parameter update (Sutton & Barto §12.2)
        with torch.no_grad():
            for a, trace_val in self.eligibility_trace.items():
                if a < self.n_actions and trace_val > 0.01:
                    q_val = self.striatum.get_q_value(state, a)
                    q_param = self.striatum.action_values[a][-1].weight
                    update = self.lr * trace_val * td_error
                    q_param.data += update * 0.01  # scaled to prevent blow-up

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
        state_np = state.detach().cpu().numpy()[0]

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

    def get_summary(self) -> dict:
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

    def _handle_motor_control(self, event) -> dict:
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
    ) -> dict:
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
    ) -> dict:
        """更新"""
        return self.bg.update(state, action, reward, next_state)

    def learn_habit(
        self,
        context: torch.Tensor,
        action: int,
    ):
        """强制形成习惯"""
        self.bg._update_habit_memory(context, action)

    def get_summary(self) -> dict:
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
    # D1/D2纹状体群体 (新增)
    'StriatalD1Population',
    'StriatalD2Population',
    # GPe/STN/GPi核团 (新增)
    'GlobusPallidusExternal',
    'SubthalamicNucleus',
    'GlobusPallidusInternal',
    # 三通路
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
