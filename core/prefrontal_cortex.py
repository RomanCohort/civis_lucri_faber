"""
前额叶皮质 - 执行功能中枢 (Prefrontal Cortex)

大脑发育最晚的区域（人类约25岁成熟），负责：
1. 成熟度系统 — 控制执行功能上限，模拟发育过程
2. 成本收益分析 — 多候选行动的即时回报/长期价值/风险/成本评估
3. 冲动抑制 — 审核来自其他脑区的冲动信号，门控放行或抑制
4. 长期规划 — 目标层级树 + 前瞻模拟 + 时间折扣
5. 工作记忆 — 7-slot Miller limit + 注意力门控写入

参考:
  - Miller & Cohen (2001) - PFC as an integrative hub
  - Casey et al. (2008) - Adolescent brain development & impulse control
  - Bechara et al. (1994) - Somatic marker hypothesis (Iowa gambling task)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque
import math


# ============ 数据类 ============

@dataclass
class CandidateEval:
    """单个候选行动的成本收益评估"""
    action: str
    immediate_reward: float = 0.0
    long_term_value: float = 0.0
    risk: float = 0.0
    effort_cost: float = 0.0
    total_score: float = 0.0


@dataclass
class GoalNode:
    """目标层级节点"""
    description: str
    priority: float = 0.5
    sub_goals: List = field(default_factory=list)
    deadline: Optional[int] = None
    progress: float = 0.0


@dataclass
class PlanStep:
    """规划步骤"""
    goal: str
    priority: float
    estimated_steps: int


# ============ 成熟度追踪器 ============

class MaturationTracker:
    """
    前额叶成熟度追踪

    模拟前额叶发育最晚的特征：
    - maturity 从 0.0 逐渐增长到 1.0
    - 控制抑制能力上限、规划深度、时间折扣
    - 未成熟：冲动难抑制、只看眼前、规划短视
    - 成熟：抑制力强、权衡长远、规划深远
    """

    def __init__(self, maturation_tau: float = 5000.0):
        self.tau = maturation_tau
        self.step_count = 0

    def advance(self, steps: int = 1):
        self.step_count += steps

    @property
    def maturity(self) -> float:
        """当前成熟度 [0, 1]"""
        return 1.0 - math.exp(-self.step_count / self.tau)

    @property
    def inhibition_capacity(self) -> float:
        """抑制能力上限 = maturity"""
        return self.maturity

    @property
    def planning_depth(self) -> int:
        """规划深度：1（不成熟）到 max_depth（成熟）"""
        return max(1, int(self.maturity * 5))

    @property
    def temporal_discount(self) -> float:
        """时间折扣率：高=只看眼前，低=重视未来"""
        return 0.9 - 0.6 * self.maturity  # 0.9(不成熟) → 0.3(成熟)

    @property
    def impulsivity_weight(self) -> float:
        """即时回报权重：高=冲动，低=理性"""
        return 0.8 - 0.6 * self.maturity  # 0.8(不成熟) → 0.2(成熟)

    def get_summary(self) -> Dict:
        return {
            'maturity': round(self.maturity, 4),
            'step_count': self.step_count,
            'inhibition_capacity': round(self.inhibition_capacity, 4),
            'planning_depth': self.planning_depth,
            'temporal_discount': round(self.temporal_discount, 4),
            'impulsivity_weight': round(self.impulsivity_weight, 4),
        }


# ============ 成本收益分析器 ============

class CostBenefitAnalyzer(nn.Module):
    """
    多维度成本收益分析

    对每个候选行动评估：
    - 即时回报 immediate_reward
    - 长期价值 long_term_value
    - 风险 risk
    - 执行成本 effort_cost

    综合得分受成熟度调制：
    - 未成熟 → 过度看重即时回报（青少年特征）
    - 成熟 → 平衡即时与长期
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.immediate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Tanh()
        )
        self.longterm_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Tanh()
        )
        self.risk_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.cost_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def evaluate(self, state: torch.Tensor, maturity: float,
                 candidates: Optional[List[str]] = None) -> List[CandidateEval]:
        """
        评估候选行动

        如果提供了 candidates 列表，对每个候选独立评估。
        否则生成一个综合评估。
        """
        if candidates is None:
            candidates = ["default"]

        results = []
        for action in candidates:
            imm = self.immediate_net(state).squeeze().item()
            ltv = self.longterm_net(state).squeeze().item()
            rsk = self.risk_net(state).squeeze().item()
            cst = self.cost_net(state).squeeze().item()

            # 综合得分：成熟度调制即时 vs 长期权重
            imp_w = 0.8 - 0.6 * maturity  # 不成熟：0.8
            lt_w = 1.0 - imp_w             # 成熟：0.8
            score = (imp_w * imm + lt_w * ltv) - 0.3 * rsk - 0.2 * cst

            results.append(CandidateEval(
                action=action,
                immediate_reward=round(imm, 4),
                long_term_value=round(ltv, 4),
                risk=round(rsk, 4),
                effort_cost=round(cst, 4),
                total_score=round(score, 4),
            ))
        return results


# ============ 冲动抑制器 ============

class ImpulseController(nn.Module):
    """
    冲动抑制控制

    接收其他脑区的冲动信号，生成抑制门控：
    - gate ∈ [0, 1]，0=完全放行冲动，1=完全抑制
    - 抑制上限 = maturity * net_output
    - 冲动累积：连续冲动若未完全抑制会累积压力

    模拟生物学：
    - 青少年 PFC 未成熟 → 难以抑制杏仁核冲动
    - 成年 PFC 成熟 → 有效门控冲动反应
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.inhibition_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.accumulated_impulse = 0.0
        self.impulse_decay = 0.9
        self.burst_threshold = 0.8

    def gate(self, state: torch.Tensor, maturity: float,
             impulse_signals: Optional[Dict[str, float]] = None) -> Dict:
        """
        计算抑制门控

        Args:
            state: 当前状态
            maturity: 成熟度 [0, 1]
            impulse_signals: 各脑区冲动强度 {"amygdala": 0.8, "habit": 0.5, ...}

        Returns:
            gate: 抑制门控 [0, 1]
            burst: 是否冲动爆发
            accumulated: 累积冲动水平
        """
        raw_inhibition = self.inhibition_net(state).squeeze().item()

        # 抑制能力受成熟度限制
        effective_inhibition = raw_inhibition * maturity

        # 处理冲动输入
        total_impulse = 0.0
        if impulse_signals:
            for source, strength in impulse_signals.items():
                total_impulse += strength
            total_impulse /= max(1, len(impulse_signals))

        # 累积冲动
        self.accumulated_impulse = (
            self.accumulated_impulse * self.impulse_decay + total_impulse * (1 - self.impulse_decay)
        )

        # 冲动爆发判定
        burst = self.accumulated_impulse > self.burst_threshold

        # 最终门控：抑制 vs 冲动
        if burst:
            gate_value = 0.0  # 冲动爆发，完全放行
        else:
            gate_value = effective_inhibition

        return {
            'gate': round(gate_value, 4),
            'burst': burst,
            'accumulated_impulse': round(self.accumulated_impulse, 4),
            'raw_inhibition': round(raw_inhibition, 4),
            'effective_inhibition': round(effective_inhibition, 4),
        }


# ============ 长期规划器 ============

class LongTermPlanner(nn.Module):
    """
    长期规划器

    维护目标层级树，支持前瞻模拟和时间折扣。
    规划深度受成熟度控制。

    - 未成熟：规划深度 1-2 步，高时间折扣
    - 成熟：规划深度 3-5 步，低时间折扣
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, max_depth: int = 5):
        super().__init__()
        self.max_depth = max_depth
        # 前瞻模拟网络：给定当前状态 + 行动，预测下一状态价值
        self.forward_model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Tanh()
        )
        self.priority_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

        # 目标层级树
        self.goals: List[GoalNode] = []
        self.plan_history: deque = deque(maxlen=100)

    def add_goal(self, description: str, priority: float = 0.5,
                 sub_goals: Optional[List[GoalNode]] = None,
                 deadline: Optional[int] = None):
        """添加目标到层级树"""
        node = GoalNode(
            description=description,
            priority=priority,
            sub_goals=sub_goals or [],
            deadline=deadline,
        )
        self.goals.append(node)

    def plan(self, state: torch.Tensor, maturity: float,
             n_candidates: int = 4) -> Dict:
        """
        生成规划

        前瞻模拟：从当前状态出发，模拟多步行动，计算累积折扣回报。
        规划深度 = int(maturity * max_depth)

        Returns:
            depth: 实际规划深度
            steps: 规划步骤列表
            cumulative_value: 累积折扣价值
            top_goal: 最优先的当前目标
        """
        depth = max(1, int(maturity * self.max_depth))
        discount = 0.9 - 0.6 * maturity  # 时间折扣

        steps = []
        cumulative_value = 0.0
        current_gamma = 1.0
        current_state = state

        for d in range(depth):
            # 为每一步选择最优行动
            best_action = 0
            best_value = -float('inf')

            for a in range(n_candidates):
                action_tensor = torch.tensor([float(a) / n_candidates])
                inp = torch.cat([current_state.squeeze(0), action_tensor])
                value = self.forward_model(inp).item()
                if value > best_value:
                    best_value = value
                    best_action = a

            cumulative_value += current_gamma * best_value
            current_gamma *= discount

            steps.append(PlanStep(
                goal=f"step_{d}",
                priority=current_gamma,
                estimated_steps=depth - d,
            ))

            # 简化：用当前状态继续（实际应用 forward_model 更新状态）
            current_state = current_state

        # 获取最优先目标
        top_goal = None
        if self.goals:
            self.goals.sort(key=lambda g: g.priority, reverse=True)
            top_goal = self.goals[0].description

        self.plan_history.append({
            'depth': depth,
            'value': round(cumulative_value, 4),
            'steps': len(steps),
        })

        return {
            'depth': depth,
            'steps': steps,
            'cumulative_value': round(cumulative_value, 4),
            'top_goal': top_goal,
            'n_goals': len(self.goals),
            'temporal_discount': round(discount, 4),
        }

    def update_goal_progress(self, goal_desc: str, progress: float):
        """更新目标进度"""
        for g in self.goals:
            if g.description == goal_desc:
                g.progress = min(1.0, g.progress + progress)
                break


# ============ 工作记忆 ============

class WorkingMemory(nn.Module):
    """
    工作记忆系统（7-slot Miller limit）

    改进：
    - 可微分的注意力读取（替代 Python list）
    - 门控写入：只有 PFC 判断为重要的信息才写入
    """

    def __init__(self, input_dim: int, n_slots: int = 7):
        super().__init__()
        self.n_slots = n_slots
        self.input_dim = input_dim

        # 记忆槽：固定大小的张量
        self.register_buffer('memory', torch.zeros(n_slots, input_dim))
        self.register_buffer('occupancy', torch.zeros(n_slots))

        # 门控写入网络
        self.write_gate = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        # 注意力读取
        self.read_attention = nn.Linear(input_dim, n_slots)

        self.write_ptr = 0

    def write(self, x: torch.Tensor):
        """
        门控写入

        只有 write_gate > 0.5 的信息才写入工作记忆
        """
        importance = self.write_gate(x).squeeze().item()
        if importance > 0.5:
            idx = self.write_ptr % self.n_slots
            self.memory[idx] = x.squeeze(0).detach()
            self.occupancy[idx] = 1.0
            self.write_ptr += 1

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """注意力加权读取"""
        if self.occupancy.sum() < 0.5:
            return torch.zeros_like(query)

        scores = self.read_attention(query)  # [1, n_slots]
        scores = scores + (self.occupancy.unsqueeze(0) - 1.0) * 100  # mask empty
        weights = F.softmax(scores, dim=-1)  # [1, n_slots]
        context = (weights @ self.memory).unsqueeze(0)  # [1, 1, input_dim] → squeeze
        return context.squeeze(0)

    def clear(self):
        self.memory.zero_()
        self.occupancy.zero_()
        self.write_ptr = 0

    @property
    def used_slots(self) -> int:
        return int((self.occupancy > 0.5).sum().item())


# ============ 主类：前额叶皮质 ============

class PrefrontalCortex(nn.Module):
    """
    前额叶皮质 — 执行功能中枢

    大脑发育最晚的区域（~25岁成熟），负责：
    1. 成熟度系统 — 控制执行功能上限
    2. 成本收益分析 — 多维度权衡利弊
    3. 冲动抑制 — 门控其他脑区的冲动信号
    4. 长期规划 — 目标层级 + 前瞻模拟
    5. 工作记忆 — 7-slot + 门控写入

    Usage:
        pfc = PrefrontalCortex(input_dim=64)

        # 每步调用
        result = pfc(
            state=state_tensor,
            impulse_signals={"amygdala": 0.7, "habit": 0.3},
            emotion_valence=-0.5,
            dopamine_level=0.6,
        )

        if result['inhibition_gate'] > 0.7:
            # PFC 抑制了冲动反应
            final_action = result['action']
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_actions: int = 4,
        maturation_tau: float = 5000.0,
        wm_slots: int = 7,
        event_bus=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # 子系统
        self.maturation = MaturationTracker(maturation_tau)
        self.cost_benefit = CostBenefitAnalyzer(input_dim, hidden_dim // 2)
        self.impulse_ctrl = ImpulseController(input_dim, hidden_dim // 2)
        self.planner = LongTermPlanner(input_dim, hidden_dim // 2)
        self.working_memory = WorkingMemory(input_dim, wm_slots)

        # 决策网络（最终行动选择）
        self.decision_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 状态融合：当前输入 + 工作记忆上下文
        self.context_blend = nn.Linear(input_dim * 2, input_dim)

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "motor_control",
                self._handle_motor_control,
                priority=1,
                name="prefrontal",
            )

    def _handle_motor_control(self, event) -> Dict:
        """Event-driven handler for motor_control events."""
        import torch as _torch
        state = event.data.get("internal_state", {})

        # Get state tensor from BG result or generate default
        state_tensor = event.data.get("state_tensor")
        if state_tensor is None:
            state_tensor = _torch.randn(1, self.input_dim)

        impulse_signals = event.data.get("impulse_signals")
        emotion_valence = state.get("limbic_valence", 0.0)
        dopamine_level = state.get("dopamine_level", 0.5)

        result = self(
            state=state_tensor,
            impulse_signals=impulse_signals,
            emotion_valence=emotion_valence,
            dopamine_level=dopamine_level,
        )

        state["pfc_inhibition"] = result["inhibition_gate"]
        state["pfc_maturity"] = result["maturity"]
        state["pfc_plan_depth"] = result["planning_depth"]

        # Check if PFC overrode BG action
        bg_action = event.data.get("bg_action")
        if bg_action is not None:
            pfc_action = result["action"].item() if hasattr(result["action"], "item") else result["action"]
            state["pfc_overrode_bg"] = (pfc_action != bg_action)

        return result

    def forward(
        self,
        state: torch.Tensor,
        candidates: Optional[List[str]] = None,
        impulse_signals: Optional[Dict[str, float]] = None,
        emotion_valence: float = 0.0,
        dopamine_level: float = 0.5,
    ) -> Dict:
        """
        执行功能中枢前向传播

        Args:
            state: [B, input_dim] 当前状态表征
            candidates: 候选行动名称列表
            impulse_signals: 来自其他脑区的冲动 {"amygdala": 0.8, "basal_ganglia": 0.5, ...}
            emotion_valence: 情绪效价 [-1, 1]
            dopamine_level: 多巴胺水平 [0, 1]

        Returns:
            action: 选择的行动
            action_logits: 行动 logits
            value: 状态价值
            inhibition_gate: 抑制门控 [0, 1]
            cost_benefit: 各候选的成本收益分析
            plan: 规划结果
            maturity: 当前成熟度
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # 推进成熟度
        self.maturation.advance()
        maturity = self.maturation.maturity

        # 1. 工作记忆读取 + 状态融合
        wm_context = self.working_memory.read(state)
        if wm_context.dim() == 1:
            wm_context = wm_context.unsqueeze(0)
        blended = torch.cat([state, wm_context], dim=-1)
        effective_input = torch.sigmoid(self.context_blend(blended))

        # 2. 成本收益分析
        cb_results = self.cost_benefit.evaluate(state, maturity, candidates)

        # 3. 冲动抑制
        inhibition_result = self.impulse_ctrl.gate(state, maturity, impulse_signals)

        # 4. 长期规划
        plan_result = self.planner.plan(state, maturity, self.num_actions)

        # 5. 决策
        action_logits = self.decision_net(effective_input)
        action = action_logits.argmax(dim=-1)

        # 6. 价值评估
        value = self.value_net(effective_input).squeeze(-1)

        # 7. 工作记忆写入
        self.working_memory.write(state)

        return {
            'action': action,
            'action_logits': action_logits,
            'value': value,
            'inhibition_gate': inhibition_result['gate'],
            'inhibition_burst': inhibition_result['burst'],
            'accumulated_impulse': inhibition_result['accumulated_impulse'],
            'cost_benefit': cb_results,
            'plan': plan_result,
            'maturity': maturity,
            'planning_depth': plan_result['depth'],
            'effective_input': effective_input,
        }

    def get_decision_explanation(self, action_id: int) -> str:
        """行动解释"""
        actions = {
            0: "explore (探索)",
            1: "exploit (利用)",
            2: "wait (等待)",
            3: "retreat (撤退)",
        }
        return actions.get(action_id, "unknown")

    def get_summary(self) -> Dict:
        """获取系统状态摘要"""
        return {
            'maturation': self.maturation.get_summary(),
            'working_memory': {
                'used_slots': self.working_memory.used_slots,
                'total_slots': self.working_memory.n_slots,
            },
            'planner': {
                'n_goals': len(self.planner.goals),
                'plan_history': len(self.planner.plan_history),
            },
            'impulse_ctrl': {
                'accumulated_impulse': round(self.impulse_ctrl.accumulated_impulse, 4),
            },
        }


__all__ = [
    'PrefrontalCortex',
    'MaturationTracker',
    'CostBenefitAnalyzer',
    'ImpulseController',
    'LongTermPlanner',
    'WorkingMemory',
    'CandidateEval',
    'GoalNode',
    'PlanStep',
]
