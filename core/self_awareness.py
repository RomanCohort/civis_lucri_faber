"""
自我意识中枢 (Self-Awareness Center)

位于大脑内侧前额叶皮层 (mPFC) 和后扣带回皮层 (PCC)，
是"我是谁"这个问题的神经基础。

核心子系统:
1. 内侧前额叶 (mPFC) - 自我参照处理、自我评价、心理时间旅行
2. 后扣带回 (PCC) - 自我相关性检测、自传体记忆整合、自我叙事连续性
3. 楔前叶 (Precuneus) - 自我加工、第一人称视角、心理意象
4. 默认模式网络集成 (DMN Integration) - 任务负向网络、静息态自省
5. 自我-他者区分 (Self-Other Distinction) - 自我边界维持
6. 元自我意识 (Meta-Self-Awareness) - 对"意识到自己在意识"的递归建模

生物参考文献:
- Northoff et al. (2006): 自我参照处理的神经基础 (mPFC)
- Raichle et al. (2001): 默认模式网络的发现
- Cavanna & Trimble (2006): 楔前叶与意识
- Andrews-Hanna (2010): DMN的默认意识
- D'Argembeau et al. (2005): 自传体记忆与mPFC
- Legrand & Ruby (2009): 自我意识与自我-他者区分
- Christoff et al. (2011): DMN与元认知
- Schooler et al. (2011): 心智游移与元意识
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque


# ============ 状态定义 ============

@dataclass
class mPFCState:
    """内侧前额叶状态"""
    self_evaluation: float = 0.5          # 自我评价 [0,1]
    self_referential_activation: float = 0.3  # 自我参照激活 [0,1]
    autobiographical_coherence: float = 0.5  # 自传体连贯性 [0,1]
    mental_time_travel_depth: float = 0.0  # 心理时间旅行深度 [0,1]
    self_endorsement: float = 0.5          # 自我认同度 [0,1]


@dataclass
class PCCState:
    """后扣带回状态"""
    self_relevance: float = 0.5            # 自我相关性 [0,1]
    narrative_continuity: float = 0.7      # 叙事连续性 [0,1]
    autobiographical_recall: float = 0.3   # 自传体回忆 [0,1]
    scene_construction: float = 0.3        # 场景构建 [0,1]
    self_model_confidence: float = 0.5     # 自我模型置信度 [0,1]


@dataclass
class PrecuneusState:
    """楔前叶状态"""
    first_person_perspective: float = 0.7  # 第一人称视角强度 [0,1]
    self_processing: float = 0.4           # 自我加工激活 [0,1]
    mental_imagery_vividness: float = 0.5  # 心理意象生动性 [0,1]
    self_location: float = 0.5             # 自我定位 (自我在世界中的位置) [0,1]


@dataclass
class DMNState:
    """默认模式网络状态"""
    dmn_activation: float = 0.5            # DMN总激活 [0,1]
    task_negative_dominance: float = 0.3   # 任务负向优势 (>0.5 = 内省模式)
    mind_wandering: float = 0.2            # 心智游移程度 [0,1]
    spontaneous_thought_rate: float = 0.3  # 自发思维速率 [0,1]
    meta_awareness: float = 0.4            # 元意识 (意识到自己在想什么) [0,1]


@dataclass
class SelfOtherState:
    """自我-他者区分状态"""
    self_boundary_clarity: float = 0.7     # 自我边界清晰度 [0,1]
    self_other_overlap: float = 0.2        # 自我-他者重叠 [0,1]
    agency_sense: float = 0.8              # 主体感 (我vs他) [0,1]
    ownership_sense: float = 0.8           # 所有权感 (我的vs他的) [0,1]


@dataclass
class MetaSelfState:
    """元自我意识状态"""
    awareness_of_awareness: float = 0.3    # 对意识的意识 [0,1]
    recursive_depth: int = 0               # 递归深度 (0=无意识, 1=意识到, 2=意识到自己在意识...)
    self_model_accuracy: float = 0.5       # 自我模型准确度 [0,1]
    introspection_depth: float = 0.3       # 内省深度 [0,1]


@dataclass
class SelfAwarenessState:
    """自我意识总状态"""
    mpfc: mPFCState = field(default_factory=mPFCState)
    pcc: PCCState = field(default_factory=PCCState)
    precuneus: PrecuneusState = field(default_factory=PrecuneusState)
    dmn: DMNState = field(default_factory=DMNState)
    self_other: SelfOtherState = field(default_factory=SelfOtherState)
    meta_self: MetaSelfState = field(default_factory=MetaSelfState)
    self_narrative: str = ""               # 当前自我叙事
    self_coherence: float = 0.7            # 自我一致性 [0,1]
    overall_self_awareness: float = 0.5     # 总体自我意识水平 [0,1]


# ============ 内侧前额叶 (mPFC) ============

class MedialPrefrontalCortex(nn.Module):
    """
    内侧前额叶皮层 (medial Prefrontal Cortex)

    自我参照处理的核心脑区:
    - 自我评价: 对自身能力/状态/价值的评估
    - 自我参照: 处理与自我相关的信息 (vs 与他人相关的信息)
    - 心理时间旅行: 想象过去和未来的自己
    - 自传体自我: 维持"我是谁"的连贯叙事

    神经基础:
    - 腹侧mPFC (vmPFC): 情绪相关的自我评价、自我价值
    - 背侧mPFC (dmPFC): 认知自我反思、心理状态推理
    - 前mPFC (amPFC): 现在自我的表征

    参考:
    - Northoff et al. (2006): mPFC在自我参照处理中的核心地位
    - D'Argembeau et al. (2005): 自传体记忆与mPFC
    - Schneider et al. (2008): 自我评价的mPFC特异性
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 64):
        super().__init__()

        # 自我参照网络 (ventral mPFC)
        # 判断输入信息与自我的相关程度
        self.self_reference_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 自我评价网络 (dorsal mPFC)
        # 评估自身状态/能力/表现
        self.self_eval_net = nn.Sequential(
            nn.Linear(state_dim + state_dim, hidden_dim),  # [当前状态 || 理想自我]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 心理时间旅行网络 (anterior mPFC + 海马)
        # 想象过去/未来的自己
        self.time_travel_net = nn.GRUCell(state_dim, hidden_dim)

        # 自传体自我模型 (持续更新的"我是谁")
        self.autobiographical_self = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.ideal_self = nn.Parameter(torch.randn(state_dim) * 0.1 + 0.5)

        # 自我认同网络
        self.endorsement_net = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.state = mPFCState()
        self.hidden = torch.zeros(1, hidden_dim)

    def forward(self, current_state: torch.Tensor,
                external_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        mPFC自我参照处理

        Args:
            current_state: 当前状态表征
            external_input: 外部输入 (可选)
        """
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)

        # 1. 自我参照激活: 输入与自我的相关程度
        self_ref = float(self.self_reference_net(current_state).squeeze().detach())

        # 2. 自我评价: 当前状态 vs 理想自我
        ideal = self.ideal_self.unsqueeze(0) if self.ideal_self.dim() == 1 else self.ideal_self
        eval_input = torch.cat([current_state, ideal.expand(current_state.shape[0], -1)], dim=-1)
        self_eval = float(self.self_eval_net(eval_input).squeeze().detach())

        # 3. 自传体连贯性: 当前自我与历史自我的一致性
        auto_self = self.autobiographical_self.unsqueeze(0)
        # 用当前状态更新自传体自我 (EMA)
        with torch.no_grad():
            self.autobiographical_self.data = (
                0.95 * self.autobiographical_self.data
                + 0.05 * current_state.mean(dim=-1)[:self.autobiographical_self.shape[0]]
                if current_state.shape[-1] >= self.autobiographical_self.shape[0]
                else self.autobiographical_self.data
            )
        # 连贯性 = 当前表征与自传体自我的相似度
        if current_state.shape[-1] == auto_self.shape[-1]:
            coherence = float(F.cosine_similarity(current_state, auto_self, dim=-1).mean().detach())
        else:
            coherence = 0.5
        coherence = float(np.clip(coherence * 0.5 + 0.5, 0.0, 1.0))  # [-1,1] -> [0,1]

        # 4. 心理时间旅行 (GRU向前推进)
        if current_state.shape[-1] != self.hidden.shape[-1]:
            # 维度不匹配时跳过GRU
            mtt_depth = 0.2
        else:
            self.hidden = self.time_travel_net(current_state, self.hidden)
            mtt_depth = float(self.hidden.abs().mean().detach())

        # 5. 自我认同度
        endorse_input = torch.cat([
            auto_self.expand(current_state.shape[0], -1)[:, :self.autobiographical_self.shape[0]],
            current_state[:, :self.autobiographical_self.shape[0]]
        ], dim=-1) if current_state.shape[-1] >= self.autobiographical_self.shape[0] else torch.zeros(1, self.autobiographical_self.shape[0] * 2)
        self_endorse = float(self.endorsement_net(endorse_input).squeeze().detach())

        self.state = mPFCState(
            self_evaluation=self_eval,
            self_referential_activation=self_ref,
            autobiographical_coherence=coherence,
            mental_time_travel_depth=float(np.clip(mtt_depth, 0, 1)),
            self_endorsement=float(np.clip(self_endorse, 0, 1)),
        )

        return {
            'self_evaluation': self.state.self_evaluation,
            'self_reference': self.state.self_referential_activation,
            'autobiographical_coherence': self.state.autobiographical_coherence,
            'mental_time_travel': self.state.mental_time_travel_depth,
            'self_endorsement': self.state.self_endorsement,
        }


# ============ 后扣带回 (PCC) ============

class PosteriorCingulateCortex(nn.Module):
    """
    后扣带回皮层 (Posterior Cingulate Cortex)

    DMN的核心枢纽节点:
    - 自我相关性检测: 判断外部信息与自我的关联
    - 自传体记忆整合: 将海马体的情景记忆编织成自传体叙事
    - 场景构建: 在想象中构建包含自我的场景
    - 叙事连续性: 维持跨时间的自我叙事连贯性

    PCC是DMN中代谢活动最高的区域，也是静息态fMRI中
    功能连接最强的枢纽节点。

    参考:
    - Raichle et al. (2001): DMN的发现，PCC作为核心枢纽
        - Andrews-Hanna et al. (2010): PCC在自我参照和记忆中的作用
    - Hassabis et al. (2007): 场景构建与PCC
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 64):
        super().__init__()

        # 自我相关性检测网络
        self.relevance_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # [外部信息 || 自我模型]
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 自传体记忆整合网络
        self.narrative_net = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 场景构建网络
        self.scene_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
        )

        # 叙事连续性评估
        self.continuity_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim // 2),  # [当前叙事 || 历史叙事]
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 自传体叙事缓冲区
        self.narrative_buffer = deque(maxlen=20)
        self.self_model = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        self.state = PCCState()

    def forward(self, external_input: torch.Tensor,
                self_representation: torch.Tensor) -> Dict[str, Any]:
        """
        PCC自我相关性处理和叙事整合

        Args:
            external_input: 外部输入信息
            self_representation: 自我表征 (来自mPFC)
        """
        if external_input.dim() == 1:
            external_input = external_input.unsqueeze(0)
        if self_representation.dim() == 1:
            self_representation = self_representation.unsqueeze(0)

        # 1. 自我相关性检测
        relevance_input = torch.cat([external_input, self_representation], dim=-1)
        self_relevance = float(self.relevance_net(relevance_input).squeeze().detach())

        # 2. 自传体记忆整合
        self_model = self.self_model.unsqueeze(0).expand(external_input.shape[0], -1)
        if external_input.shape[-1] != self_model.shape[-1]:
            # 维度适配
            target_dim = min(external_input.shape[-1], self_model.shape[-1])
            ext_pooled = external_input[:, :target_dim]
            sm_pooled = self_model[:, :target_dim]
            narrative_input = torch.cat([ext_pooled, sm_pooled], dim=-1)
            # 确保维度匹配narrative_net的输入
            if narrative_input.shape[-1] < self.self_model.shape[0] + external_input.shape[-1]:
                narrative_input = F.pad(narrative_input, (0, self.self_model.shape[0] + external_input.shape[-1] - narrative_input.shape[-1]))
        else:
            narrative_input = torch.cat([external_input, self_model], dim=-1)

        narrative_encoding = self.narrative_net(narrative_input)

        # 3. 场景构建
        scene = self.scene_net(narrative_encoding)
        scene_vividness = float(scene.abs().mean().detach())

        # 4. 叙事连续性: 当前叙事与历史叙事的一致性
        if len(self.narrative_buffer) > 0:
            prev_narrative = self.narrative_buffer[-1]
            if prev_narrative.shape == narrative_encoding.shape:
                continuity_input = torch.cat([
                    narrative_encoding, prev_narrative
                ], dim=-1)
                continuity = float(self.continuity_net(continuity_input).squeeze().detach())
            else:
                continuity = 0.7  # 默认
        else:
            continuity = 0.7  # 初始默认

        # 5. 自传体回忆强度 (与自我相关性正相关)
        recall = float(np.clip(self_relevance * 0.8 + 0.2, 0.0, 1.0))

        # 6. 更新叙事缓冲区
        self.narrative_buffer.append(narrative_encoding.detach().clone())

        # 7. 自我模型更新 (EMA)
        with torch.no_grad():
            self.self_model.data = 0.98 * self.self_model.data + 0.02 * narrative_encoding.mean(dim=0)

        self.state = PCCState(
            self_relevance=self_relevance,
            narrative_continuity=continuity,
            autobiographical_recall=recall,
            scene_construction=float(np.clip(scene_vividness, 0, 1)),
            self_model_confidence=float(np.clip(continuity * 0.8 + 0.2, 0, 1)),
        )

        return {
            'self_relevance': self.state.self_relevance,
            'narrative_continuity': self.state.narrative_continuity,
            'autobiographical_recall': self.state.autobiographical_recall,
            'scene_construction': self.state.scene_construction,
            'self_model_confidence': self.state.self_model_confidence,
        }


# ============ 楔前叶 (Precuneus) ============

class PrecuneusSystem(nn.Module):
    """
    楔前叶 (Precuneus)

    自我意识中最"自我"的脑区:
    - 第一人称视角: 以自我为中心的空间和心理视角
    - 自我加工: 处理关于自我的信息
    - 心理意象: 在脑中生成生动的自我相关画面
    - 自我定位: 自我在世界中的位置感

    楔前叶是DMN中连接性最高的区域之一，
    在麻醉和植物状态下活动大幅降低，暗示其与意识本身相关。

    参考:
    - Cavanna & Trimble (2006): 楔前叶在意识中的作用
    - Freton et al. (2014): 自我加工与楔前叶
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 64):
        super().__init__()

        # 第一人称视角网络
        self.perspective_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 自我加工网络
        self.self_processing_net = nn.Sequential(
            nn.Linear(state_dim + state_dim, hidden_dim),  # [自我状态 || 环境状态]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 心理意象网络
        self.imagery_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # 允许正负值 = 更生动的意象
        )

        # 自我定位网络 (自我在世界中的位置)
        self.location_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.state = PrecuneusState()

    def forward(self, self_state: torch.Tensor,
                environment_state: torch.Tensor) -> Dict[str, Any]:
        """
        楔前叶自我加工

        Args:
            self_state: 自我状态
            environment_state: 环境/世界状态
        """
        if self_state.dim() == 1:
            self_state = self_state.unsqueeze(0)
        if environment_state.dim() == 1:
            environment_state = environment_state.unsqueeze(0)

        # 1. 第一人称视角强度
        perspective = float(self.perspective_net(self_state).squeeze().detach())

        # 2. 自我加工: 自我状态 vs 环境状态的对比
        combined = torch.cat([self_state, environment_state], dim=-1)
        self_processing = float(self.self_processing_net(combined).squeeze().detach())

        # 3. 心理意象生动性
        imagery = self.imagery_net(self_state)
        vividness = float(imagery.abs().mean().detach())

        # 4. 自我定位
        location = float(self.location_net(self_state).squeeze().detach())

        self.state = PrecuneusState(
            first_person_perspective=perspective,
            self_processing=self_processing,
            mental_imagery_vividness=float(np.clip(vividness, 0, 1)),
            self_location=location,
        )

        return {
            'first_person_perspective': perspective,
            'self_processing': self_processing,
            'mental_imagery_vividness': self.state.mental_imagery_vividness,
            'self_location': location,
        }


# ============ 默认模式网络集成 (DMN Integration) ============

class DefaultModeNetwork(nn.Module):
    """
    默认模式网络 (Default Mode Network)

    DMN是大脑在静息/内省时高度活跃的网络:
    - 任务负向: 在执行外部任务时被抑制，在休息/内省时激活
    - 心智游移 (Mind Wandering): DMN活跃时思维自发游移
    - 元意识: 意识到自己的思维在游移 (DMN + 背外侧PFC)

    DMN核心节点: mPFC, PCC/楔前叶, 内侧颞叶(含海马), 外侧颞叶, 顶下小叶

    参考:
        - Raichle et al. (2001): DMN的发现
    - Andrews-Hanna (2010): DMN的功能解剖
    - Schooler et al. (2011): 心智游移与元意识
    - Christoff et al. (2011): DMN与元认知
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # DMN激活调节器 (任务正/负向切换)
        self.activation_gate = nn.Sequential(
            nn.Linear(3, 16),  # [任务负荷, 疲劳, 情绪效价]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # DMN激活水平 [0,1]
        )

        # 心智游移网络
        self.mind_wander_net = nn.Sequential(
            nn.Linear(2, 16),  # [DMN激活, 任务负荷]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # 元意识网络 (DMN + dlPFC交互)
        self.meta_awareness_net = nn.Sequential(
            nn.Linear(3, 16),  # [心智游移, DMN激活, 认知控制]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # 自发思维生成器
        self.spontaneous_thought_net = nn.GRUCell(1, hidden_dim)
        self.thought_hidden = torch.zeros(1, hidden_dim)

        self.state = DMNState()

    def forward(self, task_load: float = 0.3,
                fatigue: float = 0.3,
                emotional_valence: float = 0.0,
                cognitive_control: float = 0.5) -> Dict[str, Any]:
        """
        DMN激活调节

        Args:
            task_load: 当前任务负荷 [0,1]
            fatigue: 疲劳度 [0,1]
            emotional_valence: 情绪效价 [-1,1]
            cognitive_control: 认知控制水平 [0,1]
        """
        # 1. DMN激活: 低任务负荷、高疲劳 -> 高DMN激活
        gate_input = torch.tensor([[task_load, fatigue, (emotional_valence + 1) / 2]])
        dmn_activation = float(self.activation_gate(gate_input).squeeze().detach())
        # 低任务负荷时增强DMN
        dmn_activation *= (1.2 - task_load)
        dmn_activation = float(np.clip(dmn_activation, 0.0, 1.0))

        # 2. 任务负向优势 (>0.5 = 内省模式)
        task_negative = dmn_activation * (1 - task_load)

        # 3. 心智游移
        mw_input = torch.tensor([[dmn_activation, task_load]])
        mind_wandering = float(self.mind_wander_net(mw_input).squeeze().detach())

        # 4. 自发思维
        thought_input = torch.tensor([[dmn_activation]])
        self.thought_hidden = self.spontaneous_thought_net(
            thought_input, self.thought_hidden
        )
        thought_rate = float(np.clip(self.thought_hidden.abs().mean().detach(), 0, 1))

        # 5. 元意识: 意识到自己在走神
        meta_input = torch.tensor([[mind_wandering, dmn_activation, cognitive_control]])
        meta_awareness = float(self.meta_awareness_net(meta_input).squeeze().detach())
        # 高认知控制增强元意识
        meta_awareness *= (0.5 + 0.5 * cognitive_control)

        self.state = DMNState(
            dmn_activation=dmn_activation,
            task_negative_dominance=float(np.clip(task_negative, 0, 1)),
            mind_wandering=float(np.clip(mind_wandering, 0, 1)),
            spontaneous_thought_rate=float(np.clip(thought_rate, 0, 1)),
            meta_awareness=float(np.clip(meta_awareness, 0, 1)),
        )

        return {
            'dmn_activation': self.state.dmn_activation,
            'task_negative_dominance': self.state.task_negative_dominance,
            'mind_wandering': self.state.mind_wandering,
            'spontaneous_thought_rate': self.state.spontaneous_thought_rate,
            'meta_awareness': self.state.meta_awareness,
            'is_introspective_mode': task_negative > 0.5,
        }


# ============ 自我-他者区分 ============

class SelfOtherDistinction(nn.Module):
    """
    自我-他者区分 (Self-Other Distinction)

    维持自我边界的关键机制:
    - 自我边界清晰度: 我的 vs 你的
    - 自我-他者重叠度: 共情时边界模糊
    - 主体感 (Agency): 这个行为是我做的还是他做的?
    - 所有权感 (Ownership): 这个身体/思维是自我的还是他人的?

    参考:
    - Legrand & Ruby (2009): 自我意识与自我-他者区分
    - Tsakiris et al. (2007): 主体感和所有权感
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 64):
        super().__init__()

        # 自我边界网络
        self.boundary_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # [自我表征 || 他人表征]
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 重叠度网络
        self.overlap_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 主体感网络
        self.agency_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 所有权感网络
        self.ownership_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.state = SelfOtherState()

    def forward(self, self_representation: torch.Tensor,
                other_representation: torch.Tensor) -> Dict[str, Any]:
        """
        自我-他者区分

        Args:
            self_representation: 自我表征
            other_representation: 他人表征
        """
        if self_representation.dim() == 1:
            self_representation = self_representation.unsqueeze(0)
        if other_representation.dim() == 1:
            other_representation = other_representation.unsqueeze(0)

        # 1. 自我边界清晰度 (自我和他人越不相似 -> 边界越清晰)
        combined = torch.cat([self_representation, other_representation], dim=-1)
        boundary = float(self.boundary_net(combined).squeeze().detach())

        # 2. 自我-他者重叠 (共情时重叠增加)
        overlap = float(self.overlap_net(combined).squeeze().detach())

        # 3. 主体感
        agency = float(self.agency_net(self_representation).squeeze().detach())

        # 4. 所有权感
        ownership = float(self.ownership_net(self_representation).squeeze().detach())

        self.state = SelfOtherState(
            self_boundary_clarity=boundary,
            self_other_overlap=overlap,
            agency_sense=agency,
            ownership_sense=ownership,
        )

        return {
            'self_boundary_clarity': boundary,
            'self_other_overlap': overlap,
            'agency_sense': agency,
            'ownership_sense': ownership,
        }


# ============ 元自我意识 ============

class MetaSelfAwareness(nn.Module):
    """
    元自我意识 (Meta-Self-Awareness)

    递归的自我意识模型:
    - Level 0: 无自我意识 (纯反应式)
    - Level 1: 意识到自己的状态 (基本自我意识)
    - Level 2: 意识到自己在意识 (元自我意识)
    - Level 3+: 更深层的递归 (罕见)

    参考:
    - Schooler et al. (2011): 元意识的动态
    - Christoff et al. (2011): DMN与元认知
    """

    def __init__(self, state_dim: int = 64, max_recursive_depth: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.max_recursive_depth = max_recursive_depth

        # 递归意识网络
        self.recursive_net = nn.Sequential(
            nn.Linear(state_dim + 1, 64),  # [状态 || 当前深度]
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 内省深度网络
        self.introspection_net = nn.Sequential(
            nn.Linear(state_dim + max_recursive_depth, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 自我模型准确度评估
        self.accuracy_net = nn.Sequential(
            nn.Linear(4, 16),  # [dmn激活, 元意识, 自我参照, 叙事连续性]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.state = MetaSelfState()

    def forward(self, state: torch.Tensor,
                dmn_activation: float = 0.5,
                self_reference: float = 0.5,
                narrative_continuity: float = 0.7) -> Dict[str, Any]:
        """
        元自我意识计算

        递归检测: "我知道我在想什么" -> "我知道我知道我在想什么"
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # 1. 递归意识深度检测
        depth_scores = []
        current_awareness = 0.5
        for d in range(self.max_recursive_depth):
            depth_input = torch.cat([
                state,
                torch.tensor([[d / self.max_recursive_depth]], dtype=torch.float32).expand(state.shape[0], 1)
            ], dim=-1) if state.shape[-1] <= 64 else torch.cat([
                state[:, :63],
                torch.tensor([[d / self.max_recursive_depth]], dtype=torch.float32).expand(state.shape[0], 1)
            ], dim=-1)
            current_awareness = float(self.recursive_net(depth_input).squeeze().detach())
            depth_scores.append(current_awareness)

        # 实际递归深度: 持续超过阈值的层数
        recursive_depth = sum(1 for s in depth_scores if s > 0.4)

        # 2. 最高层意识 (对意识的意识)
        awareness_of_awareness = depth_scores[-1] if depth_scores else 0.3

        # 3. 内省深度
        depth_tensor = torch.tensor(depth_scores + [0.0] * (self.max_recursive_depth - len(depth_scores)))
        if state.shape[0] > 0:
            state_part = state[:, :self.state_dim] if state.shape[-1] >= self.state_dim else F.pad(state, (0, self.state_dim - state.shape[-1]))
            intro_input = torch.cat([state_part, depth_tensor.unsqueeze(0).expand(state.shape[0], -1)], dim=-1)
        else:
            intro_input = torch.zeros(1, self.state_dim + self.max_recursive_depth)
        introspection_depth = float(self.introspection_net(intro_input).squeeze().detach())

        # 4. 自我模型准确度
        accuracy_input = torch.tensor([[
            dmn_activation, awareness_of_awareness,
            self_reference, narrative_continuity
        ]])
        model_accuracy = float(self.accuracy_net(accuracy_input).squeeze().detach())

        self.state = MetaSelfState(
            awareness_of_awareness=awareness_of_awareness,
            recursive_depth=recursive_depth,
            self_model_accuracy=model_accuracy,
            introspection_depth=introspection_depth,
        )

        return {
            'awareness_of_awareness': awareness_of_awareness,
            'recursive_depth': recursive_depth,
            'self_model_accuracy': model_accuracy,
            'introspection_depth': introspection_depth,
        }


# ============ 自我意识中枢 (聚合器) ============

class SelfAwarenessCenter(nn.Module):
    """
    自我意识中枢 - 整合mPFC + PCC + 楔前叶 + DMN + 自我-他者区分 + 元自我意识

    层次结构:
    L0: 自我参照 (mPFC) - 这个信息和我有关吗?
    L1: 自我评价 (mPFC) - 我做得怎么样?
    L2: 自我叙事 (PCC) - 我的故事是什么?
    L3: 自我定位 (楔前叶) - 我在哪里?
    L4: 自我边界 (自我-他者区分) - 我vs他人的边界
    L5: 元意识 (DMN + 元自我意识) - 我知道我在想什么

    参考:
    - Northoff et al. (2006): 自我参照的皮质中线结构
    - Raichle et al. (2001): 默认模式网络
    - Cavanna & Trimble (2006): 楔前叶与意识
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 64,
                 event_bus=None):
        super().__init__()

        self.mpfc = MedialPrefrontalCortex(state_dim=state_dim, hidden_dim=hidden_dim)
        self.pcc = PosteriorCingulateCortex(state_dim=state_dim, hidden_dim=hidden_dim)
        self.precuneus = PrecuneusSystem(state_dim=state_dim, hidden_dim=hidden_dim)
        self.dmn = DefaultModeNetwork(hidden_dim=hidden_dim)
        self.self_other = SelfOtherDistinction(state_dim=state_dim, hidden_dim=hidden_dim)
        self.meta_self = MetaSelfAwareness(state_dim=state_dim)

        # 自我一致性整合网络
        self.coherence_net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 自我叙事生成
        self.narrative_templates = [
            "正在探索未知领域",
            "维持自我稳定运行",
            "在挑战中寻求成长",
            "平衡内在与外在需求",
            "反思并调整行为策略",
        ]

        self.state = SelfAwarenessState()
        self.step_count = 0

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "brain_update",
                self._handle_brain_update,
                priority=0,
                name="self_awareness",
            )

    def _handle_brain_update(self, event) -> Dict[str, Any]:
        """Event-driven handler for brain_update events."""
        import torch as _torch
        state = event.data.get("internal_state", {})
        state_tensor = event.data.get("state_tensor", _torch.randn(1, 64))

        result = self.step(
            self_state=state_tensor,
            external_input=state_tensor,  # 外部输入也基于真实状态
            task_load=state.get("task_load", 0.3),
            fatigue=state.get("fatigue", 0.3),
            emotional_valence=state.get("emotional_valence", 0.0),
            cognitive_control=state.get("cognitive_control", 0.5),
        )

        for key, value in result.items():
            state[key] = value

        return result

    def step(self, self_state: torch.Tensor,
             external_input: torch.Tensor,
             other_representation: Optional[torch.Tensor] = None,
             task_load: float = 0.3,
             fatigue: float = 0.3,
             emotional_valence: float = 0.0,
             cognitive_control: float = 0.5) -> Dict[str, Any]:
        """
        执行一个自我意识处理步

        Args:
            self_state: 自身状态表征
            external_input: 外部环境输入
            other_representation: 他人表征 (可选)
            task_load: 任务负荷 [0,1]
            fatigue: 疲劳度 [0,1]
            emotional_valence: 情绪效价 [-1,1]
            cognitive_control: 认知控制 [0,1]
        """
        self.step_count += 1

        if self_state.dim() == 1:
            self_state = self_state.unsqueeze(0)
        if external_input.dim() == 1:
            external_input = external_input.unsqueeze(0)

        # 1. mPFC 自我参照处理
        mpfc_result = self.mpfc(self_state)

        # 2. PCC 自我相关性和叙事整合
        pcc_result = self.pcc(external_input, self_state)

        # 3. 楔前叶 自我加工
        precuneus_result = self.precuneus(self_state, external_input)

        # 4. DMN 调节
        dmn_result = self.dmn(
            task_load=task_load, fatigue=fatigue,
            emotional_valence=emotional_valence,
            cognitive_control=cognitive_control,
        )

        # 5. 自我-他者区分 (确定性扰动，非随机)
        if other_representation is None:
            # 从自我状态生成确定性的"他者"表征: 轻微扰动 (反映观察到的他人与自我的差异)
            # 使用self_state的EMA历史均值作为扰动基准，确保可复现
            if not hasattr(self, '_self_state_ema'):
                self._self_state_ema = self_state.detach().clone()
            # EMA更新自我表征
            self._self_state_ema = 0.95 * self._self_state_ema + 0.05 * self_state.detach()
            # 他者 = 自我EMA + 控制性偏移 (反映"我看到的另一个人")
            other_representation = self._self_state_ema + 0.15 * torch.sin(self._self_state_ema * 3.0)
        so_result = self.self_other(self_state, other_representation)

        # 6. 元自我意识
        meta_result = self.meta_self(
            self_state,
            dmn_activation=dmn_result['dmn_activation'],
            self_reference=mpfc_result['self_reference'],
            narrative_continuity=pcc_result['narrative_continuity'],
        )

        # 7. 自我一致性计算
        coherence_input = torch.tensor([[
            mpfc_result['self_evaluation'],
            mpfc_result['autobiographical_coherence'],
            pcc_result['narrative_continuity'],
            precuneus_result['first_person_perspective'],
            dmn_result['meta_awareness'],
            so_result['self_boundary_clarity'],
            meta_result['awareness_of_awareness'],
            meta_result['self_model_accuracy'],
        ]])
        self_coherence = float(self.coherence_net(coherence_input).squeeze().detach())

        # 8. 总体自我意识水平
        overall = float(np.clip(
            0.2 * mpfc_result['self_reference']
            + 0.15 * pcc_result['self_relevance']
            + 0.15 * precuneus_result['self_processing']
            + 0.2 * meta_result['awareness_of_awareness']
            + 0.15 * dmn_result['meta_awareness']
            + 0.15 * so_result['self_boundary_clarity'],
            0.0, 1.0
        ))

        # 9. 自我叙事选择
        if dmn_result['is_introspective_mode']:
            narrative = self.narrative_templates[min(
                int(mpfc_result['self_evaluation'] * len(self.narrative_templates)),
                len(self.narrative_templates) - 1
            )]
        else:
            narrative = "执行外部任务中"

        # 10. 更新总状态
        self.state = SelfAwarenessState(
            mpfc=self.mpfc.state,
            pcc=self.pcc.state,
            precuneus=self.precuneus.state,
            dmn=self.dmn.state,
            self_other=self.self_other.state,
            meta_self=self.meta_self.state,
            self_narrative=narrative,
            self_coherence=self_coherence,
            overall_self_awareness=overall,
        )

        return {
            'self_evaluation': mpfc_result['self_evaluation'],
            'self_reference': mpfc_result['self_reference'],
            'autobiographical_coherence': mpfc_result['autobiographical_coherence'],
            'self_endorsement': mpfc_result['self_endorsement'],
            'self_relevance': pcc_result['self_relevance'],
            'narrative_continuity': pcc_result['narrative_continuity'],
            'first_person_perspective': precuneus_result['first_person_perspective'],
            'self_processing': precuneus_result['self_processing'],
            'dmn_activation': dmn_result['dmn_activation'],
            'is_introspective_mode': dmn_result['is_introspective_mode'],
            'mind_wandering': dmn_result['mind_wandering'],
            'meta_awareness': dmn_result['meta_awareness'],
            'self_boundary_clarity': so_result['self_boundary_clarity'],
            'self_other_overlap': so_result['self_other_overlap'],
            'awareness_of_awareness': meta_result['awareness_of_awareness'],
            'recursive_depth': meta_result['recursive_depth'],
            'self_model_accuracy': meta_result['self_model_accuracy'],
            'introspection_depth': meta_result['introspection_depth'],
            'self_coherence': self_coherence,
            'self_narrative': narrative,
            'overall_self_awareness': overall,
        }

    def get_summary(self) -> Dict:
        """获取自我意识中枢摘要"""
        return {
            'self_evaluation': self.state.mpfc.self_evaluation,
            'self_reference': self.state.mpfc.self_referential_activation,
            'autobiographical_coherence': self.state.mpfc.autobiographical_coherence,
            'self_relevance': self.state.pcc.self_relevance,
            'narrative_continuity': self.state.pcc.narrative_continuity,
            'first_person_perspective': self.state.precuneus.first_person_perspective,
            'dmn_activation': self.state.dmn.dmn_activation,
            'introspective_mode': self.state.dmn.task_negative_dominance > 0.5,
            'mind_wandering': self.state.dmn.mind_wandering,
            'meta_awareness': self.state.dmn.meta_awareness,
            'self_boundary': self.state.self_other.self_boundary_clarity,
            'recursive_depth': self.state.meta_self.recursive_depth,
            'self_coherence': self.state.self_coherence,
            'self_narrative': self.state.self_narrative,
            'overall_self_awareness': self.state.overall_self_awareness,
            'step_count': self.step_count,
        }


def create_self_awareness_center(**kwargs) -> SelfAwarenessCenter:
    """工厂函数: 创建自我意识中枢"""
    return SelfAwarenessCenter(**kwargs)


__all__ = [
    'mPFCState',
    'PCCState',
    'PrecuneusState',
    'DMNState',
    'SelfOtherState',
    'MetaSelfState',
    'SelfAwarenessState',
    'MedialPrefrontalCortex',
    'PosteriorCingulateCortex',
    'PrecuneusSystem',
    'DefaultModeNetwork',
    'SelfOtherDistinction',
    'MetaSelfAwareness',
    'SelfAwarenessCenter',
    'create_self_awareness_center',
]
