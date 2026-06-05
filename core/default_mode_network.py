"""
默认模式网络 (Default Mode Network, DMN)

静息态大脑的核心网络，在被动休息、心智游走、自我参照思考时活跃。

生物学基础：
  - Raichle et al. (2001): DMN首次发现
  - Buckner et al. (2008): DMN与自我参照、记忆检索
  - Andrews-Hanna (2010): DMN功能分解

核心节点：
1. 后扣带回/楔前叶 (PCC/Precuneus) — DMN核心枢纽
2. 内侧前额叶 (mPFC) — 自我参照、社会认知
3. 外侧颞叶 (LTC) — 记忆检索
4. 海马体系统 (HC) — 已有实现，本模块整合连接

功能：
- 静息态活动 (高于任务态)
- 心智游走 (mind wandering)
- 自我参照加工
- 记忆检索与整合
- 社会认知模拟

与任务正向网络(TPN)的反相关：
  DMN活跃 ↔ TPN抑制 (任务执行时)
"""

from dataclasses import dataclass, field
from typing import ClassVar, Set, Any

import numpy as np
import torch
import torch.nn as nn

from core.abstract_brain_region import AbstractBrainRegion


@dataclass
class DMNState:
    """DMN网络状态"""
    # PCC活动
    pcc_activity: float = 0.5         # 后扣带回活动 [0, 1]
    pcc_connectivity: float = 0.5     # PCC枢纽连接度 [0, 1]

    # mPFC活动
    mpfc_activity: float = 0.5        # 内侧前额叶活动 [0, 1]
    self_referential_strength: float = 0.5  # 自我参照强度 [0, 1]

    # LTC活动
    ltc_activity: float = 0.5         # 外侧颞叶活动 [0, 1]

    # 网络整体
    dmn_coherence: float = 0.5        # DMN内部一致性 [0, 1]
    mind_wandering_intensity: float = 0.0  # 心智游走强度 [0, 1]

    # TPN反相关
    tpn_suppression: float = 0.0      # 任务网络抑制 [0, 1]
    resting_vs_task: str = "resting"  # "resting" / "task" / "transition"


class PosteriorCingulateCortex(nn.Module):
    """
    后扣带回 (Posterior Cingulate Cortex, PCC)

    DMN的核心枢纽，连接所有DMN节点：
    - 接收HC、mPFC、LTC输入
    - 调节静息态活动水平
    - 与Precuneus共同形成DMN核心

    参考: Fransson & Marrelec (2008) - PCC作为DMN枢纽
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim

        # 枢纽整合网络 (接收HC、mPFC、LTC输入)
        self.hub_integration = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )

        # 静息态基线调制 (高于任务态)
        self.resting_modulation = nn.Parameter(torch.tensor(0.3))

        # 活动输出
        self.activity_output = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 连接度计算 (枢纽强度)
        self.connectivity_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.current_activity = 0.5
        self.current_connectivity = 0.5

    def forward(
        self,
        hippocampus_input: torch.Tensor,
        mpfc_input: torch.Tensor,
        ltc_input: torch.Tensor,
        is_resting: bool = True,
        task_demand: float = 0.0,
    ) -> dict[str, float]:
        """
        PCC前向传播

        Args:
            hippocampus_input: 海马体输入 [B, input_dim]
            mpfc_input: mPFC输入 [B, input_dim]
            ltc_input: LTC输入 [B, input_dim]
            is_resting: 是否静息态
            task_demand: 任务需求强度 [0, 1]

        Returns:
            pcc_activity: PCC活动 [0, 1]
            pcc_connectivity: 枢纽连接度 [0, 1]
            resting_baseline: 静息态基线贡献
        """
        if hippocampus_input.dim() == 1:
            hippocampus_input = hippocampus_input.unsqueeze(0)
        if mpfc_input.dim() == 1:
            mpfc_input = mpfc_input.unsqueeze(0)
        if ltc_input.dim() == 1:
            ltc_input = ltc_input.unsqueeze(0)

        # 枢纽整合
        combined = torch.cat([hippocampus_input, mpfc_input, ltc_input], dim=-1)
        integrated = self.hub_integration(combined)

        # 静息态调制
        resting_boost = self.resting_modulation.item() if is_resting else -self.resting_modulation.item()
        task_suppression = -0.3 * task_demand  # 任务态抑制PCC

        # 活动计算
        activity_base = self.activity_output(integrated).squeeze()
        activity = float(torch.clamp(activity_base + resting_boost + task_suppression, 0.0, 1.0))
        self.current_activity = activity

        # 连接度
        connectivity = float(self.connectivity_net(integrated).squeeze())
        self.current_connectivity = connectivity

        return {
            'pcc_activity': activity,
            'pcc_connectivity': connectivity,
            'resting_baseline': resting_boost,
            'hub_integration_strength': float(integrated.abs().mean()),
        }


class MedialPrefrontalCortex_DMN(nn.Module):
    """
    内侧前额叶 (Medial Prefrontal Cortex, mPFC)

    DMN的前部节点，负责：
    - 自我参照加工 (self-referential processing)
    - 社会认知 (social cognition)
    - 情绪评价 (emotional valuation)
    - 心智游走 (mind wandering)

    参考: D'Argembeau et al. (2005) - mPFC自我参照功能
    参考: Schilbach et al. (2008) - mPFC社会认知
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim

        # 自我参照网络
        self.self_reference_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 社会认知网络
        self.social_cognition_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 心智游走生成
        self.mind_wandering_net = nn.Sequential(
            nn.Linear(input_dim + 10, hidden_dim),  # +10 for noise
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 活动整合
        self.activity_integration = nn.Linear(3, 1)

        self.current_activity = 0.5
        self.current_self_ref = 0.5
        self.current_mind_wandering = 0.0

    def forward(
        self,
        internal_state_input: torch.Tensor,
        self_relevance: float = 0.5,
        social_context: float = 0.5,
        attention_focus: float = 0.5,
    ) -> dict[str, float]:
        """
        mPFC前向传播

        Args:
            internal_state_input: 内部状态输入 [B, input_dim]
            self_relevance: 自我相关性 [0, 1]
            social_context: 社会情境强度 [0, 1]
            attention_focus: 注意聚焦度 [0, 1] (高→抑制心智游走)

        Returns:
            mpfc_activity: mPFC活动 [0, 1]
            self_referential_strength: 自我参照强度 [0, 1]
            mind_wandering_intensity: 心智游走强度 [0, 1]
        """
        if internal_state_input.dim() == 1:
            internal_state_input = internal_state_input.unsqueeze(0)

        # 自我参照
        self_ref = self.self_reference_net(internal_state_input).squeeze()
        self_ref_strength = float(self_ref) * self_relevance

        # 社会认知
        social = self.social_cognition_net(internal_state_input).squeeze()
        social_strength = float(social) * social_context

        # 心智游走 (注意分散时增加)
        noise = torch.randn(10) * 0.1
        combined = torch.cat([internal_state_input.squeeze(0), noise]).unsqueeze(0)
        mw_base = self.mind_wandering_net(combined).squeeze()
        # 注意聚焦抑制心智游走
        mind_wandering = float(mw_base) * (1.0 - attention_focus)

        # 整合活动
        activities = torch.tensor([self_ref_strength, social_strength, mind_wandering])
        integrated_activity = float(torch.sigmoid(self.activity_integration(activities.unsqueeze(0)).squeeze()))

        self.current_activity = integrated_activity
        self.current_self_ref = self_ref_strength
        self.current_mind_wandering = mind_wandering

        return {
            'mpfc_activity': integrated_activity,
            'self_referential_strength': self_ref_strength,
            'social_cognition_strength': social_strength,
            'mind_wandering_intensity': mind_wandering,
        }


class LateralTemporalCortex(nn.Module):
    """
    外侧颞叶 (Lateral Temporal Cortex, LTC)

    DMN的外侧节点，负责：
    - 记忆检索 (memory retrieval)
    - 语言语义加工 (semantic processing)
    - 概念整合 (conceptual integration)

    参考: Binder et al. (2009) - 颞叶语义系统
    """

    def __init__(self, input_dim: int = 32):
        super().__init__()

        # 记忆检索网络
        self.memory_retrieval = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 语义整合
        self.semantic_integration = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.current_activity = 0.5

    def forward(
        self,
        memory_input: torch.Tensor,
        semantic_demand: float = 0.5,
    ) -> dict[str, float]:
        """LTC前向传播"""
        if memory_input.dim() == 1:
            memory_input = memory_input.unsqueeze(0)

        retrieval = self.memory_retrieval(memory_input).squeeze()
        semantic = self.semantic_integration(memory_input).squeeze()

        activity = float((retrieval + semantic) / 2 * (1 + 0.2 * semantic_demand))
        activity = np.clip(activity, 0.0, 1.0)
        self.current_activity = activity

        return {
            'ltc_activity': activity,
            'memory_retrieval_strength': float(retrieval),
            'semantic_integration_strength': float(semantic),
        }


class DefaultModeNetwork(AbstractBrainRegion):
    """
    默认模式网络 (Default Mode Network)

    静息态大脑的核心网络，整合PCC、mPFC、LTC三个主要节点，
    以及与海马体的连接。

    关键特性：
    - 反相关性: DMN活跃 ↔ TPN抑制
    - 心智游走: 静息态自动思维流
    - 自我意识: 自我参照加工的核心

    参考: Raichle (2015) - DMN与静息态大脑
    """

    region_name: ClassVar[str] = "dmn"

    @classmethod
    def required_keys(cls) -> Set[str]:
        return set(["hippo_retrieval", "internal_state", "attention_focus",
                    "task_demand", "self_relevance"])

    @classmethod
    def output_keys(cls) -> Set[str]:
        return set(["dmn_activity", "dmn_coherence", "pcc_activity",
                    "mpfc_activity", "ltc_activity", "mind_wandering",
                    "self_referential", "tpn_suppression"])

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        event_bus=None,
    ):
        super().__init__()

        # DMN核心节点
        self.pcc = PosteriorCingulateCortex(input_dim, hidden_dim)
        self.mpfc = MedialPrefrontalCortex_DMN(input_dim, hidden_dim)
        self.ltc = LateralTemporalCortex(input_dim)

        # 网络状态
        self.state = DMNState()

        # DMN-TPN反相关控制器
        self.anti_correlation_net = nn.Sequential(
            nn.Linear(2, 16),  # dmn_activity, task_demand
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),  # 输出 [-1, 1]
        )

        # 内部一致性计算
        self.coherence_net = nn.Sequential(
            nn.Linear(3, 16),  # pcc, mpfc, ltc
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Event-driven registration
        self.event_bus = event_bus
        if event_bus is not None:
            event_bus.subscribe(
                "brain_update",
                self._handle_brain_update,
                priority=4,
                name="dmn",
            )

    def _handle_brain_update(self, event) -> dict[str, Any]:
        """Event-driven handler for brain_update events."""
        state = event.data.get("internal_state", {})

        result = self.step(
            hippocampus_retrieval=state.get("hippo_retrieval", np.zeros(32)),
            internal_state=state.get("state_tensor", np.zeros(32)),
            attention_focus=state.get("attention_focus", 0.5),
            task_demand=state.get("task_demand", 0.0),
            self_relevance=state.get("self_relevance", 0.5),
        )

        state["dmn_activity"] = result["dmn_activity"]
        state["pcc_activity"] = result["pcc_activity"]
        state["mpfc_activity"] = result["mpfc_activity"]
        state["mind_wandering"] = result["mind_wandering"]

        return result

    def step(
        self,
        hippocampus_retrieval: np.ndarray = None,
        internal_state: np.ndarray = None,
        attention_focus: float = 0.5,
        task_demand: float = 0.0,
        self_relevance: float = 0.5,
        social_context: float = 0.5,
        **kwargs,
    ) -> dict[str, Any]:
        """执行一个DMN调节步"""
        # 默认输入
        if hippocampus_retrieval is None:
            hippocampus_retrieval = np.zeros(32)
        if internal_state is None:
            internal_state = np.zeros(32)

        # 转换为tensor
        hc_tensor = torch.FloatTensor(hippocampus_retrieval)
        internal_tensor = torch.FloatTensor(internal_state)

        # 判断静息态 vs 任务态
        is_resting = task_demand < 0.3

        # PCC计算
        pcc_result = self.pcc(
            hippocampus_input=hc_tensor,
            mpfc_input=internal_tensor,
            ltc_input=internal_tensor,
            is_resting=is_resting,
            task_demand=task_demand,
        )

        # mPFC计算
        mpfc_result = self.mpfc(
            internal_state_input=internal_tensor,
            self_relevance=self_relevance,
            social_context=social_context,
            attention_focus=attention_focus,
        )

        # LTC计算
        ltc_result = self.ltc(
            memory_input=hc_tensor,
            semantic_demand=1.0 - task_demand,
        )

        # 计算DMN整体活动
        dmn_activity = (pcc_result['pcc_activity'] +
                        mpfc_result['mpfc_activity'] +
                        ltc_result['ltc_activity']) / 3

        # 计算内部一致性
        activities = torch.tensor([
            pcc_result['pcc_activity'],
            mpfc_result['mpfc_activity'],
            ltc_result['ltc_activity'],
        ])
        coherence = float(self.coherence_net(activities.unsqueeze(0)).squeeze())

        # DMN-TPN反相关
        anti_input = torch.tensor([dmn_activity, task_demand])
        tpn_suppression = float(self.anti_correlation_net(anti_input.unsqueeze(0)).squeeze())
        tpn_suppression = (tpn_suppression + 1) / 2  # 转换到 [0, 1]

        # 更新状态
        self.state.pcc_activity = pcc_result['pcc_activity']
        self.state.mpfc_activity = mpfc_result['mpfc_activity']
        self.state.ltc_activity = ltc_result['ltc_activity']
        self.state.dmn_coherence = coherence
        self.state.mind_wandering_intensity = mpfc_result['mind_wandering_intensity']
        self.state.tpn_suppression = tpn_suppression
        self.state.resting_vs_task = "resting" if is_resting else "task"

        return {
            'dmn_activity': dmn_activity,
            'dmn_coherence': coherence,
            'pcc_activity': pcc_result['pcc_activity'],
            'pcc_connectivity': pcc_result['pcc_connectivity'],
            'mpfc_activity': mpfc_result['mpfc_activity'],
            'ltc_activity': ltc_result['ltc_activity'],
            'mind_wandering': mpfc_result['mind_wandering_intensity'],
            'self_referential': mpfc_result['self_referential_strength'],
            'tpn_suppression': tpn_suppression,
            'resting_vs_task': self.state.resting_vs_task,
        }

    def get_summary(self) -> dict[str, Any]:
        """获取DMN状态摘要"""
        return {
            'pcc_activity': self.state.pcc_activity,
            'mpfc_activity': self.state.mpfc_activity,
            'ltc_activity': self.state.ltc_activity,
            'dmn_coherence': self.state.dmn_coherence,
            'mind_wandering': self.state.mind_wandering_intensity,
            'tpn_suppression': self.state.tpn_suppression,
            'network_state': self.state.resting_vs_task,
        }


__all__ = [
    'DefaultModeNetwork',
    'PosteriorCingulateCortex',
    'MedialPrefrontalCortex_DMN',
    'LateralTemporalCortex',
    'DMNState',
]