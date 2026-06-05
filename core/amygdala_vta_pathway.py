"""
杏仁核-VTA奖赏通路 (Amygdala-VTA Reward Pathway)

实现情绪到动机的完整回路连接:
  Amygdala → VTA (多巴胺神经元) → NAc (奖赏感受)

生物学基础:
  - Cardinal et al. (2002): 杏仁核-VTA解剖连接
  - Phelps & LeDoux (2005): 情绪-奖赏整合
  - Berridge & Robinson (1998): Incentive-sensitization理论

功能通路:
  1. BLA (基底外侧杏仁核) → VTA DA神经元 → 激活奖赏
  2. CeA (中央杏仁核) → VTA GABA → 抑制奖赏 (恐惧/回避)
  3. VTA → NAc Core → wanting信号
  4. VTA → NAc Shell → liking信号

关键机制:
  - 情绪增强奖赏: 正性情绪→DA释放增加
  - 恐惧抑制奖赏: 负性情绪→DA释放减少
  - 渴求(wanting) vs 快感(liking)分离
  - 敏化(sensitization): 慢性暴露导致wanting增强
"""

from dataclasses import dataclass
from typing import ClassVar, Set, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.abstract_brain_region import AbstractBrainRegion


@dataclass
class RewardPathwayState:
    """奖赏通路状态"""
    # 杏仁核输入
    bla_activation: float = 0.0      # 基底外侧杏仁核激活 [0, 1]
    cea_activation: float = 0.0      # 中央杏仁核激活 [0, 1]

    # VTA输出
    vta_da_release: float = 0.3      # VTA多巴胺释放 [0, 1]
    vta_gaba_output: float = 0.2     # VTA GABA输出 [0, 1]

    # NAc接收
    nac_wanting: float = 0.0         # 渴求信号 [0, 1]
    nac_liking: float = 0.0          # 快感信号 [0, 1]

    # 通路状态
    emotion_reward_coupling: float = 0.5  # 情绪-奖赏耦合强度 [0, 1]
    sensitization_level: float = 0.0      # 敏化水平 [0, 1]


class AmygdalaVTAProjection(nn.Module):
    """
    杏仁核到VTA的投射

    双通路:
    1. BLA (兴奋性) → VTA DA神经元 → 促进奖赏
    2. CeA (抑制性) → VTA GABA神经元 → 抑制奖赏

    参考: Tye et al. (2011) - Amygdala-VTA解剖投射
    """

    def __init__(self, input_dim: int = 32):
        super().__init__()

        # BLA → VTA兴奋通路
        self.bla_excitatory = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # 输出 [0, 1]
        )

        # CeA → VTA抑制通路
        self.cea_inhibitory = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 情绪效价调制
        self.valence_modulation = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        amygdala_activity: torch.Tensor,
        valence: float = 0.0,      # 情绪效价 [-1, 1]
        arousal: float = 0.5,      # 唤醒水平 [0, 1]
    ) -> dict[str, float]:
        """
        杏仁核-VTA投射计算

        Args:
            amygdala_activity: 杏仁核活动 [B, input_dim]
            valence: 情绪效价 (正→兴奋BLA, 负→激活CeA)
            arousal: 唤醒水平 (放大整体输出)

        Returns:
            bla_to_vta: BLA兴奋性输出 [0, 1]
            cea_to_vta: CeA抑制性输出 [0, 1]
            net_vta_drive: VTA净驱动 (兴奋-抑制)
        """
        if amygdala_activity.dim() == 1:
            amygdala_activity = amygdala_activity.unsqueeze(0)

        # BLA兴奋性投射 (正性情绪增强)
        bla_out = self.bla_excitatory(amygdala_activity).squeeze()
        # 效价调制: 正性情绪增强BLA输出
        bla_modulated = float(bla_out) * (1.0 + 0.5 * max(0, valence))

        # CeA抑制性投射 (负性情绪增强)
        cea_out = self.cea_inhibitory(amygdala_activity).squeeze()
        # 效价调制: 负性情绪增强CeA输出
        cea_modulated = float(cea_out) * (1.0 + 0.5 * max(0, -valence))

        # 唤醒放大
        arousal_factor = 0.5 + arousal
        bla_final = np.clip(bla_modulated * arousal_factor, 0.0, 1.0)
        cea_final = np.clip(cea_modulated * arousal_factor, 0.0, 1.0)

        # VTA净驱动
        net_drive = bla_final - cea_final * 0.5  # CeA权重较低

        return {
            'bla_to_vta': bla_final,
            'cea_to_vta': cea_final,
            'net_vta_drive': float(net_drive),
            'valence_effect': 0.5 * valence,
        }


class VTADopamineRelease(nn.Module):
    """
    VTA多巴胺释放系统

    接收杏仁核输入，调控NAc多巴胺水平:
    - Tonic DA: 基线水平 (~5-20 nM)
    - Phasic DA: 突发释放 (~200-400 nM)

    参考: Schultz et al. (1997) - DA奖赏预测误差
    """

    def __init__(self, baseline_da: float = 0.3):
        super().__init__()

        # 基线Tonic DA
        self.baseline_da = baseline_da

        # Phasic释放计算
        self.phasic_release = nn.Sequential(
            nn.Linear(2, 16),  # amygdala_drive, prediction_error
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # 敏化累加器 (慢性暴露)
        self.sensitization_accumulator = 0.0
        self.sensitization_decay = 0.001  # 每步衰减

    def forward(
        self,
        amygdala_drive: float,
        prediction_error: float = 0.0,
        novelty: float = 0.0,
        chronic_exposure: float = 0.0,
    ) -> dict[str, float]:
        """
        VTA DA释放计算

        Args:
            amygdala_drive: 杏仁核净驱动 [-0.5, 1]
            prediction_error: 奖赏预测误差 [-1, 1]
            novelty: 新颖性 [0, 1]
            chronic_exposure: 慢性暴露积累 [0, 1]

        Returns:
            tonic_da: Tonic DA水平 [0, 1]
            phasic_da: Phasic DA峰值 [0, 1]
            total_da: 总DA释放 [0, 1]
            sensitization: 当前敏化水平
        """
        # Tonic DA (基线 + 杏仁核驱动)
        tonic = np.clip(self.baseline_da + 0.3 * amygdala_drive, 0.1, 0.6)

        # Phasic DA (预测误差 + 新颖性)
        input_tensor = torch.tensor([amygdala_drive, prediction_error])
        phasic_base = float(self.phasic_release(input_tensor.unsqueeze(0)).squeeze())

        # 新颖性增强Phasic
        phasic = phasic_base * (1.0 + 0.5 * novelty)

        # 敏化: 慢性暴露导致wanting增强
        self.sensitization_accumulator += chronic_exposure * 0.01
        self.sensitization_accumulator -= self.sensitization_decay
        sensitization = np.clip(self.sensitization_accumulator, 0.0, 1.0)

        # 敏化增强Phasic (但不增强liking)
        phasic_sensitized = phasic * (1.0 + 0.5 * sensitization)

        # 总DA
        total_da = np.clip(tonic + phasic_sensitized, 0.0, 1.0)

        return {
            'tonic_da': tonic,
            'phasic_da': phasic,
            'phasic_sensitized': float(phasic_sensitized),
            'total_da': total_da,
            'sensitization': sensitization,
        }


class NAcRewardReceiver(nn.Module):
    """
    Nucleus Accumbens奖赏接收器

    区分wanting (渴求) vs liking (快感):
    - NAc Core: wanting信号 (动机驱动)
    - NAc Shell: liking信号 (快感感受)

    参考: Berridge & Robinson (1998) - Incentive-sensitization
    """

    def __init__(self):
        super().__init__()

        # NAc Core wanting计算
        self.core_wanting = nn.Sequential(
            nn.Linear(2, 16),  # da_level, sensitization
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # NAc Shell liking计算 (不受敏化影响)
        self.shell_liking = nn.Sequential(
            nn.Linear(1, 16),  # da_level only
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        da_level: float,
        sensitization: float = 0.0,
        opioid_level: float = 0.0,  # 内啡肽增强liking
    ) -> dict[str, float]:
        """
        NAc奖赏计算

        Args:
            da_level: 多巴胺水平 [0, 1]
            sensitization: 敏化水平 (仅影响wanting)
            opioid_level: 内啡肽水平 (增强liking)

        Returns:
            wanting: 渴求强度 [0, 1]
            liking: 快感强度 [0, 1]
            hedonic_impact: 整体愉悦影响
        """
        # NAc Core wanting (敏化增强)
        input_wanting = torch.tensor([da_level, sensitization])
        wanting = float(self.core_wanting(input_wanting.unsqueeze(0)).squeeze())
        wanting_sensitized = wanting * (1.0 + 0.5 * sensitization)

        # NAc Shell liking (不受敏化影响，内啡肽增强)
        input_liking = torch.tensor([da_level])
        liking_base = float(self.shell_liking(input_liking.unsqueeze(0)).squeeze())
        liking = liking_base * (1.0 + 0.3 * opioid_level)

        # Hedonic impact (wanting + liking综合)
        hedonic = (wanting_sensitized + liking) / 2

        return {
            'wanting': np.clip(wanting_sensitized, 0.0, 1.0),
            'liking': np.clip(liking, 0.0, 1.0),
            'hedonic_impact': float(hedonic),
            'wanting_liking_ratio': wanting_sensitized / max(0.01, liking),
        }


class AmygdalaVTARewardPathway(AbstractBrainRegion):
    """
    杏仁核-VTA奖赏通路完整系统

    实现情绪→动机→奖赏的完整回路:
      Amygdala(BLA/CeA) → VTA → NAc(Core/Shell)

    关键功能:
    1. 情绪增强奖赏 (正性情绪→DA↑)
    2. 恐惧抑制奖赏 (负性情绪→DA↓)
    3. Wanting/Liking分离 (Berridge理论)
    4. 敏化机制 (慢性暴露→wanting增强)
    """

    region_name: ClassVar[str] = "amygdala_vta_pathway"

    @classmethod
    def required_keys(cls) -> Set[str]:
        return set(["amygdala_activity", "emotion_valence", "emotion_arousal",
                    "prediction_error", "novelty", "opioid_level"])

    @classmethod
    def output_keys(cls) -> Set[str]:
        return set(["vta_da_release", "nac_wanting", "nac_liking",
                    "sensitization", "emotion_reward_coupling"])

    def __init__(
        self,
        input_dim: int = 32,
        baseline_da: float = 0.3,
        event_bus=None,
    ):
        super().__init__()

        # 子系统
        self.amygdala_projection = AmygdalaVTAProjection(input_dim)
        self.vta_release = VTADopamineRelease(baseline_da)
        self.nac_receiver = NAcRewardReceiver()

        # 状态
        self.state = RewardPathwayState()

        # 情绪-奖赏耦合追踪
        self.coupling_history = []

        # Event-driven registration
        self.event_bus = event_bus
        if event_bus is not None:
            event_bus.subscribe(
                "brain_update",
                self._handle_brain_update,
                priority=2,
                name="amygdala_vta_pathway",
            )

    def _handle_brain_update(self, event) -> dict[str, Any]:
        """Event-driven handler."""
        state = event.data.get("internal_state", {})

        # 获取杏仁核活动
        amygdala_activity = state.get("limbic_activity", np.zeros(32))

        result = self.step(
            amygdala_activity=amygdala_activity,
            valence=state.get("limbic_valence", 0.0),
            arousal=state.get("limbic_arousal", 0.5),
            prediction_error=state.get("bg_td_error", 0.0),
        )

        state["vta_da_release"] = result["vta_da_release"]
        state["nac_wanting"] = result["nac_wanting"]
        state["nac_liking"] = result["nac_liking"]

        return result

    def step(
        self,
        amygdala_activity: np.ndarray = None,
        valence: float = 0.0,
        arousal: float = 0.5,
        prediction_error: float = 0.0,
        novelty: float = 0.0,
        chronic_exposure: float = 0.0,
        opioid_level: float = 0.0,
        **kwargs,
    ) -> dict[str, Any]:
        """执行一个奖赏通路步"""
        if amygdala_activity is None:
            amygdala_activity = np.zeros(32)

        amygdala_tensor = torch.FloatTensor(amygdala_activity)

        # 杏仁核-VTA投射
        projection_result = self.amygdala_projection(
            amygdala_activity=amygdala_tensor,
            valence=valence,
            arousal=arousal,
        )

        # VTA DA释放
        vta_result = self.vta_release(
            amygdala_drive=projection_result['net_vta_drive'],
            prediction_error=prediction_error,
            novelty=novelty,
            chronic_exposure=chronic_exposure,
        )

        # NAc接收
        nac_result = self.nac_receiver(
            da_level=vta_result['total_da'],
            sensitization=vta_result['sensitization'],
            opioid_level=opioid_level,
        )

        # 情绪-奖赏耦合计算
        coupling = abs(valence) * 0.5 + arousal * 0.3 + abs(prediction_error) * 0.2
        self.coupling_history.append(coupling)
        if len(self.coupling_history) > 100:
            self.coupling_history = self.coupling_history[-100:]

        # 更新状态
        self.state.bla_activation = projection_result['bla_to_vta']
        self.state.cea_activation = projection_result['cea_to_vta']
        self.state.vta_da_release = vta_result['total_da']
        self.state.nac_wanting = nac_result['wanting']
        self.state.nac_liking = nac_result['liking']
        self.state.sensitization_level = vta_result['sensitization']
        self.state.emotion_reward_coupling = coupling

        return {
            'bla_activation': projection_result['bla_to_vta'],
            'cea_activation': projection_result['cea_to_vta'],
            'net_vta_drive': projection_result['net_vta_drive'],
            'vta_da_release': vta_result['total_da'],
            'tonic_da': vta_result['tonic_da'],
            'phasic_da': vta_result['phasic_da'],
            'nac_wanting': nac_result['wanting'],
            'nac_liking': nac_result['liking'],
            'hedonic_impact': nac_result['hedonic_impact'],
            'sensitization': vta_result['sensitization'],
            'emotion_reward_coupling': coupling,
            'wanting_liking_ratio': nac_result['wanting_liking_ratio'],
        }

    def get_summary(self) -> dict[str, Any]:
        """获取状态摘要"""
        return {
            'bla_activation': self.state.bla_activation,
            'cea_activation': self.state.cea_activation,
            'vta_da_release': self.state.vta_da_release,
            'nac_wanting': self.state.nac_wanting,
            'nac_liking': self.state.nac_liking,
            'sensitization': self.state.sensitization_level,
            'emotion_reward_coupling': self.state.emotion_reward_coupling,
        }


__all__ = [
    'AmygdalaVTARewardPathway',
    'AmygdalaVTAProjection',
    'VTADopamineRelease',
    'NAcRewardReceiver',
    'RewardPathwayState',
]