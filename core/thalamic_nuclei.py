"""
丘脑特异性核团 (Thalamic Specific Nuclei)

实现丘脑的感觉和运动中继核团：
  - VL (Ventrolateral): 运动中继 - 小脑→运动皮层
  - LG (Lateral Geniculate): 视觉中继 - 视网膜→V1
  - MG (Medial Geniculate): 听觉中继 - 听神经→A1
  - MD (Mediodorsal): 认知中继 - 边缘系统→PFC
  - TRN (Thalamic Reticular Nucleus): GABA抑制 - 调节所有核团

生物学基础:
  - Sherman & Guillery (2006): 丘脑功能解剖
  - Jones (2007): 丘脑图谱
  - Castellotti et al. (2023): TRN振荡

功能:
  1. 感觉门控 (TRN抑制)
  2. 运动协调 (VL中继)
  3. 视觉传递 (LG中继)
  4. 听觉传递 (MG中继)
  5. 认知整合 (MD中继)
"""

from dataclasses import dataclass
from typing import ClassVar, Set, Any

import numpy as np
import torch
import torch.nn as nn

from core.abstract_brain_region import AbstractBrainRegion


@dataclass
class ThalamicNucleusState:
    """丘脑核团状态"""
    # VL运动中继
    vl_motor_signal: float = 0.0        # 运动信号强度 [0, 1]
    vl_cerebellum_input: float = 0.0    # 小脑输入 [0, 1]
    vl_motor_output: float = 0.0        # 运动皮层输出 [0, 1]

    # LG视觉中继
    lg_visual_signal: float = 0.0       # 视觉信号强度 [0, 1]
    lg_retina_input: float = 0.0        # 视网膜输入 [0, 1]
    lg_v1_output: float = 0.0           # V1输出 [0, 1]

    # MG听觉中继
    mg_auditory_signal: float = 0.0     # 听觉信号强度 [0, 1]
    mg_cochlea_input: float = 0.0       # 听神经输入 [0, 1]
    mg_a1_output: float = 0.0           # A1输出 [0, 1]

    # MD认知中继
    md_cognitive_signal: float = 0.0    # 认知信号强度 [0, 1]
    md_pfc_output: float = 0.0          # PFC输出 [0, 1]

    # TRN抑制
    trn_gating: float = 0.5             # TRN门控 [0, 1]
    trn_inhibition: float = 0.0         # TRN抑制强度 [0, 1]


class VentrolateralNucleus(nn.Module):
    """
    外侧核 (Ventrolateral Nucleus, VL)

    运动中继核团:
    - 接收: 小脑输出、基底节输出
    - 投射: 运动皮层M1、运动前区PM
    - 功能: 运动协调、运动计划传递

    参考: Asanuma et al. (1983) - VL运动中继
    """

    def __init__(
        self,
        input_dim: int = 32,
        relay_gain: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.relay_gain = relay_gain

        # 小脑→VL处理
        self.cerebellum_processor = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Tanh(),
        )

        # VL→M1输出
        self.motor_output = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # 运动信号历史
        self.motor_history = []

    def forward(
        self,
        cerebellum_input: torch.Tensor,
        basal_ganglia_input: float = 0.0,
        trn_inhibition: float = 0.0,
    ) -> dict[str, float]:
        """
        VL运动中继

        Args:
            cerebellum_input: 小脑输入 [B, input_dim]
            basal_ganglia_input: 基底节输入 (运动抑制)
            trn_inhibition: TRN抑制强度 [0, 1]

        Returns:
            motor_signal: 运动信号强度
            motor_output: 运动皮层输出
        """
        if cerebellum_input.dim() == 1:
            cerebellum_input = cerebellum_input.unsqueeze(0)

        # 小脑信号处理
        processed = self.cerebellum_processor(cerebellum_input)

        # TRN抑制调制
        effective_signal = processed * (1.0 - trn_inhibition)

        # 基底节抑制调制 (BG高活动→抑制运动)
        bg_modulation = 1.0 - 0.3 * basal_ganglia_input

        # 运动输出
        motor_out = self.motor_output(effective_signal * bg_modulation)
        motor_out_value = float(motor_out.squeeze())

        # 记录历史
        self.motor_history.append(motor_out_value)
        if len(self.motor_history) > 50:
            self.motor_history = self.motor_history[-50:]

        return {
            'vl_motor_signal': float(effective_signal.mean()),
            'vl_cerebellum_input': float(cerebellum_input.mean()),
            'vl_motor_output': motor_out_value,
            'vl_relay_efficiency': 1.0 - trn_inhibition,
        }


class LateralGeniculateNucleus(nn.Module):
    """
    外侧膝状体 (Lateral Geniculate Nucleus, LGN)

    视觉中继核团:
    - 接收: 视网膜输出 (视神经)
    - 投射: 初级视觉皮层V1
    - 功能: 视觉信号中继、视觉注意力门控

    特点:
    - 6层结构 (小细胞层+大细胞层)
    - 不同层处理不同视觉信息 (颜色/运动)
    - TRN调制视觉注意

    参考: Casagrande et al. (1994) - LGN解剖功能
    """

    def __init__(
        self,
        input_dim: int = 64,  # 视觉信号维度
        n_layers: int = 6,    # LGN层数
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers

        # 小细胞层处理 (颜色、精细细节) - 层1-4
        self.parvocellular_processor = nn.Sequential(
            nn.Linear(input_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh(),
        )

        # 大细胞层处理 (运动、粗略形状) - 层5-6
        self.magnocellular_processor = nn.Sequential(
            nn.Linear(input_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh(),
        )

        # V1输出整合
        self.v1_output = nn.Sequential(
            nn.Linear(32, 16),  # parvo + magno
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # 视觉注意历史
        self.visual_attention_history = []

    def forward(
        self,
        retina_input: torch.Tensor,
        visual_attention: float = 0.5,
        trn_inhibition: float = 0.0,
        eyes_closed: bool = False,
    ) -> dict[str, float]:
        """
        LGN视觉中继

        Args:
            retina_input: 视网膜输入 [B, input_dim]
            visual_attention: 视觉注意强度 [0, 1]
            trn_inhibition: TRN抑制强度 [0, 1]
            eyes_closed: 是否闭眼 (alpha抑制)

        Returns:
            visual_signal: 视觉信号强度
            v1_output: V1输出
        """
        if retina_input.dim() == 1:
            retina_input = retina_input.unsqueeze(0)

        # 闭眼抑制 (alpha节律)
        eyes_factor = 0.0 if eyes_closed else 1.0

        # TRN门控调制
        gating_factor = (1.0 - trn_inhibition) * visual_attention * eyes_factor

        # 分离小细胞和大细胞输入
        parvo_input = retina_input[:, :self.input_dim // 2] * gating_factor
        magno_input = retina_input[:, self.input_dim // 2:] * gating_factor

        # 小细胞层处理
        parvo_processed = self.parvocellular_processor(parvo_input)

        # 大细胞层处理
        magno_processed = self.magnocellular_processor(magno_input)

        # 整合输出
        combined = torch.cat([parvo_processed, magno_processed], dim=-1)
        v1_out = self.v1_output(combined)
        v1_out_value = float(v1_out.squeeze())

        # 记录历史
        self.visual_attention_history.append({
            'attention': visual_attention,
            'output': v1_out_value,
            'gating': gating_factor,
        })
        if len(self.visual_attention_history) > 50:
            self.visual_attention_history = self.visual_attention_history[-50:]

        return {
            'lg_visual_signal': float(combined.mean()),
            'lg_retina_input': float(retina_input.mean()),
            'lg_v1_output': v1_out_value,
            'lg_parvocellular': float(parvo_processed.mean()),
            'lg_magnocellular': float(magno_processed.mean()),
            'lg_gating_efficiency': gating_factor,
        }


class MedialGeniculateNucleus(nn.Module):
    """
    内侧膝状体 (Medial Geniculate Nucleus, MGN)

    听觉中继核团:
    - 接收: 下丘输出 (听神经通路)
    - 投射: 初级听觉皮层A1
    - 功能: 听觉信号中继、听觉注意力门控

    分区:
    - 腹侧部: 精确频率映射
    - 背侧部: 广泛频率整合
    - 内侧部: 多模态整合

    参考: Winer (2006) - MGN解剖功能
    """

    def __init__(
        self,
        input_dim: int = 32,  # 听觉信号维度
        n_freq_channels: int = 16,  # 频率通道数
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_freq_channels = n_freq_channels

        # 腹侧部处理 (精确频率)
        self.ventral_processor = nn.Sequential(
            nn.Linear(n_freq_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Tanh(),
        )

        # 背侧部处理 (广泛整合)
        self.dorsal_processor = nn.Sequential(
            nn.Linear(input_dim - n_freq_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Tanh(),
        )

        # A1输出整合
        self.a1_output = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # 听觉注意历史
        self.auditory_attention_history = []

    def forward(
        self,
        cochlea_input: torch.Tensor,
        auditory_attention: float = 0.5,
        trn_inhibition: float = 0.0,
        frequency_tuning: float = 0.5,  # 频率选择性
    ) -> dict[str, float]:
        """
        MGN听觉中继

        Args:
            cochlea_input: 听神经输入 [B, input_dim]
            auditory_attention: 听觉注意强度 [0, 1]
            trn_inhibition: TRN抑制强度 [0, 1]
            frequency_tuning: 频率选择性 [0, 1]

        Returns:
            auditory_signal: 听觉信号强度
            a1_output: A1输出
        """
        if cochlea_input.dim() == 1:
            cochlea_input = cochlea_input.unsqueeze(0)

        # TRN门控调制
        gating_factor = (1.0 - trn_inhibition) * auditory_attention

        # 分离腹侧和背侧输入
        ventral_input = cochlea_input[:, :self.n_freq_channels] * gating_factor * frequency_tuning
        dorsal_input = cochlea_input[:, self.n_freq_channels:] * gating_factor * (1.0 - frequency_tuning * 0.5)

        # 腹侧部处理
        ventral_processed = self.ventral_processor(ventral_input)

        # 背侧部处理
        dorsal_processed = self.dorsal_processor(dorsal_input)

        # 整合输出
        combined = torch.cat([ventral_processed, dorsal_processed], dim=-1)
        a1_out = self.a1_output(combined)
        a1_out_value = float(a1_out.squeeze())

        # 记录历史
        self.auditory_attention_history.append({
            'attention': auditory_attention,
            'output': a1_out_value,
            'frequency_tuning': frequency_tuning,
        })
        if len(self.auditory_attention_history) > 50:
            self.auditory_attention_history = self.auditory_attention_history[-50:]

        return {
            'mg_auditory_signal': float(combined.mean()),
            'mg_cochlea_input': float(cochlea_input.mean()),
            'mg_a1_output': a1_out_value,
            'mg_ventral': float(ventral_processed.mean()),
            'mg_dorsal': float(dorsal_processed.mean()),
            'mg_gating_efficiency': gating_factor,
        }


class MediodorsalNucleus(nn.Module):
    """
    内背侧核 (Mediodorsal Nucleus, MD)

    认知/情绪中继核团:
    - 接收: 杏仁核、海马体、嗅球
    - 投射: 前额叶皮层PFC
    - 功能: 认知整合、情绪调节、工作记忆

    参考: Goldman-Rakic & Porrino (1985) - MD-PFC连接
    """

    def __init__(
        self,
        input_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim

        # 认知信号处理
        self.cognitive_processor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh(),
        )

        # PFC输出
        self.pfc_output = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # 认知历史
        self.cognitive_history = []

    def forward(
        self,
        limbic_input: torch.Tensor,  # 杏仁核/海马输入
        working_memory_load: float = 0.5,
        trn_inhibition: float = 0.0,
    ) -> dict[str, float]:
        """
        MD认知中继

        Args:
            limbic_input: 边缘系统输入 [B, input_dim]
            working_memory_load: 工作记忆负荷 [0, 1]
            trn_inhibition: TRN抑制强度 [0, 1]

        Returns:
            cognitive_signal: 认知信号强度
            pfc_output: PFC输出
        """
        if limbic_input.dim() == 1:
            limbic_input = limbic_input.unsqueeze(0)

        # TRN门控调制
        gating_factor = 1.0 - trn_inhibition * 0.5  # MD较少受TRN抑制

        # 工作记忆调制
        wm_factor = 1.0 + 0.2 * working_memory_load

        # 认知信号处理
        processed = self.cognitive_processor(limbic_input * gating_factor * wm_factor)

        # PFC输出
        pfc_out = self.pfc_output(processed)
        pfc_out_value = float(pfc_out.squeeze())

        # 记录历史
        self.cognitive_history.append({
            'wm_load': working_memory_load,
            'output': pfc_out_value,
        })
        if len(self.cognitive_history) > 50:
            self.cognitive_history = self.cognitive_history[-50:]

        return {
            'md_cognitive_signal': float(processed.mean()),
            'md_pfc_output': pfc_out_value,
            'md_gating_efficiency': gating_factor,
        }


class ThalamicReticularNucleus(nn.Module):
    """
    丘脑网状核 (Thalamic Reticular Nucleus, TRN)

    GABA抑制性核团:
    - 包裹所有丘脑核团
    - 投射: 抑制所有丘脑relay核团
    - 功能: 感觉门控、注意力选择、睡眠振荡

    特点:
    - 纯GABA神经元 (抑制性)
    - 不投射到皮层 (只抑制丘脑)
    - 产生睡眠纺锤波振荡

    参考: Castellotti et al. (2023) - TRN振荡
    参考: Pinault (2004) - TRN解剖功能
    """

    def __init__(
        self,
        n_nuclei: int = 4,  # 控制的核团数量
        oscillation_freq: float = 13.0,  # Hz (纺锤波频率)
    ):
        super().__init__()
        self.n_nuclei = n_nuclei
        self.oscillation_freq = oscillation_freq

        # TRN振荡相位
        self.trn_phase = 0.0

        # 各核团门控强度
        self.register_buffer('gating_strength', torch.ones(n_nuclei))

        # 核团索引映射
        self.nucleus_index = {
            'VL': 0,  # 运动门控
            'LG': 1,  # 视觉门控
            'MG': 2,  # 听觉门控
            'MD': 3,  # 认知门控 (较少抑制)
        }

        # 注意力选择权重
        self.attention_weights = nn.Parameter(torch.ones(n_nuclei))

        # 睡眠状态
        self.sleep_spindle_active = False
        self.spindle_amplitude = 0.0

    def forward(
        self,
        attention_allocation: dict[str, float] = None,
        arousal_level: float = 0.5,
        sleep_stage: str = "awake",
        dt: float = 0.01,
    ) -> dict[str, float]:
        """
        TRN门控计算

        Args:
            attention_allocation: 各模态注意分配 {'visual': 0.7, 'auditory': 0.3}
            arousal_level: 唤醒水平 [0, 1]
            sleep_stage: 睡眠阶段 ("awake", "nrem1", "nrem2", "nrem3", "rem")
            dt: 时间步长(秒)

        Returns:
            trn_gating: 整体门控水平
            nucleus_inhibition: 各核团抑制强度
        """
        attention_allocation = attention_allocation or {
            'visual': 0.5,
            'auditory': 0.3,
            'motor': 0.2,
            'cognitive': 0.5,
        }

        # TRN振荡更新 (睡眠纺锤波)
        phase_increment = 2 * np.pi * self.oscillation_freq * dt
        self.trn_phase += phase_increment
        self.trn_phase %= (2 * np.pi)

        # 睡眠阶段影响
        if sleep_stage == "nrem2":
            # NREM2: 纺锤波活跃
            self.sleep_spindle_active = True
            spindle_envelope = 0.5 + 0.5 * np.sin(self.trn_phase)
            self.spindle_amplitude = spindle_envelope
        else:
            self.sleep_spindle_active = False
            self.spindle_amplitude = 0.0

        # 基础抑制水平
        # 高唤醒→低抑制 (更开放)
        # 低唤醒→高抑制 (更关闭)
        base_inhibition = 0.6 - arousal_level * 0.4

        # 睡眠纺锤波增强抑制
        base_inhibition += self.spindle_amplitude * 0.3

        # 各核团门控计算
        nucleus_inhibition = {}

        # VL (运动)
        motor_attention = attention_allocation.get('motor', 0.2)
        vl_inhibition = base_inhibition - motor_attention * 0.3
        vl_inhibition = np.clip(vl_inhibition, 0.0, 0.8)
        nucleus_inhibition['VL'] = vl_inhibition

        # LG (视觉)
        visual_attention = attention_allocation.get('visual', 0.5)
        lg_inhibition = base_inhibition - visual_attention * 0.4
        lg_inhibition = np.clip(lg_inhibition, 0.0, 0.9)
        nucleus_inhibition['LG'] = lg_inhibition

        # MG (听觉)
        auditory_attention = attention_allocation.get('auditory', 0.3)
        mg_inhibition = base_inhibition - auditory_attention * 0.35
        mg_inhibition = np.clip(mg_inhibition, 0.0, 0.85)
        nucleus_inhibition['MG'] = mg_inhibition

        # MD (认知) - 较少受TRN抑制
        cognitive_attention = attention_allocation.get('cognitive', 0.5)
        md_inhibition = (base_inhibition - cognitive_attention * 0.2) * 0.6
        md_inhibition = np.clip(md_inhibition, 0.0, 0.5)
        nucleus_inhibition['MD'] = md_inhibition

        # 更新门控强度
        for nucleus_name, inhibition in nucleus_inhibition.items():
            idx = self.nucleus_index.get(nucleus_name, 0)
            gating = 1.0 - inhibition
            self.gating_strength[idx] = gating

        return {
            'trn_gating': float(self.gating_strength.mean()),
            'trn_inhibition': base_inhibition,
            'trn_phase': self.trn_phase,
            'nucleus_inhibition': nucleus_inhibition,
            'sleep_spindle_active': self.sleep_spindle_active,
            'spindle_amplitude': self.spindle_amplitude,
        }


class ThalamicNucleiSystem(AbstractBrainRegion):
    """
    丘脑特异性核团系统

    整合所有丘脑中继核团:
    - VL: 运动中继
    - LG: 视觉中继
    - MG: 听觉中继
    - MD: 认知中继
    - TRN: 门控控制

    参考: Sherman & Guillery (2006) - Thalamus探索与探索
    """

    region_name: ClassVar[str] = "thalamic_nuclei"

    @classmethod
    def required_keys(cls) -> Set[str]:
        return set([
            "cerebellum_input", "retina_input", "cochlea_input",
            "limbic_input", "attention_allocation", "arousal_level",
            "sleep_stage"
        ])

    @classmethod
    def output_keys(cls) -> Set[str]:
        return set([
            "vl_motor_output", "lg_v1_output", "mg_a1_output",
            "md_pfc_output", "trn_gating", "thalamic_relay_status"
        ])

    def __init__(
        self,
        input_dim: int = 32,
        visual_dim: int = 64,
        auditory_dim: int = 32,
        event_bus=None,
    ):
        super().__init__()

        # 各特异性核团
        self.VL = VentrolateralNucleus(input_dim)
        self.LG = LateralGeniculateNucleus(visual_dim)
        self.MG = MedialGeniculateNucleus(auditory_dim)
        self.MD = MediodorsalNucleus(input_dim)
        self.TRN = ThalamicReticularNucleus()

        # 状态
        self.state = ThalamicNucleusState()

        # Event-driven registration
        self.event_bus = event_bus
        if event_bus is not None:
            event_bus.subscribe(
                "brain_update",
                self._handle_brain_update,
                priority=3,
                name="thalamic_nuclei",
            )

    def _handle_brain_update(self, event) -> dict[str, Any]:
        """Event-driven handler."""
        state = event.data.get("internal_state", {})

        result = self.step(
            cerebellum_input=state.get("cerebellum_output", np.zeros(32)),
            retina_input=state.get("visual_input", np.zeros(64)),
            cochlea_input=state.get("auditory_input", np.zeros(32)),
            limbic_input=state.get("limbic_output", np.zeros(32)),
            attention_allocation=state.get("attention_allocation", {}),
            arousal_level=state.get("arousal", 0.5),
            sleep_stage=state.get("sleep_stage", "awake"),
        )

        state["vl_motor_output"] = result["vl_motor_output"]
        state["lg_v1_output"] = result["lg_v1_output"]
        state["mg_a1_output"] = result["mg_a1_output"]
        state["md_pfc_output"] = result["md_pfc_output"]
        state["trn_gating"] = result["trn_gating"]

        return result

    def step(
        self,
        cerebellum_input: np.ndarray = None,
        retina_input: np.ndarray = None,
        cochlea_input: np.ndarray = None,
        limbic_input: np.ndarray = None,
        attention_allocation: dict[str, float] = None,
        arousal_level: float = 0.5,
        sleep_stage: str = "awake",
        basal_ganglia_input: float = 0.0,
        visual_attention: float = 0.5,
        auditory_attention: float = 0.3,
        working_memory_load: float = 0.5,
        eyes_closed: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """执行丘脑中继步"""

        # TRN门控计算
        trn_result = self.TRN(
            attention_allocation=attention_allocation,
            arousal_level=arousal_level,
            sleep_stage=sleep_stage,
        )

        # 获取各核团抑制
        nucleus_inhibition = trn_result['nucleus_inhibition']

        # VL运动中继
        if cerebellum_input is None:
            cerebellum_input = np.zeros(32)
        vl_result = self.VL(
            cerebellum_input=torch.FloatTensor(cerebellum_input),
            basal_ganglia_input=basal_ganglia_input,
            trn_inhibition=nucleus_inhibition['VL'],
        )

        # LG视觉中继
        if retina_input is None:
            retina_input = np.zeros(64)
        lg_result = self.LG(
            retina_input=torch.FloatTensor(retina_input),
            visual_attention=visual_attention,
            trn_inhibition=nucleus_inhibition['LG'],
            eyes_closed=eyes_closed,
        )

        # MG听觉中继
        if cochlea_input is None:
            cochlea_input = np.zeros(32)
        mg_result = self.MG(
            cochlea_input=torch.FloatTensor(cochlea_input),
            auditory_attention=auditory_attention,
            trn_inhibition=nucleus_inhibition['MG'],
        )

        # MD认知中继
        if limbic_input is None:
            limbic_input = np.zeros(32)
        md_result = self.MD(
            limbic_input=torch.FloatTensor(limbic_input),
            working_memory_load=working_memory_load,
            trn_inhibition=nucleus_inhibition['MD'],
        )

        # 更新状态
        self.state.vl_motor_signal = vl_result['vl_motor_signal']
        self.state.vl_motor_output = vl_result['vl_motor_output']
        self.state.lg_visual_signal = lg_result['lg_visual_signal']
        self.state.lg_v1_output = lg_result['lg_v1_output']
        self.state.mg_auditory_signal = mg_result['mg_auditory_signal']
        self.state.mg_a1_output = mg_result['mg_a1_output']
        self.state.md_cognitive_signal = md_result['md_cognitive_signal']
        self.state.md_pfc_output = md_result['md_pfc_output']
        self.state.trn_gating = trn_result['trn_gating']
        self.state.trn_inhibition = trn_result['trn_inhibition']

        return {
            'vl_motor_output': vl_result['vl_motor_output'],
            'vl_relay_efficiency': vl_result['vl_relay_efficiency'],
            'lg_v1_output': lg_result['lg_v1_output'],
            'lg_gating_efficiency': lg_result['lg_gating_efficiency'],
            'lg_parvocellular': lg_result['lg_parvocellular'],
            'lg_magnocellular': lg_result['lg_magnocellular'],
            'mg_a1_output': mg_result['mg_a1_output'],
            'mg_gating_efficiency': mg_result['mg_gating_efficiency'],
            'md_pfc_output': md_result['md_pfc_output'],
            'trn_gating': trn_result['trn_gating'],
            'trn_inhibition': trn_result['trn_inhibition'],
            'trn_phase': trn_result['trn_phase'],
            'sleep_spindle_active': trn_result['sleep_spindle_active'],
            'thalamic_relay_status': {
                'motor': vl_result['vl_relay_efficiency'],
                'visual': lg_result['lg_gating_efficiency'],
                'auditory': mg_result['mg_gating_efficiency'],
                'cognitive': md_result['md_gating_efficiency'],
            },
        }

    def get_summary(self) -> dict[str, Any]:
        """获取丘脑核团状态摘要"""
        return {
            'VL': {
                'motor_signal': self.state.vl_motor_signal,
                'motor_output': self.state.vl_motor_output,
            },
            'LG': {
                'visual_signal': self.state.lg_visual_signal,
                'v1_output': self.state.lg_v1_output,
            },
            'MG': {
                'auditory_signal': self.state.mg_auditory_signal,
                'a1_output': self.state.mg_a1_output,
            },
            'MD': {
                'cognitive_signal': self.state.md_cognitive_signal,
                'pfc_output': self.state.md_pfc_output,
            },
            'TRN': {
                'gating': self.state.trn_gating,
                'inhibition': self.state.trn_inhibition,
            },
        }


__all__ = [
    'VentrolateralNucleus',
    'LateralGeniculateNucleus',
    'MedialGeniculateNucleus',
    'MediodorsalNucleus',
    'ThalamicReticularNucleus',
    'ThalamicNucleiSystem',
    'ThalamicNucleusState',
]