"""
节律系统 (Rhythm System)

实现大脑的振荡节律:
1. Theta-gamma 节律耦合 - 海马和皮层的节律同步
2. 丘脑 ACh 驱动门控机制 - 注意力调制
3. Alpha/beta/gamma 节律 - 多频振荡
4. Alpha节律 (8-12Hz) - 闭眼抑制机制 (新增)

参考:
- Buzsáki (2002) - Rhythms of the brain
- Jensen & Tesche (2002) - Theta-gamma coupling in hippocampus
- Hirsch et al. (2018) - Thalamic cholinergic gating
- Berger (1929) - Alpha节律首次发现
- Pfurtscheller & Lopes da Silva (1999) - Alpha事件相关去同步(ERD)
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════
# Alpha节律 (8-12Hz) - 闭眼抑制机制
# ══════════════════════════════════════════════════════

@dataclass
class AlphaConfig:
    """Alpha节律配置

    参考: Berger (1929), Pfurtscheller & Lopes da Silva (1999)
    - 频率: 8-12 Hz (经典alpha)
    - 拓扑: 后部优势 (枕叶/顶叶)
    - 特性: 闭眼增强，睁眼抑制 (alpha blocking)
    - 功能: 放松状态、注意力空闲、视觉门控
    """
    frequency_hz: float = 10.0         # Hz (8-12)
    amplitude: float = 0.7             # 相对振幅 [0, 1]
    eyes_closed_boost: float = 0.4     # 闭眼增强因子
    attention_suppression: float = 0.5  # 注意力抑制因子
    posterior_dominance: float = 0.8   # 后部优势程度


class AlphaRhythmGenerator(nn.Module):
    """Alpha节律生成器 (8-12Hz)

    生物学基础:
    - 丘脑-皮层回路振荡
    - 后部皮层(枕叶/顶叶)优势分布
    - "闲散"状态标记：不专注于视觉任务时增强
    - Alpha blocking: 睁眼或视觉注意时抑制

    功能意义:
    1. 视觉门控: Alpha高→视觉输入抑制
    2. 放松指标: 高alpha→放松状态
    3. 注意调制: 注意力→alpha抑制(ERD)
    4. 内向思考: alpha与DMN耦合

    参考:
    - Berger (1929): Alpha节律首次记录
    - Pfurtscheller & Lopes da Silva (1999): Alpha ERD/ERS
    - Klimesch et al. (1999): Alpha与认知负荷
    - Foxe & Snyder (2011): Alpha与视觉注意
    """

    def __init__(
        self,
        config: AlphaConfig | None = None,
        n_alpha_sources: int = 4,  # 不同脑区的alpha源
    ):
        super().__init__()
        self.config = config or AlphaConfig()
        self.n_sources = n_alpha_sources

        # Alpha振荡状态
        self.alpha_phase = 0.0
        self.alpha_frequency = self.config.frequency_hz
        self.alpha_amplitude = self.config.amplitude

        # 眼睛状态 (闭眼/睁眼)
        self.eyes_closed = False

        # 注意力状态
        self.attention_level = 0.5

        # 拓扑分布 (不同脑区alpha强度)
        # 参考: posterior dominance
        self.alpha_sources = {
            'occipital': {'base_amp': 0.85, 'current_amp': 0.85},  # 枕叶 - 最强
            'parietal': {'base_amp': 0.70, 'current_amp': 0.70},   # 顶叶
            'temporal': {'base_amp': 0.40, 'current_amp': 0.40},   # 颞叶
            'frontal': {'base_amp': 0.20, 'current_amp': 0.20},    # 额叶 - 最弱
        }

        # Alpha历史追踪
        self.alpha_history: deque = deque(maxlen=100)
        self.erd_ers_history: deque = deque(maxlen=50)

        # 统计
        self.total_alpha_time = 0.0
        self.erd_count = 0  # Event-related desynchronization事件
        self.ers_count = 0  # Event-related synchronization事件

    def step(
        self,
        dt: float = 0.01,
        eyes_closed: bool = False,
        visual_attention: float = 0.5,
        cognitive_load: float = 0.3,
        internal_thought: float = 0.0,  # 内向思考程度
    ) -> dict[str, float]:
        """执行Alpha节律步骤

        Args:
            dt: 时间步长(秒)
            eyes_closed: 是否闭眼
            visual_attention: 视觉注意强度 [0, 1]
            cognitive_load: 认知负荷 [0, 1]
            internal_thought: 内向思考程度 (DMN相关)

        Returns:
            alpha_amplitude: 当前alpha振幅
            alpha_frequency: alpha频率
            alpha_blocking: 抑制程度 (ERD)
            visual_gating: 视觉门控程度
        """
        # 更新眼睛状态
        self.eyes_closed = eyes_closed
        self.attention_level = visual_attention

        # 更新Alpha相位
        phase_increment = 2 * np.pi * self.alpha_frequency * dt
        self.alpha_phase += phase_increment
        self.alpha_phase %= (2 * np.pi)

        # 计算Alpha振幅调制
        base_amplitude = self.config.amplitude

        # 1. 闭眼增强 (alpha blocking反效应)
        eyes_effect = self.config.eyes_closed_boost if eyes_closed else 0.0

        # 2. 视觉注意抑制 (ERD - Event Related Desynchronization)
        # 参考: Pfurtscheller & Lopes da Silva (1999)
        attention_effect = -self.config.attention_suppression * visual_attention

        # 3. 认知负荷抑制 (高负荷→alpha抑制)
        # 参考: Klimesch et al. (1999)
        cognitive_effect = -0.3 * cognitive_load

        # 4. 内向思考增强 (DMN耦合)
        # 参考: 内向思考时alpha增强
        internal_effect = 0.2 * internal_thought

        # 总调制
        modulation = eyes_effect + attention_effect + cognitive_effect + internal_effect

        # 更新振幅 (带平滑)
        target_amplitude = base_amplitude + modulation
        target_amplitude = np.clip(target_amplitude, 0.1, 1.0)
        self.alpha_amplitude = 0.9 * self.alpha_amplitude + 0.1 * target_amplitude

        # 拓扑分布更新
        for source_name, source_data in self.alpha_sources.items():
            # 后部区域更受视觉注意影响
            if source_name in ['occipital', 'parietal']:
                posterior_factor = self.config.posterior_dominance
                visual_factor = -0.5 * visual_attention * posterior_factor
            else:
                visual_factor = -0.2 * visual_attention

            # 闭眼对后部区域影响更大
            eyes_factor = self.config.eyes_closed_boost * 1.5 if (eyes_closed and source_name in ['occipital', 'parietal']) else 0.0

            target = source_data['base_amp'] + eyes_factor + visual_factor + internal_effect * 0.5
            target = np.clip(target, 0.05, 1.0)
            source_data['current_amp'] = 0.85 * source_data['current_amp'] + 0.15 * target

        # 计算当前振荡值
        alpha_oscillation = np.sin(self.alpha_phase) * self.alpha_amplitude

        # 视觉门控 (高alpha→抑制视觉输入)
        visual_gating = self.alpha_amplitude * 0.5  # alpha高→视觉门控强

        # ERD/ERS检测
        # 参考: 相对于基线的变化
        baseline_amp = self.config.amplitude
        current_ratio = self.alpha_amplitude / baseline_amp

        if current_ratio < 0.7:  # 显著抑制
            self.erd_count += 1
            erd_state = 'desynchronized'
        elif current_ratio > 1.3:  # 显著增强
            self.ers_count += 1
            erd_state = 'synchronized'
        else:
            erd_state = 'neutral'

        # 记录历史
        self.alpha_history.append({
            'amplitude': self.alpha_amplitude,
            'frequency': self.alpha_frequency,
            'eyes_closed': eyes_closed,
            'attention': visual_attention,
            'erd_state': erd_state,
        })

        self.erd_ers_history.append(current_ratio)
        self.total_alpha_time += dt

        return {
            'alpha_amplitude': self.alpha_amplitude,
            'alpha_frequency': self.alpha_frequency,
            'alpha_phase': self.alpha_phase,
            'alpha_blocking': 1.0 - self.alpha_amplitude / self.config.amplitude,  # 抑制程度
            'visual_gating': visual_gating,
            'erd_state': erd_state,
            'eyes_closed_boost': eyes_effect,
            'posterior_alpha': self.alpha_sources['occipital']['current_amp'],
        }

    def set_eyes_state(self, closed: bool):
        """设置眼睛状态"""
        self.eyes_closed = closed

    def get_alpha_stats(self) -> dict:
        """获取Alpha统计"""
        return {
            'current_amplitude': self.alpha_amplitude,
            'current_frequency': self.alpha_frequency,
            'eyes_closed': self.eyes_closed,
            'attention_level': self.attention_level,
            'erd_count': self.erd_count,
            'ers_count': self.ers_count,
            'total_alpha_time': self.total_alpha_time,
            'topology': {k: v['current_amp'] for k, v in self.alpha_sources.items()},
        }

    def get_visual_suppression(self) -> float:
        """获取视觉抑制程度

        高alpha→强视觉抑制
        """
        return self.alpha_amplitude * 0.5


# ══════════════════════════════════════════════════════
# Theta-gamma 节律耦合
# 参考: Jensen & Tesche (2002) - Theta-gamma coupling in hippocampus
# ══════════════════════════════════════════════════════

@dataclass
class ThetaGammaConfig:
    """Theta-gamma耦合配置参数

    参考: Buzsáki (2002), Jensen & Tesche (2002)
    - Theta频率: 4-10 Hz (海马主要节律)
    - Gamma频率: 30-100 Hz (皮层主要节律)
    - Coupling模式: Phase-amplitude coupling (PAC)
    """
    theta_freq: float = 7.0       # Hz
    gamma_freq_min: float = 30.0  # Hz
    gamma_freq_max: float = 80.0  # Hz
    coupling_strength: float = 0.5
    phase_locking: float = 0.3


class ThetaGammaCoupling(nn.Module):
    """Theta-gamma节律耦合网络

    核心:
    1. Theta节律振荡 (4-10 Hz) - 海马
    2. Gamma节律振荡 (30-100 Hz) - 皮层
    3. Phase-amplitude coupling - Theta相位调制Gamma振幅

    功能:
    - 海马-皮层节律同步
    - 时间编码 (gamma包络携带信息)
    - 注意力调制 (多巴胺调节耦合强度)

    参考:
    - Jensen & Tesche (2002): Theta-gamma coupling in hippocampus
    - Colgin et al. (2009): Hippocampal theta-gamma
    """

    def __init__(
        self,
        n_neurons: int = 64,
        config: ThetaGammaConfig | None = None,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config or ThetaGammaConfig()

        # Theta振荡器 (4-10 Hz)
        self.theta_phase = 0.0
        self.theta_freq = self.config.theta_freq
        self.theta_amplitude = 1.0

        # Gamma振荡器 (30-100 Hz)
        self.gamma_phase = 0.0
        self.gamma_freq = (self.config.gamma_freq_min + self.config.gamma_freq_max) / 2.0
        self.gamma_amplitude = 1.0

        # 耦合系数 (可学习)
        self.coupling_coef = nn.Parameter(torch.tensor(self.config.coupling_strength))

        # 神经元状态
        self.register_buffer('theta_activity', torch.zeros(n_neurons))
        self.register_buffer('gamma_activity', torch.zeros(n_neurons))

        # 节律历史
        self.theta_history: deque = deque(maxlen=100)
        self.gamma_history: deque = deque(maxlen=100)

        # 步数计数
        self.step_count = 0

    def step(self, dt: float = 0.01, dopamine_level: float = 0.5) -> dict:
        """执行一步节律更新

        Args:
            dt: 时间步长 (秒)
            dopamine_level: 多巴胺水平 [0, 1]

        Returns:
            rhythm_info: 节律信息
        """
        self.step_count += 1

        # 多巴胺调制: 高DA → 强耦合
        da_factor = 0.5 + dopamine_level * 0.5

        # Theta节律更新 (4-10 Hz)
        theta_increment = 2 * np.pi * self.theta_freq * dt
        self.theta_phase += theta_increment
        self.theta_phase %= (2 * np.pi)

        # Gamma节律更新 (30-100 Hz)
        gamma_increment = 2 * np.pi * self.gamma_freq * dt
        self.gamma_phase += gamma_increment
        self.gamma_phase %= (2 * np.pi)

        # Phase-amplitude coupling: Theta相位调制Gamma振幅
        # Phase in [-π, π]
        phase_wrapped = self.theta_phase - np.pi
        coupling_factor = np.cos(phase_wrapped) * self.coupling_coef.item() * da_factor
        self.gamma_amplitude = 1.0 + 0.5 * coupling_factor

        # 生成神经元活动
        # Theta活动: 全局振荡
        theta_activity = torch.sin(self.theta_phase) * self.theta_amplitude

        # Gamma活动: 局部振荡 + Theta调制
        gamma_activity = torch.sin(self.gamma_phase) * self.gamma_amplitude

        # 记录历史
        self.theta_history.append(theta_activity.mean().item())
        self.gamma_history.append(gamma_activity.mean().item())

        return {
            'theta_phase': self.theta_phase,
            'gamma_phase': self.gamma_phase,
            'theta_freq': self.theta_freq,
            'gamma_freq': self.gamma_freq,
            'gamma_amplitude': self.gamma_amplitude,
            'coupling_strength': self.coupling_coef.item() * da_factor,
        }

    def get_coupling_stats(self) -> dict:
        """获取耦合统计"""
        if len(self.theta_history) < 2:
            return {'theta_gamma_coupling': 0.0}

        # 计算相位-振幅耦合强度
        theta_mean = np.mean(list(self.theta_history))
        gamma_mean = np.mean(list(self.gamma_history))

        # 简化的耦合指标: 相位同步度
        theta_phase_wrapped = np.array([t % (2*np.pi) for t in self.theta_history])
        gamma_phase_wrapped = np.array([g % (2*np.pi) for g in self.gamma_history])

        # 相位差
        phase_diff = np.abs(theta_phase_wrapped - gamma_phase_wrapped)
        phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)

        coupling = np.exp(-phase_diff / (np.pi / 2))  # 0-1
        coupling_strength = coupling.mean() * self.coupling_coef.item()

        return {
            'theta_gamma_coupling': coupling_strength,
            'theta_mean': theta_mean,
            'gamma_mean': gamma_mean,
            'theta_gamma_phase_diff': phase_diff.mean(),
        }


# ══════════════════════════════════════════════════════
# 丘脑ACh驱动门控机制 (新增)
# 参考: Hirsch et al. (2018) - Thalamic cholinergic gating
# ══════════════════════════════════════════════════════

class ThalamicAChGating(nn.Module):
    """丘脑乙酰胆碱(ACh)驱动门控

    生物学基础:
    - 丘脑内侧核(MD)接收来自基底前脑的ACh
    - ACh调制丘脑门控，影响感觉信息传递
    - 高ACh → 高门控 → 注意力增强
    - 低ACh → 低门控 → 感觉过滤

    参考:
    - Hirsch et al. (2018): Cholinergic gating in thalamus
    - McCormick & Bal (1997): Thalamic relay cells

    功能:
    1. ACh信号处理 (Tonic + Phasic)
    2. 门控强度计算
    3. 感觉信息调制
    """

    def __init__(
        self,
        n_senses: int = 4,
        baseline_a_ch: float = 0.5,
        phasic_decay: float = 0.1,
        gating_threshold: float = 0.3,
    ):
        super().__init__()
        self.n_senses = n_senses
        self.baseline_a_ch = baseline_a_ch
        self.phasic_decay = phasic_decay
        self.gating_threshold = gating_threshold

        # ACh水平
        self.a_ch_tonic = baseline_a_ch  # 慢变化基线
        self.a_ch_phasic = 0.0  # 瞬时突发

        # 门控强度 (每个感觉通道)
        self.register_buffer('gating_strength', torch.ones(n_senses))

        # ACh历史
        self.a_ch_history: deque = deque(maxlen=50)

        # 突发检测
        self.burst_threshold = 0.8
        self.burst_detected = False
        self.burst_step_count = 0

    def inject_a_ch(self, level: float, phasic: bool = True):
        """
        注入ACh信号

        Args:
            level: ACh水平 [0, 1]
            phasic: 是否为phasic (突发) 信号
        """
        if phasic:
            # Phasic: 瞬时突发
            self.a_ch_phasic = level
        else:
            # Tonic: 慢变化
            self.a_ch_tonic = level

    def update(self, dt: float = 0.01) -> dict:
        """更新ACh状态

        Phasic衰减: 快速衰减
        Tonic变化: 慢速调整
        """
        # Phasic衰减
        self.a_ch_phasic *= (1.0 - self.phasic_decay)
        self.a_ch_phasic = max(0.0, self.a_ch_phasic)

        # Tonic缓慢漂移
        drift = (np.random.random() - 0.5) * 0.01
        self.a_ch_tonic = max(0.0, min(1.0, self.a_ch_tonic + drift))

        # 总ACh水平
        total_a_ch = self.a_ch_tonic + self.a_ch_phasic

        # 门控更新
        # 高ACh → 高门控
        target_gating = torch.sigmoid(
            torch.tensor((total_a_ch - self.gating_threshold) * 10.0)
        )
        self.gating_strength = self.gating_strength * 0.9 + target_gating * 0.1

        # 突发检测
        self.burst_detected = self.a_ch_phasic > self.burst_threshold
        if self.burst_detected:
            self.burst_step_count += 1
        else:
            self.burst_step_count = 0

        # 记录历史
        self.a_ch_history.append({
            'total': total_a_ch,
            'tonic': self.a_ch_tonic,
            'phasic': self.a_ch_phasic,
            'burst': self.burst_detected,
        })

        return {
            'total_a_ch': total_a_ch,
            'tonic': self.a_ch_tonic,
            'phasic': self.a_ch_phasic,
            'burst_detected': self.burst_detected,
            'avg_gating': self.gating_strength.mean().item(),
        }

    def modulate_input(
        self,
        sensory_input: torch.Tensor,
        sense_idx: int,
    ) -> torch.Tensor:
        """调制感觉输入

        Args:
            sensory_input: 感觉输入 [n_neurons]
            sense_idx: 感觉通道索引

        Returns:
            modulated: 调制后的输入
        """
        if sense_idx >= self.n_senses:
            return sensory_input

        gating = self.gating_strength[sense_idx].item()
        modulation = 0.5 + 0.5 * gating  # [0.5, 1.0]

        modulated = sensory_input * modulation
        return modulated

    def get_gating_stats(self) -> dict:
        """获取门控统计"""
        if not self.a_ch_history:
            return {
                'avg_gating': 0.5,
                'burst_count': 0,
            }

        recent = list(self.a_ch_history)[-10:]
        avg_gating = np.mean([h['total'] for h in recent])
        burst_count = sum(1 for h in recent if h['burst'])

        return {
            'avg_gating': avg_gating,
            'burst_count': burst_count,
            'total_a_ch': self.a_ch_tonic + self.a_ch_phasic,
        }


# ══════════════════════════════════════════════════════
# 综合节律系统
# ══════════════════════════════════════════════════════

class RhythmSystem(nn.Module):
    """综合节律系统

    整合:
    1. Theta-gamma耦合 (海马/皮层)
    2. 丘脑ACh门控 (注意调制)
    3. Alpha节律 (8-12Hz, 闭眼抑制)
    """

    def __init__(
        self,
        n_neurons: int = 64,
        n_senses: int = 4,
        enable_alpha: bool = True,
        event_bus=None,
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.n_senses = n_senses
        self.enable_alpha = enable_alpha

        # Theta-gamma耦合
        self.theta_gamma = ThetaGammaCoupling(n_neurons)

        # 丘脑ACh门控
        self.thalamic_gating = ThalamicAChGating(n_senses)

        # Alpha节律
        self.alpha_rhythm = AlphaRhythmGenerator() if enable_alpha else None

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "sensory_process",
                self._handle_sensory_process,
                priority=0,
                name="rhythm",
            )

    def _handle_sensory_process(self, event) -> dict:
        """Event-driven handler for sensory_process events."""
        state = event.data.get("internal_state", {})

        # 更新ACh (从internal_state)
        a_ch_level = state.get("acetylcholine", 0.5)
        is_phasic = state.get("acetylcholine_phasic", False)

        self.thalamic_gating.inject_a_ch(a_ch_level, phasic=is_phasic)
        self.thalamic_gating.update()

        # 更新Theta-gamma (从多巴胺)
        dopamine_level = state.get("dopamine_level", 0.5)
        rhythm_info = self.theta_gamma.step(dopamine_level=dopamine_level)

        # 更新Alpha (从视觉注意和认知状态)
        alpha_info = {}
        if self.enable_alpha and self.alpha_rhythm:
            alpha_info = self.alpha_rhythm.step(
                eyes_closed=state.get("eyes_closed", False),
                visual_attention=state.get("visual_attention", 0.5),
                cognitive_load=state.get("cognitive_load", 0.3),
                internal_thought=state.get("internal_thought", 0.0),
            )

        # 更新状态
        state["theta_phase"] = rhythm_info['theta_phase']
        state["gamma_phase"] = rhythm_info['gamma_phase']
        state["theta_gamma_coupling"] = rhythm_info['coupling_strength']
        state["thalamic_gating"] = self.thalamic_gating.get_gating_stats()['avg_gating']

        if alpha_info:
            state["alpha_amplitude"] = alpha_info['alpha_amplitude']
            state["alpha_blocking"] = alpha_info['alpha_blocking']

        return {
            **rhythm_info,
            **self.thalamic_gating.get_gating_stats(),
            **alpha_info,
        }

    def get_summary(self) -> dict:
        """获取系统摘要"""
        summary = {
            **self.theta_gamma.get_coupling_stats(),
            **self.thalamic_gating.get_gating_stats(),
        }
        if self.enable_alpha and self.alpha_rhythm:
            summary['alpha_stats'] = self.alpha_rhythm.get_alpha_stats()
        return summary


# ══════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════

def create_theta_gamma_coupling(**kwargs) -> ThetaGammaCoupling:
    """创建Theta-gamma耦合"""
    return ThetaGammaCoupling(**kwargs)


def create_thalamic_a_ch_gating(**kwargs) -> ThalamicAChGating:
    """创建丘脑ACh门控"""
    return ThalamicAChGating(**kwargs)


def create_rhythm_system(**kwargs) -> RhythmSystem:
    """创建综合节律系统"""
    return RhythmSystem(**kwargs)


__all__ = [
    'ThetaGammaConfig',
    'ThetaGammaCoupling',
    'ThalamicAChGating',
    'RhythmSystem',
    'AlphaConfig',
    'AlphaRhythmGenerator',
    'create_theta_gamma_coupling',
    'create_thalamic_a_ch_gating',
    'create_rhythm_system',
]
