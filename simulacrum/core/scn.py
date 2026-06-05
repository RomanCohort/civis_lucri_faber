"""
视交叉上核 (Suprachiasmatic Nucleus, SCN)

位于下丘脑上方的双核团，是哺乳动物的昼夜节律起搏器。

核心生物学通路：
视网膜 → 视网膜下丘脑束 (RHT) → SCN → 松果体 → 褪黑素

SCN内部机制：
1. 分子钟：转录-翻译反馈环 (TTFL)
   - CLOCK/BMAL1 → 驱动PER/CRY转录
   - PER/CRY积累 → 抑制CLOCK/BMAL1 → 周期重启
   - 自然周期约24.2小时（略长于24h，需光照校准）

2. 光感受入：
   - 视网膜含黑视蛋白的ipRGC细胞 → 直接投射SCN
   - 蓝光(460-480nm)最有效
   - 光脉冲可相位前移（清晨）或相位后移（傍晚）

3. 松果体控制：
   - SCN → PVN(室旁核) → IML(中间外侧柱) → 颈上神经节 → 松果体
   - 夜间：交感神经激活 → NE释放 → β-肾上腺素受体 → NAT酶激活 → 褪黑素合成
   - 白天：SCN抑制该通路 → 褪黑素接近零

4. 输出信号：
   - 睡眠-觉醒门控
   - 体温节律（夜间下降）
   - 皮质醇晨峰（CAR: Cortisol Awakening Response）
   - 认知性能节律（午后低谷）

核心类：
1. MolecularClock - 分子钟机制
2. PinealGland - 松果体褪黑素合成
3. SuprachiasmaticNucleus - 完整SCN系统
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch.nn as nn


class LightType(Enum):
    """光照类型"""
    DARKNESS = 0       # 黑暗
    DIM = 1            # 昏暗（室内灯光）
    INDOOR = 2         # 室内正常照明
    OUTDOOR = 3        # 户外自然光
    BRIGHT_BLUE = 4    # 强蓝光（电子屏幕）


@dataclass
class MolecularClockState:
    """
    分子钟状态

    对应TTFL（转录-翻译反馈环）的核心组分：
    - per_cry: PER/CRY蛋白水平（抑制性臂）
    - clock_bmal: CLOCK/BMAL1活性（驱动性臂）
    - period: 当前内源性周期（小时）
    """
    per_cry: float = 0.0          # PER/CRY蛋白积累 [0, 1]
    clock_bmal: float = 1.0       # CLOCK/BMAL1转录活性 [0, 1]
    period: float = 24.2          # 内源性周期（小时，人类约24.2h）
    phase: float = 0.0            # 当前相位 [0, 2π]
    amplitude: float = 1.0        # 振幅（光照强化，黑暗衰减）


@dataclass
class CircadianOutput:
    """SCN输出信号"""
    melatonin: float = 0.0        # 松果体褪黑素水平 [0, 1]
    core_temperature: float = 37.0  # 核心体温 (°C)
    cortisol_rhythm: float = 0.3   # 皮质醇节律分量 [0, 1]
    alertness: float = 0.5         # 认知警觉度 [0, 1]
    sleep_pressure: float = 0.0    # 睡眠压力（Process S） [0, 1]
    wake_drive: float = 0.5        # 觉醒驱动（Process C） [0, 1]
    chronotype: str = "intermediate"  # 时型：morning/intermediate/evening


class MolecularClock:
    """
    分子钟：转录-翻译反馈环

    简化的van der Pol振荡器模型：
    dx/dt = (π/12) * [A * cos(2π*t/τ) + γ * x]

    参数：
        intrinsic_period: 内源性周期（小时）
        amplitude: 振荡振幅
        damping: 阻尼系数（无光照时振幅衰减速率）
    """

    def __init__(
        self,
        intrinsic_period: float = 24.2,
        amplitude: float = 1.0,
        damping: float = 0.02,
    ):
        self.state = MolecularClockState(
            period=intrinsic_period,
            amplitude=amplitude,
        )
        self.damping = damping
        self._step_size = 1.0 / 60.0  # 每步 = 1分钟

    def tick(self, light_intensity: float = 0.0) -> float:
        """
        推进分子钟一个时间步

        Args:
            light_intensity: 光照强度 [0, 1]

        Returns:
            当前相位 [0, 2π]
        """
        s = self.state
        dt = self._step_size  # 小时

        # PER/CRY积累和降解（负反馈）
        # 高CLOCK/BMAL1 → PER/CRY增加
        # PER/CRY积累 → 抑制CLOCK/BMAL1
        per_cry_rate = s.clock_bmal * 0.8 - s.per_cry * 0.5
        s.per_cry += per_cry_rate * dt * 60  # 缩放到分钟步长
        s.per_cry = np.clip(s.per_cry, 0.0, 1.0)

        # CLOCK/BMAL1受PER/CRY抑制
        bmal_rate = (1.0 - s.per_cry) * 0.6 - s.clock_bmal * 0.3
        s.clock_bmal += bmal_rate * dt * 60
        s.clock_bmal = np.clip(s.clock_bmal, 0.0, 1.0)

        # 相位推进（基于内源性周期）
        phase_rate = 2 * np.pi / (s.period * 60)  # 每分钟相位增量
        s.phase += phase_rate

        # 振幅维持/衰减: 连续sigmoid替代硬阈值
        # 光照是SCN最强的授时因子, 低光照时振幅衰减
        light_factor = 1.0 / (1.0 + np.exp(-15.0 * (light_intensity - 0.05)))
        s.amplitude = np.clip(
            s.amplitude + light_factor * 0.002 - self.damping * dt * (1.0 - light_factor),
            0.3, 1.0
        )

        # 相位保持在 [0, 2π]
        s.phase = s.phase % (2 * np.pi)

        return s.phase

    def apply_light_pulse(
        self,
        light_intensity: float,
        circadian_hour: float,
    ) -> float:
        """
        光脉冲相位响应（PRC: Phase Response Curve）

        对应生物学：
        - 清晨光照 → 相位前移（advance，生物钟提前）
        - 傍晚光照 → 相位后移（delay，生物钟推后）
        - 夜间中期光照 → 相位后移（最大延迟）

        Args:
            light_intensity: 光照强度 [0, 1]
            circadian_hour: 昼夜时（0=midnight, 12=noon, 24=midnight）

        Returns:
            相位偏移量（弧度）
        """
        # PRC曲线（简化）
        if 4 <= circadian_hour < 10:
            # 清晨：相位前移（advance zone）
            shift = -0.05 * light_intensity * np.sin(
                (circadian_hour - 4) / 6 * np.pi
            )
        elif 16 <= circadian_hour < 22:
            # 傍晚/夜间早期：相位后移（delay zone）
            shift = 0.04 * light_intensity * np.sin(
                (circadian_hour - 16) / 6 * np.pi
            )
        else:
            # 其他时间：小或无响应
            shift = 0.01 * light_intensity

        self.state.phase += shift
        return shift


class PinealGland:
    """
    松果体 (Pineal Gland)

    褪黑素合成通路：
    色氨酸 → 5-HT(血清素) → NAT酶激活 → 褪黑素

    SCN控制路径：
    SCN(夜间激活) → PVN → IML → 颈上神经节 → NE释放
    → β-肾上腺素受体 → cAMP ↑ → NAT酶激活 → 褪黑素合成 ↑

    白天：SCN通过GABA抑制该通路 → 褪黑素≈0
    夜间：SCN解除抑制 → NE释放 → 褪黑素合成

    参数：
        synthesis_rate: 合成速率
        degradation_rate: 降解速率（褪黑素半衰期约20-50分钟）
        max_output: 最大褪黑素输出
    """

    def __init__(
        self,
        synthesis_rate: float = 0.1,
        degradation_rate: float = 0.05,
        max_output: float = 1.0,
    ):
        self.synthesis_rate = synthesis_rate
        self.degradation_rate = degradation_rate
        self.max_output = max_output

        self.melatonin_level = 0.0
        self.serotonin_precursor = 0.6  # 色氨酸前体储备

    def synthesize(self, scn_night_signal: float, light_suppression: float) -> float:
        """
        合成褪黑素

        Args:
            scn_night_signal: SCN夜间信号 [0, 1]（夜间高，白天低）
            light_suppression: 光照抑制因子 [0, 1]（光照越强，抑制越强）

        Returns:
            褪黑素水平 [0, 1]
        """
        # 光照直接抑制褪黑素（即使在夜间，光照也能快速抑制）
        effective_signal = scn_night_signal * (1.0 - light_suppression * 0.9)

        # NAT酶活性受SCN夜间信号驱动
        nat_activity = effective_signal * self.synthesis_rate * self.serotonin_precursor

        # 褪黑素合成
        self.melatonin_level += nat_activity

        # 褪黑素降解（半衰期约30分钟）
        self.melatonin_level -= self.degradation_rate * self.melatonin_level

        # 血清素前体消耗和补充
        self.serotonin_precursor -= nat_activity * 0.1
        self.serotonin_precursor += 0.01  # 持续补充
        self.serotonin_precursor = np.clip(self.serotonin_precursor, 0.2, 1.0)

        self.melatonin_level = np.clip(self.melatonin_level, 0.0, self.max_output)

        return self.melatonin_level


class SuprachiasmaticNucleus(nn.Module):
    """
    视交叉上核 (SCN) - 昼夜节律起搏器

    完整通路：
    光 → ipRGC → RHT → SCN分子钟 → PVN → 松果体 → 褪黑素

    SCN输出：
    1. 褪黑素节律（松果体控制）
    2. 体温节律（夜间下降约0.5-1°C）
    3. 皮质醇晨峰（CAR，觉醒后30-45分钟达峰）
    4. 认知警觉度节律（午后低谷约14:00-16:00）
    5. 睡眠压力 vs 觉醒驱动（Process S × Process C）

    参数：
        intrinsic_period: 内源性周期（小时）
        chronotype_offset: 时型偏移（>0=晨型人，<0=夜型人）
    """

    def __init__(
        self,
        intrinsic_period: float = 24.2,
        chronotype_offset: float = 0.0,
    ):
        super().__init__()

        # 子系统
        self.molecular_clock = MolecularClock(
            intrinsic_period=intrinsic_period,
        )
        self.pineal = PinealGland()

        # 时型（chronotype）
        self.chronotype_offset = chronotype_offset
        if chronotype_offset > 0.5:
            self.chronotype = "morning"
        elif chronotype_offset < -0.5:
            self.chronotype = "evening"
        else:
            self.chronotype = "intermediate"

        # 光照历史（用于计算近日光照总量）
        self._light_history: list[float] = []
        self._max_light_history = 60  # 最近60步

        # 状态
        self.output = CircadianOutput(chronotype=self.chronotype)
        self._step_count = 0

        # Process S（睡眠压力，腺苷积累）
        self._sleep_pressure = 0.0
        self._adenosine_level = 0.0

    def _compute_circadian_hour(self) -> float:
        """
        将分子钟相位转换为昼夜时

        相位0 → CT0 (midnight)
        相位π → CT12 (noon)
        """
        hour = (self.molecular_clock.state.phase / (2 * np.pi)) * 24.0
        hour = (hour + self.chronotype_offset) % 24.0
        return hour

    def _compute_night_signal(self, circadian_hour: float) -> float:
        """
        SCN夜间信号（控制松果体的信号）— 连续sigmoid过渡

        生物学：SCN在主观夜间"解除"对松果体的抑制
        使用余弦包络实现平滑的日夜过渡
        """
        night_start = 19.0 + self.chronotype_offset
        night_end = 7.0 + self.chronotype_offset
        if night_start > 24: night_start -= 24
        if night_end > 24: night_end -= 24

        # 计算到夜间窗口中心的角度
        night_mid = (night_start + 6.0) % 24
        diff = circadian_hour - night_mid
        if diff > 12: diff -= 24
        elif diff < -12: diff += 24

        # 余弦包络: center=1.0, 边缘平滑过渡到~0.1
        cosine = np.cos(np.pi * diff / 12.0)
        signal = 0.45 * (cosine + 1.0)  # [0, 0.9]
        return max(0.1, signal)

    def _compute_temperature_rhythm(self, circadian_hour: float) -> float:
        """
        核心体温节律

        生物学：
        - 最低温（nadir）约在04:00-05:00
        - 最高温约在18:00-19:00
        - 振幅约0.5-1.0°C
        """
        # 最低温在CT4-5
        temp_nadir = 36.4 + self.chronotype_offset * 0.1
        temp_peak = 37.2 + self.chronotype_offset * 0.1
        temp_mean = (temp_nadir + temp_peak) / 2
        temp_amplitude = (temp_peak - temp_nadir) / 2

        # 余弦函数：nadir在CT5，peak在CT17
        temp = temp_mean - temp_amplitude * np.cos(
            2 * np.pi * (circadian_hour - 5) / 24
        )
        return temp

    def _compute_cortisol_rhythm(self, circadian_hour: float) -> float:
        """
        皮质醇昼夜节律

        生物学：
        - 晨峰（CAR）：觉醒后30-45分钟达峰
        - 日间逐渐下降
        - 午夜低谷（nadir约00:00-02:00）
        """
        # 皮质醇在CT6-8达峰（觉醒时），CT0-2最低
        cortisol = 0.3 + 0.5 * np.exp(
            -((circadian_hour - 7.5) ** 2) / (2 * 3.0 ** 2)
        )
        # 午后小峰
        cortisol += 0.1 * np.exp(
            -((circadian_hour - 16) ** 2) / (2 * 1.5 ** 2)
        )
        return np.clip(cortisol, 0.05, 1.0)

    def _compute_alertness(self, circadian_hour: float) -> float:
        """
        认知警觉度节律

        生物学：
        - 上午高峰（09:00-11:00）
        - 午后低谷（14:00-16:00，post-lunch dip）
        - 晚间高峰（18:00-20:00，wake maintenance zone）
        - 夜间急剧下降
        """
        # 基础节律
        alertness = 0.5

        # 上午峰
        alertness += 0.3 * np.exp(
            -((circadian_hour - 10) ** 2) / (2 * 2 ** 2)
        )

        # 午后低谷
        alertness -= 0.2 * np.exp(
            -((circadian_hour - 15) ** 2) / (2 * 1.5 ** 2)
        )

        # 晚间维持区
        alertness += 0.2 * np.exp(
            -((circadian_hour - 19) ** 2) / (2 * 1.5 ** 2)
        )

        return np.clip(alertness, 0.1, 1.0)

    def _update_sleep_pressure(self, is_awake: bool = True):
        """
        Process S：睡眠压力（腺苷积累）

        觉醒时：腺苷逐渐积累 → 睡眠压力上升
        睡眠时：腺苷清除 → 睡眠压力下降
        """
        if is_awake:
            self._adenosine_level += 0.002  # 每步（分钟）积累
            self._adenosine_level = min(1.0, self._adenosine_level)
        else:
            # NREM3清除最快，REM较慢
            self._adenosine_level *= 0.98  # 缓慢清除

        self._sleep_pressure = self._adenosine_level

    def step(
        self,
        light_input: float = 0.3,
        light_type: LightType = LightType.INDOOR,
        is_awake: bool = True,
        elapsed_minutes: float = 1.0,
    ) -> CircadianOutput:
        """
        SCN完整步进

        通路：光 → ipRGC → SCN分子钟 → 松果体 → 褪黑素 + 其他输出

        Args:
            light_input: 光照强度 [0, 1]
            light_type: 光照类型
            is_awake: 是否清醒
            elapsed_minutes: 经过分钟数

        Returns:
            CircadianOutput: 所有昼夜输出信号
        """
        self._step_count += 1

        # 1. 光照预处理（蓝光加权）
        # 蓝光(460-480nm)对SCN最有效
        blue_weight = {
            LightType.DARKNESS: 0.0,
            LightType.DIM: 0.3,
            LightType.INDOOR: 0.5,
            LightType.OUTDOOR: 0.7,
            LightType.BRIGHT_BLUE: 1.0,
        }.get(light_type, 0.5)
        effective_light = light_input * blue_weight

        # 2. 分子钟推进
        circadian_hour = self._compute_circadian_hour()
        phase = self.molecular_clock.tick(effective_light)

        # 3. 光相位响应（PRC）
        self.molecular_clock.apply_light_pulse(effective_light, circadian_hour)

        # 4. 松果体褪黑素合成
        night_signal = self._compute_night_signal(circadian_hour)
        light_suppression = effective_light
        melatonin = self.pineal.synthesize(night_signal, light_suppression)

        # 5. 体温节律
        temperature = self._compute_temperature_rhythm(circadian_hour)

        # 6. 皮质醇节律
        cortisol_rhythm = self._compute_cortisol_rhythm(circadian_hour)

        # 7. 认知警觉度
        alertness = self._compute_alertness(circadian_hour)
        # 高褪黑素降低警觉
        alertness *= (1.0 - melatonin * 0.5)

        # 8. Process S
        self._update_sleep_pressure(is_awake)

        # 9. 觉醒驱动（Process C，SCN驱动）
        wake_drive = 0.5 + 0.4 * np.cos(
            2 * np.pi * (circadian_hour - 10) / 24  # 峰值在CT10
        )

        # 10. 光照历史
        self._light_history.append(effective_light)
        if len(self._light_history) > self._max_light_history:
            self._light_history.pop(0)

        # 更新输出
        self.output = CircadianOutput(
            melatonin=melatonin,
            core_temperature=temperature,
            cortisol_rhythm=cortisol_rhythm,
            alertness=alertness,
            sleep_pressure=self._sleep_pressure,
            wake_drive=wake_drive,
            chronotype=self.chronotype,
        )

        return self.output

    def get_circadian_hour(self) -> float:
        """获取当前昼夜时"""
        return self._compute_circadian_hour()

    def get_summary(self) -> dict:
        """获取SCN摘要"""
        return {
            'circadian_hour': round(self._compute_circadian_hour(), 2),
            'phase': round(self.molecular_clock.state.phase, 3),
            'period': round(self.molecular_clock.state.period, 2),
            'amplitude': round(self.molecular_clock.state.amplitude, 3),
            'melatonin': round(self.output.melatonin, 4),
            'temperature': round(self.output.core_temperature, 2),
            'cortisol_rhythm': round(self.output.cortisol_rhythm, 4),
            'alertness': round(self.output.alertness, 4),
            'sleep_pressure': round(self.output.sleep_pressure, 4),
            'wake_drive': round(self.output.wake_drive, 4),
            'chronotype': self.chronotype,
            'per_cry': round(self.molecular_clock.state.per_cry, 4),
            'clock_bmal': round(self.molecular_clock.state.clock_bmal, 4),
        }


def create_scn(
    intrinsic_period: float = 24.2,
    chronotype_offset: float = 0.0,
) -> SuprachiasmaticNucleus:
    """创建视交叉上核系统"""
    return SuprachiasmaticNucleus(
        intrinsic_period=intrinsic_period,
        chronotype_offset=chronotype_offset,
    )


__all__ = [
    'LightType',
    'MolecularClockState',
    'CircadianOutput',
    'MolecularClock',
    'PinealGland',
    'SuprachiasmaticNucleus',
    'create_scn',
]
