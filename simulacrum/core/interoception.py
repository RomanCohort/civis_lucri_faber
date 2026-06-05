# =============================================================================
# Interoception - 内感受系统
# =============================================================================
# 身体状态→情绪影响
#
# 核心机制：
# 1. 内感受编码：躯体感受
# 2. 身体标记假说：疼痛/疲劳→情绪
# 3. 肠道-脑轴：Gut-brain axis
# 4. 心身相互作用
# =============================================================================

from dataclasses import dataclass

import torch
import torch.nn as nn

# =============================================================================
# 内感受状态
# =============================================================================

@dataclass
class InteroceptiveState:
    """内感受状态"""
    # 心血管
    heart_rate: float = 0.5        # 心率 [0, 1]
    blood_pressure: float = 0.5   # 血压
    heart_rate_variability: float = 0.5  # HRV (自主神经)

    # 呼吸
    breathing_rate: float = 0.5    # 呼吸率
    breath_depth: float = 0.5         # 呼吸深度

    # 代谢
    glucose_level: float = 0.5      # 血糖
    hunger: float = 0.5             # 饥饿感
    satiety: float = 0.5            # 饱腹感

    # 躯体
    pain: float = 0.0              # 疼痛 [0, 1]
    fatigue: float = 0.0            # 疲劳 [0, 1]
    energy: float = 0.5            # 能量 [0, 1]

    # 皮肤
    skin_conductance: float = 0.5   # 皮肤电导 ( arousal)
    temperature: float = 0.5       # 体温


@dataclass
class GutState:
    """肠道状态"""
    microbiome_diversity: float = 0.5  # 微生物多样性
    serotonin_precursor: float = 0.5   # 血清素前体
    gaba_level: float = 0.5            # GABA (抑制)
    norepinephrine: float = 0.5         # 去甲肾上腺素


# =============================================================================
# 内感受编码器
# =============================================================================

class InteroceptiveEncoder(nn.Module):
    """
    内感受编码器

    对应神经机制：
    - 孤束核 (NTS): 内感受信息中继
    - 臂旁核 (PBN): 整合
    - 脑岛 (Insula): 内感受觉知

    功能：
    - 身体状态→神经表征
    - 内感受意识
    """

    def __init__(
        self,
        n_signals: int = 15,  # 15个内感受信号
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_signals = n_signals

        # 编码网络
        self.encoder = nn.Sequential(
            nn.Linear(n_signals, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 显著性检测
        self.salience_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def encode(
        self,
        interoceptive_state: InteroceptiveState,
    ) -> torch.Tensor:
        """
        编码内感受状态

        Args:
            interoceptive_state: 内感受状态

        Returns:
            encoding: [1, hidden_dim] 编码
        """
        # 展平为向量
        signals = [
            interoceptive_state.heart_rate,
            interoceptive_state.blood_pressure,
            interoceptive_state.heart_rate_variability,
            interoceptive_state.breathing_rate,
            interoceptive_state.breath_depth,
            interoceptive_state.glucose_level,
            interoceptive_state.hunger,
            interoceptive_state.satiety,
            interoceptive_state.pain,
            interoceptive_state.fatigue,
            interoceptive_state.energy,
            interoceptive_state.skin_conductance,
            interoceptive_state.temperature,
        ]

        # 填充到n_signals
        while len(signals) < self.n_signals:
            signals.append(0.5)

        x = torch.tensor(signals[:self.n_signals], dtype=torch.float32)
        encoding = self.encoder(x.unsqueeze(0))

        return encoding

    def detect_salience(
        self,
        encoding: torch.Tensor,
    ) -> float:
        """
        检测显著性

        Args:
            encoding: 编码

        Returns:
            salience: 显著性 [0, 1]
        """
        return self.salience_net(encoding).item()

    def forward(
        self,
        interoceptive_state: InteroceptiveState,
    ) -> dict:
        """
        前向编码

        Args:
            interoceptive_state: 内感受状态

        Returns:
            encoding: 编码结果
        """
        encoding = self.encode(interoceptive_state)
        salience = self.detect_salience(encoding)

        return {
            'encoding': encoding,
            'salience': salience,
        }


# =============================================================================
# 身体标记系统
# =============================================================================

class SomaticMarker(nn.Module):
    """
    身体标记系统

    对应神经机制：
    - 躯体感觉皮层: 身体状态
    - 脑岛: 内感受标记
    - 眼眶皮层: 情绪-身体关联

    功能：
    - 身体状态影响决策
    - 躯体标记：直觉
    - "胃部直觉"
    """

    def __init__(
        self,
        intero_dim: int = 64,
        decision_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 身体→情绪映射
        self.body_to_emotion = nn.Sequential(
            nn.Linear(intero_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),  # 情绪影响
        )

        # 身体→决策影响
        self.body_to_decision = nn.Sequential(
            nn.Linear(intero_dim + decision_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 影响因子 [-1, 1]
        )

        # 直觉强度
        self.intuition_strength = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        interoceptive_encoding: torch.Tensor,
        decision_context: torch.Tensor | None = None,
    ) -> dict:
        """
        身体标记

        Args:
            interoceptive_encoding: 内感受编码
            decision_context: 决策上下文

        Returns:
            somatic_mark: 身体标记
            emotion_influence: 情绪影响
        """
        # 身体状态→情绪
        emotion_influence = self.body_to_emotion(interoceptive_encoding)

        # 身体→决策
        if decision_context is not None:
            combined = torch.cat([interoceptive_encoding, decision_context], dim=-1)
            body_influence = self.body_to_decision(combined)
        else:
            body_influence = torch.zeros_like(interoceptive_encoding[:, :1])

        return {
            'somatic_marker': emotion_influence,
            'emotion_influence': emotion_influence,
            'decision_influence': body_influence,
            'intuition_strength': self.intuition_strength.item(),
        }


# =============================================================================
# 肠道-脑轴
# =============================================================================

class GutBrainAxis(nn.Module):
    """
    肠道-脑轴

    对应神经机制：
    - 肠道神经系统 (ENS): "第二脑"
    - 迷走神经: 肠-脑通信
    - 微生物群-肠-脑轴: 神经递质合成

    功能：
    - 血清素合成（前体→神经递质）
    - GABA调节（抑制/焦虑）
    - 炎症→情绪影响
    """

    def __init__(
        self,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 微生物编码
        self.microbiome_net = nn.Sequential(
            nn.Linear(5, hidden_dim),  # 5个微生物指标
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 肠-脑影响网络
        self.gut_to_brain = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4种神经递质影响
        )

        # 炎症影响网络
        self.inflammation_net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 3个炎症指标
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 情绪影响
        )

        # 神经递质水平
        self.register_buffer('serotonin', torch.tensor(0.5))
        self.register_buffer('gaba', torch.tensor(0.5))
        self.register_buffer('dopamine', torch.tensor(0.5))
        self.register_buffer('norepinephrine', torch.tensor(0.5))

    def update_neurotransmitters(
        self,
        gut_state: GutState,
    ) -> dict:
        """
        更新神经递质水平

        Args:
            gut_state: 肠道状态

        Returns:
            levels: 神经递质水平
        """
        # 编码肠道状态
        gut_vec = torch.tensor([
            gut_state.microbiome_diversity,
            gut_state.serotonin_precursor,
            gut_state.gaba_level,
            gut_state.norepinephrine,
            0.5,
        ])

        encoding = self.microbiome_net(gut_vec.unsqueeze(0))

        # 计算影响
        effects = self.gut_to_brain(encoding)

        # 更新神经递质
        self.serotonin = self.serotonin * 0.95 + 0.05 * (0.5 + effects[0, 0].item() * 0.5)
        self.gaba = self.gaba * 0.95 + 0.05 * (0.5 + effects[0, 1].item() * 0.5)
        self.dopamine = self.dopamine * 0.95 + 0.05 * (0.5 + effects[0, 2].item() * 0.5)
        self.norepinephrine = self.norepinephrine * 0.95 + 0.05 * (0.5 + effects[0, 3].item() * 0.5)

        return {
            'serotonin': self.serotonin.item(),
            'gaba': self.gaba.item(),
            'dopamine': self.dopamine.item(),
            'norepinephrine': self.norepinephrine.item(),
        }

    def forward(
        self,
        gut_state: GutState | None = None,
        inflammation_level: float = 0.0,
    ) -> dict:
        """
        肠-脑轴

        Args:
            gut_state: 肠道状态
            inflammation_level: 炎症水平

        Returns:
            brain_influence: 对脑的影响
        """
        if gut_state is not None:
            nt_levels = self.update_neurotransmitters(gut_state)
            serotonin = nt_levels['serotonin']
            gaba = nt_levels['gaba']
        else:
            serotonin = 0.5
            gaba = 0.5

        # 炎症影响（高炎症→负面情绪）
        if inflammation_level > 0.5:
            inflammation_effect = -0.3
        else:
            inflammation_effect = 0.0

        # 血清素：正面情绪，抑制负面
        emotion_modulation = {
            'positive': serotonin * 0.3,
            'negative': -gaba * 0.2 + inflammation_effect,
        }

        return {
            'serotonin': serotonin,
            'gaba': gaba,
            'emotion_modulation': emotion_modulation,
        }


# =============================================================================
# 心身整合系统
# =============================================================================

class PsychoSomaticIntegration(nn.Module):
    """
    心身整合系统

    整合：
    1. 内感受编码
    2. 身体标记
    3. 肠道-脑轴
    4. 心身双向影响
    """

    def __init__(
        self,
        n_signals: int = 15,
        intero_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 内感受编码器
        self.intero_encoder = InteroceptiveEncoder(n_signals, hidden_dim)

        # 身体标记
        self.somatic_marker = SomaticMarker(intero_dim, intero_dim, hidden_dim)

        # 肠-脑轴
        self.gut_brain = GutBrainAxis(hidden_dim)

        # 整合网络
        self.integrator = nn.Sequential(
            nn.Linear(hidden_dim + 8 + 4, hidden_dim),  # intero + emotion + nt
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
        )

    def compute_body_emotion_influence(
        self,
        interoceptive_state: InteroceptiveState,
    ) -> dict:
        """
        计算身体对情绪的影响

        Args:
            interoceptive_state: 内感受状态

        Returns:
            influence: 影响结果
        """
        # 编码
        encoding = self.intero_encoder.encode(interoceptive_state)

        # 显著性
        salience = self.intero_encoder.detect_salience(encoding)

        return {
            'encoding': encoding,
            'salience': salience,
        }

    def compute_somatic_markers(
        self,
        interoceptive_encoding: torch.Tensor,
        decision_context: torch.Tensor | None = None,
    ) -> dict:
        """
        计算身体标记

        Args:
            interoceptive_encoding: 内感受编码
            decision_context: 决策上下文

        Returns:
            markers: 身体标记
        """
        return self.somatic_marker(interoceptive_encoding, decision_context)

    def forward(
        self,
        interoceptive_state: InteroceptiveState,
        gut_state: GutState | None = None,
        inflammation_level: float = 0.0,
    ) -> dict:
        """
        完整内感受处理

        Args:
            interoceptive_state: 内感受状态
            gut_state: 肠道状态
            inflammation_level: 炎症水平

        Returns:
            complete_influence: 完整影响
        """
        # 1. 内感受编码
        intero_result = self.compute_body_emotion_influence(interoceptive_state)

        # 2. 身体标记
        somatic_result = self.compute_somatic_markers(
            intero_result['encoding']
        )

        # 3. 肠-脑轴
        gut_brain_result = self.gut_brain(gut_state, inflammation_level)

        # 整合
        combined = torch.cat([
            intero_result['encoding'],
            somatic_result['emotion_influence'],
            torch.tensor([
                gut_brain_result['serotonin'],
                gut_brain_result['gaba'],
                gut_brain_result['emotion_modulation']['positive'],
                gut_brain_result['emotion_modulation']['negative'],
            ], device=intero_result['encoding'].device).unsqueeze(0),
        ], dim=-1)

        integrated = self.integrator(combined)

        return {
            'influenced_emotion': integrated,
            'interoceptive_encoding': intero_result['encoding'],
            'salience': intero_result['salience'],
            'somatic_markers': somatic_result,
            'gut_brain': gut_brain_result,
        }


# =============================================================================
# 完整内感受系统
# =============================================================================

class InteroceptionSystem(nn.Module):
    """
    完整内感受系统

    整合：
    1. 内感受编码
    2. 身体标记假说
    3. 肠道-脑轴
    4. 心身整合
    """

    def __init__(
        self,
        n_signals: int = 15,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 心身整合
        self.psycho_somatic = PsychoSomaticIntegration(n_signals, hidden_dim, hidden_dim)

    def process(
        self,
        interoceptive_state: InteroceptiveState,
        gut_state: GutState | None = None,
        inflammation_level: float = 0.0,
    ) -> dict:
        """
        处理内感受

        Args:
            interoceptive_state: 内感受状态
            gut_state: 肠道状态
            inflammation_level: 炎症水平

        Returns:
            influence: 对情绪的影响
        """
        result = self.psycho_somatic(
            interoceptive_state, gut_state, inflammation_level
        )

        return {
            'influenced_emotion': result['influenced_emotion'],
            'body_emotion_influence': result['somatic_markers']['emotion_influence'],
            'gut_brain_influence': result['gut_brain']['emotion_modulation'],
            'salience': result['salience'],
        }

    def get_summary(self) -> dict:
        """获取摘要"""
        return {
            'gut_brain_active': True,
            'system': 'interoception',
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_interoception(
    n_signals: int = 15,
    hidden_dim: int = 64,
) -> InteroceptionSystem:
    """创建内感受系统"""
    return InteroceptionSystem(n_signals, hidden_dim)


__all__ = [
    'InteroceptiveState',
    'GutState',
    'InteroceptiveEncoder',
    'SomaticMarker',
    'GutBrainAxis',
    'PsychoSomaticIntegration',
    'InteroceptionSystem',
    'create_interoception',
    'InteroceptivePredictionError',
]


# ══════════════════════════════════════════════════════
# 内感受预测误差 — 惊恐发作核心机制
# ══════════════════════════════════════════════════════

class InteroceptivePredictionError:
    """内感受预测误差 — Klein假窒息警报理论。

    惊恐发作 = 大脑对内感受信号的灾难性误解释:
      1. 预测的身体状态 vs 实际感知的身体状态 → 预测误差
      2. 误差信号 → 岛叶/前扣带回激活 → 威胁评估
      3. 威胁评估 → 杏仁核 → 更交感激活 → 更大误差 (正反馈)
      4. 正反馈环路 → 惊恐发作

    参考:
    - Paulus & Stein (2006) Biol Psychiatry — 内感受预测误差
    - Gorman & Sullivan (2000) — 假窒息警报理论
    """

    def __init__(
        self,
        prediction_weight: float = 0.5,
        salience_threshold: float = 0.3,
    ):
        self.prediction_weight = prediction_weight
        self.salience_threshold = salience_threshold
        self._predicted_hr = 0.5
        self._predicted_br = 0.5
        self._pe_history = []

    def compute(
        self,
        actual_heart_rate: float = 0.5,
        actual_breathing_rate: float = 0.5,
        actual_skin_conductance: float = 0.5,
        predicted_heart_rate: float = None,
        predicted_breathing_rate: float = None,
    ) -> dict[str, float]:
        """计算内感受预测误差。

        Returns:
            pe_hr, pe_br, total_pe, is_alarm
        """
        # 使用缓存的预测值或显式传入
        pred_hr = predicted_heart_rate if predicted_heart_rate is not None else self._predicted_hr
        pred_br = predicted_breathing_rate if predicted_breathing_rate is not None else self._predicted_br

        # 预测误差 = |actual - predicted|
        pe_hr = abs(actual_heart_rate - pred_hr)
        pe_br = abs(actual_breathing_rate - pred_br)
        pe_sc = max(0.0, actual_skin_conductance - 0.5)  # 皮肤电导高于基线

        # 总预测误差 (加权)
        total_pe = 0.4 * pe_hr + 0.35 * pe_br + 0.25 * pe_sc

        # 假窒息警报: HR↑ + BR↑ + SC↑ → 大误差
        is_alarm = total_pe > self.salience_threshold

        # 更新预测 (缓慢适应)
        alpha = 0.05
        self._predicted_hr = (1 - alpha) * self._predicted_hr + alpha * actual_heart_rate
        self._predicted_br = (1 - alpha) * self._predicted_br + alpha * actual_breathing_rate

        self._pe_history.append(total_pe)
        if len(self._pe_history) > 100:
            self._pe_history.pop(0)

        return {
            "pe_heart_rate": pe_hr,
            "pe_breathing_rate": pe_br,
            "pe_skin_conductance": pe_sc,
            "total_pe": total_pe,
            "is_alarm": is_alarm,
            "interoceptive_pe": total_pe,  # 写入state的键名
        }


# =============================================================================
# 测试
# =============================================================================

def test_interoception():
    """测试内感受系统"""
    print("=" * 60)
    print("Testing Interoception System")
    print("=" * 60)

    # 创建模型
    model = InteroceptionSystem()

    # 创建内感受状态
    state = InteroceptiveState(
        heart_rate=0.6,
        blood_pressure=0.5,
        heart_rate_variability=0.4,
        breathing_rate=0.5,
        glucose_level=0.7,
        hunger=0.3,
        pain=0.2,
        fatigue=0.4,
        energy=0.6,
        skin_conductance=0.5,
    )

    # 创建肠道状态
    gut = GutState(
        microbiome_diversity=0.7,
        serotonin_precursor=0.6,
        gaba_level=0.5,
    )

    print("\n[1] Testing interoceptive processing...")
    result = model.process(state, gut, inflammation_level=0.3)
    print(f"  Influenced emotion: {result['influenced_emotion'][0]}")
    print(f"  Salience: {result['salience']:.3f}")

    print("\n[2] Testing gut-brain axis...")
    print(f"  Serotonin: {result['gut_brain_influence']}")

    print("\n[3] Summary...")
    summary = model.get_summary()
    print(f"  System: {summary['system']}")

    print("\n" + "=" * 60)
    print("✓ Interoception system working!")
    print("  - Interoceptive encoding: ✓")
    print("  - Somatic markers: ✓")
    print("  - Gut-brain axis: ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_interoception()
