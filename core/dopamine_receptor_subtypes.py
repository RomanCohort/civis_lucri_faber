"""
多巴胺受体亚型扩展 (Dopamine Receptor Subtypes)

扩展完整的DA受体家族，实现药物选择性作用模拟。

受体亚型:
  D1 (D1R/D5R家族):
    - D1: 高表达于纹状体GABAergic MSN (直接通路)
    - D5: 皮层、海马、丘脑 (认知功能)

  D2 (D2R/D3R/D4R家族):
    - D2: 高表达于纹状体GABAergic MSN (间接通路)
          两种剪接变体: D2S (短) / D2L (长)
    - D3: limbic系统 (情绪、动机)
    - D4: PFC (注意力、认知)

生物学基础:
  - Seeman (2005): D2高/低亲和力态
  - Missale et al. (1998): DA受体家族综述
  - Beaulieu & Gainetdinov (2011): DA信号通路
"""

from dataclasses import dataclass
from typing import ClassVar, Set, Any

import numpy as np
import torch
import torch.nn as nn


# 受体亲和力参数 (Kd, nM)
# 参考: Seeman et al. (2005), Wang et al. (2008)
RECEPTOR_KD = {
    'D1': {'high': 0.5, 'low': 5.0},     # D1亲和力
    'D2': {'high': 2.0, 'low': 20.0},    # D2亲和力 (已存在)
    'D3': {'high': 1.0, 'low': 10.0},    # D3亲和力 (高亲和力)
    'D4': {'high': 5.0, 'low': 50.0},    # D4亲和力 (中等)
    'D5': {'high': 0.3, 'low': 3.0},     # D5亲和力 (最高)
}

# 受体密度分布 (相对表达比例)
RECEPTOR_DISTRIBUTION = {
    'striatum': {'D1': 0.4, 'D2': 0.35, 'D3': 0.1, 'D4': 0.05, 'D5': 0.1},
    'pfc': {'D1': 0.15, 'D2': 0.1, 'D3': 0.05, 'D4': 0.45, 'D5': 0.25},
    'limbic': {'D1': 0.1, 'D2': 0.15, 'D3': 0.55, 'D4': 0.15, 'D5': 0.05},
    'hippocampus': {'D1': 0.2, 'D2': 0.1, 'D3': 0.1, 'D4': 0.2, 'D5': 0.4},
    'thalamus': {'D1': 0.05, 'D2': 0.1, 'D3': 0.05, 'D4': 0.2, 'D5': 0.6},
}

# 受体功能映射
RECEPTOR_FUNCTIONS = {
    'D1': {
        'signaling': 'Gs → cAMP↑ → PKA激活',
        'effect': '兴奋性',
        'function': ['运动促进', '奖励学习', '工作记忆'],
        'target_region': ['纹状体MSN-D1', 'PFC', '海马'],
    },
    'D2': {
        'signaling': 'Gi → cAMP↓ → 抑制',
        'effect': '抑制性',
        'function': ['运动抑制', '奖励预测', '冲动控制'],
        'target_region': ['纹状体MSN-D2', 'PFC'],
    },
    'D3': {
        'signaling': 'Gi → cAMP↓ → 抑制',
        'effect': '抑制性',
        'function': ['情绪调节', '动机控制', '奖励敏感性'],
        'target_region': ['NAc shell', '嗅结节', '岛叶'],
    },
    'D4': {
        'signaling': 'Gi → cAMP↓ → 抑制',
        'effect': '抑制性',
        'function': ['注意力', '认知灵活性', '冲动抑制'],
        'target_region': ['PFC', '海马', '杏仁核'],
    },
    'D5': {
        'signaling': 'Gs → cAMP↑ → PKA激活',
        'effect': '兴奋性',
        'function': ['认知功能', '运动', '奖励'],
        'target_region': ['PFC', '海马', '丘脑'],
    },
}


@dataclass
class ReceptorOccupancyState:
    """受体占有率状态"""
    D1_occupancy: float = 0.0
    D2_occupancy: float = 0.0
    D3_occupancy: float = 0.0
    D4_occupancy: float = 0.0
    D5_occupancy: float = 0.0

    # D2剪接变体
    D2S_occupancy: float = 0.0  # 短型 (突触前)
    D2L_occupancy: float = 0.0  # 长型 (突触后)

    # 功能性输出
    net_excitatory_signal: float = 0.0  # D1/D5兴奋性
    net_inhibitory_signal: float = 0.0  # D2/D3/D4抑制性


class DopamineReceptor(nn.Module):
    """
    单一DA受体类

    Langmuir吸附方程计算占有率:
    occupancy = [DA] / ([DA] + Kd)
    """

    def __init__(
        self,
        receptor_type: str = 'D1',
        expression_density: float = 0.5,
    ):
        super().__init__()

        self.receptor_type = receptor_type
        self.density = expression_density

        # 亲和力参数
        kd = RECEPTOR_KD.get(receptor_type, {'high': 2.0, 'low': 20.0})
        self.Kd_high = kd['high']
        self.Kd_low = kd['low']

        # 受体功能属性
        self.is_excitatory = receptor_type in ['D1', 'D5']
        self.signaling_pathway = RECEPTOR_FUNCTIONS.get(receptor_type, {}).get('signaling', '')

        # 内部状态追踪
        self.current_occupancy = 0.0
        self.activation_history = []

    def forward(
        self,
        da_concentration_nM: float,
        antipsychotic_block: float = 0.0,
        use_high_affinity: bool = True,
    ) -> dict[str, float]:
        """
        计算受体占有率

        Args:
            da_concentration_nM: 多巴胺浓度 (nM)
            antipsychotic_block: 抗精神病药物阻断率 [0, 1]
            use_high_affinity: 使用高亲和力态

        Returns:
            occupancy: 受体占有率 [0, 1]
            effective_signal: 有效信号强度
            activation_state: 激活状态描述
        """
        # 选择Kd
        Kd = self.Kd_high if use_high_affinity else self.Kd_low

        # Langmuir占有率
        occupancy = da_concentration_nM / (da_concentration_nM + Kd)

        # 药物阻断修正
        effective_occupancy = occupancy * (1.0 - antipsychotic_block)

        # 表达密度修正
        effective_signal = effective_occupancy * self.density

        self.current_occupancy = effective_occupancy
        self.activation_history.append(effective_occupancy)
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]

        return {
            'occupancy': np.clip(effective_occupancy, 0.0, 1.0),
            'effective_signal': np.clip(effective_signal, 0.0, 1.0),
            'receptor_type': self.receptor_type,
            'signaling_pathway': self.signaling_pathway,
            'is_excitatory': self.is_excitatory,
            'activation_state': f"{self.receptor_type} active: {effective_occupancy:.2%}",
        }


class D3Receptor(DopamineReceptor):
    """
    D3受体 (情绪-动机调节)

    特点:
    - 高亲和力 (Kd ~1nM)
    - limbic系统高表达 (NAc shell、嗅结节)
    - 情绪调节、动机控制、奖励敏感性

    参考: Sokoloff et al. (1990) - D3受体发现
    """

    def __init__(self, limbic_density: float = 0.55):
        super().__init__(receptor_type='D3', expression_density=limbic_density)
        self.limbic_density = limbic_density

    def forward(
        self,
        da_concentration_nM: float,
        emotional_context: float = 0.5,
        drug_block: float = 0.0,
    ) -> dict[str, float]:
        """D3特异性计算"""
        base_result = super().forward(da_concentration_nM, drug_block)

        # D3情绪调制: 高情绪负荷增强响应
        emotion_modulation = 1.0 + 0.3 * emotional_context

        # limbic特异性信号
        limbic_signal = base_result['effective_signal'] * emotion_modulation * self.limbic_density

        return {
            **base_result,
            'emotion_modulation': emotion_modulation,
            'limbic_signal': np.clip(limbic_signal, 0.0, 1.0),
            'reward_sensitivity': np.clip(limbic_signal * 1.5, 0.0, 1.0),
        }


class D4Receptor(DopamineReceptor):
    """
    D4受体 (注意力-认知)

    特点:
    - PFC高表达 (~45% of DA receptors in PFC)
    - 注意力调节、认知灵活性
    - ADHD关联受体

    参考: Tarazi et al. (1999) - D4受体分布
    参考: Swanson et al. (2000) - ADHD与D4关联
    """

    def __init__(self, pfc_density: float = 0.45):
        super().__init__(receptor_type='D4', expression_density=pfc_density)
        self.pfc_density = pfc_density

    def forward(
        self,
        da_concentration_nM: float,
        attention_demand: float = 0.5,
        cognitive_load: float = 0.3,
        drug_block: float = 0.0,
    ) -> dict[str, float]:
        """D4特异性计算"""
        base_result = super().forward(da_concentration_nM, drug_block)

        # D4认知调制: 高注意需求增强响应
        cognitive_modulation = 1.0 + 0.5 * attention_demand + 0.2 * cognitive_load

        # PFC特异性信号
        pfc_signal = base_result['effective_signal'] * cognitive_modulation * self.pfc_density

        return {
            **base_result,
            'cognitive_modulation': cognitive_modulation,
            'pfc_signal': np.clip(pfc_signal, 0.0, 1.0),
            'attention_flexibility': np.clip(pfc_signal * 1.2, 0.0, 1.0),
            'impulse_inhibition': np.clip(1.0 - pfc_signal * 0.5, 0.0, 1.0),
        }


class D5Receptor(DopamineReceptor):
    """
    D5受体 (皮层认知)

    特点:
    - 最高亲和力 (Kd ~0.3nM)
    - 皮层、海马、丘脑高表达
    - 认知功能、运动协调

    参考: Sunahara et al. (1991) - D5受体克隆
    """

    def __init__(self, cortical_density: float = 0.25):
        super().__init__(receptor_type='D5', expression_density=cortical_density)
        self.cortical_density = cortical_density

    def forward(
        self,
        da_concentration_nM: float,
        cognitive_state: float = 0.5,
        drug_block: float = 0.0,
    ) -> dict[str, float]:
        """D5特异性计算"""
        base_result = super().forward(da_concentration_nM, drug_block, use_high_affinity=True)

        # D5认知调制
        cognitive_modulation = 1.0 + 0.3 * cognitive_state

        # 皮层特异性信号
        cortical_signal = base_result['effective_signal'] * cognitive_modulation * self.cortical_density

        return {
            **base_result,
            'cognitive_modulation': cognitive_modulation,
            'cortical_signal': np.clip(cortical_signal, 0.0, 1.0),
            'working_memory_enhancement': np.clip(cortical_signal * 1.3, 0.0, 1.0),
        }


class DopamineReceptorFamily(nn.Module):
    """
    完整DA受体家族

    统一管理所有DA受体亚型，计算:
    - 各受体占有率
    - 兴奋性 vs 抑制性平衡
    - 药物选择性作用
    """

    def __init__(
        self,
        region: str = 'striatum',
    ):
        super().__init__()

        self.region = region
        distribution = RECEPTOR_DISTRIBUTION.get(region, RECEPTOR_DISTRIBUTION['striatum'])

        # 创建所有受体亚型
        self.D1 = DopamineReceptor('D1', distribution['D1'])
        self.D2 = DopamineReceptor('D2', distribution['D2'])
        self.D3 = D3Receptor(distribution['D3'])
        self.D4 = D4Receptor(distribution['D4'])
        self.D5 = D5Receptor(distribution['D5'])

        # 状态追踪
        self.state = ReceptorOccupancyState()

    def forward(
        self,
        da_concentration_nM: float,
        emotional_context: float = 0.5,
        attention_demand: float = 0.5,
        cognitive_state: float = 0.5,
        drug_blocks: dict[str, float] = None,  # 药物选择性阻断
    ) -> dict[str, Any]:
        """
        完整受体家族计算

        Args:
            da_concentration_nM: DA浓度 (nM)
            emotional_context: 情绪负荷 [0, 1]
            attention_demand: 注意需求 [0, 1]
            cognitive_state: 认知状态 [0, 1]
            drug_blocks: 各受体药物阻断率 {'D2': 0.7, 'D3': 0.5, ...}

        Returns:
            各受体占有率及综合信号
        """
        drug_blocks = drug_blocks or {}

        # 计算各受体响应
        d1_result = self.D1.forward(da_concentration_nM, drug_blocks.get('D1', 0.0))
        d2_result = self.D2.forward(da_concentration_nM, drug_blocks.get('D2', 0.0))
        d3_result = self.D3.forward(da_concentration_nM, emotional_context, drug_blocks.get('D3', 0.0))
        d4_result = self.D4.forward(da_concentration_nM, attention_demand, cognitive_state, drug_blocks.get('D4', 0.0))
        d5_result = self.D5.forward(da_concentration_nM, cognitive_state, drug_blocks.get('D5', 0.0))

        # 更新状态
        self.state.D1_occupancy = d1_result['occupancy']
        self.state.D2_occupancy = d2_result['occupancy']
        self.state.D3_occupancy = d3_result['occupancy']
        self.state.D4_occupancy = d4_result['occupancy']
        self.state.D5_occupancy = d5_result['occupancy']

        # 兴奋性信号 (D1 + D5)
        self.state.net_excitatory_signal = (
            d1_result['effective_signal'] + d5_result['effective_signal']
        ) / 2

        # 抑制性信号 (D2 + D3 + D4)
        self.state.net_inhibitory_signal = (
            d2_result['effective_signal'] +
            d3_result['effective_signal'] +
            d4_result['effective_signal']
        ) / 3

        return {
            'receptor_occupancy': {
                'D1': d1_result['occupancy'],
                'D2': d2_result['occupancy'],
                'D3': d3_result['occupancy'],
                'D4': d4_result['occupancy'],
                'D5': d5_result['occupancy'],
            },
            'receptor_signals': {
                'D1': d1_result['effective_signal'],
                'D2': d2_result['effective_signal'],
                'D3_limbic': d3_result['limbic_signal'],
                'D4_pfc': d4_result['pfc_signal'],
                'D5_cortical': d5_result['cortical_signal'],
            },
            'net_excitatory': self.state.net_excitatory_signal,
            'net_inhibitory': self.state.net_inhibitory_signal,
            'E_I_balance': self.state.net_excitatory_signal - self.state.net_inhibitory_signal,
            'functional_outputs': {
                'reward_sensitivity': d3_result['reward_sensitivity'],
                'attention_flexibility': d4_result['attention_flexibility'],
                'working_memory': d5_result['working_memory_enhancement'],
                'impulse_inhibition': d4_result['impulse_inhibition'],
            },
            'region': self.region,
        }

    def get_state_summary(self) -> dict[str, float]:
        """获取状态摘要"""
        return {
            'D1': self.state.D1_occupancy,
            'D2': self.state.D2_occupancy,
            'D3': self.state.D3_occupancy,
            'D4': self.state.D4_occupancy,
            'D5': self.state.D5_occupancy,
            'net_excitatory': self.state.net_excitatory_signal,
            'net_inhibitory': self.state.net_inhibitory_signal,
        }


# 药物选择性阻断配置 (典型抗精神病药物)
DRUG_SELECTIVITY = {
    'haloperidol': {'D2': 0.85, 'D3': 0.70, 'D4': 0.50, 'D1': 0.30, 'D5': 0.20},
    'clozapine': {'D4': 0.60, 'D3': 0.50, 'D2': 0.40, 'D1': 0.20, 'D5': 0.15},  # 非典型
    'risperidone': {'D2': 0.75, 'D3': 0.60, 'D4': 0.45, 'D1': 0.25, 'D5': 0.15},
    'aripiprazole': {'D2': 0.50, 'D3': 0.40, 'D4': 0.35, 'D1': 0.10, 'D5': 0.10},  # 部分激动
    'olanzapine': {'D2': 0.55, 'D3': 0.45, 'D4': 0.40, 'D1': 0.20, 'D5': 0.15},
}


def apply_drug_block(
    receptor_family: DopamineReceptorFamily,
    drug_name: str,
    occupancy_rate: float = 0.7,
) -> dict[str, float]:
    """
    应用药物阻断

    Args:
        receptor_family: 受体家族实例
        drug_name: 药物名称
        occupancy_rate: 实际占位率 [0, 1]

    Returns:
        实际阻断率字典
    """
    selectivity = DRUG_SELECTIVITY.get(drug_name, {'D2': 0.5})
    blocks = {r: s * occupancy_rate for r, s in selectivity.items()}
    return blocks


__all__ = [
    'DopamineReceptor',
    'D3Receptor',
    'D4Receptor',
    'D5Receptor',
    'DopamineReceptorFamily',
    'ReceptorOccupancyState',
    'RECEPTOR_KD',
    'RECEPTOR_DISTRIBUTION',
    'RECEPTOR_FUNCTIONS',
    'DRUG_SELECTIVITY',
    'apply_drug_block',
]