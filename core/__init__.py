"""Civis Lucri-Faber Core Modules"""

import sys
import os

# 处理相对导入
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from civis_lucri_faber.core.agent import CivisLucriFaber
except ImportError:
    from core.agent import CivisLucriFaber
from civis_lucri_faber.core.curiosity import CuriosityEngine
from civis_lucri_faber.core.information_gain import InformationGainCalculator
from civis_lucri_faber.core.meta_learning import MetaLearner, ActiveLearner
from civis_lucri_faber.core.self_alignment import SelfAlignmentModule
from civis_lucri_faber.core.thermodynamics import ThermodynamicsSystem
from civis_lucri_faber.core.metabolic_budget import (
    MetabolicBudget,
    MetabolicCostCalculator,
    PeriodicStarvation,
    MetabolicState,
    create_metabolic_budget,
)
from civis_lucri_faber.core.sleep import (
    SleepSystem,
    SleepController,
    SleepStage,
    MemoryReplayer,
    create_sleep_system,
)
from civis_lucri_faber.core.neuromodulation_integration import (
    NeuromodulationIntegration,
    RewardModulation,
    TemporalDiscount,
    AttentionModulator,
    create_neuromodulation_integration,
)
from civis_lucri_faber.core.cerebello_spinal import (
    CerebelloSpinalCoordination,
    SpinalCord,
    CentralPatternGenerator,
    ReflexPathway,
    Cerebellum,
    create_cerebello_spinal,
)
from civis_lucri_faber.core.basal_ganglia import (
    BasalGangliaSystem,
    BasalGanglia,
    create_basal_ganglia,
)
from civis_lucri_faber.core.hippocampus import (
    Hippocampus,
    create_hippocampus,
)
from civis_lucri_faber.core.limbic import (
    LimbicSystem,
    Amygdala,
    Thalamus,
    create_limbic_system,
)
from civis_lucri_faber.core.neurotransmitter import (
    NeurotransmitterSystem,
    DopamineSystem,
    SerotoninSystem,
    AcetylcholineSystem,
    create_neurotransmitter_system,
)

__all__ = [
    "CivisLucriFaber",
    "CuriosityEngine",
    "InformationGainCalculator",
    "MetaLearner",
    "ActiveLearner",
    "SelfAlignmentModule",
    "ThermodynamicsSystem",
    "MetabolicBudget",
    "MetabolicCostCalculator",
    "PeriodicStarvation",
    "MetabolicState",
    "create_metabolic_budget",
    # 睡眠系统
    "SleepSystem",
    "SleepController",
    "SleepStage",
    "MemoryReplayer",
    "create_sleep_system",
    # 神经调制集成
    "NeuromodulationIntegration",
    "RewardModulation",
    "TemporalDiscount",
    "AttentionModulator",
    "create_neuromodulation_integration",
    # 小脑-脊髓
    "CerebelloSpinalCoordination",
    "SpinalCord",
    "CentralPatternGenerator",
    "ReflexPathway",
    "Cerebellum",
    "create_cerebello_spinal",
    # 基底神经节
    "BasalGangliaSystem",
    "BasalGanglia",
    "create_basal_ganglia",
    # 海马体
    "Hippocampus",
    "create_hippocampus",
    # 边缘系统
    "LimbicSystem",
    "Amygdala",
    "Thalamus",
    "create_limbic_system",
    # 神经递质
    "NeurotransmitterSystem",
    "DopamineSystem",
    "SerotoninSystem",
    "AcetylcholineSystem",
    "create_neurotransmitter_system",
]