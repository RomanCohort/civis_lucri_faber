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
]