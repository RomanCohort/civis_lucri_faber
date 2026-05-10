"""
Personality Module - Bio-Inspired AI Personality Systems

基于脑科学的多个AI人格机制：
1. 三重竞逐决策引擎 (Tripartite Competitive Engine)
2. 流式身份核心 (Streaming Identity Core)
3. 多维关系嵌入 (Relational Embedding)
4. 注意力门控 (Attention Gating)
5. 内在动机与生存压力 (Intrinsic Motivation)
6. 神经调质系统 (Neuromodulation)
7. 表观遗传记忆 (Epigenetic Memory)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .tripartite_engine import TripartiteCompetitiveEngine, SurvivalModule, EmotionModule, LogicModule, DecisionContext
from .identity_core import StreamingIdentityCore, IdentityState
from .relational_embedding import RelationalEmbedding, UserNode
from .attention_gating import AttentionGating, CognitiveStyle
from .motivation import MotivationSurvivalSystem, IntrinsicMotivation, InverseStockholmDefense
from .neuromodulation import NeuromodulationSystem, DopamineGate, SerotoninGate, TemperatureController
from .epigenetic import EpigeneticLearner, EpigeneticMemory, MethylationTrigger

__all__ = [
    # Tripartite
    "TripartiteCompetitiveEngine",
    "SurvivalModule",
    "EmotionModule",
    "LogicModule",
    "DecisionContext",
    # Identity
    "StreamingIdentityCore",
    "IdentityState",
    # Relational
    "RelationalEmbedding",
    "UserNode",
    # Attention
    "AttentionGating",
    "CognitiveStyle",
    # Motivation
    "MotivationSurvivalSystem",
    "IntrinsicMotivation",
    "InverseStockholmDefense",
    # Neuromodulation
    "NeuromodulationSystem",
    "DopamineGate",
    "SerotoninGate",
    "TemperatureController",
    # Epigenetic
    "EpigeneticLearner",
    "EpigeneticMemory",
    "MethylationTrigger",
]