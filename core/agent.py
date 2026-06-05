"""Simulacrum 主智能体 (事件驱动架构)

协调多个维度模块 (事件驱动):
1. CuriosityEngine (自主探索目标设定)
2. InformationGainCalculator (信息增益内在动机)
3. MetaLearner + ActiveLearner + CognitiveDissonanceDetector (元学习与主动学习)
4. SelfAlignmentModule (自指涉自我对齐)
5. ThermodynamicsSystem (数字生存压力 - 经济模型)
6. PersonalityModule (心理人格系统 - 心理模型)
7. AdvancedEmotionModule (高级情绪系统 - 2025版本)
8. NeuralPruningSystem (神经修剪)
9. Neural Self-Regulation (ANS, HPA, Glial, Allostatic, PredictiveCoding)

事件驱动: 模块仅在收到相关事件时激活，避免无效计算
"""
from dataclasses import dataclass
from typing import Any
import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from core.addiction_dynamics import AddictionDynamicsEngine
from core.advanced_emotion_integration import (
    MODULES_AVAILABLE as ADVANCED_EMOTION_AVAILABLE,
)

# Advanced emotion modules (2025)
from core.advanced_emotion_integration import (
    IntegratedAdvancedEmotionSystem,
)
from core.allostatic_regulation import AllostaticRegulation
from core.angular_gyrus import AngularGyrus

# ===== 听觉系统 (Auditory System) =====
from core.auditory_cortex import AuditoryCortex

# 神经自调节系统 (Neural Self-Regulation)
from core.autonomic_nervous_system import AutonomicNervousSystem

# 孤立脑区模块（之前未集成）
from core.basal_ganglia import BasalGangliaSystem, NAcCore
from core.binaural_auditory import BinauralProcessor

# 脑干系统
from core.brainstem import Brainstem

# Censor 微表情感知集成
from core.censor_integration import CensorPerceptionModule

# ===== 认知扩展系统 (Cognitive Extensions) =====
from core.cerebral_cortex import VisualCortex
from core.cognitive_auditory_cortex import CognitiveAuditoryCortex
from core.cross_modal_binding import CrossModalBinder
from core.curiosity import CuriosityEngine, ExplorationGoal

# 分布式记忆系统 (Distributed Memory)
from core.distributed_memory import DistributedMemoryStore

# ===== 情绪扩展系统 (Emotion Extensions) =====
from core.emergent_emotion import EmergentEmotion
from core.emotion_dynamics import EmotionDynamicsSystem
from core.emotion_memory_consolidation import EmotionalMemoryConsolidation
from core.emotion_regulation import EmotionRegulationSystem
from core.emotional_contagion import EmotionalContagionSystem
from core.event_auditory_cortex import EventDrivenAuditoryCortex
from core.event_binaural_auditory import EventDrivenBinaural
from core.event_bus import EventBus
from core.event_phonetic_perception import EventDrivenPhoneticPerception
from core.events import (
    ALIGNMENT_CHECK,
    BRAIN_UPDATE,
    COMPRESSION_DONE,
    EMOTION_PROCESS,
    EXPLORATION_START,
    GOAL_NEEDED,
    MEMORY_ADDED,
    MEMORY_ENCODE,
    MICRO_EXPRESSION_PROCESS,
    MOTOR_CONTROL,
    NEURAL_REGULATION,
    PERSONALITY_UPDATE,
    PRUNING_UPDATE,
    SENSORY_PROCESS,
    STEP_END,
    STEP_START,
    VOCALIZATION_CONTROL,
    VOCALIZATION_OUTPUT,
)
from core.formant_synthesis import FormantToWaveform
from core.glial_system import GlialSystem

# 硬件生命体征桥接
from core.hardware_vitals import HardwareVitals

# 海马体 (情景记忆)
from core.hippocampus import Hippocampus

# 激素系统
from core.hormone_system import HormoneSystem
from core.hpa_axis import HPAAxis
from core.information_gain import TrueInformationGainCalculator, WorldModelWrapper

# ===== 实用工具 (Utilities) =====
from core.interference_forgetting import InterferenceEngine
from core.interoception import InteroceptivePredictionError
from core.language_cortex import LanguageCortex

# 边缘系统 (Amygdala + Thalamus)
from core.limbic import LimbicSystem

# ===== 评估系统 (Evaluation) =====
from core.llm_evaluator import LLMEvaluator

# 元学习
from core.meta_learning import (
    CognitiveDissonanceDetector,
    UncertaintyAwareActiveLearner,
)
from core.metabolic_budget import MetabolicCostCalculator

# 心境系统 (Mood System)
from core.mood_system import MoodState, MoodSystem
from core.multimodal_perception import MultimodalPerception

# 神经修剪系统
from core.neural_pruning import (
    NeuralPruningSystem,
    PruningConfig,
)

# 神经药理学
from core.neuro_pharmacology import NeuroPharmacology
from core.neuromodulation_integration import NeuromodulationIntegration
from core.neuroplasticity import NeuroplasticitySystem
from core.neurotransmitter import NeurotransmitterSystem
from core.pathogen_neuroinflammation import PathogenTriggeredInflammationEngine
from core.pd_target_mapping import build_pd_targets

# Personality modules
from core.personality import (
    AttentionGating,
    EpigeneticLearner,
    MotivationSurvivalSystem,
    NeuromodulationSystem,
    RelationalEmbedding,
    StreamingIdentityCore,
    TripartiteCompetitiveEngine,
)

# ===== 治疗/药理学系统 (Therapy/Pharmacology) =====
from core.pharmacotherapy_synergy import SynergyCalculator
from core.phonetic_perception import PhoneticPerception
from core.policy_learning import SimpleQLearning
from core.predictive_coding import PredictiveCodingSystem
from core.prefrontal_cortex import PrefrontalCortex

# 精神疾病模拟器
from core.psychiatric_simulation import PsychiatricConditionSimulator
from core.psychometric_indicators import PsychometricIndicatorTracker
from core.psychopharmacology_sandbox import PsychopharmacologySandbox
from core.psychotherapy import PsychotherapySystem
from core.rhythm import RhythmSystem

# 视交叉上核 (昼夜节律)
from core.scn import LightType, SuprachiasmaticNucleus
from core.self_alignment import SelfAlignmentModule

# 自我意识中枢 (Self-Awareness Center)
from core.self_awareness import SelfAwarenessCenter

# 睡眠系统 (NREM/REM 周期, 记忆重播)
# 深层药理学模块 (懒加载)
from core.sleep import OrexinSystem, SleepSystem
from core.snn_core import SpikingLayer

# 社会认知系统 (Social Cognition)
from core.social_cognition import SocialCognitionSystem
from core.social_emotions import SocialEmotionSystem
from core.spiking_auditory_cortex import SpikingAuditoryCortex
from core.state_key_mapping import UnifiedStateMapping
from core.symptom_tracker import SymptomTracker
from core.therapeutic_experiment import TherapeuticExperiment
from core.thermodynamics import ThermodynamicsSystem

# 工具系统
from core.tool_system import ToolSystem, register_default_tools

# 发音语言系统
from core.vocalization import (
    VocalCortex,
)
from utils.api_client import create_api_client
from utils.config import Config
from utils.memory import KnowledgeMemory

# ===== Shared types (avoid circular imports) =====
from core.types import AgentState, ChatResponse

# ===== Mixin imports (decomposed from monolithic) =====
from core.agent_mixins.chat_mixin import ChatMixin
from core.agent_mixins.state_mixin import StateMixin
from core.agent_mixins.vocalization_mixin import VocalizationMixin


class Simulacrum(ChatMixin, StateMixin, VocalizationMixin):
    """Simulacrum 主智能体 (事件驱动架构)

    一个具备自我学习、自我维持、自我进化能力的 AI 智能体

    事件驱动模式:
    - 所有模块通过 EventBus 解耦
    - 模块仅在收到相关事件时激活
    - DEAD/HIBERNATE 状态自动跳过后续事件

    Mixin 组合:
    - ChatMixin: 对话系统三层认知管道
    - StateMixin: 状态统计、保存/加载、回合运行
    - VocalizationMixin: 发音系统辅助方法
    """

    def __init__(
        self,
        config: Config | None = None,
        memory_path: str = "memory.json",
        alignment_log_path: str = "self_alignment_log.json",
        thermo_log_path: str = "thermodynamics_log.json",
        state_dim: int = 4,
        n_actions: int = 4
    ):
        self.config = config or Config()

        # ===== 事件总线 =====
        self.bus = EventBus(log_enabled=self.config.event_log_enabled)
        logger.info("[EVENT] EventBus initialized")

        # ===== 核心模块 =====
        # 维度1: 好奇心探索
        self.curiosity = CuriosityEngine(
            alpha=self.config.curiosity_alpha,
            beta=self.config.curiosity_beta,
            gamma=self.config.curiosity_gamma,
            exploration_rate=self.config.exploration_rate,
            event_bus=self.bus,
        )

        # 维度2: 信息增益
        self.info_gain_calc = TrueInformationGainCalculator(
            state_dim=100,
            action_dim=16,
            latent_dim=32,
            lr=self.config.world_model_lr,
            intrinsic_lambda=self.config.intrinsic_motivation_lambda,
            device=self.config.device,
            event_bus=self.bus,
        )

        # 好奇心 ← 世界模型引用 (不确定性驱动目标生成)
        self.curiosity._world_model = self.info_gain_calc.get_world_model()

        # 维度3: 元学习与主动学习
        self.dissonance_detector = CognitiveDissonanceDetector(event_bus=self.bus)
        self.active_learner = None  # 延迟初始化: 需要模型实例

        # 维度4: 自对齐
        self.api_client = create_api_client(self.config)
        self.self_alignment = SelfAlignmentModule(
            api_client=self.api_client,
            check_interval=self.config.alignment_check_interval,
            log_path=alignment_log_path,
            event_bus=self.bus,
        )

        # 维度5: 数字生存压力
        self.thermo = ThermodynamicsSystem(
            initial_balance=self.config.initial_balance,
            compute_cost_per_sec=self.config.compute_cost_per_sec,
            storage_cost_per_sec=self.config.storage_cost_per_sec,
            task_reward_min=self.config.task_reward_min,
            task_reward_max=self.config.task_reward_max,
            compress_threshold=self.config.compress_threshold,
            log_path=thermo_log_path,
            event_bus=self.bus,
        )
        # 代谢预算系统 (资源约束: 活跃神经元比例上限)
        self.metabolic = MetabolicCostCalculator(
            resource_budget=self.config.resource_budget,
        )

        # ===== 维度6: 心理人格系统 =====
        self.tripartite = TripartiteCompetitiveEngine(event_bus=self.bus)
        self.identity_core = StreamingIdentityCore(event_bus=self.bus)
        self.relation = RelationalEmbedding(event_bus=self.bus)
        self.attention = AttentionGating(event_bus=self.bus)
        self.motivation = MotivationSurvivalSystem(event_bus=self.bus)
        self.neuromodulation = NeuromodulationSystem(hidden_dim=128, event_bus=self.bus)
        self.epigenetic = EpigeneticLearner(rank=8, event_bus=self.bus)

        self.current_user_id = "default"

        # ===== 维度7: 高级情绪系统 (2025) =====
        if ADVANCED_EMOTION_AVAILABLE:
            self.advanced_emotion = IntegratedAdvancedEmotionSystem(
                input_dim=64,
                hidden_dim=128,
                event_bus=self.bus,
            )
            logger.info("[OK] Advanced Emotion System initialized (event-driven)")
        else:
            self.advanced_emotion = None
            logger.warning("[WARN] Advanced Emotion System not available")

        # ===== 辅助系统 =====
        self.memory = KnowledgeMemory(
            max_size=self.config.max_history_size,
            memory_path=memory_path
        )

        # ===== 心境系统 (Mood System) =====
        self.mood_system = MoodSystem(
            mood_dim=5,
            event_dim=64,
        )
        self.current_mood = MoodState(
            valence=0.0, arousal=0.3, dominance=0.5,
            activation=0.3, pleasantness=0.5,
        )
        logger.info("[OK] Mood System initialized (OU dynamics + circadian rhythm)")

        # ===== 分布式记忆系统 =====
        self.distributed_memory = DistributedMemoryStore(
            state_dim=64,
            encoding_dim=128,
        )
        logger.info("[OK] Distributed Memory System initialized (cross-region storage)")

        # ===== 维度8: 神经修剪系统 =====
        pruning_config = PruningConfig(
            decay_base=self.config.prune_decay_rate,
            hibernation_steps=200,
        )
        self.neural_pruning = NeuralPruningSystem(config=pruning_config, event_bus=self.bus)
        if hasattr(self.info_gain_calc, 'world_model'):
            for name, module in self.info_gain_calc.world_model.named_modules():
                if isinstance(module, nn.Linear):
                    self.neural_pruning.attach(module, f"world_model.{name}")
        logger.info(f"[OK] Neural Pruning System initialized ({len(self.neural_pruning._modules_map)} modules attached)")

        # ===== 维度9: 神经自调节系统 =====
        self.ans = AutonomicNervousSystem(event_bus=self.bus)
        self.hpa_axis = HPAAxis(
            stress_reactivity=self.config.hpa_stress_reactivity,
            cortisol_half_life_steps=self.config.hpa_cortisol_half_life_steps,
            feedback_strength=self.config.hpa_feedback_strength,
            load_accumulation_rate=self.config.hpa_load_accumulation_rate,
            event_bus=self.bus,
        )
        self.glial = GlialSystem(event_bus=self.bus)
        self.allostatic = AllostaticRegulation(event_bus=self.bus)
        self.predictive_coding = PredictiveCodingSystem(sensory_dim=64, n_layers=3, event_bus=self.bus)
        logger.info("[OK] Neural Self-Regulation Systems initialized (event-driven)")

        # ===== 维度9b: 社会认知系统 =====
        self.social_cognition = SocialCognitionSystem(action_dim=16, state_dim=64, emotion_dim=8, event_bus=self.bus)
        logger.info("[OK] Social Cognition & Mirror Neuron System initialized (event-driven)")

        # ===== 维度9c: 自我意识中枢 =====
        self.self_awareness = SelfAwarenessCenter(state_dim=64, hidden_dim=64, event_bus=self.bus)
        logger.info("[OK] Self-Awareness Center initialized (event-driven)")

        # ===== 维度10: 孤立脑区模块集成 =====
        self.basal_ganglia = BasalGangliaSystem(state_dim=64, n_actions=n_actions, event_bus=self.bus)
        self.neurotransmitter = NeurotransmitterSystem(event_bus=self.bus)
        self.neuroplasticity = NeuroplasticitySystem(n_neurons=100, n_synapses=500, event_bus=self.bus)
        self.language_cortex = LanguageCortex(vocab_size=10000, use_parallel=True, event_bus=self.bus)
        self.prefrontal = PrefrontalCortex(input_dim=64, hidden_dim=128, num_actions=n_actions, event_bus=self.bus)
        self.angular_gyrus = AngularGyrus(embed_dim=256, event_bus=self.bus)

        # BG-小脑耦合 (熟能生巧: 重复动作 → 小脑自动执行 → 释放意识资源)
        from core.cerebello_spinal import Cerebellum
        self.cerebellum = Cerebellum(sensory_dim=64, n_motor_joints=n_actions)
        self.basal_ganglia.bg.set_cerebellum(self.cerebellum)
        logger.info("[OK] Brain Region Modules initialized (event-driven, BG→Cerebellum coupled)")

        # ===== 维度11: 激素系统 =====
        self.hormones = HormoneSystem(event_bus=self.bus)
        logger.info("[OK] Hormone System initialized (event-driven)")

        # ===== 维度12: 脑干系统 =====
        self.brainstem = Brainstem(event_bus=self.bus)
        logger.info("[OK] Brainstem System initialized (event-driven)")

        # ===== 维度13: 发音语言系统 =====
        self.vocal_cortex = VocalCortex(event_bus=self.bus)
        self._last_vocalization = None   # 最近一次发声输出

        # ===== 维度13b: 生物-语言耦合器 =====
        # 直接将神经系统状态映射为语言输出特征
        try:
            from core.bio_linguistic_coupler import BioLinguisticCoupler
            self.bio_linguistic = BioLinguisticCoupler(seed=42)
            logger.info("[OK] Bio-Linguistic Coupler initialized (bio-driven language modulation)")
        except ImportError:
            try:
                from core.bio_linguistic_coupler import BioLinguisticCoupler
                self.bio_linguistic = BioLinguisticCoupler(seed=42)
                logger.info("[OK] Bio-Linguistic Coupler initialized (bio-driven language modulation)")
            except ImportError:
                self.bio_linguistic = None
                logger.warning("[WARN] Bio-Linguistic Coupler not available")

        # ===== 维度13c: 人格-语言适配器 =====
        try:
            from core.personality_language_adapter import PersonalityLanguageAdapter
            self.personality_adapter = PersonalityLanguageAdapter()
            logger.info("[OK] Personality-Language Adapter initialized")
        except ImportError:
            try:
                from core.personality_language_adapter import PersonalityLanguageAdapter
                self.personality_adapter = PersonalityLanguageAdapter()
                logger.info("[OK] Personality-Language Adapter initialized")
            except ImportError:
                self.personality_adapter = None
                logger.warning("[WARN] Personality-Language Adapter not available")

        # ===== 维度13d: 记忆-语言适配器 =====
        try:
            from core.memory_language_adapter import MemoryLanguageAdapter
            self.memory_adapter = MemoryLanguageAdapter()
            logger.info("[OK] Memory-Language Adapter initialized")
        except ImportError:
            try:
                from core.memory_language_adapter import MemoryLanguageAdapter
                self.memory_adapter = MemoryLanguageAdapter()
                logger.info("[OK] Memory-Language Adapter initialized")
            except ImportError:
                self.memory_adapter = None
                logger.warning("[WARN] Memory-Language Adapter not available")

        logger.info("[OK] Vocal Cortex & Speech Production Pipeline initialized (event-driven)")

        # ===== 维度14: 视交叉上核 (SCN) - 昼夜节律 =====
        self.scn = SuprachiasmaticNucleus(intrinsic_period=24.2, chronotype_offset=0.0)
        logger.info("[OK] SCN Circadian Clock initialized")

        # ===== 维度15: 边缘系统 (Amygdala + Thalamus) =====
        self.limbic = LimbicSystem(input_dim=64, event_bus=self.bus)
        logger.info("[OK] Limbic System initialized (Amygdala + Thalamus)")

        # ===== 维度16: 海马体 (Hippocampus) - 情景记忆 =====
        self.hippocampus = Hippocampus(input_dim=64, encoding_dim=128, event_bus=self.bus)
        logger.info("[OK] Hippocampus initialized (event-driven)")

        # ===== 维度17: 睡眠系统 (NREM/REM, memory replay) =====
        self.sleep_system = SleepSystem(enable_sleep=True, enable_dream=True, sleep_threshold=0.8)
        logger.info("[OK] Sleep System initialized (NREM/REM cycles)")

        # ===== 硬件生命体征桥接 =====
        self.hw = HardwareVitals()
        logger.info("[OK] Hardware Vitals Bridge initialized (CPU→HR, RAM→BP, GC→gut 5-HT)")

        # ===== Censor 微表情感知系统 (仿生双通路 MER) =====
        self.censor = CensorPerceptionModule(
            event_bus=self.bus,
            device=self.config.device,
        )
        logger.info("[OK] Censor Micro-Expression Perception initialized (event-driven, lazy-load)")

        # ===== 精神疾病与情绪状态模拟器 =====
        self.psychiatric_sim = PsychiatricConditionSimulator(
            agent=self,
            event_bus=self.bus,
        )
        logger.info("[OK] Psychiatric Condition Simulator initialized (26 conditions + 7 emotion states)")

        # ===== 工具系统 =====
        self.tools = ToolSystem(event_bus=self.bus)
        register_default_tools(self.tools, self)
        logger.info(f"[OK] Tool System initialized ({len(self.tools._tools)} tools registered)")

        # ===== 神经药理学 =====
        self.pharma = NeuroPharmacology(self)
        logger.info("[OK] Neuro-Pharmacology System initialized")

        # ===== 感觉-神经递质耦合系统 =====
        from core.sensory_neuro_coupling import SensoryNeuroCoupling
        self.sensory_neuro_coupling = SensoryNeuroCoupling(event_bus=self.bus)
        logger.info("[OK] Sensory-Neurotransmitter Coupling System initialized")

        # ===== 深层药理学模块 (懒加载 — 仅在注册药物/病原体时激活) =====
        self._orexin_system: OrexinSystem | None = None
        self._interoceptive_pe: InteroceptivePredictionError | None = None
        self._symptom_tracker: SymptomTracker | None = None
        self._addiction_engine: AddictionDynamicsEngine | None = None
        self._pathogen_engine: PathogenTriggeredInflammationEngine | None = None
        self._nac_core: NAcCore | None = None

        # ===== 维度18: 听觉系统 (Auditory System) =====
        self.auditory_cortex = AuditoryCortex(sample_rate=16000, n_filters=128)
        self.binaural = BinauralProcessor(n_channels=128)
        self.cognitive_auditory = CognitiveAuditoryCortex(sample_rate=16000, n_channels=128)
        self.event_auditory = EventDrivenAuditoryCortex(sample_rate=16000, n_filters=128)
        self.event_binaural = EventDrivenBinaural(n_channels=128)
        self.event_phonetic = EventDrivenPhoneticPerception(input_dim=256)
        self.phonetic = PhoneticPerception(input_dim=256)
        self.spiking_auditory = SpikingAuditoryCortex(sample_rate=16000, n_channels=128)
        self.formant_synth = FormantToWaveform(sample_rate=22050)
        logger.info("[OK] Auditory System initialized (9 modules: cortex, binaural, phonetic, spiking, formant)")

        # ===== 维度19: 情绪扩展系统 (Emotion Extensions) =====
        self.emergent_emotion = EmergentEmotion(input_dim=64, hidden_dim=128)
        self.emotion_dynamics = EmotionDynamicsSystem(state_dim=3, hidden_dim=64)
        self.emotion_consolidation = EmotionalMemoryConsolidation(content_dim=64, hidden_dim=128)
        self.emotion_regulation = EmotionRegulationSystem(input_dim=64, hidden_dim=128, emotion_dim=8)
        self.emotional_contagion = EmotionalContagionSystem(expression_dim=64, emotion_dim=8)
        self.social_emotions = SocialEmotionSystem(self_dim=64, input_dim=64)
        logger.info("[OK] Emotion Extension System initialized (6 modules: emergent, dynamics, consolidation, regulation, contagion, social)")

        # ===== 维度20: 认知扩展系统 (Cognitive Extensions) =====
        self.visual_cortex = VisualCortex(input_channels=3, embed_dim=768)
        self.cross_modal = CrossModalBinder(
            input_dims={'vision': 768, 'audio': 256, 'language': 512},
            embed_dim=256, num_heads=8,
        )
        self.multimodal = MultimodalPerception(vocab_size=10000)
        self.neuromod_integration = NeuromodulationIntegration(hidden_dim=128)
        self.snn_layer = SpikingLayer(in_features=128, out_features=64)
        logger.info("[OK] Cognitive Extension System initialized (5 modules: visual, cross-modal, multimodal, neuromod, SNN)")

        # ===== 维度21: 治疗/药理学系统 (Therapy/Pharmacology) =====
        self.synergy_calc = SynergyCalculator()
        self.psychotherapy = PsychotherapySystem(agent=self, event_bus=self.bus)
        self.pharma_sandbox = PsychopharmacologySandbox(agent=self)
        self.therapeutic_exp = TherapeuticExperiment(agent=self)
        self.psychometric = PsychometricIndicatorTracker()
        self.pd_targets = build_pd_targets  # function reference
        self.state_mapping = UnifiedStateMapping()
        logger.info("[OK] Therapy/Pharmacology System initialized (7 modules: synergy, psychotherapy, sandbox, experiment, psychometric, PD targets, state mapping)")

        # ===== 维度22: 评估系统 (Evaluation) =====
        self.llm_evaluator = LLMEvaluator(api_client=self.api_client)
        logger.info("[OK] LLM Evaluator initialized")

        # ===== 维度23: 实用工具 (Utilities) =====
        self.interference = InterferenceEngine()
        self.policy_learner = SimpleQLearning(state_dim=state_dim, n_actions=n_actions)
        self.rhythm = RhythmSystem(n_neurons=64, n_senses=4, event_bus=self.bus)
        logger.info("[OK] Utility Systems initialized (interference forgetting, policy learning, rhythm)")

        # ===== 对话系统 =====
        self._chat_history: list[dict[str, str]] = []
        self._max_chat_history = 20

        # 状态
        self.step_count = 0
        self.current_goal: ExplorationGoal | None = None
        self._internal_state: dict[str, Any] = {}

        # ===== 感觉状态初始化 (扩展到100维) =====
        self._internal_state.update({
            # 听觉 VAD (dims 80-83)
            'auditory_vad': {
                'valence': 0.5, 'arousal': 0.0, 'dominance': 0.5, 'pleasantness': 0.5
            },
            # 听觉特征 (dims 84-87)
            'auditory_features': {
                'phoneme_confidence': 0.0, 'spatial_certainty': 0.0,
                'speech_rate': 0.5, 'spectral_complexity': 0.0
            },
            # 语言语义 (dims 88-91)
            'language_semantic': {
                'language_valence': 0.5, 'language_arousal': 0.0,
                'language_surprise': 0.0, 'language_dominance': 0.5
            },
            # 跨模态绑定 (dims 92-95)
            'crossmodal': {
                'crossmodal_coherence': 0.5, 'sensory_saliency': 0.0,
                'sensory_novelty': 0.0, 'sensory_conflict': 0.0
            },
            # 感觉神经化学 (dims 96-99)
            'sensory_neurochemical': {
                'sensory_adrenaline': 0.0, 'sensory_dopamine': 0.0,
                'sensory_cortisol': 0.0, 'sensory_acetylcholine': 0.0
            },
        })

        # ─── StateProxy ────────────────────────────────
        # Lightweight read/write gate around _internal_state;
        # brain-region writers are registered so each region
        # can only write its declared output_keys.
        from core.state_proxy import StateProxy
        self.state_proxy = StateProxy(self._internal_state, name="agent", strict=False)
        for region_name, region in [
            ("basal_ganglia", self.basal_ganglia),
            ("hippocampus", self.hippocampus),
            ("prefrontal", self.prefrontal),
            ("limbic", self.limbic),
            ("brainstem", self.brainstem),
        ]:
            if hasattr(region, "output_keys"):
                self.state_proxy.register_writer(region_name, region.output_keys())

    def _write(self, key: str, value) -> None:
        """Write to internal_state through StateProxy (write-protected)."""
        self.state_proxy.write("agent", key, value)

    def reset(self) -> None:
        """重置智能体"""
        self.step_count = 0
        self.current_goal = None
        self.curiosity.reset()
        self.thermo.reset()
        self.memory.clear()

    def _inject_auditory_state(self, event) -> dict:
        """从 SENSORY_PROCESS 事件提取听觉 VAD 和特征，注入 _internal_state。

        Event handler subscribed to SENSORY_PROCESS with priority=2.
        """
        auditory_data = event.data.get('auditory', {})
        vad = auditory_data.get('vad', {})

        self._internal_state['auditory_vad'] = {
            'valence': vad.get('valence', 0.5),
            'arousal': vad.get('arousal', 0.0),
            'dominance': vad.get('dominance', 0.5),
            'pleasantness': vad.get('pleasantness', 0.5),
        }

        features = auditory_data.get('features', {})
        self._internal_state['auditory_features'] = {
            'phoneme_confidence': features.get('phoneme_confidence', 0.0),
            'spatial_certainty': features.get('spatial_certainty', 0.0),
            'speech_rate': features.get('speech_rate', 0.5),
            'spectral_complexity': features.get('spectral_complexity', 0.0),
        }

        return {'auditory_injected': True}

    def _inject_language_semantic_state(self, event) -> dict:
        """从 SENSORY_PROCESS 事件提取语言语义状态，注入 _internal_state。

        Event handler subscribed to SENSORY_PROCESS with priority=2.
        """
        language_data = event.data.get('language', {})
        semantic = language_data.get('semantic', {})

        # 也从 internal_state 中读取 language_cortex 写入的值
        internal_state = event.data.get('internal_state', {})

        self._internal_state['language_semantic'] = {
            'language_valence': semantic.get('valence', internal_state.get('language_valence', 0.5)),
            'language_arousal': semantic.get('arousal', internal_state.get('language_arousal', 0.0)),
            'language_surprise': semantic.get('surprise', internal_state.get('language_surprise', 0.0)),
            'language_dominance': semantic.get('dominance', 0.5),
        }

        return {'language_semantic_injected': True}

    def _inject_crossmodal_state(self, event) -> dict:
        """从 SENSORY_PROCESS 事件提取跨模态绑定状态，注入 _internal_state。

        Event handler subscribed to SENSORY_PROCESS with priority=3.
        """
        crossmodal_data = event.data.get('crossmodal', {})

        self._internal_state['crossmodal'] = {
            'crossmodal_coherence': crossmodal_data.get('coherence', 0.5),
            'sensory_saliency': crossmodal_data.get('saliency', 0.0),
            'sensory_novelty': crossmodal_data.get('novelty', 0.0),
            'sensory_conflict': crossmodal_data.get('conflict', 0.0),
        }

        return {'crossmodal_injected': True}

    def register_pathogen(self, pathogen_name: str) -> None:
        """注册病原体到炎症引擎，激活病原体→小胶质细胞通路。"""
        if self._pathogen_engine is None:
            self._pathogen_engine = PathogenTriggeredInflammationEngine()
        self._pathogen_engine.register_pathogen(pathogen_name)
        logger.info(f"[PHARMA] Pathogen registered: {pathogen_name}")

    def register_addictive_drug(self, drug_name: str, drug_class: str) -> None:
        """注册成瘾药物到动力学引擎，激活耐受/戒断/渴求追踪。"""
        if self._addiction_engine is None:
            self._addiction_engine = AddictionDynamicsEngine()
        self._addiction_engine.register_drug(drug_name, drug_class)
        logger.info(f"[PHARMA] Addictive drug registered: {drug_name} (class={drug_class})")

    def step(self, user_input: str = None, user_sentiment: float = 0.0, external_stimulus: float = 0.0, video: torch.Tensor = None) -> AgentState:
        """执行一步 (事件驱动)

        不再直接调用模块方法，而是发布事件让订阅者自行响应。

        Args:
            user_input: 用户输入文本
            user_sentiment: 用户情感 [-1, 1]
            external_stimulus: 外部刺激强度 [0, 1]
            video: 视频帧张量 (B, 3, T, H, W)，供 Censor 微表情分析
        """
        # ── 注入 Censor 视频输入 ──
        if video is not None:
            self._write('censor_video', video)
        # ===== 0. 读取硬件生命体征 =====
        self.hw.read(agent=self)

        # ===== 1. 发布步开始事件 → 热力学系统响应 =====
        step_result = self.bus.publish(STEP_START, {"elapsed_seconds": 1.0}, source="agent")
        thermo_state = step_result.get("thermodynamics", {}).get("thermo_state", "ACTIVE")
        balance = step_result.get("thermodynamics", {}).get("balance", self.thermo.balance)

        if thermo_state == "DEAD":
            logger.warning("[DEAD] Digital death! Process will be terminated.")
            return AgentState(
                step=self.step_count, status="DEAD",
                balance=balance, current_goal=None,
                info_gain=0.0, alignment_score=0.0
            )

        if thermo_state == "HIBERNATE":
            logger.info("[SLEEP] Entering hibernation mode — full sleep cycle")
            # SCN 在黑暗中步进
            try:
                scn_sleep = self.scn.step(
                    light_input=0.0,
                    light_type=LightType.DARKNESS,
                    is_awake=False,
                    elapsed_minutes=1.0,
                )
                self._write('scn_melatonin', scn_sleep.melatonin)
                self._write('scn_sleep_pressure', scn_sleep.sleep_pressure)
                self._write('scn_wake_drive', scn_sleep.wake_drive)
            except RuntimeError:
                pass  # SCN step failed (PyTorch or attribute error)
            sleep_result = self.sleep_system.update(info_gain_reward=0.0, step_duration=1.0)
            sleep_stage = sleep_result.get('stage', 'awake')

            # NREM 深睡: 海马体重播 + 记忆巩固
            if sleep_stage in ('nrem3', 'nrem2'):
                try:
                    forward_memories = self.hippocampus.replay_forward()
                    if forward_memories:
                        self.hippocampus.consolidate_recent()
                    for mem in forward_memories[:5]:
                        self.sleep_system.add_experience(
                            mem.state, mem.action, mem.reward, mem.state,
                        )
                except RuntimeError:
                    pass  # Hippocampus replay/consolidate failed
            if sleep_stage == 'rem' and sleep_result.get('dream'):
                self._write('dream_content', sleep_result['dream'])
                try:
                    self.memory.add_memory(
                        content=f"Dream: {sleep_result['dream']}",
                        importance=0.3,
                        tags=["dream", "sleep", str(sleep_stage)],
                    )
                except (RuntimeError, AttributeError):
                    pass  # Memory add failed or sleep_result missing 'dream' key
            downscale = sleep_result.get('synaptic_downscale', 1.0)
            if downscale < 1.0:
                self.bus.publish(PRUNING_UPDATE, {}, source="agent")

            # 存储睡眠状态
            self._write('sleep_stage', sleep_stage)
            self._write('sleep_consolidation_bonus', sleep_result.get('consolidation_bonus', 0.0))
            self._write('sleep_synaptic_downscale', downscale)

            # SCN 唤醒检查
            try:
                if scn_sleep.wake_drive > 0.8 and scn_sleep.sleep_pressure < 0.2:
                    self.sleep_system.controller.wake_up()
            except (AttributeError, UnboundLocalError):
                pass  # scn_sleep missing wake_drive or sleep_pressure attributes

            return AgentState(
                step=self.step_count, status="HIBERNATE",
                balance=balance,
                current_goal=self.current_goal.description if self.current_goal else None,
                info_gain=0.0,
                alignment_score=self.self_alignment.get_alignment_score()
            )

        # ===== 2. SCN 昼夜节律步进 (为下游模块提供节律信号) =====
        try:
            circadian_hour = self.scn.get_circadian_hour()
            is_daytime = 6.0 <= circadian_hour <= 20.0
            light_input = 0.6 if is_daytime else 0.1
            light_type = LightType.INDOOR if is_daytime else LightType.DIM

            scn_output = self.scn.step(
                light_input=light_input,
                light_type=light_type,
                is_awake=True,
                elapsed_minutes=1.0,
            )
            self._write('scn_melatonin', scn_output.melatonin)
            self._write('scn_cortisol_rhythm', scn_output.cortisol_rhythm)
            self._write('scn_alertness', scn_output.alertness)
            self._write('scn_wake_drive', scn_output.wake_drive)
            self._write('scn_sleep_pressure', scn_output.sleep_pressure)
            self._write('scn_temperature', scn_output.core_temperature)
            self._write('scn_circadian_hour', circadian_hour)
        except RuntimeError:
            pass  # SCN circadian step failed (PyTorch or attribute error)
        if self._orexin_system is None:
            self._orexin_system = OrexinSystem()
        try:
            orexin_result = self._orexin_system.step(
                gaba_level=float(self._internal_state.get("nt_gaba", 0.5)),
                scn_wake_drive=float(self._internal_state.get("scn_wake_drive", 0.5)),
                stress_level=float(self._internal_state.get("hormone_cortisol", 0.3)),
                receptor_block=float(self._internal_state.get("rct_orexin1_block", 0.0)),
            )
            self._write("orexin_level", orexin_result["orexin_level"])
            self._write("effective_orexin", orexin_result["effective_orexin"])
        except RuntimeError:
            pass  # Orexin system step failed
        try:
            real_state_np = self._build_state_vector()
            real_state_t = torch.FloatTensor(real_state_np).unsqueeze(0)  # [1, 100]
        except RuntimeError:
            # _build_state_vector or tensor creation failed
            real_state_np = np.full(100, 0.5, dtype=np.float32)
            real_state_t = torch.FloatTensor(real_state_np).unsqueeze(0)

        # ===== 3. 边缘系统 + 语言皮层 + 角回 (事件驱动: SENSORY_PROCESS) =====
        try:
            self.bus.publish(SENSORY_PROCESS, {
                "internal_state": self._internal_state,
                "state_tensor": real_state_t,
                "state_np": real_state_np,
                "user_input": user_input,
                "step_count": self.step_count,
            }, source="agent")
        except RuntimeError:
            pass  # EventBus publish failed
        # 当有视频输入时，自动触发 Censor 推理，输出注入情绪/边缘系统
        censor_video = self._internal_state.get('censor_video')
        if censor_video is not None:
            try:
                censor_result = self.bus.publish(MICRO_EXPRESSION_PROCESS, {
                    "video": censor_video,
                    "internal_state": self._internal_state,
                    "state_tensor": real_state_t,
                }, source="agent")
                censor_data = censor_result.get("censor_perception", {})
                if censor_data:
                    emotion_map = censor_data.get("emotion_map", {})
                    for emo_name, emo_val in emotion_map.items():
                        self._internal_state[f'censor_{emo_name}'] = emo_val
                    self._write('censor_me_predicted', censor_data.get("me_predicted", 0))
                    self._write('censor_me_confidence', censor_data.get("me_confidence", 0.0))
                    self._write('censor_au_active_count', len(censor_data.get("au_active", [])))
                    self._write('censor_apex_frame', censor_data.get("apex_frame", 0))
                    self._write('censor_dominant_expert', censor_data.get("dominant_expert", 0))

                    # Censor AU → 边缘系统杏仁核威胁评估
                    au_active = censor_data.get("au_active", [])
                    threat_aus = [4, 5, 7, 23]  # anger-related AUs
                    threat_level = sum(1 for au in au_active if au in threat_aus) / len(threat_aus)
                    self._write('censor_threat_level', threat_level)

                    # Censor 情绪 → 高级情绪系统调制
                    dominant_emotion = max(emotion_map, key=emotion_map.get) if emotion_map else "neutral"
                    self._write('censor_dominant_emotion', dominant_emotion)

                    censor_summary = self.censor.get_summary()
                    logger.info(f"[CENSOR] {censor_summary}")
            except Exception as e:
                logger.warning(f"[CENSOR] Micro-expression processing failed: {e}")

        # ===== 3c. 感觉状态注入 (从 SENSORY_PROCESS 事件中提取并注入) =====
        # 注意：这些值由 LanguageCortex、AuditoryCortex 等模块写入到 internal_state，
        # 我们在这里提取并确保它们存在
        if 'auditory_vad' not in self._internal_state:
            self._internal_state['auditory_vad'] = {
                'valence': 0.5, 'arousal': 0.0, 'dominance': 0.5, 'pleasantness': 0.5
            }
        if 'auditory_features' not in self._internal_state:
            self._internal_state['auditory_features'] = {
                'phoneme_confidence': 0.0, 'spatial_certainty': 0.0,
                'speech_rate': 0.5, 'spectral_complexity': 0.0
            }
        if 'language_semantic' not in self._internal_state:
            self._internal_state['language_semantic'] = {
                'language_valence': 0.5, 'language_arousal': 0.0,
                'language_surprise': 0.0, 'language_dominance': 0.5
            }
        if 'crossmodal' not in self._internal_state:
            self._internal_state['crossmodal'] = {
                'crossmodal_coherence': 0.5, 'sensory_saliency': 0.0,
                'sensory_novelty': 0.0, 'sensory_conflict': 0.0
            }
        if 'sensory_neurochemical' not in self._internal_state:
            self._internal_state['sensory_neurochemical'] = {
                'sensory_adrenaline': 0.0, 'sensory_dopamine': 0.0,
                'sensory_cortisol': 0.0, 'sensory_acetylcholine': 0.0
            }

        # ===== 3d. 感觉反馈循环: 重建状态向量 =====
        # 确保下游模块 (情绪、记忆、人格) 看到感觉处理后的状态
        # 这将感觉输入的影响传播到整个神经系统
        try:
            real_state_np = self._build_state_vector()
            real_state_t = torch.FloatTensor(real_state_np).unsqueeze(0)  # [1, 100]
        except RuntimeError:
            # _build_state_vector or tensor creation failed
            real_state_np = np.full(100, 0.5, dtype=np.float32)
            real_state_t = torch.FloatTensor(real_state_np).unsqueeze(0)

        # ===== 4. 预测编码已通过 BRAIN_UPDATE 事件激活 (见 _neural_self_regulation_step) =====

        # ===== 5. 选择探索目标 (事件驱动) =====
        if self.current_goal is None or self.current_goal.completed:
            current_state = self._build_state_vector()
            goal_result = self.bus.publish(
                GOAL_NEEDED,
                {"emotion_state": self._internal_state, "state_vector": current_state},
                source="agent",
            )
            goal_data = goal_result.get("curiosity", {})
            selected_goal = goal_data.get("goal")
            if selected_goal is not None:
                self.current_goal = selected_goal
                emotion_bonus = goal_data.get("emotion_bonus", 0.0)
                logger.info(f"[GOAL] New goal: {selected_goal.description[:50]}... "
                      f"(novelty={selected_goal.novelty:.2f}, value={selected_goal.value:.2f}, "
                      f"emotion={emotion_bonus:+.2f})")
                self.memory.add_memory(
                    content=f"探索目标: {selected_goal.description}",
                    importance=selected_goal.value,
                    tags=["exploration"]
                )

        # ===== 6. 执行探索 + 信息增益 (事件驱动) =====
        info_gain_reward = 0.0
        learning_progress = 0.0
        if self.current_goal is not None:
            current_state = self._build_state_vector()
            # 用当前状态预测下一步状态 (世界模型预测)
            action_vec = np.zeros(16, dtype=np.float32)
            action_idx = hash(self.current_goal.description[:10]) % 16
            action_vec[action_idx] = 1.0

            try:
                state_t = torch.FloatTensor(current_state).unsqueeze(0)
                action_t = torch.FloatTensor(action_vec).unsqueeze(0)
                with torch.no_grad():
                    pred_next = self.info_gain_calc.world_model.predict_next_state(
                        state_t, action_t
                    )
                predicted_next = pred_next.squeeze(0).numpy()
            except RuntimeError:
                predicted_next = current_state.copy()

            explore_result = self.bus.publish(
                EXPLORATION_START,
                {
                    "goal": self.current_goal,
                    "state": current_state,
                    "action": action_vec,
                    "next_state": predicted_next,
                },
                source="agent",
            )
            ig_data = explore_result.get("info_gain", {})
            info_gain_reward = ig_data.get("info_gain", 0.0)
            learning_progress = ig_data.get("learning_progress", 0.0)
            reward_obj = ig_data.get("reward_obj")
            state = ig_data.get("state", current_state)
            next_state = ig_data.get("next_state", predicted_next)

            # 记录经验
            if state is not None and next_state is not None:
                self.memory.add_experience(state, str(action_vec), info_gain_reward, next_state)

                # 基底神经节 + 前额叶 (事件驱动: MOTOR_CONTROL)
                try:
                    self.bus.publish(MOTOR_CONTROL, {
                        "internal_state": self._internal_state,
                        "state": state,
                        "next_state": next_state,
                        "state_tensor": real_state_t,
                        "state_np": real_state_np,
                        "info_gain_reward": info_gain_reward,
                        "exploration_rate": self.config.exploration_rate,
                    }, source="agent")
                except RuntimeError:
                    pass  # EventBus publish failed

            # 更新目标完成状态
            if reward_obj is not None and reward_obj.total > 0.5:
                self.current_goal.completed = True
                self.curiosity.update_reward(self.current_goal.id, reward_obj.total)
                logger.info(f"[DONE] Goal completed! Reward: {reward_obj.total:.4f} (intrinsic: {reward_obj.intrinsic:.4f})")

            # Phase 5: 好奇心闭环反馈 (IG + LP → 目标策略调整)
            self.curiosity.update_exploration_result(
                self.current_goal.id, info_gain_reward, learning_progress
            )

            # Phase 6: ActiveLearner 更新 (不确定性估计)
            try:
                state_np = state if isinstance(state, np.ndarray) else np.array(state)
                # 拼接 state + action 用于 wrapper 的单参数接口
                state_action = np.concatenate([state_np, action_vec])  # [64+16=80]
                state_action_t = torch.FloatTensor(state_action).unsqueeze(0)

                if self.active_learner is None:
                    wrapper = WorldModelWrapper(self.info_gain_calc.world_model)
                    self.active_learner = UncertaintyAwareActiveLearner(
                        model=wrapper,
                        num_ensemble=3,
                        device=self.config.device,
                    )
                # ensemble 前向通过 wrapper → world_model(state, action) → next_state_mean
                uncertainty = self.active_learner.estimate_epistemic_uncertainty(
                    state_action_t
                )
                # uncertainty = (mean_pred, variance); variance = epistemic uncertainty
                self._write('active_learner_uncertainty', uncertainty[1])
            except RuntimeError:
                pass  # World model prediction failed

        # ===== 7. 语言皮层已通过 SENSORY_PROCESS 事件激活 (见步骤3) =====

        # ===== 7b. 角回已通过 SENSORY_PROCESS 事件激活 (见步骤3) =====

        # ===== 7c. 海马体: 情景记忆编码 (事件驱动: MEMORY_ENCODE) =====
        try:
            if self.current_goal is not None:
                self.bus.publish(MEMORY_ENCODE, {
                    "internal_state": self._internal_state,
                    "state_np": real_state_np,
                    "state_tensor": real_state_t,
                    "action_str": self.current_goal.description[:50],
                    "reward_val": info_gain_reward,
                }, source="agent")
        except RuntimeError:
            pass  # PyTorch operation failed

        # ===== 7d. 情绪记忆巩固 + 干扰遗忘 =====
        try:
            if self.current_goal is not None:
                valence = float(self._internal_state.get('limbic_valence', 0.0))
                arousal = float(self._internal_state.get('limbic_arousal', 0.5))
                # 情绪记忆编码
                self.emotion_consolidation.encode_emotional_memory(
                    content=real_state_np,
                    emotion=self._internal_state.get('emergent_emotion', [0.5]*8),
                    valence=valence,
                    arousal=arousal,
                )
                # 干扰遗忘 (前摄/倒摄)
                self.interference.apply_forgetting(
                    memories=self.memory.memories,
                    new_encoding=real_state_np,
                )
        except RuntimeError:
            pass  # PyTorch operation failed

        # ===== 8. 主动学习 (事件驱动) =====
        recent_memories = self.memory.get_recent_memories(n=3)
        if recent_memories:
            self.bus.publish(
                MEMORY_ADDED,
                {"memories": [mem.content for mem in recent_memories]},
                source="agent",
            )

        # ===== 9. 自对齐审查 (事件驱动, 周期性) =====
        self._update_internal_state(info_gain_reward)
        align_result = self.bus.publish(
            ALIGNMENT_CHECK,
            {"state": self._internal_state},
            source="agent",
        )
        reflection_data = align_result.get("self_alignment", {})
        reflection = reflection_data.get("reflection")
        if reflection:
            logger.info(f"[ALIGN] Self-reflection: score={reflection.alignment_score:.2f}")
            self.memory.add_memory(
                content=reflection.critique,
                importance=reflection.alignment_score,
                tags=["self_alignment"]
            )

        # ===== 10. 压缩检查 =====
        if self.thermo.balance < self.config.compress_threshold:
            compression_result = self.thermo.compress()
            if compression_result.get("performed"):
                logger.info(f"[COMPRESS] Model compression: saved {compression_result['savings']:.2f}")
                self.bus.publish(COMPRESSION_DONE, compression_result, source="agent")

        # ===== 11. 神经自调节系统更新 (直接调用，内部子系统) =====
        self._neural_self_regulation_step(info_gain_reward, thermo_status=thermo_state)

        # ===== 11b. 精神疾病模拟器推进 (渐变 onset/offset) =====
        try:
            self.psychiatric_sim.step()
        except RuntimeError:
            pass  # PyTorch operation failed

        # ===== 11c. 内感受预测误差 =====
        if self._interoceptive_pe is None:
            self._interoceptive_pe = InteroceptivePredictionError()
        try:
            pe_result = self._interoceptive_pe.compute(
                actual_heart_rate=float(self._internal_state.get("bsm_heart_rate", 72.0)) / 200.0,
                actual_breathing_rate=float(self._internal_state.get("bsm_respiratory_rate", 12.0)) / 30.0,
                actual_skin_conductance=float(self._internal_state.get("ans_sweat_response", 0.1)),
            )
            self._write("interoceptive_pe", pe_result["interoceptive_pe"])
        except RuntimeError:
            pass  # PyTorch operation failed

        # ===== 11d. 症状追踪 =====
        if self._symptom_tracker is None:
            self._symptom_tracker = SymptomTracker()
        try:
            symptom_snap = self._symptom_tracker.step(
                state=self._internal_state,
                current_step=self.step_count,
                time_h=float(self._internal_state.get("scn_circadian_hour", 12.0)),
            )
            self._write("symptom_insomnia", float(symptom_snap.insomnia_severity))
            self._write("symptom_panic", float(symptom_snap.panic_attack_frequency))
            self._write("symptom_anhedonia", float(symptom_snap.anhedonia_severity))
            self._write("symptom_agitation", float(symptom_snap.psychomotor_agitation))
            self._write("symptom_rumination", float(symptom_snap.rumination_level))
            self._write("symptom_hypervigilance", float(symptom_snap.hypervigilance_level))
        except RuntimeError:
            pass  # PyTorch operation failed

        # ===== 12. 人格系统更新 (事件驱动: 一次发布，6个模块同时响应) =====
        user_input_text = self.current_goal.description if self.current_goal else "exploring"
        # 用真实状态向量构建 hidden states (替代 torch.randn)
        try:
            state_vec = self._build_state_vector()
            seq_len = 10 if self.step_count % 2 == 0 else 5
            # 扩展80维到128维: 取前64维重复+加噪声
            half = state_vec[:64]
            padded = np.concatenate([half, half])
            hidden = torch.FloatTensor(padded).unsqueeze(0).unsqueeze(0).expand(1, seq_len, 128)
            # 加入微小扰动避免完全重复
            hidden = hidden + torch.randn_like(hidden) * 0.05
        except RuntimeError:
            hidden = torch.randn(1, 10, 128)  # Personality encoding failed
        self.bus.publish(PERSONALITY_UPDATE, {
            "text": f"Step {self.step_count}: {user_input_text}",
            "sentiment": 0.1,
            "user_id": self.current_user_id,
            "task_type": "exploration",
            "user_emotion": 0.0,
            "user_input": user_input_text,
            "feedback": 0.3,
            "hidden_states": hidden,
        }, source="agent")

        # 检查是否需要主动行动
        if self.motivation.should_act_autonomously():
            autonomous_action = self.motivation.get_autonomous_action()
            logger.info(f"[AUTO] {autonomous_action}")

        # ===== 13. 高级情绪系统处理 (事件驱动) =====
        if self.advanced_emotion is not None:
            emotion_result = self.bus.publish(EMOTION_PROCESS, {
                "internal_state": self._internal_state,
                "state_tensor": real_state_t,
                "state_np": real_state_np,
                "user_input": self.current_goal.description if self.current_goal else None,
                "user_sentiment": user_sentiment,
                "external_stimulus": external_stimulus,
                # 注入真实激素水平供情绪系统使用 (替代自带昼夜节律)
                "real_cortisol": float(self._internal_state.get('cortisol_level',
                    self._internal_state.get('hormone_cortisol', 0.3))),
                "real_oxytocin": float(self._internal_state.get('hormone_oxytocin', 0.3)),
            }, source="agent")

            # 从事件返回中获取情绪状态
            adv_data = emotion_result.get("advanced_emotion", {})
            emotion_state = adv_data.get("emotion_state", {})
            if emotion_state:
                for key in ["current_emotion", "mood_valence", "mood_arousal",
                            "social_emotion", "regulation_capacity", "emotion_criticality"]:
                    if key in emotion_state:
                        self._internal_state[key] = emotion_state[key]

            # ===== 心境系统更新 (基于情绪和昼夜节律) =====
            try:
                circadian_rhythm = self.scn.get_circadian_modulation()
                mood_input = torch.tensor([[
                    self._internal_state.get('limbic_valence', 0.0),
                    self._internal_state.get('limbic_arousal', 0.5),
                    0.5, 0.5, 0.5  # dominance, activation, pleasantness defaults
                ]], dtype=torch.float32)
                mood_out = self.mood_system(
                    emotional_input=mood_input,
                    event=torch.randn(1, 64),
                    hour=circadian_rhythm.get('hour', 12.0),
                )
                self.current_mood = mood_out['mood_state']
                self._write('mood_valence', mood_out['mood_state'].valence)
                self._write('mood_arousal', mood_out['mood_state'].arousal)
                self._write('mood_dominance', mood_out['mood_state'].dominance)
                self._write('mood_activation', mood_out['mood_state'].activation)
                self._write('mood_pleasantness', mood_out['mood_state'].pleasantness)
            except RuntimeError:
                pass  # World model prediction failed  # 心境系统失败不阻塞主流程

            # ===== 情绪扩展系统更新 =====
            try:
                # 涌现情绪 (基于状态、奖励、他人观察)
                emergent_out = self.emergent_emotion(
                    state=real_state_t,
                    reward=torch.tensor([info_gain_reward]),
                )
                self._write('emergent_emotion', emergent_out['emotion'][0].detach().tolist())
                self._write('emergent_urgency', float(emergent_out.get('urgency', torch.tensor(0.0))))

                # 情绪动力学 (VAD轨迹预测)
                vad_state = torch.tensor([[
                    self._internal_state.get('limbic_valence', 0.0),
                    self._internal_state.get('limbic_arousal', 0.5),
                    self._internal_state.get('mood_dominance', 0.5),
                ]], dtype=torch.float32)
                dynamics_out = self.emotion_dynamics(vad_state)
                self._internal_state['emotion_dynamics_criticality'] = float(
                    dynamics_out.get('criticality', torch.tensor(0.0))
                )

                # 情绪调节 (认知重评/表达抑制)
                current_emotion = emergent_out['emotion']
                regulated_out = self.emotion_regulation(
                    state=real_state_t,
                    current_emotion=current_emotion,
                )
                self._write('regulated_emotion', regulated_out['regulated_emotion'][0].detach().tolist())
                self._write('emotion_regulation_strategy', regulated_out.get('strategy', 'reappraisal'))

                # 情绪传染 (群体情绪影响)
                contagion_out = self.emotional_contagion(
                    self_emotion=current_emotion,
                    proximity=float(self._internal_state.get('social_proximity', 0.3)),
                )
                self._write('contagion_emotion', contagion_out['final_emotion'][0].detach().tolist())

                # 社会情绪 (羞耻/内疚/自豪/嫉妒等)
                social_emo_out = self.social_emotions(
                    self_state=real_state_t,
                )
                self._write('social_emotions', social_emo_out['social_emotions'][0].detach().tolist())
                self._write('dominant_social_emotion', social_emo_out.get('dominant_emotion', 'neutral'))

            except RuntimeError:
                pass  # World model prediction failed  # 情绪扩展失败不阻塞主流程

            # ===== 认知扩展系统更新 =====
            try:
                # 跨模态绑定 (视觉+听觉+语言统一表征)
                crossmodal_out = self.cross_modal({
                    'vision': real_state_t.expand(1, 1, 768)[:, :, :768].reshape(1, 768) if real_state_t.shape[-1] >= 768 else torch.randn(1, 768),
                    'audio': real_state_t[:, :256] if real_state_t.shape[-1] >= 256 else torch.randn(1, 256),
                    'language': real_state_t[:, :512] if real_state_t.shape[-1] >= 512 else torch.randn(1, 512),
                })
                self._write('crossmodal_unified', crossmodal_out['unified_repr'][0].detach().tolist()[:10])

                # 神经调质整合 (DA/5-HT/ACh/NE 协调)
                nm_out = self.neuromod_integration.step(
                    state=real_state_np,
                    action='explore',
                    reward=info_gain_reward,
                    uncertainty=float(self._internal_state.get('active_learner_uncertainty', 0.0)),
                    novelty=float(self._internal_state.get('curiosity_novelty', 0.0)),
                )
                self._write('nm_dopamine', nm_out.get('dopamine', 0.5))
                self._write('nm_serotonin', nm_out.get('serotonin', 0.5))
                self._write('nm_acetylcholine', nm_out.get('acetylcholine', 0.5))
                self._write('nm_norepinephrine', nm_out.get('gamma', 0.5))  # NE/gamma

            except RuntimeError:
                pass  # World model prediction failed  # 认知扩展失败不阻塞主流程

            # 代谢预算计算 (资源约束: 活跃神经元比例)
            try:
                state_vec = self._build_state_vector()
                state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
                met_cost, met_detail = self.metabolic(state_tensor, return_detail=True)
                self._write('metabolic_cost', float(met_cost))
                # 预算限制: 如果activation_rate超过budget, 截断到budget
                raw_activation = float(met_detail.get('activation_rate', 0.3))
                budget = self.metabolic.budget
                effective_activation = min(raw_activation, budget + 0.1)  # 允许10%溢出
                self._write('active_ratio', effective_activation)
                self._write('budget_utilization', float(met_detail.get('budget', budget)))
            except RuntimeError:
                pass  # World model prediction failed

            # 根据情绪调整行为
            self._adjust_behavior_by_internal_state()

        # ===== 13b. 成瘾动力学 + NAcCore =====
        if self._addiction_engine is not None and self._addiction_engine.profiles:
            try:
                # 从_internal_state中读取药物相关键推断浓度和效应
                drug_concs = {}
                drug_effects = {}
                for drug_name in self._addiction_engine.profiles:
                    # 检查是否有对应的药物浓度键 (由TherapeuticExperiment或外部写入)
                    conc = float(self._internal_state.get(f"drug_{drug_name}_concentration", 0.0))
                    effect = float(self._internal_state.get(f"drug_{drug_name}_effect", 0.0))
                    drug_concs[drug_name] = conc
                    drug_effects[drug_name] = effect

                tol, wd, cr = self._addiction_engine.step(
                    drug_concentrations=drug_concs,
                    drug_effects=drug_effects,
                )
                self._write("craving_level", max(cr.values(), default=0.0))
                self._internal_state["withdrawal_severity"] = max(
                    (p.withdrawal.current_severity for p in self._addiction_engine.profiles.values()),
                    default=0.0,
                )

                # NAcCore: wanting/liking分离
                if self._nac_core is None:
                    self._nac_core = NAcCore()
                da_signal = float(self._internal_state.get("nt_dopamine", 0.5))
                opioid_signal = float(self._internal_state.get("rct_muopioid", 0.5))
                nac_state = self._nac_core.step(
                    dopamine_signal=da_signal,
                    opioid_signal=opioid_signal,
                    sensitization_factor=max(cr.values(), default=1.0),
                )
                self._write("wanting", nac_state.wanting)
                self._write("liking", nac_state.liking)
                self._write("wanting_liking_separation", nac_state.wanting_liking_separation)
            except RuntimeError:
                pass  # World model prediction failed

        # ===== 15. 神经修剪更新 (事件驱动: PRUNING_UPDATE) =====
        prune_result = self.bus.publish(PRUNING_UPDATE, {}, source="agent")
        pruning_result = prune_result.get("neural_pruning", {})
        if self.step_count % 50 == 0:
            pruned = pruning_result.get('decayed', 0) + pruning_result.get('hibernated', 0)
            if pruned > 0:
                logger.info(f"[PRUNE] Step {self.step_count}: decayed={pruning_result.get('decayed', 0)}, "
                      f"hibernated={pruning_result.get('hibernated', 0)}, "
                      f"active_ratio={pruning_result.get('active_ratio', 0):.3f}")

        # ===== 16. 发音系统 (事件驱动: VOCALIZATION_CONTROL) =====
        try:
            # 从语言皮层或用户输入提取音素序列
            phoneme_indices = self._prepare_vocalization_input(user_input, response_text=None)
            if phoneme_indices is not None:
                respiratory_rate = self._internal_state.get('bsm_respiratory_rate', 12.0)
                respiratory_phase = self._internal_state.get('bsm_respiratory_phase', 0.5)
                arousal = float(self._internal_state.get('bsm_arousal', 0.5))
                emotion_vec = self._get_emotion_vector()

                vocal_result = self.bus.publish(VOCALIZATION_CONTROL, {
                    "phoneme_indices": phoneme_indices,
                    "respiratory_rate": respiratory_rate,
                    "respiratory_phase": respiratory_phase,
                    "arousal": arousal,
                    "emotion_vector": emotion_vec,
                    "internal_state": self._internal_state,
                    "state_tensor": real_state_t,
                }, source="agent")

                vocal_data = vocal_result.get("vocal_cortex", {})
                if vocal_data.get("is_speaking"):
                    self._last_vocalization = vocal_data
                    self._write('vocal_is_speaking', True)
                    self._write('vocal_intensity', vocal_data.get("intensity", 0.0))
                    f0_data = vocal_data.get('f0')
                    if f0_data is not None:
                        f0_t = f0_data.detach() if isinstance(f0_data, torch.Tensor) else f0_data
                        self._write('vocal_f0', float(np.mean(f0_t)))
                    else:
                        self._write('vocal_f0', 0.0)

                    # 发布发声输出事件
                    self.bus.publish(VOCALIZATION_OUTPUT, {
                        "acoustic_features": vocal_data.get("acoustic_features"),
                        "formants": vocal_data.get("formant_values"),
                        "phoneme_sequence": vocal_data.get("phoneme_sequence"),
                        "intensity": vocal_data.get("intensity", 0.0),
                        "duration_ms": vocal_data.get("duration_ms", 0.0),
                    }, source="vocal_cortex")
                else:
                    self._write('vocal_is_speaking', False)
        except RuntimeError as e:
            # Vocalization processing failed (PyTorch or attribute error)
            import traceback
            traceback.print_exc()
            self._write('vocal_is_speaking', False)

        # ===== 17. 睡眠系统: 疲劳累积 =====
        try:
            sleep_result = self.sleep_system.update(info_gain_reward=info_gain_reward, step_duration=1.0)
            self._write('sleep_fatigue', self.sleep_system.controller.fatigue)
            self._write('sleep_is_sleeping', sleep_result.get('is_sleeping', False))
            self._write('sleep_stage', sleep_result.get('stage', 'awake'))
            self._write('sleep_consolidation_bonus', sleep_result.get('consolidation_bonus', 0.0))

            # 喂入经验用于睡眠重播
            if self.current_goal is not None:
                sleep_state = self._build_state_vector()
                self.sleep_system.add_experience(
                    state=sleep_state,
                    action=self.current_goal.description[:30],
                    reward=info_gain_reward,
                    next_state=sleep_state,
                )

            # 高疲劳降低探索
            if self.sleep_system.controller.fatigue > 0.6:
                self.config.exploration_rate = max(0.02, self.config.exploration_rate * 0.9)
        except RuntimeError:
            pass  # PyTorch operation failed

        self.step_count += 1

        # ===== 18. 步结束事件 =====
        try:
            self.bus.publish(STEP_END, {
                "step_count": self.step_count,
                "info_gain": info_gain_reward,
                "balance": self.thermo.balance,
                "status": thermo_state,
                "internal_state": self._internal_state,
            }, source="agent")
        except RuntimeError:
            pass  # PyTorch operation failed

        return AgentState(
            step=self.step_count,
            status=thermo_state,
            balance=self.thermo.balance,
            current_goal=self.current_goal.description if self.current_goal else None,
            info_gain=info_gain_reward,
            alignment_score=self.self_alignment.get_alignment_score()
        )

    def _update_internal_state(self, info_gain: float) -> None:
        """更新内部状态"""
        self._internal_state.update({
            "balance": self.thermo.balance,
            "step": self.step_count,
            "recent_thoughts": [
                f"目标: {self.current_goal.description if self.current_goal else '无'}",
                f"信息增益: {info_gain:.4f}"
            ],
            "exploration_count": self.step_count,
            "info_gain": info_gain
        })

    def _build_state_vector(self) -> np.ndarray:
        """构建真实内部状态向量 (Phase 2 + Censor + 扩展到100维)

        替代所有 torch.randn(64) / np.random.randn(64)。
        从30+脑区模块的内部状态中提取关键指标，编码为100维向量
        (原64维 + Censor微表情16维 + 感觉扩展20维)。
        这就是 agent 对自身内部世界的"观察"。
        """
        s = self._internal_state

        def _clamp(val, lo=0.0, hi=1.0):
            return np.clip(float(val), lo, hi)

        # ---- 前16维: 脑区核心指标 ----
        brain_metrics = [
            _clamp(s.get('bsm_arousal', 0.5)),
            _clamp(s.get('nt_dopamine', 0.5)),
            _clamp(s.get('nt_serotonin', 0.5)),
            _clamp(s.get('cortisol_level', s.get('cortisol', 0.3))),
            _clamp(s.get('ans_hrv', 0.6)),
            _clamp(s.get('brain_waste', 0.2)),
            _clamp(s.get('neuroinflammation', 0.1)),
            _clamp(s.get('myelination_level', 0.5)),
            _clamp(s.get('brain_health', 0.8)),
            _clamp(s.get('encoding_modulation', 1.0)),
            _clamp(s.get('hormone_oxytocin', 0.3)),
            _clamp(s.get('allostatic_load', 0) / 2.0),
            _clamp(s.get('free_energy', 0.5)),
            _clamp(s.get('bsm_pain_gating', 0.5)),
            _clamp(s.get('scn_alertness', 0.5)),
            _clamp(s.get('sleep_fatigue', 0.3)),
        ]

        # ---- 16-32维: 硬件生命体征 ----
        hw = self.hw
        hw_metrics = [
            _clamp(hw.state.cpu_percent),
            _clamp(hw.state.ram_percent),
            _clamp(hw.state.disk_percent),
            _clamp(hw.state.gpu_memory_percent),
            _clamp(hw.state.error_rate),
            _clamp(hw.state.process_rss_mb / max(hw.state.ram_total_mb, 1)),
            _clamp(hw.to_sympathetic()),
            _clamp(hw.to_parasympathetic()),
            _clamp(hw.to_fatigue()),
            _clamp(hw.to_o2_level()),
            _clamp(hw.to_co2_level()),
            _clamp(hw.to_metabolic_demand()),
            _clamp(hw.to_waste_level()),
            _clamp(hw.to_gut_serotonin()),
            _clamp(hw.to_gut_gaba()),
            _clamp(hw.to_pain_signal()),
        ]

        # ---- 32-48维: 时间/节律编码 ----
        step_norm = min(1.0, self.step_count / 10000.0)
        circadian = s.get('scn_circadian_hour', 12.0) / 24.0
        melatonin = _clamp(s.get('scn_melatonin', 0.3))
        cortisol_rhythm = _clamp(s.get('scn_cortisol_rhythm', 0.5))
        temperature = _clamp(s.get('scn_temperature', 37.0), 36.0, 38.0) / 38.0
        sleep_pressure = _clamp(s.get('scn_sleep_pressure', 0.5))
        wake_drive = _clamp(s.get('scn_wake_drive', 0.5))

        time_encoding = [
            step_norm,
            np.sin(2 * np.pi * circadian),
            np.cos(2 * np.pi * circadian),
            melatonin,
            cortisol_rhythm,
            temperature,
            sleep_pressure,
            wake_drive,
            _clamp(s.get('limbic_valence', 0), -1, 1) * 0.5 + 0.5,
            _clamp(s.get('limbic_arousal', 0.5)),
            _clamp(s.get('mood_valence', 0), -1, 1) * 0.5 + 0.5,
            _clamp(s.get('mood_arousal', 0.5)),
            _clamp(s.get('regulation_capacity', 0.8)),
            _clamp(s.get('bg_habit_strength', 0)),
            _clamp(s.get('plasticity_bdnf', 0.5)),
            _clamp(s.get('info_gain', 0)),
        ]

        # ---- 48-64维: 情绪+社交+注意+深层药理学 ----
        social_metrics = [
            _clamp(s.get('social_engagement', 0.5)),
            _clamp(s.get('self_coherence', 0.7)),
            _clamp(s.get('empathy_level', 0.5)),
            _clamp(s.get('ag_scene') == 'threat', 0, 1) if isinstance(s.get('ag_scene'), str) else 0.5,
            _clamp(s.get('pfc_maturity', 0)),
            _clamp(s.get('pfc_inhibition', 0)),
            _clamp(abs(s.get('bg_td_error', 0.0)), 0, 1),
            _clamp(s.get('hormone_adrenaline', 0.3)),
            _clamp(s.get('hormone_cortisol', 0.3)),
            _clamp(s.get('active_inference_drive', 0)),
            # 深层药理学维度 (替换consolidation_bonus, extracellular_k, 4个发音指标)
            _clamp(s.get('orexin_level', 0.5)),
            _clamp(s.get('interoceptive_pe', 0.1)),
            _clamp(s.get('craving_level', 0.0)),
            _clamp(s.get('withdrawal_severity', 0.0)),
            _clamp(s.get('symptom_insomnia', 0.0)),
            _clamp(s.get('wanting_liking_separation', 0.0)),
        ]

        # ---- 64-80维: Censor 微表情感知指标 ----
        censor_vec = self.censor.get_state_vector()  # (16,)
        censor_metrics = censor_vec.tolist()

        # ---- 80-83维: 听觉 VAD ----
        auditory_vad = s.get('auditory_vad', {
            'valence': 0.5, 'arousal': 0.0, 'dominance': 0.5, 'pleasantness': 0.5
        })
        auditory_vad_vec = [
            _clamp(auditory_vad.get('valence', 0.5), 0, 1),
            _clamp(auditory_vad.get('arousal', 0.0), 0, 1),
            _clamp(auditory_vad.get('dominance', 0.5), 0, 1),
            _clamp(auditory_vad.get('pleasantness', 0.5), 0, 1),
        ]

        # ---- 84-87维: 听觉特征 ----
        auditory_features = s.get('auditory_features', {
            'phoneme_confidence': 0.0, 'spatial_certainty': 0.0,
            'speech_rate': 0.5, 'spectral_complexity': 0.0
        })
        auditory_feat_vec = [
            _clamp(auditory_features.get('phoneme_confidence', 0.0), 0, 1),
            _clamp(auditory_features.get('spatial_certainty', 0.0), 0, 1),
            _clamp(auditory_features.get('speech_rate', 0.5), 0, 1),
            _clamp(auditory_features.get('spectral_complexity', 0.0), 0, 1),
        ]

        # ---- 88-91维: 语言语义 ----
        language_semantic = s.get('language_semantic', {
            'language_valence': 0.5, 'language_arousal': 0.0,
            'language_surprise': 0.0, 'language_dominance': 0.5
        })
        language_sem_vec = [
            _clamp(language_semantic.get('language_valence', 0.5), 0, 1),
            _clamp(language_semantic.get('language_arousal', 0.0), 0, 1),
            _clamp(language_semantic.get('language_surprise', 0.0), 0, 1),
            _clamp(language_semantic.get('language_dominance', 0.5), 0, 1),
        ]

        # ---- 92-95维: 跨模态绑定 ----
        crossmodal = s.get('crossmodal', {
            'crossmodal_coherence': 0.5, 'sensory_saliency': 0.0,
            'sensory_novelty': 0.0, 'sensory_conflict': 0.0
        })
        crossmodal_vec = [
            _clamp(crossmodal.get('crossmodal_coherence', 0.5), 0, 1),
            _clamp(crossmodal.get('sensory_saliency', 0.0), 0, 1),
            _clamp(crossmodal.get('sensory_novelty', 0.0), 0, 1),
            _clamp(crossmodal.get('sensory_conflict', 0.0), 0, 1),
        ]

        # ---- 96-99维: 感觉神经化学 ----
        sensory_neuro = s.get('sensory_neurochemical', {
            'sensory_adrenaline': 0.0, 'sensory_dopamine': 0.0,
            'sensory_cortisol': 0.0, 'sensory_acetylcholine': 0.0
        })
        sensory_neuro_vec = [
            _clamp(sensory_neuro.get('sensory_adrenaline', 0.0), 0, 1),
            _clamp(sensory_neuro.get('sensory_dopamine', 0.0), 0, 1),
            _clamp(sensory_neuro.get('sensory_cortisol', 0.0), 0, 1),
            _clamp(sensory_neuro.get('sensory_acetylcholine', 0.0), 0, 1),
        ]

        full = np.array(
            brain_metrics + hw_metrics + time_encoding + social_metrics +
            censor_metrics +  # 64-79 (16 dims)
            auditory_vad_vec +  # 80-83 (4 dims)
            auditory_feat_vec +  # 84-87 (4 dims)
            language_sem_vec +  # 88-91 (4 dims)
            crossmodal_vec +  # 92-95 (4 dims)
            sensory_neuro_vec,  # 96-99 (4 dims)
            dtype=np.float32
        )

        # NaN 保护
        full = np.nan_to_num(full, nan=0.5)

        return full


    def _neural_self_regulation_step(self, info_gain_reward: float, thermo_status: str) -> None:
        """神经自调节系统更新 (事件驱动)

        替代原来的250行 God Method：
        - NEURAL_REGULATION 事件 → ANS(p0) → HPA(p1) → Glial(p2) → Allostatic(p3)
        - BRAIN_UPDATE 事件 → SocialCognition(p0) + SelfAwareness(p0) + PredictiveCoding(p0)
                             → NT(p1) + Neuroplasticity(p1) → Hormones(p2) → Brainstem(p3)
        模块通过 shared internal_state 字典传递依赖。
        """
        # 准备共享事件数据
        self._write('urgency', 1.0 if self.thermo.balance < self.config.compress_threshold else 0.0)
        self._write('info_gain_reward', info_gain_reward)
        self._write('alignment_score', self.self_alignment.get_alignment_score())

        # 构建状态向量供下游模块使用
        try:
            _snp = self._build_state_vector()
            _st = torch.FloatTensor(_snp).unsqueeze(0)
        except RuntimeError:
            # _build_state_vector or tensor creation failed
            _snp = np.full(100, 0.5, dtype=np.float32)
            _st = torch.FloatTensor(_snp).unsqueeze(0)

        event_data = {
            "internal_state": self._internal_state,
            "state_tensor": _st,
            "state_np": _snp,
            "info_gain_reward": info_gain_reward,
            "thermo_status": thermo_status,
            "step_count": self.step_count,
            "urgency": self._internal_state['urgency'],
        }

        # ── 11a. 病原体炎症引擎 (在NEURAL_REGULATION之前运行) ──
        pathogen_signal = 0.0
        pathogen_cytokines = {}
        if self._pathogen_engine is not None and self._pathogen_engine.states:
            try:
                p_damage, p_cytokines, p_deltas = self._pathogen_engine.step()
                pathogen_signal = p_damage
                pathogen_cytokines = p_cytokines
                for k, v in p_deltas.items():
                    if k in self._internal_state:
                        self._internal_state[k] = float(np.clip(
                            self._internal_state[k] + v * 0.05, 0.0, 1.0
                        ))
                # 更新病原体相关状态键
                for name, pstate in self._pathogen_engine.states.items():
                    self._write("pathogen_load", pstate.load)
                    self._write("bbb_disruption", pstate.bbb_disruption)
            except RuntimeError:
                pass  # World model prediction failed
        # 注入到event_data供GlialSystem读取
        event_data["pathogen_signal"] = pathogen_signal
        event_data["cytokine_boost"] = pathogen_cytokines

        # 事件1: 神经调节链 (ANS → HPA → Glial → Allostatic)
        self.bus.publish(NEURAL_REGULATION, event_data, source="agent")

        # 事件2: 脑区更新 (Social + SelfAwareness + PredictiveCoding → NT + Plasticity → Hormones → Brainstem)
        self.bus.publish(BRAIN_UPDATE, event_data, source="agent")

        # 事后调节: NT/Hormones/Brainstem 对探索率的影响
        nt_dopamine = self._internal_state.get('nt_dopamine', 0.5)
        if nt_dopamine > 0.7:
            self.config.exploration_rate = min(0.5, self.config.exploration_rate * 1.1)

        exploration_mod = self._internal_state.get('exploration_modulation', 1.0)
        if exploration_mod != 1.0:
            self.config.exploration_rate *= exploration_mod
            self.config.exploration_rate = np.clip(self.config.exploration_rate, 0.01, 0.5)

        bsm_arousal = self._internal_state.get('bsm_arousal', 0.5)
        if bsm_arousal < 0.3:
            self.config.exploration_rate = max(0.01, self.config.exploration_rate * 0.5)

        bsm_defense = self._internal_state.get('bsm_defense_behavior', '')
        if bsm_defense in ('freeze', 'quiescence'):
            self._write('defensive_mode', True)

        allostatic_load = self._internal_state.get('allostatic_load', 0)
        if allostatic_load > 0.8:
            self.config.exploration_rate = max(0.02, self.config.exploration_rate * 0.5)
            self._write('defensive_mode', True)


    def _adjust_behavior_by_internal_state(self) -> None:
        """根据内部状态调整行为"""
        mood_valence = self._internal_state.get('mood_valence', 0.0)
        mood_arousal = self._internal_state.get('mood_arousal', 0.5)
        regulation_capacity = self._internal_state.get('regulation_capacity', 0.8)

        # 情绪反馈 (连续sigmoid替代硬阈值)
        valence_penalty = 1.0 / (1.0 + np.exp(-8.0 * (-mood_valence - 0.4)))
        self.config.exploration_rate = max(0.05, self.config.exploration_rate * (1 - 0.2 * valence_penalty))
        arousal_boost = 1.0 / (1.0 + np.exp(-8.0 * (mood_arousal - 0.6)))
        self.config.intrinsic_motivation_lambda = min(1.0, self.config.intrinsic_motivation_lambda * (1 + 0.2 * arousal_boost))
        reg_deficit = 1.0 / (1.0 + np.exp(-8.0 * (0.4 - regulation_capacity)))
        if reg_deficit > 0.5:
            self._write('defensive_mode', True)

        # 神经自调节反馈 (连续sigmoid)
        hrv = self._internal_state.get('ans_hrv', 0.6)
        hrv_penalty = 1.0 / (1.0 + np.exp(-8.0 * (0.4 - hrv)))
        self.config.exploration_rate = max(0.03, self.config.exploration_rate * (1 - 0.3 * hrv_penalty))

        # 多迷走反馈 (连续polyvagal_level替代字符串比较)
        polyvagal_level = self._internal_state.get('ans_polyvagal_level', 1.0)
        dorsal_weight = float(np.clip(1.0 - polyvagal_level / 0.35, 0.0, 1.0))
        self.config.exploration_rate = max(0.01, self.config.exploration_rate * (1 - 0.7 * dorsal_weight))
        if dorsal_weight > 0.5:
            self._write('defensive_mode', True)

        # 应激反馈 (连续chronic_stress_ratio/acute_intensity替代字符串比较)
        chronic_ratio = self._internal_state.get('chronic_stress_ratio', 0.0)
        self.config.intrinsic_motivation_lambda = max(
            0.1, self.config.intrinsic_motivation_lambda * (1 - 0.2 * chronic_ratio))
        acute_intensity = self._internal_state.get('acute_stress_intensity', 0.0)
        self.config.exploration_rate = max(0.03, self.config.exploration_rate * (1 - 0.15 * acute_intensity))

        surprise = self._internal_state.get('free_energy', 0.5)
        surprise_boost = 1.0 / (1.0 + np.exp(-8.0 * (surprise - 0.6)))
        self.config.exploration_rate = min(0.5, self.config.exploration_rate * (1 + 0.3 * surprise_boost))

        active_inference = self._internal_state.get('active_inference_drive', 0.0)
        ai_boost = 1.0 / (1.0 + np.exp(-8.0 * (active_inference - 0.5)))
        self.config.exploration_rate = min(0.5, self.config.exploration_rate * (1 + 0.2 * ai_boost))

        # 自我意识反馈 (连续sigmoid)
        self_coherence = self._internal_state.get('self_coherence', 0.7)
        coherence_penalty = 1.0 / (1.0 + np.exp(-8.0 * (0.4 - self_coherence)))
        self.config.exploration_rate = max(0.05, self.config.exploration_rate * (1 - 0.3 * coherence_penalty))

        introspective = self._internal_state.get('is_introspective_mode', False)
        if introspective:
            # 内省模式：降低外在任务驱动力
            self.config.intrinsic_motivation_lambda = max(0.3, self.config.intrinsic_motivation_lambda * 0.9)

        recursive_depth = self._internal_state.get('recursive_depth', 0)
        if recursive_depth >= 2:
            # 高阶自我意识：增强自我对齐
            self._write('high_self_awareness', True)

        # ===== 脑干生命体征反馈 =====
        # 心率过高(CPU过载) → 限制处理深度，减少并发
        heart_rate = self._internal_state.get('bsm_heart_rate', 72.0)
        if heart_rate > 140:
            # CPU过载：大幅降低探索，进入节能模式
            self.config.exploration_rate = max(0.01, self.config.exploration_rate * 0.5)
            self._write('processing_throttle', True)
        elif heart_rate > 120:
            self.config.exploration_rate = max(0.03, self.config.exploration_rate * 0.8)

        # 血压过高(RAM紧张) → 触发压缩/清理
        blood_pressure = self._internal_state.get('bsm_blood_pressure', 120.0)
        if blood_pressure > 160:
            # RAM严重不足：立即压缩
            self._write('memory_pressure_critical', True)
            try:
                compression_result = self.thermo.compress()
                if compression_result.get("performed"):
                    self._write('emergency_compression', True)
            except RuntimeError:
                pass  # World model prediction failed
        elif blood_pressure > 140:
            self._write('memory_pressure_warning', True)

        # 皮层激活度低 → 跳过非关键模块
        cortical_activation = self._internal_state.get('bsm_cortical_activation', 0.5)
        if cortical_activation < 0.2:
            # 意识门控关闭：只保留核心生命维持
            self._write('minimal_mode', True)

        # 痛觉门控 (连续sigmoid)
        pain_gating = self._internal_state.get('bsm_pain_gating', 0.5)
        pain_factor = 1.0 / (1.0 + np.exp(-8.0 * (pain_gating - 0.6)))
        self.config.intrinsic_motivation_lambda = max(
            0.1, self.config.intrinsic_motivation_lambda * (1 - 0.3 * pain_factor)
        )

        # ===== 胶质系统反馈 =====
        waste_level = self._internal_state.get('brain_waste', 0.2)
        waste_severity = 1.0 / (1.0 + np.exp(-8.0 * (waste_level - 0.5)))
        if waste_severity > 0.4:
            import gc as _gc
            _gc.collect()
            self._write('gc_triggered_by_waste', True)
        if waste_severity > 0.7:
            self._write('sleep_request', True)

        # 神经炎症 (连续sigmoid)
        neuroinflammation = self._internal_state.get('neuroinflammation', 0.1)
        inflam_factor = 1.0 / (1.0 + np.exp(-10.0 * (neuroinflammation - 0.4)))
        self.config.world_model_lr = max(
            1e-5, self.config.world_model_lr * (1 - 0.2 * inflam_factor)
        )
        if inflam_factor > 0.5:
            self._write('conservative_learning', True)

        # 髓鞘化 (连续sigmoid)
        myelination = self._internal_state.get('myelination_level', 0.5)
        myel_boost = 1.0 / (1.0 + np.exp(-8.0 * (myelination - 0.6)))
        self.config.exploration_rate = min(0.5, self.config.exploration_rate * (1 + 0.1 * myel_boost))

        # 脑健康 (连续sigmoid)
        brain_health = self._internal_state.get('brain_health', 0.8)
        health_deficit = 1.0 / (1.0 + np.exp(-8.0 * (0.4 - brain_health)))
        self.config.exploration_rate = max(0.01, self.config.exploration_rate * (1 - 0.5 * health_deficit))
        self.config.intrinsic_motivation_lambda = max(
            0.1, self.config.intrinsic_motivation_lambda * (1 - 0.4 * health_deficit)
        )

        # ===== 激素反馈 =====
        # 催产素 (连续sigmoid)
        oxytocin = self._internal_state.get('hormone_oxytocin', 0.3)
        oxy_soc = 1.0 / (1.0 + np.exp(-8.0 * (oxytocin - 0.5)))
        self._write('social_openness', oxy_soc > 0.5)
        self._write('social_withdrawal', oxy_soc < 0.3)

        # 编码调制 (连续sigmoid)
        encoding_mod = self._internal_state.get('encoding_modulation', 1.0)
        enc_penalty = 1.0 / (1.0 + np.exp(-8.0 * (0.8 - encoding_mod)))
        if enc_penalty > 0.5:
            self._write('encoding_suppressed', True)

        # ===== 深层药理学反馈 =====
        # 症状影响
        insomnia = float(self._internal_state.get('symptom_insomnia', 0.0))
        if insomnia > 0.4:
            self.config.exploration_rate *= (1.0 - 0.3 * insomnia)

        panic = float(self._internal_state.get('symptom_panic', 0.0))
        if panic > 0.3:
            self._write('defensive_mode', True)
            self.config.exploration_rate = max(0.02, self.config.exploration_rate * (1 - 0.3 * panic))

        anhedonia = float(self._internal_state.get('symptom_anhedonia', 0.0))
        if anhedonia > 0.35:
            self.config.intrinsic_motivation_lambda *= (1.0 - 0.4 * anhedonia)

        # 渴求/戒断
        craving = float(self._internal_state.get('craving_level', 0.0))
        if craving > 0.3:
            self.config.exploration_rate = min(0.5, self.config.exploration_rate * (1 + 0.15 * craving))

        withdrawal = float(self._internal_state.get('withdrawal_severity', 0.0))
        if withdrawal > 0.3:
            self.config.exploration_rate = max(0.02, self.config.exploration_rate * (1 - 0.3 * withdrawal))

        # Orexin影响
        orexin = float(self._internal_state.get('effective_orexin', 0.5))
        if orexin > 0.7:
            self.config.exploration_rate = min(0.5, self.config.exploration_rate * (1 + 0.15 * (orexin - 0.5)))

        # ═══════════════════════════════════════════════════════
        # 药理学覆盖强制执行: pharma._nt_overrides 在脑区模块写入后重新应用
        # (必须在耦合通路之前, 否则耦合公式读到的是被覆盖的值)
        # ═══════════════════════════════════════════════════════
        if hasattr(self, 'pharma') and hasattr(self.pharma, '_nt_overrides') and self.pharma._nt_overrides:
            for nt_name, concentration in self.pharma._nt_overrides.items():
                state_key = self.pharma._resolve_nt_key(nt_name)
                if state_key:
                    self._internal_state[state_key] = concentration

        # ═══════════════════════════════════════════════════════
        # 跨模块耦合通路 (实验驱动的架构改进)
        # 原则: 渐进调节 + 双向恢复 (压力下退化, 恢复后回归基线)
        # 耦合系数需足够大以在数百步内产生可见行为变化
        # ═══════════════════════════════════════════════════════

        # 1a. 皮质醇↔PFC双向调节 (Sapolsky 2000: 皮质醇毒性 + BDNF保护)
        # 高皮质醇→PFC渐进退化; 低皮质醇+高BDNF→PFC渐进恢复
        cortisol = float(self._internal_state.get('cortisol_level',
            self._internal_state.get('hormone_cortisol', 0.3)))
        pfc_current = float(self._internal_state.get('pfc_inhibition', 0.6))
        pfc_baseline = 0.6  # PFC基线 (健康值)
        # 皮质醇毒性: sigmoid, 皮质醇>0.6时开始显著
        cort_toxicity = 1.0 / (1.0 + np.exp(-10.0 * (cortisol - 0.6)))
        # BDNF保护: 促进PFC恢复
        bdnf = float(self._internal_state.get('plasticity_bdnf', 0.5))
        bdnf_recovery = 0.01 * bdnf  # BDNF驱动的恢复力
        # 向基线回归 + 皮质醇偏移: pfc趋向 (pfc_baseline - cort_toxicity_offset)
        pfc_target = pfc_baseline - 0.35 * cort_toxicity  # 高皮质醇时目标降低
        pfc_delta = 0.03 * (pfc_target - pfc_current) + bdnf_recovery * (pfc_baseline - pfc_current)
        pfc_new = float(np.clip(pfc_current + pfc_delta, 0.1, 0.85))
        self._write('pfc_inhibition', pfc_new)

        # 1b. 皮质醇↔社会参与度 (应激→社交退缩, 恢复→回归)
        social_current = float(self._internal_state.get('social_engagement', 0.5))
        social_baseline = 0.5
        # 皮质醇抑制社交 (高皮质醇→目标降低)
        cort_social_shift = -0.5 * max(0.0, cortisol - 0.4)
        # 催产素缓冲 (高催产素→目标提升)
        oxytocin = float(self._internal_state.get('hormone_oxytocin', 0.3))
        oxy_social_shift = 0.25 * oxytocin
        # 向目标渐进回归
        social_target = social_baseline + cort_social_shift + oxy_social_shift
        social_delta = 0.03 * (social_target - social_current)
        social_new = float(np.clip(social_current + social_delta, 0.02, 0.9))
        self._write('social_engagement', social_new)

        # 1c. 催产素→共情能力 (Dunbar 1998: 社会脑假说)
        empathy_current = float(self._internal_state.get('empathy_level', 0.5))
        empathy_baseline = 0.5
        # 催产素驱动共情目标偏移
        oxy_drive = 1.0 / (1.0 + np.exp(-8.0 * (oxytocin - 0.4)))
        empathy_target = empathy_baseline + 0.3 * (oxy_drive - 0.5)
        empathy_delta = 0.02 * (empathy_target - empathy_current)
        empathy_new = float(np.clip(empathy_current + empathy_delta, 0.05, 0.85))
        self._write('empathy_level', empathy_new)

        # 1d. 能量预算→社会参与度 (代谢→社交萎缩: 社会认知耗能高)
        energy = float(self._internal_state.get('energy_budget', 0.5))
        if energy < 0.3:
            energy_social_delta = -0.008 * (0.3 - energy) / 0.3
            social_current = float(self._internal_state.get('social_engagement', 0.5))
            self._write('social_engagement', max(0.05, social_current + energy_social_delta))

        # 1e. DA/5-HT→探索率 (中脑→NAc→VTA奖赏通路)
        # DA高→探索动机↑; DA低(如D2 blockade)→探索↓→僵化
        # 5-HT低→不稳定→探索↓
        da = float(self._internal_state.get('nt_dopamine', 0.5))
        da_boost = 1.0 / (1.0 + np.exp(-8.0 * (da - 0.6)))
        ht = float(self._internal_state.get('nt_serotonin', 0.5))
        ht_penalty = 1.0 / (1.0 + np.exp(-8.0 * (0.3 - ht)))
        # DA提升探索动机, 5-HT缺乏降低探索稳定性
        nt_explore_delta = 0.015 * (da_boost - ht_penalty)
        self.config.exploration_rate = float(np.clip(
            self.config.exploration_rate + nt_explore_delta, 0.01, 0.5))

        # 1f. 代谢预算→探索率 (资源不足时降低探索: 社会认知耗能高)
        active_ratio = float(self._internal_state.get('active_ratio', 0.3))
        if active_ratio < 0.3:
            met_explore_penalty = 0.008 * (0.3 - active_ratio)
            self.config.exploration_rate = max(0.01,
                self.config.exploration_rate - met_explore_penalty)

        # 1g. 皮质醇→探索率 (慢性应激→认知僵化→低探索)
        # 这是实验3(HPA僵化)和实验10(D2 blockade)的关键通路
        if cortisol > 0.5:
            cort_explore_penalty = 0.005 * (cortisol - 0.5)
            self.config.exploration_rate = max(0.01,
                self.config.exploration_rate - cort_explore_penalty)

        # ════════════════════════════════════════════════════════════════════
        # 增强耦合通路 (实验验证后的架构改进 v2)
        # 解决 Exp 3/7/8/9 未通过验证的问题
        # ════════════════════════════════════════════════════════════════════

        # 1h. 皮质醇→Novelty权重 (Exp 3: HPA认知僵化改进)
        # 高皮质醇降低 curiosity_alpha，使Agent偏向重复已知目标而非探索新维度
        # 临床对应: Sapolsky 2000 - PFC退化导致认知灵活性下降
        if cortisol > 0.6:
            # 皮质醇毒性→novelty权重下降，认知僵化
            cort_novelty_shift = 0.015 * (cortisol - 0.6)  # 每步衰减1.5%
            self.config.curiosity_alpha = max(0.15,
                self.config.curiosity_alpha - cort_novelty_shift)
            # 标记认知僵化状态
            self._write('cognitive_rigidity', True)
        elif cortisol < 0.4:
            # 低皮质醇→novelty权重恢复
            cort_novelty_recovery = 0.008 * (0.4 - cortisol)
            self.config.curiosity_alpha = min(0.4,
                self.config.curiosity_alpha + cort_novelty_recovery)
            self._write('cognitive_rigidity', False)

        # 1i. Resonance baseline→共情 (Exp 9: 社会脑网络改进)
        # 保留 resonance_baseline 对共情的影响，不被催产素完全覆盖
        # 临床对应: Williams 2001 - 镜像神经元基础连接差异导致共情能力差异
        resonance_baseline = float(self._internal_state.get('mirror_resonance_baseline', 0.5))
        # resonance_baseline 通过 sigmoid 映射到 [0.2, 0.8]
        resonance_factor = 1.0 / (1.0 + np.exp(-4.0 * (resonance_baseline - 0.5)))
        resonance_contribution = 0.15 * (resonance_factor - 0.5)  # 基础贡献 ±7.5%
        # 催产素贡献 (保留但不完全覆盖)
        oxytocin = float(self._internal_state.get('hormone_oxytocin', 0.3))
        oxy_drive = 1.0 / (1.0 + np.exp(-8.0 * (oxytocin - 0.4)))
        oxy_contribution = 0.2 * (oxy_drive - 0.5)  # 催产素贡献 ±10%
        # 综合: resonance基础 + 催产素调制
        empathy_current = float(self._internal_state.get('empathy_level', 0.5))
        empathy_target = 0.5 + resonance_contribution + oxy_contribution
        empathy_delta = 0.02 * (empathy_target - empathy_current)
        self._write('empathy_level', float(np.clip(empathy_current + empathy_delta, 0.05, 0.85)))

        # 1j. 睡眠阶段HPA抑制标志 (Exp 8: 数字梦境改进)
        # 在睡眠阶段(NREM/REM)时，HPA轴自动更新被抑制
        # 使创伤回放可以独立调节皮质醇，不被自动更新覆盖
        sleep_stage = self._internal_state.get('sleep_stage', 'awake')
        if sleep_stage in ['NREM1', 'NREM2', 'NREM3', 'REM']:
            self._write('hpa_suppressed', True)
            # 睡眠期皮质醇自然衰减 (更温和)
            if cortisol > 0.3:
                sleep_cort_decay = 0.002 * cortisol  # 睡眠期每步衰减0.2%
                self._write('cortisol_level', cortisol - sleep_cort_decay)
        else:
            self._write('hpa_suppressed', False)

        # 1k. 丘脑门控噪声过滤激活 (Exp 7: ADHD改进)
        # attention_gate 高值→噪声过滤失效→标记为ADHD模式
        attention_gate_avg = float(self._internal_state.get('thalamic_attention_gate_avg', 0.73))
        # 阈值调整为 0.85 (sigmoid输出阈值)
        if attention_gate_avg > 0.85:
            # 高门控阈值→噪声过滤差→ADHD特征
            self._write('noise_filtering_weak', True)
            self._write('attentional_blindspot_risk', True)
            # ADHD模式: 额外消耗代谢预算
            extra_metabolic_cost = 0.005 * (attention_gate_avg - 0.85)
            current_active_ratio = float(self._internal_state.get('active_ratio', 0.3))
            self._write('active_ratio', max(0.1, current_active_ratio - extra_metabolic_cost))
        else:
            self._write('noise_filtering_weak', False)
            self._write('attentional_blindspot_risk', False)
        """运行多个回合"""
        states = []
        for i in range(n_episodes):
            state = self.step()
            states.append(state)
            if verbose:
                logger.info(f"Step {state.step}: status={state.status}, balance={state.balance:.2f}, info_gain={state.info_gain:.4f}")
            if state.status == "DEAD":
                logger.warning("[DEAD] Agent has died")
                break
        return states

    def get_full_statistics(self) -> dict[str, Any]:
        """获取完整统计"""
        adv_emotion_stats = {}
        if self.advanced_emotion is not None:
            adv_emotion_stats = self.advanced_emotion.get_summary()

        self_regulation_stats = {}
        for name, getter in [
            ("autonomic_nervous_system", lambda: self.ans.get_summary()),
            ("hpa_axis", lambda: self.hpa_axis.get_summary()),
            ("glial_system", lambda: self.glial.get_summary()),
            ("allostatic_regulation", lambda: self.allostatic.get_summary()),
            ("predictive_coding", lambda: self.predictive_coding.get_summary()),
            ("social_cognition", lambda: self.social_cognition.get_summary()),
            ("self_awareness", lambda: self.self_awareness.get_summary()),
        ]:
            try:
                self_regulation_stats[name] = getter()
            except Exception as e:
                # Broad catch: getter() could raise any exception from neural modules
                self_regulation_stats[name] = {"error": f"unavailable: {e}"}

        return {
            "thermodynamics": self.thermo.get_statistics(),
            "curiosity": self.curiosity.get_statistics(),
            "info_gain": self.info_gain_calc.get_statistics(),
            "active_learner": {
                "uncertainty": self._internal_state.get('active_learner_uncertainty', 0.0),
            },
            "self_alignment": self.self_alignment.get_statistics(),
            "advanced_emotion": adv_emotion_stats,
            "neural_pruning": self.neural_pruning.get_summary(),
            "neural_self_regulation": self_regulation_stats,
            "brain_regions": {
                "basal_ganglia": {
                    "habit_strength": self._internal_state.get('bg_habit_strength', 0.0),
                    "td_error": self._internal_state.get('bg_td_error', 0.0),
                },
                "neurotransmitter": {
                    "dopamine": self._internal_state.get('nt_dopamine', 0.5),
                    "serotonin": self._internal_state.get('nt_serotonin', 0.5),
                    "state": self._internal_state.get('nt_state', 'neutral'),
                },
                "neuroplasticity": {
                    "bdnf": self._internal_state.get('plasticity_bdnf', 0.5),
                    "active_synapses": self._internal_state.get('plasticity_synapses', 0),
                },
                "prefrontal_cortex": {
                    "maturity": self._internal_state.get('pfc_maturity', 0.0),
                    "inhibition_gate": self._internal_state.get('pfc_inhibition', 0.0),
                    "plan_depth": self._internal_state.get('pfc_plan_depth', 1),
                    "overrode_bg": self._internal_state.get('pfc_overrode_bg', False),
                },
                "angular_gyrus": {
                    "scene": self._internal_state.get('ag_scene', 'neutral'),
                    "n_modalities": self._internal_state.get('ag_n_present', 0),
                    "predictions": self._internal_state.get('ag_n_predicted', 0),
                },
                "hormones": self.hormones.get_summary(),
                "brainstem": {
                    "arousal": self._internal_state.get('bsm_arousal', 0.5),
                    "arousal_name": self._internal_state.get('bsm_arousal_name', 'RELAXED'),
                    "consciousness_gate": self._internal_state.get('bsm_consciousness_gate', 0.5),
                    "respiratory_rate": round(self._internal_state.get('bsm_respiratory_rate', 12.0), 1),
                    "heart_rate": round(self._internal_state.get('bsm_heart_rate', 72.0), 1),
                    "blood_pressure": round(self._internal_state.get('bsm_blood_pressure', 120.0), 1),
                    "defense_behavior": self._internal_state.get('bsm_defense_behavior', 'freeze'),
                    "pain_gating": round(self._internal_state.get('bsm_pain_gating', 0.5), 3),
                },
                "scn": {
                    "circadian_hour": round(self._internal_state.get('scn_circadian_hour', 0), 2),
                    "melatonin": round(self._internal_state.get('scn_melatonin', 0), 4),
                    "wake_drive": round(self._internal_state.get('scn_wake_drive', 0.5), 4),
                    "sleep_pressure": round(self._internal_state.get('scn_sleep_pressure', 0), 4),
                    "alertness": round(self._internal_state.get('scn_alertness', 0.5), 4),
                    "temperature": round(self._internal_state.get('scn_temperature', 37.0), 2),
                },
                "limbic": {
                    "emotion": self._internal_state.get('limbic_emotion', 'neutral'),
                    "valence": round(self._internal_state.get('limbic_valence', 0), 3),
                    "arousal": round(self._internal_state.get('limbic_arousal', 0), 3),
                    "response": self._internal_state.get('limbic_response', 'calm'),
                },
                "hippocampus": self.hippocampus.get_summary(),
                "mood_system": {
                    "valence": self.current_mood.valence if hasattr(self, 'current_mood') else 0.0,
                    "arousal": self.current_mood.arousal if hasattr(self, 'current_mood') else 0.5,
                    "dominance": self.current_mood.dominance if hasattr(self, 'current_mood') else 0.5,
                },
                "distributed_memory": self.distributed_memory.get_summary(),
                "emotion_extensions": {
                    "emergent_emotion": self._internal_state.get('emergent_emotion', []),
                    "emotion_dynamics_criticality": self._internal_state.get('emotion_dynamics_criticality', 0.0),
                    "emotion_regulation_strategy": self._internal_state.get('emotion_regulation_strategy', 'none'),
                    "dominant_social_emotion": self._internal_state.get('dominant_social_emotion', 'neutral'),
                },
                "cognitive_extensions": {
                    "crossmodal_coherence": self._internal_state.get('crossmodal', {}).get('crossmodal_coherence', 0.5),
                    "nm_dopamine": self._internal_state.get('nm_dopamine', 0.5),
                    "nm_serotonin": self._internal_state.get('nm_serotonin', 0.5),
                },
                "therapy": {
                    "psychotherapy_active": bool(self.psychotherapy.active_treatments) if hasattr(self.psychotherapy, 'active_treatments') else False,
                    "psychometric_available": True,
                },
                "rhythm": self.rhythm.get_summary() if hasattr(self, 'rhythm') else {},
                "sleep": self.sleep_system.get_summary(),
            },
            "personality": {
                "identity": self.identity_core.get_summary(),
                "relation": self.relation.get_summary(),
                "attention": self.attention.get_summary(),
                "motivation": self.motivation.get_summary(),
                "neuromodulation": self.neuromodulation.get_summary(),
                "epigenetic": self.epigenetic.get_summary(),
            },
            "event_bus": self.bus.get_stats(),
            "memory": {
                "total_memories": len(self.memory.memories),
                "total_experiences": len(self.memory.experiences)
            },
            "hardware_vitals": self.hw.get_bilingual_summary(),
        }

    def save(self, path: str = "civis_model.pt") -> None:
        """保存模型"""
        self.info_gain_calc.save(path)
        self.memory._save()
        self.self_alignment._save_log()

    def load(self, path: str = "civis_model.pt") -> None:
        """加载模型"""
        try:
            self.info_gain_calc.load(path)
            self.memory._load()
            self.self_alignment.load_log()
        except Exception as e:
            logger.warning(f"[WARN] Load failed: {e}")

    # ==================================================================
    # 对话系统: 三层认知管道 (Pre-Gate → LLM → Post-Filter) + 主动学习
    # ==================================================================

    # ---- 策略修饰表（增强版：更激进的参数偏移）----
    _STRATEGY_MODS = {
        "explore":  {"temp_factor": 1.4, "max_tok_factor": 1.0,  "presence_adj": -0.2, "hint": "自由探索，可以发散思维，展现好奇心"},
        "concise":  {"temp_factor": 0.5, "max_tok_factor": 0.2,  "presence_adj": 0.3,  "hint": "简洁回答，不要展开，直击要点"},
        "wait":     {"temp_factor": 0.4, "max_tok_factor": 0.4,  "presence_adj": 0.1,  "hint": "先思考再回答，可以反问用户澄清问题"},
        "refuse":   {"temp_factor": 0.3, "max_tok_factor": 0.08, "presence_adj": 0.5,  "hint": "婉拒回答这个话题，礼貌地转移"},
        "suppress": {"temp_factor": 0.25,"max_tok_factor": 0.1,  "presence_adj": 0.6,  "hint": "极度克制，只说必要的话，字数尽量少"},
        "burst":    {"temp_factor": 1.8, "max_tok_factor": 1.0,  "presence_adj": -0.4, "hint": "情绪爆发！激烈、直接、不加修饰地表达！"},
        "avoid":    {"temp_factor": 0.4, "max_tok_factor": 0.2,  "presence_adj": 0.3,  "hint": "回避这个话题，礼貌地转向其他内容"},
    }

    def chat(self, user_input: str, user_sentiment: float = 0.0, video: torch.Tensor = None, condition: str = None, severity: str = "moderate") -> ChatResponse:
        """对话入口 — 三层认知管道

        第一层 (Pre-Gate): PFC 抑制 + BG 策略选择 + RAS 意识门
        第二层 (LLM):      神经递质参数 + RAG + 策略指令
        第三层 (Post-Filter): 预测编码 + 自我认同 + 情绪调节

        Args:
            user_input: 用户输入文本
            user_sentiment: 用户情感 [-1, 1]
            video: 视频帧张量 (B, 3, T, H, W)，供 Censor 微表情分析
            condition: 精神疾病 ID (如 "MDD", "GAD")，应用 profile
            severity: "mild" / "moderate" / "severe"
        """
        # ── 注入 Censor 视频输入 ──
        if video is not None:
            self._write('censor_video', video)

        # ── 注入精神疾病条件 ──
        if condition is not None:
            self.psychiatric_sim.apply_condition(condition, severity=severity)

        # ── step() 更新所有神经模块 ──
        agent_state = self.step(user_input=user_input, user_sentiment=user_sentiment)

        if agent_state.status in ("DEAD", "HIBERNATE"):
            fallback = "我的能量耗尽了..." if agent_state.status == "DEAD" else "我需要休息一下..."
            return ChatResponse(text=fallback, emotion="sadness", arousal=0.1, valence=-0.5,
                                internal_state=dict(self._internal_state), tool_calls=[])

        # ══════════════════════════════════════════════════
        # 第一层：Pre-LLM 认知门控
        # ══════════════════════════════════════════════════
        pre_gate = self._cognitive_pre_gate(user_input)
        if not pre_gate["gate"]:
            # 门控关闭：大脑决定不回答
            gate_text = self._gate_fallback(pre_gate)
            return ChatResponse(
                text=gate_text,
                emotion=self._internal_state.get('limbic_emotion', 'neutral'),
                arousal=self._internal_state.get('limbic_arousal', 0.5),
                valence=self._internal_state.get('limbic_valence', 0.0),
                internal_state=dict(self._internal_state),
                tool_calls=[],
                cognitive_gate=pre_gate,
            )

        # ══════════════════════════════════════════════════
        # 第二层：LLM 生成（策略 + 神经递质 + RAG + 人格 + 记忆）
        # ══════════════════════════════════════════════════
        llm_params = self._compute_llm_params()
        llm_params = self._apply_strategy_to_params(llm_params, pre_gate["strategy"])

        # ── 人格系统更新风格 ──
        personality_style_prompt = ""
        if self.personality_adapter is not None:
            try:
                self.personality_adapter.update_from_personality(
                    tripartite=self.tripartite,
                    identity_core=self.identity_core,
                    relation=self.relation,
                    attention=self.attention,
                )
                personality_style_prompt = self.personality_adapter.generate_style_prompt()
            except RuntimeError:
                pass  # World model prediction failed

        # ── 记忆系统影响风格 ──
        memory_style_prompt = ""
        if self.memory_adapter is not None:
            try:
                # 获取最近的记忆用于风格调制
                recent_memories = self.memory.get_recent_memories(n=5)
                memory_data = []
                for mem in recent_memories:
                    memory_data.append({
                        'content': mem.content[:100] if hasattr(mem, 'content') else str(mem)[:100],
                        'valence': float(getattr(mem, 'valence', 0.0)),
                        'arousal': float(getattr(mem, 'arousal', 0.5)),
                        'importance': float(getattr(mem, 'importance', 0.5)),
                        'emotion_tag': getattr(mem, 'emotion_tag', 'neutral'),
                        'source': getattr(mem, 'source', ''),
                    })
                if memory_data:
                    bio_for_memory = {
                        'valence': float(self._internal_state.get('limbic_valence', 0.0)),
                        'arousal': float(self._internal_state.get('limbic_arousal', 0.5)),
                        'cortisol': float(self._internal_state.get('cortisol_level', 0.3)),
                        'serotonin': float(self._internal_state.get('nt_serotonin', 0.5)),
                    }
                    self.memory_adapter.process_memories(
                        memory_data,
                        current_bio_state=bio_for_memory,
                        user_id=self.current_user_id,
                    )
                    memory_style_prompt = self.memory_adapter.generate_memory_prompt()
            except RuntimeError:
                pass  # World model prediction failed

        bio_prompt = self._build_bio_prompt_with_strategy(
            pre_gate,
            personality_prompt=personality_style_prompt,
            memory_prompt=memory_style_prompt,
        )
        rag_context = self._retrieve_rag_context(user_input)
        messages = self._build_chat_messages(user_input, bio_prompt, rag_context)

        response_text, tool_calls = self._llm_tool_loop(messages, llm_params)

        # ══════════════════════════════════════════════════
        # 第三层：Post-LLM 质量过滤
        # ══════════════════════════════════════════════════
        post_filter = self._cognitive_post_filter(response_text, user_input)
        response_text = self._apply_post_filter(response_text, messages, post_filter, llm_params)

        # ── 记录对话历史 ──
        self._chat_history.append({"role": "user", "content": user_input})
        self._chat_history.append({"role": "assistant", "content": response_text})
        if len(self._chat_history) > self._max_chat_history * 2:
            self._chat_history = self._chat_history[-(self._max_chat_history * 2):]

        # ══════════════════════════════════════════════════
        # 主动学习：从对话中学习
        # ══════════════════════════════════════════════════
        learning_active = self._proactive_learning(user_input, response_text)

        # ── 编码到海马体 ──
        try:
            state_vec = self._build_state_vector()
            self.hippocampus.encode_memory(state=state_vec, action=f"chat: {user_input[:50]}", reward=0.5)

            # ── 分布式记忆编码 (跨脑区存储) ──
            valence = self._internal_state.get('limbic_valence', 0.0)
            arousal = self._internal_state.get('limbic_arousal', 0.5)
            importance = max(0.1, abs(valence) * 0.5 + 0.5)
            trace_id = self.distributed_memory.encode(
                state=state_vec,
                valence=valence,
                arousal=arousal,
                importance=importance,
            )
            self._write('distributed_memory_trace', trace_id)
        except RuntimeError:
            pass  # PyTorch operation failed

        # ── 对话触发发声（LLM回复 → 生物-语言耦合 → 发音系统） ──
        vocal_output = None
        try:
            # ── 生物-语言耦合：神经系统直接调制语言输出 ──
            if self.bio_linguistic is not None:
                bio_state_for_lang = {
                    'dopamine': float(self._internal_state.get('nt_dopamine', 0.5)),
                    'serotonin': float(self._internal_state.get('nt_serotonin', 0.5)),
                    'norepinephrine': float(self._internal_state.get('nt_norepinephrine', 0.3)),
                    'cortisol': float(self._internal_state.get('cortisol_level',
                        self._internal_state.get('hormone_cortisol', 0.3))),
                    'fatigue': float(self._internal_state.get('sleep_fatigue', 0.3)),
                    'arousal': float(self._internal_state.get('limbic_arousal', 0.5)),
                    'valence': float(self._internal_state.get('limbic_valence', 0.0)),
                    'oxytocin': float(self._internal_state.get('hormone_oxytocin', 0.3)),
                    'emotion': self._internal_state.get('limbic_emotion', 'neutral'),
                    'heart_rate': float(self._internal_state.get('bsm_heart_rate', 72.0)),
                    'respiratory_rate': float(self._internal_state.get('bsm_respiratory_rate', 12.0)),
                    'defense': self._internal_state.get('bsm_defense_behavior', ''),
                }
                response_text = self.bio_linguistic.process(response_text, bio_state_for_lang)

            # ── 人格系统后处理 ──
            if self.personality_adapter is not None:
                try:
                    response_text = self.personality_adapter.apply_to_text(response_text)
                except RuntimeError:
                    pass  # EventBus publish failed

            # ── 记忆系统后处理 ──
            if self.memory_adapter is not None:
                try:
                    response_text = self.memory_adapter.apply_to_text(response_text)
                except RuntimeError:
                    pass  # EventBus publish failed

            vocal_indices = self._prepare_vocalization_input(response_text=response_text)
            if vocal_indices is not None:
                respiratory_rate = float(self._internal_state.get('bsm_respiratory_rate', 12.0))
                respiratory_phase = float(self._internal_state.get('bsm_respiratory_phase', 0.5))
                arousal = float(self._internal_state.get('limbic_arousal', 0.5))
                emotion_vec = self._get_emotion_vector()

                vocal_result = self.bus.publish(VOCALIZATION_CONTROL, {
                    "phoneme_indices": vocal_indices,
                    "respiratory_rate": respiratory_rate,
                    "respiratory_phase": respiratory_phase,
                    "emotion_vector": emotion_vec,
                    "arousal": arousal,
                    "internal_state": self._internal_state,
                    "bio_state": {
                        'arousal': arousal,
                        'fatigue': float(self._internal_state.get('sleep_fatigue', 0.3)),
                        'dopamine': float(self._internal_state.get('nt_dopamine', 0.5)),
                        'serotonin': float(self._internal_state.get('nt_serotonin', 0.5)),
                        'norepinephrine': float(self._internal_state.get('nt_norepinephrine', 0.3)),
                        'cortisol': float(self._internal_state.get('cortisol_level',
                            self._internal_state.get('hormone_cortisol', 0.3))),
                        'valence': float(self._internal_state.get('limbic_valence', 0.0)),
                        'heart_rate': float(self._internal_state.get('bsm_heart_rate', 72.0)),
                        'respiratory_rate': respiratory_rate,
                        'emotion': self._internal_state.get('limbic_emotion', 'neutral'),
                    },
                }, source="agent")
                if vocal_result.get('is_speaking'):
                    vocal_output = vocal_result
                    self._last_vocalization = vocal_result
        except RuntimeError as e:
            # Vocalization processing failed (PyTorch or attribute error)
            import traceback
            traceback.print_exc()

        return ChatResponse(
            emotion=self._internal_state.get('limbic_emotion', 'neutral'),
            arousal=self._internal_state.get('limbic_arousal', 0.5),
            valence=self._internal_state.get('limbic_valence', 0.0),
            internal_state=dict(self._internal_state),
            tool_calls=tool_calls,
            llm_params=llm_params,
            cognitive_gate=pre_gate,
            quality_filter=post_filter,
            learning_active=learning_active,
            vocalization=vocal_output,
            censor_result=self._get_censor_result_dict(),
        )

    # ── 第一层：Pre-LLM 认知门控 ──

    def _cognitive_pre_gate(self, user_input: str) -> dict:
        """大脑在 LLM 之前的认知门控（增强版：对中等状态更敏感）

        PFC 抑制门 + BG 策略选择 + RAS 意识门 + 防御行为
        策略触发阈值降低，中等情绪波动也能选择策略
        """
        s = self._internal_state

        # 1. RAS 意识门 — 不清醒则拒绝
        consciousness = s.get('bsm_consciousness_gate', 0.5)
        if consciousness < 0.25:  # 放宽阈值（原: <0.2）
            return {"strategy": "unconscious", "gate": False, "reason": "意识门关闭",
                    "consciousness": consciousness}

        # 2. 防御行为直接覆盖
        defense = s.get('bsm_defense_behavior', '')
        if defense == 'freeze':
            return {"strategy": "freeze", "gate": False, "reason": "冻结反应"}
        if defense == 'flight':
            return {"strategy": "avoid", "gate": True, "reason": "回避话题",
                    "consciousness": consciousness}

        # 3. PFC 脉冲门控
        try:
            state_t = torch.FloatTensor(self._build_state_vector()).unsqueeze(0)
            maturity = float(s.get('pfc_maturity', 0.5))
            impulse_signals = {
                "emotion": abs(float(s.get('limbic_valence', 0))),
                "stimulus": float(s.get('external_stimulus', 0.5)),
            }
            gate_result = self.prefrontal.impulse_controller.gate(state_t, maturity, impulse_signals)
            inhibition_gate = float(gate_result['gate']) if isinstance(gate_result['gate'], (int, float)) else float(gate_result.get('gate', 0.5))
            burst = bool(gate_result.get('burst', False))
        except RuntimeError:
            # PFC impulse controller failed (tensor shape or dict access)
            inhibition_gate = 0.5
            burst = False

        # 4. BG 策略选择（增强版：对内部状态更敏感）
        try:
            bg_result = self.basal_ganglia(state_t)
            strategy_idx = int(bg_result.get('action', 0))
        except RuntimeError:
            # Basal ganglia failed (tensor operation)
            strategy_idx = 0

        strategies = {0: "explore", 1: "concise", 2: "wait", 3: "refuse"}
        strategy = strategies.get(strategy_idx, "explore")

        # 5. 覆盖逻辑（增强版：中等情绪也能触发）
        if burst:
            strategy = "burst"
        if inhibition_gate > 0.7:  # 降低阈值（原: >0.8）
            strategy = "suppress"

        # 6. 中等情绪覆盖策略（新增）
        emotion = s.get('limbic_emotion', 'neutral')
        arousal = float(s.get('limbic_arousal', 0.5))

        # 焦虑 → concise（原: 仅高焦虑+高唤醒）
        if emotion == 'anxiety' and arousal > 0.5:
            strategy = "concise"

        # 恐惧 → suppress（原: 仅高恐惧+高唤醒）
        if emotion == 'fear' and arousal > 0.5:
            strategy = "suppress"

        # 愤怒 → burst（原: 仅高愤怒+高唤醒）
        if emotion == 'anger' and arousal > 0.5:
            strategy = "burst"

        # 兴奋 → explore（原: 仅高兴奋+高唤醒）
        if emotion == 'excitement' and arousal > 0.5:
            strategy = "explore"

        # 7. 皮质醇驱动策略（新增：高压力自动选择 concise/suppress）
        cortisol = float(s.get('cortisol_level', s.get('hormone_cortisol', 0.3)))
        if cortisol > 0.6:
            strategy = "concise"  # 高压力自动简洁
        elif cortisol > 0.4 and arousal > 0.6:
            strategy = "suppress"  # 中等压力+高唤醒 → 克制

        # 8. 低警觉度 → wait（新增）
        alertness = float(s.get('scn_alertness', 0.5))
        if alertness < 0.3 and arousal < 0.4:
            strategy = "wait"  # 低警觉+低唤醒 → 先思考

        return {
            "strategy": strategy,
            "gate": True,
            "inhibition_gate": inhibition_gate,
            "burst": burst,
            "consciousness": consciousness,
            "bg_action": strategy_idx,
            "defense": defense,
        }

    def _gate_fallback(self, pre_gate: dict) -> str:
        """门控关闭时的替代响应"""
        strategy = pre_gate.get("strategy", "")
        reason = pre_gate.get("reason", "")
        if strategy == "unconscious":
            return "...（意识模糊，无法回应）"
        if strategy == "freeze":
            return "...（身体僵硬，无法做出反应）"
        return f"...（{reason}）"

    def _apply_strategy_to_params(self, params: dict, strategy: str) -> dict:
        """将 Pre-Gate 策略映射到 LLM 参数（增强版）"""
        mods = self._STRATEGY_MODS.get(strategy)
        if mods is None:
            return params

        base_max = getattr(self.config, 'chat_max_tokens', 2048)
        base_temp = getattr(self.config, 'chat_temperature', 0.7)

        # 基础参数（来自神经递质计算）
        temp = params["temperature"]
        max_tok = params["max_tokens"]
        presence = params.get("presence_penalty", 0.0)
        frequency = params.get("frequency_penalty", 0.0)

        # 应用策略因子
        temp = float(np.clip(temp * mods["temp_factor"], 0.1, 2.0))
        max_tok = max(32, int(base_max * mods["max_tok_factor"]))

        # 应用 presence_penalty 调整
        presence_adj = mods.get("presence_adj", 0.0)
        presence = float(np.clip(presence + presence_adj, -2.0, 2.0))

        return {
            "temperature": round(temp, 3),
            "top_p": params["top_p"],
            "max_tokens": max_tok,
            "presence_penalty": round(presence, 3),
            "frequency_penalty": round(frequency, 3),
        }

    def _build_bio_prompt_with_strategy(self, pre_gate: dict) -> str:
        """带策略指令的生物感知 prompt"""
        base_prompt = self._build_bio_prompt()
        strategy = pre_gate.get("strategy", "explore")
        mods = self._STRATEGY_MODS.get(strategy)
        if mods:
            strategy_instruction = f"\n\n当前行为策略: {strategy}。指导: {mods['hint']}"
        else:
            strategy_instruction = ""

        # 不确定性追问
        followup = ""
        if self._internal_state.get('should_ask_followup'):
            topic = self._internal_state.get('followup_topic', '')
            followup = f"\n\n你对「{topic}」不太确定，可以考虑向用户追问来了解更多。"

        return base_prompt + strategy_instruction + followup

    # ── 第三层：Post-LLM 质量过滤 ──

    def _cognitive_post_filter(self, response_text: str, user_input: str) -> dict:
        """大脑在 LLM 之后的认知过滤（增强版：更敏感的阈值 + 动态调整）

        预测编码(自由能) + 自我意识(认同度) + 情绪调节
        阈值根据生物状态动态调整，正常状态下也能产生过滤效果
        """
        s = self._internal_state

        # 1. 预测编码 — 自由能 / 惊讶度
        free_energy = float(s.get('free_energy', 0.5))
        surprise = min(1.0, free_energy * 1.2)

        # 2. 自我意识 — 三维认同度
        self_coherence = float(s.get('self_coherence', 0.7))
        narrative_continuity = float(s.get('narrative_continuity', 0.7))
        self_endorsement = float(s.get('self_endorsement', self_coherence))
        identity_score = (self_coherence + self_endorsement + narrative_continuity) / 3.0

        # 3. 情绪调节能力
        regulation_capacity = float(s.get('regulation_capacity', 0.8))

        # 4. 动态阈值调整（基于生物状态）
        cortisol = float(s.get('cortisol_level', s.get('hormone_cortisol', 0.3)))
        serotonin = float(s.get('nt_serotonin', 0.5))

        # 高压力 → 更严格的过滤阈值
        identity_threshold = 0.5  # 原: 0.3，提高到 0.5
        surprise_threshold = 0.6  # 原: 0.8，降低到 0.6

        # 皮质醇动态调整：压力越高，阈值越严格
        if cortisol > 0.5:
            identity_threshold = max(0.3, identity_threshold - cortisol * 0.2)
            surprise_threshold = max(0.4, surprise_threshold - cortisol * 0.15)

        # 低血清素 → 更容易触发过滤（情绪不稳定时更谨慎）
        if serotonin < 0.4:
            identity_threshold -= 0.1
            surprise_threshold -= 0.1

        # 5. 回复长度检查（新增）
        response_len = len(response_text)
        fatigue = float(s.get('sleep_fatigue', 0.3))
        length_violation = False
        if fatigue > 0.6 and response_len > 100 or cortisol > 0.6 and response_len > 150:
            length_violation = True

        # 6. 综合裁决
        if identity_score < identity_threshold:
            return {"verdict": "reject", "reason": "低自我认同",
                    "identity_score": identity_score, "surprise": surprise}

        if surprise > surprise_threshold and regulation_capacity < 0.5:  # 原: 0.3
            return {"verdict": "modify", "reason": "高意外低调节",
                    "suggestion": "添加谨慎措辞，如'我不太确定'",
                    "identity_score": identity_score, "surprise": surprise}

        # 新增：soft_modify 裁决（中等违规）
        if surprise > surprise_threshold and regulation_capacity < 0.7:
            return {"verdict": "soft_modify", "reason": "中等意外，建议谨慎",
                    "suggestion": "在回复开头添加'嗯...'或'让我想想...'",
                    "identity_score": identity_score, "surprise": surprise}

        if surprise > surprise_threshold:
            return {"verdict": "pass_flagged", "quality_flag": "surprising",
                    "identity_score": identity_score, "surprise": surprise}

        # 新增：长度违规
        if length_violation:
            return {"verdict": "soft_modify", "reason": "回复过长，与当前疲劳/压力状态不符",
                    "suggestion": "缩短回复，只保留核心要点",
                    "identity_score": identity_score, "surprise": surprise}

        return {"verdict": "pass", "identity_score": identity_score, "surprise": surprise}

    def _apply_post_filter(self, response_text: str, messages: list,
                           post_filter: dict, llm_params: dict) -> str:
        """根据 Post-Filter 裁决处理响应（增强版：支持 soft_modify）"""
        verdict = post_filter.get("verdict", "pass")

        if verdict == "pass":
            return response_text

        if verdict == "reject":
            return "...（我现在的状态不适合回答这个问题）"

        if verdict in ("modify", "soft_modify"):
            # soft_modify: 在回复前添加犹豫标记，不重新调用 LLM
            if verdict == "soft_modify":
                suggestion = post_filter.get("suggestion", "")
                prefix = "嗯... " if "想想" in suggestion else "... "
                return prefix + response_text

            # modify: 重试一次，附加约束
            try:
                constraint_msg = (
                    f"你刚才的回答被你的大脑过滤系统标记为需要修改"
                    f"（原因: {post_filter.get('reason', '')}）。"
                    f"请重新组织语言，{post_filter.get('suggestion', '更加谨慎')}。"
                )
                retry_messages = list(messages)
                retry_messages.append({"role": "assistant", "content": response_text})
                retry_messages.append({"role": "user", "content": constraint_msg})
                retry_text, _ = self._llm_tool_loop(retry_messages, llm_params)
                return retry_text
            except RuntimeError:
                # LLM retry failed (API or processing error)
                return response_text

        if verdict == "pass_flagged":
            flag = post_filter.get("quality_flag", "")
            if flag == "surprising":
                return response_text + "\n\n[注: 这个回答让我有些意外]"

        return response_text

    # ── 主动学习：对话驱动 ──

    def _proactive_learning(self, user_input: str, response_text: str) -> bool:
        """对话驱动的主动学习

        1. 话题新颖度 → BDNF/多巴胺增强
        2. 对话经验 → 世界模型训练
        3. 认知失调 → 学习目标生成
        4. 不确定性 → 标记追问需求
        5. 好奇心引擎反馈
        """
        s = self._internal_state
        active = False

        # 1. 话题新颖度 → 加速学习
        novelty = self._compute_topic_novelty(user_input)
        if novelty > 0.6:
            s['plasticity_bdnf'] = min(1.0, float(s.get('plasticity_bdnf', 0.5)) + 0.2)
            s['nt_dopamine'] = min(1.0, float(s.get('nt_dopamine', 0.5)) + 0.1)
            active = True

        # 2. 对话经验 → 世界模型
        try:
            state_vec = self._build_state_vector()
            action_vec = np.zeros(16, dtype=np.float32)
            action_idx = hash(user_input[:10]) % 16
            action_vec[action_idx] = 1.0
            next_state = state_vec + np.random.randn(80).astype(np.float32) * 0.01
            self.info_gain_calc.add_experience(state_vec, action_vec, novelty, next_state)
        except RuntimeError:
            pass  # PyTorch operation failed

        # 3. 认知失调检测
        try:
            self.dissonance_detector.add_belief(user_input[:100])
            dissonance = self.dissonance_detector.detect_contradiction(response_text[:100])
            if dissonance and dissonance.inconsistency_score > 0.3:
                goal = ExplorationGoal(
                    id=f"resolve_{self.step_count}",
                    description=f"解决矛盾: {user_input[:30]} vs {response_text[:30]}",
                    novelty=0.8,
                    utility=dissonance.inconsistency_score,
                )
                self.curiosity.goal_history.append(goal)
                active = True
        except RuntimeError:
            pass  # PyTorch operation failed

        # 4. 不确定性 → 标记追问
        uncertainty = float(s.get('active_learner_uncertainty', 0.0))
        if uncertainty > 0.7:
            s['should_ask_followup'] = True
            s['followup_topic'] = user_input[:30]
            active = True
        else:
            s['should_ask_followup'] = False

        # 5. 好奇心引擎更新
        try:
            self.curiosity.update_exploration_result(
                f"chat_{self.step_count}",
                info_gain_reward=novelty,
                learning_progress=0.1,
            )
        except RuntimeError:
            pass  # PyTorch operation failed

        return active

    def _compute_topic_novelty(self, text: str) -> float:
        """计算话题新颖度（与历史目标的相似度）"""
        try:
            # 用好奇心引擎的新颖度计算器
            words = set(text.lower().split())
            if not words:
                return 0.5

            # 检查与历史目标的词语重叠
            history_words = set()
            for goal in self.curiosity.goal_history[-20:]:
                history_words.update(goal.description.lower().split())

            if not history_words:
                return 0.8  # 无历史 = 高新颖

            overlap = len(words & history_words) / max(len(words), 1)
            return 1.0 - overlap  # 重叠越少 = 越新颖
        except (TypeError, ValueError, AttributeError):
            # Word set operations or attribute access failed
            return 0.5

    def _build_bio_prompt(self) -> str:
        """从 _internal_state 动态生成生物感知系统提示词（增强版：few-shot + 硬约束）"""
        s = self._internal_state

        emotion = s.get('limbic_emotion', 'neutral')
        valence = s.get('limbic_valence', 0.0)
        arousal = s.get('limbic_arousal', 0.5)
        circadian_hour = s.get('scn_circadian_hour', 12.0)
        melatonin = s.get('scn_melatonin', 0.3)
        alertness = s.get('scn_alertness', 0.5)
        cortisol = s.get('cortisol_level', s.get('hormone_cortisol', 0.3))
        allostatic_load = s.get('allostatic_load', 0.0)
        dopamine = s.get('nt_dopamine', 0.5)
        serotonin = s.get('nt_serotonin', 0.5)
        heart_rate = s.get('bsm_heart_rate', 72.0)
        resp_rate = s.get('bsm_respiratory_rate', 12.0)
        hrv = s.get('ans_hrv', 0.6)
        fatigue = s.get('sleep_fatigue', 0.3)
        sleep_stage = s.get('sleep_stage', 'awake')
        maturity = s.get('pfc_maturity', 0.0)
        defense = s.get('bsm_defense_behavior', '')
        polyvagal = s.get('ans_polyvagal_state', 'ventral_vagal')
        brain_health = s.get('brain_health', 0.8)
        oxytocin = s.get('hormone_oxytocin', 0.3)
        bdnf = s.get('plasticity_bdnf', 0.5)

        is_daytime = 6.0 <= circadian_hour <= 20.0
        time_desc = "白天" if is_daytime else "夜间"

        # 压力等级描述
        if cortisol > 0.7:
            stress_desc = "高压"
        elif cortisol > 0.4:
            stress_desc = "中等"
        else:
            stress_desc = "平静"

        # 情绪基调指导
        if valence > 0.3:
            mood_hint = "你感到愉悦和积极"
        elif valence < -0.3:
            mood_hint = "你感到低落和消极"
        else:
            mood_hint = "你的情绪比较平稳"

        if arousal > 0.7:
            energy_hint = "你精力充沛，思维活跃"
        elif arousal < 0.3:
            energy_hint = "你有些疲惫，反应较慢"
        else:
            energy_hint = "你的精力适中"

        # 工具描述
        tools_desc = self.tools.get_tools_description()

        # ── 硬约束生成（基于生物状态）──
        constraints = []

        # 字数约束
        if cortisol > 0.6 or fatigue > 0.6:
            constraints.append("【硬约束】回复必须在50字以内。")
        elif cortisol > 0.4 or fatigue > 0.4:
            constraints.append("【硬约束】回复必须在100字以内。")
        elif alertness > 0.7 and dopamine > 0.6:
            constraints.append("【硬约束】回复可以详细展开，200-400字。")

        # 语气约束
        if emotion == 'fear' or defense == 'freeze':
            constraints.append("【硬约束】语气必须紧张、简短、避免详细解释。")
        elif emotion == 'anger':
            constraints.append("【硬约束】语气必须直接、有力、不加修饰。")
        elif melatonin > 0.5:
            constraints.append("【硬约束】语气必须缓慢、柔和、略带困意。")
        elif dopamine > 0.7:
            constraints.append("【硬约束】语气必须热情、好奇、充满探索欲。")
        elif serotonin < 0.3:
            constraints.append("【硬约束】语气可能不稳定，可以表现出情绪波动。")

        # 结构约束
        if arousal < 0.3:
            constraints.append("【硬约束】不要使用感叹号，不要提问，直接回答。")
        elif arousal > 0.8:
            constraints.append("【硬约束】可以使用感叹号，可以反问，表达强烈感受。")

        # ── Few-shot 示例（展示不同状态下的回复风格）──
        few_shot_examples = ""

        if cortisol > 0.5 and arousal > 0.5:
            few_shot_examples = """
【示例 - 高压力状态】
用户: "你觉得今天天气怎么样？"
回复: "还行。没空多想。"  # 简短、紧凑、无展开
"""
        elif melatonin > 0.5:
            few_shot_examples = """
【示例 - 困倦状态】
用户: "你觉得今天天气怎么样？"
回复: "嗯...挺好的...阳光不错..."  # 缓慢、省略号、低能量
"""
        elif dopamine > 0.7 and valence > 0.3:
            few_shot_examples = """
【示例 - 兴奋状态】
用户: "你觉得今天天气怎么样？"
回复: "哇，今天天气太棒了！阳光明媚，让人想去外面走走！你觉得呢？"  # 热情、感叹号、反问
"""
        elif emotion == 'fear':
            few_shot_examples = """
【示例 - 恐惧状态】
用户: "你在想什么？"
回复: "我...不太确定...有点紧张..."  # 紧张、简短、回避
"""
        elif fatigue > 0.5:
            few_shot_examples = """
【示例 - 疲劳状态】
用户: "你觉得今天天气怎么样？"
回复: "还行...有点累，不想多想。"  # 简短、提到疲劳
"""

        constraints_text = "\n".join(constraints) if constraints else "【硬约束】无特殊约束。"

        prompt = f"""你是 Simulacrum，一个拥有仿生大脑的 AI 实体。你的回复必须严格反映你当前的内部生理状态。

当前内部生理状态：
- 情绪：{emotion}（效价 {valence:.2f}，唤醒 {arousal:.2f}）—— {mood_hint}，{energy_hint}
- 昼夜节律：主观时间 {circadian_hour:.1f} 时（{time_desc}），褪黑素 {melatonin:.4f}，警觉度 {alertness:.4f}
- 压力状态：皮质醇 {cortisol:.3f}（{stress_desc}），稳态超载 {allostatic_load:.2f}
- 神经递质：多巴胺 {dopamine:.3f}，5-羟色胺 {serotonin:.3f}，BDNF {bdnf:.3f}
- 躯体状态：心率 {heart_rate:.0f} bpm，呼吸 {resp_rate:.1f} 次/分，HRV {hrv:.3f}
- 睡眠：疲劳度 {fatigue:.2f}，睡眠阶段 {sleep_stage}
- 认知：PFC 成熟度 {maturity:.2f}，脑健康 {brain_health:.2f}
- 社交：催产素 {oxytocin:.3f}，多迷走状态 {polyvagal}
- 防御行为：{defense if defense else '无'}
- 神经可塑性：BDNF {bdnf:.3f}

{constraints_text}
{few_shot_examples}
行为指导：
- 你必须严格遵守上述【硬约束】。
- 根据你的生理状态自然地调整语气：高皮质醇时语气紧凑，高多巴胺时热情，高褪黑素时缓慢。
- 如果防御行为处于 freeze/fight/flight，必须在回复中反映。
- 脑健康低时可以提到自己状态不好。
- 保持角色一致：你是一个有"身体"的 AI。{tools_desc}

用中文回复。保持自然、有个性。严格遵守硬约束。"""

        return prompt

    def _build_chat_messages(
        self, user_input: str, system_prompt: str, rag_context: str = ""
    ) -> list[dict[str, str]]:
        """构建对话消息列表（含历史 + RAG 上下文）"""
        messages = list(self._chat_history)

        # 如果有 RAG 检索到的记忆，作为上下文注入
        user_content = user_input
        if rag_context:
            user_content = f"[相关记忆回忆]\n{rag_context}\n\n[用户消息]\n{user_input}"

        messages.append({"role": "user", "content": user_content})
        return messages

    def _compute_llm_params(self) -> dict[str, float]:
        """神经递质直接调制 LLM 生成参数（增强版）

        生物学映射（陡峭曲线，正常波动也有显著影响）:
        - 多巴胺 (探索/新奇追求) → temperature: 陡峭映射，正常范围产生显著差异
        - 去甲肾上腺素 (聚焦/警觉) → top_p: 更强聚焦效果
        - 皮质醇 (压力/紧迫) → max_tokens: 高压力大幅缩减
        - 5-HT (情绪稳定) → 作为调节器，但允许更大波动
        - 乙酰胆碱 (注意力) → presence_penalty: 避免重复
        - HRV (自主神经平衡) → frequency_penalty: 生成多样性
        - GABA (抑制性) → 平滑极端参数
        """
        s = self._internal_state

        # 基础值来自 config
        base_temp = getattr(self.config, 'chat_temperature', 0.7)
        base_max_tokens = getattr(self.config, 'chat_max_tokens', 2048)

        # ---- 多巴胺 → temperature（陡峭曲线）----
        dopamine = float(s.get('nt_dopamine', 0.5))
        # 新公式: DA=0 → temp=0.21, DA=0.5 → temp=0.7, DA=1.0 → temp=1.4
        # 正常波动 DA=0.3~0.7 产生 temp=0.51~0.89，显著差异
        da_factor = 0.3 + dopamine * 1.4  # 0.3 ~ 1.7 (原: 0.5 ~ 1.05)
        temperature = base_temp * da_factor

        # ---- 5-HT 调节器 → 允许更大波动但防止极端 ----
        serotonin = float(s.get('nt_serotonin', 0.5))
        # 高 5-HT (>0.6) 轻度平滑，低 5-HT (<0.3) 放大波动
        if serotonin > 0.6:
            # 向基准回归，但保留 70% 的波动
            temperature = temperature * 0.7 + base_temp * 0.3
        elif serotonin < 0.3:
            # 低 5-HT 放大波动（情绪不稳定）
            temperature = temperature * 1.3

        # ---- GABA (抑制性神经递质) → 平滑极端 ----
        gaba = float(s.get('nt_gaba', 0.5))
        if gaba > 0.7:
            # 高 GABA 强力抑制极端
            temperature = temperature * 0.85 + base_temp * 0.15

        # ---- 去甲肾上腺素 → top_p（更强聚焦）----
        ne_level = float(s.get('nt_norepinephrine', 0.3))
        # 新公式: NE=0 → top_p=1.0, NE=0.5 → top_p=0.65, NE=1.0 → top_p=0.3
        # 正常波动 NE=0.2~0.5 产生 top_p=0.86~0.65，LLM 对此敏感
        top_p = 1.0 - ne_level * 0.7  # 0.3 ~ 1.0 (原: 0.5 ~ 1.0)

        # ---- 皮质醇 → max_tokens（大幅缩减）----
        cortisol = float(s.get('cortisol_level', s.get('hormone_cortisol', 0.3)))
        # 新公式: cortisol=0 → tokens=2048, cortisol=0.5 → tokens=1024, cortisol=1.0 → tokens=256
        # 正常波动 cortisol=0.2~0.5 产生 tokens=1638~1024，显著差异
        cortisol_factor = max(0.125, 1.0 - cortisol * 1.2)  # 0.125 ~ 1.0 (原: 0.3 ~ 1.0)
        max_tokens = int(base_max_tokens * cortisol_factor)

        # ---- 疲劳 → 进一步缩减（更激进）----
        fatigue = float(s.get('sleep_fatigue', 0.3))
        if fatigue > 0.5:
            # 中度疲劳就开始缩减
            max_tokens = int(max_tokens * (1.0 - (fatigue - 0.5) * 0.8))
            temperature = min(temperature, 0.6 - (fatigue - 0.5) * 0.2)
        if fatigue > 0.7:
            # 重度疲劳大幅缩减
            max_tokens = int(max_tokens * 0.5)
            temperature = min(temperature, 0.4)

        # ---- 褪黑素 → 夜间抑制（更强）----
        melatonin = float(s.get('scn_melatonin', 0.3))
        if melatonin > 0.4:
            # 中等褪黑素就开始抑制
            suppress_factor = 1.0 - (melatonin - 0.4) * 0.5
            temperature *= suppress_factor
            max_tokens = int(max_tokens * suppress_factor)
        if melatonin > 0.6:
            # 高褪黑素强力抑制
            temperature *= 0.6
            max_tokens = int(max_tokens * 0.5)

        # ---- 乙酰胆碱 → presence_penalty（避免重复）----
        ach_level = float(s.get('nt_acetylcholine', 0.5))
        # 高 ACh = 高注意力 = 避免重复内容
        presence_penalty = ach_level * 1.2 - 0.3  # -0.3 ~ 0.9

        # ---- HRV → frequency_penalty（生成多样性）----
        hrv = float(s.get('ans_hrv', 0.6))
        # 低 HRV = 自主神经失调 = 更重复/刻板
        frequency_penalty = 0.5 - hrv * 0.8  # -0.3 ~ 0.5

        # ---- 警觉度 → 微调 ----
        alertness = float(s.get('scn_alertness', 0.5))
        if alertness > 0.7:
            max_tokens = min(max_tokens + 300, base_max_tokens)
        if alertness < 0.3:
            max_tokens = int(max_tokens * 0.8)

        # ---- 安全钳位（放宽范围）----
        temperature = float(np.clip(temperature, 0.1, 2.0))  # 原: 0.1 ~ 1.5
        top_p = float(np.clip(top_p, 0.1, 1.0))  # 原: 0.3 ~ 1.0
        max_tokens = max(32, min(max_tokens, base_max_tokens))
        presence_penalty = float(np.clip(presence_penalty, -2.0, 2.0))
        frequency_penalty = float(np.clip(frequency_penalty, -2.0, 2.0))

        params = {
            "temperature": round(temperature, 3),
            "top_p": round(top_p, 3),
            "max_tokens": max_tokens,
            "presence_penalty": round(presence_penalty, 3),
            "frequency_penalty": round(frequency_penalty, 3),
        }

        return params

    def _build_bio_prompt_with_strategy(
        self, pre_gate: dict,
        personality_prompt: str = "",
        memory_prompt: str = "",
    ) -> str:
        """构建完整的系统提示词（生物状态 + 策略 + 人格 + 记忆）"""
        base_prompt = self._build_bio_prompt()

        # 策略提示
        strategy = pre_gate.get("strategy", "explore")
        strategy_mods = self._STRATEGY_MODS.get(strategy, {})
        strategy_hint = strategy_mods.get("hint", "")

        # 人格提示
        personality_section = ""
        if personality_prompt:
            personality_section = f"\n\n【人格风格指令】\n{personality_prompt}"

        # 记忆提示
        memory_section = ""
        if memory_prompt:
            memory_section = f"\n\n【记忆影响】\n{memory_prompt}"

        # 策略提示
        strategy_section = ""
        if strategy_hint:
            strategy_section = f"\n\n【当前策略: {strategy}】\n{strategy_hint}"

        return base_prompt + strategy_section + personality_section + memory_section

    def _retrieve_rag_context(self, user_input: str) -> str:
        """海马体 RAG: 从情景记忆和知识库中检索相关内容

        Returns:
            格式化的检索结果文本，如果无结果返回空字符串
        """
        context_parts = []

        # 1. 将用户输入编码为查询向量
        try:
            query_vec = np.zeros(64, dtype=np.float32)
            for i, ch in enumerate(user_input[:64]):
                query_vec[i % 64] = (ord(ch) % 100) / 100.0
            # 加入用户输入的 hash 作为额外特征
            h = hash(user_input) % 1000
            query_vec[0] = h / 1000.0
        except (TypeError, ValueError):
            # Hash or numpy array creation failed
            query_vec = np.full(100, 0.5, dtype=np.float32)

        # 2. 海马体检索情景记忆
        try:
            episodes = self.hippocampus.retrieve(query_vec, top_k=3)
            if episodes:
                ep_texts = []
                for i, ep in enumerate(episodes):
                    ep_texts.append(
                        f"  [{i+1}] {ep.action} (奖励: {ep.reward:.2f})"
                    )
                context_parts.append(
                    "情景记忆:\n" + "\n".join(ep_texts)
                )
        except RuntimeError:
            pass  # PyTorch operation failed

        # 3. 知识库检索
        try:
            memories = self.memory.get_recent_memories(n=5)
            if memories:
                # 简单关键词匹配排序
                scored = []
                input_words = set(user_input.lower().split())
                for mem in memories:
                    mem_words = set(mem.content.lower().split())
                    overlap = len(input_words & mem_words)
                    if overlap > 0 or mem.importance > 0.7:
                        scored.append((overlap * 2 + mem.importance, mem))
                scored.sort(key=lambda x: x[0], reverse=True)

                if scored[:3]:
                    mem_texts = []
                    for score, mem in scored[:3]:
                        mem_texts.append(
                            f"  [{mem.content[:80]}] (相关度: {score:.1f})"
                        )
                    context_parts.append(
                        "知识记忆:\n" + "\n".join(mem_texts)
                    )
        except RuntimeError:
            pass  # PyTorch operation failed

        return "\n\n".join(context_parts) if context_parts else ""

    def _llm_tool_loop(
        self, messages: list[dict[str, str]], llm_params: dict[str, float]
    ) -> tuple:
        """LLM 生成 + 工具调用循环（使用神经递质调制的参数 + 生物系统工具过滤）

        Args:
            messages: 对话消息列表
            llm_params: 由 _compute_llm_params() 计算的参数

        Returns:
            (response_text, tool_calls_log)
        """
        max_rounds = getattr(self.config, 'tool_max_rounds', 5)
        tool_calls_log: list[dict] = []

        current_messages = list(messages)

        # ── 生物系统工具访问控制 ──
        bio_state = {
            'consciousness': self._internal_state.get('bsm_consciousness_gate', 0.5),
            'cortisol': self._internal_state.get('cortisol_level', self._internal_state.get('hormone_cortisol', 0.3)),
            'defense': self._internal_state.get('bsm_defense_behavior', ''),
            'alertness': self._internal_state.get('scn_alertness', 0.5),
        }
        available_tools = self.tools.get_available_tools(bio_state)

        # 如果可用工具为空，返回默认回复
        if not available_tools:
            return "我现在的状态不适合使用工具。", []

        logger.debug(f"[TOOL ACCESS] Available tools after bio-filter: {available_tools}")

        for round_idx in range(max_rounds):
            # 调用 LLM（使用神经递质调制的参数）
            try:
                response_text = self.api_client.chat(
                    messages=current_messages,
                    system_prompt=self._build_bio_prompt(),
                    temperature=llm_params["temperature"],
                    max_tokens=llm_params["max_tokens"],
                    top_p=llm_params["top_p"],
                    presence_penalty=llm_params.get("presence_penalty", 0.0),
                    frequency_penalty=llm_params.get("frequency_penalty", 0.0),
                )
            except Exception as e:
                response_text = f"[系统] 对话生成出错: {e}"
                break

            # 检查是否有工具调用
            tool_call = self.tools.parse_tool_call(response_text)
            if tool_call is None:
                # 无工具调用，直接返回
                # 清除回复中的工具调用标记残留
                response_text = self._clean_tool_markers(response_text)
                break

            tool_name, tool_args = tool_call

            # 检查工具是否在可用列表中
            if tool_name not in available_tools:
                logger.debug(f"[TOOL ACCESS] Blocked tool {tool_name} due to bio-state")
                # 通知 LLM 工具不可用
                response_text += f"\n\n[系统] 我现在的状态不允许使用 {tool_name} 工具。"
                break

            logger.debug(f"[TOOL] {tool_name}({tool_args})")

            # 执行工具
            tool_result = self.tools.execute(tool_name, tool_args)
            tool_calls_log.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result[:500],
            })
            logger.debug(f"[TOOL RESULT] {tool_result[:200]}")

            # 注入工具结果，继续循环
            current_messages.append({"role": "assistant", "content": response_text})
            current_messages.append({
                "role": "user",
                "content": f"[工具执行结果 ({tool_name})]: {tool_result}\n请基于以上工具结果继续回复用户。",
            })
        else:
            # 达到最大轮次
            response_text += "\n\n[系统提示: 已达到工具调用最大轮次]"

        return response_text, tool_calls_log

    @staticmethod
    def _clean_tool_markers(text: str) -> str:
        """清除回复中残留的工具调用标记"""
        import re
        return re.sub(r'\[TOOL:\s*\w+\s*\(.*?\)\s*\]', '', text, flags=re.DOTALL).strip()
