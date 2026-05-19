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

# 前额叶皮质
from civis_lucri_faber.core.prefrontal_cortex import (
    PrefrontalCortex,
    MaturationTracker,
    CostBenefitAnalyzer,
    ImpulseController,
    LongTermPlanner,
    WorkingMemory,
    CandidateEval,
    GoalNode,
    PlanStep,
)

# 角回 — 跨模态翻译器
from civis_lucri_faber.core.angular_gyrus import (
    AngularGyrus,
    ModalityProjector,
    TranslationMatrix,
    SemanticInterlingua,
    CrossModalPredictor,
    TemporalBindingBuffer,
    SceneDetector,
)

# 高级情绪系统
try:
    from civis_lucri_faber.core.advanced_emotion_integration import (
        IntegratedAdvancedEmotionSystem,
        AdvancedEmotionState,
        create_advanced_emotion_system,
        MODULES_AVAILABLE as ADVANCED_EMOTION_AVAILABLE,
    )
except ImportError:
    try:
        from core.advanced_emotion_integration import (
            IntegratedAdvancedEmotionSystem,
            AdvancedEmotionState,
            create_advanced_emotion_system,
            MODULES_AVAILABLE as ADVANCED_EMOTION_AVAILABLE,
        )
    except ImportError:
        ADVANCED_EMOTION_AVAILABLE = False

# 情绪调节系统
from civis_lucri_faber.core.emotion_regulation import (
    EmotionRegulationSystem,
    PrefrontalRegulation,
    MetabolicRegulation,
    MetabolicState,
    SocialRegulation,
)

# 心境系统
from civis_lucri_faber.core.mood_system import (
    MoodSystem,
    MoodState,
)

# 情绪记忆巩固
from civis_lucri_faber.core.emotion_memory_consolidation import (
    EmotionalMemoryConsolidation,
    SleepStageData,
)

# 社会情绪
from civis_lucri_faber.core.social_emotions import (
    SocialEmotionSystem,
)

# 情绪传染
from civis_lucri_faber.core.emotional_contagion import (
    EmotionalContagionSystem,
)

# 内感受系统
from civis_lucri_faber.core.interoception import (
    InteroceptionSystem,
    InteroceptiveState,
    GutState,
)

# 情绪动力学
from civis_lucri_faber.core.emotion_dynamics import (
    EmotionDynamicsSystem,
)

# 神经修剪系统
from civis_lucri_faber.core.neural_pruning import (
    NeuralPruningSystem,
    PruningConfig,
    NeuronState,
    ActivityTracker,
    create_neural_pruning_system,
)

# 神经自调节系统
from civis_lucri_faber.core.autonomic_nervous_system import (
    AutonomicNervousSystem,
    ANSState,
    SympatheticBranch,
    ParasympatheticBranch,
    BaroreceptorReflex,
    PolyvagalSystem,
    create_autonomic_nervous_system,
)
from civis_lucri_faber.core.hpa_axis import (
    HPAAxis,
    HPAState,
    HypothalamicCRH,
    PituitaryACTH,
    AdrenalCortex,
    NegativeFeedbackLoop,
    AllostaticLoadTracker,
    create_hpa_axis,
)
from civis_lucri_faber.core.glial_system import (
    GlialSystem,
    GlialState,
    AstrocyteSystem,
    MicrogliaSystem,
    OligodendrocyteSystem,
    create_glial_system,
)
from civis_lucri_faber.core.allostatic_regulation import (
    AllostaticRegulation,
    AllostaticState,
    PredictiveRegulator,
    LoadAccumulator,
    RegimeSelector,
    create_allostatic_regulation,
)
from civis_lucri_faber.core.predictive_coding import (
    PredictiveCodingSystem,
    GenerativeLayer,
    HierarchicalGenerativeModel,
    PrecisionModulator,
    ActiveInferenceController,
    create_predictive_coding_system,
)

# 社会认知系统
from civis_lucri_faber.core.social_cognition import (
    SocialCognitionSystem,
    SocialCognitionState,
    MirrorNeuronSystem,
    MirrorState,
    TheoryOfMind,
    ToMState,
    EmpathyCircuit,
    EmpathyState,
    ImitationLearning,
    ImitationState,
    SocialPredictor,
    create_social_cognition,
)

# 脑干系统
from civis_lucri_faber.core.brainstem import (
    Brainstem,
    BrainstemState,
    RespiratoryPhase,
    ArousalLevel,
    DefensiveBehavior,
    RespiratoryRhythmGenerator,
    ReticularActivatingSystem,
    PeriaqueductalGray,
    MedullaryCardiovascularCenter,
    create_brainstem,
)

# 发音语言系统 (Bio-Inspired Vocalization)
from civis_lucri_faber.core.vocalization import (
    VocalCortex,
    VocalTract,
    ArticulatoryPlanner,
    SpeechProductionPipeline,
    FormantSynthesizer,
    VocalizationOutput,
    PHONEMES,
    PHONEME_TO_IDX,
    N_PHONEMES,
    text_to_phoneme_indices,
    describe_phoneme,
)

# 共振峰波形合成器 (Formant -> Waveform)
from civis_lucri_faber.core.formant_synthesis import (
    FormantToWaveform,
    create_formant_synthesizer,
)

# 心理治疗系统 (Psychotherapy)
from civis_lucri_faber.core.psychotherapy import (
    TherapyModality,
    SessionFrequency,
    TherapyPhase,
    TherapySession,
    TherapyProgress,
    PsychotherapySystem,
    conduct_therapy_session,
    compute_resistance,
    compute_compliance,
    THERAPY_TARGETS,
    create_psychotherapy_system,
)

# 药物-治疗协同引擎 (Pharmacotherapy Synergy)
from civis_lucri_faber.core.pharmacotherapy_synergy import (
    SynergyType,
    SynergyRecord,
    SynergyCalculator,
    SYNERGY_MATRIX,
    create_synergy_calculator,
)

# 计算精神药理学沙盒 (Psychopharmacology Sandbox)
from civis_lucri_faber.core.psychopharmacology_sandbox import (
    ExperimentMode,
    TreatmentArm,
    ExperimentDesign,
    ArmResult,
    ExperimentResult,
    PsychopharmacologySandbox,
    create_sandbox,
)

# 统一状态键映射
from civis_lucri_faber.core.state_key_mapping import UnifiedStateMapping

# PD靶点映射
from civis_lucri_faber.core.pd_target_mapping import (
    PDTarget,
    build_pd_targets,
    compute_pd_deltas,
)

# 治疗实验 (Therapeutic Experiment)
from civis_lucri_faber.core.therapeutic_experiment import (
    DrugConfig,
    TherapyConfig,
    ExperimentConfig,
    TreatmentTimepoint,
    TherapeuticResult,
    TherapeuticExperiment,
)

# 心理测量指标
from civis_lucri_faber.core.psychometric_indicators import (
    PsychometricSnapshot,
    PsychometricIndicatorTracker,
)

# LLM临床评估器
from civis_lucri_faber.core.llm_evaluator import (
    LLMEvaluationResult,
    LLMEvaluator,
)

# 治疗报告生成器
from civis_lucri_faber.core.therapeutic_report import generate_markdown_report

# 药物相互作用 (DDI)
from civis_lucri_faber.core.drug_pipeline.ddi import (
    DDIPairRecord,
    DDIResult,
    assess_ddi,
    combine_pd_deltas,
    compute_step_ke_modifiers,
)

# 受体亚型药效动力学
from civis_lucri_faber.core.drug_pipeline.receptor_pd import (
    ReceptorSubtype,
    ReceptorPDTarget,
    RECEPTOR_REGISTRY,
    DRUG_RECEPTOR_AFFINITY,
    build_receptor_pd_targets,
    compute_receptor_deltas,
    aggregate_receptor_to_nt,
)

# 成瘾动力学
from civis_lucri_faber.core.addiction_dynamics import (
    ToleranceState,
    WithdrawalState,
    CravingState,
    AddictionProfile,
    AddictionDynamicsEngine,
)

# 病原体神经炎症
from civis_lucri_faber.core.pathogen_neuroinflammation import (
    PathogenProfile,
    PathogenState,
    PathogenTriggeredInflammationEngine,
    PATHOGEN_REGISTRY,
)

# 症状追踪器
from civis_lucri_faber.core.symptom_tracker import (
    SymptomEpisode,
    PersistentSymptom,
    SymptomSnapshot,
    SymptomTracker,
)

# 硬件生命体征桥接
from civis_lucri_faber.core.hardware_vitals import (
    HardwareState,
    HardwareVitals,
    create_hardware_vitals,
)

# 自我意识中枢
from civis_lucri_faber.core.self_awareness import (
    SelfAwarenessCenter,
    mPFCState,
    PCCState,
    PrecuneusState,
    DMNState,
    SelfOtherState,
    MetaSelfState,
    SelfAwarenessState,
    MedialPrefrontalCortex,
    PosteriorCingulateCortex,
    PrecuneusSystem,
    DefaultModeNetwork,
    SelfOtherDistinction,
    MetaSelfAwareness,
    create_self_awareness_center,
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
    # 前额叶皮质
    "PrefrontalCortex",
    "MaturationTracker",
    "CostBenefitAnalyzer",
    "ImpulseController",
    "LongTermPlanner",
    "WorkingMemory",
    "CandidateEval",
    "GoalNode",
    "PlanStep",
    # 角回
    "AngularGyrus",
    "ModalityProjector",
    "TranslationMatrix",
    "SemanticInterlingua",
    "CrossModalPredictor",
    "TemporalBindingBuffer",
    "SceneDetector",
    # 高级情绪系统
    "IntegratedAdvancedEmotionSystem",
    "AdvancedEmotionState",
    "create_advanced_emotion_system",
    "ADVANCED_EMOTION_AVAILABLE",
    # 情绪调节
    "EmotionRegulationSystem",
    "PrefrontalRegulation",
    "MetabolicRegulation",
    "SocialRegulation",
    # 心境
    "MoodSystem",
    # 情绪记忆
    "EmotionalMemoryConsolidation",
    "SleepStageData",
    # 社会情绪
    "SocialEmotionSystem",
    # 情绪传染
    "EmotionalContagionSystem",
    # 内感受
    "InteroceptionSystem",
    "InteroceptiveState",
    "GutState",
    # 情绪动力学
    "EmotionDynamicsSystem",
    # 神经修剪
    "NeuralPruningSystem",
    "PruningConfig",
    "NeuronState",
    "ActivityTracker",
    "create_neural_pruning_system",
    # 神经自调节 - ANS
    "AutonomicNervousSystem",
    "ANSState",
    "SympatheticBranch",
    "ParasympatheticBranch",
    "BaroreceptorReflex",
    "PolyvagalSystem",
    "create_autonomic_nervous_system",
    # 神经自调节 - HPA轴
    "HPAAxis",
    "HPAState",
    "HypothalamicCRH",
    "PituitaryACTH",
    "AdrenalCortex",
    "NegativeFeedbackLoop",
    "AllostaticLoadTracker",
    "create_hpa_axis",
    # 神经自调节 - 胶质系统
    "GlialSystem",
    "GlialState",
    "AstrocyteSystem",
    "MicrogliaSystem",
    "OligodendrocyteSystem",
    "create_glial_system",
    # 神经自调节 - 稳态调节
    "AllostaticRegulation",
    "AllostaticState",
    "PredictiveRegulator",
    "LoadAccumulator",
    "RegimeSelector",
    "create_allostatic_regulation",
    # 神经自调节 - 预测编码
    "PredictiveCodingSystem",
    "GenerativeLayer",
    "HierarchicalGenerativeModel",
    "PrecisionModulator",
    "ActiveInferenceController",
    "create_predictive_coding_system",
    # 社会认知
    "SocialCognitionSystem",
    "SocialCognitionState",
    "MirrorNeuronSystem",
    "MirrorState",
    "TheoryOfMind",
    "ToMState",
    "EmpathyCircuit",
    "EmpathyState",
    "ImitationLearning",
    "ImitationState",
    "SocialPredictor",
    "create_social_cognition",
    # 脑干
    "Brainstem",
    "BrainstemState",
    "RespiratoryPhase",
    "ArousalLevel",
    "DefensiveBehavior",
    "RespiratoryRhythmGenerator",
    "ReticularActivatingSystem",
    "PeriaqueductalGray",
    "MedullaryCardiovascularCenter",
    "create_brainstem",
    # 发音语言系统
    "VocalCortex",
    "VocalTract",
    "ArticulatoryPlanner",
    "SpeechProductionPipeline",
    "FormantSynthesizer",
    "VocalizationOutput",
    "PHONEMES",
    "PHONEME_TO_IDX",
    "N_PHONEMES",
    "text_to_phoneme_indices",
    "describe_phoneme",
    # 共振峰波形合成
    "FormantToWaveform",
    "create_formant_synthesizer",
    # 心理治疗系统
    "TherapyModality",
    "SessionFrequency",
    "TherapyPhase",
    "TherapySession",
    "TherapyProgress",
    "PsychotherapySystem",
    "conduct_therapy_session",
    "compute_resistance",
    "compute_compliance",
    "THERAPY_TARGETS",
    "create_psychotherapy_system",
    # 药物-治疗协同引擎
    "SynergyType",
    "SynergyRecord",
    "SynergyCalculator",
    "SYNERGY_MATRIX",
    "create_synergy_calculator",
    # 计算精神药理学沙盒
    "ExperimentMode",
    "TreatmentArm",
    "ExperimentDesign",
    "ArmResult",
    "ExperimentResult",
    "PsychopharmacologySandbox",
    "create_sandbox",
    # 统一状态键映射
    "UnifiedStateMapping",
    # PD靶点映射
    "PDTarget",
    "build_pd_targets",
    "compute_pd_deltas",
    # 治疗实验
    "DrugConfig",
    "TherapyConfig",
    "ExperimentConfig",
    "TreatmentTimepoint",
    "TherapeuticResult",
    "TherapeuticExperiment",
    # 心理测量指标
    "PsychometricSnapshot",
    "PsychometricIndicatorTracker",
    # LLM临床评估器
    "LLMEvaluationResult",
    "LLMEvaluator",
    # 治疗报告生成器
    "generate_markdown_report",
    # 药物相互作用 (DDI)
    "DDIPairRecord",
    "DDIResult",
    "assess_ddi",
    "combine_pd_deltas",
    "compute_step_ke_modifiers",
    # 受体亚型药效动力学
    "ReceptorSubtype",
    "ReceptorPDTarget",
    "RECEPTOR_REGISTRY",
    "DRUG_RECEPTOR_AFFINITY",
    "build_receptor_pd_targets",
    "compute_receptor_deltas",
    "aggregate_receptor_to_nt",
    # 成瘾动力学
    "ToleranceState",
    "WithdrawalState",
    "CravingState",
    "AddictionProfile",
    "AddictionDynamicsEngine",
    # 病原体神经炎症
    "PathogenProfile",
    "PathogenState",
    "PathogenTriggeredInflammationEngine",
    "PATHOGEN_REGISTRY",
    # 症状追踪器
    "SymptomEpisode",
    "PersistentSymptom",
    "SymptomSnapshot",
    "SymptomTracker",
    # 自我意识中枢
    "SelfAwarenessCenter",
    "mPFCState",
    "PCCState",
    "PrecuneusState",
    "DMNState",
    "SelfOtherState",
    "MetaSelfState",
    "SelfAwarenessState",
    "MedialPrefrontalCortex",
    "PosteriorCingulateCortex",
    "PrecuneusSystem",
    "DefaultModeNetwork",
    "SelfOtherDistinction",
    "MetaSelfAwareness",
    "create_self_awareness_center",
]