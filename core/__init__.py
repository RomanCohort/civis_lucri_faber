"""Simulacrum Core Modules"""

import sys
import os

# 处理相对导入
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 尝试从simulacrum导入，失败则从core导入
def _import_module(module_name, symbols):
    """Helper to import from simulacrum.core or fall back to core"""
    try:
        module = __import__(f'simulacrum.core.{module_name}', fromlist=symbols)
        return {s: getattr(module, s) for s in symbols}
    except ImportError:
        module = __import__(f'core.{module_name}', fromlist=symbols)
        return {s: getattr(module, s) for s in symbols}

# 导入所有模块
try:
    from simulacrum.core.agent import Simulacrum
except ImportError:
    from core.agent import Simulacrum

try:
    from simulacrum.core.curiosity import CuriosityEngine
except ImportError:
    from core.curiosity import CuriosityEngine

try:
    from simulacrum.core.information_gain import InformationGainCalculator
except ImportError:
    from core.information_gain import InformationGainCalculator

try:
    from simulacrum.core.meta_learning import MetaLearner, ActiveLearner
except ImportError:
    from core.meta_learning import MetaLearner, ActiveLearner

try:
    from simulacrum.core.self_alignment import SelfAlignmentModule
except ImportError:
    from core.self_alignment import SelfAlignmentModule

try:
    from simulacrum.core.thermodynamics import ThermodynamicsSystem
except ImportError:
    from core.thermodynamics import ThermodynamicsSystem

try:
    from simulacrum.core.metabolic_budget import (
        MetabolicBudget, MetabolicCostCalculator, PeriodicStarvation, MetabolicState, create_metabolic_budget,
    )
except ImportError:
    from core.metabolic_budget import (
        MetabolicBudget, MetabolicCostCalculator, PeriodicStarvation, MetabolicState, create_metabolic_budget,
    )

try:
    from simulacrum.core.sleep import SleepSystem, SleepController, SleepStage, MemoryReplayer, create_sleep_system
except ImportError:
    from core.sleep import SleepSystem, SleepController, SleepStage, MemoryReplayer, create_sleep_system

try:
    from simulacrum.core.neuromodulation_integration import (
        NeuromodulationIntegration, RewardModulation, TemporalDiscount, AttentionModulator, create_neuromodulation_integration,
    )
except ImportError:
    from core.neuromodulation_integration import (
        NeuromodulationIntegration, RewardModulation, TemporalDiscount, AttentionModulator, create_neuromodulation_integration,
    )

try:
    from simulacrum.core.cerebello_spinal import (
        CerebelloSpinalCoordination, SpinalCord, CentralPatternGenerator, ReflexPathway, Cerebellum, create_cerebello_spinal,
    )
except ImportError:
    from core.cerebello_spinal import (
        CerebelloSpinalCoordination, SpinalCord, CentralPatternGenerator, ReflexPathway, Cerebellum, create_cerebello_spinal,
    )

try:
    from simulacrum.core.basal_ganglia import BasalGangliaSystem, BasalGanglia, create_basal_ganglia
except ImportError:
    from core.basal_ganglia import BasalGangliaSystem, BasalGanglia, create_basal_ganglia

try:
    from simulacrum.core.hippocampus import Hippocampus, create_hippocampus
except ImportError:
    from core.hippocampus import Hippocampus, create_hippocampus

try:
    from simulacrum.core.limbic import LimbicSystem, Amygdala, Thalamus, create_limbic_system
except ImportError:
    from core.limbic import LimbicSystem, Amygdala, Thalamus, create_limbic_system

try:
    from simulacrum.core.neurotransmitter import (
        NeurotransmitterSystem, DopamineSystem, SerotoninSystem, AcetylcholineSystem, create_neurotransmitter_system,
    )
except ImportError:
    from core.neurotransmitter import (
        NeurotransmitterSystem, DopamineSystem, SerotoninSystem, AcetylcholineSystem, create_neurotransmitter_system,
    )

try:
    from simulacrum.core.prefrontal_cortex import (
        PrefrontalCortex, MaturationTracker, CostBenefitAnalyzer, ImpulseController, LongTermPlanner, WorkingMemory, CandidateEval, GoalNode, PlanStep,
    )
except ImportError:
    from core.prefrontal_cortex import (
        PrefrontalCortex, MaturationTracker, CostBenefitAnalyzer, ImpulseController, LongTermPlanner, WorkingMemory, CandidateEval, GoalNode, PlanStep,
    )

try:
    from simulacrum.core.angular_gyrus import (
        AngularGyrus, ModalityProjector, TranslationMatrix, SemanticInterlingua, CrossModalPredictor, TemporalBindingBuffer, SceneDetector,
    )
except ImportError:
    from core.angular_gyrus import (
        AngularGyrus, ModalityProjector, TranslationMatrix, SemanticInterlingua, CrossModalPredictor, TemporalBindingBuffer, SceneDetector,
    )

# 高级情绪系统
try:
    from simulacrum.core.advanced_emotion_integration import (
        IntegratedAdvancedEmotionSystem, AdvancedEmotionState, create_advanced_emotion_system, MODULES_AVAILABLE as ADVANCED_EMOTION_AVAILABLE,
    )
except ImportError:
    try:
        from core.advanced_emotion_integration import (
            IntegratedAdvancedEmotionSystem, AdvancedEmotionState, create_advanced_emotion_system, MODULES_AVAILABLE as ADVANCED_EMOTION_AVAILABLE,
        )
    except ImportError:
        ADVANCED_EMOTION_AVAILABLE = False

try:
    from simulacrum.core.emotion_regulation import EmotionRegulationSystem, PrefrontalRegulation, MetabolicRegulation, MetabolicState, SocialRegulation
except ImportError:
    from core.emotion_regulation import EmotionRegulationSystem, PrefrontalRegulation, MetabolicRegulation, MetabolicState, SocialRegulation

try:
    from simulacrum.core.mood_system import MoodSystem, MoodState
except ImportError:
    from core.mood_system import MoodSystem, MoodState

try:
    from simulacrum.core.emotion_memory_consolidation import EmotionalMemoryConsolidation, SleepStageData
except ImportError:
    from core.emotion_memory_consolidation import EmotionalMemoryConsolidation, SleepStageData

try:
    from simulacrum.core.social_emotions import SocialEmotionSystem
except ImportError:
    from core.social_emotions import SocialEmotionSystem

try:
    from simulacrum.core.emotional_contagion import EmotionalContagionSystem
except ImportError:
    from core.emotional_contagion import EmotionalContagionSystem

try:
    from simulacrum.core.interoception import InteroceptionSystem, InteroceptiveState, GutState
except ImportError:
    from core.interoception import InteroceptionSystem, InteroceptiveState, GutState

try:
    from simulacrum.core.emotion_dynamics import EmotionDynamicsSystem
except ImportError:
    from core.emotion_dynamics import EmotionDynamicsSystem

try:
    from simulacrum.core.neural_pruning import NeuralPruningSystem, PruningConfig, NeuronState, ActivityTracker, create_neural_pruning_system
except ImportError:
    from core.neural_pruning import NeuralPruningSystem, PruningConfig, NeuronState, ActivityTracker, create_neural_pruning_system

try:
    from simulacrum.core.autonomic_nervous_system import (
        AutonomicNervousSystem, ANSState, SympatheticBranch, ParasympatheticBranch, BaroreceptorReflex, PolyvagalSystem, create_autonomic_nervous_system,
    )
except ImportError:
    from core.autonomic_nervous_system import (
        AutonomicNervousSystem, ANSState, SympatheticBranch, ParasympatheticBranch, BaroreceptorReflex, PolyvagalSystem, create_autonomic_nervous_system,
    )

try:
    from simulacrum.core.hpa_axis import (
        HPAAxis, HPAState, HypothalamicCRH, PituitaryACTH, AdrenalCortex, NegativeFeedbackLoop, AllostaticLoadTracker, create_hpa_axis,
    )
except ImportError:
    from core.hpa_axis import (
        HPAAxis, HPAState, HypothalamicCRH, PituitaryACTH, AdrenalCortex, NegativeFeedbackLoop, AllostaticLoadTracker, create_hpa_axis,
    )

try:
    from simulacrum.core.glial_system import GlialSystem, GlialState, AstrocyteSystem, MicrogliaSystem, OligodendrocyteSystem, create_glial_system
except ImportError:
    from core.glial_system import GlialSystem, GlialState, AstrocyteSystem, MicrogliaSystem, OligodendrocyteSystem, create_glial_system

try:
    from simulacrum.core.allostatic_regulation import AllostaticRegulation, AllostaticState, PredictiveRegulator, LoadAccumulator, RegimeSelector, create_allostatic_regulation
except ImportError:
    from core.allostatic_regulation import AllostaticRegulation, AllostaticState, PredictiveRegulator, LoadAccumulator, RegimeSelector, create_allostatic_regulation

try:
    from simulacrum.core.predictive_coding import PredictiveCodingSystem, GenerativeLayer, HierarchicalGenerativeModel, PrecisionModulator, ActiveInferenceController, create_predictive_coding_system
except ImportError:
    from core.predictive_coding import PredictiveCodingSystem, GenerativeLayer, HierarchicalGenerativeModel, PrecisionModulator, ActiveInferenceController, create_predictive_coding_system

try:
    from simulacrum.core.social_cognition import (
        SocialCognitionSystem, SocialCognitionState, MirrorNeuronSystem, MirrorState, TheoryOfMind, ToMState, EmpathyCircuit, EmpathyState, ImitationLearning, ImitationState, SocialPredictor, create_social_cognition,
    )
except ImportError:
    from core.social_cognition import (
        SocialCognitionSystem, SocialCognitionState, MirrorNeuronSystem, MirrorState, TheoryOfMind, ToMState, EmpathyCircuit, EmpathyState, ImitationLearning, ImitationState, SocialPredictor, create_social_cognition,
    )

try:
    from simulacrum.core.brainstem import (
        Brainstem, BrainstemState, RespiratoryPhase, ArousalLevel, DefensiveBehavior, RespiratoryRhythmGenerator, ReticularActivatingSystem, PeriaqueductalGray, MedullaryCardiovascularCenter, create_brainstem,
    )
except ImportError:
    from core.brainstem import (
        Brainstem, BrainstemState, RespiratoryPhase, ArousalLevel, DefensiveBehavior, RespiratoryRhythmGenerator, ReticularActivatingSystem, PeriaqueductalGray, MedullaryCardiovascularCenter, create_brainstem,
    )

try:
    from simulacrum.core.vocalization import (
        VocalCortex, VocalTract, ArticulatoryPlanner, SpeechProductionPipeline, FormantSynthesizer, VocalizationOutput, PHONEMES, PHONEME_TO_IDX, N_PHONEMES, text_to_phoneme_indices, describe_phoneme,
    )
except ImportError:
    from core.vocalization import (
        VocalCortex, VocalTract, ArticulatoryPlanner, SpeechProductionPipeline, FormantSynthesizer, VocalizationOutput, PHONEMES, PHONEME_TO_IDX, N_PHONEMES, text_to_phoneme_indices, describe_phoneme,
    )

try:
    from simulacrum.core.formant_synthesis import FormantToWaveform, create_formant_synthesizer
except ImportError:
    from core.formant_synthesis import FormantToWaveform, create_formant_synthesizer

try:
    from simulacrum.core.psychotherapy import (
        TherapyModality, SessionFrequency, TherapyPhase, TherapySession, TherapyProgress, PsychotherapySystem, conduct_therapy_session, compute_resistance, compute_compliance, THERAPY_TARGETS, create_psychotherapy_system,
    )
except ImportError:
    from core.psychotherapy import (
        TherapyModality, SessionFrequency, TherapyPhase, TherapySession, TherapyProgress, PsychotherapySystem, conduct_therapy_session, compute_resistance, compute_compliance, THERAPY_TARGETS, create_psychotherapy_system,
    )

try:
    from simulacrum.core.pharmacotherapy_synergy import SynergyType, SynergyRecord, SynergyCalculator, SYNERGY_MATRIX, create_synergy_calculator
except ImportError:
    from core.pharmacotherapy_synergy import SynergyType, SynergyRecord, SynergyCalculator, SYNERGY_MATRIX, create_synergy_calculator

try:
    from simulacrum.core.psychopharmacology_sandbox import ExperimentMode, TreatmentArm, ExperimentDesign, ArmResult, ExperimentResult, PsychopharmacologySandbox, create_sandbox
except ImportError:
    from core.psychopharmacology_sandbox import ExperimentMode, TreatmentArm, ExperimentDesign, ArmResult, ExperimentResult, PsychopharmacologySandbox, create_sandbox

try:
    from simulacrum.core.state_key_mapping import UnifiedStateMapping
except ImportError:
    from core.state_key_mapping import UnifiedStateMapping

try:
    from simulacrum.core.pd_target_mapping import PDTarget, build_pd_targets, compute_pd_deltas
except ImportError:
    from core.pd_target_mapping import PDTarget, build_pd_targets, compute_pd_deltas

try:
    from simulacrum.core.therapeutic_experiment import DrugConfig, TherapyConfig, ExperimentConfig, TreatmentTimepoint, TherapeuticResult, TherapeuticExperiment
except ImportError:
    from core.therapeutic_experiment import DrugConfig, TherapyConfig, ExperimentConfig, TreatmentTimepoint, TherapeuticResult, TherapeuticExperiment

try:
    from simulacrum.core.psychometric_indicators import PsychometricSnapshot, PsychometricIndicatorTracker
except ImportError:
    from core.psychometric_indicators import PsychometricSnapshot, PsychometricIndicatorTracker

try:
    from simulacrum.core.llm_evaluator import LLMEvaluationResult, LLMEvaluator
except ImportError:
    from core.llm_evaluator import LLMEvaluationResult, LLMEvaluator

try:
    from simulacrum.core.therapeutic_report import generate_markdown_report
except ImportError:
    from core.therapeutic_report import generate_markdown_report

try:
    from simulacrum.core.drug_pipeline.ddi import DDIPairRecord, DDIResult, assess_ddi, combine_pd_deltas, compute_step_ke_modifiers
except ImportError:
    from core.drug_pipeline.ddi import DDIPairRecord, DDIResult, assess_ddi, combine_pd_deltas, compute_step_ke_modifiers

try:
    from simulacrum.core.drug_pipeline.receptor_pd import ReceptorSubtype, ReceptorPDTarget, RECEPTOR_REGISTRY, DRUG_RECEPTOR_AFFINITY, build_receptor_pd_targets, compute_receptor_deltas, aggregate_receptor_to_nt
except ImportError:
    from core.drug_pipeline.receptor_pd import ReceptorSubtype, ReceptorPDTarget, RECEPTOR_REGISTRY, DRUG_RECEPTOR_AFFINITY, build_receptor_pd_targets, compute_receptor_deltas, aggregate_receptor_to_nt

try:
    from simulacrum.core.addiction_dynamics import ToleranceState, WithdrawalState, CravingState, AddictionProfile, AddictionDynamicsEngine
except ImportError:
    from core.addiction_dynamics import ToleranceState, WithdrawalState, CravingState, AddictionProfile, AddictionDynamicsEngine

try:
    from simulacrum.core.pathogen_neuroinflammation import PathogenProfile, PathogenState, PathogenTriggeredInflammationEngine, PATHOGEN_REGISTRY
except ImportError:
    from core.pathogen_neuroinflammation import PathogenProfile, PathogenState, PathogenTriggeredInflammationEngine, PATHOGEN_REGISTRY

try:
    from simulacrum.core.symptom_tracker import SymptomEpisode, PersistentSymptom, SymptomSnapshot, SymptomTracker
except ImportError:
    from core.symptom_tracker import SymptomEpisode, PersistentSymptom, SymptomSnapshot, SymptomTracker

try:
    from simulacrum.core.hardware_vitals import HardwareState, HardwareVitals, create_hardware_vitals
except ImportError:
    from core.hardware_vitals import HardwareState, HardwareVitals, create_hardware_vitals

try:
    from simulacrum.core.self_awareness import (
        SelfAwarenessCenter, mPFCState, PCCState, PrecuneusState, DMNState, SelfOtherState, MetaSelfState, SelfAwarenessState, MedialPrefrontalCortex, PosteriorCingulateCortex, PrecuneusSystem, DefaultModeNetwork, SelfOtherDistinction, MetaSelfAwareness, create_self_awareness_center,
    )
except ImportError:
    from core.self_awareness import (
        SelfAwarenessCenter, mPFCState, PCCState, PrecuneusState, DMNState, SelfOtherState, MetaSelfState, SelfAwarenessState, MedialPrefrontalCortex, PosteriorCingulateCortex, PrecuneusSystem, DefaultModeNetwork, SelfOtherDistinction, MetaSelfAwareness, create_self_awareness_center,
    )

__all__ = [
    "Simulacrum", "CuriosityEngine", "InformationGainCalculator", "MetaLearner", "ActiveLearner", "SelfAlignmentModule", "ThermodynamicsSystem",
    "MetabolicBudget", "MetabolicCostCalculator", "PeriodicStarvation", "MetabolicState", "create_metabolic_budget",
    "SleepSystem", "SleepController", "SleepStage", "MemoryReplayer", "create_sleep_system",
    "NeuromodulationIntegration", "RewardModulation", "TemporalDiscount", "AttentionModulator", "create_neuromodulation_integration",
    "CerebelloSpinalCoordination", "SpinalCord", "CentralPatternGenerator", "ReflexPathway", "Cerebellum", "create_cerebello_spinal",
    "BasalGangliaSystem", "BasalGanglia", "create_basal_ganglia",
    "Hippocampus", "create_hippocampus",
    "LimbicSystem", "Amygdala", "Thalamus", "create_limbic_system",
    "NeurotransmitterSystem", "DopamineSystem", "SerotoninSystem", "AcetylcholineSystem", "create_neurotransmitter_system",
    "PrefrontalCortex", "MaturationTracker", "CostBenefitAnalyzer", "ImpulseController", "LongTermPlanner", "WorkingMemory", "CandidateEval", "GoalNode", "PlanStep",
    "AngularGyrus", "ModalityProjector", "TranslationMatrix", "SemanticInterlingua", "CrossModalPredictor", "TemporalBindingBuffer", "SceneDetector",
    "IntegratedAdvancedEmotionSystem", "AdvancedEmotionState", "create_advanced_emotion_system", "ADVANCED_EMOTION_AVAILABLE",
    "EmotionRegulationSystem", "PrefrontalRegulation", "MetabolicRegulation", "SocialRegulation",
    "MoodSystem", "EmotionalMemoryConsolidation", "SleepStageData", "SocialEmotionSystem", "EmotionalContagionSystem",
    "InteroceptionSystem", "InteroceptiveState", "GutState", "EmotionDynamicsSystem",
    "NeuralPruningSystem", "PruningConfig", "NeuronState", "ActivityTracker", "create_neural_pruning_system",
    "AutonomicNervousSystem", "ANSState", "SympatheticBranch", "ParasympatheticBranch", "BaroreceptorReflex", "PolyvagalSystem", "create_autonomic_nervous_system",
    "HPAAxis", "HPAState", "HypothalamicCRH", "PituitaryACTH", "AdrenalCortex", "NegativeFeedbackLoop", "AllostaticLoadTracker", "create_hpa_axis",
    "GlialSystem", "GlialState", "AstrocyteSystem", "MicrogliaSystem", "OligodendrocyteSystem", "create_glial_system",
    "AllostaticRegulation", "AllostaticState", "PredictiveRegulator", "LoadAccumulator", "RegimeSelector", "create_allostatic_regulation",
    "PredictiveCodingSystem", "GenerativeLayer", "HierarchicalGenerativeModel", "PrecisionModulator", "ActiveInferenceController", "create_predictive_coding_system",
    "SocialCognitionSystem", "SocialCognitionState", "MirrorNeuronSystem", "MirrorState", "TheoryOfMind", "ToMState", "EmpathyCircuit", "EmpathyState", "ImitationLearning", "ImitationState", "SocialPredictor", "create_social_cognition",
    "Brainstem", "BrainstemState", "RespiratoryPhase", "ArousalLevel", "DefensiveBehavior", "RespiratoryRhythmGenerator", "ReticularActivatingSystem", "PeriaqueductalGray", "MedullaryCardiovascularCenter", "create_brainstem",
    "VocalCortex", "VocalTract", "ArticulatoryPlanner", "SpeechProductionPipeline", "FormantSynthesizer", "VocalizationOutput", "PHONEMES", "PHONEME_TO_IDX", "N_PHONEMES", "text_to_phoneme_indices", "describe_phoneme",
    "FormantToWaveform", "create_formant_synthesizer",
    "TherapyModality", "SessionFrequency", "TherapyPhase", "TherapySession", "TherapyProgress", "PsychotherapySystem", "conduct_therapy_session", "compute_resistance", "compute_compliance", "THERAPY_TARGETS", "create_psychotherapy_system",
    "SynergyType", "SynergyRecord", "SynergyCalculator", "SYNERGY_MATRIX", "create_synergy_calculator",
    "ExperimentMode", "TreatmentArm", "ExperimentDesign", "ArmResult", "ExperimentResult", "PsychopharmacologySandbox", "create_sandbox",
    "UnifiedStateMapping", "PDTarget", "build_pd_targets", "compute_pd_deltas",
    "DrugConfig", "TherapyConfig", "ExperimentConfig", "TreatmentTimepoint", "TherapeuticResult", "TherapeuticExperiment",
    "PsychometricSnapshot", "PsychometricIndicatorTracker", "LLMEvaluationResult", "LLMEvaluator", "generate_markdown_report",
    "DDIPairRecord", "DDIResult", "assess_ddi", "combine_pd_deltas", "compute_step_ke_modifiers",
    "ReceptorSubtype", "ReceptorPDTarget", "RECEPTOR_REGISTRY", "DRUG_RECEPTOR_AFFINITY", "build_receptor_pd_targets", "compute_receptor_deltas", "aggregate_receptor_to_nt",
    "ToleranceState", "WithdrawalState", "CravingState", "AddictionProfile", "AddictionDynamicsEngine",
    "PathogenProfile", "PathogenState", "PathogenTriggeredInflammationEngine", "PATHOGEN_REGISTRY",
    "SymptomEpisode", "PersistentSymptom", "SymptomSnapshot", "SymptomTracker",
    "HardwareState", "HardwareVitals", "create_hardware_vitals",
    "SelfAwarenessCenter", "mPFCState", "PCCState", "PrecuneusState", "DMNState", "SelfOtherState", "MetaSelfState", "SelfAwarenessState", "MedialPrefrontalCortex", "PosteriorCingulateCortex", "PrecuneusSystem", "DefaultModeNetwork", "SelfOtherDistinction", "MetaSelfAwareness", "create_self_awareness_center",
]
