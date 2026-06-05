"""Simulacrum Core Modules - Lazy-loaded brain region components."""

import os
import sys

# Handle relative imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Mapping of exported names to their source modules
_SUBMODULES = {
    # Agent
    "Simulacrum": "core.agent",
    # Cognitive systems
    "CuriosityEngine": "core.curiosity",
    "InformationGainCalculator": "core.information_gain",
    "ActiveLearner": "core.meta_learning",
    "MetaLearner": "core.meta_learning",
    "SelfAlignmentModule": "core.self_alignment",
    "ThermodynamicsSystem": "core.thermodynamics",
    # Metabolic system
    "MetabolicBudget": "core.metabolic_budget",
    "MetabolicCostCalculator": "core.metabolic_budget",
    "MetabolicState": "core.metabolic_budget",
    "PeriodicStarvation": "core.metabolic_budget",
    "create_metabolic_budget": "core.metabolic_budget",
    # Sleep system
    "SleepSystem": "core.sleep",
    "SleepController": "core.sleep",
    "SleepStage": "core.sleep",
    "MemoryReplayer": "core.sleep",
    "create_sleep_system": "core.sleep",
    # Neuromodulation
    "NeuromodulationIntegration": "core.neuromodulation_integration",
    "RewardModulation": "core.neuromodulation_integration",
    "TemporalDiscount": "core.neuromodulation_integration",
    "AttentionModulator": "core.neuromodulation_integration",
    "create_neuromodulation_integration": "core.neuromodulation_integration",
    # Motor coordination
    "CerebelloSpinalCoordination": "core.cerebello_spinal",
    "SpinalCord": "core.cerebello_spinal",
    "CentralPatternGenerator": "core.cerebello_spinal",
    "ReflexPathway": "core.cerebello_spinal",
    "Cerebellum": "core.cerebello_spinal",
    "create_cerebello_spinal": "core.cerebello_spinal",
    # Basal ganglia
    "BasalGangliaSystem": "core.basal_ganglia",
    "BasalGanglia": "core.basal_ganglia",
    "create_basal_ganglia": "core.basal_ganglia",
    # Memory
    "Hippocampus": "core.hippocampus",
    "create_hippocampus": "core.hippocampus",
    # Limbic system
    "LimbicSystem": "core.limbic",
    "Amygdala": "core.limbic",
    "Thalamus": "core.limbic",
    "create_limbic_system": "core.limbic",
    # Neurotransmitters
    "NeurotransmitterSystem": "core.neurotransmitter",
    "DopamineSystem": "core.neurotransmitter",
    "SerotoninSystem": "core.neurotransmitter",
    "AcetylcholineSystem": "core.neurotransmitter",
    "create_neurotransmitter_system": "core.neurotransmitter",
    # Prefrontal cortex
    "PrefrontalCortex": "core.prefrontal_cortex",
    "MaturationTracker": "core.prefrontal_cortex",
    "CostBenefitAnalyzer": "core.prefrontal_cortex",
    "ImpulseController": "core.prefrontal_cortex",
    "LongTermPlanner": "core.prefrontal_cortex",
    "WorkingMemory": "core.prefrontal_cortex",
    "CandidateEval": "core.prefrontal_cortex",
    "GoalNode": "core.prefrontal_cortex",
    "PlanStep": "core.prefrontal_cortex",
    # Angular gyrus
    "AngularGyrus": "core.angular_gyrus",
    "ModalityProjector": "core.angular_gyrus",
    "TranslationMatrix": "core.angular_gyrus",
    "SemanticInterlingua": "core.angular_gyrus",
    "CrossModalPredictor": "core.angular_gyrus",
    "TemporalBindingBuffer": "core.angular_gyrus",
    "SceneDetector": "core.angular_gyrus",
    # Advanced emotion
    "IntegratedAdvancedEmotionSystem": "core.advanced_emotion_integration",
    "AdvancedEmotionState": "core.advanced_emotion_integration",
    "create_advanced_emotion_system": "core.advanced_emotion_integration",
    # Emotion regulation
    "EmotionRegulationSystem": "core.emotion_regulation",
    "PrefrontalRegulation": "core.emotion_regulation",
    "MetabolicRegulation": "core.emotion_regulation",
    "SocialRegulation": "core.emotion_regulation",
    # Mood system
    "MoodSystem": "core.mood_system",
    "MoodState": "core.mood_system",
    "EmotionalMemoryConsolidation": "core.emotion_memory_consolidation",
    "SleepStageData": "core.emotion_memory_consolidation",
    "SocialEmotionSystem": "core.social_emotions",
    "EmotionalContagionSystem": "core.emotional_contagion",
    # Interoception
    "InteroceptionSystem": "core.interoception",
    "InteroceptiveState": "core.interoception",
    "GutState": "core.interoception",
    "EmotionDynamicsSystem": "core.emotion_dynamics",
    # Neural pruning
    "NeuralPruningSystem": "core.neural_pruning",
    "PruningConfig": "core.neural_pruning",
    "NeuronState": "core.neural_pruning",
    "ActivityTracker": "core.neural_pruning",
    "create_neural_pruning_system": "core.neural_pruning",
    # Autonomic nervous system
    "AutonomicNervousSystem": "core.autonomic_nervous_system",
    "ANSState": "core.autonomic_nervous_system",
    "SympatheticBranch": "core.autonomic_nervous_system",
    "ParasympatheticBranch": "core.autonomic_nervous_system",
    "BaroreceptorReflex": "core.autonomic_nervous_system",
    "PolyvagalSystem": "core.autonomic_nervous_system",
    "create_autonomic_nervous_system": "core.autonomic_nervous_system",
    # HPA axis
    "HPAAxis": "core.hpa_axis",
    "HPAState": "core.hpa_axis",
    "HypothalamicCRH": "core.hpa_axis",
    "PituitaryACTH": "core.hpa_axis",
    "AdrenalCortex": "core.hpa_axis",
    "NegativeFeedbackLoop": "core.hpa_axis",
    "AllostaticLoadTracker": "core.hpa_axis",
    "create_hpa_axis": "core.hpa_axis",
    # Glial system
    "GlialSystem": "core.glial_system",
    "GlialState": "core.glial_system",
    "AstrocyteSystem": "core.glial_system",
    "MicrogliaSystem": "core.glial_system",
    "OligodendrocyteSystem": "core.glial_system",
    "create_glial_system": "core.glial_system",
    # Allostatic regulation
    "AllostaticRegulation": "core.allostatic_regulation",
    "AllostaticState": "core.allostatic_regulation",
    "PredictiveRegulator": "core.allostatic_regulation",
    "LoadAccumulator": "core.allostatic_regulation",
    "RegimeSelector": "core.allostatic_regulation",
    "create_allostatic_regulation": "core.allostatic_regulation",
    # Predictive coding
    "PredictiveCodingSystem": "core.predictive_coding",
    "GenerativeLayer": "core.predictive_coding",
    "HierarchicalGenerativeModel": "core.predictive_coding",
    "PrecisionModulator": "core.predictive_coding",
    "ActiveInferenceController": "core.predictive_coding",
    "create_predictive_coding_system": "core.predictive_coding",
    # Social cognition
    "SocialCognitionSystem": "core.social_cognition",
    "SocialCognitionState": "core.social_cognition",
    "MirrorNeuronSystem": "core.social_cognition",
    "MirrorState": "core.social_cognition",
    "TheoryOfMind": "core.social_cognition",
    "ToMState": "core.social_cognition",
    "EmpathyCircuit": "core.social_cognition",
    "EmpathyState": "core.social_cognition",
    "ImitationLearning": "core.social_cognition",
    "ImitationState": "core.social_cognition",
    "SocialPredictor": "core.social_cognition",
    "create_social_cognition": "core.social_cognition",
    # Brainstem
    "Brainstem": "core.brainstem",
    "BrainstemState": "core.brainstem",
    "RespiratoryPhase": "core.brainstem",
    "ArousalLevel": "core.brainstem",
    "DefensiveBehavior": "core.brainstem",
    "RespiratoryRhythmGenerator": "core.brainstem",
    "ReticularActivatingSystem": "core.brainstem",
    "PeriaqueductalGray": "core.brainstem",
    "MedullaryCardiovascularCenter": "core.brainstem",
    "create_brainstem": "core.brainstem",
    # Vocalization
    "VocalCortex": "core.vocalization",
    "VocalTract": "core.vocalization",
    "ArticulatoryPlanner": "core.vocalization",
    "SpeechProductionPipeline": "core.vocalization",
    "FormantSynthesizer": "core.vocalization",
    "VocalizationOutput": "core.vocalization",
    "PHONEMES": "core.vocalization",
    "PHONEME_TO_IDX": "core.vocalization",
    "N_PHONEMES": "core.vocalization",
    "text_to_phoneme_indices": "core.vocalization",
    "describe_phoneme": "core.vocalization",
    "FormantToWaveform": "core.formant_synthesis",
    "create_formant_synthesizer": "core.formant_synthesis",
    # Psychotherapy
    "PsychotherapySystem": "core.psychotherapy",
    "TherapyModality": "core.psychotherapy",
    "SessionFrequency": "core.psychotherapy",
    "TherapyPhase": "core.psychotherapy",
    "TherapySession": "core.psychotherapy",
    "TherapyProgress": "core.psychotherapy",
    "conduct_therapy_session": "core.psychotherapy",
    "compute_resistance": "core.psychotherapy",
    "compute_compliance": "core.psychotherapy",
    "THERAPY_TARGETS": "core.psychotherapy",
    "create_psychotherapy_system": "core.psychotherapy",
    # Pharmacotherapy synergy
    "SynergyType": "core.pharmacotherapy_synergy",
    "SynergyRecord": "core.pharmacotherapy_synergy",
    "SynergyCalculator": "core.pharmacotherapy_synergy",
    "SYNERGY_MATRIX": "core.pharmacotherapy_synergy",
    "create_synergy_calculator": "core.pharmacotherapy_synergy",
    # Psychopharmacology sandbox
    "PsychopharmacologySandbox": "core.psychopharmacology_sandbox",
    "ExperimentMode": "core.psychopharmacology_sandbox",
    "TreatmentArm": "core.psychopharmacology_sandbox",
    "ExperimentDesign": "core.psychopharmacology_sandbox",
    "ArmResult": "core.psychopharmacology_sandbox",
    "ExperimentResult": "core.psychopharmacology_sandbox",
    "create_sandbox": "core.psychopharmacology_sandbox",
    # State mapping
    "UnifiedStateMapping": "core.state_key_mapping",
    "PDTarget": "core.pd_target_mapping",
    "build_pd_targets": "core.pd_target_mapping",
    "compute_pd_deltas": "core.pd_target_mapping",
    # Therapeutic experiment
    "DrugConfig": "core.therapeutic_experiment",
    "TherapyConfig": "core.therapeutic_experiment",
    "ExperimentConfig": "core.therapeutic_experiment",
    "TreatmentTimepoint": "core.therapeutic_experiment",
    "TherapeuticResult": "core.therapeutic_experiment",
    "TherapeuticExperiment": "core.therapeutic_experiment",
    # Psychometrics
    "PsychometricSnapshot": "core.psychometric_indicators",
    "PsychometricIndicatorTracker": "core.psychometric_indicators",
    # LLM evaluation
    "LLMEvaluationResult": "core.llm_evaluator",
    "LLMEvaluator": "core.llm_evaluator",
    # Therapeutic report
    "generate_markdown_report": "core.therapeutic_report",
    # Drug-drug interaction
    "DDIPairRecord": "core.drug_pipeline.ddi",
    "DDIResult": "core.drug_pipeline.ddi",
    "assess_ddi": "core.drug_pipeline.ddi",
    "combine_pd_deltas": "core.drug_pipeline.ddi",
    "compute_step_ke_modifiers": "core.drug_pipeline.ddi",
    # Receptor PD
    "ReceptorSubtype": "core.drug_pipeline.receptor_pd",
    "ReceptorPDTarget": "core.drug_pipeline.receptor_pd",
    "RECEPTOR_REGISTRY": "core.drug_pipeline.receptor_pd",
    "DRUG_RECEPTOR_AFFINITY": "core.drug_pipeline.receptor_pd",
    "build_receptor_pd_targets": "core.drug_pipeline.receptor_pd",
    "compute_receptor_deltas": "core.drug_pipeline.receptor_pd",
    "aggregate_receptor_to_nt": "core.drug_pipeline.receptor_pd",
    # Addiction dynamics
    "ToleranceState": "core.addiction_dynamics",
    "WithdrawalState": "core.addiction_dynamics",
    "CravingState": "core.addiction_dynamics",
    "AddictionProfile": "core.addiction_dynamics",
    "AddictionDynamicsEngine": "core.addiction_dynamics",
    # Pathogen neuroinflammation
    "PathogenProfile": "core.pathogen_neuroinflammation",
    "PathogenState": "core.pathogen_neuroinflammation",
    "PathogenTriggeredInflammationEngine": "core.pathogen_neuroinflammation",
    "PATHOGEN_REGISTRY": "core.pathogen_neuroinflammation",
    # Symptom tracking
    "SymptomEpisode": "core.symptom_tracker",
    "PersistentSymptom": "core.symptom_tracker",
    "SymptomSnapshot": "core.symptom_tracker",
    "SymptomTracker": "core.symptom_tracker",
    # Hardware vitals
    "HardwareState": "core.hardware_vitals",
    "HardwareVitals": "core.hardware_vitals",
    "create_hardware_vitals": "core.hardware_vitals",
    # Self-awareness
    "SelfAwarenessCenter": "core.self_awareness",
    "mPFCState": "core.self_awareness",
    "PCCState": "core.self_awareness",
    "PrecuneusState": "core.self_awareness",
    "DMNState": "core.self_awareness",
    "SelfOtherState": "core.self_awareness",
    "MetaSelfState": "core.self_awareness",
    "SelfAwarenessState": "core.self_awareness",
    "MedialPrefrontalCortex": "core.self_awareness",
    "PosteriorCingulateCortex": "core.self_awareness",
    "PrecuneusSystem": "core.self_awareness",
    "DefaultModeNetwork": "core.self_awareness",
    "SelfOtherDistinction": "core.self_awareness",
    "MetaSelfAwareness": "core.self_awareness",
    "create_self_awareness_center": "core.self_awareness",
    # Emotion state manager
    "EmotionStateManager": "core.emotion_state_manager",
    "EmotionSnapshot": "core.emotion_state_manager",
    "EmotionHistoryEntry": "core.emotion_state_manager",
    "create_emotion_state_manager": "core.emotion_state_manager",
    # Animation mapper
    "VTuberPlatform": "core.animation_mapper",
    "EmotionLabel": "core.animation_mapper",
    "VADState": "core.animation_mapper",
    "AnimationParameter": "core.animation_mapper",
    "AnimationState": "core.animation_mapper",
    "AnimationMapper": "core.animation_mapper",
    "SmoothInterpolator": "core.animation_mapper",
    "create_animation_mapper": "core.animation_mapper",
    "vad_to_animation": "core.animation_mapper",
    "vad_to_emotion_label": "core.animation_mapper",
    "LIVE2D_PARAMS": "core.animation_mapper",
    "VRM_BLENDSHAPES": "core.animation_mapper",
}

# Special case: ADVANCED_EMOTION_AVAILABLE is a boolean constant, not a class/function
_ADVANCED_EMOTION_AVAILABLE = None  # Will be computed lazily


def _get_advanced_emotion_available():
    """Check if advanced emotion integration is available."""
    global _ADVANCED_EMOTION_AVAILABLE
    if _ADVANCED_EMOTION_AVAILABLE is None:
        try:
            from core.advanced_emotion_integration import MODULES_AVAILABLE
            _ADVANCED_EMOTION_AVAILABLE = MODULES_AVAILABLE
        except ImportError:
            _ADVANCED_EMOTION_AVAILABLE = False
    return _ADVANCED_EMOTION_AVAILABLE


def __getattr__(name):
    """Lazy-import on first access."""
    if name == "ADVANCED_EMOTION_AVAILABLE":
        return _get_advanced_emotion_available()
    if name in _SUBMODULES:
        import importlib
        module = importlib.import_module(_SUBMODULES[name])
        return getattr(module, name)
    raise AttributeError(f"module 'core' has no attribute {name!r}")


def __dir__():
    """Return all public names for tab completion."""
    return list(_SUBMODULES.keys()) + ["ADVANCED_EMOTION_AVAILABLE"]


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
    "EmotionStateManager", "EmotionSnapshot", "EmotionHistoryEntry", "create_emotion_state_manager",
    # Animation mapper
    "VTuberPlatform", "EmotionLabel", "VADState", "AnimationParameter", "AnimationState",
    "AnimationMapper", "SmoothInterpolator", "create_animation_mapper", "vad_to_animation",
    "vad_to_emotion_label", "LIVE2D_PARAMS", "VRM_BLENDSHAPES",
]