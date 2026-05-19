"""事件类型定义

所有事件类型常量，按生命周期分组。
"""

# ===== 生命周期事件 =====
STEP_START = "step_start"             # 每步开始
STEP_END = "step_end"                 # 每步结束

# ===== 热力学事件 =====
THERMO_STATE = "thermo_state"         # 系统状态更新
HIBERNATE_ENTER = "hibernate_enter"   # 进入休眠
SYSTEM_DEAD = "system_dead"           # 数字死亡
COMPRESSION_NEEDED = "compression_needed"  # 需要压缩

# ===== 探索事件 =====
GOAL_NEEDED = "goal_needed"           # 需要新目标
GOAL_SELECTED = "goal_selected"       # 目标已选定
EXPLORATION_START = "exploration_start"   # 开始探索
EXPLORATION_DONE = "exploration_done"     # 探索完成
INFO_GAIN_COMPUTED = "info_gain_computed" # 信息增益计算完毕

# ===== 学习事件 =====
MEMORY_ADDED = "memory_added"         # 新记忆添加
DISSONANCE_DETECTED = "dissonance_detected"  # 认知失调检测
ALIGNMENT_CHECK = "alignment_check"   # 自对齐检查

# ===== 人格事件 =====
PERSONALITY_UPDATE = "personality_update"  # 人格系统更新

# ===== 情绪事件 =====
EMOTION_PROCESS = "emotion_process"   # 情绪处理请求
EMOTION_UPDATED = "emotion_updated"   # 情绪更新完毕

# ===== 压缩事件 =====
COMPRESSION_DONE = "compression_done"  # 压缩完成

# ===== 神经调节事件 =====
NEURAL_REGULATION = "neural_regulation"  # 神经自调节 (ANS, HPA, Glial, Allostatic)
BRAIN_UPDATE = "brain_update"            # 脑区更新 (Social, SelfAwareness, NT, Neuroplasticity, Hormones, Brainstem)
SENSORY_PROCESS = "sensory_process"      # 感觉处理 (Limbic, LanguageCortex, AngularGyrus)
MOTOR_CONTROL = "motor_control"          # 运动控制 (BG, PFC, Cerebellum)
VOCALIZATION_CONTROL = "vocalization_control"  # 发音控制 (VocalCortex, ArticulatoryPlanner)
VOCALIZATION_OUTPUT = "vocalization_output"    # 发音输出 (声学特征, 共振峰)
MEMORY_ENCODE = "memory_encode"          # 记忆编码 (Hippocampus)
PRUNING_UPDATE = "pruning_update"        # 神经修剪 (NeuralPruning)
OUTPUT_FILTER = "output_filter"           # Post-LLM 输出过滤 (PredictiveCoding, SelfAwareness)
LEARNING_TRIGGER = "learning_trigger"     # 对话驱动的学习触发 (Curiosity, Neuroplasticity)

# ===== 微表情感知事件 (Censor集成) =====
MICRO_EXPRESSION_PROCESS = "micro_expression_process"    # 微表情处理请求 (Censor)
MICRO_EXPRESSION_DETECTED = "micro_expression_detected"  # 微表情检测完成 (AU, ME, Emotion)

# ===== 精神疾病模拟事件 =====
PSYCHIATRIC_CONDITION_CHANGE = "psychiatric_condition_change"  # 精神疾病条件变化

# ===== 心理治疗事件 =====
THERAPY_SESSION_START = "therapy_session_start"    # 治疗session开始
THERAPY_SESSION_END = "therapy_session_end"        # 治疗session结束
THERAPY_PROGRESS_UPDATE = "therapy_progress_update"  # 治疗进展更新

# ===== 协同/实验事件 =====
SYNERGY_CALCULATED = "synergy_calculated"          # 协同因子计算完毕
EXPERIMENT_START = "experiment_start"              # 沙盒实验开始
EXPERIMENT_END = "experiment_end"                  # 沙盒实验结束

# ===== 治疗实验事件 =====
THERAPEUTIC_EFFECT_APPLIED = "therapeutic_effect_applied"    # PK/PD效应应用到状态
THERAPEUTIC_TIMEPOINT_REACHED = "therapeutic_timepoint_reached"  # 观测时间点到达
LLM_EVALUATION_COMPLETED = "llm_evaluation_completed"        # LLM评估完成


# 所有事件类型列表
ALL_EVENTS = [
    STEP_START, STEP_END,
    THERMO_STATE, HIBERNATE_ENTER, SYSTEM_DEAD, COMPRESSION_NEEDED,
    GOAL_NEEDED, GOAL_SELECTED, EXPLORATION_START, EXPLORATION_DONE, INFO_GAIN_COMPUTED,
    MEMORY_ADDED, DISSONANCE_DETECTED, ALIGNMENT_CHECK,
    PERSONALITY_UPDATE,
    EMOTION_PROCESS, EMOTION_UPDATED,
    COMPRESSION_DONE,
    NEURAL_REGULATION, BRAIN_UPDATE, SENSORY_PROCESS,
    MOTOR_CONTROL, VOCALIZATION_CONTROL, VOCALIZATION_OUTPUT,
    MEMORY_ENCODE, PRUNING_UPDATE,
    OUTPUT_FILTER, LEARNING_TRIGGER,
    MICRO_EXPRESSION_PROCESS, MICRO_EXPRESSION_DETECTED,
    PSYCHIATRIC_CONDITION_CHANGE,
    THERAPY_SESSION_START, THERAPY_SESSION_END, THERAPY_PROGRESS_UPDATE,
    SYNERGY_CALCULATED, EXPERIMENT_START, EXPERIMENT_END,
    THERAPEUTIC_EFFECT_APPLIED, THERAPEUTIC_TIMEPOINT_REACHED, LLM_EVALUATION_COMPLETED,
]
