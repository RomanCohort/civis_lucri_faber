"""
社会认知与镜像神经元系统 (Social Cognition & Mirror Neuron System)

实现人类"共情机制"的完整神经模型：
1. 镜像神经元系统 (Mirror Neuron System) - 观察-执行匹配、动作共振
2. 心理理论 (Theory of Mind) - 推断他人信念、意图、视角
3. 共情回路 (Empathy Circuit) - 情感共情 + 认知共情 + 同情关怀
4. 模仿学习 (Imitation Learning) - 观察学习、技能获取
5. 社会预测 (Social Prediction) - 行为建模、交互结果预测

核心原理:
- 镜像神经元: 看到别人做动作时，自己脑中对应的运动区域也会激活
  (Rizzolatti et al., 1996 - 猕猴F5区镜像神经元发现)
- 打哈欠传染: 镜像共振的外在表现
- 看到别人受伤自己也会觉得疼: 情感共情 (anterior insula + ACC)
- 理解他人意图: 心理理论 (TPJ + mPFC)

生物参考文献:
- Rizzolatti & Craighero (2004): 镜像神经元系统综述
- Gallese (2001): 具身模拟理论
- Premack & Woodruff (1978): 心理理论
- Singer et al. (2004): 共情的神经基础
- de Waal (2008): 共情的进化层次模型
- Heyes (2010): 镜像神经元的学习起源 (ASL模型)
- Dunbar (1998): 社会脑假说
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============ 状态定义 ============

@dataclass
class MirrorState:
    """镜像神经元状态"""
    resonance_level: float = 0.0         # 镜像共振强度 [0,1]
    matched_action: str | None = None # 匹配到的动作
    motor_simulation: float = 0.0        # 运动模拟激活 [0,1]
    contagion_susceptibility: float = 0.5 # 传染易感性 [0,1]
    yawning_trigger: float = 0.0         # 打哈欠触发 [0,1]
    pain_resonance: float = 0.0          # 疼痛共振 [0,1]


@dataclass
class ToMState:
    """心理理论状态"""
    inferred_belief: float = 0.5          # 推断的他人信念 [0,1]
    inferred_intent: float = 0.5          # 推断的他人意图 [0,1]
    perspective_distance: float = 0.5     # 视角差异度 [0,1]
    mental_state_confidence: float = 0.3  # 心理状态推断置信度 [0,1]
    predicted_action: str | None = None # 预测的他人行动


@dataclass
class EmpathyState:
    """共情状态"""
    affective_empathy: float = 0.0        # 情感共情 (感受他人感受) [0,1]
    cognitive_empathy: float = 0.0        # 认知共情 (理解他人想法) [0,1]
    compassion: float = 0.0               # 同情关怀 (帮助动机) [0,1]
    personal_distress: float = 0.0        # 个人痛苦 (共情过载) [0,1]
    empathy_regulation: float = 0.8       # 共情调节能力 [0,1]


@dataclass
class ImitationState:
    """模仿学习状态"""
    observed_action: str | None = None
    copy_accuracy: float = 0.0            # 模仿精度 [0,1]
    learning_progress: float = 0.0        # 学习进度 [0,1]
    skill_acquired: dict[str, float] = field(default_factory=dict)


@dataclass
class SocialCognitionState:
    """社会认知总状态"""
    mirror: MirrorState = field(default_factory=MirrorState)
    tom: ToMState = field(default_factory=ToMState)
    empathy: EmpathyState = field(default_factory=EmpathyState)
    imitation: ImitationState = field(default_factory=ImitationState)
    social_prediction_accuracy: float = 0.5  # 社会预测准确度 [0,1]
    overall_social_capacity: float = 0.7     # 总体社会能力 [0,1]


# ============ 镜像神经元系统 ============

class MirrorNeuronSystem(nn.Module):
    """
    镜像神经元系统 (Mirror Neuron System)

    核心机制: 观察-执行匹配
    - 看到别人做动作 → 自己脑中对应的运动区域也激活
    - 看到/听到别人打哈欠 → 自己也想打哈欠
    - 看到别人疼痛 → 自己的疼痛区域也激活 (前脑岛 + ACC)

    神经基础:
    - F5区 (猕猴) / Broca区 (人类): 动作观察-执行匹配
    - 前脑岛 (Anterior Insula): 情感共振 (厌恶、疼痛)
    - 前扣带回 (ACC): 疼痛共振
    - 顶下小叶 (IPL): 动作理解

    参考:
    - Rizzolatti et al. (1996): 猕猴F5区镜像神经元
    - Rizzolatti & Craighero (2004): 镜像神经元系统综述
    - Gallese (2001): 具身模拟理论
    """

    def __init__(self, action_dim: int = 16, hidden_dim: int = 64,
                 n_known_actions: int = 8):
        super().__init__()

        # 动作观察网络 (IPL + STS)
        self.observation_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 动作-执行匹配网络 (F5/Broca)
        self.matching_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_known_actions),
            nn.Softmax(dim=-1),
        )

        # 运动模拟网络 (运动皮层)
        self.motor_simulation_net = nn.Sequential(
            nn.Linear(n_known_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # 共振强度调节 (个体差异)
        self.resonance_baseline = nn.Parameter(torch.tensor(0.5))

        # 打哈欠传染的特定模型
        self.yawn_susceptibility = nn.Parameter(torch.tensor(0.4))

        # 疼痛共振网络 (前脑岛 + ACC)
        self.pain_resonance_net = nn.Sequential(
            nn.Linear(2, 16),  # [他人疼痛强度, 亲密度]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.state = MirrorState()
        self.action_labels = [f"action_{i}" for i in range(n_known_actions)]

    def forward(self, observed_action: torch.Tensor,
                self_action_repertoire: torch.Tensor | None = None,
                pain_observed: float = 0.0,
                proximity: float = 0.5) -> dict[str, Any]:
        """
        镜像神经元激活

        Args:
            observed_action: 观察到的动作表征
            self_action_repertoire: 自身动作库 (用于匹配)
            pain_observed: 观察到的他人疼痛强度 [0,1]
            proximity: 与观察对象的关系亲密度 [0,1]
        """
        # 1. 动作观察与编码
        observed_features = self.observation_net(observed_action)

        # 2. 动作匹配 (观察到的动作 vs 已知动作库)
        action_probs = self.matching_net(observed_features)
        best_action_idx = action_probs.argmax(dim=-1)
        match_confidence = action_probs.max(dim=-1)[0]

        # 3. 运动模拟 (匹配到的动作激活对应运动模式)
        motor_output = self.motor_simulation_net(action_probs)

        # 4. 计算镜像共振强度
        # 共振 = 基线 × 匹配置信度 × 亲密度调节
        resonance = float(
            torch.sigmoid(self.resonance_baseline).detach() * match_confidence.mean().detach()
        )
        resonance *= (0.5 + 0.5 * proximity)  # 亲密度调节
        resonance = float(np.clip(resonance, 0.0, 1.0))

        # 5. 打哈欠传染模型
        # 打哈欠是镜像共振的典型外在表现
        # 易感性受亲密度、疲劳度、共振基线影响
        yawn_trigger = self._compute_yawn_contagion(resonance, proximity)

        # 6. 疼痛共振 (Singer et al., 2004)
        # 看到亲人疼痛 → 自己的痛觉区域也激活
        pain_input = torch.tensor([[pain_observed, proximity]], dtype=torch.float32)
        pain_res = float(self.pain_resonance_net(pain_input).squeeze().detach())

        # 7. 传染易感性 (高共振 = 高易感)
        susceptibility = float(np.clip(resonance * 1.2, 0.0, 1.0))

        # 更新状态
        self.state = MirrorState(
            resonance_level=resonance,
            matched_action=self.action_labels[best_action_idx.item()]
            if best_action_idx.numel() > 0 else None,
            motor_simulation=float(motor_output.abs().mean().detach()),
            contagion_susceptibility=susceptibility,
            yawning_trigger=yawn_trigger,
            pain_resonance=pain_res,
        )

        return {
            'resonance_level': resonance,
            'matched_action': self.state.matched_action,
            'match_confidence': float(match_confidence.mean()),
            'motor_simulation': self.state.motor_simulation,
            'contagion_susceptibility': susceptibility,
            'yawning_trigger': yawn_trigger,
            'pain_resonance': pain_res,
        }

    def _compute_yawn_contagion(self, resonance: float,
                                 proximity: float) -> float:
        """
        打哈欠传染模型

        受以下因素影响:
        - 镜像共振基线 (个体差异)
        - 关系亲密度 (亲人/朋友 > 陌生人)
        - 共情能力

        参考: Platek et al. (2003) - 打哈欠传染与自我意识/共情的关系
        """
        base_susceptibility = float(torch.sigmoid(self.yawn_susceptibility).detach())
        # 亲密度调制: 越亲密越容易传染
        modulated = base_susceptibility * (0.3 + 0.7 * proximity)
        # 共振放大
        trigger = modulated * resonance * 1.5
        return float(np.clip(trigger, 0.0, 1.0))


# ============ 心理理论 (Theory of Mind) ============

class TheoryOfMind(nn.Module):
    """
    心理理论 (Theory of Mind / ToM)

    推断他人的心理状态:
    - 信念追踪: 他人知道什么? (False belief task)
    - 意图推断: 他人想要做什么?
    - 视角采择: 从他人的角度看问题

    神经基础:
    - 颞顶联合区 (TPJ): 视角采择、信念推理
    - 内侧前额叶 (mPFC): 心理状态推理
    - 楔前叶 (Precuneus): 自传体记忆、自我-他人区分
    - STS: 意图从生物运动中推断

    参考:
    - Premack & Woodruff (1978): 心理理论原始论文
    - Saxe & Kanwisher (2003): TPJ在ToM中的特异性
    - Wimmer & Perner (1983): 错误信念任务
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 64):
        super().__init__()

        # 信念推理网络 (mPFC)
        self.belief_net = nn.Sequential(
            nn.Linear(state_dim + state_dim, hidden_dim),  # [自身状态 || 他人行为]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 信念概率 [0,1]
        )

        # 意图推断网络 (TPJ + STS)
        self.intent_net = nn.Sequential(
            nn.Linear(state_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3种意图: 合作/竞争/中立
            nn.Softmax(dim=-1),
        )

        # 视角采择网络 (TPJ + Precuneus)
        self.perspective_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 视角距离 [0,1], 0=完全一致
        )

        # 置信度评估 (元认知)
        self.confidence_net = nn.Sequential(
            nn.Linear(3, 16),  # [信念, 意图确定性, 视角距离]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # 他人心理状态的内部模型
        self.other_model = nn.GRUCell(state_dim, hidden_dim)
        self.other_state = nn.Parameter(torch.zeros(hidden_dim), requires_grad=False)

        self.state = ToMState()
        self.intent_labels = ["cooperative", "competitive", "neutral"]

    def forward(self, self_state: torch.Tensor,
                other_behavior: torch.Tensor,
                context: torch.Tensor | None = None) -> dict[str, Any]:
        """
        心理理论推理

        Args:
            self_state: 自身状态表征
            other_behavior: 观察到的他人行为
            context: 社交上下文 (可选)
        """
        # 1. 更新他人心理模型 (GRU持续追踪)
        if other_behavior.dim() == 1:
            other_behavior = other_behavior.unsqueeze(0)
        if self_state.dim() == 1:
            self_state = self_state.unsqueeze(0)
        self.other_state.data = self.other_model(
            other_behavior, self.other_state.unsqueeze(0)
        ).squeeze(0)

        # 2. 信念推理
        combined = torch.cat([self_state, other_behavior], dim=-1)
        if combined.dim() == 1:
            combined = combined.unsqueeze(0)
        belief = float(self.belief_net(combined).squeeze())

        # 3. 意图推断
        intent_probs = self.intent_net(combined)
        intent_idx = intent_probs.argmax(dim=-1)
        intent_confidence = float(intent_probs.max(dim=-1)[0].squeeze())

        # 4. 视角采择 (推断自己与他人视角的差异)
        perspective_dist = float(self.perspective_net(self_state).squeeze())

        # 5. 综合置信度
        conf_input = torch.tensor([[
            belief, intent_confidence, perspective_dist
        ]], dtype=torch.float32)
        confidence = float(self.confidence_net(conf_input).squeeze())

        # 6. 预测他人下一步行动
        predicted_intent = self.intent_labels[intent_idx.squeeze().item()]

        # 更新状态
        self.state = ToMState(
            inferred_belief=belief,
            inferred_intent=intent_confidence,
            perspective_distance=perspective_dist,
            mental_state_confidence=confidence,
            predicted_action=predicted_intent,
        )

        return {
            'inferred_belief': belief,
            'inferred_intent': predicted_intent,
            'intent_confidence': intent_confidence,
            'perspective_distance': perspective_dist,
            'mental_state_confidence': confidence,
            'predicted_action': predicted_intent,
        }


# ============ 共情回路 ============

class EmpathyCircuit(nn.Module):
    """
    共情回路 (Empathy Circuit)

    de Waal (2008) 共情的进化层次模型:
    1. 情感传染 (Emotional Contagion) - 最底层, 自动化的
    2. 共情关切 (Sympathetic Concern) - 关心他人福祉
    3. 视角采择共情 (Empathic Perspective-Taking) - 认知层面

    神经基础:
    - 前脑岛 (Anterior Insula): 情感共情
    - 前扣带回 (ACC): 情感共鸣 + 痛觉共情
    - TPJ + mPFC: 认知共情 / 视角采择
    - Oxytocin系统: 信任和亲社会行为

    参考:
    - Singer et al. (2004): 共情的神经相关性
    - de Waal (2008): 共情的进化
    - Decety & Jackson (2004): 共情的功能架构
    """

    def __init__(self, emotion_dim: int = 8, hidden_dim: int = 64):
        super().__init__()

        # 情感共情网络 (前脑岛 + ACC)
        # 感受他人的情感 = 自身产生类似的情感反应
        self.affective_net = nn.Sequential(
            nn.Linear(emotion_dim + 2, hidden_dim),  # [他人情绪 || 亲密度 || 相似度]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, emotion_dim),  # 产生的共情情绪
            nn.Sigmoid(),
        )

        # 认知共情网络 (TPJ + mPFC)
        # 理解他人的想法 (不一定要感受到)
        self.cognitive_net = nn.Sequential(
            nn.Linear(emotion_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 认知共情程度 [0,1]
        )

        # 同情关怀网络 (腹侧纹状体 + Oxytocin)
        # 产生帮助他人的动机
        self.compassion_net = nn.Sequential(
            nn.Linear(emotion_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # 同情关怀强度 [0,1]
        )

        # 个人痛苦网络 (杏仁核)
        # 过度共情导致自身痛苦 (共情过载)
        self.distress_net = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # 个人痛苦程度 [0,1]
        )

        # 共情调节网络 (PFC top-down)
        # 调节共情强度, 防止过载
        self.regulation_net = nn.Sequential(
            nn.Linear(3, 16),  # [情感共情, 认知共情, 个人痛苦]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # 调节能力 [0,1]
        )

        # 催产素水平 (促进亲社会行为)
        self.oxytocin_level = nn.Parameter(torch.tensor(0.5))

        self.state = EmpathyState()

    def forward(self, other_emotion: torch.Tensor,
                proximity: float = 0.5,
                similarity: float = 0.5) -> dict[str, Any]:
        """
        共情回路激活

        Args:
            other_emotion: 他人情绪状态 (8维)
            proximity: 关系亲密度 [0,1]
            similarity: 自我-他人相似度 [0,1]
        """
        if other_emotion.dim() == 1:
            other_emotion = other_emotion.unsqueeze(0)

        context = torch.tensor([[proximity, similarity]], dtype=torch.float32)
        context = context.expand(other_emotion.shape[0], -1)

        # 1. 情感共情: 感受他人的感受
        affect_input = torch.cat([other_emotion, context], dim=-1)
        affective_response = self.affective_net(affect_input)
        affective_strength = float(affective_response.mean())

        # 催产素促进情感共情 (尤其在亲密关系中)
        oxytocin = float(torch.sigmoid(self.oxytocin_level))
        affective_strength *= (0.7 + 0.3 * oxytocin)

        # 2. 认知共情: 理解他人的想法
        cognitive_strength = float(self.cognitive_net(affect_input).squeeze())

        # 3. 同情关怀: 产生帮助动机
        compassion = float(self.compassion_net(affect_input).squeeze())
        # 催产素增强同情关怀
        compassion *= (0.6 + 0.4 * oxytocin)

        # 4. 个人痛苦: 过度共情导致自身痛苦
        distress = float(self.distress_net(other_emotion).squeeze())
        # 个人痛苦在情感共情过强时加剧
        distress *= (1.0 + affective_strength * 0.5)
        distress = float(np.clip(distress, 0.0, 1.0))

        # 5. 共情调节: PFC下调过强的共情反应
        reg_input = torch.tensor([[affective_strength, cognitive_strength, distress]])
        regulation = float(self.regulation_net(reg_input).squeeze())

        # 调节后降低个人痛苦
        distress = distress * (1.0 - regulation * 0.5)

        # 更新状态
        self.state = EmpathyState(
            affective_empathy=float(np.clip(affective_strength, 0, 1)),
            cognitive_empathy=float(np.clip(cognitive_strength, 0, 1)),
            compassion=float(np.clip(compassion, 0, 1)),
            personal_distress=float(np.clip(distress, 0, 1)),
            empathy_regulation=regulation,
        )

        return {
            'affective_empathy': self.state.affective_empathy,
            'cognitive_empathy': self.state.cognitive_empathy,
            'compassion': self.state.compassion,
            'personal_distress': self.state.personal_distress,
            'empathy_regulation': regulation,
            'oxytocin_level': oxytocin,
            'empathic_emotion': affective_response.squeeze(0).detach(),
        }


# ============ 模仿学习系统 ============

class ImitationLearning(nn.Module):
    """
    模仿学习系统 (Imitation Learning)

    通过观察他人来学习:
    - 动作复制: 观察并模仿具体动作
    - 技能获取: 从多次观察中提炼技能
    - 社会学习: 基于观察的行为调整

    神经基础:
    - 镜像神经元: 动作观察 -> 运动模拟 -> 执行
    - 前额叶: 目标理解和策略提取
    - 小脑: 运动误差修正

    参考:
    - Heyes (2010): 镜像神经元的学习起源 (ASL模型)
    - Bandura (1977): 社会学习理论
    - Meltzoff & Moore (1977): 新生儿模仿
    """

    def __init__(self, action_dim: int = 16, hidden_dim: int = 64):
        super().__init__()

        # 观察编码网络
        self.observation_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 动作复制网络 (尝试产生相同的动作)
        self.copy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # 技能库 (已习得的技能)
        self.skill_memory = {}  # skill_name -> weight tensor
        self.skill_acquisition_count = {}  # skill_name -> observation count

        # 误差修正网络 (小脑类比)
        self.error_correction = nn.Sequential(
            nn.Linear(action_dim * 2, hidden_dim // 2),  # [目标 || 实际]
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),
        )

        self.state = ImitationState()
        self.learning_rate = 0.01

    def forward(self, observed_action: torch.Tensor,
                skill_name: str | None = None) -> dict[str, Any]:
        """
        模仿学习

        Args:
            observed_action: 观察到的动作
            skill_name: 动作/技能名称
        """
        if observed_action.dim() == 1:
            observed_action = observed_action.unsqueeze(0)

        # 1. 编码观察到的动作
        encoded = self.observation_encoder(observed_action)

        # 2. 尝试复制
        copied_action = self.copy_net(encoded)

        # 3. 计算模仿精度
        copy_error = F.mse_loss(copied_action, observed_action)
        copy_accuracy = float(1.0 - torch.clamp(copy_error, 0.0, 1.0))

        # 4. 如果有技能名, 更新技能库
        if skill_name:
            if skill_name not in self.skill_memory:
                self.skill_memory[skill_name] = copied_action.detach().clone()
                self.skill_acquisition_count[skill_name] = 1
            else:
                # EMA更新: 逐渐逼近观察到的动作
                old = self.skill_memory[skill_name]
                self.skill_memory[skill_name] = (
                    0.8 * old + 0.2 * copied_action.detach()
                )
                self.skill_acquisition_count[skill_name] = (
                    self.skill_acquisition_count.get(skill_name, 0) + 1
                )

        # 5. 误差修正 (用观察到的动作作为目标)
        correction_input = torch.cat([observed_action, copied_action], dim=-1)
        corrected = copied_action + self.error_correction(correction_input) * 0.1

        # 6. 学习进度
        progress = float(np.clip(
            min(1.0, self.skill_acquisition_count.get(skill_name, 0) / 20.0)
            if skill_name else 0.0, 0.0, 1.0
        ))

        # 更新状态
        self.state = ImitationState(
            observed_action=skill_name,
            copy_accuracy=copy_accuracy,
            learning_progress=progress,
            skill_acquired={
                k: float(v.abs().mean()) for k, v in self.skill_memory.items()
            },
        )

        return {
            'copy_accuracy': copy_accuracy,
            'learning_progress': progress,
            'corrected_action': corrected.squeeze(0).detach(),
            'skills_known': len(self.skill_memory),
            'skill_acquired': skill_name in self.skill_memory if skill_name else False,
        }


# ============ 社会预测系统 ============

class SocialPredictor(nn.Module):
    """
    社会预测系统 (Social Prediction)

    建模他人的行为模式, 预测交互结果:
    - 行为模式识别
    - 交互结果预测
    - 社会规范内化

    参考:
    - Dunbar (1998): 社会脑假说
    - Frith & Frith (2006): 社会交互中的预测
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 64):
        super().__init__()

        # 他人行为建模 (LSTM)
        self.behavior_lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.behavior_hidden = None

        # 交互结果预测
        self.outcome_net = nn.Sequential(
            nn.Linear(hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 积极/消极/中立
            nn.Softmax(dim=-1),
        )

        # 预测准确度追踪
        self.prediction_history = deque(maxlen=100)
        self.accuracy_buffer = deque(maxlen=50)

        self.state = SocialCognitionState()

    def forward(self, other_behavior_sequence: torch.Tensor,
                self_planned_action: torch.Tensor) -> dict[str, Any]:
        """
        社会预测

        Args:
            other_behavior_sequence: 他人行为序列 [seq_len, state_dim]
            self_planned_action: 自身计划的行动 [state_dim]
        """
        if other_behavior_sequence.dim() == 2:
            other_behavior_sequence = other_behavior_sequence.unsqueeze(0)

        # 1. 建模他人行为模式
        lstm_out, self.behavior_hidden = self.behavior_lstm(
            other_behavior_sequence, self.behavior_hidden
        )
        other_representation = lstm_out[:, -1, :]  # 取最后时刻

        # 2. 预测交互结果
        if self_planned_action.dim() == 1:
            self_planned_action = self_planned_action.unsqueeze(0)
        combined = torch.cat([other_representation, self_planned_action], dim=-1)
        outcome_probs = self.outcome_net(combined)

        predicted_outcome = ["positive", "negative", "neutral"][
            outcome_probs.argmax(dim=-1).item()
        ]
        confidence = float(outcome_probs.max(dim=-1)[0])

        return {
            'predicted_outcome': predicted_outcome,
            'prediction_confidence': confidence,
            'outcome_distribution': outcome_probs.squeeze(0).detach().tolist(),
            'other_representation': other_representation.squeeze(0).detach(),
        }

    def update_accuracy(self, predicted: str, actual: str):
        """更新预测准确度"""
        correct = predicted == actual
        self.accuracy_buffer.append(1.0 if correct else 0.0)
        return float(np.mean(self.accuracy_buffer)) if self.accuracy_buffer else 0.5


# ============ 社会认知系统 (聚合器) ============

class SocialCognitionSystem(nn.Module):
    """
    社会认知与镜像神经元系统 - 聚合器

    整合镜像神经元、心理理论、共情回路、模仿学习和社会预测。

    层次结构 (de Waal, 2008):
    底层: 情感传染 (自动化的)
    中层: 共情关切 (同情)
    高层: 视角采择 (认知共情/ToM)

    参考:
    - Rizzolatti & Craighero (2004): 镜像神经元
    - de Waal (2008): 共情的进化层次
    - Premack & Woodruff (1978): 心理理论
    """

    def __init__(self, action_dim: int = 16, state_dim: int = 64,
                 emotion_dim: int = 8, hidden_dim: int = 64,
                 event_bus=None):
        super().__init__()

        self.mirror = MirrorNeuronSystem(
            action_dim=action_dim, hidden_dim=hidden_dim
        )
        self.tom = TheoryOfMind(
            state_dim=state_dim, hidden_dim=hidden_dim
        )
        self.empathy = EmpathyCircuit(
            emotion_dim=emotion_dim, hidden_dim=hidden_dim
        )
        self.imitation = ImitationLearning(
            action_dim=action_dim, hidden_dim=hidden_dim
        )
        self.social_predictor = SocialPredictor(
            state_dim=state_dim, hidden_dim=hidden_dim
        )

        # 社会能力整合网络
        self.capacity_net = nn.Sequential(
            nn.Linear(8, 32),  # [共振, ToM置信, 情感共情, 认知共情,
                               #  同情, 痛苦, 模仿精度, 预测置信]
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.state = SocialCognitionState()
        self.step_count = 0

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "brain_update",
                self._handle_brain_update,
                priority=0,
                name="social_cognition",
            )

    def _handle_brain_update(self, event) -> dict[str, Any]:
        """Event-driven handler for brain_update events."""
        import torch as _torch
        state = event.data.get("internal_state", {})
        pain_observed = state.get("pain_observed", 0.0)
        proximity = state.get("social_proximity", 0.5)
        state_tensor = event.data.get("state_tensor", _torch.randn(1, 64))

        result = self.step(
            observed_action=_torch.randn(1, 16),  # 无真实观测动作
            self_state=state_tensor,
            other_behavior=state_tensor,  # 用自身状态近似他人
            other_emotion=state_tensor[:, :8] if state_tensor.shape[-1] >= 8 else _torch.randn(1, 8),
            pain_observed=pain_observed,
            proximity=proximity,
            similarity=0.5,
        )

        state["mirror_resonance"] = result["mirror_resonance"]
        state["yawning_trigger"] = result["yawning_trigger"]
        state["pain_resonance"] = result["pain_resonance"]
        state["affective_empathy"] = result["affective_empathy"]
        state["cognitive_empathy"] = result["cognitive_empathy"]
        state["compassion"] = result["compassion"]
        state["social_capacity"] = result["overall_social_capacity"]
        state["inferred_intent"] = result["inferred_intent"]

        return result

    def step(self, observed_action: torch.Tensor,
             self_state: torch.Tensor,
             other_behavior: torch.Tensor,
             other_emotion: torch.Tensor,
             pain_observed: float = 0.0,
             proximity: float = 0.5,
             similarity: float = 0.5,
             skill_name: str | None = None) -> dict[str, Any]:
        """
        执行一个社会认知步

        Args:
            observed_action: 观察到的动作 [action_dim]
            self_state: 自身状态 [state_dim]
            other_behavior: 他人行为 [state_dim]
            other_emotion: 他人情绪 [emotion_dim]
            pain_observed: 观察到的疼痛 [0,1]
            proximity: 关系亲密度 [0,1]
            similarity: 自我-他人相似度 [0,1]
            skill_name: 技能名称 (用于模仿学习)
        """
        self.step_count += 1

        # 1. 镜像神经元激活
        mirror_result = self.mirror(
            observed_action, pain_observed=pain_observed, proximity=proximity
        )

        # 2. 心理理论推理
        tom_result = self.tom(self_state, other_behavior)

        # 3. 共情回路激活
        empathy_result = self.empathy(
            other_emotion, proximity=proximity, similarity=similarity
        )

        # 4. 模仿学习
        imitation_result = self.imitation(observed_action, skill_name=skill_name)

        # 5. 社会预测
        behavior_seq = other_behavior.unsqueeze(0) if other_behavior.dim() == 1 else other_behavior
        prediction_result = self.social_predictor(
            behavior_seq.unsqueeze(0) if behavior_seq.dim() == 2 else behavior_seq,
            self_state
        )

        # 6. 计算总体社会能力
        capacity_input = torch.tensor([[
            mirror_result['resonance_level'],
            tom_result['mental_state_confidence'],
            empathy_result['affective_empathy'],
            empathy_result['cognitive_empathy'],
            empathy_result['compassion'],
            empathy_result['personal_distress'],
            imitation_result['copy_accuracy'],
            prediction_result['prediction_confidence'],
        ]])
        social_capacity = float(self.capacity_net(capacity_input).squeeze())

        # 7. 更新总状态
        self.state = SocialCognitionState(
            mirror=self.mirror.state,
            tom=self.tom.state,
            empathy=self.empathy.state,
            imitation=self.imitation.state,
            social_prediction_accuracy=prediction_result['prediction_confidence'],
            overall_social_capacity=social_capacity,
        )

        return {
            # 镜像神经元
            'mirror_resonance': mirror_result['resonance_level'],
            'yawning_trigger': mirror_result['yawning_trigger'],
            'pain_resonance': mirror_result['pain_resonance'],
            'matched_action': mirror_result['matched_action'],
            'contagion_susceptibility': mirror_result['contagion_susceptibility'],
            # 心理理论
            'inferred_belief': tom_result['inferred_belief'],
            'inferred_intent': tom_result['inferred_intent'],
            'tom_confidence': tom_result['mental_state_confidence'],
            'perspective_distance': tom_result['perspective_distance'],
            # 共情
            'affective_empathy': empathy_result['affective_empathy'],
            'cognitive_empathy': empathy_result['cognitive_empathy'],
            'compassion': empathy_result['compassion'],
            'personal_distress': empathy_result['personal_distress'],
            'empathy_regulation': empathy_result['empathy_regulation'],
            'oxytocin_level': empathy_result['oxytocin_level'],
            # 模仿
            'copy_accuracy': imitation_result['copy_accuracy'],
            'learning_progress': imitation_result['learning_progress'],
            'skills_known': imitation_result['skills_known'],
            # 社会预测
            'predicted_outcome': prediction_result['predicted_outcome'],
            'prediction_confidence': prediction_result['prediction_confidence'],
            # 综合
            'overall_social_capacity': social_capacity,
        }

    def get_summary(self) -> dict:
        """获取社会认知系统摘要"""
        return {
            'mirror_resonance': self.state.mirror.resonance_level,
            'yawning_trigger': self.state.mirror.yawning_trigger,
            'pain_resonance': self.state.mirror.pain_resonance,
            'tom_confidence': self.state.tom.mental_state_confidence,
            'affective_empathy': self.state.empathy.affective_empathy,
            'cognitive_empathy': self.state.empathy.cognitive_empathy,
            'compassion': self.state.empathy.compassion,
            'personal_distress': self.state.empathy.personal_distress,
            'skills_known': len(self.state.imitation.skill_acquired),
            'social_capacity': self.state.overall_social_capacity,
            'step_count': self.step_count,
        }


def create_social_cognition(**kwargs) -> SocialCognitionSystem:
    """工厂函数: 创建社会认知系统"""
    return SocialCognitionSystem(**kwargs)


__all__ = [
    'MirrorState',
    'ToMState',
    'EmpathyState',
    'ImitationState',
    'SocialCognitionState',
    'MirrorNeuronSystem',
    'TheoryOfMind',
    'EmpathyCircuit',
    'ImitationLearning',
    'SocialPredictor',
    'SocialCognitionSystem',
    'create_social_cognition',
]
