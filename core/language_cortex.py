"""
语言皮层 - 并行+串行混合

并行处理: GRU (整句一起处理)
串行处理: SSM (逐词流式)

use_parallel=True: 并行GRU
use_parallel=False: 串行SSM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ============ SSM组件 (串行) + 类生物门控 ============

# ============ 剪枝机制 ============

class AdaptivePruner(nn.Module):
    """
    自适应剪枝器

    根据:
    1. 重要性 (参数 magnitude)
    2. 冗余度 (激活相关性)
    3. 贡献度 (梯度)
    """
    def __init__(self):
        super().__init__()

        self.prune_ratio = nn.Parameter(torch.tensor(0.3))  # 剪枝比例
        self.granularity = 'channel'  # 粒度

    def compute_importance(self, module: nn.Module):
        """计算重要性分数"""
        importance = {}
        for name, param in module.named_parameters():
            if 'weight' in name:
                # 幅值重要性
                importance[name] = param.abs().mean()
        return importance

    def prune(self, module: nn.Module):
        """执行剪枝"""
        importance = self.compute_importance(module)
        # 保留top-k
        threshold = sorted(importance.values())[int(len(importance) * self.prune_ratio)]
        return threshold

    def get_sparsity(self, module: nn.Module):
        """计算稀疏度"""
        total = sum(p.numel() for p in module.parameters())
        if total == 0:
            return 0

        zero_params = sum((p == 0).sum() for p in module.parameters())
        return (zero_params.float() / total).item()


class DynamicExpertPruner(nn.Module):
    """
    动态专家剪枝

    MOE中只激活top-k专家
    其余专家不参与计算 → 节省算力
    """
    def __init__(self, n_experts: int = 4, top_k: int = 1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # 专家使用计数器
        self.usage_count = nn.Parameter(torch.zeros(n_experts))

    def forward(self, gate_weights: torch.Tensor):
        """
        选择top-k专家

        Returns:
            selected_indices: top-k专家索引
            mask: 稀疏掩码
        """
        # Top-k选择
        top_k_indices = gate_weights.topk(self.top_k, dim=-1)[1]

        # 创建稀疏掩码
        mask = torch.zeros_like(gate_weights)
        mask[0, top_k_indices[0]] = 1

        # 更新使用计数
        with torch.no_grad():
            self.usage_count[top_k_indices[0]] += 1

        return top_k_indices[0].item(), mask

    def get_active_ratio(self):
        """活跃专家比例"""
        return (self.usage_count > 0).float().mean().item()


class SynapticDepression(nn.Module):
    """
    突触效能 Depression

    生物启发的"遗忘剪枝"
    长期不使用的连接逐渐弱化
    """
    def __init__(self, decay_rate: float = 0.01):
        super().__init__()
        self.decay_rate = decay_rate

    def apply_decay(self, param: nn.Parameter):
        """应用衰减"""
        with torch.no_grad():
            param.data *= (1 - self.decay_rate)

    def apply_LTD(self, param: nn.Parameter, activity: float):
        """长期 Depression"""
        # 低活动 → 更强衰减
        decay = self.decay_rate * (1 - activity)
        with torch.no_grad():
            param.data *= (1 - decay)


class OjaRule(nn.Module):
    """
    Oja学习规则

    突触可塑性模型:
    Δw = η * y * (x - wy)
    强调: 共同激活的神经元连接增强
    """
    def __init__(self, learning_rate: float = 0.01):
        super().__init__()
        self.eta = learning_rate

    def update(self, weight: nn.Parameter, pre: torch.Tensor, post: torch.Tensor):
        """更新权重"""
        # Oja规则
        delta = self.eta * post * (pre - weight * post)
        with torch.no_grad():
            weight.data += delta

class EmotionRegulation(nn.Module):
    """
    情绪调节策略

    Gross情绪调节模型:
    - 认知重评 (Cognitive Reappraisal): 改变对事件的解释
    - 表达抑制 (Suppression): 压抑情绪表达
    - 正念 (Mindfulness): 觉察当下
    - 注意力转移 (Attentional Deployment): 转移注意力
    """
    def __init__(self):
        super().__init__()

        # 调节策略强度
        self.cognitive_reappraisal = nn.Parameter(torch.zeros(1))
        self.suppression = nn.Parameter(torch.zeros(1))
        self.mindfulness = nn.Parameter(torch.zeros(1))
        self.attentional_shift = nn.Parameter(torch.zeros(1))

    def regulate(self, emotion_intensity: torch.Tensor, strategy: str = None):
        """
        调节情绪强度

        Args:
            emotion_intensity: 原始情绪强度
            strategy: 指定策略 (None则自动选择)
        """
        if strategy is None:
            # 自动选择: 正念优先，其次认知重评
            strategy = 'mindfulness' if torch.sigmoid(self.mindfulness) > 0.5 else 'reappraisal'

        if strategy == 'reappraisal':
            # 认知重评: 降低情绪影响
            return emotion_intensity * (1 - torch.sigmoid(self.cognitive_reappraisal) * 0.5)
        elif strategy == 'suppression':
            # 压抑: 不推荐，可能反噬
            return emotion_intensity * (1 - torch.sigmoid(self.suppression) * 0.8)
        elif strategy == 'mindfulness':
            # 正念: 保持觉察，减少反应
            return emotion_intensity * (1 - torch.sigmoid(self.mindfulness) * 0.4)
        elif strategy == 'attentional_shift':
            # 注意力转移
            return emotion_intensity * (1 - torch.sigmoid(self.attentional_shift) * 0.3)

        return emotion_intensity


class CognitiveBias(nn.Module):
    """
    认知偏差

    常见认知偏差:
    - 确认偏差 (Confirmation Bias)
    - 锚定效应 (Anchoring)
    - 损失厌恶 (Loss Aversion)
    - 可用性启发 (Availability)
    - 后见之明 (Hindsight)
    """
    def __init__(self, dim: int = 256):
        super().__init__()

        # 各偏差强度
        self.confirmation_bias = nn.Parameter(torch.zeros(1))
        self.anchoring = nn.Parameter(torch.zeros(1))
        self.loss_aversion = nn.Parameter(torch.zeros(1))  # 损失厌恶 > 收益
        self.availability = nn.Parameter(torch.zeros(1))
        self.hindsight = nn.Parameter(torch.zeros(1))

    def apply_bias(self, decision_logits: torch.Tensor, context: str = 'default'):
        """
        应用认知偏差到决策

        损失厌恶: 对损失比对收益更敏感
        """
        # 损失厌恶
        if 'loss' in context:
            loss_weight = 1 + torch.sigmoid(self.loss_aversion) * 0.5
            decision_logits = decision_logits * loss_weight

        # 确认偏差: 增强与已有信念一致的选择
        if 'belief' in context:
            confirmation = torch.sigmoid(self.confirmation_bias) * 0.3
            decision_logits = decision_logits * (1 + confirmation)

        # 锚定效应
        if 'anchor' in context:
            anchor_effect = torch.tanh(self.anchoring) * 0.2
            decision_logits = decision_logits + anchor_effect

        # 可用性启发: 近期事件权重高
        if 'recent' in context:
            avail = torch.sigmoid(self.availability) * 0.2
            decision_logits = decision_logits * (1 + avail)

        return decision_logits


class Metacognition(nn.Module):
    """
    元认知 (Metacognition)

    "对认知的认知"
    - 觉察自己的思维过程
    - 监控认知状态
    - 调节认知策略
    """
    def __init__(self, dim: int = 256):
        super().__init__()

        # 元认知监控
        self.monitoring = nn.Linear(dim, 1)  # 监控认知状态

        # 元认知调节
        self.strategy_selector = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4种策略
        )

        # 认知状态估计
        self.cognitive_state = nn.Parameter(torch.zeros(1))  # 当前认知状态

    def monitor(self, thought: torch.Tensor):
        """
        监控思考过程

        返回: 认知状态 (清晰度)
        """
        clarity = torch.sigmoid(self.monitoring(thought))
        with torch.no_grad():
            self.cognitive_state.data = clarity
        return clarity

    def self_regulate(self, thought: torch.Tensor, task_type: str = 'reasoning'):
        """
        自我调节策略选择

        根据任务类型和当前状态选择策略
        """
        strategy_logits = self.strategy_selector(thought)
        strategy_names = ['analyze', 'memorize', 'creative', 'critical']

        # 根据任务调整
        if task_type == 'memory':
            strategy_logits[0, 1] += 0.5
        elif task_type == 'creative':
            strategy_logits[0, 2] += 0.5

        selected = strategy_logits.argmax(dim=-1).item()
        return strategy_names[selected], F.softmax(strategy_logits, dim=-1)

class PlutchikEmotion(nn.Module):
    """
    Plutchik情绪轮 (8种基本情绪)

    8种基本情绪:
    - 喜悦(Joy) ↔ 悲伤(Sadness)
    - 信任(Trust) ↔ 厌恶(Disgust)
    - 恐惧(Fear) ↔ 愤怒(Anger)
    - 惊讶(Surprise) ↔ 期待(Anticipation)
    """
    EMOTION_NAMES = ['joy', 'sadness', 'trust', 'disgust', 'fear', 'anger', 'surprise', 'anticipation']

    def __init__(self):
        super().__init__()
        # 8维情绪向量 (对称)
        self.emotion_vector = nn.Parameter(torch.zeros(8))

    def forward(self):
        """返回各情绪强度"""
        # 使用sigmoid保证非负
        intensities = torch.sigmoid(self.emotion_vector)
        return {name: intensities[i].item() for i, name in enumerate(self.EMOTION_NAMES)}

    def get_primary(self):
        """获取主导情绪"""
        intensities = torch.sigmoid(self.emotion_vector)
        idx = intensities.argmax().item()
        return self.EMOTION_NAMES[idx], intensities[idx].item()

    def affect_behavior(self, behavior_logits: torch.Tensor):
        """
        情绪影响行为决策

        Plutchik理论:
        - 高兴 → 风险寻求
        - 悲伤 → 风险规避
        - 愤怒 → 快速决策
        - 恐惧 → 拖延/回避
        """
        joy = torch.sigmoid(self.emotion_vector[0])  # 喜悦
        fear = torch.sigmoid(self.emotion_vector[4])  # 恐惧
        anger = torch.sigmoid(self.emotion_vector[5])  # 愤怒

        # 影响
        effect = torch.zeros_like(behavior_logits)
        effect += (joy - 0.5) * 0.3    # 喜悦增加风险偏好
        effect -= (fear - 0.5) * 0.3   # 恐惧增加保守
        effect += anger * 0.2          # 愤怒增加激进

        return behavior_logits + effect


class DualProcessCognition(nn.Module):
    """
    双过程认知 (System 1 / System 2)

    - System 1: 快速、直觉、自动化的
    - System 2: 慢速、理性、审慎的
    """
    def __init__(self, dim: int = 256):
        super().__init__()

        # System 1: 快速通路
        self.system1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
        )

        # System 2: 慢速通路 (更深的推理)
        self.system2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        # 系统选择器 (基于认知负荷)
        self.cognitive_load = nn.Parameter(torch.zeros(1))
        self.cognitive_capacity = 7  # Miller 7±2

    def forward(self, input_emb: torch.Tensor, task_difficulty: float = 0.5):
        """
        根据任务难度选择系统

        低难度 → System 1
        高难度 → System 2
        """
        # 计算当前认知负荷
        load = torch.sigmoid(self.cognitive_load)

        if task_difficulty > 0.6 or load > 0.7:
            # 高负荷，使用System 2
            output = self.system2(input_emb)
            system = 'System2'
        else:
            # 低负荷，使用System 1
            output = self.system1(input_emb)
            system = 'System1'

        # 学习: 任务完成增加负荷容量
        with torch.no_grad():
            self.cognitive_load.data *= 0.95  # 衰减
            self.cognitive_load.data += 0.01 * task_difficulty

        return output, system


class EmbodiedCognition(nn.Module):
    """
    具身认知 (Embodied Cognition)

    理论: 认知源于��体与环境的交互
    实现: 情感状态影响感知处理
    """
    def __init__(self):
        super().__init__()

        # 身体状态
        self.energy = nn.Parameter(torch.zeros(1))      # 能量水平
        self.fatigue = nn.Parameter(torch.zeros(1))    # 疲劳程度
        self.stress = nn.Parameter(torch.zeros(1))     # 压力水平

        # 身体对感知的影响
        self.body_affect_perception = nn.Linear(3, 1)

    def forward(self, perception_input: torch.Tensor):
        """
        身体状态影响感知

        例如:
        - 疲劳时更易分心
        - 高压下感知变窄
        - 饥饿时情绪波动大
        """
        body_state = torch.cat([
            torch.sigmoid(self.energy),
            torch.sigmoid(self.fatigue),
            torch.sigmoid(self.stress),
        ])

        # 疲劳降低感知效率
        fatigue_effect = -self.body_affect_perception(body_state.unsqueeze(0)) * 0.3

        # 压力使感知变窄
        stress_effect = -torch.sigmoid(self.stress) * 0.2

        return perception_input * (1 + fatigue_effect + stress_effect)

    def update_body(self, sleep: float, food: float, social: float):
        """根据生理需求更新"""
        with torch.no_grad():
            self.energy.data += 0.01 * (sleep * 0.3 + food * 0.3 + social * 0.4)
            self.energy.data = torch.clamp(self.energy.data, 0, 1)

            self.fatigue.data += 0.01 * (1 - sleep)
            self.fatigue.data = torch.clamp(self.fatigue.data, 0, 1)

            self.stress.data += 0.01 * (1 - social)
            self.stress.data = torch.clamp(self.stress.data, 0, 1)


class CognitiveLoadManager(nn.Module):
    """
    认知负荷管理器

    理论: 工作记忆容量有限 (7±2)
    实现: 监控并调节认知负荷
    """
    def __init__(self, capacity: int = 7):
        super().__init__()
        self.capacity = capacity
        self.current_load = nn.Parameter(torch.zeros(1))

    def check_load(self):
        """检查是否超载"""
        return torch.sigmoid(self.current_load) > (self.capacity / 10)

    def allocate(self, amount: float):
        """分配认知资源"""
        with torch.no_grad():
            new_load = self.current_load + amount
            if new_load > self.capacity:
                return False  # 超载
            self.current_load = new_load
            return True

    def release(self):
        """释放认知资源"""
        with torch.no_grad():
            self.current_load.data *= 0.8

class Neuromodulator(nn.Module):
    """
    神经调节器 (顶层)

    模拟大脑中的神经递质系统:
    - 多巴胺 (Dopamine): 奖励/动机
    - 血清素 (Serotonin): 情绪稳定/风险
    - 去甲肾上腺素 (Norepinephrine): 唤醒/注意
    """
    def __init__(self, dim: int = 256):
        super().__init__()

        # 三种神经调节系统
        self.dopamine = nn.Parameter(torch.zeros(1))    # 奖励预期
        self.serotonin = nn.Parameter(torch.zeros(1))   # 情绪稳定
        self.norepinephrine = nn.Parameter(torch.zeros(1))  # 唤醒度

        # 调节器输出
        self.modulator_out = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
        )

    def forward(self, context: torch.Tensor = None):
        """返回调节信号"""
        signals = torch.cat([
            torch.sigmoid(self.dopamine),
            torch.sigmoid(self.serotonin),
            torch.sigmoid(self.norepinephrine),
        ], dim=-1)

        modulation = self.modulator_out(signals)
        return {
            'dopamine': torch.sigmoid(self.dopamine),
            'serotonin': torch.sigmoid(self.serotonin),
            'norepinephrine': torch.sigmoid(self.norepinephrine),
            'modulation': modulation,
        }

    def update(self, reward: float, surprise: float, arousal: float):
        """根据经验更新调节器"""
        with torch.no_grad():
            # 多巴胺: 奖励学习
            self.dopamine.data += 0.01 * (reward - 0.5)
            self.dopamine.data = torch.clamp(self.dopamine.data, -2, 2)

            # 血清素: 情绪稳定
            self.serotonin.data -= 0.01 * arousal
            self.serotonin.data = torch.clamp(self.serotonin.data, -2, 2)

            # 去甲肾上腺素: 唤醒调节
            self.norepinephrine.data += 0.01 * (surprise - 0.5)
            self.norepinephrine.data = torch.clamp(self.norepinephrine.data, -2, 2)


class MoodState(nn.Module):
    """
    心境状态 (中层)

    比情绪更持久，影响整体认知风格:
    - 乐观/悲观
    - 平静/焦虑
    - 自信/犹豫
    """
    def __init__(self):
        super().__init__()

        # 心境维度
        self.mood_valence = nn.Parameter(torch.zeros(1))    # 乐观度
        self.mood_arousal = nn.Parameter(torch.zeros(1))   # 焦虑度
        self.mood_dominance = nn.Parameter(torch.zeros(1))   # 自信度

        # 心境记忆 (更长衰减)
        self.mood_decay = 0.99

    def forward(self):
        return {
            'optimism': torch.tanh(self.mood_valence),
            'anxiety': torch.sigmoid(self.mood_arousal),
            'confidence': torch.sigmoid(self.mood_dominance),
        }

    def read_mood(self):
        """读取当前心境"""
        return torch.cat([self.forward()['optimism'], self.forward()['anxiety'], self.forward()['confidence']])

    def mood_affect_decision(self, base_logits: torch.Tensor):
        """心境影响决策"""
        m = self.forward()

        # 乐观 → 更冒险 (更高的logits)
        # 焦虑 → 更保守 (抑制高logits)
        # 自信 → 更果断 (放大差异)

        mood_effect = torch.zeros_like(base_logits)
        mood_effect += m['optimism'] * 0.3    # 乐观提升
        mood_effect -= m['anxiety'] * 0.2    # 焦虑抑制
        mood_effect += (m['confidence'] - 0.5) * 0.3  # 自信调节

        return base_logits + mood_effect


class BioGate(nn.Module):
    """
    类生物门控机制 (底层)

    结合:
    1. 输入内容 (content)
    2. 膜电位累积 (membrane potential) - 历史状态
    3. 情绪状态 (emotional state) - VAD
    4. 心境影响 (mood) - 中层调节
    """
    def __init__(self, dim: int = 256, n_experts: int = 4):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts

        # 内容门控
        self.content_gate = nn.Linear(dim, n_experts)

        # 膜电位累积器
        self.membrane_potential = nn.Parameter(torch.zeros(n_experts))
        self.membrane_decay = 0.9

        # 情绪向量 VAD
        self.emotion_vector = nn.Parameter(torch.zeros(3))

        # 心境状态 (中层)
        self.mood = MoodState()

    def forward(self, input_emb: torch.Tensor, return_emotion: bool = False):
        """
        Args:
            input_emb: [B, dim]
            return_emotion: 是否返回情绪状态
        Returns:
            expert_idx: 选择的专家ID
            gate_weights: 各专家权重
            emotion: (可选)情绪状态
        """
        # === 步骤1: 内容门控 ===
        content_logits = self.content_gate(input_emb)

        # === 步骤2: 膜电位影响 ===
        membrane_effect = self.membrane_potential.unsqueeze(0)

        # === 步骤3: 情绪影响 ===
        emotion_effect = torch.tanh(self.emotion_vector.sum()) * 0.3

        # === 步骤4: 心境影响 (中层调节) ===
        mood_effect = self.mood.mood_affect_decision(content_logits)

        # === 综合门控 ===
        gate_logits = content_logits + membrane_effect + emotion_effect + mood_effect
        gate_weights = F.softmax(gate_logits, dim=-1)

        expert_idx = gate_weights.argmax(dim=-1)

        # 更新膜电位
        with torch.no_grad():
            updates = torch.zeros_like(self.membrane_potential)
            updates[expert_idx] = 0.1
            self.membrane_potential.data = self.membrane_potential.data * self.membrane_decay + updates

        if return_emotion:
            emotion_state = {
                'valence': torch.tanh(self.emotion_vector[0]),
                'arousal': torch.sigmoid(self.emotion_vector[1]),
                'dominance': torch.sigmoid(self.emotion_vector[2]),
            }
            return expert_idx, gate_weights, emotion_state

        return expert_idx, gate_weights


class SSMStateUpdate(nn.Module):
    """选择性状态更新 + 生物门控"""
    def __init__(self, dim: int = 256):
        super().__init__()
        # 生物门控
        self.bio_gate = BioGate(dim, n_experts=4)

        # 多个"专家"候选
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
            for _ in range(4)
        ])

        self.gate_proj = nn.Linear(dim * 2, dim)
        self.candidate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh())

    def forward(self, state, input_emb, apply_gate=True):
        if apply_gate:
            expert_idx, gate_weights = self.bio_gate(input_emb)
            # tensor转整数
            expert_idx = expert_idx.item() if expert_idx.numel() == 1 else expert_idx[0].item()
            expert_output = self.experts[expert_idx](input_emb)
        else:
            expert_output = input_emb

        combined = torch.cat([state, input_emb], dim=-1)
        update_gate = torch.sigmoid(self.gate_proj(combined))

        new_state = state + update_gate * (self.candidate(combined) - state)
        info_gain = gate_weights.max()

        return new_state, info_gain


class WorkingMemory(nn.Module):
    """工作记忆 (7槽) + 情绪影响读写"""
    def __init__(self, dim: int = 256):
        super().__init__()
        self.register_buffer('slots', torch.zeros(7, dim))
        self.importance = nn.Parameter(torch.ones(7))

        # 情绪影响阅读门控
        self.emotion_gate = nn.Linear(dim, 1)
        self.emotion_bias = nn.Parameter(torch.zeros(7))  # 情绪偏置

    def read(self, query, emotion_state=None):
        # 基础分数
        scores = F.softmax(self.importance, dim=0)

        # 情绪影响: 负面情绪降低重要性感知
        if emotion_state is not None:
            neg_bias = emotion_state['valence'] * -0.5
            scores = scores + neg_bias

        top_k = min(3, 7)
        top_idx = scores.argsort(descending=True)[:top_k]
        return self.slots[top_idx].mean(dim=0, keepdim=True)

    def write(self, state, info_gain, emotion_state=None):
        # 情绪影响写入: 高唤醒抑制写入
        arousal_penalty = 0.0
        if emotion_state is not None:
            arousal_penalty = emotion_state['arousal'] * 0.2

        threshold = 0.3 - arousal_penalty
        if info_gain > threshold:
            lowest = self.importance.argmin()
            self.slots[lowest] = state[0].detach()
            self.importance.data[lowest] = info_gain


class SemanticEncoder(nn.Module):
    """层级语义"""
    def __init__(self, dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU()) for _ in range(3)
        ])

    def forward(self, x):
        outs = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0: outs['word'] = x
            elif i == 1: outs['phrase'] = x
            else: outs['sentence'] = x
        return outs


# ============ 并行处理组件 ============

class ParallelEncoder(nn.Module):
    """并行GRU处理"""
    def __init__(self, dim: int = 256):
        super().__init__()
        hidden = dim * 2  # 增大隐藏层
        self.gru = nn.GRU(dim, hidden, batch_first=True, bidirectional=True, num_layers=2)
        self.proj = nn.Linear(hidden * 4, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, embeddings):
        output, _ = self.gru(embeddings)
        # 双向拼接首尾
        combined = torch.cat([output[:, -1, :], output[:, 0, :]], dim=-1)
        return F.relu(self.proj(combined))


# ============ 完整语言皮层 ============

class LanguageCortex(nn.Module):
    """
    语言皮层 + 认知心理学

    use_parallel=True: 并行GRU (批量快)
    use_parallel=False: 串行SSM + Bio-Gating + 心理学机制

    心理学机制:
    - Plutchik情绪轮
    - 双过程认知 (System 1/2)
    - 具身认知
    - 认知负荷管理
    """
    def __init__(
        self,
        vocab_size: int = 10000,
        use_parallel: bool = True,
        event_bus=None,
    ):
        super().__init__()
        self.use_parallel = use_parallel
        embed_dim = 256

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if use_parallel:
            self.encoder = ParallelEncoder(embed_dim)
        else:
            self.ssm = SSMStateUpdate(embed_dim)
            self.memory = WorkingMemory(embed_dim)
            self.semantic = SemanticEncoder(embed_dim)

            # 心理学组件
            self.plutchik = PlutchikEmotion()
            self.dual_process = DualProcessCognition(embed_dim)
            self.embodied = EmbodiedCognition()
            self.cognitive_load = CognitiveLoadManager()

            # 扩展心理学组件
            self.emotion_regulation = EmotionRegulation()
            self.cognitive_bias = CognitiveBias(embed_dim)
            self.metacognition = Metacognition(embed_dim)

            # 剪枝组件
            self.expert_pruner = DynamicExpertPruner(n_experts=4, top_k=1)
            self.synaptic_depression = SynapticDepression(decay_rate=0.01)
            self.oja = OjaRule(learning_rate=0.01)

        # 情感头
        self.emotion = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        # Event-driven registration
        if event_bus is not None:
            event_bus.subscribe(
                "sensory_process",
                self._handle_sensory_process,
                priority=1,
                name="language_cortex",
            )

    @staticmethod
    def _text_to_tokens(text: str):
        """确定性文本到 token 编码（相同文本产生相同 token）"""
        import torch as _torch
        words = text.split()
        if not words:
            return _torch.tensor([[0]])
        # 用字符 hash 映射到 vocab 空间，确定性且可复现
        tokens = []
        for word in words:
            token_id = hash(word) % 10000
            tokens.append(token_id)
        return _torch.tensor([tokens])

    def _handle_sensory_process(self, event) -> Dict:
        """Event-driven handler for sensory_process events."""
        import torch as _torch
        user_input = event.data.get("user_input")
        if user_input is None:
            return {}

        # Convert user_input to tokens tensor
        # 使用确定性字符编码代替随机 token，相同文本产生相同结果
        if isinstance(user_input, str):
            tokens = self._text_to_tokens(user_input)
        else:
            tokens = user_input if isinstance(user_input, _torch.Tensor) else _torch.randint(0, 10000, (1, 10))

        result = self(tokens)

        state = event.data.get("internal_state", {})
        state["language_valence"] = result["valence"].item() if hasattr(result["valence"], "item") else result["valence"]
        state["language_arousal"] = result["arousal"].item() if hasattr(result["arousal"], "item") else result["arousal"]
        state["language_surprise"] = result["surprise"] if isinstance(result["surprise"], float) else float(result["surprise"])

        return result

    def forward(self, tokens: torch.Tensor, return_emotion: bool = False) -> Dict:
        """
        Args:
            tokens: [B, T]
            return_emotion: 是否返回情绪状态
        """
        B, T = tokens.shape
        embeddings = self.embedding(tokens)

        if self.use_parallel:
            final_state = self.encoder(embeddings)
            semantic = {'word': final_state, 'phrase': final_state, 'sentence': final_state}
            surprise = 0.5
        else:
            # 串行处理 + 生物门控
            state = torch.zeros(B, 256, device=embeddings.device)
            emotion_state = None

            for t in range(T):
                emb = embeddings[:, t, :]  # [B, dim]
                state, info_gain = self.ssm(state, emb, apply_gate=True)

                # 情绪反馈
                if return_emotion:
                    _, _, emotion_state = self.ssm.bio_gate(emb, return_emotion=True)

            # 读取记忆
            final_state = self.memory.read(embeddings[:, -1, :], emotion_state)
            semantic = {'word': final_state, 'phrase': final_state, 'sentence': final_state}
            surprise = info_gain if isinstance(info_gain, float) else info_gain.item()

        # 情感特征
        emotion_out = self.emotion(final_state)
        valence = torch.tanh(emotion_out[:, 0])
        arousal = torch.sigmoid(emotion_out[:, 1])

        result = {
            'features': final_state,
            'valence': valence,
            'arousal': arousal,
            'semantic': semantic,
            'surprise': surprise,
        }

        if return_emotion and self.ssm.bio_gate.emotion_vector is not None:
            result['emotion_state'] = {
                'valence': torch.tanh(self.ssm.bio_gate.emotion_vector[0]),
                'arousal': torch.sigmoid(self.ssm.bio_gate.emotion_vector[1]),
                'dominance': torch.sigmoid(self.ssm.bio_gate.emotion_vector[2]),
            }

        return result


def create_language_cortex(
    vocab_size: int = 10000,
    use_parallel: bool = True,
) -> LanguageCortex:
    return LanguageCortex(vocab_size, use_parallel)


__all__ = ['LanguageCortex', 'create_language_cortex']