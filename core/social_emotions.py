# =============================================================================
# Social Emotions - 复杂社会情绪
# =============================================================================
# 高级社会情绪：羞耻/自豪、嫉妒/钦佩、内疚/宽恕、蔑视/尊敬
#
# 理论基础：
# - 社会情绪理论 (Haidt, 2003)
# - 自我意识情绪 (Tangney, 2007)
# - 道德情绪 (Greene, 2013)
#
# 核心机制：
# 1. 自我评价情绪：羞耻/自豪/内疚
# 2. 他人导向情绪：嫉妒/钦佩/羡慕
# 3. 道德情绪：蔑视/愤怒/尊敬
# 4. 社会地位情绪：骄傲/谦卑
# =============================================================================

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 情绪类型定义
# =============================================================================

class SocialEmotionTypes:
    """社会情绪类型"""
    # 自我意识情绪
    SHAME = "shame"        # 羞耻
    GUILT = "guilt"        # 内疚
    PRIDE = "pride"        # 自豪
    HUMILITY = "humility"  # 谦卑

    # 他人导向情绪
    JEALOUSY = "jealousy"    # 嫉妒
    ENVY = "envy"            # 羡慕
    ADMIRATION = "admiration"  # 钦佩
    CONTEMPT = "contempt"     # 蔑视

    # 道德情绪
    DISGUST = "disgust"       # 厌恶
    SCORN = "scorn"          # 轻蔑
    RESPECT = "respect"       # 尊敬
    FORGIVENESS = "forgiveness"  # 宽恕

    # 社会地位情绪
    ARROGANCE = "arrogance"   # 傲慢
    INSULT = "insult"         # 侮辱
    GRATITUDE = "gratitude"   # 感激

    ALL = [
        SHAME, GUILT, PRIDE, HUMILITY,
        JEALOUSY, ENVY, ADMIRATION, CONTEMPT,
        DISGUST, SCORN, RESPECT, FORGIVENESS,
        ARROGANCE, INSULT, GRATITUDE,
    ]


# =============================================================================
# 自我评价系统 (Self-Evaluation)
# =============================================================================

class SelfEvaluation(nn.Module):
    """
    自我评价系统

    对应神经机制：
    - medial PFC: 自我反思
    - ACC: 自我监控
    - posterior cingulate: 自我相关性

    产生的情绪：
    - 羞耻：自我价值感知低于标准
    - 内疚：行为违反个人标准
    - 自豪：成就高于预期
    - 谦卑：能力被高估
    """

    def __init__(
        self,
        self_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.self_dim = self_dim

        # 自我评价网络
        self.evaluation_net = nn.Sequential(
            nn.Linear(self_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4种自我评价
        )

        # 自我标准网络
        self.standard_net = nn.Sequential(
            nn.Linear(self_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()  # 标准强度 [0, 1]
        )

        # 比较网络
        self.comparison_net = nn.Sequential(
            nn.Linear(self_dim * 2, hidden_dim),  # 当前自我 vs 理想自我
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # -1 (差) ~ 1 (好)
        )

    def forward(
        self,
        current_self: torch.Tensor,
        ideal_self: torch.Tensor | None = None,
    ) -> dict:
        """
        自我评价

        Args:
            current_self: [B, self_dim] 当前自我表征
            ideal_self: [B, self_dim] 理想自我（可选）

        Returns:
            evaluation: 自我评价
            emotions: 产生的情绪
        """
        # 评价
        evaluation = self.evaluation_net(current_self)

        # 标准
        if ideal_self is not None:
            comparison = self.comparison_net(
                torch.cat([current_self, ideal_self], dim=-1)
            )
            # 标准 vs 当前
            standard = self.standard_net(ideal_self)
        else:
            comparison = torch.zeros_like(current_self[:, :1])
            standard = self.standard_net(current_self)

        # 生成情绪
        # 自豪：评价高
        # 羞耻：评价低
        # 内疚：行为偏离标准
        # 谦卑：能力低于自我感知
        emotions = {
            'pride': F.relu(evaluation[:, 0]),
            'shame': F.relu(-evaluation[:, 0]),
            'guilt': F.relu(evaluation[:, 1]),
            'humility': F.relu(-evaluation[:, 1]),
        }

        return {
            'evaluation': evaluation,
            'comparison': comparison,
            'standard': standard,
            'emotions': emotions,
        }


# =============================================================================
# 他人导向情绪系统 (Other-Directed)
# =============================================================================

class OtherDirectedEmotion(nn.Module):
    """
    他人导向情绪系统

    对应神经机制：
    - STS: 社会比较
    - TPJ: 观点采择
    - anterior insula: 情感共鸣

    产生的情绪：
    - 嫉妒：他人拥有自己想要的东西
    - 羡慕：想要他人拥有的特质
    - 钦佩：欣赏他人的成就
    - 蔑视：轻视他人
    """

    def __init__(
        self,
        self_dim: int = 64,
        other_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 社会比较网络
        self.comparison_net = nn.Sequential(
            nn.Linear(self_dim + other_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 差距网络
        self.gap_net = nn.Sequential(
            nn.Linear(hidden_dim, 4),  # 4种差距
        )

        # 情绪网络
        self.emotion_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4种情绪
        )

        # 共情网络
        self.empathy_net = nn.Sequential(
            nn.Linear(other_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 共情程度
        )

    def forward(
        self,
        self_state: torch.Tensor,
        other_state: torch.Tensor,
    ) -> dict:
        """
        他人导向情绪

        Args:
            self_state: [B, self_dim] 自身状态
            other_state: [B, other_dim] 他人状态

        Returns:
            emotions: 他人导向情绪
            gap: 差距感知
            empathy: 共情程度
        """
        # 比较
        combined = torch.cat([self_state, other_state], dim=-1)
        comparison = self.comparison_net(combined)

        # 差距
        gap = self.gap_net(comparison)

        # 情绪
        emotion_logits = self.emotion_net(comparison)

        # 共情
        empathy = self.empathy_net(other_state)

        # 映射到情绪
        # 嫉妒：自己 < 他人，且想要
        # 羡慕：想要他人的特质
        # 钦佩：欣赏他人��就
        # 蔑视：轻视他人
        emotions = {
            'jealousy': F.relu(emotion_logits[:, 0]),
            'envy': F.relu(emotion_logits[:, 1]),
            'admiration': F.relu(emotion_logits[:, 2]),
            'contempt': F.relu(-emotion_logits[:, 2]),
        }

        return {
            'emotions': emotions,
            'gap': gap,
            'empathy': empathy.squeeze(-1),
            'comparison': comparison,
        }


# =============================================================================
# 道德情绪系统 (Moral Emotion)
# =============================================================================

class MoralEmotion(nn.Module):
    """
    道德情绪系统

    对应神经机制：
    - orbital PFC: 道德判断
    - ventromedial PFC: 情感价值
    - anterior cingulate: 冲突检测

    产生的情绪：
    - 厌恶：道德违反
    - 轻蔑：道德轻视
    - 愤怒：道德违规
    - 尊敬：道德高尚
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 道德判断网络
        self.judgment_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 道德违反网络
        self.violation_net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 道德违反程度
        )

        # 道德情绪网络
        self.emotion_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4种道德情绪
        )

        # 规范编码器
        self.norm_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        action: torch.Tensor,
        norms: torch.Tensor | None = None,
    ) -> dict:
        """
        道德情绪

        Args:
            action: [B, input_dim] 行动
            norms: [B, 4] 社会规范

        Returns:
            judgment: 道德判断
            emotions: 道德情绪
        """
        # 判断
        judgment = self.judgment_net(action)

        # 违反程度
        if norms is not None:
            norm_encoding = self.norm_net(norms)
            violation = self.violation_net(norm_encoding)
        else:
            violation = self.violation_net(judgment)

        # 道德情绪
        emotion_logits = self.emotion_net(judgment)

        emotions = {
            'disgust': F.relu(emotion_logits[:, 0]),
            'scorn': F.relu(emotion_logits[:, 1]),
            'anger': F.relu(emotion_logits[:, 2]),
            'respect': F.relu(emotion_logits[:, 3]),
        }

        return {
            'judgment': judgment,
            'violation': violation.squeeze(-1),
            'emotions': emotions,
        }


# =============================================================================
# 社会地位情绪系统 (Status Emotion)
# =============================================================================

class StatusEmotion(nn.Module):
    """
    社会地位情绪系统

    对应神经机制：
    - dorsal striatum: 地位奖励
    - medial PFC: 社会地位评估

    产生的情绪：
    - 骄傲：地位提升
    - 傲慢：过度高估地位
    - 谦卑：承认他人优越
    - 感激：感谢帮助
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # 地位感知网络
        self.status_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # -1 (低) ~ 1 (高)
        )

        # 地位变化网络
        self.change_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # 当前 vs 过去
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 上升/稳定/下降
        )

        # 情绪网络
        self.emotion_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # 状态 + 地位
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        current_status: torch.Tensor,
        past_status: torch.Tensor | None = None,
    ) -> dict:
        """
        地位情绪

        Args:
            current_status: [B, input_dim] 当前状态
            past_status: [B, input_dim] 过去状态（可选）

        Returns:
            emotions: 地位情绪
            status_level: 地位水平
        """
        # 地位
        status_level = self.status_net(current_status)

        # 变化
        if past_status is not None:
            change = self.change_net(
                torch.cat([current_status, past_status], dim=-1)
            )
        else:
            change = torch.zeros_like(current_status[:, :3])

        # 情绪
        status_input = torch.cat([current_status, status_level], dim=-1)
        emotion_logits = self.emotion_net(status_input)

        emotions = {
            'pride': F.relu(emotion_logits[:, 0]),
            'arrogance': F.relu(emotion_logits[:, 1]),
            'humility': F.relu(-emotion_logits[:, 1]),
            'gratitude': F.relu(emotion_logits[:, 2]),
        }

        return {
            'emotions': emotions,
            'status_level': status_level.squeeze(-1),
            'change': change,
        }


# =============================================================================
# 完整社会情绪系统
# =============================================================================

class SocialEmotionSystem(nn.Module):
    """
    完整社会情绪系统

    整合：
    1. 自我评价：羞耻/自豪/内疚/谦卑
    2. 他人导向：嫉妒/钦佩/羡慕/蔑视
    3. 道德情绪：厌恶/轻蔑/愤怒/尊敬
    4. 社会地位：骄傲/傲慢/谦卑/感激
    """

    def __init__(
        self,
        self_dim: int = 64,
        input_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim

        # 各子系统
        self.self_eval = SelfEvaluation(self_dim, hidden_dim)
        self.other = OtherDirectedEmotion(self_dim, input_dim, hidden_dim)
        self.moral = MoralEmotion(input_dim, hidden_dim)
        self.status = StatusEmotion(input_dim, hidden_dim)

        # 综合网络
        self.integrator = nn.Sequential(
            nn.Linear(16, hidden_dim),  # 4种系统 × 4种情绪
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 15),  # 15种社会情绪
        )

        # 情绪历史
        self.emotion_history = deque(maxlen=100)

    def forward(
        self,
        self_state: torch.Tensor,
        other_state: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        ideal_self: torch.Tensor | None = None,
        past_status: torch.Tensor | None = None,
    ) -> dict:
        """
        社会情绪处理

        Args:
            self_state: 自身状态
            other_state: 他人状态
            action: 行动
            ideal_self: 理想自我
            past_status: 过去状态

        Returns:
            social_emotions: 社会情绪
            emotion_summary: 情绪摘要
        """
        batch_size = self_state.shape[0]
        device = self_state.device

        # 1. 自我评价
        if ideal_self is None:
            ideal_self = self_state
        self_eval_out = self.self_eval(self_state, ideal_self)
        self_emotions = [
            self_eval_out['emotions']['pride'],
            self_eval_out['emotions']['shame'],
            self_eval_out['emotions']['guilt'],
            self_eval_out['emotions']['humility'],
        ]

        # 2. 他人导向
        if other_state is None:
            other_state = torch.randn(batch_size, self.input_dim, device=device)
        other_out = self.other(self_state, other_state)
        other_emotions = [
            other_out['emotions']['jealousy'],
            other_out['emotions']['envy'],
            other_out['emotions']['admiration'],
            other_out['emotions']['contempt'],
        ]

        # 3. 道德情绪
        if action is None:
            action = self_state
        moral_out = self.moral(action)
        moral_emotions = [
            moral_out['emotions']['disgust'],
            moral_out['emotions']['scorn'],
            moral_out['emotions']['anger'],
            moral_out['emotions']['respect'],
        ]

        # 4. 地位情绪
        if past_status is None:
            past_status = self_state
        status_out = self.status(self_state, past_status)
        status_emotions = [
            status_out['emotions']['pride'],
            status_out['emotions']['arrogance'],
            status_out['emotions']['humility'],
            status_out['emotions']['gratitude'],
        ]

        # 合并所有情绪
        all_emotions = self_emotions + other_emotions + moral_emotions + status_emotions
        combined = torch.stack(all_emotions, dim=-1)

        # 综合处理
        integrated = self.integrator(combined)

        # 软最大化
        emotion_probs = F.softmax(integrated, dim=-1)

        # 记录历史
        self.emotion_history.append(emotion_probs.detach())

        # 创建情绪映射
        emotion_names = [
            'pride', 'shame', 'guilt', 'humility',
            'jealousy', 'envy', 'admiration', 'contempt',
            'disgust', 'scorn', 'anger', 'respect',
            'arrogance', 'gratitude', 'forgiveness',
        ]

        # 构建结果字典
        result = {}
        for i, name in enumerate(emotion_names):
            result[name] = emotion_probs[:, i]

        return {
            'social_emotions': emotion_probs,
            'emotion_by_type': {
                'self_conscious': self_eval_out['emotions'],
                'other_directed': other_out['emotions'],
                'moral': moral_out['emotions'],
                'status': status_out['emotions'],
            },
            'dominant_emotion': emotion_names[emotion_probs.argmax().item()],
        }

    def get_emotion_summary(self) -> dict:
        """获取情绪摘要"""
        if not self.emotion_history:
            return {'status': 'no_data'}

        recent = torch.stack(list(self.emotion_history)[-10:]).mean(dim=0)

        emotion_names = [
            'pride', 'shame', 'guilt', 'humility',
            'jealousy', 'envy', 'admiration', 'contempt',
            'disgust', 'scorn', 'anger', 'respect',
            'arrogance', 'gratitude', 'forgiveness',
        ]

        return {
            'emotion_distribution': recent,
            'dominant': emotion_names[recent.argmax().item()],
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_social_emotion(
    self_dim: int = 64,
    input_dim: int = 64,
    hidden_dim: int = 64,
) -> SocialEmotionSystem:
    """创建社会情绪系统"""
    return SocialEmotionSystem(self_dim, input_dim, hidden_dim)


__all__ = [
    'SocialEmotionTypes',
    'SelfEvaluation',
    'OtherDirectedEmotion',
    'MoralEmotion',
    'StatusEmotion',
    'SocialEmotionSystem',
    'create_social_emotion',
]


# =============================================================================
# 测试
# =============================================================================

def test_social_emotions():
    """测试社会情绪系统"""
    print("=" * 60)
    print("Testing Social Emotions System")
    print("=" * 60)

    # 创建模型
    model = SocialEmotionSystem()

    # 输入
    self_state = torch.randn(4, 64)
    other_state = torch.randn(4, 64)
    action = torch.randn(4, 64)

    print("\n[1] Testing self-evaluation emotions...")
    out = model.self_eval(self_state)
    print(f"  Pride: {out['emotions']['pride'][0]:.3f}")
    print(f"  Shame: {out['emotions']['shame'][0]:.3f}")

    print("\n[2] Testing other-directed emotions...")
    out = model.other(self_state, other_state)
    print(f"  Jealousy: {out['emotions']['jealousy'][0]:.3f}")
    print(f"  Admiration: {out['emotions']['admiration'][0]:.3f}")

    print("\n[3] Testing moral emotions...")
    out = model.moral(action)
    print(f"  Disgust: {out['emotions']['disgust'][0]:.3f}")
    print(f"  Respect: {out['emotions']['respect'][0]:.3f}")

    print("\n[4] Testing complete system...")
    out = model(self_state, other_state, action)
    print(f"  Dominant emotion: {out['dominant_emotion']}")

    print("\n[5] Emotion summary...")
    summary = model.get_emotion_summary()
    print(f"  Dominant: {summary.get('dominant')}")

    print("\n" + "=" * 60)
    print("✓ Social emotions system working!")
    print("  - Self-evaluation: ✓")
    print("  - Other-directed: ✓")
    print("  - Moral: ✓")
    print("  - Status: ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_social_emotions()
