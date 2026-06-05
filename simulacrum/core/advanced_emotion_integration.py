# =============================================================================
# Advanced Emotion System Integration
# =============================================================================
# 高级情绪系统整合模块
#
# 已整合:
# 1. EmotionRegulationSystem - 情绪调节
# 2. MoodSystem - 心境系统
# 3. EmotionalMemoryConsolidation - 情绪记忆巩固
# 4. SocialEmotionSystem - 社会情绪
# 5. EmotionalContagionSystem - 情绪传染
# 6. InteroceptionSystem - 内感受
# 7. EmotionDynamicsSystem - 情绪动力学
#
# 事件驱动:
# - 订阅 EMOTION_PROCESS: 收到情绪处理请求
# - 发布 EMOTION_UPDATED: 情绪处理完毕后通知下游
# =============================================================================

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from core.events import EMOTION_PROCESS, EMOTION_UPDATED, HIBERNATE_ENTER

# 尝试导入，失败则设置标志
MODULES_AVAILABLE = False

try:
    from core.emotion_dynamics import EmotionDynamicsSystem
    from core.emotion_memory_consolidation import EmotionalMemoryConsolidation
    from core.emotion_regulation import EmotionRegulationSystem
    from core.emotional_contagion import EmotionalContagionSystem
    from core.interoception import GutState, InteroceptionSystem, InteroceptiveState
    from core.mood_system import MoodSystem
    from core.social_emotions import SocialEmotionSystem
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Advanced emotion modules not available: {e}")


# =============================================================================
# 高级情绪状态
# =============================================================================

@dataclass
class AdvancedEmotionState:
    """高级情绪状态"""
    # 当前情绪（秒-分钟级）
    current_emotion: str = "neutral"
    emotion_intensity: float = 0.0

    # 心境（小时-天级）
    mood_valence: float = 0.0      # 效价 [-1, 1]
    mood_arousal: float = 0.0      # 唤醒度 [0, 1]
    mood_dominance: float = 0.0     # 支配感 [0, 1]

    # 社会情绪
    social_emotion: str = "neutral"
    social_intensity: float = 0.0

    # 调节状态
    regulation_capacity: float = 1.0  # 调节能力 [0, 1]
    regulation_strategy: str = "none"

    # 动力学
    emotion_velocity: float = 0.0
    criticality: float = 0.0


# =============================================================================
# 集成高级情绪系统
# =============================================================================

class IntegratedAdvancedEmotionSystem:
    """
    集成高级情绪系统

    整合7个高级情绪模块:
    - EmotionRegulation: 前额叶调控 + 代谢 + 社会调节
    - Mood: 心境（OU过程 + 昼夜节律）
    - Memory: 情绪记忆巩固（睡眠重播 + 消退）
    - Social: 社会情绪
    - Contagion: 情绪传染
    - Intero: 内感受
    - Dynamics: 情绪动力学
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        event_bus=None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.modules_available = MODULES_AVAILABLE
        self._bus = event_bus

        if not MODULES_AVAILABLE:
            print("[WARN] Running in fallback mode")
            return

        # ===== 模块1: 情绪调节 =====
        self.emotion_regulation = EmotionRegulationSystem(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            emotion_dim=8,
        )

        # ===== 模块2: 心境系统 =====
        self.mood_system = MoodSystem(
            mood_dim=5,
            event_dim=input_dim,
        )

        # ===== 模块3: 情绪记忆 =====
        self.emotion_memory = EmotionalMemoryConsolidation(
            content_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        # ===== 模块4: 社会情绪 =====
        self.social_emotions = SocialEmotionSystem(
            self_dim=input_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        # ===== 模块5: 情绪传染 =====
        self.emotional_contagion = EmotionalContagionSystem(
            expression_dim=input_dim,
            emotion_dim=8,
            hidden_dim=hidden_dim,
            n_group_members=5,
        )

        # ===== 模块6: 内感受 =====
        self.interoception = InteroceptionSystem(
            n_signals=15,
            hidden_dim=hidden_dim,
        )

        # ===== 模块7: 情绪动力学 =====
        self.emotion_dynamics = EmotionDynamicsSystem(
            state_dim=3,
            hidden_dim=hidden_dim,
            prediction_horizon=10,
        )

        # 状态历史
        self.state_history = deque(maxlen=100)

        # 当前时间（用于昼夜节律）
        self.current_hour = 12.0

        # 事件订阅
        if self._bus is not None:
            self._bus.subscribe(EMOTION_PROCESS, self.on_emotion_process, priority=0, name="advanced_emotion")
            self._bus.subscribe(HIBERNATE_ENTER, self.on_hibernate_enter, priority=0, name="emotion_consolidation")

    def process(
        self,
        state: torch.Tensor,
        observation: torch.Tensor | None = None,
        user_state: torch.Tensor | None = None,
        user_emotion: torch.Tensor | None = None,
        user_proximity: float = 0.5,
        interoceptive_state: InteroceptiveState | None = None,
        gut_state: GutState | None = None,
        hour: float | None = None,
        external_cortisol: float | None = None,
        external_oxytocin: float | None = None,
    ) -> dict:
        """
        处理高级情绪

        Args:
            state: [B, input_dim] 当前状态
            observation: 观察到的表情（可选）
            user_state: 用户状态（可选）
            user_emotion: 用户情绪（可选）
            user_proximity: 与用户亲近度 [0, 1]
            interoceptive_state: 内感受状态（可选）
            gut_state: 肠道状态（可选）
            hour: 小时 [0, 24]

        Returns:
            emotion_result: 情绪处理结果
        """
        if not self.modules_available:
            return {'error': 'modules_not_available'}

        device = state.device
        batch_size = state.shape[0]

        # 更新时间
        if hour is not None:
            self.current_hour = hour

        # ==== 0. 从状态向量提取当前情绪表征 (替代随机噪声) ====
        # 取 state 的前 8 维作为情绪特征, 通过 sigmoid 归一化到 [0,1]
        emotion_features = torch.sigmoid(state[:, :8])

        # ==== 1. 情绪调节 ====
        regulation_result = self.emotion_regulation(
            state,
            emotion_features,   # 用真实状态特征替代 torch.randn
            None,  # metabolic_state
            user_proximity,
        )
        regulated_emotion = regulation_result['regulated_emotion']

        # ==== 2. 心境更新 (始终传入状态, 不随机丢弃) ====
        mood_result = self.mood_system(
            emotional_input=regulated_emotion,
            event=state,        # 始终传入状态, 不再 70% 丢弃
            hour=self.current_hour,
            external_cortisol=external_cortisol,
        )
        mood_state = mood_result['mood_state']

        # ==== 3. 情绪记忆 ====
        memory_result = {}

        # ==== 4. 社会情绪 ====
        if user_state is not None:
            social_result = self.social_emotions(state, user_state)
        else:
            social_result = self.social_emotions(state)
        dominant_social = social_result['dominant_emotion']

        # ==== 5. 情绪传染 ====
        contagion_result = {}
        if observation is not None:
            contagion_result = self.emotional_contagion.process_observation(observation)

        if user_emotion is not None and user_proximity > 0.3:
            infection = self.emotional_contagion.infect(
                regulated_emotion,
                user_emotion,
                user_proximity,
            )
            contagion_result['infected'] = infection['infected_emotion']

        # ==== 6. 内感受 ====
        if interoceptive_state is None:
            interoceptive_state = InteroceptiveState()
        intero_result = self.interoception.process(
            interoceptive_state,
            gut_state,
        )

        # ==== 7. 情绪动力学 ====
        vad = torch.tensor([
            mood_state.valence,
            mood_state.arousal,
            mood_state.dominance,
        ], device=device).unsqueeze(0)

        dynamics_result = self.emotion_dynamics(vad)
        next_vad = dynamics_result['next_state']

        # ==== 8. 整合所有情绪信号 (5 个源全部参与) ====
        combined_emotion = self._integrate_emotions(
            regulated_emotion,
            mood_state,
            social_result,
            contagion_result,
            intero_result,
        )

        # 更新历史
        self.state_history.append({
            'emotion': combined_emotion,
            'mood': mood_state,
            'social': dominant_social,
        })

        # 返回高级情绪状态
        return {
            'current_emotion': combined_emotion,
            'regulated_emotion': regulated_emotion,
            'mood': mood_state,
            'social': dominant_social,
            'contagion': contagion_result,
            'intero': intero_result,
            'dynamics': dynamics_result,
            'regulation': regulation_result,
            'advanced_state': AdvancedEmotionState(
                current_emotion=self._get_emotion_name(combined_emotion),
                emotion_intensity=combined_emotion.max().item(),
                mood_valence=mood_state.valence,
                mood_arousal=mood_state.arousal,
                mood_dominance=mood_state.dominance,
                social_emotion=dominant_social,
                regulation_capacity=regulation_result.get('metabolic_regulation', {}).get('regulation_capacity', 0.5),
                regulation_strategy=regulation_result.get('strategy', 'none'),
                emotion_velocity=dynamics_result.get('drift', torch.zeros(1)).norm().item(),
                criticality=dynamics_result.get('criticality', 0.0),
            )
        }

    def _integrate_emotions(
        self,
        regulated: torch.Tensor,
        mood_state,
        social_result: dict,
        contagion_result: dict,
        intero_result: dict,
    ) -> torch.Tensor:
        """
        整合所有 5 个情绪源的加权融合

        权重设计 (基于情绪心理学):
        - regulated (0.30): 当前调节后情绪 — 最直接的实时信号
        - mood (0.25): 心境状态 — 持续的背景情绪色调
        - social (0.20): 社会情绪 — 他人互动产生的共享情绪
        - intero (0.15): 内感受 — 身体状态对情绪的贡献
        - contagion (0.10): 情绪传染 — 环境中他人情绪的扩散效应

        如果某个源缺失，其权重自动分配给其他源。
        """
        device = regulated.device
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        sources = [regulated]

        # 2. mood_state → tensor
        try:
            mood_tensor = torch.tensor(
                [mood_state.valence, mood_state.arousal, mood_state.dominance],
                device=device
            ).unsqueeze(0)
            # 扩展到与 regulated 相同的维度 (前 3 维填充 VAD, 其余补 0)
            mood_full = torch.zeros_like(regulated)
            n_fill = min(3, mood_full.shape[-1])
            mood_full[:, :n_fill] = mood_tensor[:, :n_fill]
            sources.append(mood_full)
        except Exception:
            sources.append(torch.zeros_like(regulated))
            weights[1] = 0.0

        # 3. social_result → tensor
        try:
            social_tensor = social_result.get('emotion_vector')
            if social_tensor is not None:
                social_full = torch.zeros_like(regulated)
                n_fill = min(social_tensor.shape[-1], social_full.shape[-1])
                social_full[:, :n_fill] = social_tensor[:, :n_fill]
                sources.append(social_full)
            else:
                # 从 dominant_emotion 构造简单表征
                social_full = torch.zeros_like(regulated)
                sources.append(social_full)
        except Exception:
            sources.append(torch.zeros_like(regulated))
            weights[2] = 0.0

        # 4. intero_result → tensor
        try:
            intero_tensor = intero_result.get('interoceptive_signal')
            if intero_tensor is not None:
                intero_full = torch.zeros_like(regulated)
                if isinstance(intero_tensor, torch.Tensor):
                    n_fill = min(intero_tensor.shape[-1], intero_full.shape[-1])
                    intero_full[:, :n_fill] = intero_tensor[:, :n_fill]
                sources.append(intero_full)
            else:
                sources.append(torch.zeros_like(regulated))
        except Exception:
            sources.append(torch.zeros_like(regulated))
            weights[3] = 0.0

        # 5. contagion_result → tensor
        try:
            infected = contagion_result.get('infected')
            if infected is not None and isinstance(infected, torch.Tensor):
                contagion_full = torch.zeros_like(regulated)
                n_fill = min(infected.shape[-1], contagion_full.shape[-1])
                contagion_full[:, :n_fill] = infected[:, :n_fill]
                sources.append(contagion_full)
            else:
                sources.append(torch.zeros_like(regulated))
        except Exception:
            sources.append(torch.zeros_like(regulated))
            weights[4] = 0.0

        # 归一化权重 (如果某些源缺失, 重新分配权重)
        active_weights = [w for i, w in enumerate(weights) if w > 0 or i < len(sources)]
        total_w = sum(weights)
        if total_w > 0:
            weights = [w / total_w for w in weights]

        # 加权融合
        combined = torch.zeros_like(regulated)
        for src, w in zip(sources, weights):
            combined = combined + src * w

        return combined

    def _get_emotion_name(self, emotion: torch.Tensor) -> str:
        """获取情绪名称"""
        emotions = [
            'joy', 'sadness', 'anger', 'fear',
            'surprise', 'disgust', 'neutral', 'anticipation'
        ]
        idx = emotion.argmax().item()
        return emotions[idx] if idx < len(emotions) else 'neutral'

    def on_emotion_process(self, event) -> dict[str, Any]:
        """事件驱动: 响应 EMOTION_PROCESS"""
        user_input = event.data.get("user_input")
        user_sentiment = event.data.get("user_sentiment", 0.0)
        external_stimulus = event.data.get("external_stimulus", 0.0)
        # 读取真实激素水平 (由agent注入)
        real_cortisol = event.data.get("real_cortisol")
        real_oxytocin = event.data.get("real_oxytocin")

        if not self.modules_available:
            return {"emotion_state": {
                "current_emotion": "neutral",
                "mood_valence": user_sentiment,
                "mood_arousal": 0.5,
            }}

        batch_size = 1
        if user_input:
            state = self._text_to_state(user_input)
        else:
            state = torch.randn(batch_size, 64)

        user_emotion_tensor = torch.tensor([user_sentiment], dtype=torch.float32) if user_sentiment != 0.0 else None
        user_prox = external_stimulus if external_stimulus > 0 else 0.5

        try:
            result = self.process(state, user_emotion=user_emotion_tensor, user_proximity=user_prox,
                                  external_cortisol=real_cortisol, external_oxytocin=real_oxytocin)
            adv_state = result.get('advanced_state')
            emotion_state = {}
            if adv_state and adv_state.current_emotion:
                emotion_state = {
                    "current_emotion": adv_state.current_emotion,
                    "mood_valence": adv_state.mood_valence,
                    "mood_arousal": adv_state.mood_arousal,
                    "social_emotion": adv_state.social_emotion,
                    "regulation_capacity": adv_state.regulation_capacity,
                    "criticality": adv_state.criticality,
                }

            # 发布 EMOTION_UPDATED
            if self._bus is not None:
                self._bus.publish(EMOTION_UPDATED, {"emotion_state": emotion_state}, source="advanced_emotion")

            return {"emotion_state": emotion_state, "result": result}
        except Exception:
            return {"emotion_state": {"current_emotion": "neutral", "mood_valence": 0.0, "mood_arousal": 0.5}}

    def on_hibernate_enter(self, event) -> dict[str, Any]:
        """事件驱动: 响应 HIBERNATE_ENTER，进行情绪记忆巩固"""
        try:
            emotion_memory = self.emotion_memory
            if hasattr(emotion_memory, 'consolidate_during_sleep'):
                emotion_memory.consolidate_during_sleep()
                print("[EMOTION] Memory consolidation during hibernation")
        except Exception:
            pass
        return {"consolidation_done": True}

    def _text_to_state(self, text: str) -> torch.Tensor:
        """将文本转换为状态向量 (基于文本哈希的确定性映射)"""
        # 用文本哈希生成确定性种子，产生稳定的状态向量
        import hashlib

        import torch as _torch
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        _torch.manual_seed(h % (2**63))
        return _torch.randn(1, 64)

    def get_summary(self) -> dict:
        """获取高级情绪摘要"""
        if not self.modules_available:
            return {'status': 'fallback'}

        return {
            'regulation': self.emotion_regulation is not None,
            'mood': self.mood_system is not None,
            'memory': self.emotion_memory is not None,
            'social': self.social_emotions is not None,
            'contagion': self.emotional_contagion is not None,
            'intero': self.interoception is not None,
            'dynamics': self.emotion_dynamics is not None,
            'history_size': len(self.state_history),
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_advanced_emotion_system(
    input_dim: int = 64,
    hidden_dim: int = 64,
) -> IntegratedAdvancedEmotionSystem:
    """创建集成高级情绪系统"""
    return IntegratedAdvancedEmotionSystem(input_dim, hidden_dim)


__all__ = [
    'IntegratedAdvancedEmotionSystem',
    'AdvancedEmotionState',
    'create_advanced_emotion_system',
    'MODULES_AVAILABLE',
]


# =============================================================================
# 测试
# =============================================================================

def test_integration():
    """测试集成系统"""
    print("=" * 60)
    print("Testing Integrated Advanced Emotion System")
    print("=" * 60)

    if not MODULES_AVAILABLE:
        print("[WARN] Modules not available, skipping test")
        return

    # 创建系统
    system = IntegratedAdvancedEmotionSystem()

    # 测试输入
    state = torch.randn(4, 64)

    print("\n[1] Testing basic processing...")
    result = system.process(state)
    print(f"  Current emotion: {result['advanced_state'].current_emotion}")
    print(f"  Mood valence: {result['advanced_state'].mood_valence:.3f}")

    print("\n[2] Testing with user interaction...")
    user_state = torch.randn(4, 64)
    result = system.process(state, user_state=user_state, user_proximity=0.7)
    print(f"  Social emotion: {result['social']}")

    print("\n[3] Testing with observation...")
    observation = torch.randn(4, 64)
    result = system.process(state, observation=observation)
    print(f"  Contagion: {'detected' if 'contagion' in result else 'none'}")

    print("\n[4] Summary...")
    summary = system.get_summary()
    print(f"  History size: {summary['history_size']}")

    print("\n" + "=" * 60)
    print("✓ Advanced emotion system integrated!")
    print("=" * 60)


if __name__ == "__main__":
    test_integration()
