"""新脑区系统测试"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from civis_lucri_faber.core.basal_ganglia import (
    BasalGanglia,
    BasalGangliaSystem,
    create_basal_ganglia,
)
from civis_lucri_faber.core.hippocampus import (
    Hippocampus,
    create_hippocampus,
)
from civis_lucri_faber.core.limbic import (
    Amygdala,
    Thalamus,
    LimbicSystem,
    create_limbic_system,
)


def test_basal_ganglia():
    """测试基底神经节"""
    print("=== Test Basal Ganglia ===")

    bg = create_basal_ganglia(state_dim=64, n_actions=4)

    # 选择动作
    state = torch.randn(1, 64)
    result = bg(state, epsilon=0.1)

    print(f"Action: {result['action']}")
    print(f"Q-values: {result['q_values'].round(3)}")
    print(f"Dopamine: {result['dopamine']:.3f}")
    print(f"Habit: {result['habit_strength']:.3f}")

    # 更新
    next_state = torch.randn(1, 64)
    update_result = bg.update(state, result['action'], 1.0, next_state)
    print(f"TD error: {update_result['td_error']:.3f}")

    # 习惯形成
    for i in range(10):
        bg.learn_habit(torch.randn(1, 64), 0)

    print(f"Habit strength: {bg.bg.get_habit_strength():.3f}")
    print("[PASS] Basal Ganglia test\n")


def test_hippocampus():
    """测试海马体"""
    print("=== Test Hippocampus ===")

    hip = create_hippocampus(input_dim=64, encoding_dim=128)

    # 编码记忆
    state = np.random.randn(64)
    encoding = hip.encode_memory(state, "move_forward", 1.0)
    print(f"Encoding shape: {encoding.shape}")

    # 编码多个
    for i in range(5):
        s = np.random.randn(64)
        hip.encode_memory(s, f"action_{i}", np.random.randn())

    print(f"Memory count: {len(hip.episodic_memory)}")

    # 检索
    retrieved = hip.retrieve(state, top_k=3)
    print(f"Retrieved: {len(retrieved)} memories")

    # 正向回放
    replay = hip.replay_forward()
    print(f"Forward replay: {len(replay)}")

    # 想象未来
    current = torch.randn(1, 64)
    future = hip.imagine_future(current, n_steps=3)
    print(f"Future想象: {len(future)} steps")

    print(f"Summary: {hip.get_summary()}")
    print("[PASS] Hippocampus test\n")


def test_limbic():
    """测试边缘系统"""
    print("=== Test Limbic System (Amygdala + Thalamus) ===")

    limbic = create_limbic_system(input_dim=64)

    # 情绪处理
    state = torch.randn(1, 64)
    result = limbic(state)

    print(f"Emotion: {result['emotion']}")
    print(f"Valence: {result['valence']:.3f}")
    print(f"Arousal: {result['arousal']:.3f}")
    print(f"Response: {result['response']}")
    print(f"Emotional attention: {result['emotional_attention']:.3f}")

    # 丘脑时间信息
    sensory = [torch.randn(1, 64) for _ in range(4)]
    thalamus_result = limbic.thalamus.process(sensory, state)
    print(f"Thalamus timing: {thalamus_result.get('temporal_features') is not None}")

    print(f"Summary: {limbic.get_summary()}")
    print("[PASS] Limbic test\n")


def test_amygdala_fear():
    """测试恐惧条件化"""
    print("=== Test Amygdala Fear Conditioning ===")

    from civis_lucri_faber.core.limbic import FearConditioning, AmygdalaNucleus

    fear = FearConditioning()
    amygdala = AmygdalaNucleus(64)

    # 学习恐惧
    cue = np.random.randn(64)
    fear.learn_fear(cue, -1.0)  # 负性刺激

    # 测试
    fear_level = fear.detect_fear(cue)
    print(f"Fear level: {fear_level:.3f}")

    # 情绪评估
    state = torch.randn(1, 64)
    emotion = amygdala(state)
    print(f"Emotion: {emotion['emotion']}, valence: {emotion['valence']:.3f}")
    print("[PASS] Fear test\n")


def test_hippocampus_spatial():
    """测试空间表征"""
    print("=== Test Hippocampus Spatial ===")

    hip = create_hippocampus(input_dim=64)

    # 直接测试CA3关联功能
    hip.encode_memory(np.random.randn(64), "location_A", 1.0)
    hip.encode_memory(np.random.randn(64), "location_B", 0.5)
    hip.encode_memory(np.random.randn(64), "transition_AB", 2.0)

    hip.link_episodes(0, 1)
    hip.link_episodes(1, 2)

    print("Linked episodes created")
    print("[PASS] Spatial test\n")


def test_integration():
    """测试完整整合"""
    print("=== Test Brain Integration ===")

    # 创建各系统
    bg = create_basal_ganglia(64, 4)
    hip = create_hippocampus(64)
    limbic = create_limbic_system(64)

    # 模拟经验
    state = torch.randn(1, 64)
    state_np = state.numpy()[0]

    # 1. 边缘系统处理情绪
    emotion_result = limbic(state)
    print(f"Valence: {emotion_result['valence']:.3f}")

    # 2. 海马体编码
    hip.encode_memory(state_np, "explore", emotion_result['valence'])

    # 3. 基底神经节选择动作
    bg_result = bg(state, epsilon=0.1)
    print(f"Action: {bg_result['action']}")

    # 4. 更新
    bg.update(state, bg_result['action'], emotion_result['valence'], torch.randn(1, 64))

    # 摘要
    print(f"BG: {bg.get_summary()['habit_strength']:.3f}")
    print(f"Hip: {hip.get_summary()['memory_count']}")
    print(f"Limbic: {limbic.get_summary()}")
    print("[PASS] Integration test\n")


if __name__ == "__main__":
    test_basal_ganglia()
    test_hippocampus()
    test_limbic()
    test_amygdala_fear()
    test_hippocampus_spatial()
    test_integration()
    print("=" * 50)
    print("All brain system tests passed!")