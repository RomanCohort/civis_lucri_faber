"""神经递质系统测试"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from civis_lucri_faber.core.neurotransmitter import (
    NeurotransmitterSystem,
    DopamineSystem,
    SerotoninSystem,
    AcetylcholineSystem,
    create_neurotransmitter_system,
)


def test_dopamine():
    """测试多巴胺系统"""
    print("=== Test Dopamine System ===")

    da = DopamineSystem(baseline=0.5)

    # 奖励信号
    for i in range(5):
        reward = np.random.randn()
        expectation = 0.0
        signal = da.compute_reward_signal(reward, expectation)
        print(f"Reward={reward:.2f}: dopamine={signal:.3f}")

    # 运动信号
    motor = da.compute_motor_signal(0.8)
    print(f"Motor signal: {motor:.3f}")

    # 认知信号
    cog = da.compute_cognitive_signal(0.5)
    print(f"Cognitive signal: {cog:.3f}")

    print(f"Summary: {da.get_summary()}")
    print("[PASS] Dopamine test\n")


def test_serotonin():
    """测试血清素系统"""
    print("=== Test Serotonin System ===")

    st = SerotoninSystem(baseline=0.5)

    # 情绪
    for i in range(5):
        reward = np.random.randn() * 0.5
        if np.random.random() > 0.5:
            mood = st.compute_mood(reward, 0)
        else:
            mood = st.compute_mood(0, abs(reward))
        print(f"Reward={reward:.2f}: mood={st.mood:.3f}")

    # 睡眠信号
    for hour in [6, 12, 18, 24]:
        sleep = st.compute_sleep_signal(hour)
        print(f"Hour {hour}: sleep_wake={sleep:.3f}")

    print("[PASS] Serotonin test\n")


def test_acetylcholine():
    """测试乙酰胆碱"""
    print("=== Test Acetylcholine System ===")

    ach = AcetylcholineSystem()

    # 注意力
    for i in range(3):
        novelty = np.random.random()
        salience = np.random.random()
        att = ach.compute_attention(novelty, salience)
        print(f"Novelty={novelty:.2f}, Salience={salience:.2f}: attention={att:.3f}")

    # 记忆巩固
    mem = ach.compute_memory_consolidation(0.7)
    print(f"Memory consolidation: {mem:.3f}")
    print("[PASS] Acetylcholine test\n")


def test_integration():
    """测试整合系统"""
    print("=== Test Neurotransmitter System ===")

    nt = create_neurotransmitter_system()

    # 模拟场景
    scenarios = [
        {'reward': 1.0, 'novelty': 0.8, 'salience': 0.6, 'threat': 0.1},
        {'reward': -0.5, 'novelty': 0.2, 'salience': 0.3, 'threat': 0.8},
        {'reward': 0.5, 'novelty': 0.5, 'salience': 0.4, 'threat': 0.2},
    ]

    for s in scenarios:
        result = nt.step(**s)
        print(f"State: {result['state']}, Motivation: {result['motivation']:.2f}, Arousal: {result['arousal']:.2f}")
        print(f"  DA={result['dopamine']:.2f}, 5-HT={result['serotonin']:.2f}, ACh={result['acetylcholine']:.2f}")
        print(f"  NE={result['norepinephrine']:.2f}, EP={result['endorphin']:.2f}")

    # 调制
    att_mod = nt.get_attention_modulation(0.5)
    learn_mod = nt.get_learning_modulation()
    motor_mod = nt.get_motor_modulation()

    print(f"Attention modulation: {att_mod:.3f}")
    print(f"Learning modulation: {learn_mod:.3f}")
    print(f"Motor modulation: {motor_mod:.3f}")

    # E/I平衡
    e_i = nt.get_e_i_balance()
    print(f"E/I balance: {e_i:.3f}")

    print(f"\nSummary: {nt.get_summary()}")
    print("[PASS] Integration test\n")


def test_stress_response():
    """测试应激反应"""
    print("=== Test Stress Response ===")

    nt = create_neurotransmitter_system()

    # 无威胁
    result = nt.step(threat=0.1)
    print(f"No threat: {result['state']}")

    # 高威胁
    result = nt.step(threat=0.9)
    print(f"High threat: {result['state']}, NE={result['norepinephrine']:.2f}")

    # 恢复
    for i in range(3):
        result = nt.step(threat=0.0)
    print(f"After recovery: {result['state']}")
    print("[PASS] Stress test\n")


def test_reward_learning():
    """测试奖励学习"""
    print("=== Test Reward Learning ===")

    nt = create_neurotransmitter_system()

    expectations = [0.5, 0.7, 0.9]  # 期望递增
    rewards = [1.0, 0.3, 0.6]

    for i, (exp, rew) in enumerate(zip(expectations, rewards)):
        result = nt.step(reward=rew, expectation=exp)
        rpe = rew - exp  # Reward Prediction Error
        print(f"Exp={exp:.1f}, Rew={rew:.1f}: RPE={rpe:.2f}, DA={result['dopamine']:.2f}, State={result['state']}")

    print("[PASS] Reward Learning test\n")


if __name__ == "__main__":
    test_dopamine()
    test_serotonin()
    test_acetylcholine()
    test_integration()
    test_stress_response()
    test_reward_learning()
    print("=" * 50)
    print("All neurotransmitter tests passed!")