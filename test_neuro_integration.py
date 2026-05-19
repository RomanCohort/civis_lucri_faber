"""神经调制集成测试"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulacrum.core.neuromodulation_integration import (
    NeuromodulationIntegration,
    RewardModulation,
    TemporalDiscount,
    AttentionModulator,
    create_neuromodulation_integration,
)


def test_reward_modulation():
    """测试奖赏调制"""
    print("=== 测试 RewardModulation (多巴胺) ===")

    dopamine = RewardModulation(gamma=0.95, baseline_alpha=0.01)

    # 模拟状态
    for i in range(5):
        state = torch.tensor(np.random.randn(64), dtype=torch.float32)
        reward = np.random.randn(1).item() * 2  # 放大奖励

        # 计算TD误差
        next_state = torch.tensor(np.random.randn(64), dtype=torch.float32)
        td_error = dopamine.compute_td_error(reward, state, next_state)

        # 多巴胺信号
        dop_signal = dopamine.get_dopamine_signal(td_error)

        print(f"Step {i+1}: reward={reward:.2f}, td_error={td_error:.2f}, dopamine={dop_signal:.2f}")

    print(f"Baseline: {dopamine.baseline:.3f}")
    print("[PASS] RewardModulation test\n")


def test_temporal_discount():
    """测试时间折扣"""
    print("=== 测试 TemporalDiscount (血清素) ===")

    discount = TemporalDiscount(base_gamma=0.99, min_gamma=0.5)

    # 测试不同不确定性下的gamma
    uncertainties = [0.0, 0.2, 0.5, 0.8, 1.0]
    for u in uncertainties:
        gamma = discount.compute_gamma(uncertainty=u)
        serotonin = discount.compute_serotonin_signal(reward_std=0.1)
        print(f"Uncertainty={u:.1f}: gamma={gamma:.3f}, serotonin={serotonin:.2f}")

    # 更新波动性
    rewards = [1.0, -0.5, 2.0, 0.5, -1.0]
    for r in rewards:
        discount.update_volatility(r)

    print(f"Reward volatility: {discount.reward_volatility:.3f}")

    # 计算折扣返回
    gamma = 0.9
    discounted = discount.get_discounted_return(rewards, gamma)
    print(f"Discounted return (gamma={gamma}): {discounted:.3f}")

    print("[PASS] TemporalDiscount test\n")


def test_attention_modulator():
    """测试注意力调制"""
    print("=== 测试 AttentionModulator (乙酰胆碱) ===")

    acetylcholine = AttentionModulator(hidden_dim=64)

    # 测试不同新奇度下的聚焦
    novelties = [0.0, 0.3, 0.6, 1.0]
    for n in novelties:
        hidden = torch.randn(1, 64)
        focus = acetylcholine.compute_focus(hidden, novelty=n)
        print(f"Novelty={n:.1f}: focus={focus:.2f}")

    # 应用门控
    hidden = torch.randn(4, 10, 64)
    acetylcholine.focus_strength = 0.3
    gated = acetylcholine.apply_gating(hidden)
    print(f"Gate applied: {hidden.numel()} -> {gated.numel()} (non-zero: {(gated != 0).sum()})")

    print("[PASS] AttentionModulator test\n")


def test_integration():
    """测试完整集成"""
    print("=== 测试 NeuromodulationIntegration ===")

    neuro = create_neuromodulation_integration(hidden_dim=64, gamma=0.95)

    # 模拟轨迹
    for step in range(10):
        state = np.random.randn(64)
        action = f"action_{step}"
        reward = np.random.randn(1).item()

        # 随机next_state
        next_state = np.random.randn(64) if step < 9 else None

        result = neuro.step(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            uncertainty=0.3,
            novelty=0.5 if step % 2 == 0 else 0.1,
        )

        if step % 3 == 0:
            print(f"Step {step}: dopamine={result['dopamine']:.2f}, "
                  f"serotonin={result['serotonin']:.2f}, "
                  f"acetylcholine={result['acetylcholine']:.2f}")

    # 测试增强奖励
    enhanced = neuro.get_increased_reward(base_reward=1.0, consolidation_bonus=0.3)
    print(f"Enhanced reward: 1.0 -> {enhanced:.2f}")

    # 测试学习率调整
    lr = neuro.get_learning_rate_adjustment(base_lr=0.001)
    print(f"Adjusted LR: 0.001 -> {lr:.4f}")

    # 摘要
    summary = neuro.get_summary()
    print(f"Summary: {summary}")

    print("[PASS] NeuromodulationIntegration test\n")


def test_td_learning_signal():
    """测试TD学习信号"""
    print("=== 测试 TD Learning Signal ===")

    dopamine = RewardModulation(gamma=0.99)

    # 简单轨迹
    states = [torch.tensor(np.random.randn(64), dtype=torch.float32) for _ in range(5)]
    rewards = [1.0, 1.0, 1.0, 1.0, 10.0]  # 最后大奖

    cumulative = 0.0
    for i in range(len(rewards)):
        reward = rewards[i]
        next_state = states[i+1] if i < len(rewards)-1 else None

        td_error = dopamine.compute_td_error(
            reward,
            states[i],
            next_state
        )
        dop_signal = dopamine.get_dopamine_signal(td_error)

        print(f"Step {i+1}: reward={reward:.1f}, td_error={td_error:.2f}, dopamine={dop_signal:.2f}")

    print("[PASS] TD Learning Signal test\n")


if __name__ == "__main__":
    test_reward_modulation()
    test_temporal_discount()
    test_attention_modulator()
    test_integration()
    test_td_learning_signal()
    print("=" * 50)
    print("All neuromodulation integration tests passed!")