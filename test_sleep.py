"""睡眠系统测试"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from civis_lucri_faber.core.sleep import (
    SleepSystem,
    SleepController,
    SleepStage,
    MemoryReplayer,
    create_sleep_system,
)


def test_sleep_controller():
    """测试睡眠控制器"""
    print("=== 测试 SleepController ===")

    controller = SleepController(
        awake_to_sleep_threshold=0.2,
        sleep_cycle_duration=10,  # 短周期便于测试
        rem_duration=2,
        nrem2_duration=4,
        nrem3_duration=3,
        enable_dream=True,
    )

    # 模拟清醒时的疲惫积累
    for i in range(5):
        controller.update_fatigue(info_gain_reward=0.0, step_duration=1.0)  # 没有探索时
        print(f"Step {i+1}: fatigue={controller.fatigue:.2f}, should_sleep={controller.should_sleep()}")

    # 应该入睡了
    assert controller.should_sleep(), "Should enter sleep when fatigue > threshold"

    # 进入睡眠
    controller.enter_sleep()
    print(f"Entered sleep: stage={controller.current_cycle.current_stage.value}")

    # 模拟睡眠周期
    for i in range(15):
        stage = controller.step()
        if i % 3 == 0:
            print(f"  Cycle step {i+1}: stage={stage.value}, consolidation={controller.get_consolidation_bonus():.2f}")

    # 测试突触缩减
    downscale = controller.get_synaptic_downscale_factor()
    print(f"Synaptic downscale: {downscale:.2f}")

    # 摘要
    summary = controller.get_summary()
    print(f"Summary: {summary['current_stage']}, cycles={summary['total_cycles']}")

    print("[PASS] SleepController test\n")


def test_memory_replayer():
    """测试记忆回放"""
    print("=== 测试 MemoryReplayer ===")

    replayer = MemoryReplayer(priority_alpha=0.6, batch_size=4)

    # 添加经验
    for i in range(10):
        state = np.random.randn(64)
        action = f"action_{i}"
        reward = np.random.randn(1).item()
        next_state = np.random.randn(64)
        replayer.add_experience(state, action, reward, next_state)

    print(f"Buffer size: {len(replayer.replay_buffer)}")

    # 采样
    samples = replayer.sample(n=3)
    print(f"Sampled: {len(samples)} memories")
    for s in samples:
        print(f"  priority={s.priority:.3f}, reward={s.reward:.2f}")

    # 统计
    stats = replayer.get_statistics()
    print(f"Stats: {stats}")

    print("[PASS] MemoryReplayer test\n")


def test_sleep_system():
    """测试完整睡眠系统"""
    print("=== 测试 SleepSystem ===")

    sleep_sys = create_sleep_system(
        enable_sleep=True,
        enable_dream=True,
        sleep_threshold=0.6,
    )

    # 模拟一些步骤
    for step in range(20):
        # 添加经验
        state = np.random.randn(64)
        action = f"action_{step}"
        reward = np.random.randn(1).item()
        next_state = np.random.randn(64)
        sleep_sys.add_experience(state, action, reward, next_state)

        # 更新
        info_gain = 0.2 if step < 10 else 0.0  # 前10步有探索，后10步空闲
        result = sleep_sys.update(info_gain_reward=info_gain, step_duration=1.0)

        if step % 5 == 0:
            print(f"Step {step}: stage={result['stage']}, sleeping={result['is_sleeping']}")
            if 'consolidation_bonus' in result and result['consolidation_bonus'] > 0:
                print(f"  Consolidation bonus: {result['consolidation_bonus']:.2f}")
            if result.get('dream'):
                print(f"  Dream: {result['dream'][:50]}...")

    # 摘要
    summary = sleep_sys.get_summary()
    print(f"\nSummary: {summary['controller']['current_stage']}, cycles={summary['controller']['total_cycles']}")

    print("[PASS] SleepSystem test\n")


def test_sleep_dream_generation():
    """测试梦境生成"""
    print("=== 测试梦境生成 ===")

    from civis_lucri_faber.core.sleep import MemoryReplay, SleepController

    # 创建一些记忆
    memories = [
        MemoryReplay(
            state=np.random.randn(64),
            action=f"action_{i}",
            reward=np.random.randn(1).item(),
            next_state=np.random.randn(64),
            priority=np.random.random(),
            timestamp=i,
        )
        for i in range(5)
    ]

    controller = SleepController(enable_dream=True, dream_creativity=0.8)
    controller.enter_sleep()
    controller.current_cycle.current_stage = SleepStage.REM

    # 生成梦境
    dream = controller.generate_dream(memories)
    print(f"Generated dream: {dream}")

    assert dream is not None, "Dream should be generated in REM stage"
    print("[PASS] Dream generation test\n")


if __name__ == "__main__":
    test_sleep_controller()
    test_memory_replayer()
    test_sleep_system()
    test_sleep_dream_generation()
    print("=" * 50)
    print("All sleep system tests passed!")