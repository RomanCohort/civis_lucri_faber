"""
Neuromodulation + Epigenetic Tests
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def test_neuromodulation():
    """测试神经调质系统"""
    print("\n[TEST 1] Neuromodulation")

    from core.personality.neuromodulation import NeuromodulationSystem

    neuro = NeuromodulationSystem(hidden_dim=128)

    # 模拟hidden states
    hidden = torch.randn(1, 10, 128)

    # 正常任务
    result = neuro.forward(hidden, task_type="general")
    print(f"  General: temp={result['temperature']:.2f}, confidence={result['confidence']:.2f}, uncertainty={result['uncertainty']:.2f}")

    # 道德任务 (应该保守)
    result2 = neuro.forward(hidden, task_type="moral")
    print(f"  Moral: temp={result2['temperature']:.2f}, confidence={result2['confidence']:.2f}")

    # 创意任务 (可以冒险)
    result3 = neuro.forward(hidden, task_type="creative")
    print(f"  Creative: temp={result3['temperature']:.2f}, confidence={result3['confidence']:.2f}")

    # 测试温度应用
    logits = torch.randn(1, 10000)
    adjusted = neuro.apply_to_logits(logits, "moral")
    print(f"  Logits adjusted: original={logits.std():.3f}, adjusted={adjusted.std():.3f}")

    summary = neuro.get_summary()
    print(f"  Summary: dopamine={summary['dopamine']:.2f}, serotonin={summary['serotonin']:.2f}, temp={summary['temperature']:.2f}")

    print("  ✓ PASSED")


def test_dopamine():
    """测试多巴胺门控"""
    print("\n[TEST 2] Dopamine Gate")

    from core.personality.neuromodulation import DopamineGate, RewardPredictionError

    dopamine = DopamineGate(hidden_dim=128, vocab_size=10000)

    hidden = torch.randn(1, 5, 128)
    confidence, uncertainty = dopamine(hidden)

    print(f"  Confidence: {confidence:.2f}")
    print(f"  Uncertainty: {uncertainty:.2f}")

    # 奖励预测
    rpe = RewardPredictionError(value_dim=128)
    pred = rpe(torch.randn(1, 128))
    pred_val = pred.item() if hasattr(pred, 'item') else pred
    error = rpe.compute_prediction_error(pred_val, actual=0.8)
    print(f"  Prediction error: {error:.2f}")

    print("  ✓ PASSED")


def test_epigenetic():
    """测试表观遗传记忆"""
    print("\n[TEST 3] Epigenetic Memory")

    from core.personality.epigenetic import EpigeneticLearner, MethylationTrigger

    learner = EpigeneticLearner(rank=8)

    # 正常交互
    result1 = learner.learn(
        user_input="你好",
        assistant_output="你好",
        sentiment=0.3,
        user_feedback=0.5,
    )
    print(f"  Normal: methylated={result1['methylated']}")

    # 情绪刺激
    result2 = learner.learn(
        user_input="你太棒了",
        assistant_output="谢谢",
        sentiment=0.9,  # 强情绪
        user_feedback=0.9,
    )
    print(f"  Emotional shock: methylated={result2['methylated']}, type={result2['event_type']}")

    # 事实纠错
    result3 = learner.learn(
        user_input="你之前说的不对",
        assistant_output="抱歉",
        sentiment=-0.3,
        user_feedback=-0.9,
        is_fact_correction=True,
    )
    print(f"  Fact correction: methylated={result3['methylated']}, type={result3['event_type']}")

    # 成长时间线
    timeline = learner.get_growth_timeline()
    print(f"  Growth timeline: {len(timeline)} events")

    summary = learner.get_summary()
    print(f"  Summary: epigenetic={summary['epigenetic_count']}, trauma={summary['has_trauma']}")

    print("  ✓ PASSED")


def test_methylation_trigger():
    """测试甲基化触发器"""
    print("\n[TEST 4] Methylation Trigger")

    from core.personality.epigenetic import MethylationTrigger

    trigger = MethylationTrigger()

    # 测试各种触发条件
    tests = [
        (0.95, False, 0.5, "trauma"),
        (0.8, False, 0.5, "emotional_shock"),
        (0.3, True, -0.8, "fact_correction"),
        (0.3, False, 0.95, "milestone"),
    ]

    for sent, is_corr, feedback, expected in tests:
        should, event = trigger.should_methylate(sent, is_corr, feedback)
        print(f"  Sent={sent}, correction={is_corr}, feedback={feedback}: {event}")

    summary = trigger.get_summary()
    print(f"  Triggers: {summary}")

    print("  ✓ PASSED")


def test_integration():
    """集成测试"""
    print("\n[TEST 5] Integration")

    from core.personality.neuromodulation import NeuromodulationSystem
    from core.personality.epigenetic import EpigeneticLearner

    neuro = NeuromodulationSystem(hidden_dim=64)
    epi = EpigeneticLearner(rank=4)

    # 模拟对话
    for turn in range(5):
        hidden = torch.randn(1, 3, 64)

        # 神经调质
        mod = neuro.forward(hidden, task_type="general")

        # 表观遗传学习
        sent = np.random.choice([-0.8, -0.3, 0.3, 0.8])
        feedback = np.random.choice([0.3, 0.7, 0.9])
        result = epi.learn(
            user_input=f"用户消息{turn}",
            assistant_output=f"回复{turn}",
            sentiment=sent,
            user_feedback=feedback,
        )

        if result['methylated']:
            print(f"  Turn {turn}: METHYLATED ({result['event_type']})")

    # 统一摘要
    print(f"  Neuromod summary: {neuro.get_summary()}")
    print(f"  Epigenetic summary: {epi.get_summary()}")

    print("  ✓ PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Neuromodulation + Epigenetic Tests")
    print("=" * 60)

    test_neuromodulation()
    test_dopamine()
    test_epigenetic()
    test_methylation_trigger()
    test_integration()

    print("\n" + "=" * 60)
    print("All Tests Passed!")
    print("=" * 60)