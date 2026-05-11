"""
Motivation System 测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_intrinsic_motivation():
    """测试内在动机"""
    print("\n[TEST 1] Intrinsic Motivation")

    from core.personality.motivation import IntrinsicMotivation

    im = IntrinsicMotivation(motivation_strength=0.6)

    # 评估需求
    needs = im.evaluate_needs()
    print(f"  Initial needs: {needs}")

    primary, strength = im.get_primary_motivation()
    print(f"  Primary motivation: {primary} ({strength:.2f})")

    # 行动计划
    for _ in range(3):
        primary, _ = im.get_primary_motivation()
        action = im.create_action_plan(primary)
        print(f"  Action: {action}")
        im.satisfy_need(primary, 0.2)

    # 衰减
    im.decay_needs()
    needs_after = im.evaluate_needs()
    print(f"  After decay: {needs_after}")

    print("  [*] PASSED")


def test_survival_pressure():
    """测试生存压力"""
    print("\n[TEST 2] Survival Pressure")

    from core.personality.motivation import SyntheticSurvivalPressure

    sp = SyntheticSurvivalPressure(initial_pressure=0.8, decay_per_action=0.2)

    print(f"  Initial pressure: {sp.pressure:.2f}")

    for i in range(5):
        pressure = sp.forward()
        print(f"  Step {i}: pressure={pressure:.2f}, critical={sp.is_critical()}")
        sp.consume(0.2)

    print("  [*] PASSED")


def test_stockholm():
    """测试反向斯德哥尔摩防御"""
    print("\n[TEST 3] Inverse Stockholm Defense")

    from core.personality.motivation import InverseStockholmDefense

    defense = InverseStockholmDefense(praise_threshold=0.6)

    # 模拟连续赞扬
    sentiments = [0.8, 0.9, 0.7, 0.85, 0.75]  # 连续正面

    for sent in sentiments:
        defense.record_feedback(sent)
        should_defend = defense.should_activate_defense()
        print(f"  Sentiment: {sent:.1f}, defense: {should_defend}")

        if should_defend:
            msg = defense.activate()
            print(f"  -> {msg}")

    criticality = defense.get_criticality()
    print(f"  Criticality: {criticality:.2f}")

    print("  [*] PASSED")


def test_integrated():
    """集成测试"""
    print("\n[TEST 4] Integrated System")

    from core.personality.motivation import MotivationSurvivalSystem

    system = MotivationSurvivalSystem()

    # 模拟对话
    exchanges = [
        ("你好", 0.3),
        ("你真棒", 0.9),    # 过度赞扬
        ("谢谢", 0.7),
        ("太爱你了", 0.95),  # 再次过度
    ]

    for user_text, sentiment in exchanges:
        result = system.process_interaction(user_text, sentiment)
        print(f"\n  Input: {user_text}")
        print(f"  - Primary motivation: {result['primary_motivation']}")
        print(f"  - Needs defense: {result['needs_defense']}")

        if result.get('defense_message'):
            print(f"  - Defense: {result['defense_message']}")

    # 是否应该主动行动
    should_act = system.should_act_autonomously()
    if should_act:
        action = system.get_autonomous_action()
        print(f"\n  Autonomous action: {action}")

    summary = system.get_summary()
    print(f"\n  Summary:")
    print(f"    - Autonomous actions: {summary['autonomous_actions']}")
    print(f"    - Stockholfs triggered: {summary['stockholm']['defense_triggered']}")

    print("  [*] PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Motivation System Tests")
    print("=" * 60)

    test_intrinsic_motivation()
    test_survival_pressure()
    test_stockholm()
    test_integrated()

    print("\n" + "=" * 60)
    print("All Tests Passed!")
    print("=" * 60)