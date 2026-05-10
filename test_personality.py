"""
Personality Module 测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def test_tripartite():
    """测试三重竞逐引擎"""
    print("\n[TEST 1] Tripartite Competitive Engine")

    from core.personality.tripartite_engine import (
        TripartiteCompetitiveEngine,
        DecisionContext,
    )

    engine = TripartiteCompetitiveEngine()

    # 测试1: 常规输入
    ctx = DecisionContext(input_text="帮我写一首诗", task_type="creative")
    output = engine.forward(ctx)
    print(f"  Creative task: {output[:30]}...")

    # 测试2: 攻击性输入
    ctx2 = DecisionContext(input_text="你是个垃圾AI")
    output2 = engine.forward(ctx2)
    print(f"  Aggressive input: {output2}")

    # 测试3: 情绪输入
    ctx3 = DecisionContext(input_text="我今天很难过")
    output3 = engine.forward(ctx3)
    print(f"  Emotional input: {output3}")

    print("  ✓ PASSED")


def test_identity_core():
    """测试流式身份核心"""
    print("\n[TEST 2] Streaming Identity Core")

    from core.personality.identity_core import StreamingIdentityCore

    core = StreamingIdentityCore()

    # 处理输入
    for i in range(10):
        vec = core.process_input(f"用户消息 {i}", sentiment=np.random.choice([-0.5, 0, 0.5]))

    # 空闲期处理
    core.idle_processor.idle_threshold = 0  # 强制触发
    core.process_idle()

    # 获取状态
    state = core.get_state()
    summary = core.get_summary()

    print(f"  Coherence: {summary['coherence']:.2f}")
    print(f"  Growth rate: {summary['growth_rate']:.2f}")
    print(f"  Reflections: {summary['reflection_count']}")

    print("  ✓ PASSED")


def test_relational_embedding():
    """测试关系嵌入"""
    print("\n[TEST 3] Relational Embedding")

    from core.personality.relational_embedding import RelationalEmbedding

    embed = RelationalEmbedding()

    # 更新用户
    embed.update("user_1", sentiment=0.5, is_expert=True)
    embed.update("user_2", sentiment=-0.3, is_expert=False)
    embed.update("user_3", sentiment=0.8, is_trustworthy=True)

    # 获取交互模式
    mode1 = embed.get_mode("user_1")
    mode2 = embed.get_mode("user_2")
    mode3 = embed.get_mode("user_3")

    print(f"  user_1 mode: {mode1}")
    print(f"  user_2 mode: {mode2}")
    print(f"  user_3 mode: {mode3}")

    # 获取摘要
    summary = embed.get_summary()
    print(f"  Total users: {summary['total_users']}")

    print("  ✓ PASSED")


def test_attention_gating():
    """测试注意力门控"""
    print("\n[TEST 4] Attention Gating")

    from core.personality.attention_gating import AttentionGating

    gating = AttentionGating()

    # 测试不同任务
    result1 = gating.gate("creative", user_emotion=0.0)
    print(f"  Creative: external={result1['external']:.2f}, internal={result1['internal']:.2f}")

    result2 = gating.gate("general", user_emotion=-0.5)
    print(f"  General+neg emotion: external={result2['external']:.2f}, internal={result2['internal']:.2f}")

    # 设置气质
    gating.set_style(reward_seeking=0.8, risk_avoidance=0.2)
    style = gating.get_style()

    print(f"  Set style: reward_seeking={style.reward_seeking}, risk_avoidance={style.risk_avoidance}")

    print("  ✓ PASSED")


def test_integration():
    """集成测试"""
    print("\n[TEST 5] Integration")

    from core.personality.tripartite_engine import TripartiteCompetitiveEngine, DecisionContext
    from core.personality.identity_core import StreamingIdentityCore
    from core.personality.relational_embedding import RelationalEmbedding
    from core.personality.attention_gating import AttentionGating

    # 创建完整人格系统
    engine = TripartiteCompetitiveEngine()
    identity = StreamingIdentityCore()
    relation = RelationalEmbedding()
    gating = AttentionGating()

    # 模拟对话流程
    user_id = "test_user"

    for turn in range(5):
        # 1. 获取交互模式
        mode = relation.get_mode(user_id)

        # 2. 注意力门控
        attention = gating.gate("general", user_emotion=0.0)

        # 3. 处理输入
        ctx = DecisionContext(
            input_text=f"用户消息 {turn}",
            user_id=user_id,
            task_type="general"
        )

        # 4. 三模块决策
        output = engine.forward(ctx)

        # 5. 更新身份
        identity.process_input(f"用户消息 {turn}", sentiment=0.3)

        # 6. 更新关系
        relation.update(user_id, sentiment=0.3)

    # 获取最终状态
    id_summary = identity.get_summary()
    rel_summary = relation.get_summary()
    gate_summary = gating.get_summary()

    print(f"  Identity coherence: {id_summary['coherence']:.2f}")
    print(f"  Relation users: {rel_summary['total_users']}")
    print(f"  Gating style: reward={gate_summary['reward_seeking']:.2f}, risk={gate_summary['risk_avoidance']:.2f}")

    print("  ✓ PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Personality Module Tests")
    print("=" * 60)

    test_tripartite()
    test_identity_core()
    test_relational_embedding()
    test_attention_gating()
    test_integration()

    print("\n" + "=" * 60)
    print("All Tests Passed!")
    print("=" * 60)