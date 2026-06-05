"""
前额叶皮质测试 - Prefrontal Cortex Tests

验证执行功能中枢的五个子系统：
1. 成熟度系统 — 发育过程对认知能力的影响
2. 成本收益分析 — 多维度权衡利弊
3. 冲动抑制 — 门控与冲动爆发
4. 长期规划 — 目标层级 + 前瞻模拟
5. 工作记忆 — 门控写入 + 注意力读取
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import importlib.util

# 直接导入模块文件
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "prefrontal_cortex",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "prefrontal_cortex.py")
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

PrefrontalCortex = _mod.PrefrontalCortex
MaturationTracker = _mod.MaturationTracker


def test_maturation():
    """成熟度系统：验证发育过程"""
    print("\n" + "=" * 60)
    print("测试: 成熟度系统")
    print("=" * 60)

    pfc = PrefrontalCortex(input_dim=64, maturation_tau=500.0)

    # 初始状态（不成熟）
    summary = pfc.maturation.get_summary()
    print(f"  初始成熟度: {summary['maturity']:.4f}")
    print(f"  初始抑制能力: {summary['inhibition_capacity']:.4f}")
    print(f"  初始规划深度: {summary['planning_depth']}")
    print(f"  初始时间折扣: {summary['temporal_discount']:.4f}")
    print(f"  初始冲动权重: {summary['impulsivity_weight']:.4f}")

    assert summary['maturity'] < 0.01, "初始应该接近0"
    assert summary['planning_depth'] == 1, "不成熟时规划深度应为1"
    assert summary['temporal_discount'] > 0.8, "不成熟时时间折扣应该高"

    # 推进大量步数（成熟）
    pfc.maturation.advance(3000)
    summary = pfc.maturation.get_summary()
    print(f"\n  3000步后成熟度: {summary['maturity']:.4f}")
    print(f"  3000步后抑制能力: {summary['inhibition_capacity']:.4f}")
    print(f"  3000步后规划深度: {summary['planning_depth']}")
    print(f"  3000步后时间折扣: {summary['temporal_discount']:.4f}")

    assert summary['maturity'] > 0.9, "长期后应该接近1"
    assert summary['planning_depth'] >= 4, "成熟时规划深度应该高"
    assert summary['temporal_discount'] < 0.4, "成熟时时间折扣应该低"
    print("  [PASS]")


def test_cost_benefit():
    """成本收益分析：验证多维度评估"""
    print("\n" + "=" * 60)
    print("测试: 成本收益分析")
    print("=" * 60)

    pfc = PrefrontalCortex(input_dim=64, maturation_tau=100.0)
    state = torch.randn(1, 64)

    # 不成熟状态
    pfc.maturation.advance(1)
    cb_immature = pfc.cost_benefit.evaluate(state, maturity=pfc.maturation.maturity,
                                             candidates=["explore", "exploit", "wait"])
    print(f"  不成熟时即时回报权重: {pfc.maturation.impulsivity_weight:.4f}")
    for r in cb_immature:
        print(f"    {r.action}: score={r.total_score:.4f} (imm={r.immediate_reward:.3f}, "
              f"ltv={r.long_term_value:.3f}, risk={r.risk:.3f}, cost={r.effort_cost:.3f})")

    # 成熟状态
    pfc.maturation.advance(1000)
    cb_mature = pfc.cost_benefit.evaluate(state, maturity=pfc.maturation.maturity,
                                           candidates=["explore", "exploit", "wait"])
    print(f"\n  成熟时即时回报权重: {pfc.maturation.impulsivity_weight:.4f}")
    for r in cb_mature:
        print(f"    {r.action}: score={r.total_score:.4f}")

    # 不成熟时应该更看重即时回报
    assert pfc.maturation.impulsivity_weight < 0.3, "成熟后冲动权重应低"
    print("  [PASS]")


def test_impulse_control():
    """冲动抑制：验证门控和冲动爆发"""
    print("\n" + "=" * 60)
    print("测试: 冲动抑制")
    print("=" * 60)

    pfc = PrefrontalCortex(input_dim=64, maturation_tau=100.0)
    state = torch.randn(1, 64)

    # 不成熟 + 强冲动 → 抑制应该弱
    pfc.maturation.advance(1)
    result_low = pfc.impulse_ctrl.gate(
        state, maturity=pfc.maturation.maturity,
        impulse_signals={"amygdala": 0.8}
    )
    print(f"  不成熟 + 强冲动: gate={result_low['gate']:.4f}, "
          f"effective_inh={result_low['effective_inhibition']:.4f}")

    # 成熟 + 强冲动 → 抑制应该强
    pfc.maturation.advance(1000)
    result_high = pfc.impulse_ctrl.gate(
        state, maturity=pfc.maturation.maturity,
        impulse_signals={"amygdala": 0.8}
    )
    print(f"  成熟 + 强冲动: gate={result_high['gate']:.4f}, "
          f"effective_inh={result_high['effective_inhibition']:.4f}")

    assert result_high['effective_inhibition'] > result_low['effective_inhibition'], \
        "成熟后抑制应该更强"

    # 测试冲动累积爆发
    pfc.impulse_ctrl.accumulated_impulse = 0.0
    for i in range(20):
        result = pfc.impulse_ctrl.gate(
            state, maturity=0.9,
            impulse_signals={"amygdala": 0.9, "habit": 0.8}
        )

    print(f"  连续强冲动后: accumulated={result['accumulated_impulse']:.4f}, "
          f"burst={result['burst']}, gate={result['gate']:.4f}")

    if result['burst']:
        print("  冲动爆发！gate=0，完全放行冲动")
    print("  [PASS]")


def test_planning():
    """长期规划：验证规划深度和目标管理"""
    print("\n" + "=" * 60)
    print("测试: 长期规划")
    print("=" * 60)

    pfc = PrefrontalCortex(input_dim=64, maturation_tau=200.0)
    state = torch.randn(1, 64)

    # 添加目标
    pfc.planner.add_goal("长期生存", priority=0.9)
    pfc.planner.add_goal("信息探索", priority=0.6)
    pfc.planner.add_goal("自我对齐", priority=0.7, deadline=100)

    # 不成熟规划
    pfc.maturation.advance(10)
    plan_immature = pfc.planner.plan(state, maturity=pfc.maturation.maturity)
    print(f"  不成熟 (maturity={pfc.maturation.maturity:.3f}):")
    print(f"    规划深度: {plan_immature['depth']}")
    print(f"    累积价值: {plan_immature['cumulative_value']:.4f}")
    print(f"    时间折扣: {plan_immature['temporal_discount']:.4f}")

    # 成熟规划
    pfc.maturation.advance(1000)
    plan_mature = pfc.planner.plan(state, maturity=pfc.maturation.maturity)
    print(f"\n  成熟 (maturity={pfc.maturation.maturity:.3f}):")
    print(f"    规划深度: {plan_mature['depth']}")
    print(f"    累积价值: {plan_mature['cumulative_value']:.4f}")
    print(f"    时间折扣: {plan_mature['temporal_discount']:.4f}")
    print(f"    最优先目标: {plan_mature['top_goal']}")

    assert plan_mature['depth'] > plan_immature['depth'], "成熟后规划应该更深"
    assert plan_mature['temporal_discount'] < plan_immature['temporal_discount'], \
        "成熟后时间折扣应该更低"

    # 更新目标进度
    pfc.planner.update_goal_progress("长期生存", 0.3)
    assert pfc.planner.goals[0].progress == 0.3, "目标进度应该更新"
    print("  [PASS]")


def test_working_memory():
    """工作记忆：门控写入 + 注意力读取"""
    print("\n" + "=" * 60)
    print("测试: 工作记忆")
    print("=" * 60)

    pfc = PrefrontalCortex(input_dim=64, wm_slots=7)

    # 初始为空
    print(f"  初始已用槽位: {pfc.working_memory.used_slots}")
    assert pfc.working_memory.used_slots == 0

    # 写入多个
    n_written = 0
    for i in range(10):
        x = torch.randn(1, 64)
        pfc.working_memory.write(x)
        if pfc.working_memory.used_slots > n_written:
            n_written = pfc.working_memory.used_slots

    print(f"  写入10次后已用槽位: {pfc.working_memory.used_slots} (最多7)")
    assert pfc.working_memory.used_slots <= 7, "不应超过Miller limit"

    # 读取
    query = torch.randn(1, 64)
    context = pfc.working_memory.read(query)
    print(f"  读取上下文形状: {context.shape}")
    assert context.shape == (1, 64), "读取应返回 [1, 64]"

    # 清空
    pfc.working_memory.clear()
    assert pfc.working_memory.used_slots == 0
    print("  清空后已用槽位: 0")
    print("  [PASS]")


def test_full_pfc():
    """完整前额叶前向传播"""
    print("\n" + "=" * 60)
    print("测试: 完整 PFC 前向传播")
    print("=" * 60)

    pfc = PrefrontalCortex(input_dim=64, num_actions=4, maturation_tau=200.0)
    state = torch.randn(1, 64)

    # 初始（不成熟）
    result = pfc(
        state=state,
        candidates=["explore", "exploit", "wait", "retreat"],
        impulse_signals={"amygdala": 0.7, "basal_ganglia": 0.5},
        emotion_valence=-0.3,
        dopamine_level=0.6,
    )

    print(f"  不成熟 (maturity={result['maturity']:.4f}):")
    print(f"    行动: {result['action'].item()} ({pfc.get_decision_explanation(result['action'].item())})")
    print(f"    价值: {result['value'].item():.4f}")
    print(f"    抑制门控: {result['inhibition_gate']:.4f}")
    print(f"    规划深度: {result['planning_depth']}")
    print(f"    成本收益: {len(result['cost_benefit'])} 个候选")
    for cb in result['cost_benefit']:
        print(f"      {cb.action}: score={cb.total_score:.4f}")

    # 推进到成熟
    for _ in range(50):
        result = pfc(
            state=torch.randn(1, 64),
            impulse_signals={"amygdala": 0.3},
            dopamine_level=0.5,
        )

    print(f"\n  成熟 (maturity={result['maturity']:.4f}):")
    print(f"    抑制门控: {result['inhibition_gate']:.4f}")
    print(f"    规划深度: {result['planning_depth']}")
    print(f"    累积冲动: {result['accumulated_impulse']:.4f}")

    summary = pfc.get_summary()
    print(f"\n  系统摘要:")
    print(f"    成熟度: {summary['maturation']['maturity']:.4f}")
    print(f"    工作记忆: {summary['working_memory']['used_slots']}/{summary['working_memory']['total_slots']} slots")
    print(f"    目标数: {summary['planner']['n_goals']}")
    print("  [PASS]")


# ============ 运行 ============

if __name__ == "__main__":
    print("=" * 60)
    print("前额叶皮质测试 - Prefrontal Cortex Tests")
    print("=" * 60)

    tests = [
        test_maturation,
        test_cost_benefit,
        test_impulse_control,
        test_planning,
        test_working_memory,
        test_full_pfc,
    ]

    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed} passed, {failed} failed")
    print("=" * 60)
