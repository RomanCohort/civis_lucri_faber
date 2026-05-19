"""实验六: 胶质淋巴系统的时间窗口验证 (Glymphatic Timing).

对应疾病: 阿尔茨海默症、睡眠障碍与毒性蛋白累积

机制链:
  高强度学习 → 代谢废物(brain_waste)累积 → GlialSystem清除
  → 清除时机影响: 持续清洗=灾难性遗忘; 不清洗=系统中毒
  → NREM3深度睡眠清除效率最高 (factor=2.0, 20x于清醒态)

3组对比:
  Group A (Continuous): 持续清洗 (无节律, 高clearance)
  Group B (Sleep-gated): 仅夜间/睡眠时清洗 (NREM3)
  Group C (Gamma-triggered): γ波(40Hz)间歇触发清洗

3 Phases:
  Phase 1 (Learning, 300步): 高强度学习产生废物
  Phase 2 (Clearance, 400步): 按组别策略清洗
  Phase 3 (Recall, 300步): 测试记忆保留 + 系统性能

测量:
  - Memory Retention: 清洗后记忆准确率
  - Toxicity Index: 未清理废物对性能的拖累
  - Clearance Efficiency: 各组清洗效率对比
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from typing import Dict, List

from simulacrum.core.agent import Simulacrum
from simulacrum.utils.config import Config


def read_metrics(agent: Simulacrum) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "brain_waste": float(s.get("brain_waste", 0.2)),
        "brain_health": float(s.get("brain_health", 0.8)),
        "neuroinflammation": float(s.get("neuroinflammation", 0.1)),
        "exploration_rate": float(agent.config.exploration_rate),
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "symptom_anhedonia": float(s.get("symptom_anhedonia", 0.0)),
        "balance": float(agent.thermo.balance),
        "active_ratio": float(s.get("active_ratio", 0.3)),
        "metabolic_cost": float(s.get("metabolic_cost", 0.0)),
        "plasticity_bdnf": float(s.get("plasticity_bdnf", 0.5)),
    }


def run_group(group_name: str, strategy: str) -> Dict:
    """运行一个实验组

    Args:
        group_name: 组名
        strategy: "continuous" | "sleep_gated" | "gamma"
    """
    print(f"\n{'='*60}")
    print(f"Group: {group_name} (strategy={strategy})")
    print(f"{'='*60}")

    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        seed=42,
    )
    agent = Simulacrum(config=config)

    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}
    waste_cleared_total = 0.0

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])

    # 记录Phase1结束时的"记忆基线" (用exploration_entropy近似)
    phase1_goals = []

    # ── Phase 1: Intensive Learning (300步) ──
    print("\n[Phase 1] INTENSIVE LEARNING — 高强度学习产生废物 (300步)")
    for step in range(300):
        # 高强度刺激 → 高神经活动 → 产生大量废物
        agent.step(user_input=None, external_stimulus=0.7)

        # 额外注入废物 (模拟高强度认知活动)
        current_waste = agent._internal_state.get("brain_waste", 0.2)
        agent._internal_state["brain_waste"] = min(1.0, current_waste + 0.002)

        record()
        phase1_goals.append(
            agent.current_goal.description if agent.current_goal else "none"
        )

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Waste={m['brain_waste']:.3f} "
                  f"Health={m['brain_health']:.3f} "
                  f"Inflam={m['neuroinflammation']:.3f} "
                  f"Explore={m['exploration_rate']:.4f}")

    # Phase1 baseline metrics
    phase1_waste = np.mean(history["brain_waste"][-50:])
    phase1_health = np.mean(history["brain_health"][-50:])
    phase1_explore = np.mean(history["exploration_rate"][-50:])

    # ── Phase 2: Clearance (400步) ──
    print(f"\n[Phase 2] CLEARANCE — 策略: {strategy} (400步)")

    # 记录清洗前的废物水平
    pre_clearance_waste = agent._internal_state.get("brain_waste", 0.2)

    for step in range(400):
        total_step = 300 + step

        if strategy == "continuous":
            # 持续清洗: 强制降低废物 (模拟无节律的高clearance)
            current_waste = agent._internal_state.get("brain_waste", 0.2)
            agent._internal_state["brain_waste"] = max(0.0, current_waste - 0.008)
            waste_cleared_total += 0.008

        elif strategy == "sleep_gated":
            # 仅在"睡眠"阶段清洗 (模拟SCN夜间 + NREM3)
            # 每10步一个"睡眠周期", 其中4步为NREM3
            cycle_pos = step % 10
            if cycle_pos < 4:  # NREM3 phase
                current_waste = agent._internal_state.get("brain_waste", 0.2)
                # NREM3: 高效清除 (factor=2.0)
                clearance = 0.016  # 2x normal
                agent._internal_state["brain_waste"] = max(0.0, current_waste - clearance)
                waste_cleared_total += clearance
            # 觉醒期不清洗 → 废物自然累积

        elif strategy == "gamma":
            # γ波(40Hz)间歇触发清洗
            # 40Hz = 每25ms一个周期, 模拟为每25步触发一次
            if step % 25 == 0:
                current_waste = agent._internal_state.get("brain_waste", 0.2)
                clearance = 0.012  # gamma-triggered burst
                agent._internal_state["brain_waste"] = max(0.0, current_waste - clearance)
                waste_cleared_total += clearance

        agent.step(user_input=None, external_stimulus=0.1)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Waste={m['brain_waste']:.3f} "
                  f"Health={m['brain_health']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Inflam={m['neuroinflammation']:.3f}")

    # ── Phase 3: Recall (300步) ──
    print("\n[Phase 3] RECALL — 测试记忆保留 + 系统性能 (300步)")
    for step in range(300):
        total_step = 700 + step
        agent.step(user_input=None, external_stimulus=0.1)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Waste={m['brain_waste']:.3f} "
                  f"Health={m['brain_health']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Anhedonia={m['symptom_anhedonia']:.3f}")

    # ── 计算结果 ──
    phase3_waste = np.mean(history["brain_waste"][-50:])
    phase3_health = np.mean(history["brain_health"][-50:])
    phase3_explore = np.mean(history["exploration_rate"][-50:])

    # Memory retention: Phase3探索率保持相对于Phase1的比例
    memory_retention = phase3_explore / max(phase1_explore, 0.001)

    # Toxicity index: Phase2-3中废物平均水平 (越高=毒性越强)
    toxicity_index = np.mean(history["brain_waste"][300:])

    # Clearance efficiency: 总清除量 / Phase1废物水平
    clearance_efficiency = waste_cleared_total / max(phase1_waste, 0.01)

    result = {
        "group": group_name,
        "strategy": strategy,
        "phase1_waste": phase1_waste,
        "phase1_health": phase1_health,
        "phase1_explore": phase1_explore,
        "phase3_waste": phase3_waste,
        "phase3_health": phase3_health,
        "phase3_explore": phase3_explore,
        "memory_retention": memory_retention,
        "toxicity_index": toxicity_index,
        "clearance_efficiency": clearance_efficiency,
        "waste_cleared_total": waste_cleared_total,
        "waste_trajectory": history["brain_waste"][::10],
        "health_trajectory": history["brain_health"][::10],
        "explore_trajectory": history["exploration_rate"][::10],
    }
    return result


def run_experiment():
    print("=" * 70)
    print("实验六: 胶质淋巴系统的时间窗口验证 (REAL Simulacrum AGENT)")
    print("3 Strategies: Continuous / Sleep-gated / Gamma-triggered")
    print("=" * 70)

    results = {}
    results["Continuous"] = run_group("Continuous", "continuous")
    results["SleepGated"] = run_group("SleepGated", "sleep_gated")
    results["GammaTrigger"] = run_group("GammaTrigger", "gamma")

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验六 结果汇总 (REAL Simulacrum AGENT)")
    print("=" * 70)

    print(f"\n{'指标':<25s} {'Continuous':>12s} {'SleepGated':>12s} {'Gamma':>12s}")
    print("-" * 61)
    print(f"{'Phase1 废物水平':<25s} "
          f"{results['Continuous']['phase1_waste']:>12.3f} "
          f"{results['SleepGated']['phase1_waste']:>12.3f} "
          f"{results['GammaTrigger']['phase1_waste']:>12.3f}")
    print(f"{'Phase3 废物水平':<25s} "
          f"{results['Continuous']['phase3_waste']:>12.3f} "
          f"{results['SleepGated']['phase3_waste']:>12.3f} "
          f"{results['GammaTrigger']['phase3_waste']:>12.3f}")
    print(f"{'Phase3 脑健康':<25s} "
          f"{results['Continuous']['phase3_health']:>12.3f} "
          f"{results['SleepGated']['phase3_health']:>12.3f} "
          f"{results['GammaTrigger']['phase3_health']:>12.3f}")
    print(f"{'记忆保留率':<25s} "
          f"{results['Continuous']['memory_retention']:>12.3f} "
          f"{results['SleepGated']['memory_retention']:>12.3f} "
          f"{results['GammaTrigger']['memory_retention']:>12.3f}")
    print(f"{'毒性指数':<25s} "
          f"{results['Continuous']['toxicity_index']:>12.3f} "
          f"{results['SleepGated']['toxicity_index']:>12.3f} "
          f"{results['GammaTrigger']['toxicity_index']:>12.3f}")
    print(f"{'清除效率':<25s} "
          f"{results['Continuous']['clearance_efficiency']:>12.2f} "
          f"{results['SleepGated']['clearance_efficiency']:>12.2f} "
          f"{results['GammaTrigger']['clearance_efficiency']:>12.2f}")

    # 关键验证
    print(f"\nKey validation:")
    # SleepGated应有最佳记忆保留
    best_memory = max(results.values(), key=lambda r: r["memory_retention"])
    print(f"  Best memory retention: {best_memory['group']} ({best_memory['memory_retention']:.3f}) "
          f"{'[PASS] Sleep-gated optimal' if best_memory['group'] == 'SleepGated' else '[INFO] Unexpected winner'}")
    # Continuous应有最低毒性但低记忆保留
    continuous_low_tox = results["Continuous"]["toxicity_index"] < results["SleepGated"]["toxicity_index"]
    print(f"  Continuous toxicity: {results['Continuous']['toxicity_index']:.3f} "
          f"{'[PASS] Lowest toxicity' if continuous_low_tox else '[INFO] Not lowest'}")
    # Continuous记忆保留应最低 (过度清洗 = 灾难性遗忘)
    continuous_worst_mem = results["Continuous"]["memory_retention"] <= results["SleepGated"]["memory_retention"]
    print(f"  Continuous memory loss: retention={results['Continuous']['memory_retention']:.3f} "
          f"{'[PASS] Catastrophic forgetting' if continuous_worst_mem else '[INFO] Memory preserved'}")
    # Gamma应在毒性和记忆之间取得平衡
    gamma_balanced = (results["GammaTrigger"]["toxicity_index"] < results["SleepGated"]["toxicity_index"]
                      and results["GammaTrigger"]["memory_retention"] > results["Continuous"]["memory_retention"])
    print(f"  Gamma balance: "
          f"{'[PASS] Balanced toxicity+memory' if gamma_balanced else '[INFO] Not balanced'}")

    return results


if __name__ == "__main__":
    run_experiment()
