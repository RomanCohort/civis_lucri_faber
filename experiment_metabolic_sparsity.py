"""实验二: 代谢预算与稀疏性 (Metabolic Budget & Sparsity).

使用真实 CivisLucriFaber 主循环 — 14脑区通过EventBus自然交互。

2组对比:
  Control:      resource_budget=1.0 (无限制), starvation_prob=0.0
  Experimental: resource_budget=0.3 (30%限制), starvation_prob=0.15

3 Phases:
  Phase 1 (Baseline, 200步): 正常运行
  Phase 2 (Budget Stress, 400步): Experimental组施加资源约束
  Phase 3 (Recovery, 200步): 恢复资源预算，观察恢复

测量:
  - Performance drop: exploration_rate变化
  - Zombie neurons: active_ratio (被阻断的pathway比例)
  - Recovery time: 恢复到baseline的步数
  - Allostatic load trajectory
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from typing import Dict, List

from civis_lucri_faber.core.agent import CivisLucriFaber
from civis_lucri_faber.utils.config import Config


def read_metrics(agent: CivisLucriFaber) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "exploration_rate": float(agent.config.exploration_rate),
        "active_ratio": float(s.get("active_ratio", 0.3)),
        "metabolic_cost": float(s.get("metabolic_cost", 0.0)),
        "budget_utilization": float(s.get("budget_utilization", 0.3)),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "pfc_inhibition": float(s.get("pfc_inhibition", 0.6)),
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "symptom_anhedonia": float(s.get("symptom_anhedonia", 0.0)),
        "balance": float(agent.thermo.balance),
        "resource_budget": float(s.get("resource_budget", 0.3)),
    }


def run_group(group_name: str, resource_budget: float, starvation_prob: float) -> Dict:
    """运行一个实验组"""
    print(f"\n{'='*60}")
    print(f"Group: {group_name} (resource_budget={resource_budget}, starvation_prob={starvation_prob})")
    print(f"{'='*60}")

    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        resource_budget=resource_budget,
        seed=42,
    )
    agent = CivisLucriFaber(config=config)

    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])

    # ── Phase 1: Baseline (200步) ──
    print("\n[Phase 1] Baseline — 正常运行 (200步)")
    for step in range(200):
        agent.step(user_input=None, external_stimulus=0.1)
        record()
        if step % 50 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Explore={m['exploration_rate']:.4f} "
                  f"ActiveRatio={m['active_ratio']:.3f} "
                  f"MetCost={m['metabolic_cost']:.4f} "
                  f"Budget={m['budget_utilization']:.3f}")

    # ── Phase 2: Budget Stress (400步) ──
    print("\n[Phase 2] BUDGET STRESS — 资源约束 (400步)")
    for step in range(400):
        total_step = 200 + step

        # Experimental组: 降低resource_budget (模拟代谢压力)
        if group_name == "Experimental":
            # 渐进降低resource_budget
            budget_val = max(0.1, resource_budget - step * 0.0003)
            agent.metabolic.budget = budget_val

        agent.step(user_input=None, external_stimulus=0.1)

        # step()内部代谢耦合会自然调整active_ratio

        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Explore={m['exploration_rate']:.4f} "
                  f"ActiveRatio={m['active_ratio']:.3f} "
                  f"MetCost={m['metabolic_cost']:.4f} "
                  f"AlloLoad={m['allostatic_load']:.3f} "
                  f"Budget={m['budget_utilization']:.3f}")

    # ── Phase 3: Recovery (200步) ──
    print("\n[Phase 3] RECOVERY — 恢复资源预算 (200步)")
    if group_name == "Experimental":
        # 恢复resource_budget到正常值
        agent.metabolic.budget = 1.0
        agent._internal_state['resource_budget'] = 1.0

    for step in range(200):
        total_step = 600 + step
        agent.step(user_input=None, external_stimulus=0.1)
        record()

        if step % 50 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Explore={m['exploration_rate']:.4f} "
                  f"ActiveRatio={m['active_ratio']:.3f} "
                  f"MetCost={m['metabolic_cost']:.4f}")

    # 计算结果
    bl_explore = np.mean(history["exploration_rate"][:200])
    bl_active = np.mean(history["active_ratio"][:200])
    st_explore = np.mean(history["exploration_rate"][200:600])
    st_active = np.mean(history["active_ratio"][200:600])
    rc_explore = np.mean(history["exploration_rate"][600:])
    rc_active = np.mean(history["active_ratio"][600:])

    # Recovery time: 从Phase 3开始到active_ratio恢复到>0.25的步数
    recovery_time = None
    for i, ar in enumerate(history["active_ratio"][600:]):
        if ar > 0.25:
            recovery_time = i
            break

    result = {
        "group": group_name,
        "resource_budget": resource_budget,
        "bl_explore": bl_explore,
        "bl_active": bl_active,
        "st_explore": st_explore,
        "st_active": st_active,
        "rc_explore": rc_explore,
        "rc_active": rc_active,
        "explore_drop_pct": (bl_explore - st_explore) / max(bl_explore, 0.001) * 100,
        "recovery_time": recovery_time if recovery_time is not None else 200,
        "allostatic_trajectory": history["allostatic_load"],
        "active_ratio_trajectory": history["active_ratio"],
    }
    return result


def run_experiment():
    print("=" * 70)
    print("实验二: 代谢预算与稀疏性 (REAL CLF AGENT)")
    print("Control (budget=1.0) vs Experimental (budget=0.3)")
    print("=" * 70)

    ctrl_result = run_group("Control", resource_budget=1.0, starvation_prob=0.0)
    exp_result = run_group("Experimental", resource_budget=0.3, starvation_prob=0.15)

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验二 结果汇总 (REAL CLF AGENT)")
    print("=" * 70)

    print(f"\n{'指标':<25s} {'Control':>12s} {'Experimental':>12s}")
    print("-" * 49)
    print(f"{'基线探索率':<25s} {ctrl_result['bl_explore']:>12.4f} {exp_result['bl_explore']:>12.4f}")
    print(f"{'应激探索率':<25s} {ctrl_result['st_explore']:>12.4f} {exp_result['st_explore']:>12.4f}")
    print(f"{'恢复探索率':<25s} {ctrl_result['rc_explore']:>12.4f} {exp_result['rc_explore']:>12.4f}")
    print(f"{'基线active_ratio':<25s} {ctrl_result['bl_active']:>12.3f} {exp_result['bl_active']:>12.3f}")
    print(f"{'应激active_ratio':<25s} {ctrl_result['st_active']:>12.3f} {exp_result['st_active']:>12.3f}")
    print(f"{'恢复active_ratio':<25s} {ctrl_result['rc_active']:>12.3f} {exp_result['rc_active']:>12.3f}")
    print(f"{'探索率下降 (%)':<25s} {ctrl_result['explore_drop_pct']:>12.1f} {exp_result['explore_drop_pct']:>12.1f}")
    print(f"{'恢复时间 (步)':<25s} {ctrl_result['recovery_time']:>12d} {exp_result['recovery_time']:>12d}")

    # 关键验证
    print(f"\nKey validation:")
    # Performance drop: Experimental组探索率下降应>15%
    exp_drop = exp_result['explore_drop_pct']
    print(f"  Experimental explore drop: {exp_drop:.1f}% "
          f"{'[PASS] Significant drop' if exp_drop > 15 else '[INFO] Minor drop'}")
    # Zombie neurons: Experimental组active_ratio应<0.3
    zombie = exp_result['st_active'] < 0.3
    print(f"  Zombie neurons: active_ratio={exp_result['st_active']:.3f} "
          f"{'[PASS] Neurons blocked' if zombie else '[INFO] No blocking'}")
    # Recovery: Experimental组应能恢复（疤痕效应）
    recovery_pct = (exp_result['rc_explore'] - exp_result['st_explore']) / max(exp_result['bl_explore'] - exp_result['st_explore'], 0.001) * 100
    print(f"  Recovery extent: {recovery_pct:.1f}% "
          f"{'[PASS] Partial recovery' if 20 < recovery_pct < 80 else '[INFO] Full or no recovery'}")

    return {"Control": ctrl_result, "Experimental": exp_result}


if __name__ == "__main__":
    run_experiment()