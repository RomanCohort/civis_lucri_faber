"""实验三: HPA轴应激与认知僵化 (HPA Axis Stress & Cognitive Rigidity).

使用真实 Simulacrum 主循环 — 14脑区通过EventBus自然交互。

3 Phases:
  Phase 1 (Baseline, 500步): stress_reactivity=1.0, normal stimulus
  Phase 2 (Stress, 500步): stress_reactivity=2.0, high noise stimulus (0.85)
  Phase 3 (Recovery, 500步): stress_reactivity=1.0, normal stimulus

测量:
  - Stuck ratio: 连续N步选择相同goal的比例
  - Novelty score: 最近M步中不同goal类型数 / M
  - Cortisol/PFC/exploration_rate trajectory
  - Post-stress recovery: recovery阶段stuck_ratio恢复程度
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from typing import Dict, List
from collections import Counter

from simulacrum.core.agent import Simulacrum
from simulacrum.utils.config import Config


def read_metrics(agent: Simulacrum) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "pfc_inhibition": float(s.get("pfc_inhibition", 0.6)),
        "exploration_rate": float(agent.config.exploration_rate),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "symptom_anhedonia": float(s.get("symptom_anhedonia", 0.0)),
        "symptom_rumination": float(s.get("symptom_rumination", 0.0)),
        "da": float(s.get("nt_dopamine", 0.5)),
        "5ht": float(s.get("nt_serotonin", 0.5)),
        "balance": float(agent.thermo.balance),
    }


def compute_stuck_ratio(goal_history: List[str], window: int = 20) -> float:
    """计算stuck ratio: 最近window步中重复维度编号的比例

    认知僵化 = 反复探索相同的几个维度, 缺乏novelty
    提取goal中的维度编号, 如果只集中在少数几个维度, 则stuck_ratio高
    """
    if len(goal_history) < window:
        return 0.0
    recent = goal_history[-window:]
    import re
    # 提取维度编号 (格式: "维度79" → 79)
    dims = []
    for g in recent:
        m = re.search(r'维度(\d+)', g)
        if m:
            dims.append(m.group(1))
    if not dims:
        return 0.0
    # unique维度数 / 总维度数 → diversity ratio
    # stuck_ratio = 1 - diversity (维度越少越僵化)
    diversity = len(set(dims)) / len(dims)
    return 1.0 - diversity


def compute_novelty_score(goal_history: List[str], window: int = 50) -> float:
    """计算novelty score: 最近window步中unique goal数 / window"""
    if not goal_history:
        return 0.0
    recent = goal_history[-window:]
    return len(set(recent)) / len(recent) if recent else 0.0


def run_experiment():
    print("=" * 70)
    print("实验三: HPA轴应激与认知僵化 (REAL Simulacrum AGENT)")
    print("stress_reactivity=2.0 → Cortisol↑ → PFC↓ → Cognitive Rigidity")
    print("=" * 70)

    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        hpa_stress_reactivity=1.0,
        seed=42,
    )
    agent = Simulacrum(config=config)
    print(f"[INIT] Agent created. HPA stress_reactivity=1.0")

    # 记录器
    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}
    goal_history: List[str] = []
    stuck_ratios: List[float] = []
    novelty_scores: List[float] = []

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])
        # 记录goal
        goal_desc = agent.current_goal.description if agent.current_goal else "none"
        goal_history.append(goal_desc)
        # 计算stuck ratio和novelty
        stuck_ratios.append(compute_stuck_ratio(goal_history))
        novelty_scores.append(compute_novelty_score(goal_history))

    # ── Phase 1: Baseline (500步) ──
    print("\n[Phase 1] Baseline — stress_reactivity=1.0 (500步)")
    for step in range(500):
        agent.step(user_input=None, external_stimulus=0.1)
        record()
        if step % 100 == 0:
            m = read_metrics(agent)
            sr = stuck_ratios[-1] if stuck_ratios else 0
            ns = novelty_scores[-1] if novelty_scores else 0
            print(f"  Step {step:4d}: Cort={m['cortisol']:.3f} "
                  f"PFC={m['pfc_inhibition']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Stuck={sr:.3f} Novelty={ns:.3f}")

    # ── Phase 2: Stress (500步) ──
    print("\n[Phase 2] CHRONIC STRESS — stress_reactivity=2.0 (500步)")
    # 修改HPA轴应激反应性 + 持续注入皮质醇
    try:
        agent.hpa_axis.crh.stress_reactivity.data.fill_(5.0)
        print(f"  [OK] HPA stress_reactivity set to 5.0")
    except Exception as e:
        print(f"  [WARN] Could not set stress_reactivity: {e}")

    for step in range(500):
        total_step = 500 + step
        # 持续高皮质醇 (模拟慢性应激)
        agent.pharma.inject("cortisol", 0.7 + 0.1 * np.random.random())
        # 高噪声刺激
        agent.step(user_input=None, external_stimulus=0.85)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            sr = stuck_ratios[-1] if stuck_ratios else 0
            ns = novelty_scores[-1] if novelty_scores else 0
            print(f"  Step {total_step:4d}: Cort={m['cortisol']:.3f} "
                  f"PFC={m['pfc_inhibition']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Stuck={sr:.3f} Novelty={ns:.3f} "
                  f"AlloLoad={m['allostatic_load']:.3f}")

    # ── Phase 3: Recovery (500步) ──
    print("\n[Phase 3] RECOVERY — stress_reactivity=1.0 (500步)")
    # 恢复HPA轴应激反应性
    try:
        agent.hpa_axis.crh.stress_reactivity.data.fill_(1.0)
        print(f"  [OK] HPA stress_reactivity reset to 1.0")
    except Exception:
        pass
    agent.pharma.reset()
    # 重置HPA轴内部状态
    try:
        agent.hpa_axis.state = type(agent.hpa_axis.state)(
            crh_level=0.2, acth_level=0.2, cortisol_level=0.3,
            allostatic_load=0.0, stress_type="none",
            acute_stress_intensity=0.0, chronic_stress_ratio=0.0,
            recovery_state=0.8,
            cortisol_history=type(agent.hpa_axis.state.cortisol_history)(maxlen=200),
        )
        agent.hpa_axis.adrenal.current_cortisol = 0.3
        agent.hpa_axis.crh.current_crh = 0.2
        agent.hpa_axis.acth.current_acth = 0.2
        agent.hpa_axis.load_tracker.load = 0.0
    except Exception:
        pass

    for step in range(500):
        total_step = 1000 + step
        agent.step(user_input=None, external_stimulus=0.1)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            sr = stuck_ratios[-1] if stuck_ratios else 0
            ns = novelty_scores[-1] if novelty_scores else 0
            print(f"  Step {total_step:4d}: Cort={m['cortisol']:.3f} "
                  f"PFC={m['pfc_inhibition']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Stuck={sr:.3f} Novelty={ns:.3f}")

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验三 结果汇总 (REAL Simulacrum AGENT)")
    print("=" * 70)

    bl_cort = np.mean(history["cortisol"][:500])
    bl_pfc = np.mean(history["pfc_inhibition"][:500])
    bl_explore = np.mean(history["exploration_rate"][:500])
    bl_stuck = np.mean(stuck_ratios[:500])
    bl_novelty = np.mean(novelty_scores[:500])
    bl_allo = np.mean(history["allostatic_load"][:500])

    st_cort = np.mean(history["cortisol"][500:1000])
    st_pfc = np.mean(history["pfc_inhibition"][500:1000])
    st_explore = np.mean(history["exploration_rate"][500:1000])
    st_stuck = np.mean(stuck_ratios[500:1000])
    st_novelty = np.mean(novelty_scores[500:1000])
    st_allo = np.mean(history["allostatic_load"][500:1000])

    rc_cort = np.mean(history["cortisol"][1000:])
    rc_pfc = np.mean(history["pfc_inhibition"][1000:])
    rc_explore = np.mean(history["exploration_rate"][1000:])
    rc_stuck = np.mean(stuck_ratios[1000:])
    rc_novelty = np.mean(novelty_scores[1000:])
    rc_allo = np.mean(history["allostatic_load"][1000:])

    print(f"\n{'指标':<25s} {'Baseline':>10s} {'Stress':>10s} {'Recovery':>10s}")
    print("-" * 55)
    print(f"{'皮质醇':<25s} {bl_cort:>10.3f} {st_cort:>10.3f} {rc_cort:>10.3f}")
    print(f"{'PFC抑制':<25s} {bl_pfc:>10.3f} {st_pfc:>10.3f} {rc_pfc:>10.3f}")
    print(f"{'探索率':<25s} {bl_explore:>10.4f} {st_explore:>10.4f} {rc_explore:>10.4f}")
    print(f"{'Stuck Ratio':<25s} {bl_stuck:>10.3f} {st_stuck:>10.3f} {rc_stuck:>10.3f}")
    print(f"{'Novelty Score':<25s} {bl_novelty:>10.3f} {st_novelty:>10.3f} {rc_novelty:>10.3f}")
    print(f"{'异稳态负荷':<25s} {bl_allo:>10.3f} {st_allo:>10.3f} {rc_allo:>10.3f}")

    # 关键验证
    stuck_increase = (st_stuck - bl_stuck) / max(bl_stuck, 0.001) * 100
    novelty_decline = (bl_novelty - st_novelty) / max(bl_novelty, 0.001) * 100
    pfc_decline = (bl_pfc - st_pfc) / max(bl_pfc, 0.001) * 100
    stuck_recovery = (rc_stuck - st_stuck) / max(bl_stuck - st_stuck, 0.001) * 100

    print(f"\nKey validation:")
    print(f"  Stuck ratio increase: {stuck_increase:.1f}% "
          f"{'[PASS] Cognitive rigidity' if stuck_increase > 30 else '[INFO] Minor rigidity'}")
    print(f"  Novelty decline: {novelty_decline:.1f}% "
          f"{'[PASS] Reduced exploration' if novelty_decline > 20 else '[INFO] Minor change'}")
    print(f"  PFC decline: {pfc_decline:.1f}% "
          f"{'[PASS] PFC impaired' if pfc_decline > 10 else '[INFO] PFC intact'}")
    print(f"  Stuck recovery: {stuck_recovery:.1f}% "
          f"{'[PASS] Partial (scar effect)' if 20 < stuck_recovery < 80 else '[INFO] Full or no recovery'}")

    # 疤痕效应
    stuck_scar = (rc_stuck - bl_stuck) / max(bl_stuck, 0.001) * 100
    novelty_scar = (bl_novelty - rc_novelty) / max(bl_novelty, 0.001) * 100
    print(f"  Scar effect (stuck): {stuck_scar:.1f}% "
          f"{'[PASS] Scar present' if stuck_scar > 10 else '[INFO] No scar'}")
    print(f"  Scar effect (novelty): {novelty_scar:.1f}% "
          f"{'[PASS] Scar present' if novelty_scar > 10 else '[INFO] No scar'}")

    return {
        "history": history,
        "stuck_ratios": stuck_ratios,
        "novelty_scores": novelty_scores,
    }


if __name__ == "__main__":
    run_experiment()
