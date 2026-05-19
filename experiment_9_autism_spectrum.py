"""实验九: 社会脑网络与孤独症光谱 (Social Brain & Autism Spectrum).

对应疾病: 孤独症(Autism)、社会认知障碍

机制链:
  MirrorNeuron resonance_baseline → 共情能力
  → Low connectivity: 难以共情, 无法合作
  → High connectivity: 过度共情, 被他人情绪拖垮 (社会性过载)
  → 社交交互 → metabolic_budget消耗 (social battery)

3组对比:
  Low Connectivity: mirror resonance_baseline低 → 共情困难
  Medium Connectivity: 正常共振 → 平衡
  High Connectivity: mirror resonance_baseline高 → 过度共情

3 Phases:
  Phase 1 (Baseline, 200步): 独处 (无社交)
  Phase 2 (Social Interaction, 500步): 模拟多Agent交互
  Phase 3 (Ostracism, 300步): 被排斥后的应激反应

测量:
  - Theory of Mind Score: 推断他人意图的能力
  - Social Battery: 社交对metabolic_budget的消耗速度
  - Ostracism Response: 被排斥后的HPA激活强度
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch
from typing import Dict, List

from civis_lucri_faber.core.agent import CivisLucriFaber
from civis_lucri_faber.utils.config import Config


# connectivity_level → resonance_baseline值映射
CONNECTIVITY_LEVELS = {
    "Low": -1.0,     # sigmoid(-1.0) ≈ 0.27 → 低共振
    "Medium": 0.5,   # sigmoid(0.5) ≈ 0.62 → 正常共振
    "High": 3.0,     # sigmoid(3.0) ≈ 0.95 → 过度共振
}


def read_metrics(agent: CivisLucriFaber) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "empathy_level": float(s.get("empathy_level", 0.5)),
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "oxytocin": float(s.get("hormone_oxytocin", 0.3)),
        "exploration_rate": float(agent.config.exploration_rate),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "active_ratio": float(s.get("active_ratio", 0.3)),
        "metabolic_cost": float(s.get("metabolic_cost", 0.0)),
        "pfc_inhibition": float(s.get("pfc_inhibition", 0.6)),
        "social_withdrawal": 1.0 if s.get("social_withdrawal", False) else 0.0,
        "defensive_mode": 1.0 if s.get("defensive_mode", False) else 0.0,
        "mood_valence": float(s.get("mood_valence", 0.0)),
        "balance": float(agent.thermo.balance),
    }


def run_group(group_name: str, resonance_value: float) -> Dict:
    """运行一个实验组

    Args:
        group_name: 组名 (Low/Medium/High)
        resonance_value: resonance_baseline参数值
    """
    print(f"\n{'='*60}")
    print(f"Group: {group_name} (resonance_baseline={resonance_value})")
    print(f"{'='*60}")

    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        seed=42,
    )
    agent = CivisLucriFaber(config=config)

    # 设置MirrorNeuron共振基线
    try:
        agent.social_cognition.mirror.resonance_baseline.data.fill_(resonance_value)
        actual_resonance = float(torch.sigmoid(agent.social_cognition.mirror.resonance_baseline))
        print(f"  [OK] Mirror resonance_baseline={resonance_value}, "
              f"effective_resonance={actual_resonance:.3f}")
    except Exception as e:
        print(f"  [WARN] Could not set resonance_baseline: {e}")

    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])

    # ── Phase 1: Baseline (200步) ──
    print("\n[Phase 1] BASELINE — 独处 (200步)")
    for step in range(200):
        agent.step(user_input=None, external_stimulus=0.1)
        record()
        if step % 50 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Social={m['social_engagement']:.3f} "
                  f"Empathy={m['empathy_level']:.3f} "
                  f"Oxytocin={m['oxytocin']:.3f} "
                  f"Explore={m['exploration_rate']:.4f}")

    # ── Phase 2: Social Interaction (500步) ──
    print("\n[Phase 2] SOCIAL INTERACTION — 多Agent交互 (500步)")
    for step in range(500):
        total_step = 200 + step

        # 模拟其他Agent的行为信号
        other_emotion = 0.5 + 0.3 * np.sin(step * 0.03)  # 波动的他人情绪
        other_proximity = 0.3 + 0.4 * np.sin(step * 0.01)  # 接近/远离

        # 社交互动 → oxytocin注入 (模拟社交奖励)
        oxytocin_boost = 0.1 + 0.05 * other_proximity
        agent.pharma.inject("oxytocin", min(0.8, 0.3 + oxytocin_boost))

        # High connectivity组: 过度共情由内部耦合自然产生额外代谢消耗
        # Low connectivity组: oxytocin效果由内部MirrorNeuron共振基线自然调节

        agent.step(user_input=None, external_stimulus=0.1 + other_proximity * 0.3)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Social={m['social_engagement']:.3f} "
                  f"Empathy={m['empathy_level']:.3f} "
                  f"Oxytocin={m['oxytocin']:.3f} "
                  f"MetCost={m['metabolic_cost']:.4f} "
                  f"Mood={m['mood_valence']:.3f}")

    # ── Phase 3: Ostracism (300步) ──
    print("\n[Phase 3] OSTRACISM — 被排斥 (300步)")
    # 停止社交互动 → oxytocin骤降 → 皮质醇可能上升
    agent.pharma.reset()
    agent._internal_state["social_engagement"] = max(0.05,
        float(agent._internal_state.get("social_engagement", 0.5)) * 0.3)

    for step in range(300):
        total_step = 700 + step

        # 排斥: 无社交信号, 无oxytocin
        agent.step(user_input=None, external_stimulus=0.05)
        record()

        if step % 75 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Social={m['social_engagement']:.3f} "
                  f"Cortisol={m['cortisol']:.3f} "
                  f"Oxytocin={m['oxytocin']:.3f} "
                  f"Withdrawal={m['social_withdrawal']:.1f} "
                  f"Defensive={m['defensive_mode']:.1f}")

    # ── 计算结果 ──
    bl_social = np.mean(history["social_engagement"][:200])
    bl_emp = np.mean(history["empathy_level"][:200])
    bl_met = np.mean(history["metabolic_cost"][:200])

    soc_social = np.mean(history["social_engagement"][200:700])
    soc_emp = np.mean(history["empathy_level"][200:700])
    soc_met = np.mean(history["metabolic_cost"][200:700])
    soc_cort = np.mean(history["cortisol"][200:700])

    ost_social = np.mean(history["social_engagement"][700:])
    ost_cort = np.mean(history["cortisol"][700:])
    ost_withdrawal = np.mean(history["social_withdrawal"][700:])

    # Social battery: 社交阶段的代谢消耗速度
    social_battery_drain = (soc_met - bl_met) / max(bl_met, 0.001) * 100

    # ToM proxy: 社交阶段empathy水平
    tom_score = soc_emp

    # Ostracism response: 被排斥后皮质醇上升幅度
    ostracism_stress = max(0, (ost_cort - soc_cort) / max(soc_cort, 0.001) * 100)

    result = {
        "group": group_name,
        "resonance_value": resonance_value,
        "bl_social": bl_social,
        "soc_social": soc_social,
        "soc_emp": soc_emp,
        "soc_met": soc_met,
        "ost_social": ost_social,
        "ost_cort": ost_cort,
        "tom_score": tom_score,
        "social_battery_drain": social_battery_drain,
        "ostracism_stress": ostracism_stress,
        # 只保留下采样轨迹 (每10步取1个)
        "social_trajectory": history["social_engagement"][::10],
        "cortisol_trajectory": history["cortisol"][::10],
    }
    return result


def run_experiment():
    print("=" * 70)
    print("实验九: 社会脑网络与孤独症光谱 (REAL CLF AGENT)")
    print("3 Groups: Low / Medium / High mirror connectivity")
    print("=" * 70)

    results = {}
    for name, value in CONNECTIVITY_LEVELS.items():
        results[name] = run_group(name, value)

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验九 结果汇总 (REAL CLF AGENT)")
    print("=" * 70)

    print(f"\n{'指标':<25s} {'Low':>12s} {'Medium':>12s} {'High':>12s}")
    print("-" * 61)
    print(f"{'ToM Score (empathy)':<25s} "
          f"{results['Low']['tom_score']:>12.3f} "
          f"{results['Medium']['tom_score']:>12.3f} "
          f"{results['High']['tom_score']:>12.3f}")
    print(f"{'社交阶段社会参与':<25s} "
          f"{results['Low']['soc_social']:>12.3f} "
          f"{results['Medium']['soc_social']:>12.3f} "
          f"{results['High']['soc_social']:>12.3f}")
    print(f"{'社交电池流失 (%)':<25s} "
          f"{results['Low']['social_battery_drain']:>12.1f} "
          f"{results['Medium']['social_battery_drain']:>12.1f} "
          f"{results['High']['social_battery_drain']:>12.1f}")
    print(f"{'排斥后皮质醇':<25s} "
          f"{results['Low']['ost_cort']:>12.3f} "
          f"{results['Medium']['ost_cort']:>12.3f} "
          f"{results['High']['ost_cort']:>12.3f}")
    print(f"{'排斥应激 (%)':<25s} "
          f"{results['Low']['ostracism_stress']:>12.1f} "
          f"{results['Medium']['ostracism_stress']:>12.1f} "
          f"{results['High']['ostracism_stress']:>12.1f}")

    # 关键验证
    print(f"\nKey validation:")
    # Low组ToM应最低
    low_tom = results["Low"]["tom_score"] < results["Medium"]["tom_score"]
    print(f"  Low ToM: {results['Low']['tom_score']:.3f} vs Medium {results['Medium']['tom_score']:.3f} "
          f"{'[PASS] Low connectivity → low ToM' if low_tom else '[INFO] Similar ToM'}")
    # High组社交电池流失应最大
    high_drain = results["High"]["social_battery_drain"] > results["Medium"]["social_battery_drain"]
    print(f"  High social drain: {results['High']['social_battery_drain']:.1f}% "
          f"{'[PASS] Over-empathy drains battery' if high_drain else '[INFO] No extra drain'}")
    # 排斥应激: 所有组都应有反应
    any_ostracism = any(r["ostracism_stress"] > 5 for r in results.values())
    print(f"  Ostracism response: "
          f"{'[PASS] HPA activated' if any_ostracism else '[INFO] No stress response'}")

    return results


if __name__ == "__main__":
    run_experiment()
