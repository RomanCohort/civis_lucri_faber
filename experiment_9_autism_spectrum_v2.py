"""实验九v2: 社会脑网络与孤独症光谱 (Social Brain & Autism Spectrum).

使用真实 Agent 内部状态变化 — 无人工 pharma.inject().

关键改进:
  1. 移除 np.sin() 人工社交信号曲线
  2. 移除人工 pharma.inject("oxytocin")
  3. 使用不规律社交事件时间表
  4. oxytocin 由 Agent 内部社交交互自然调节
  5. resonance_baseline 保留真实影响 (Exp 9改进)
  6. Agent 状态真实性验证

对应疾病: 孤独症(Autism)、社会认知障碍

机制链:
  MirrorNeuron resonance_baseline → 共情能力
  → Low connectivity: 难以共情, 无法合作
  → High connectivity: 过度共情, 被他人情绪拖垮 (社会性过载)
  → 社交交互 → metabolic_budget消耗 (social battery)

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
from typing import Dict, List, Tuple
import random

from simulacrum.core.agent import Simulacrum
from simulacrum.utils.config import Config


CONNECTIVITY_LEVELS = {
    "Low": -1.0,
    "Medium": 0.5,
    "High": 3.0,
}


def generate_social_schedule(n_steps: int, interaction_probability: float = 0.3,
                             ostracism_probability: float = 0.02) -> List[Dict]:
    """生成不规律社交事件时间表

    返回每个step的事件:
    - type: "approach", "withdraw", "emotional_exchange", "neutral", "ostracism"
    - other_emotion: 他人情绪强度 [-1, 1]
    - other_proximity: 接近程度 [0, 1]
    """
    schedule = []
    ostracism_mode = False
    ostracism_remaining = 0

    for step in range(n_steps):
        if ostracism_remaining > 0:
            # 被排斥期: 无社交信号
            schedule.append({
                "type": "ostracism",
                "other_emotion": 0.0,
                "other_proximity": 0.0,
                "social_signal": 0.0,
            })
            ostracism_remaining -= 1
        else:
            if ostracism_mode:
                ostracism_mode = False

            if random.random() < ostracism_probability:
                # 进入排斥期
                ostracism_remaining = random.randint(20, 60)
                ostracism_mode = True
                schedule.append({
                    "type": "ostracism_start",
                    "other_emotion": 0.0,
                    "other_proximity": 0.0,
                    "social_signal": 0.0,
                })
            elif random.random() < interaction_probability:
                # 社交互动
                proximity = random.uniform(0.2, 0.8)
                emotion = random.uniform(-0.5, 0.8)  # 他人情绪波动

                if random.random() < 0.15:
                    # 亲密接触
                    schedule.append({
                        "type": "approach",
                        "other_emotion": emotion,
                        "other_proximity": min(1.0, proximity + 0.3),
                        "social_signal": proximity * 0.5 + abs(emotion) * 0.3,
                    })
                elif random.random() < 0.2:
                    # 退缩
                    schedule.append({
                        "type": "withdraw",
                        "other_emotion": emotion,
                        "other_proximity": max(0.1, proximity - 0.2),
                        "social_signal": max(0.05, proximity * 0.3),
                    })
                else:
                    # 正常交换
                    schedule.append({
                        "type": "emotional_exchange",
                        "other_emotion": emotion,
                        "other_proximity": proximity,
                        "social_signal": proximity * 0.4 + abs(emotion) * 0.2,
                    })
            else:
                # 独处
                schedule.append({
                    "type": "neutral",
                    "other_emotion": 0.0,
                    "other_proximity": random.uniform(0.05, 0.15),
                    "social_signal": 0.05,
                })
    return schedule


def read_metrics(agent: Simulacrum) -> Dict[str, float]:
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
        "mirror_resonance_baseline": float(s.get("mirror_resonance_baseline", 0.5)),
    }


def validate_data_authenticity(history: Dict[str, List[float]]) -> Tuple[bool, str]:
    """验证数据真实性"""
    oxytocin_std = np.std(history["oxytocin"])
    empathy_std = np.std(history["empathy_level"])

    if oxytocin_std < 0.05:
        return False, f"催产素轨迹过于平滑 (STD={oxytocin_std:.4f})"
    if empathy_std < 0.05:
        return False, f"共情轨迹过于平滑 (STD={empathy_std:.4f})"

    oxytocin_range = np.max(history["oxytocin"]) - np.min(history["oxytocin"])
    if oxytocin_range < 0.15:
        return False, f"催产素变化幅度过小 (range={oxytocin_range:.4f})"

    return True, f"数据真实性验证通过 (Oxytocin STD={oxytocin_std:.4f}, range={oxytocin_range:.4f})"


def run_group(group_name: str, resonance_value: float) -> Dict:
    """运行一个实验组"""
    print(f"\n{'='*60}")
    print(f"Group: {group_name} (resonance_baseline={resonance_value})")
    print(f"{'='*60}")

    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        hpa_stress_reactivity=1.5,
        seed=42,
    )
    agent = Simulacrum(config=config)

    # 设置MirrorNeuron共振基线
    try:
        agent.social_cognition.mirror.resonance_baseline.data.fill_(resonance_value)
        actual_resonance = float(torch.sigmoid(agent.social_cognition.mirror.resonance_baseline))
        print(f"  [OK] Mirror resonance_baseline={resonance_value}, "
              f"effective_resonance={actual_resonance:.3f}")
        agent._internal_state["mirror_resonance_baseline"] = resonance_value
    except Exception as e:
        print(f"  [WARN] Could not set resonance_baseline: {e}")

    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}
    oxytocin_sources = []

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])
        # v2: oxytocin 由 Agent 内部自然调节 (无pharma.inject)
        oxytocin_sources.append(True)

    # ── Phase 1: Baseline (200步) ──
    print("\n[Phase 1] BASELINE — 独处 (200步)")
    baseline_schedule = generate_social_schedule(200, interaction_probability=0.05)

    for step in range(200):
        event = baseline_schedule[step]
        agent.step(user_input=None, external_stimulus=event["social_signal"])
        record()
        if step % 50 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Social={m['social_engagement']:.3f} "
                  f"Empathy={m['empathy_level']:.3f} "
                  f"Oxytocin={m['oxytocin']:.3f} "
                  f"Explore={m['exploration_rate']:.4f}")

    # ── Phase 2: Social Interaction (500步) ──
    print("\n[Phase 2] SOCIAL INTERACTION — 不规律社交事件 (500步)")
    print("  >>> Oxytocin 由 Agent 内部自然调节 <<<")

    social_schedule = generate_social_schedule(500, interaction_probability=0.4, ostracism_probability=0.01)

    for step in range(500):
        total_step = 200 + step
        event = social_schedule[step]

        # 社交信号由事件决定 (不使用 pharma.inject)
        # Agent 内部会根据 social_engagement 自然调节 oxytocin
        social_signal = event["social_signal"]
        agent.step(user_input=None, external_stimulus=social_signal)

        # 记录他人情绪信号 (供 MirrorNeuron 使用)
        agent._internal_state["other_agent_emotion"] = event["other_emotion"]
        agent._internal_state["other_agent_proximity"] = event["other_proximity"]

        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            event_type = event["type"]
            print(f"  Step {total_step:4d}: Social={m['social_engagement']:.3f} "
                  f"Empathy={m['empathy_level']:.3f} "
                  f"Oxytocin={m['oxytocin']:.3f} "
                  f"MetCost={m['metabolic_cost']:.4f} "
                  f"Mood={m['mood_valence']:.3f} "
                  f"Event={event_type}")

    # ── Phase 3: Ostracism (300步) ──
    print("\n[Phase 3] OSTRACISM — 被排斥 (300步)")
    print("  >>> 观察 HPA 自然应激反应 <<<")

    ostracism_schedule = generate_social_schedule(300, interaction_probability=0.02, ostracism_probability=0.8)

    for step in range(300):
        total_step = 700 + step
        event = ostracism_schedule[step]
        agent.step(user_input=None, external_stimulus=event["social_signal"])
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
    bl_oxy = np.mean(history["oxytocin"][:200])

    soc_social = np.mean(history["social_engagement"][200:700])
    soc_emp = np.mean(history["empathy_level"][200:700])
    soc_met = np.mean(history["metabolic_cost"][200:700])
    soc_cort = np.mean(history["cortisol"][200:700])
    soc_oxy = np.mean(history["oxytocin"][200:700])

    ost_social = np.mean(history["social_engagement"][700:])
    ost_cort = np.mean(history["cortisol"][700:])
    ost_oxy = np.mean(history["oxytocin"][700:])
    ost_withdrawal = np.mean(history["social_withdrawal"][700:])

    social_battery_drain = (soc_met - bl_met) / max(bl_met, 0.001) * 100
    tom_score = soc_emp
    ostracism_stress = max(0, (ost_cort - soc_cort) / max(soc_cort, 0.001) * 100)

    authentic, auth_msg = validate_data_authenticity(history)

    result = {
        "group": group_name,
        "resonance_value": resonance_value,
        "bl_social": bl_social,
        "soc_social": soc_social,
        "soc_emp": soc_emp,
        "soc_met": soc_met,
        "soc_oxy": soc_oxy,
        "ost_social": ost_social,
        "ost_cort": ost_cort,
        "ost_oxy": ost_oxy,
        "tom_score": tom_score,
        "social_battery_drain": social_battery_drain,
        "ostracism_stress": ostracism_stress,
        "authentic": authentic,
        "auth_msg": auth_msg,
        "social_trajectory": history["social_engagement"][::10],
        "cortisol_trajectory": history["cortisol"][::10],
        "oxytocin_trajectory": history["oxytocin"][::10],
    }
    return result


def run_experiment():
    print("=" * 70)
    print("实验九v2: 社会脑网络与孤独症光谱 (NATIVE OXYTOCIN)")
    print("3 Groups: Low / Medium / High mirror connectivity")
    print("=" * 70)

    results = {}
    for name, value in CONNECTIVITY_LEVELS.items():
        results[name] = run_group(name, value)

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验九v2 结果汇总 (NATIVE OXYTOCIN)")
    print("=" * 70)

    print(f"\n[数据验证]")
    for name, result in results.items():
        print(f"  {name}: {result['auth_msg']}")

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
    print(f"{'社交阶段Oxytocin':<25s} "
          f"{results['Low']['soc_oxy']:>12.3f} "
          f"{results['Medium']['soc_oxy']:>12.3f} "
          f"{results['High']['soc_oxy']:>12.3f}")
    print(f"{'社交电池流失 (%)':<25s} "
          f"{results['Low']['social_battery_drain']:>12.1f} "
          f"{results['Medium']['social_battery_drain']:>12.1f} "
          f"{results['High']['social_battery_drain']:>12.1f}")
    print(f"{'排斥后皮质醇':<25s} "
          f"{results['Low']['ost_cort']:>12.3f} "
          f"{results['Medium']['ost_cort']:>12.3f} "
          f"{results['High']['ost_cort']:>12.3f}")
    print(f"{'排斥后Oxytocin':<25s} "
          f"{results['Low']['ost_oxy']:>12.3f} "
          f"{results['Medium']['ost_oxy']:>12.3f} "
          f"{results['High']['ost_oxy']:>12.3f}")
    print(f"{'排斥应激 (%)':<25s} "
          f"{results['Low']['ostracism_stress']:>12.1f} "
          f"{results['Medium']['ostracism_stress']:>12.1f} "
          f"{results['High']['ostracism_stress']:>12.1f}")

    # 关键验证
    print(f"\nKey validation:")
    print(f"  Oxytocin来源: Agent内部自然调节 (无pharma.inject)")
    print(f"  数据真实性: "
          f"Low={'PASS' if results['Low']['authentic'] else 'FAIL'}, "
          f"Medium={'PASS' if results['Medium']['authentic'] else 'FAIL'}, "
          f"High={'PASS' if results['High']['authentic'] else 'FAIL'}")

    low_tom = results["Low"]["tom_score"] < results["Medium"]["tom_score"]
    print(f"  Low ToM: {results['Low']['tom_score']:.3f} vs Medium {results['Medium']['tom_score']:.3f} "
          f"{'[PASS] Low connectivity → low ToM' if low_tom else '[INFO] Similar ToM'}")

    high_drain = results["High"]["social_battery_drain"] > results["Medium"]["social_battery_drain"]
    print(f"  High social drain: {results['High']['social_battery_drain']:.1f}% "
          f"{'[PASS] Over-empathy drains battery' if high_drain else '[INFO] No extra drain'}")

    # Exp 9改进验证: resonance_baseline保留影响
    oxy_diff = results["High"]["soc_oxy"] - results["Low"]["soc_oxy"]
    print(f"  Resonance baseline保留: High oxytocin={results['High']['soc_oxy']:.3f} vs Low={results['Low']['soc_oxy']:.3f} "
          f"{'[PASS] Baseline influences empathy' if oxy_diff > 0.1 else '[INFO] Similar oxytocin levels'}")

    return results


if __name__ == "__main__":
    run_experiment()