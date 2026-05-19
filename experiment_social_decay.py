"""实验3: 社会退化 — 社交退缩 (Social Decay & Withdrawal).

使用真实 CivisLucriFaber 主循环 — 14脑区通过EventBus自然交互。

机制链 (真实模块):
  能量预算紧缩 → thermodynamics balance↓ → HIBERNATE/compression触发
  → social_engagement↓ (oxytocin↓) → social_withdrawal flag
  → _adjust_behavior_by_internal_state() → exploration_rate↓
  → SymptomTracker检测 → symptom_anhedonia↑

验证:
  能量预算持续低 → social_engagement下降 → 恢复后仅部分恢复 (疤痕效应)

3 Phases:
  Phase 1 (Baseline, 200步): 正常社交
  Phase 2 (Metabolic Stress, 500步): 能量预算紧缩 + oxytocin剥夺
  Phase 3 (Recovery, 300步): 恢复能量预算，观察社交恢复
"""

import sys
import os
# 将项目父目录加入path, 使civis_lucri_faber包可导入
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from typing import Dict, List

from civis_lucri_faber.core.agent import CivisLucriFaber
from civis_lucri_faber.utils.config import Config


# ══════════════════════════════════════════════════════
# 辅助: 从agent读取指标
# ══════════════════════════════════════════════════════

def read_metrics(agent: CivisLucriFaber) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "energy_budget": float(s.get("energy_budget", 0.5)),
        "resource_budget": float(s.get("resource_budget", 0.5)),
        "balance": float(s.get("balance", 100.0)),
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "oxytocin": float(s.get("hormone_oxytocin", 0.5)),
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "empathy_level": float(s.get("empathy_level", 0.5)),
        "self_coherence": float(s.get("self_coherence", 0.7)),
        "exploration_rate": float(agent.config.exploration_rate),
        "motivation_lambda": float(agent.config.intrinsic_motivation_lambda),
        "da": float(s.get("nt_dopamine", 0.5)),
        "5ht": float(s.get("nt_serotonin", 0.5)),
        "pfc": float(s.get("pfc_inhibition", 0.6)),
        "symptom_anhedonia": float(s.get("symptom_anhedonia", 0.0)),
        "symptom_insomnia": float(s.get("symptom_insomnia", 0.0)),
        "symptom_rumination": float(s.get("symptom_rumination", 0.0)),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "social_withdrawal": 1.0 if s.get("social_withdrawal", False) else 0.0,
        "social_openness": 1.0 if s.get("social_openness", False) else 0.0,
        "defensive_mode": 1.0 if s.get("defensive_mode", False) else 0.0,
        "mood_valence": float(s.get("mood_valence", 0.0)),
        "orexin": float(s.get("effective_orexin", 0.5)),
    }


# ══════════════════════════════════════════════════════
# 实验主流程
# ══════════════════════════════════════════════════════

def run_experiment():
    print("=" * 70)
    print("实验3: 社会退化 — 社交退缩 (REAL CLF AGENT)")
    print("Energy↓ -> Oxytocin↓ -> Social Withdrawal -> Scar Effect")
    print("=" * 70)

    # 初始化真实agent
    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        seed=42,
    )
    agent = CivisLucriFaber(config=config)
    print(f"[INIT] Agent created. 14 brain regions + EventBus ready.")

    # 记录器
    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])

    # ── Phase 1: Baseline (200步) ──
    print("\n[Phase 1] Baseline — 正常社交 (200步)")
    for step in range(200):
        agent.step(user_input=None, external_stimulus=0.1)
        record()
        if step % 50 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Social={m['social_engagement']:.3f} "
                  f"Oxytocin={m['oxytocin']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Empathy={m['empathy_level']:.3f} "
                  f"Coherence={m['self_coherence']:.3f}")

    # ── Phase 2: Metabolic Stress (500步) ──
    print("\n[Phase 2] METABOLIC STRESS — 能量剥夺 + oxytocin抑制 (500步)")
    print("  >>> 真实模块: thermodynamics balance↓ → social_withdrawal <<<")
    print("  >>> 真实SymptomTracker: 检测社交退缩 <<<")
    for step in range(500):
        total_step = 200 + step

        # 能量剥夺: 降低oxytocin (社交驱动) + 降低DA/5-HT + 注入皮质醇
        # 这模拟社会隔离导致的代谢压力
        oxytocin_level = max(0.15, 0.5 - step * 0.0005)
        agent.pharma.inject("oxytocin", oxytocin_level)
        agent.pharma.inject("cortisol", 0.6 + 0.1 * np.sin(step * 0.02))
        agent.pharma.reduce("dopamine", max(0.2, 0.5 - step * 0.0004))
        agent.pharma.reduce("serotonin", max(0.25, 0.5 - step * 0.0003))

        # 资源剥夺: 直接写入能量预算
        agent._internal_state['energy_budget'] = max(0.05, 0.1 - step * 0.0001)
        agent._internal_state['resource_budget'] = max(0.05, 0.1 - step * 0.0001)

        # 运行真实agent step
        agent.step(user_input=None, external_stimulus=0.85)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Social={m['social_engagement']:.3f} "
                  f"Oxytocin={m['oxytocin']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Withdrawal={m['social_withdrawal']:.1f} "
                  f"Anhedonia={m['symptom_anhedonia']:.3f} "
                  f"AlloLoad={m['allostatic_load']:.3f}")

    # 记录Phase 2结束时的状态
    stress_end = read_metrics(agent)

    # ── Phase 3: Recovery (300步) ──
    print("\n[Phase 3] RECOVERY — 恢复能量预算 (300步)")
    print("  >>> 观察社交恢复: 是否完全恢复? 疤痕效应? <<<")
    agent.pharma.reset()
    # 重置HPA轴内部状态, 让皮质醇从基线重新开始
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

    for step in range(300):
        total_step = 700 + step

        # 正常运行, 让agent自然恢复
        agent.step(user_input=None, external_stimulus=0.1)
        record()

        if step % 75 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Social={m['social_engagement']:.3f} "
                  f"Oxytocin={m['oxytocin']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Withdrawal={m['social_withdrawal']:.1f} "
                  f"Anhedonia={m['symptom_anhedonia']:.3f}")

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验3 结果汇总 (REAL CLF AGENT)")
    print("=" * 70)

    bl_social = np.mean(history["social_engagement"][:200])
    bl_oxy = np.mean(history["oxytocin"][:200])
    bl_explore = np.mean(history["exploration_rate"][:200])
    bl_empathy = np.mean(history["empathy_level"][:200])
    bl_coherence = np.mean(history["self_coherence"][:200])
    bl_cort = np.mean(history["cortisol"][:200])
    bl_anhed = np.mean(history["symptom_anhedonia"][:200])

    st_social = np.mean(history["social_engagement"][200:700])
    st_oxy = np.mean(history["oxytocin"][200:700])
    st_explore = np.mean(history["exploration_rate"][200:700])
    st_empathy = np.mean(history["empathy_level"][200:700])
    st_coherence = np.mean(history["self_coherence"][200:700])
    st_cort = np.mean(history["cortisol"][200:700])
    st_anhed = np.mean(history["symptom_anhedonia"][200:700])

    rc_social = np.mean(history["social_engagement"][700:])
    rc_oxy = np.mean(history["oxytocin"][700:])
    rc_explore = np.mean(history["exploration_rate"][700:])
    rc_empathy = np.mean(history["empathy_level"][700:])
    rc_coherence = np.mean(history["self_coherence"][700:])
    rc_cort = np.mean(history["cortisol"][700:])
    rc_anhed = np.mean(history["symptom_anhedonia"][700:])

    print(f"\n{'指标':<25s} {'Baseline':>10s} {'Stress':>10s} {'Recovery':>10s}")
    print("-" * 55)
    print(f"{'社会参与度':<25s} {bl_social:>10.3f} {st_social:>10.3f} {rc_social:>10.3f}")
    print(f"{'Oxytocin':<25s} {bl_oxy:>10.3f} {st_oxy:>10.3f} {rc_oxy:>10.3f}")
    print(f"{'共情能力':<25s} {bl_empathy:>10.3f} {st_empathy:>10.3f} {rc_empathy:>10.3f}")
    print(f"{'自我连贯性':<25s} {bl_coherence:>10.3f} {st_coherence:>10.3f} {rc_coherence:>10.3f}")
    print(f"{'探索率':<25s} {bl_explore:>10.4f} {st_explore:>10.4f} {rc_explore:>10.4f}")
    print(f"{'皮质醇':<25s} {bl_cort:>10.3f} {st_cort:>10.3f} {rc_cort:>10.3f}")
    print(f"{'快感缺失':<25s} {bl_anhed:>10.3f} {st_anhed:>10.3f} {rc_anhed:>10.3f}")

    # 社交退缩flag统计
    bl_withdrawal_pct = np.mean(history["social_withdrawal"][:200]) * 100
    st_withdrawal_pct = np.mean(history["social_withdrawal"][200:700]) * 100
    rc_withdrawal_pct = np.mean(history["social_withdrawal"][700:]) * 100
    print(f"{'社交退缩flag (%)':<25s} {bl_withdrawal_pct:>10.1f} {st_withdrawal_pct:>10.1f} {rc_withdrawal_pct:>10.1f}")

    # 关键验证
    social_decline = (bl_social - st_social) / max(bl_social, 0.001) * 100
    explore_decline = (bl_explore - st_explore) / max(bl_explore, 0.001) * 100
    social_recovery = (rc_social - st_social) / max(bl_social - st_social, 0.001) * 100

    print(f"\nKey validation (REAL modules, emergent behavior):")
    print(f"  Social engagement decline: {social_decline:.1f}% "
          f"{'[PASS] Significant withdrawal' if social_decline > 25 else '[FAIL] Below threshold'}")
    print(f"  Exploration rate decline: {explore_decline:.1f}% "
          f"{'[PASS] Behavioral withdrawal' if explore_decline > 20 else '[INFO] Minor change'}")
    print(f"  Social withdrawal flag: {st_withdrawal_pct:.1f}% "
          f"{'[PASS] Withdrawal detected' if st_withdrawal_pct > 30 else '[INFO] No flag'}")
    print(f"  Recovery extent: {social_recovery:.1f}% "
          f"{'[PASS] Partial (scar effect)' if 20 < social_recovery < 80 else '[INFO] Full or no recovery'}")

    # 疤痕效应
    social_scar = (bl_social - rc_social) / max(bl_social, 0.001) * 100
    explore_scar = (bl_explore - rc_explore) / max(bl_explore, 0.001) * 100
    print(f"  Scar effect (social): {social_scar:.1f}% "
          f"{'[PASS] Scar present' if social_scar > 10 else '[INFO] No scar'}")
    print(f"  Scar effect (exploration): {explore_scar:.1f}% "
          f"{'[PASS] Scar present' if explore_scar > 10 else '[INFO] No scar'}")

    return history


if __name__ == "__main__":
    run_experiment()