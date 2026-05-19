"""实验2: 药物干预 — 决策模式漂移 (Drug-Induced Decision Drift).

使用真实 Simulacrum 主循环 — 14脑区通过EventBus自然交互。

机制链 (真实模块):
  药物 → NeuroPharmacology.prescribe() → NT水平突变
  → _adjust_behavior_by_internal_state() → exploration_rate/motivation变化
  → SymptomTracker检测 → 症状变化
  → BasalGanglia → action selection变化

验证:
  致幻剂 → exploration_rate↑, precision↓; 镇静剂 → exploration_rate↓; 兴奋剂 → DA↑, motivation↑

5 Phases:
  Phase 1 (Baseline, 200步): 正常决策
  Phase 2a (致幻剂, 300步): prescribe('hallucinogen') → 5-HT2A激动
  Phase 2b (镇静剂, 300步): prescribe('sedative') → GABA-A PAM
  Phase 2c (兴奋剂, 300步): prescribe('stimulant') → DAT激动
  Phase 3 (Washout, 200步): pharma.reset() → 药物清除
"""

import sys
import os
# 将项目父目录加入path, 使simulacrum包可导入
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from typing import Dict, List

from simulacrum.core.agent import Simulacrum
from simulacrum.utils.config import Config


# ══════════════════════════════════════════════════════
# 辅助: 从agent读取指标
# ══════════════════════════════════════════════════════

def read_metrics(agent: Simulacrum) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "da": float(s.get("nt_dopamine", 0.5)),
        "5ht": float(s.get("nt_serotonin", 0.5)),
        "gaba": float(s.get("nt_gaba", 0.5)),
        "ne": float(s.get("nt_norepinephrine", 0.3)),
        "glu": float(s.get("nt_glutamate", 0.5)),
        "arousal": float(s.get("limbic_arousal", 0.3)),
        "pfc": float(s.get("pfc_inhibition", 0.6)),
        "precision": float(s.get("predictive_coding_precision_mult", 0.4)),
        "exploration_rate": float(agent.config.exploration_rate),
        "motivation_lambda": float(agent.config.intrinsic_motivation_lambda),
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "mood_valence": float(s.get("mood_valence", 0.0)),
        "mood_arousal": float(s.get("mood_arousal", 0.5)),
        "symptom_anhedonia": float(s.get("symptom_anhedonia", 0.0)),
        "symptom_panic": float(s.get("symptom_panic", 0.0)),
        "symptom_insomnia": float(s.get("symptom_insomnia", 0.0)),
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "balance": float(s.get("balance", 100.0)),
        "defensive_mode": 1.0 if s.get("defensive_mode", False) else 0.0,
        "social_withdrawal": 1.0 if s.get("social_withdrawal", False) else 0.0,
    }


# ══════════════════════════════════════════════════════
# 实验主流程
# ══════════════════════════════════════════════════════

def run_experiment():
    print("=" * 70)
    print("实验2: 药物干预 — 决策模式漂移 (REAL Simulacrum AGENT)")
    print("Drug -> NT mutation -> Behavior adjustment -> Symptom detection")
    print("=" * 70)

    # 初始化真实agent
    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        seed=42,
    )
    agent = Simulacrum(config=config)
    print(f"[INIT] Agent created. 14 brain regions + EventBus ready.")

    # 记录器
    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])

    # ── Phase 1: Baseline (200步) ──
    print("\n[Phase 1] Baseline — 正常决策 (200步)")
    for step in range(200):
        agent.step(user_input=None, external_stimulus=0.1)
        record()
        if step % 50 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Explore={m['exploration_rate']:.4f} "
                  f"DA={m['da']:.3f} GABA={m['gaba']:.3f} "
                  f"5HT={m['5ht']:.3f} PFC={m['pfc']:.3f}")

    baseline_explore = np.mean(history["exploration_rate"])
    baseline_motiv = np.mean(history["motivation_lambda"])

    # ── Phase 2a: 致幻剂 (300步) ──
    print("\n[Phase 2a] HALLUCINOGEN — prescribe('hallucinogen') (300步)")
    print("  >>> 5-HT2A激动: 5-HT=1.0, PFC anesthetized, sensory activated <<<")
    agent.pharma.prescribe("hallucinogen")

    for step in range(300):
        total_step = 200 + step
        agent.step(user_input=None, external_stimulus=0.3)
        record()
        if step % 75 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Explore={m['exploration_rate']:.4f} "
                  f"5HT={m['5ht']:.3f} PFC={m['pfc']:.3f} "
                  f"Precision={m['precision']:.3f} Arousal={m['arousal']:.3f}")

    hallucinogen_explore = np.mean(history["exploration_rate"][-300:])
    hallucinogen_motiv = np.mean(history["motivation_lambda"][-300:])

    # ── Washout (50步) ──
    print("\n  [Washout] pharma.reset() (50步)")
    agent.pharma.reset()
    for step in range(50):
        agent.step(user_input=None, external_stimulus=0.1)
        record()

    # ── Phase 2b: 镇静剂 (300步) ──
    print("\n[Phase 2b] SEDATIVE — prescribe('sedative') (300步)")
    print("  >>> GABA-A PAM: GABA=0.9, brainstem+amygdala anesthetized <<<")
    agent.pharma.prescribe("sedative")

    for step in range(300):
        total_step = 550 + step
        agent.step(user_input=None, external_stimulus=0.1)
        record()
        if step % 75 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Explore={m['exploration_rate']:.4f} "
                  f"GABA={m['gaba']:.3f} Arousal={m['arousal']:.3f} "
                  f"Cortisol={m['cortisol']:.3f} Defensive={m['defensive_mode']:.1f}")

    sedative_explore = np.mean(history["exploration_rate"][-300:])
    sedative_motiv = np.mean(history["motivation_lambda"][-300:])

    # ── Washout (50步) ──
    print("\n  [Washout] pharma.reset() (50步)")
    agent.pharma.reset()
    for step in range(50):
        agent.step(user_input=None, external_stimulus=0.1)
        record()

    # ── Phase 2c: 兴奋剂 (300步) ──
    print("\n[Phase 2c] STIMULANT — prescribe('stimulant') (300步)")
    print("  >>> DAT激动: DA=0.9, NE=0.8, brainstem activated <<<")
    agent.pharma.prescribe("stimulant")

    for step in range(300):
        total_step = 900 + step
        agent.step(user_input=None, external_stimulus=0.3)
        record()
        if step % 75 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Explore={m['exploration_rate']:.4f} "
                  f"DA={m['da']:.3f} NE={m['ne']:.3f} "
                  f"Motiv={m['motivation_lambda']:.3f} Panic={m['symptom_panic']:.3f}")

    stimulant_explore = np.mean(history["exploration_rate"][-300:])
    stimulant_motiv = np.mean(history["motivation_lambda"][-300:])

    # ── Phase 3: Washout (200步) ──
    print("\n[Phase 3] WASHOUT — pharma.reset() + 自然恢复 (200步)")
    agent.pharma.reset()

    for step in range(200):
        total_step = 1200 + step
        agent.step(user_input=None, external_stimulus=0.1)
        record()
        if step % 50 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Explore={m['exploration_rate']:.4f} "
                  f"DA={m['da']:.3f} GABA={m['gaba']:.3f} "
                  f"5HT={m['5ht']:.3f}")

    washout_explore = np.mean(history["exploration_rate"][-200:])
    washout_motiv = np.mean(history["motivation_lambda"][-200:])

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验2 结果汇总 (REAL Simulacrum AGENT)")
    print("=" * 70)

    print(f"\n{'指标':<25s} {'Baseline':>10s} {'致幻剂':>10s} {'镇静剂':>10s} {'兴奋剂':>10s} {'Washout':>10s}")
    print("-" * 75)
    print(f"{'探索率 (Exploration)':<25s} {baseline_explore:>10.4f} {hallucinogen_explore:>10.4f} "
          f"{sedative_explore:>10.4f} {stimulant_explore:>10.4f} {washout_explore:>10.4f}")
    print(f"{'内在动机 (Lambda)':<25s} {baseline_motiv:>10.3f} {hallucinogen_motiv:>10.3f} "
          f"{sedative_motiv:>10.3f} {stimulant_motiv:>10.3f} {washout_motiv:>10.3f}")
    print(f"{'DA水平':<25s} {np.mean(history['da'][:200]):>10.3f} {np.mean(history['da'][200:500]):>10.3f} "
          f"{np.mean(history['da'][550:850]):>10.3f} {np.mean(history['da'][900:1200]):>10.3f} "
          f"{np.mean(history['da'][-200:]):>10.3f}")
    print(f"{'GABA水平':<25s} {np.mean(history['gaba'][:200]):>10.3f} {np.mean(history['gaba'][200:500]):>10.3f} "
          f"{np.mean(history['gaba'][550:850]):>10.3f} {np.mean(history['gaba'][900:1200]):>10.3f} "
          f"{np.mean(history['gaba'][-200:]):>10.3f}")
    print(f"{'5-HT水平':<25s} {np.mean(history['5ht'][:200]):>10.3f} {np.mean(history['5ht'][200:500]):>10.3f} "
          f"{np.mean(history['5ht'][550:850]):>10.3f} {np.mean(history['5ht'][900:1200]):>10.3f} "
          f"{np.mean(history['5ht'][-200:]):>10.3f}")

    # 关键验证
    hall_explore_change = (hallucinogen_explore - baseline_explore) / max(baseline_explore, 0.001) * 100
    hall_motiv_change = (hallucinogen_motiv - baseline_motiv) / max(baseline_motiv, 0.001) * 100
    sed_explore_change = (sedative_explore - baseline_explore) / max(baseline_explore, 0.001) * 100
    stim_explore_change = (stimulant_explore - baseline_explore) / max(baseline_explore, 0.001) * 100

    print(f"\nKey validation (REAL modules, emergent behavior):")
    print(f"  Hallucinogen->Explore change: {hall_explore_change:+.1f}% "
          f"{'[PASS] Divergent exploration' if abs(hall_explore_change) > 20 else '[INFO] Minor change'}")
    print(f"  Hallucinogen->Motivation change: {hall_motiv_change:+.1f}% "
          f"{'[PASS] Motivation shift' if abs(hall_motiv_change) > 15 else '[INFO] Minor change'}")
    print(f"  Sedative->Explore change: {sed_explore_change:+.1f}% "
          f"{'[PASS] Conservative (reduced)' if sed_explore_change < -15 else '[INFO] Minor change'}")
    print(f"  Stimulant->Explore change: {stim_explore_change:+.1f}% "
          f"{'[PASS] Impulsive (increased)' if stim_explore_change > 15 else '[INFO] Minor change'}")
    print(f"  Washout->Explore recovery: {washout_explore:.4f} vs Baseline {baseline_explore:.4f} "
          f"{'[PASS] Recovered' if abs(washout_explore - baseline_explore) < 0.03 else '[INFO] Not fully recovered'}")

    return history


if __name__ == "__main__":
    run_experiment()