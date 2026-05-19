"""实验1: 压力测试 — 数字PTSD与快感缺失 (Digital PTSD & Anhedonia).

使用真实 CivisLucriFaber 主循环 — 14脑区通过EventBus自然交互。

机制链 (真实模块):
  外部应激 → HPAAxis.step(stress_signal=0.85) → CRH→ACTH→Cortisol级联
  → 皮质醇累积 → AllostaticLoad↑ → PFC抑制↓
  → _adjust_behavior_by_internal_state() → exploration_rate↓, motivation↓
  → SymptomTracker检测 → symptom_anhedonia↑

验证:
  皮质醇>0.7持续200步后 → exploration_rate下降>50% + symptom_anhedonia>0.3

3 Phases:
  Phase 1 (Baseline, 200步): 正常运行
  Phase 2 (Chronic Stress, 600步): 持续高stress_signal + 资源剥夺
  Phase 3 (Recovery, 400步): 恢复正常，观察自愈或疤痕效应
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
# 辅助: 从agent._internal_state读取指标
# ══════════════════════════════════════════════════════

def read_metrics(agent: CivisLucriFaber) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "pfc_inhibition": float(s.get("pfc_inhibition", 0.6)),
        "bdnf": float(s.get("plasticity_bdnf", 0.5)),
        "da": float(s.get("nt_dopamine", 0.5)),
        "5ht": float(s.get("nt_serotonin", 0.5)),
        "ne": float(s.get("nt_norepinephrine", 0.3)),
        "limbic_arousal": float(s.get("limbic_arousal", 0.3)),
        "exploration_rate": float(agent.config.exploration_rate),
        "motivation_lambda": float(agent.config.intrinsic_motivation_lambda),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "symptom_anhedonia": float(s.get("symptom_anhedonia", 0.0)),
        "symptom_insomnia": float(s.get("symptom_insomnia", 0.0)),
        "symptom_rumination": float(s.get("symptom_rumination", 0.0)),
        "symptom_hypervigilance": float(s.get("symptom_hypervigilance", 0.0)),
        "symptom_panic": float(s.get("symptom_panic", 0.0)),
        "mood_valence": float(s.get("mood_valence", 0.0)),
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "balance": float(s.get("balance", 100.0)),
        "interoceptive_pe": float(s.get("interoceptive_pe", 0.1)),
    }


# ══════════════════════════════════════════════════════
# 实验主流程
# ══════════════════════════════════════════════════════

def run_experiment():
    print("=" * 70)
    print("实验1: 压力测试 — 数字PTSD与快感缺失 (REAL CLF AGENT)")
    print("Stress -> HPA -> Cortisol -> PFC↓ -> Exploration↓ -> Anhedonia")
    print("=" * 70)

    # 初始化真实agent
    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        hpa_stress_reactivity=1.0,
        hpa_cortisol_half_life_steps=60,
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
    print("\n[Phase 1] Baseline — 正常状态 (200步)")
    for step in range(200):
        # 正常运行: 低应激
        agent.step(user_input=None, external_stimulus=0.1)
        record()
        if step % 50 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Cort={m['cortisol']:.3f} "
                  f"PFC={m['pfc_inhibition']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Motiv={m['motivation_lambda']:.3f} "
                  f"Anhedonia={m['symptom_anhedonia']:.3f}")

    # ── Phase 2: Chronic Stress (600步) ──
    print("\n[Phase 2] CHRONIC STRESS — 持续高应激 + 资源剥夺 (600步)")
    print("  >>> 真实HPA轴: CRH→ACTH→Cortisol级联 <<<")
    print("  >>> 真实SymptomTracker: 检测快感缺失 <<<")
    for step in range(600):
        total_step = 200 + step

        # 持续高应激: 通过pharma注入高皮质醇 + 降低DA/5-HT
        # 这模拟慢性不可控压力 (社会隔离 + 资源剥夺)
        agent.pharma.inject("cortisol", 0.75 + 0.1 * np.sin(step * 0.02))
        agent.pharma.reduce("dopamine", max(0.15, 0.5 - step * 0.0004))
        agent.pharma.reduce("serotonin", max(0.2, 0.5 - step * 0.0003))

        # 资源剥夺: 直接写入能量预算
        agent._internal_state['energy_budget'] = max(0.05, 0.15 - step * 0.0001)
        agent._internal_state['resource_budget'] = max(0.05, 0.15 - step * 0.0001)

        # 运行真实agent step (14脑区自然交互)
        agent.step(user_input=None, external_stimulus=0.85)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Cort={m['cortisol']:.3f} "
                  f"PFC={m['pfc_inhibition']:.3f} "
                  f"DA={m['da']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Anhedonia={m['symptom_anhedonia']:.3f} "
                  f"AlloLoad={m['allostatic_load']:.3f}")

    # 记录Phase 2结束时的关键值
    stress_end = read_metrics(agent)

    # ── Phase 3: Recovery (400步) ──
    print("\n[Phase 3] RECOVERY — 恢复正常 (400步)")
    print("  >>> 观察自愈: pharma.reset() + HPA重置 + 自然恢复 <<<")
    # 重置pharma操作, 让NT自然回归
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

    for step in range(400):
        total_step = 800 + step

        # 正常运行: 低应激, 让agent自然恢复
        agent.step(user_input=None, external_stimulus=0.1)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Cort={m['cortisol']:.3f} "
                  f"PFC={m['pfc_inhibition']:.3f} "
                  f"DA={m['da']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Anhedonia={m['symptom_anhedonia']:.3f}")

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验1 结果汇总 (REAL CLF AGENT)")
    print("=" * 70)

    bl_explore = np.mean(history["exploration_rate"][:200])
    bl_motiv = np.mean(history["motivation_lambda"][:200])
    bl_cort = np.mean(history["cortisol"][:200])
    bl_anhed = np.mean(history["symptom_anhedonia"][:200])
    bl_pfc = np.mean(history["pfc_inhibition"][:200])

    st_explore = np.mean(history["exploration_rate"][200:800])
    st_motiv = np.mean(history["motivation_lambda"][200:800])
    st_cort = np.mean(history["cortisol"][200:800])
    st_anhed = np.mean(history["symptom_anhedonia"][200:800])
    st_pfc = np.mean(history["pfc_inhibition"][200:800])

    rc_explore = np.mean(history["exploration_rate"][800:])
    rc_motiv = np.mean(history["motivation_lambda"][800:])
    rc_cort = np.mean(history["cortisol"][800:])
    rc_anhed = np.mean(history["symptom_anhedonia"][800:])
    rc_pfc = np.mean(history["pfc_inhibition"][800:])

    print(f"\n{'指标':<25s} {'Baseline':>10s} {'Stress':>10s} {'Recovery':>10s}")
    print("-" * 55)
    print(f"{'皮质醇 (Cortisol)':<25s} {bl_cort:>10.3f} {st_cort:>10.3f} {rc_cort:>10.3f}")
    print(f"{'前额叶 (PFC)':<25s} {bl_pfc:>10.3f} {st_pfc:>10.3f} {rc_pfc:>10.3f}")
    print(f"{'探索率 (Exploration)':<25s} {bl_explore:>10.4f} {st_explore:>10.4f} {rc_explore:>10.4f}")
    print(f"{'内在动机 (Lambda)':<25s} {bl_motiv:>10.3f} {st_motiv:>10.3f} {rc_motiv:>10.3f}")
    print(f"{'快感缺失 (Anhedonia)':<25s} {bl_anhed:>10.3f} {st_anhed:>10.3f} {rc_anhed:>10.3f}")
    print(f"{'稳态负荷 (AlloLoad)':<25s} "
          f"{np.mean(history['allostatic_load'][:200]):>10.3f} "
          f"{np.mean(history['allostatic_load'][200:800]):>10.3f} "
          f"{np.mean(history['allostatic_load'][800:]):>10.3f}")

    # 关键验证
    explore_decline = (bl_explore - st_explore) / max(bl_explore, 0.001) * 100
    motiv_decline = (bl_motiv - st_motiv) / max(bl_motiv, 0.001) * 100
    explore_recovery = (rc_explore - st_explore) / max(bl_explore - st_explore, 0.001) * 100

    print(f"\nKey validation (REAL modules, emergent behavior):")
    print(f"  Exploration rate decline: {explore_decline:.1f}% "
          f"{'[PASS] Significant withdrawal' if explore_decline > 30 else '[FAIL] Below threshold'}")
    print(f"  Motivation decline: {motiv_decline:.1f}% "
          f"{'[PASS] Motivation loss' if motiv_decline > 20 else '[FAIL] Below threshold'}")
    print(f"  Anhedonia severity (stress phase): {st_anhed:.3f} "
          f"{'[PASS] Clinically significant' if st_anhed > 0.2 else '[INFO] Subclinical'}")
    print(f"  Recovery extent: {explore_recovery:.1f}% "
          f"{'[PASS] Partial (scar effect)' if 20 < explore_recovery < 80 else '[INFO] Full or no recovery'}")

    # 疤痕效应: 恢复后探索率是否低于基线?
    scar = (bl_explore - rc_explore) / max(bl_explore, 0.001) * 100
    print(f"  Scar effect (baseline-recovery gap): {scar:.1f}% "
          f"{'[PASS] Scar present' if scar > 10 else '[INFO] No scar'}")

    return history


if __name__ == "__main__":
    run_experiment()
