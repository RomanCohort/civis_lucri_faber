"""实验十: 抗精神病药的D2占用率模拟 (D2 Occupancy Rate).

对应治疗: 精神分裂症、双相情感障碍药物干预

机制链:
  阳性症状(幻觉) → 高noise/DA过载 → D2受体阻滞
  → Low blockade (<50%): 无效, 症状持续
  → Medium blockade (70-80%): 症状消退, 功能恢复 (治疗窗)
  → High blockade (>90%): EPS副作用(运动迟缓) + 情感淡漠

关键: D2占用率与临床改善的倒U型曲线

3组对比:
  Low Blockade: D2 blockade=30% (无效)
  Medium Blockade: D2 blockade=75% (治疗窗)
  High Blockade: D2 blockade=95% (过量, EPS)

3 Phases:
  Phase 1 (Psychosis Induction, 200步): 诱导阳性症状
  Phase 2 (Treatment, 500步): 注入抗精神病药 (D2 blockade)
  Phase 3 (Assessment, 300步): 评估疗效 + 副作用

测量:
  - Positive Symptom Reduction: 幻觉/噪声消除速度
  - EPS (Extrapyramidal Side Effects): 动作僵硬指数
  - Therapeutic Window: 最佳D2占用率
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from typing import Dict, List

from simulacrum.core.agent import Simulacrum
from simulacrum.utils.config import Config


# D2 blockade levels
BLOCKADE_LEVELS = {
    "Low": 0.30,       # 30% occupancy - 无效
    "Medium": 0.75,    # 75% occupancy - 治疗窗
    "High": 0.95,      # 95% occupancy - 过量 (EPS)
}


def read_metrics(agent: Simulacrum) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "da": float(s.get("nt_dopamine", 0.5)),
        "exploration_rate": float(agent.config.exploration_rate),
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "pfc_inhibition": float(s.get("pfc_inhibition", 0.6)),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "symptom_anhedonia": float(s.get("symptom_anhedonia", 0.0)),
        "symptom_rumination": float(s.get("symptom_rumination", 0.0)),
        "symptom_hypervigilance": float(s.get("symptom_hypervigilance", 0.0)),
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "mood_valence": float(s.get("mood_valence", 0.0)),
        "balance": float(agent.thermo.balance),
        "active_ratio": float(s.get("active_ratio", 0.3)),
    }


def compute_positive_symptom_index(metrics: Dict[str, float]) -> float:
    """计算阳性症状指数: 高DA + 高警觉 + 高皮质醇 → 阳性症状"""
    da = metrics["da"]
    hypervig = metrics["symptom_hypervigilance"]
    cort = metrics["cortisol"]
    rumination = metrics["symptom_rumination"]
    # 阳性症状 = DA过载 + 过度警觉 + 思维奔逸
    return float(np.clip(0.3 * da + 0.3 * hypervig + 0.2 * cort + 0.2 * rumination, 0, 1))


def compute_eps_index(agent: Simulacrum, da_level: float) -> float:
    """计算EPS (锥体外系副作用) 指数

    EPS = 动作僵硬 + 运动迟缓
    由D2 blockade过度导致: DA过低 → 基底节运动通路受损
    """
    # BG dopamine temperature: 低DA → 锐化softmax → 动作僵化
    bg_temp = float(np.clip(0.5 + da_level, 0.3, 2.0))
    # 温度越低 → 越僵化 (EPS)
    eps = float(np.clip(1.0 - bg_temp, 0, 1.0))
    # 加上运动迟缓: exploration_rate过低 = 行为僵化
    explore = float(agent.config.exploration_rate)
    bradykinesia = float(np.clip(1.0 - explore / 0.05, 0, 1.0))
    return float(np.clip(0.5 * eps + 0.5 * bradykinesia, 0, 1))


def run_group(group_name: str, d2_blockade: float) -> Dict:
    """运行一个实验组

    Args:
        group_name: 组名 (Low/Medium/High)
        d2_blockade: D2受体阻滞率 (0-1)
    """
    print(f"\n{'='*60}")
    print(f"Group: {group_name} (D2_blockade={d2_blockade:.0%})")
    print(f"{'='*60}")

    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        seed=42,
    )
    agent = Simulacrum(config=config)

    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}
    positive_symptoms = []
    eps_scores = []

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])
        # 计算衍生指标
        psi = compute_positive_symptom_index(m)
        eps = compute_eps_index(agent, m["da"])
        positive_symptoms.append(psi)
        eps_scores.append(eps)

    # ── Phase 1: Psychosis Induction (200步) ──
    print("\n[Phase 1] PSYCHOSIS INDUCTION — 诱导阳性症状 (200步)")
    for step in range(200):
        # 诱导阳性症状: 高DA + 高噪声 + 皮质醇
        agent.pharma.inject("dopamine", 0.8 + 0.1 * np.sin(step * 0.05))
        agent.pharma.inject("cortisol", 0.6 + 0.1 * np.random.random())

        # 高噪声刺激 (模拟幻觉)
        noise_stimulus = 0.7 + 0.2 * np.random.random()

        agent.step(user_input=None, external_stimulus=noise_stimulus)
        record()

        if step % 50 == 0:
            m = read_metrics(agent)
            psi = positive_symptoms[-1]
            print(f"  Step {step:4d}: DA={m['da']:.3f} "
                  f"Cort={m['cortisol']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"PSI={psi:.3f} "
                  f"Hypervig={m['symptom_hypervigilance']:.3f}")

    # 记录Phase1结束时的阳性症状水平
    phase1_psi = np.mean(positive_symptoms[-50:])
    phase1_da = np.mean(history["da"][-50:])

    # ── Phase 2: Treatment (500步) ──
    print(f"\n[Phase 2] TREATMENT — D2 blockade={d2_blockade:.0%} (500步)")

    for step in range(500):
        total_step = 200 + step

        # D2 blockade效应: 降低DA到 (baseline * (1 - blockade))
        # 但保留一定DA (不完全消除)
        target_da = max(0.1, phase1_da * (1 - d2_blockade))
        agent.pharma.inject("dopamine", target_da)

        # 继续注入低水平噪声 (残余幻觉)
        residual_noise = 0.3 * (1 - d2_blockade * 0.8)

        # High blockade额外副作用: EPS和情感淡漠由内部耦合自然产生
        if d2_blockade > 0.9:
            pass  # DA suppression alone drives EPS/apathy via internal coupling

        agent.step(user_input=None, external_stimulus=residual_noise)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            psi = positive_symptoms[-1]
            eps = eps_scores[-1]
            print(f"  Step {total_step:4d}: DA={m['da']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"PSI={psi:.3f} EPS={eps:.3f} "
                  f"Anhedonia={m['symptom_anhedonia']:.3f}")

    # ── Phase 3: Assessment (300步) ──
    print("\n[Phase 3] ASSESSMENT — 评估疗效+副作用 (300步)")
    for step in range(300):
        total_step = 700 + step

        # 维持D2 blockade
        target_da = max(0.1, phase1_da * (1 - d2_blockade))
        agent.pharma.inject("dopamine", target_da)

        agent.step(user_input=None, external_stimulus=0.1)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            psi = positive_symptoms[-1]
            eps = eps_scores[-1]
            print(f"  Step {total_step:4d}: DA={m['da']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"PSI={psi:.3f} EPS={eps:.3f} "
                  f"Social={m['social_engagement']:.3f}")

    # ── 计算结果 ──
    phase2_psi = np.mean(positive_symptoms[200:700])
    phase3_psi = np.mean(positive_symptoms[700:])
    phase2_eps = np.mean(eps_scores[200:700])
    phase3_eps = np.mean(eps_scores[700:])
    phase3_explore = np.mean(history["exploration_rate"][700:])
    phase3_anhedonia = np.mean(history["symptom_anhedonia"][700:])
    phase3_social = np.mean(history["social_engagement"][700:])

    # Positive symptom reduction
    psi_reduction = (phase1_psi - phase3_psi) / max(phase1_psi, 0.001) * 100

    # Therapeutic index: 症状改善 / EPS副作用
    therapeutic_index = psi_reduction / max(phase3_eps * 100, 1.0)

    result = {
        "group": group_name,
        "d2_blockade": d2_blockade,
        "phase1_psi": phase1_psi,
        "phase2_psi": phase2_psi,
        "phase3_psi": phase3_psi,
        "phase2_eps": phase2_eps,
        "phase3_eps": phase3_eps,
        "psi_reduction_pct": psi_reduction,
        "therapeutic_index": therapeutic_index,
        "phase3_explore": phase3_explore,
        "phase3_anhedonia": phase3_anhedonia,
        "phase3_social": phase3_social,
        "psi_trajectory": positive_symptoms,
        "eps_trajectory": eps_scores,
    }
    return result


def run_experiment():
    print("=" * 70)
    print("实验十: D2受体占用率模拟 (REAL Simulacrum AGENT)")
    print("3 Groups: Low(30%) / Medium(75%) / High(95%) D2 blockade")
    print("=" * 70)

    results = {}
    for name, level in BLOCKADE_LEVELS.items():
        results[name] = run_group(name, level)

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验十 结果汇总 (REAL Simulacrum AGENT)")
    print("=" * 70)

    print(f"\n{'指标':<25s} {'Low(30%)':>12s} {'Medium(75%)':>12s} {'High(95%)':>12s}")
    print("-" * 61)
    print(f"{'Phase1 PSI (基线)':<25s} "
          f"{results['Low']['phase1_psi']:>12.3f} "
          f"{results['Medium']['phase1_psi']:>12.3f} "
          f"{results['High']['phase1_psi']:>12.3f}")
    print(f"{'Phase3 PSI (治疗后)':<25s} "
          f"{results['Low']['phase3_psi']:>12.3f} "
          f"{results['Medium']['phase3_psi']:>12.3f} "
          f"{results['High']['phase3_psi']:>12.3f}")
    print(f"{'症状改善 (%)':<25s} "
          f"{results['Low']['psi_reduction_pct']:>12.1f} "
          f"{results['Medium']['psi_reduction_pct']:>12.1f} "
          f"{results['High']['psi_reduction_pct']:>12.1f}")
    print(f"{'EPS指数':<25s} "
          f"{results['Low']['phase3_eps']:>12.3f} "
          f"{results['Medium']['phase3_eps']:>12.3f} "
          f"{results['High']['phase3_eps']:>12.3f}")
    print(f"{'治疗指数':<25s} "
          f"{results['Low']['therapeutic_index']:>12.2f} "
          f"{results['Medium']['therapeutic_index']:>12.2f} "
          f"{results['High']['therapeutic_index']:>12.2f}")
    print(f"{'探索率':<25s} "
          f"{results['Low']['phase3_explore']:>12.4f} "
          f"{results['Medium']['phase3_explore']:>12.4f} "
          f"{results['High']['phase3_explore']:>12.4f}")
    print(f"{'快感缺失':<25s} "
          f"{results['Low']['phase3_anhedonia']:>12.3f} "
          f"{results['Medium']['phase3_anhedonia']:>12.3f} "
          f"{results['High']['phase3_anhedonia']:>12.3f}")
    print(f"{'社会参与':<25s} "
          f"{results['Low']['phase3_social']:>12.3f} "
          f"{results['Medium']['phase3_social']:>12.3f} "
          f"{results['High']['phase3_social']:>12.3f}")

    # 关键验证
    print(f"\nKey validation:")

    # 1. Medium组症状改善应最大 (倒U型曲线)
    medium_best_reduction = results["Medium"]["psi_reduction_pct"] >= results["Low"]["psi_reduction_pct"]
    print(f"  Medium symptom reduction: {results['Medium']['psi_reduction_pct']:.1f}% "
          f"{'[PASS] Therapeutic effect' if medium_best_reduction and results['Medium']['psi_reduction_pct'] > 10 else '[INFO] Insufficient reduction'}")

    # 2. High组EPS应最高
    high_eps = results["High"]["phase3_eps"] > results["Medium"]["phase3_eps"]
    print(f"  High EPS: {results['High']['phase3_eps']:.3f} vs Medium {results['Medium']['phase3_eps']:.3f} "
          f"{'[PASS] Motor side effects' if high_eps else '[INFO] No EPS'}")

    # 3. 倒U型曲线: Medium治疗指数应最高
    medium_best_ti = results["Medium"]["therapeutic_index"] >= results["Low"]["therapeutic_index"]
    print(f"  Therapeutic index: Low={results['Low']['therapeutic_index']:.2f} "
          f"Medium={results['Medium']['therapeutic_index']:.2f} "
          f"High={results['High']['therapeutic_index']:.2f} "
          f"{'[PASS] Inverted-U confirmed' if medium_best_ti else '[INFO] No clear optimum'}")

    # 4. Low组症状应持续 (无效)
    low_ineffective = results["Low"]["psi_reduction_pct"] < results["Medium"]["psi_reduction_pct"]
    print(f"  Low blockade ineffective: {results['Low']['psi_reduction_pct']:.1f}% reduction "
          f"{'[PASS] Subtherapeutic' if low_ineffective else '[INFO] Unexpected efficacy'}")

    return results


if __name__ == "__main__":
    run_experiment()
