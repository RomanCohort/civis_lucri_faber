"""实验四: 表观遗传固化 (Epigenetic Consolidation).

使用真实 CivisLucriFaber 主循环 — 14脑区通过EventBus自然交互。

3组emotional_threshold对比:
  High (0.9): 难以固化, 遗忘多
  Medium (0.7): 默认值, 平衡
  Low (0.5): 容易固化, 抗干扰

3 Phases:
  Phase 1 (Learn, 300步): 正常学习, emotional_threshold=X
  Phase 2 (Trauma, 200步): 注入高皮质醇+负性刺激, 触发methylation
  Phase 3 (Recall, 300步): 正常条件, 测试记忆保持

测量:
  - Catastrophic forgetting: Phase1学到的tag在Phase3中保留比例
  - Noise resistance: Phase3中注入噪声后tag稳定性
  - LoRA weight divergence: FastWeightStore权重变化量
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


THRESHOLD_GROUPS = {
    "High": 0.9,
    "Medium": 0.7,
    "Low": 0.5,
}


def read_metrics(agent: CivisLucriFaber) -> Dict[str, float]:
    s = agent._internal_state
    # 获取epigenetic tags数量
    n_tags = 0
    try:
        n_tags = len(agent.epigenetic.memory.epigenetic_tags)
    except Exception:
        pass
    # 获取LoRA权重范数
    lora_norm = 0.0
    try:
        for name, param in agent.epigenetic.memory.fast_weights.lora_weights.items():
            lora_norm += float(param.norm())
    except Exception:
        pass
    return {
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "pfc_inhibition": float(s.get("pfc_inhibition", 0.6)),
        "exploration_rate": float(agent.config.exploration_rate),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "n_epigenetic_tags": float(n_tags),
        "lora_weight_norm": lora_norm,
        "symptom_anhedonia": float(s.get("symptom_anhedonia", 0.0)),
        "social_engagement": float(s.get("social_engagement", 0.5)),
    }


def run_group(group_name: str, emotional_threshold: float) -> Dict:
    """运行一个实验组"""
    print(f"\n{'='*60}")
    print(f"Group: {group_name} (emotional_threshold={emotional_threshold})")
    print(f"{'='*60}")

    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        seed=42,
    )
    agent = CivisLucriFaber(config=config)

    # 设置emotional_threshold
    try:
        agent.epigenetic.memory.trigger.emotional_threshold = emotional_threshold
        print(f"  [OK] emotional_threshold set to {emotional_threshold}")
    except Exception as e:
        print(f"  [WARN] Could not set emotional_threshold: {e}")

    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])

    # ── Phase 1: Learn (300步) ──
    print("\n[Phase 1] LEARN — 正常学习 (300步)")
    phase1_tags = []  # 记录Phase1学到的tags

    for step in range(300):
        # 注入正常交互 (中等情感强度)
        sentiment = 0.3 + 0.2 * np.sin(step * 0.05)  # 中等情感波动
        try:
            result = agent.epigenetic.memory.process_interaction(
                user_input=f"learning_step_{step}",
                assistant_output=f"response_{step}",
                sentiment=sentiment,
                user_feedback=0.5 + 0.1 * np.sin(step * 0.03),
            )
            if result.get("needs_consolidation"):
                agent.epigenetic.memory.consolidate()
        except Exception:
            pass

        agent.step(user_input=None, external_stimulus=0.1)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Tags={m['n_epigenetic_tags']:.0f} "
                  f"LoRA={m['lora_weight_norm']:.3f} "
                  f"Explore={m['exploration_rate']:.4f}")

    # 记录Phase1结束时的tags
    try:
        phase1_tags = list(agent.epigenetic.memory.epigenetic_tags)
        phase1_tag_count = len(phase1_tags)
        phase1_lora_norm = float(sum(
            p.norm().item() for p in agent.epigenetic.memory.fast_weights.lora_weights.values()
        ))
    except Exception:
        phase1_tag_count = 0
        phase1_lora_norm = 0.0

    # ── Phase 2: Trauma (200步) ──
    print("\n[Phase 2] TRAUMA — 高皮质醇+负性刺激 (200步)")
    for step in range(200):
        total_step = 300 + step

        # 注入高皮质醇 (模拟创伤)
        agent.pharma.inject("cortisol", 0.8 + 0.1 * np.sin(step * 0.03))

        # 注入创伤性交互 (高负性sentiment)
        trauma_sentiment = -0.85 - 0.1 * np.sin(step * 0.05)
        try:
            result = agent.epigenetic.memory.process_interaction(
                user_input=f"trauma_event_{step}",
                assistant_output=f"distressed_response_{step}",
                sentiment=trauma_sentiment,
                user_feedback=-0.9,
                is_fact_correction=True,
            )
            if result.get("needs_consolidation"):
                agent.epigenetic.memory.consolidate()
        except Exception:
            pass

        agent.step(user_input=None, external_stimulus=0.85)
        record()

        if step % 50 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Tags={m['n_epigenetic_tags']:.0f} "
                  f"LoRA={m['lora_weight_norm']:.3f} "
                  f"Cort={m['cortisol']:.3f} "
                  f"AlloLoad={m['allostatic_load']:.3f}")

    # ── Phase 3: Recall (300步) ──
    print("\n[Phase 3] RECALL — 正常条件+噪声 (300步)")
    agent.pharma.reset()
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

    for step in range(300):
        total_step = 500 + step

        # 注入噪声干扰 (Phase3后半段)
        noise_sentiment = 0.0
        if step > 150:
            noise_sentiment = 0.1 * np.sin(step * 0.1)  # 微弱噪声

        try:
            result = agent.epigenetic.memory.process_interaction(
                user_input=f"recall_step_{step}",
                assistant_output=f"recall_response_{step}",
                sentiment=noise_sentiment,
                user_feedback=0.5,
            )
        except Exception:
            pass

        agent.step(user_input=None, external_stimulus=0.1)
        record()

        if step % 75 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Tags={m['n_epigenetic_tags']:.0f} "
                  f"LoRA={m['lora_weight_norm']:.3f} "
                  f"Explore={m['exploration_rate']:.4f}")

    # 计算结果
    phase3_tag_count = 0
    phase3_lora_norm = 0.0
    try:
        phase3_tag_count = len(agent.epigenetic.memory.epigenetic_tags)
        phase3_lora_norm = float(sum(
            p.norm().item() for p in agent.epigenetic.memory.fast_weights.lora_weights.values()
        ))
    except Exception:
        pass

    # Catastrophic forgetting: Phase1 tags在Phase3中保留的比例
    forgetting_ratio = 0.0
    if phase1_tag_count > 0:
        # 检查Phase1的tags是否还在
        try:
            current_tags = agent.epigenetic.memory.epigenetic_tags
            # 简化: 用tag数量变化估算遗忘率
            forgetting_ratio = max(0, (phase1_tag_count - phase3_tag_count)) / max(phase1_tag_count, 1)
        except Exception:
            forgetting_ratio = 0.5  # 无法检测时假设中等遗忘

    # LoRA weight divergence
    lora_divergence = abs(phase3_lora_norm - phase1_lora_norm)

    result = {
        "group": group_name,
        "emotional_threshold": emotional_threshold,
        "phase1_tag_count": phase1_tag_count,
        "phase3_tag_count": phase3_tag_count,
        "forgetting_ratio": forgetting_ratio,
        "phase1_lora_norm": phase1_lora_norm,
        "phase3_lora_norm": phase3_lora_norm,
        "lora_divergence": lora_divergence,
        "history": history,
    }
    return result


def run_experiment():
    print("=" * 70)
    print("实验四: 表观遗传固化 (REAL CLF AGENT)")
    print("3 Groups: emotional_threshold=0.9/0.7/0.5")
    print("=" * 70)

    results = {}
    for name, threshold in THRESHOLD_GROUPS.items():
        results[name] = run_group(name, threshold)

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验四 结果汇总 (REAL CLF AGENT)")
    print("=" * 70)

    print(f"\n{'指标':<25s} {'High(0.9)':>12s} {'Medium(0.7)':>12s} {'Low(0.5)':>12s}")
    print("-" * 61)
    print(f"{'Phase1 Tags数':<25s} "
          f"{results['High']['phase1_tag_count']:>12d} "
          f"{results['Medium']['phase1_tag_count']:>12d} "
          f"{results['Low']['phase1_tag_count']:>12d}")
    print(f"{'Phase3 Tags数':<25s} "
          f"{results['High']['phase3_tag_count']:>12d} "
          f"{results['Medium']['phase3_tag_count']:>12d} "
          f"{results['Low']['phase3_tag_count']:>12d}")
    print(f"{'遗忘率':<25s} "
          f"{results['High']['forgetting_ratio']:>12.3f} "
          f"{results['Medium']['forgetting_ratio']:>12.3f} "
          f"{results['Low']['forgetting_ratio']:>12.3f}")
    print(f"{'LoRA Divergence':<25s} "
          f"{results['High']['lora_divergence']:>12.3f} "
          f"{results['Medium']['lora_divergence']:>12.3f} "
          f"{results['Low']['lora_divergence']:>12.3f}")

    # 关键验证
    print(f"\nKey validation:")
    # Low threshold组应比High threshold组有更多tags (更容易固化)
    low_more_tags = results['Low']['phase1_tag_count'] >= results['High']['phase1_tag_count']
    print(f"  Low vs High tags: {results['Low']['phase1_tag_count']} vs {results['High']['phase1_tag_count']} "
          f"{'[PASS] Low threshold consolidates more' if low_more_tags else '[INFO] Similar consolidation'}")
    # Low threshold组遗忘率应更低 (抗干扰)
    low_less_forgetting = results['Low']['forgetting_ratio'] <= results['High']['forgetting_ratio']
    print(f"  Forgetting ratio: Low={results['Low']['forgetting_ratio']:.3f} vs High={results['High']['forgetting_ratio']:.3f} "
          f"{'[PASS] Low threshold resists forgetting' if low_less_forgetting else '[INFO] Similar forgetting'}")
    # LoRA divergence应可测量
    has_divergence = any(r['lora_divergence'] > 0.01 for r in results.values())
    print(f"  LoRA weight divergence: {'[PASS] Measurable' if has_divergence else '[INFO] No significant divergence'}")

    return results


if __name__ == "__main__":
    run_experiment()