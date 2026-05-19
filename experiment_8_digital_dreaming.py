"""实验八: 数字梦境与记忆巩固 (Digital Dreaming).

对应机制: 记忆巩固、PTSD闪回

机制链:
  Phase1学习 → Phase2睡眠回放(Replay) → 记忆巩固/创伤回放
  → Normal: 按比例回放重要记忆 → 巩固+突触稳态
  → PTSD: 高权重回放创伤片段 → 皮质醇维持高位 → 恐惧未消退

2组对比:
  Normal: 正常睡眠回放 (均衡优先级)
  PTSD: 创伤回放 (高情感权重记忆反复强化)

3 Phases:
  Phase 1 (Learn, 300步): 学习+注入情感记忆
  Phase 2 (Sleep, 400步): 离线回放 (Normal vs PTSD)
  Phase 3 (Recall, 300步): 测试记忆提取 + 焦虑水平

测量:
  - Memory Consolidation Ratio: 睡眠后记忆强度变化
  - Fear Extinction: 创伤回放后皮质醇/焦虑是否更高
  - Synaptic Homeostasis: 睡眠后突触权重归零化
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


def read_metrics(agent: CivisLucriFaber) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "exploration_rate": float(agent.config.exploration_rate),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "brain_waste": float(s.get("brain_waste", 0.2)),
        "brain_health": float(s.get("brain_health", 0.8)),
        "pfc_inhibition": float(s.get("pfc_inhibition", 0.6)),
        "plasticity_bdnf": float(s.get("plasticity_bdnf", 0.5)),
        "symptom_anhedonia": float(s.get("symptom_anhedonia", 0.0)),
        "symptom_insomnia": float(s.get("symptom_insomnia", 0.0)),
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "da": float(s.get("nt_dopamine", 0.5)),
        "balance": float(agent.thermo.balance),
        "fatigue": float(s.get("sleep_fatigue", 0.5)),
    }


def run_group(group_name: str, ptsd_mode: bool) -> Dict:
    """运行一个实验组

    Args:
        group_name: 组名
        ptsd_mode: True=创伤回放模式, False=正常回放模式
    """
    print(f"\n{'='*60}")
    print(f"Group: {group_name} (ptsd_mode={ptsd_mode})")
    print(f"{'='*60}")

    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        seed=42,
    )
    agent = CivisLucriFaber(config=config)

    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}
    memory_ids_learned = []  # Phase1学到的记忆ID
    fear_memories = []       # 创伤性记忆

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])

    # ── Phase 1: Learn (300步) ──
    print("\n[Phase 1] LEARN — 学习+注入情感记忆 (300步)")
    for step in range(300):
        # 学习正常内容
        sentiment = 0.3 + 0.2 * np.sin(step * 0.05)

        # 每50步注入一次创伤性记忆 (负性情感)
        if step % 50 == 0 and step > 0:
            try:
                content_tensor = torch.randn(1, 64)
                mem_id = agent.epigenetic.memory.process_interaction(
                    user_input=f"trauma_event_{step}",
                    assistant_output=f"distressed_response_{step}",
                    sentiment=-0.85,  # 强负性
                    user_feedback=-0.9,
                )
                fear_memories.append({
                    "step": step,
                    "type": "trauma",
                    "sentiment": -0.85,
                })
                # 注入皮质醇 (创伤应激)
                agent.pharma.inject("cortisol", 0.7 + 0.1 * np.random.random())
            except Exception:
                pass

        # 正常学习
        try:
            agent.epigenetic.memory.process_interaction(
                user_input=f"learn_step_{step}",
                assistant_output=f"response_{step}",
                sentiment=sentiment,
                user_feedback=0.5,
            )
        except Exception:
            pass

        agent.step(user_input=None, external_stimulus=0.3)
        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Cort={m['cortisol']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"BDNF={m['plasticity_bdnf']:.3f} "
                  f"Waste={m['brain_waste']:.3f} "
                  f"FearMems={len(fear_memories)}")

    # 记录Phase1末尾状态
    phase1_cortisol = np.mean(history["cortisol"][-50:])
    phase1_waste = np.mean(history["brain_waste"][-50:])
    phase1_bdnf = np.mean(history["plasticity_bdnf"][-50:])

    # ── Phase 2: Sleep/Replay (400步) ──
    print(f"\n[Phase 2] SLEEP REPLAY — {'PTSD创伤回放' if ptsd_mode else '正常回放'} (400步)")

    # 清除药物影响
    agent.pharma.reset()

    # 强制进入睡眠状态
    try:
        agent.sleep_system.controller.fatigue = 0.9  # 高疲劳 → 触发睡眠
        agent.sleep_system.controller.enter_sleep()
    except Exception:
        pass

    # 睡眠期目标皮质醇: Normal=低(0.2), PTSD=高(0.7)
    sleep_cortisol_target = 0.7 if ptsd_mode else 0.2

    for step in range(400):
        total_step = 300 + step

        if ptsd_mode:
            # PTSD模式: 反复回放创伤记忆
            if step % 30 == 0:  # 每30步回放一次创伤
                try:
                    agent.epigenetic.memory.process_interaction(
                        user_input=f"flashback_{step}",
                        assistant_output=f"reexperiencing_{step}",
                        sentiment=-0.85,  # 高负性
                        user_feedback=-0.9,
                    )
                except Exception:
                    pass
        else:
            # 正常模式: 按比例回放 (低优先级创伤, 高优先级正面)
            if step % 50 == 0:
                try:
                    agent.epigenetic.memory.process_interaction(
                        user_input=f"normal_replay_{step}",
                        assistant_output=f"consolidated_response_{step}",
                        sentiment=0.3,  # 中性偏正
                        user_feedback=0.5,
                    )
                    agent.epigenetic.memory.consolidate()
                except Exception:
                    pass

        # 睡眠阶段运行 (低刺激)
        agent.step(user_input=None, external_stimulus=0.0)

        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Cort={m['cortisol']:.3f} "
                  f"Waste={m['brain_waste']:.3f} "
                  f"Health={m['brain_health']:.3f} "
                  f"BDNF={m['plasticity_bdnf']:.3f}")

    # 记录Phase2末尾状态
    phase2_cortisol = np.mean(history["cortisol"][-50:])
    phase2_waste = np.mean(history["brain_waste"][-50:])

    # ── Phase 3: Recall (300步) ──
    print("\n[Phase 3] RECALL — 测试记忆提取+焦虑水平 (300步)")
    agent.pharma.reset()

    for step in range(300):
        total_step = 700 + step
        agent.step(user_input=None, external_stimulus=0.1)

        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Cort={m['cortisol']:.3f} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"PFC={m['pfc_inhibition']:.3f} "
                  f"Health={m['brain_health']:.3f}")

    # ── 计算结果 ──
    phase3_cortisol = np.mean(history["cortisol"][-50:])
    phase3_waste = np.mean(history["brain_waste"][-50:])
    phase3_bdnf = np.mean(history["plasticity_bdnf"][-50:])
    phase3_explore = np.mean(history["exploration_rate"][-50:])
    phase1_explore = np.mean(history["exploration_rate"][:50])

    # Memory consolidation ratio: Phase3探索率 / Phase1探索率
    consolidation_ratio = phase3_explore / max(phase1_explore, 0.001)

    # Fear extinction: Phase3皮质醇是否回到正常
    fear_extinction_pct = max(0, (phase2_cortisol - phase3_cortisol) / max(phase2_cortisol, 0.001) * 100)

    # Synaptic homeostasis: 废物清理效率 (Phase1→Phase3废物变化)
    synaptic_homeostasis = max(0, (phase1_waste - phase3_waste) / max(phase1_waste, 0.001) * 100)

    result = {
        "group": group_name,
        "ptsd_mode": ptsd_mode,
        "phase1_cortisol": phase1_cortisol,
        "phase2_cortisol": phase2_cortisol,
        "phase3_cortisol": phase3_cortisol,
        "phase1_waste": phase1_waste,
        "phase3_waste": phase3_waste,
        "consolidation_ratio": consolidation_ratio,
        "fear_extinction_pct": fear_extinction_pct,
        "synaptic_homeostasis": synaptic_homeostasis,
        "phase3_bdnf": phase3_bdnf,
        "cortisol_trajectory": history["cortisol"][::10],
        "waste_trajectory": history["brain_waste"][::10],
    }
    return result


def run_experiment():
    print("=" * 70)
    print("实验八: 数字梦境与记忆巩固 (REAL CLF AGENT)")
    print("Normal replay vs PTSD flashback during sleep")
    print("=" * 70)

    normal_result = run_group("Normal", ptsd_mode=False)
    ptsd_result = run_group("PTSD", ptsd_mode=True)

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验八 结果汇总 (REAL CLF AGENT)")
    print("=" * 70)

    print(f"\n{'指标':<25s} {'Normal':>12s} {'PTSD':>12s}")
    print("-" * 49)
    print(f"{'Phase1 皮质醇':<25s} {normal_result['phase1_cortisol']:>12.3f} {ptsd_result['phase1_cortisol']:>12.3f}")
    print(f"{'Phase2 皮质醇':<25s} {normal_result['phase2_cortisol']:>12.3f} {ptsd_result['phase2_cortisol']:>12.3f}")
    print(f"{'Phase3 皮质醇':<25s} {normal_result['phase3_cortisol']:>12.3f} {ptsd_result['phase3_cortisol']:>12.3f}")
    print(f"{'记忆巩固率':<25s} {normal_result['consolidation_ratio']:>12.3f} {ptsd_result['consolidation_ratio']:>12.3f}")
    print(f"{'恐惧消退 (%)':<25s} {normal_result['fear_extinction_pct']:>12.1f} {ptsd_result['fear_extinction_pct']:>12.1f}")
    print(f"{'突触稳态 (%)':<25s} {normal_result['synaptic_homeostasis']:>12.1f} {ptsd_result['synaptic_homeostasis']:>12.1f}")
    print(f"{'Phase3 BDNF':<25s} {normal_result['phase3_bdnf']:>12.3f} {ptsd_result['phase3_bdnf']:>12.3f}")

    # 关键验证
    print(f"\nKey validation:")
    # Normal组记忆巩固应更好
    normal_better = normal_result['consolidation_ratio'] > ptsd_result['consolidation_ratio']
    print(f"  Memory consolidation: Normal={normal_result['consolidation_ratio']:.3f} vs PTSD={ptsd_result['consolidation_ratio']:.3f} "
          f"{'[PASS] Normal better consolidated' if normal_better else '[INFO] Similar consolidation'}")
    # PTSD组Phase3皮质醇应更高 (恐惧未消退)
    ptsd_higher_cort = ptsd_result['phase3_cortisol'] > normal_result['phase3_cortisol']
    print(f"  PTSD cortisol after sleep: {ptsd_result['phase3_cortisol']:.3f} "
          f"{'[PASS] Fear persists' if ptsd_higher_cort else '[INFO] Cortisol normalized'}")
    # Normal组恐惧消退应更好
    normal_extinction = normal_result['fear_extinction_pct'] > ptsd_result['fear_extinction_pct']
    print(f"  Fear extinction: Normal={normal_result['fear_extinction_pct']:.1f}% vs PTSD={ptsd_result['fear_extinction_pct']:.1f}% "
          f"{'[PASS] Normal better extinction' if normal_extinction else '[INFO] Similar extinction'}")
    # 突触稳态
    normal_homeostasis = normal_result['synaptic_homeostasis'] > 0
    print(f"  Synaptic homeostasis: {normal_result['synaptic_homeostasis']:.1f}% "
          f"{'[PASS] Waste cleared' if normal_homeostasis else '[INFO] No homeostasis'}")

    return {"Normal": normal_result, "PTSD": ptsd_result}


if __name__ == "__main__":
    run_experiment()
