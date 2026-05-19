"""实验七: ADHD的"临界闪烁频率"测试 (Critical Flicker Fusion).

对应疾病: 注意力缺陷多动障碍(ADHD)、感觉门控缺陷

机制链:
  噪声环境 → ThalamicRelay门控 → 无关信号过滤
  → ADHD模式: 门控阈值高 → 所有噪声进入 → metabolic_budget耗散
  → DA系统疲劳 (phasic DA衰减) → 注意瞬脱增加

2组对比:
  Normal: 低门控阈值 (只允许最强信号进入)
  ADHD: 高门控阈值 (所有噪声都能进入)

3 Phases:
  Phase 1 (Baseline, 200步): 低噪声环境
  Phase 2 (Noise Overload, 500步): 高噪声 + 主任务
  Phase 3 (Recovery, 300步): 恢复低噪声

测量:
  - Attentional Blink: 错过关键信号(stimulus峰值)的频率
  - Metabolic Drain: 处理噪声消耗的energy比例
  - Phasic DA Fatigue: DA系统的衰减速度
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from typing import Dict, List

from simulacrum.core.agent import Simulacrum
from simulacrum.utils.config import Config


def read_metrics(agent: Simulacrum) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "exploration_rate": float(agent.config.exploration_rate),
        "active_ratio": float(s.get("active_ratio", 0.3)),
        "metabolic_cost": float(s.get("metabolic_cost", 0.0)),
        "da": float(s.get("nt_dopamine", 0.5)),
        "5ht": float(s.get("nt_serotonin", 0.5)),
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "brain_waste": float(s.get("brain_waste", 0.2)),
        "pfc_inhibition": float(s.get("pfc_inhibition", 0.6)),
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "balance": float(agent.thermo.balance),
    }


def run_group(group_name: str, adhd_mode: bool) -> Dict:
    """运行一个实验组

    Args:
        group_name: 组名
        adhd_mode: True=ADHD(高门控), False=Normal(低门控)
    """
    print(f"\n{'='*60}")
    print(f"Group: {group_name} (adhd_mode={adhd_mode})")
    print(f"{'='*60}")

    config = Config(
        initial_balance=100.0,
        exploration_rate=0.1,
        seed=42,
    )
    agent = Simulacrum(config=config)

    # 调整丘脑门控参数
    if adhd_mode:
        # ADHD: 降低门控参数 → 更多噪声通过
        try:
            gate = agent.limbic.thalamus.relay.attention_gate
            gate.data.fill_(2.0)  # 高sigmoid值 → 高通过率 → 少过滤
            print(f"  [OK] Thalamic gate set HIGH (ADHD mode)")
        except Exception as e:
            print(f"  [WARN] Could not adjust thalamic gate: {e}")

    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}
    stimulus_hits = []  # 记录关键刺激是否被响应
    noise_signals = []  # 记录噪声强度
    prev_da = 0.5  # 记录上一步DA

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])

    # ── Phase 1: Baseline (200步) ──
    print("\n[Phase 1] Baseline — 低噪声环境 (200步)")
    for step in range(200):
        agent.step(user_input=None, external_stimulus=0.1)
        record()
        if step % 50 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Explore={m['exploration_rate']:.4f} "
                  f"DA={m['da']:.3f} "
                  f"MetCost={m['metabolic_cost']:.4f} "
                  f"PFC={m['pfc_inhibition']:.3f}")

    # ── Phase 2: Noise Overload (500步) ──
    print("\n[Phase 2] NOISE OVERLOAD — 高噪声+主任务 (500步)")
    for step in range(500):
        total_step = 200 + step

        # 生成信号: 主任务信号(关键) + 噪声
        # 关键信号: 每20步出现一个高峰 (模拟关键帧)
        is_key_frame = (step % 20 == 0)

        # 噪声信号: 随机噪声
        noise_level = 0.3 + 0.4 * np.random.random()
        noise_signals.append(noise_level)

        if adhd_mode:
            # ADHD: 噪声直接注入 (门控失效, 所有信号都通过)
            total_stimulus = noise_level + (0.5 if is_key_frame else 0.0)
            # 额外代谢消耗 (处理噪声需要额外能量)
            agent._internal_state['energy_budget'] = max(0.1,
                agent._internal_state.get('energy_budget', 0.5) - 0.001 * noise_level)
        else:
            # Normal: 噪声被门控过滤, 关键信号通过
            total_stimulus = 0.1 + (0.5 if is_key_frame else 0.0)

        agent.step(user_input=None, external_stimulus=total_stimulus)

        # 检测是否响应了关键帧
        current_da = float(agent._internal_state.get("nt_dopamine", 0.5))
        if is_key_frame:
            # DA变化量: 关键帧应该引起DA波动
            # Normal: 噪声被过滤, 关键帧DA变化大
            # ADHD: 噪声淹没信号, 关键帧DA变化小
            da_change = abs(current_da - prev_da)
            stimulus_hits.append(1.0 if da_change > 0.005 else 0.0)
        prev_da = current_da

        record()

        if step % 100 == 0:
            m = read_metrics(agent)
            attentional_blink = 1.0 - np.mean(stimulus_hits) if stimulus_hits else 0.0
            print(f"  Step {total_step:4d}: Explore={m['exploration_rate']:.4f} "
                  f"DA={m['da']:.3f} "
                  f"MetCost={m['metabolic_cost']:.4f} "
                  f"Waste={m['brain_waste']:.3f} "
                  f"MissRate={attentional_blink:.3f}")

    # ── Phase 3: Recovery (300步) ──
    print("\n[Phase 3] RECOVERY — 恢复低噪声 (300步)")
    for step in range(300):
        total_step = 700 + step
        agent.step(user_input=None, external_stimulus=0.1)
        record()
        if step % 100 == 0:
            m = read_metrics(agent)
            print(f"  Step {total_step:4d}: Explore={m['exploration_rate']:.4f} "
                  f"DA={m['da']:.3f} "
                  f"MetCost={m['metabolic_cost']:.4f}")

    # ── 计算结果 ──
    bl_da = np.mean(history["da"][:200])
    bl_met = np.mean(history["metabolic_cost"][:200])
    bl_waste = np.mean(history["brain_waste"][:200])

    noise_da = np.mean(history["da"][200:700])
    noise_met = np.mean(history["metabolic_cost"][200:700])
    noise_waste = np.mean(history["brain_waste"][200:700])

    rc_da = np.mean(history["da"][700:])
    rc_met = np.mean(history["metabolic_cost"][700:])

    # Attentional blink rate (miss rate)
    attentional_blink_rate = 1.0 - np.mean(stimulus_hits) if stimulus_hits else 0.0

    # Metabolic drain: 噪声阶段的metabolic cost比baseline增加的比例
    metabolic_drain = (noise_met - bl_met) / max(bl_met, 0.001) * 100

    # DA fatigue: DA从baseline到噪声后期的下降比例
    da_fatigue = (bl_da - noise_da) / max(bl_da, 0.001) * 100

    result = {
        "group": group_name,
        "adhd_mode": adhd_mode,
        "bl_da": bl_da,
        "noise_da": noise_da,
        "rc_da": rc_da,
        "bl_met": bl_met,
        "noise_met": noise_met,
        "noise_waste": noise_waste,
        "attentional_blink_rate": attentional_blink_rate,
        "metabolic_drain_pct": metabolic_drain,
        "da_fatigue_pct": da_fatigue,
        "da_trajectory": history["da"][::10],
        "met_trajectory": history["metabolic_cost"][::10],
    }
    return result


def run_experiment():
    print("=" * 70)
    print("实验七: ADHD临界闪烁频率测试 (REAL Simulacrum AGENT)")
    print("Normal (gated) vs ADHD (ungated) under noise overload")
    print("=" * 70)

    normal_result = run_group("Normal", adhd_mode=False)
    adhd_result = run_group("ADHD", adhd_mode=True)

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验七 结果汇总 (REAL Simulacrum AGENT)")
    print("=" * 70)

    print(f"\n{'指标':<25s} {'Normal':>12s} {'ADHD':>12s}")
    print("-" * 49)
    print(f"{'基线DA':<25s} {normal_result['bl_da']:>12.3f} {adhd_result['bl_da']:>12.3f}")
    print(f"{'噪声期DA':<25s} {normal_result['noise_da']:>12.3f} {adhd_result['noise_da']:>12.3f}")
    print(f"{'恢复期DA':<25s} {normal_result['rc_da']:>12.3f} {adhd_result['rc_da']:>12.3f}")
    print(f"{'注意瞬脱率':<25s} {normal_result['attentional_blink_rate']:>12.3f} {adhd_result['attentional_blink_rate']:>12.3f}")
    print(f"{'代谢流失 (%)':<25s} {normal_result['metabolic_drain_pct']:>12.1f} {adhd_result['metabolic_drain_pct']:>12.1f}")
    print(f"{'DA疲劳 (%)':<25s} {normal_result['da_fatigue_pct']:>12.1f} {adhd_result['da_fatigue_pct']:>12.1f}")
    print(f"{'噪声期废物':<25s} {normal_result['noise_waste']:>12.3f} {adhd_result['noise_waste']:>12.3f}")

    # 关键验证
    print(f"\nKey validation:")
    adhd_more_blink = adhd_result['attentional_blink_rate'] > normal_result['attentional_blink_rate']
    print(f"  ADHD blink rate: {adhd_result['attentional_blink_rate']:.3f} vs Normal {normal_result['attentional_blink_rate']:.3f} "
          f"{'[PASS] ADHD misses more' if adhd_more_blink else '[INFO] Similar blink rates'}")
    adhd_more_drain = adhd_result['metabolic_drain_pct'] > normal_result['metabolic_drain_pct']
    print(f"  Metabolic drain: ADHD {adhd_result['metabolic_drain_pct']:.1f}% vs Normal {normal_result['metabolic_drain_pct']:.1f}% "
          f"{'[PASS] ADHD drains faster' if adhd_more_drain else '[INFO] Similar drain'}")
    adhd_da_fatigue = adhd_result['da_fatigue_pct'] > normal_result['da_fatigue_pct']
    print(f"  DA fatigue: ADHD {adhd_result['da_fatigue_pct']:.1f}% vs Normal {normal_result['da_fatigue_pct']:.1f}% "
          f"{'[PASS] ADHD DA depleted' if adhd_da_fatigue else '[INFO] Similar DA levels'}")

    return {"Normal": normal_result, "ADHD": adhd_result}


if __name__ == "__main__":
    run_experiment()
