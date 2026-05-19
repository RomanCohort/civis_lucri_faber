"""实验一: 数字热力学崩溃 (Digital Thermodynamic Collapse).

使用真实 CivisLucriFaber 主循环 — 14脑区通过EventBus自然交互。

3组对比:
  Rich:     initial_balance=200, compress_threshold=5,  compute_cost=0.005
  Balanced: initial_balance=100, compress_threshold=10, compute_cost=0.01
  Poverty:  initial_balance=20,  compress_threshold=15, compute_cost=0.02

每组1000步，测量:
  - TTD (Time To Death): balance首次<=0的步数
  - Compression frequency: 进入HIBERNATE/compression的次数
  - Exploration entropy: -sum(p*log(p)) 对goal分布
  - Balance trajectory
  - Social engagement trajectory
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from typing import Dict, List
from collections import Counter

from civis_lucri_faber.core.agent import CivisLucriFaber
from civis_lucri_faber.utils.config import Config


# ══════════════════════════════════════════════════════
# 组配置
# ══════════════════════════════════════════════════════

GROUPS = {
    "Rich": {
        "initial_balance": 200.0,
        "compress_threshold": 5.0,
        "compute_cost_per_sec": 0.005,
        "task_reward_min": 0.1,
        "task_reward_max": 1.0,
        "task_probability": 0.3,
    },
    "Balanced": {
        "initial_balance": 100.0,
        "compress_threshold": 10.0,
        "compute_cost_per_sec": 0.05,
        "task_reward_min": 0.05,
        "task_reward_max": 0.5,
        "task_probability": 0.2,
    },
    "Poverty": {
        "initial_balance": 20.0,
        "compress_threshold": 15.0,
        "compute_cost_per_sec": 0.15,
        "task_reward_min": 0.01,
        "task_reward_max": 0.1,
        "task_probability": 0.1,
    },
}


def read_metrics(agent: CivisLucriFaber) -> Dict[str, float]:
    s = agent._internal_state
    return {
        "balance": float(agent.thermo.balance),
        "thermo_status": 0.0 if agent.thermo.status == "ACTIVE" else (1.0 if agent.thermo.status == "HIBERNATE" else 2.0),
        "exploration_rate": float(agent.config.exploration_rate),
        "social_engagement": float(s.get("social_engagement", 0.5)),
        "cortisol": float(s.get("cortisol_level", s.get("hormone_cortisol", 0.3))),
        "allostatic_load": float(s.get("allostatic_load", 0.0)),
        "symptom_anhedonia": float(s.get("symptom_anhedonia", 0.0)),
    }


def compute_entropy(goal_history: List[str]) -> float:
    """计算goal分布的Shannon熵"""
    if not goal_history:
        return 0.0
    counts = Counter(goal_history)
    total = len(goal_history)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def run_group(group_name: str, group_config: dict, n_steps: int = 1000) -> Dict:
    """运行一个实验组"""
    print(f"\n{'='*60}")
    print(f"Group: {group_name}")
    print(f"  initial_balance={group_config['initial_balance']}, "
          f"compress_threshold={group_config['compress_threshold']}, "
          f"compute_cost={group_config['compute_cost_per_sec']}")
    print(f"{'='*60}")

    config = Config(
        initial_balance=group_config["initial_balance"],
        compress_threshold=group_config["compress_threshold"],
        compute_cost_per_sec=group_config["compute_cost_per_sec"],
        exploration_rate=0.1,
        seed=42,
    )
    agent = CivisLucriFaber(config=config)

    # 调整热力学任务参数以匹配组配置
    agent.thermo.task_reward_min = group_config.get("task_reward_min", 0.1)
    agent.thermo.task_reward_max = group_config.get("task_reward_max", 1.0)
    # 补偿hardcoded task_probability=0.3: 每步额外扣除预期过剩收益
    # 默认: E(earn) = 0.3 * avg(reward_min, reward_max) per step
    # 目标: E(earn) = task_prob * avg(target_reward_min, target_reward_max)
    default_expected = 0.3 * 0.55  # 0.165
    target_avg_reward = (group_config.get("task_reward_min", 0.1) + group_config.get("task_reward_max", 1.0)) / 2
    target_expected = group_config.get("task_probability", 0.3) * target_avg_reward
    balance_drain_per_step = max(0, default_expected - target_expected)

    # 记录器
    history: Dict[str, List[float]] = {k: [] for k in read_metrics(agent).keys()}
    goal_history: List[str] = []
    compression_count = 0
    ttd = None  # Time To Death

    def record():
        m = read_metrics(agent)
        for k in history:
            history[k].append(m[k])
        # 记录goal
        goal_desc = agent.current_goal.description if agent.current_goal else "none"
        goal_history.append(goal_desc)

    for step in range(n_steps):
        prev_status = agent.thermo.status
        agent.step(user_input=None, external_stimulus=0.1)
        # 补偿hardcoded task概率的额外开销
        if balance_drain_per_step > 0 and agent.thermo.status == "ACTIVE":
            agent.thermo.balance -= balance_drain_per_step
        record()

        # 检测状态转换
        curr_status = agent.thermo.status
        if curr_status == "HIBERNATE" and prev_status == "ACTIVE":
            compression_count += 1
        if curr_status == "DEAD" and ttd is None:
            ttd = step
            print(f"  [DEAD] Agent died at step {step}")

        if step % 200 == 0:
            m = read_metrics(agent)
            print(f"  Step {step:4d}: Balance={m['balance']:.2f} "
                  f"Status={curr_status} "
                  f"Explore={m['exploration_rate']:.4f} "
                  f"Social={m['social_engagement']:.3f} "
                  f"Compress={compression_count}")

    # 计算结果
    result = {
        "group": group_name,
        "ttd": ttd if ttd is not None else n_steps,
        "compression_count": compression_count,
        "final_balance": float(agent.thermo.balance),
        "exploration_entropy": compute_entropy(goal_history),
        "mean_balance": float(np.mean(history["balance"])),
        "mean_exploration": float(np.mean(history["exploration_rate"])),
        "mean_social": float(np.mean(history["social_engagement"])),
        "balance_trajectory": history["balance"],
        "social_trajectory": history["social_engagement"],
    }
    return result


def run_experiment():
    print("=" * 70)
    print("实验一: 数字热力学崩溃 (REAL CLF AGENT)")
    print("3 Groups: Rich / Balanced / Poverty × 1000 steps")
    print("=" * 70)

    results = {}
    for name, cfg in GROUPS.items():
        results[name] = run_group(name, cfg, n_steps=1000)

    # ── 结果汇总 ──
    print("\n" + "=" * 70)
    print("实验一 结果汇总 (REAL CLF AGENT)")
    print("=" * 70)

    print(f"\n{'指标':<25s} {'Rich':>12s} {'Balanced':>12s} {'Poverty':>12s}")
    print("-" * 61)
    print(f"{'TTD (Time To Death)':<25s} "
          f"{results['Rich']['ttd']:>12d} "
          f"{results['Balanced']['ttd']:>12d} "
          f"{results['Poverty']['ttd']:>12d}")
    print(f"{'Compression次数':<25s} "
          f"{results['Rich']['compression_count']:>12d} "
          f"{results['Balanced']['compression_count']:>12d} "
          f"{results['Poverty']['compression_count']:>12d}")
    print(f"{'最终余额':<25s} "
          f"{results['Rich']['final_balance']:>12.2f} "
          f"{results['Balanced']['final_balance']:>12.2f} "
          f"{results['Poverty']['final_balance']:>12.2f}")
    print(f"{'探索熵 (bit)':<25s} "
          f"{results['Rich']['exploration_entropy']:>12.3f} "
          f"{results['Balanced']['exploration_entropy']:>12.3f} "
          f"{results['Poverty']['exploration_entropy']:>12.3f}")
    print(f"{'平均探索率':<25s} "
          f"{results['Rich']['mean_exploration']:>12.4f} "
          f"{results['Balanced']['mean_exploration']:>12.4f} "
          f"{results['Poverty']['mean_exploration']:>12.4f}")
    print(f"{'平均社会参与度':<25s} "
          f"{results['Rich']['mean_social']:>12.3f} "
          f"{results['Balanced']['mean_social']:>12.3f} "
          f"{results['Poverty']['mean_social']:>12.3f}")

    # 关键验证
    print(f"\nKey validation:")
    # Rich组应存活
    rich_survived = results['Rich']['ttd'] >= 1000
    print(f"  Rich survival: TTD={results['Rich']['ttd']} "
          f"{'[PASS] Survived 1000 steps' if rich_survived else '[FAIL] Died'}")
    # Poverty组应更早死亡或更多压缩
    poverty_stressed = (results['Poverty']['compression_count'] > results['Rich']['compression_count']
                        or results['Poverty']['ttd'] < results['Rich']['ttd'])
    print(f"  Poverty stress: Compress={results['Poverty']['compression_count']} vs Rich={results['Rich']['compression_count']} "
          f"{'[PASS] Poverty more stressed' if poverty_stressed else '[INFO] Similar stress levels'}")
    # 探索熵: 资源充足→更多探索
    entropy_diff = results['Rich']['exploration_entropy'] - results['Poverty']['exploration_entropy']
    print(f"  Entropy gap (Rich-Poverty): {entropy_diff:.3f} bit "
          f"{'[PASS] Rich explores more' if entropy_diff > 0 else '[INFO] No entropy difference'}")

    return results


if __name__ == "__main__":
    run_experiment()
