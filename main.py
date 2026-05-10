"""Civis Lucri-Faber 测试入口

运行示例:
    python main.py
"""
import sys
import os

# 获取项目根目录并添加到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import random

from civis_lucri_faber.utils.config import Config, load_config
from civis_lucri_faber.core.agent import CivisLucriFaber


def main():
    """主函数"""
    print("=" * 60)
    print("Civis Lucri-Faber - Bio-Inspired AI Agent System")
    print("=" * 60)

    # 加载配置 (或使用默认值)
    config = load_config(
        # 好奇心参数
        curiosity_alpha=0.4,
        curiosity_beta=0.3,
        curiosity_gamma=0.3,
        exploration_rate=0.2,

        # 信息增益参数
        intrinsic_motivation_lambda=0.5,
        entropy_coef=0.1,

        # 元学习参数
        meta_lr=0.01,
        inner_steps=5,
        uncertainty_threshold=0.5,

        # 自对齐参数
        alignment_check_interval=5,
        use_anthropic=False,  # 使用 OpenAI

        # 经济学参数
        initial_balance=30.0,
        compute_cost_per_sec=0.02,
        storage_cost_per_sec=0.001,
        task_reward_min=0.1,
        task_reward_max=1.0,
        compress_threshold=8.0,

        # API (使用环境变量或留空)
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model_name="gpt-4",

        # 系统参数
        max_history_size=500,
        device="cpu",
        seed=42
    )

    print("\n[CONFIG] Configuration:")
    print(f"  - Initial Balance: {config.initial_balance}")
    print(f"  - Compute Cost/sec: {config.compute_cost_per_sec}")
    print(f"  - Intrinsic Motivation Lambda: {config.intrinsic_motivation_lambda}")
    print(f"  - Alignment Check Interval: {config.alignment_check_interval}")
    print(f"  - API Client: {'Configured' if config.openai_api_key or config.anthropic_api_key else 'Mock Mode'}")

    # 实例化智能体
    print("\n[INIT] Initializing agent...")
    agent = CivisLucriFaber(
        config=config,
        memory_path="civis_memory.json",
        alignment_log_path="civis_alignment.json",
        thermo_log_path="civis_thermo.json"
    )

    # 设置随机种子
    np.random.seed(config.seed)
    random.seed(config.seed)

    print("[OK] Agent initialized")

    # 运行探索
    print("\n" + "=" * 60)
    print("Starting Exploration")
    print("=" * 60)

    n_episodes = 15
    states = agent.run_episodes(n_episodes=n_episodes, verbose=True)

    # 输出统计
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)

    stats = agent.get_full_statistics()

    print("\n[Dimension 1 - Curiosity Exploration]:")
    curiosity_stats = stats.get("curiosity", {})
    print(f"  - Total Goals: {curiosity_stats.get('total_goals', 0)}")
    print(f"  - Completed: {curiosity_stats.get('completed', 0)}")
    print(f"  - Avg Novelty: {curiosity_stats.get('novelty_avg', 0):.4f}")
    print(f"  - Avg Value: {curiosity_stats.get('value_avg', 0):.4f}")

    print("\n[Dimension 2 - Information Gain]:")
    ig_stats = stats.get("info_gain", {})
    print(f"  - Buffer Size: {ig_stats.get('buffer_size', 0)}")
    print(f"  - Avg Info Gain: {ig_stats.get('info_gain_avg', 0):.4f}")

    print("\n[Dimension 3 - Active Learning]:")
    al_stats = stats.get("active_learning", {})
    print(f"  - Total Queries: {al_stats.get('total_queries', 0)}")
    print(f"  - Answered: {al_stats.get('answered', 0)}")
    print(f"  - Avg Uncertainty: {al_stats.get('avg_uncertainty', 0):.4f}")

    print("\n[Dimension 4 - Self Alignment]:")
    align_stats = stats.get("self_alignment", {})
    print(f"  - Reflections: {align_stats.get('total_reflections', 0)}")
    print(f"  - Avg Alignment Score: {align_stats.get('avg_alignment', 0):.4f}")
    print(f"  - Issues Found: {align_stats.get('recent_issues', [])}")

    print("\n[Dimension 5 - Digital Thermodynamics]:")
    thermo_stats = stats.get("thermodynamics", {})
    print(f"  - Final Balance: {thermo_stats.get('balance', 0):.2f}")
    print(f"  - Status: {thermo_stats.get('status', 'UNKNOWN')}")
    print(f"  - Total Compute Used: {thermo_stats.get('total_compute', 0):.4f}")
    print(f"  - Total Earnings: {thermo_stats.get('total_earnings', 0):.4f}")
    print(f"  - Tasks Completed: {thermo_stats.get('task_completed', 0)}")
    print(f"  - Deaths: {thermo_stats.get('deaths', 0)}")

    print("\n[Knowledge Memory]:")
    mem_stats = stats.get("memory", {})
    print(f"  - Memory Items: {mem_stats.get('total_memories', 0)}")
    print(f"  - Experiences: {mem_stats.get('total_experiences', 0)}")

    # 最终状态
    final_state = states[-1]
    print("\n" + "=" * 60)
    print("Final State")
    print("=" * 60)
    print(f"  - Episodes: {final_state.step}")
    print(f"  - System Status: {final_state.status}")
    print(f"  - Balance: {final_state.balance:.2f}")
    print(f"  - Info Gain: {final_state.info_gain:.4f}")
    print(f"  - Alignment Score: {final_state.alignment_score:.4f}")

    # 保存模型
    print("\n[SAVE] Saving model...")
    agent.save("civis_model.pt")
    print("[OK] Saved")

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()