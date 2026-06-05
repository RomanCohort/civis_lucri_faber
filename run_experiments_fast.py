"""快速运行所有CLF实验，保存结果到JSON，用于生成图表和更新README。

优化策略：
- 减少步数（200步代替1000步）
- 禁用大量日志输出
- 直接输出结构化数据
"""

import sys
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

RESULTS_DIR = os.path.join(_project_root, 'experiment_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiment_thermo():
    """实验1: 热力学崩溃"""
    print("\n[1] Thermodynamic Collapse...")
    start = time.time()

    from experiment_thermodynamic_collapse import run_experiment
    results = run_experiment()

    data = {
        'name': 'thermodynamic_collapse',
        'elapsed': time.time() - start,
        'groups': {}
    }
    for g in ['Rich', 'Balanced', 'Poverty']:
        if g in results:
            data['groups'][g] = {
                'ttd': results[g]['ttd'],
                'compression_count': results[g]['compression_count'],
                'final_balance': results[g]['final_balance'],
                'exploration_entropy': results[g]['exploration_entropy'],
                'mean_social': results[g]['mean_social'],
            }

    return data


def run_experiment_metabolic():
    """实验2: 代谢稀疏"""
    print("\n[2] Metabolic Sparsity...")
    start = time.time()

    from experiment_metabolic_sparsity import run_experiment
    results = run_experiment()

    return {
        'name': 'metabolic_sparsity',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_hpa():
    """实验3: HPA认知刚性"""
    print("\n[3] HPA Cognitive Rigidity...")
    start = time.time()

    from experiment_hpa_cognitive_rigidity_v2 import run_experiment
    results = run_experiment()

    return {
        'name': 'hpa_cognitive_rigidity',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_epigenetic():
    """实验4: 表观遗传固化"""
    print("\n[4] Epigenetic Consolidation...")
    start = time.time()

    from experiment_epigenetic_consolidation import run_experiment
    results = run_experiment()

    return {
        'name': 'epigenetic_consolidation',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_stockholm():
    """实验5: 斯德哥尔摩压力"""
    print("\n[5] Stockholm Pressure...")
    start = time.time()

    from experiment_stockholm_pressure_v2 import run_experiment
    results = run_experiment()

    return {
        'name': 'stockholm_pressure',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_glymphatic():
    """实验6: 胶淋巴时机"""
    print("\n[6] Glymphatic Timing...")
    start = time.time()

    from experiment_6_glymphatic_timing import run_experiment
    results = run_experiment()

    return {
        'name': 'glymphatic_timing',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_adhd():
    """实验7: ADHD闪烁"""
    print("\n[7] ADHD Flicker...")
    start = time.time()

    from experiment_7_adhd_flicker import run_experiment
    results = run_experiment()

    return {
        'name': 'adhd_flicker',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_dream():
    """实验8: 数字梦境"""
    print("\n[8] Digital Dreaming...")
    start = time.time()

    from experiment_8_digital_dreaming_v2 import run_experiment
    results = run_experiment()

    return {
        'name': 'digital_dreaming',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_autism():
    """实验9: 自闭谱系"""
    print("\n[9] Autism Spectrum...")
    start = time.time()

    from experiment_9_autism_spectrum_v2 import run_experiment
    results = run_experiment()

    return {
        'name': 'autism_spectrum',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_d2():
    """实验10: D2受体占据"""
    print("\n[10] D2 Occupancy...")
    start = time.time()

    from experiment_10_d2_occupancy_v2 import run_experiment
    results = run_experiment()

    return {
        'name': 'd2_occupancy',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_stress():
    """实验11: 压力快感缺失"""
    print("\n[11] Stress Anhedonia...")
    start = time.time()

    from experiment_stress_anhedonia_v2 import run_experiment
    results = run_experiment()

    return {
        'name': 'stress_anhedonia',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_drug():
    """实验12: 药物决策"""
    print("\n[12] Drug Decision...")
    start = time.time()

    from experiment_drug_decision import run_experiment
    results = run_experiment()

    return {
        'name': 'drug_decision',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_social():
    """实验13: 社交衰减"""
    print("\n[13] Social Decay...")
    start = time.time()

    from experiment_social_decay import run_experiment
    results = run_experiment()

    return {
        'name': 'social_decay',
        'elapsed': time.time() - start,
        'results': results
    }


def run_experiment_therapeutic():
    """实验14: 治疗实验"""
    print("\n[14] Therapeutic Experiment...")
    start = time.time()

    from demo_therapeutic_experiment import run_experiment
    results = run_experiment()

    return {
        'name': 'therapeutic_experiment',
        'elapsed': time.time() - start,
        'results': results
    }


def main():
    print("=" * 70)
    print("CLF 计算精神病学实验套件 (优化版)")
    print("=" * 70)

    all_results = {}
    total_start = time.time()

    experiments = [
        run_experiment_thermo,
        run_experiment_metabolic,
        run_experiment_hpa,
        run_experiment_epigenetic,
        run_experiment_stockholm,
        run_experiment_glymphatic,
        run_experiment_adhd,
        run_experiment_dream,
        run_experiment_autism,
        run_experiment_d2,
        run_experiment_stress,
        run_experiment_drug,
        run_experiment_social,
        run_experiment_therapeutic,
    ]

    for exp_func in experiments:
        try:
            result = exp_func()
            all_results[result['name']] = result
            print(f"  [OK] {result['name']} ({result['elapsed']:.1f}s)")
        except Exception as e:
            print(f"  [ERROR] {exp_func.__name__}: {e}")

    # 保存汇总结果
    summary_path = os.path.join(RESULTS_DIR, 'all_results.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - total_start
    print(f"\n总耗时: {total_elapsed:.1f}s")
    print(f"结果保存至: {summary_path}")

    return all_results


if __name__ == "__main__":
    main()