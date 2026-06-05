"""并行运行所有计算精神病学实验，结果保存到 docs/experiment_results/"""
import subprocess, sys, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

EXPERIMENTS = [
    ('01_thermo',            'experiment_thermodynamic_collapse.py'),
    ('02_metabolic',         'experiment_metabolic_sparsity.py'),
    ('03_hpa',               'experiment_hpa_cognitive_rigidity_v2.py'),
    ('04_epigenetic',        'experiment_epigenetic_consolidation.py'),
    ('05_stockholm',         'experiment_stockholm_pressure_v2.py'),
    ('06_glymphatic',        'experiment_6_glymphatic_timing.py'),
    ('07_adhd',              'experiment_7_adhd_flicker.py'),
    ('08_dream',             'experiment_8_digital_dreaming_v2.py'),
    ('09_autism',            'experiment_9_autism_spectrum_v2.py'),
    ('10_d2',                'experiment_10_d2_occupancy_v2.py'),
    ('11_stress',            'experiment_stress_anhedonia_v2.py'),
    ('12_drug',              'experiment_drug_decision.py'),
    ('13_social',            'experiment_social_decay.py'),
    ('14_therapeutic',       'demo_therapeutic_experiment.py'),
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'experiment_results')


def run_one(name, script):
    """运行单个实验，保存结果到文件"""
    t0 = time.time()
    outfile = os.path.join(RESULTS_DIR, f'{name}.txt')
    base = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, script]

    try:
        with open(outfile, 'w', encoding='utf-8') as f:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=base,
                env={**os.environ, 'PYTHONWARNINGS': 'ignore',
                     'PYTHONPATH': base}
            )
            for line in proc.stdout:
                # 过滤大量重复WARNING
                if 'WARNING' in line and any(k in line for k in ['dimension', 'size', 'mismatch', 'torch']):
                    continue
                f.write(line)
            proc.wait()
            elapsed = time.time() - t0
            f.write(f"\n\n=== META ===\nName: {name}\nReturnCode: {proc.returncode}\nElapsed: {elapsed:.1f}s\n")
        return (name, proc.returncode, elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(f"ERROR: {e}\n")
        return (name, -1, elapsed)


def run_overdose():
    """运行overdose实验（特殊调用方式）"""
    t0 = time.time()
    name = '15_overdose'
    outfile = os.path.join(RESULTS_DIR, f'{name}.txt')
    base = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, '-c',
           'import warnings; warnings.filterwarnings("ignore"); '
           'from core.sertraline_overdose_experiment import run_experiment; run_experiment()']
    try:
        with open(outfile, 'w', encoding='utf-8') as f:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=base,
                env={**os.environ, 'PYTHONWARNINGS': 'ignore',
                     'PYTHONPATH': base}
            )
            for line in proc.stdout:
                if 'WARNING' in line and any(k in line for k in ['dimension', 'size', 'mismatch', 'torch']):
                    continue
                f.write(line)
            proc.wait()
            elapsed = time.time() - t0
            f.write(f"\n\n=== META ===\nName: {name}\nReturnCode: {proc.returncode}\nElapsed: {elapsed:.1f}s\n")
        return (name, proc.returncode, elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(f"ERROR: {e}\n")
        return (name, -1, elapsed)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    print(f"=== 并行运行 {len(EXPERIMENTS)+1} 个实验 ===")
    print(f"结果目录: {RESULTS_DIR}\n")

    t_start = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {}
        for name, script in EXPERIMENTS:
            f = pool.submit(run_one, name, script)
            futures[f] = name
        # overdose单独提交
        f = pool.submit(run_overdose)
        futures[f] = '15_overdose'

        for f in as_completed(futures):
            name = futures[f]
            try:
                rname, rc, elapsed = f.result()
                results[rname] = (rc, elapsed)
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"  [{status}] {rname} ({elapsed:.1f}s)")
            except Exception as e:
                results[name] = (-1, 0)
                print(f"  [ERROR] {name}: {e}")

    total = time.time() - t_start

    # 写汇总
    summary = os.path.join(RESULTS_DIR, 'SUMMARY.txt')
    with open(summary, 'w', encoding='utf-8') as f:
        f.write(f"CLF 计算精神病学实验汇总\n")
        f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total:.1f}s\n")
        f.write("=" * 70 + "\n\n")
        for name in sorted(results):
            rc, elapsed = results[name]
            status = "SUCCESS" if rc == 0 else f"FAILED(rc={rc})"
            f.write(f"## {name} [{status}] ({elapsed:.1f}s)\n\n")
            rfile = os.path.join(RESULTS_DIR, f'{name}.txt')
            if os.path.exists(rfile):
                with open(rfile, 'r', encoding='utf-8') as rf:
                    lines = rf.readlines()
                    tail = lines[-50:] if len(lines) > 50 else lines
                    f.writelines(tail)
            f.write('\n' + '-' * 70 + '\n\n')

    print(f"\n汇总: {summary}")
    print(f"总耗时: {total:.1f}s")


if __name__ == '__main__':
    main()
