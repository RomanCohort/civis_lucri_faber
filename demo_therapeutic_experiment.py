"""治疗实验演示 — MDD + Sertraline + CBT 联合治疗 + DDI多药交互。

对比7个实验臂:
  1. Sertraline 100mg 单用
  2. CBT weekly 单用
  3. Sertraline 100mg + CBT weekly (联合)
  4. 安慰剂 (无治疗)
  5. Sertraline + Fluoxetine (双SSRI: CYP2D6/2C19抑制 + Bliss独立)
  6. Haloperidol + Amphetamine (DA拮抗 + CYP2D6抑制)
  7. Diazepam + Morphine (禁忌: CNS呼吸抑制协同)

运行8周治疗 + 2周随访，生成报告+8张图。
"""

import sys
import os

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.therapeutic_experiment import (
    DrugConfig,
    TherapyConfig,
    ExperimentConfig,
    TherapeuticExperiment,
)
from core.therapeutic_report import generate_markdown_report


def run_demo():
    """运行4臂治疗实验演示。"""

    # ── 通用配置 ──
    base_config = dict(
        condition="MDD",
        severity="moderate",
        duration_steps=13440,     # 8周 × 7天 × 24h × 10步/h
        follow_up_steps=3360,    # 2周随访
        observation_interval=10,
        steps_per_hour=10.0,
    )

    # ── 臂1: Sertraline 单用 ──
    arm1_config = ExperimentConfig(
        **base_config,
        drugs=[DrugConfig(
            name_or_smiles="sertraline",
            dose_mg=100.0,
            freq_per_day=1.0,
        )],
        therapies=[],
    )

    # ── 臂2: CBT 单用 ──
    arm2_config = ExperimentConfig(
        **base_config,
        drugs=[],
        therapies=[TherapyConfig(
            modality="CBT",
            frequency="weekly",
            intensity=0.7,
        )],
    )

    # ── 臂3: Sertraline + CBT 联合 ──
    arm3_config = ExperimentConfig(
        **base_config,
        drugs=[DrugConfig(
            name_or_smiles="sertraline",
            dose_mg=100.0,
            freq_per_day=1.0,
        )],
        therapies=[TherapyConfig(
            modality="CBT",
            frequency="weekly",
            intensity=0.7,
        )],
    )

    # ── 臂4: 安慰剂 ──
    arm4_config = ExperimentConfig(
        **base_config,
        drugs=[],
        therapies=[],
    )

    # ── 臂5: Sertraline + Fluoxetine (双SSRI: DDI) ──
    arm5_config = ExperimentConfig(
        **base_config,
        drugs=[
            DrugConfig(name_or_smiles="sertraline", dose_mg=100.0, freq_per_day=1.0),
            DrugConfig(name_or_smiles="fluoxetine", dose_mg=20.0, freq_per_day=1.0),
        ],
        therapies=[],
    )

    # ── 臂6: Haloperidol + Amphetamine (DA拮抗: DDI) ──
    arm6_config = ExperimentConfig(
        **base_config,
        drugs=[
            DrugConfig(name_or_smiles="haloperidol", dose_mg=5.0, freq_per_day=2.0),
            DrugConfig(name_or_smiles="amphetamine", dose_mg=10.0, freq_per_day=1.0),
        ],
        therapies=[],
    )

    # ── 臂7: Diazepam + Morphine (禁忌: BZD+阿片) ──
    arm7_config = ExperimentConfig(
        **base_config,
        drugs=[
            DrugConfig(name_or_smiles="diazepam", dose_mg=5.0, freq_per_day=2.0),
            DrugConfig(name_or_smiles="morphine", dose_mg=10.0, freq_per_day=4.0),
        ],
        therapies=[],
    )

    # ── 运行各臂 ──
    arm_results = {}
    arms = [
        ("Sertraline", arm1_config),
        ("CBT", arm2_config),
        ("Sertraline+CBT", arm3_config),
        ("Placebo", arm4_config),
        ("Sertraline+Fluoxetine", arm5_config),
        ("Haloperidol+Amphetamine", arm6_config),
        ("Diazepam+Morphine", arm7_config),
    ]

    for label, config in arms:
        print(f"\n{'='*60}")
        print(f"  Running arm: {label}")
        print(f"{'='*60}")

        try:
            exp = TherapeuticExperiment(config=config)
            exp.prepare()
            result = exp.run()
            arm_results[label] = result

            print(f"  Remission rate: {result.remission_rate:.1%}")
            print(f"  Relapse rate:   {result.relapse_rate:.1%}")
            print(f"  Peak drug effect: {result.peak_drug_effect:.3f}")
            print(f"  Therapy sessions: {result.total_therapy_sessions}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── 生成报告 ──
    if arm_results:
        # 用联合臂作为主报告
        main_result = arm_results.get("Sertraline+CBT",
                       list(arm_results.values())[0])

        report_path = generate_markdown_report(
            result=main_result,
            arm_results=arm_results,
            figure_dir="docs/figures/therapeutic",
            report_path="docs/therapeutic_experiment_report.md",
        )
        print(f"\nReport generated: {report_path}")
    else:
        print("\nNo results to report.")


if __name__ == "__main__":
    run_demo()
