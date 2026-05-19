"""治疗实验报告生成器 — 8张图 + Markdown报告。

从TherapeuticResult生成完整的可视化报告:
  1. 症状严重度时程 (PHQ-9/GAD-7轨迹)
  2. 神经递质动态 (DA, 5-HT, NE, GABA, Cortisol)
  3. 药物浓度 & PD效应曲线
  4. 脑区变化 (PFC, 边缘系统, HPA, ANS)
  5. 心理测量仪表盘 (5量表 + CGI)
  6. 协同因子轨迹
  7. 治疗结果对比 (多臂实验)
  8. LLM评估时间线
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyBboxPatch

from simulacrum.core.therapeutic_experiment import TherapeuticResult, TreatmentTimepoint
from simulacrum.core.psychometric_indicators import PsychometricSnapshot

# ── 配色方案 ──
COLORS = {
    "phq9": "#f38ba8",       # 粉红 — 抑郁
    "gad7": "#fab387",       # 橙 — 焦虑
    "cognitive": "#89b4fa",  # 蓝 — 认知
    "emotion_reg": "#cba6f7", # 紫 — 情绪调节
    "social": "#a6e3a1",     # 绿 — 社会功能
    "global": "#f5c2e7",     # 浅粉 — 综合
    "da": "#f38ba8",         # 多巴胺
    "sht": "#89b4fa",        # 5-HT
    "ne": "#fab387",         # NE
    "gaba": "#a6e3a1",       # GABA
    "cort": "#f9e2af",       # 皮质醇
    "pfc": "#89b4fa",        # PFC
    "valence": "#cba6f7",    # 效价
    "arousal": "#fab387",    # 唤醒
    "hpa": "#f38ba8",        # HPA
    "drug": "#74c7ec",       # 药物浓度
    "pd": "#94e2d5",         # PD效应
    "synergy": "#f5c2e7",    # 协同
    "therapy": "#a6e3a1",    # 疗法
    "treatment": "#89b4fa",  # 治疗期
    "followup": "#f9e2af",   # 随访期
}

# 中文支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _extract_timeseries(
    trajectory: List[TreatmentTimepoint],
) -> Dict[str, np.ndarray]:
    """从轨迹提取所有时序数据为numpy数组。"""
    n = len(trajectory)
    if n == 0:
        return {}

    data: Dict[str, np.ndarray] = {}
    data["time_h"] = np.array([t.time_h for t in trajectory])
    data["step"] = np.array([t.step for t in trajectory])
    data["phase"] = [t.phase for t in trajectory]

    # 药物浓度和PD效应
    all_drug_ids = set()
    for t in trajectory:
        all_drug_ids.update(t.drug_concentrations.keys())
    for did in all_drug_ids:
        data[f"conc_{did}"] = np.array([
            t.drug_concentrations.get(did, 0.0) for t in trajectory
        ])
        data[f"pd_{did}"] = np.array([
            t.drug_pd_effects.get(did, 0.0) for t in trajectory
        ])

    # 神经递质
    nt_keys = ["nt_dopamine", "nt_serotonin", "nt_norepinephrine",
               "nt_gaba", "cortisol_level"]
    for key in nt_keys:
        data[key] = np.array([
            t.neurotransmitters.get(key, 0.5) for t in trajectory
        ])

    # 脑区
    br_keys = ["prefrontal_maturity", "limbic_valence",
               "limbic_arousal", "hpa_axis_stress_reactivity_mult"]
    for key in br_keys:
        data[key] = np.array([
            t.brain_regions.get(key, 0.5) for t in trajectory
        ])

    # 疗法
    data["therapy_skill"] = np.array([t.therapy_skill for t in trajectory])
    data["synergy_factor"] = np.array([t.synergy_factor for t in trajectory])
    data["therapy_sessions"] = np.array([t.therapy_sessions_total for t in trajectory])

    # 心理测量
    pm_keys = ["depression_severity", "anxiety_level", "cognitive_function",
               "emotional_regulation", "social_functioning", "global_symptom_severity"]
    for key in pm_keys:
        vals = []
        for t in trajectory:
            if t.psychometrics and key in t.psychometrics:
                vals.append(t.psychometrics[key])
            else:
                vals.append(float("nan"))
        data[key] = np.array(vals)

    return data


def _add_phase_shading(ax, data: Dict[str, np.ndarray], alpha: float = 0.08):
    """添加治疗期/随访期背景色。"""
    time_h = data["time_h"]
    phases = data["phase"]

    # 找到阶段切换点
    treatment_end = None
    for i, ph in enumerate(phases):
        if ph == "follow_up" and treatment_end is None:
            treatment_end = time_h[i]
            break

    if treatment_end is not None:
        ax.axvspan(0, treatment_end, alpha=alpha, color=COLORS["treatment"],
                   label="Treatment")
        ax.axvspan(treatment_end, time_h[-1], alpha=alpha, color=COLORS["followup"],
                   label="Follow-up")


# ══════════════════════════════════════════════════════════════
# 8张图
# ══════════════════════════════════════════════════════════════

def fig1_symptom_severity(
    data: Dict[str, np.ndarray],
    save_path: str,
    title_suffix: str = "",
):
    """图1: 症状严重度时程 (PHQ-9/GAD-7轨迹)。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    t = data["time_h"]

    ax.plot(t, data["depression_severity"], color=COLORS["phq9"],
            linewidth=2, label="PHQ-9 (Depression)")
    ax.plot(t, data["anxiety_level"], color=COLORS["gad7"],
            linewidth=2, label="GAD-7 (Anxiety)")
    ax.plot(t, data["global_symptom_severity"] * 10, color=COLORS["global"],
            linewidth=1.5, linestyle="--", label="Global Severity (x10)")

    # 临床阈值线
    ax.axhline(y=5, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(t[0] + 1, 5.3, "Remission threshold", fontsize=8, color="gray")
    ax.axhline(y=10, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(t[0] + 1, 10.3, "Moderate threshold", fontsize=8, color="gray")

    _add_phase_shading(ax, data)
    ax.set_xlabel("Time (h)", fontsize=12)
    ax.set_ylabel("Score (0-10)", fontsize=12)
    ax.set_title(f"Symptom Severity Over Time{title_suffix}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig2_neurotransmitter_dynamics(
    data: Dict[str, np.ndarray],
    save_path: str,
    title_suffix: str = "",
):
    """图2: 神经递质动态。"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    t = data["time_h"]

    nt_plots = [
        ("nt_dopamine", "Dopamine (DA)", COLORS["da"]),
        ("nt_serotonin", "Serotonin (5-HT)", COLORS["sht"]),
        ("nt_norepinephrine", "Norepinephrine (NE)", COLORS["ne"]),
        ("nt_gaba", "GABA", COLORS["gaba"]),
        ("cortisol_level", "Cortisol", COLORS["cort"]),
    ]

    for idx, (key, label, color) in enumerate(nt_plots):
        ax = axes[idx // 3][idx % 3]
        ax.plot(t, data[key], color=color, linewidth=2)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        _add_phase_shading(ax, data)
        if idx >= 3:
            ax.set_xlabel("Time (h)", fontsize=10)

    # 第6格: 综合雷达图 (最后一个时间点 vs 基线)
    ax = axes[1][2]
    ax.axis("off")
    if len(t) > 1:
        categories = ["DA", "5-HT", "NE", "GABA", "Cort"]
        baseline_vals = [data[k][0] for k in
                         ["nt_dopamine", "nt_serotonin", "nt_norepinephrine",
                          "nt_gaba", "cortisol_level"]]
        final_vals = [data[k][-1] for k in
                      ["nt_dopamine", "nt_serotonin", "nt_norepinephrine",
                       "nt_gaba", "cortisol_level"]]

        # 简易柱状对比
        x = np.arange(len(categories))
        width = 0.35
        ax_bar = fig.add_subplot(axes[1][2])
        ax_bar.bar(x - width / 2, baseline_vals, width, label="Baseline",
                   color="gray", alpha=0.5)
        ax_bar.bar(x + width / 2, final_vals, width, label="Final",
                   color="#89b4fa", alpha=0.8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(categories, fontsize=9)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.legend(fontsize=8)
        ax_bar.set_title("Baseline vs Final", fontsize=10, fontweight="bold")
        ax_bar.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Neurotransmitter Dynamics{title_suffix}", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig3_drug_concentration_pd(
    data: Dict[str, np.ndarray],
    save_path: str,
    title_suffix: str = "",
):
    """图3: 药物浓度 & PD效应曲线。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    t = data["time_h"]

    # 找所有药物ID
    drug_ids = sorted(set(
        k.replace("conc_", "", 1) for k in data if k.startswith("conc_")
    ))

    drug_colors = ["#74c7ec", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7"]

    for i, did in enumerate(drug_ids):
        c = drug_colors[i % len(drug_colors)]
        conc_key = f"conc_{did}"
        pd_key = f"pd_{did}"
        if conc_key in data:
            ax1.plot(t, data[conc_key], color=c, linewidth=2,
                     label=f"{did} Conc")
        if pd_key in data:
            ax2.plot(t, data[pd_key], color=c, linewidth=2,
                     label=f"{did} PD Effect")

    ax1.set_ylabel("Concentration (mg/L)", fontsize=11)
    ax1.set_title(f"Drug Concentration{title_suffix}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    _add_phase_shading(ax1, data)

    ax2.set_xlabel("Time (h)", fontsize=11)
    ax2.set_ylabel("PD Effect (0-Emax)", fontsize=11)
    ax2.set_title("Pharmacodynamic Effect", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    _add_phase_shading(ax2, data)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig4_brain_regions(
    data: Dict[str, np.ndarray],
    save_path: str,
    title_suffix: str = "",
):
    """图4: 脑区变化。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = data["time_h"]

    plots = [
        ("prefrontal_maturity", "PFC Maturity", COLORS["pfc"]),
        ("limbic_valence", "Limbic Valence", COLORS["valence"]),
        ("limbic_arousal", "Limbic Arousal", COLORS["arousal"]),
        ("hpa_axis_stress_reactivity_mult", "HPA Reactivity", COLORS["hpa"]),
    ]

    for idx, (key, label, color) in enumerate(plots):
        ax = axes[idx // 2][idx % 2]
        ax.plot(t, data[key], color=color, linewidth=2)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        _add_phase_shading(ax, data)
        if idx >= 2:
            ax.set_xlabel("Time (h)", fontsize=11)

    plt.suptitle(f"Brain Region Dynamics{title_suffix}", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig5_psychometric_dashboard(
    data: Dict[str, np.ndarray],
    save_path: str,
    title_suffix: str = "",
):
    """图5: 心理测量仪表盘 (5量表轨迹 + 综合严重度)。"""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    t = data["time_h"]

    # 5个量表轨迹
    scales = [
        ("depression_severity", "PHQ-9 (Depression)", COLORS["phq9"]),
        ("anxiety_level", "GAD-7 (Anxiety)", COLORS["gad7"]),
        ("cognitive_function", "MoCA (Cognition)", COLORS["cognitive"]),
        ("emotional_regulation", "DERS (Emotion Reg.)", COLORS["emotion_reg"]),
        ("social_functioning", "Social Functioning", COLORS["social"]),
    ]

    for i, (key, label, color) in enumerate(scales):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.plot(t, data[key], color=color, linewidth=2)
        ax.fill_between(t, 0, data[key], alpha=0.15, color=color)
        ax.set_ylim(0, 10.5)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        _add_phase_shading(ax, data)
        if i >= 3:
            ax.set_xlabel("Time (h)", fontsize=9)

    # 综合严重度
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, data["global_symptom_severity"], color=COLORS["global"],
            linewidth=2.5)
    ax.fill_between(t, 0, data["global_symptom_severity"],
                    alpha=0.2, color=COLORS["global"])
    ax.axhline(y=0.3, color="green", linestyle=":", alpha=0.5)
    ax.text(t[0] + 1, 0.32, "Remission", fontsize=8, color="green")
    ax.axhline(y=0.6, color="orange", linestyle=":", alpha=0.5)
    ax.text(t[0] + 1, 0.62, "Moderate", fontsize=8, color="orange")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (h)", fontsize=9)
    ax.set_ylabel("Severity", fontsize=9)
    ax.set_title("Global Symptom Severity", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    _add_phase_shading(ax, data)

    plt.suptitle(f"Psychometric Dashboard{title_suffix}", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig6_synergy_trajectory(
    data: Dict[str, np.ndarray],
    save_path: str,
    title_suffix: str = "",
):
    """图6: 协同因子 + 疗法技能轨迹。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    t = data["time_h"]

    ax1.plot(t, data["synergy_factor"], color=COLORS["synergy"],
             linewidth=2, label="Synergy Factor")
    ax1.fill_between(t, 0, data["synergy_factor"], alpha=0.15,
                     color=COLORS["synergy"])
    ax1.set_ylabel("Synergy Factor", fontsize=11)
    ax1.set_title(f"Drug-Therapy Synergy{title_suffix}", fontsize=12,
                  fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    _add_phase_shading(ax1, data)

    ax2.plot(t, data["therapy_skill"], color=COLORS["therapy"],
             linewidth=2, label="Therapy Skill")
    # 疗法会话标记
    sessions = data["therapy_sessions"]
    session_steps = []
    prev = 0
    for i, s in enumerate(sessions):
        if s > prev:
            session_steps.append(i)
            prev = s
    if session_steps:
        ax2.scatter(t[session_steps], data["therapy_skill"][session_steps],
                    color=COLORS["therapy"], s=30, zorder=5, alpha=0.7,
                    label="Session")

    ax2.set_xlabel("Time (h)", fontsize=11)
    ax2.set_ylabel("Therapy Skill", fontsize=11)
    ax2.set_title("Therapy Skill Accumulation", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    _add_phase_shading(ax2, data)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig7_arm_comparison(
    results: Dict[str, TherapeuticResult],
    save_path: str,
):
    """图7: 多臂实验对比 (不同治疗条件)。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    arm_colors = ["#f38ba8", "#89b4fa", "#a6e3a1", "#fab387", "#cba6f7",
                  "#74c7ec", "#f9e2af"]
    arm_labels = list(results.keys())

    # 提取所有臂的数据
    all_data = {}
    for label, result in results.items():
        all_data[label] = _extract_timeseries(result.trajectory)

    # 7a: PHQ-9对比
    ax = axes[0][0]
    for i, label in enumerate(arm_labels):
        d = all_data[label]
        if "depression_severity" in d:
            ax.plot(d["time_h"], d["depression_severity"],
                    color=arm_colors[i % len(arm_colors)],
                    linewidth=2, label=label)
    ax.axhline(y=5, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("PHQ-9 Score", fontsize=11)
    ax.set_title("Depression Severity Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10.5)

    # 7b: GAD-7对比
    ax = axes[0][1]
    for i, label in enumerate(arm_labels):
        d = all_data[label]
        if "anxiety_level" in d:
            ax.plot(d["time_h"], d["anxiety_level"],
                    color=arm_colors[i % len(arm_colors)],
                    linewidth=2, label=label)
    ax.axhline(y=5, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("GAD-7 Score", fontsize=11)
    ax.set_title("Anxiety Level Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10.5)

    # 7c: 综合严重度对比
    ax = axes[1][0]
    for i, label in enumerate(arm_labels):
        d = all_data[label]
        if "global_symptom_severity" in d:
            ax.plot(d["time_h"], d["global_symptom_severity"],
                    color=arm_colors[i % len(arm_colors)],
                    linewidth=2, label=label)
    ax.axhline(y=0.3, color="green", linestyle=":", alpha=0.5)
    ax.set_xlabel("Time (h)", fontsize=11)
    ax.set_ylabel("Global Severity", fontsize=11)
    ax.set_title("Global Symptom Severity Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 7d: 缓解率/复发率柱状图
    ax = axes[1][1]
    remission_rates = [results[l].remission_rate for l in arm_labels]
    relapse_rates = [results[l].relapse_rate for l in arm_labels]
    x = np.arange(len(arm_labels))
    width = 0.35
    ax.bar(x - width / 2, remission_rates, width, label="Remission Rate",
           color="#a6e3a1", alpha=0.8)
    ax.bar(x + width / 2, relapse_rates, width, label="Relapse Rate",
           color="#f38ba8", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(arm_labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Outcome Summary", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Multi-Arm Treatment Comparison", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig8_llm_evaluation_timeline(
    llm_evaluations: List[Dict],
    data: Dict[str, np.ndarray],
    save_path: str,
    title_suffix: str = "",
):
    """图8: LLM评估时间线。"""
    if not llm_evaluations:
        # 无LLM评估时生成空图
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No LLM evaluations available",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax.transAxes)
        ax.set_title("LLM Clinical Evaluation Timeline", fontsize=14,
                     fontweight="bold")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    t = data["time_h"]

    # 提取评估数据
    eval_times = []
    cgi_vals = []
    func_vals = []
    severity_map = {"minimal": 1, "mild": 2, "moderate": 3,
                    "moderately_severe": 4, "severe": 5}
    sev_vals = []
    progress_map = {"full_remission": 1, "significant": 2, "moderate": 3,
                    "minimal": 4, "no_change": 5}
    prog_vals = []

    for ev in llm_evaluations:
        step = ev.get("timepoint", ev.get("step", 0))
        # 找到对应时间
        if "time_h" in ev:
            eval_times.append(ev["time_h"])
        else:
            idx = np.searchsorted(data["step"], step)
            idx = min(idx, len(t) - 1)
            eval_times.append(t[idx])

        cgi_vals.append(float(ev.get("clinical_global_impression", 4)))
        func_vals.append(float(ev.get("functional_improvement", 0)))
        sev_str = str(ev.get("severity_assessment", "moderate")).lower()
        sev_vals.append(severity_map.get(sev_str, 3))
        prog_str = str(ev.get("treatment_progress", "no_change")).lower()
        prog_vals.append(progress_map.get(prog_str, 5))

    # 8a: CGI-I
    ax = axes[0]
    ax.plot(t, data["global_symptom_severity"] * 6 + 1,
            color="gray", alpha=0.3, linewidth=1, label="Global Severity (scaled)")
    ax.scatter(eval_times, cgi_vals, color=COLORS["phq9"], s=80, zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.plot(eval_times, cgi_vals, color=COLORS["phq9"], linewidth=1.5, alpha=0.5)
    for i, (et, cv) in enumerate(zip(eval_times, cgi_vals)):
        ax.annotate(f"{cv:.0f}", (et, cv), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_ylabel("CGI-I (1-7)", fontsize=11)
    ax.set_title("Clinical Global Impression - Improvement", fontsize=12,
                 fontweight="bold")
    ax.set_ylim(0.5, 7.5)
    ax.axhline(y=3, color="green", linestyle=":", alpha=0.4)
    ax.text(t[0] + 1, 3.2, "Minimally improved", fontsize=8, color="green")
    ax.grid(True, alpha=0.3)

    # 8b: 功能改善
    ax = axes[1]
    ax.scatter(eval_times, func_vals, color=COLORS["social"], s=80, zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.plot(eval_times, func_vals, color=COLORS["social"], linewidth=1.5, alpha=0.5)
    ax.set_ylabel("Functional Improvement (0-1)", fontsize=11)
    ax.set_title("Functional Improvement", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 8c: 严重度 + 进展
    ax = axes[2]
    ax.scatter(eval_times, sev_vals, color=COLORS["gad7"], s=80, zorder=5,
               marker="s", label="Severity", edgecolors="white", linewidths=1.5)
    ax.scatter(eval_times, prog_vals, color=COLORS["therapy"], s=80, zorder=5,
               marker="^", label="Progress", edgecolors="white", linewidths=1.5)
    ax.set_xlabel("Time (h)", fontsize=11)
    ax.set_ylabel("Category (1=best, 5=worst)", fontsize=11)
    ax.set_title("Severity & Progress Assessment", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 5.5)

    plt.suptitle(f"LLM Clinical Evaluation Timeline{title_suffix}", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════
# Markdown报告
# ══════════════════════════════════════════════════════════════

def generate_markdown_report(
    result: TherapeuticResult,
    arm_results: Optional[Dict[str, TherapeuticResult]] = None,
    figure_dir: str = "docs/figures/therapeutic",
    report_path: str = "docs/therapeutic_report.md",
) -> str:
    """生成完整的Markdown报告。

    Args:
        result: 主实验结果
        arm_results: 多臂对比结果 (可选)
        figure_dir: 图片保存目录
        report_path: 报告保存路径

    Returns:
        报告文件路径
    """
    os.makedirs(figure_dir, exist_ok=True)

    cfg = result.config
    data = _extract_timeseries(result.trajectory)

    # 生成8张图
    title_suf = f" — {cfg.condition} ({cfg.severity})"
    fig1_symptom_severity(data, f"{figure_dir}/fig1_symptom_severity.png", title_suf)
    fig2_neurotransmitter_dynamics(data, f"{figure_dir}/fig2_neurotransmitter.png", title_suf)
    fig3_drug_concentration_pd(data, f"{figure_dir}/fig3_drug_pd.png", title_suf)
    fig4_brain_regions(data, f"{figure_dir}/fig4_brain_regions.png", title_suf)
    fig5_psychometric_dashboard(data, f"{figure_dir}/fig5_psychometric_dashboard.png", title_suf)
    fig6_synergy_trajectory(data, f"{figure_dir}/fig6_synergy.png", title_suf)

    if arm_results:
        fig7_arm_comparison(arm_results, f"{figure_dir}/fig7_arm_comparison.png")

    fig8_llm_evaluation_timeline(
        result.llm_evaluations, data,
        f"{figure_dir}/fig8_llm_timeline.png", title_suf,
    )

    # ── 汇总统计 ──
    phq9_vals = data.get("depression_severity", np.array([]))
    gad7_vals = data.get("anxiety_level", np.array([]))
    phq9_start = float(phq9_vals[0]) if len(phq9_vals) > 0 else 0
    phq9_end = float(phq9_vals[-1]) if len(phq9_vals) > 0 else 0
    gad7_start = float(gad7_vals[0]) if len(gad7_vals) > 0 else 0
    gad7_end = float(gad7_vals[-1]) if len(gad7_vals) > 0 else 0

    # 药物信息
    drug_lines = []
    for dc in cfg.drugs:
        drug_lines.append(f"- {dc.name_or_smiles} {dc.dose_mg}mg, "
                         f"{dc.freq_per_day}x/day")
    therapy_lines = []
    for tc in cfg.therapies:
        therapy_lines.append(f"- {tc.modality} ({tc.frequency}, "
                            f"intensity={tc.intensity})")

    # LLM评估摘要
    llm_summary = ""
    if result.llm_evaluations:
        last_ev = result.llm_evaluations[-1]
        llm_summary = f"""
### LLM Clinical Assessment (Final)

| Field | Value |
|-------|-------|
| Diagnostic Summary | {last_ev.get('diagnostic_summary', 'N/A')} |
| Severity | {last_ev.get('severity_assessment', 'N/A')} |
| Treatment Progress | {last_ev.get('treatment_progress', 'N/A')} |
| Relapse Risk | {last_ev.get('relapse_risk', 'N/A')} |
| Side Effect Risk | {last_ev.get('side_effect_risk', 'N/A')} |
| CGI-I | {last_ev.get('clinical_global_impression', 'N/A')} |
| Functional Improvement | {last_ev.get('functional_improvement', 'N/A')} |

**Recommendations:**
"""
        for adj in last_ev.get("recommended_adjustments", []):
            llm_summary += f"- {adj}\n"

    # 病理警告
    all_warnings = []
    for tp in result.trajectory:
        all_warnings.extend(tp.warnings)
    unique_warnings = list(dict.fromkeys(all_warnings))[:10]

    # ── 组装Markdown ──
    md = f"""# Computational Psychopharmacology — Therapeutic Experiment Report

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Condition | {cfg.condition} |
| Severity | {cfg.severity} |
| Duration | {cfg.duration_steps} steps ({cfg.duration_steps / cfg.steps_per_hour:.0f} h) |
| Follow-up | {cfg.follow_up_steps} steps ({cfg.follow_up_steps / cfg.steps_per_hour:.0f} h) |
| Observation Interval | Every {cfg.observation_interval} steps |

### Drug Regimen
{chr(10).join(drug_lines) if drug_lines else "- None (placebo)"}

### Psychotherapy
{chr(10).join(therapy_lines) if therapy_lines else "- None"}

---

## Outcome Summary

| Metric | Value |
|--------|-------|
| Remission Rate | {result.remission_rate:.1%} |
| Relapse Rate | {result.relapse_rate:.1%} |
| Peak Drug Effect | {result.peak_drug_effect:.3f} |
| Total Therapy Sessions | {result.total_therapy_sessions} |
| PHQ-9 Change | {phq9_start:.1f} → {phq9_end:.1f} ({phq9_end - phq9_start:+.1f}) |
| GAD-7 Change | {gad7_start:.1f} → {gad7_end:.1f} ({gad7_end - gad7_start:+.1f}) |

{llm_summary}

---

## Figures

### 1. Symptom Severity Over Time
![Symptom Severity]({figure_dir}/fig1_symptom_severity.png)

### 2. Neurotransmitter Dynamics
![Neurotransmitter]({figure_dir}/fig2_neurotransmitter.png)

### 3. Drug Concentration & PD Effect
![Drug PD]({figure_dir}/fig3_drug_pd.png)

### 4. Brain Region Dynamics
![Brain Regions]({figure_dir}/fig4_brain_regions.png)

### 5. Psychometric Dashboard
![Dashboard]({figure_dir}/fig5_psychometric_dashboard.png)

### 6. Drug-Therapy Synergy
![Synergy]({figure_dir}/fig6_synergy.png)

### 7. Multi-Arm Comparison
![Comparison]({figure_dir}/fig7_arm_comparison.png)

### 8. LLM Evaluation Timeline
![LLM Timeline]({figure_dir}/fig8_llm_timeline.png)

---

## Pathological State Warnings

"""
    if unique_warnings:
        for w in unique_warnings:
            md += f"- {w}\n"
    else:
        md += "No pathological states detected.\n"

    md += f"""
---

*Report generated by Simulacrum Computational Psychopharmacology Sandbox*
*Condition: {cfg.condition} | Severity: {cfg.severity} | Steps: {cfg.duration_steps + cfg.follow_up_steps}*
"""

    # 写入文件
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    return report_path


__all__ = [
    "fig1_symptom_severity",
    "fig2_neurotransmitter_dynamics",
    "fig3_drug_concentration_pd",
    "fig4_brain_regions",
    "fig5_psychometric_dashboard",
    "fig6_synergy_trajectory",
    "fig7_arm_comparison",
    "fig8_llm_evaluation_timeline",
    "generate_markdown_report",
]
