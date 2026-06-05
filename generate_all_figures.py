"""Generate detailed figures for all 10 Simulacrum experiments.

Uses matplotlib to generate PNG images saved to docs/figures/.
Each experiment generates 2-4 key figures.

Usage:
    python generate_all_figures.py          # all experiments
    python generate_all_figures.py 1 5 10   # only experiments 1, 5, 10
"""

import sys
import os
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = os.path.join(_project_root, 'docs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Global style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'legend.fontsize': 8,
    'figure.figsize': (10, 6),
})

COLORS = {
    'blue': '#2563EB',
    'red': '#DC2626',
    'green': '#16A34A',
    'orange': '#EA580C',
    'purple': '#9333EA',
    'gray': '#6B7280',
}


def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ========================================================================
# Experiment 1: Thermodynamic Collapse
# Return: dict keyed by 'Rich','Balanced','Poverty'
#   {group, ttd, compression_count, final_balance, exploration_entropy,
#    mean_balance, mean_exploration, mean_social, balance_trajectory, social_trajectory}
# ========================================================================

def fig_exp1():
    print("\n[Fig 1] Thermodynamic Collapse")
    from experiment_thermodynamic_collapse import run_experiment
    results = run_experiment()

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig)

    groups = ['Rich', 'Balanced', 'Poverty']
    colors = [COLORS['blue'], COLORS['green'], COLORS['red']]

    # 1a: Balance trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    for g, c in zip(groups, colors):
        traj = results[g]['balance_trajectory']
        ax1.plot(traj, label=g, color=c, alpha=0.8, linewidth=1)
    ax1.set_title('Balance Trajectory')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Balance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1b: Social engagement trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    for g, c in zip(groups, colors):
        traj = results[g]['social_trajectory']
        ax2.plot(traj, label=g, color=c, alpha=0.8, linewidth=1)
    ax2.set_title('Social Engagement')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Social Engagement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 1c: Summary bar chart (normalized)
    ax3 = fig.add_subplot(gs[0, 2])
    metrics = ['TTD', 'Compressions', 'Entropy (bit)', 'Final Balance']
    vals = np.array([
        [results[g]['ttd'] for g in groups],
        [results[g]['compression_count'] for g in groups],
        [results[g]['exploration_entropy'] for g in groups],
        [results[g]['final_balance'] for g in groups],
    ])
    x = np.arange(len(metrics))
    width = 0.25
    for i, (g, c) in enumerate(zip(groups, colors)):
        normalized = vals[:, i] / np.maximum(vals.max(axis=1), 1)
        ax3.bar(x + i * width, normalized, width, label=g, color=c, alpha=0.8)
    ax3.set_title('Group Comparison (normalized)')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(metrics, rotation=15, fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 1: Digital Thermodynamic Collapse', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'exp1_balance_trajectory.png')


# ========================================================================
# Experiment 2: Metabolic Sparsity
# Return: {'Control': {...}, 'Experimental': {...}}
#   {group, resource_budget, bl_explore, bl_active, st_explore, st_active,
#    rc_explore, rc_active, explore_drop_pct, recovery_time,
#    allostatic_trajectory, active_ratio_trajectory}
# NOTE: No explore_trajectory, stress_active is st_active
# ========================================================================

def fig_exp2():
    print("\n[Fig 2] Metabolic Sparsity")
    from experiment_metabolic_sparsity import run_experiment
    results = run_experiment()

    ctrl, exp = results['Control'], results['Experimental']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 2a: Active ratio trajectory
    ax = axes[0]
    ax.plot(ctrl['active_ratio_trajectory'], label='Control', color=COLORS['blue'], alpha=0.8)
    ax.plot(exp['active_ratio_trajectory'], label='Experimental', color=COLORS['red'], alpha=0.8)
    ax.axvline(200, color='gray', linestyle='--', alpha=0.5, label='Stress onset')
    ax.axvline(600, color='gray', linestyle=':', alpha=0.5, label='Recovery')
    ax.set_title('Active Ratio (Neuron Utilization)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Active Ratio')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2b: Allostatic load trajectory
    ax = axes[1]
    ax.plot(ctrl['allostatic_trajectory'], label='Control', color=COLORS['blue'], alpha=0.8)
    ax.plot(exp['allostatic_trajectory'], label='Experimental', color=COLORS['red'], alpha=0.8)
    ax.axvline(200, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(600, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Allostatic Load')
    ax.set_xlabel('Step')
    ax.set_ylabel('Allostatic Load')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2c: Bar chart comparison
    ax = axes[2]
    labels = ['BL Active', 'Stress Active', 'RC Active', 'Explore Drop%']
    ctrl_vals = [ctrl['bl_active'], ctrl['st_active'], ctrl['rc_active'], ctrl['explore_drop_pct']]
    exp_vals = [exp['bl_active'], exp['st_active'], exp['rc_active'], exp['explore_drop_pct']]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, ctrl_vals, width, label='Control', color=COLORS['blue'], alpha=0.8)
    ax.bar(x + width/2, exp_vals, width, label='Experimental', color=COLORS['red'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Phase Comparison')

    fig.suptitle('Experiment 2: Metabolic Sparsity & Zombie Neurons', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'exp2_active_ratio.png')


# ========================================================================
# Experiment 3: HPA Cognitive Rigidity
# Return: {history: {cortisol:[], pfc_inhibition:[], exploration_rate:[],
#            allostatic_load:[], social_engagement:[], ...},
#          stuck_ratios: [], novelty_scores: []}
# Single run with 3 phases: Baseline(0-500), Stress(500-1000), Recovery(1000-1500)
# ========================================================================

def fig_exp3():
    print("\n[Fig 3] HPA Cognitive Rigidity")
    from experiment_hpa_cognitive_rigidity import run_experiment
    results = run_experiment()

    history = results['history']
    stuck_ratios = results['stuck_ratios']
    novelty_scores = results['novelty_scores']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    n = len(history['cortisol'])

    # 3a: Cortisol trajectory
    ax = axes[0, 0]
    ax.plot(history['cortisol'], color=COLORS['red'], alpha=0.7, linewidth=0.5)
    ax.axvline(500, color='gray', linestyle='--', alpha=0.5, label='Stress onset')
    ax.axvline(1000, color='gray', linestyle=':', alpha=0.5, label='Recovery')
    ax.set_title('Cortisol Trajectory')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cortisol')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3b: PFC inhibition
    ax = axes[0, 1]
    ax.plot(history['pfc_inhibition'], color=COLORS['blue'], alpha=0.7, linewidth=0.5)
    ax.axvline(500, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(1000, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('PFC Inhibition')
    ax.set_xlabel('Step')
    ax.set_ylabel('PFC Inhibition')
    ax.grid(True, alpha=0.3)

    # 3c: Stuck ratio & novelty
    ax = axes[1, 0]
    if stuck_ratios:
        ax.plot(stuck_ratios, color=COLORS['orange'], alpha=0.7, linewidth=0.5, label='Stuck Ratio')
    if novelty_scores:
        ax2 = ax.twinx()
        ax2.plot(novelty_scores, color=COLORS['purple'], alpha=0.5, linewidth=0.5, label='Novelty')
        ax2.set_ylabel('Novelty Score', color=COLORS['purple'])
        ax2.legend(loc='upper right', fontsize=7)
    ax.axvline(500, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(1000, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Cognitive Rigidity (Stuck Ratio & Novelty)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Stuck Ratio', color=COLORS['orange'])
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3d: Phase averages bar chart (compute from history)
    ax = axes[1, 1]
    bl = 500
    st = 1000
    phases = ['Baseline', 'Stress', 'Recovery']
    phase_avgs = {
        'Cortisol': [
            np.mean(history['cortisol'][:bl]),
            np.mean(history['cortisol'][bl:st]),
            np.mean(history['cortisol'][st:]),
        ],
        'PFC': [
            np.mean(history['pfc_inhibition'][:bl]),
            np.mean(history['pfc_inhibition'][bl:st]),
            np.mean(history['pfc_inhibition'][st:]),
        ],
        'Explore Rate': [
            np.mean(history['exploration_rate'][:bl]),
            np.mean(history['exploration_rate'][bl:st]),
            np.mean(history['exploration_rate'][st:]),
        ],
    }
    x = np.arange(len(phases))
    width = 0.25
    for i, (metric, vals) in enumerate(phase_avgs.items()):
        ax.bar(x + i * width, vals, width, label=metric, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(phases)
    ax.set_title('Phase Averages')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 3: HPA Stress -> Cognitive Rigidity', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'exp3_rigidity_trajectory.png')


# ========================================================================
# Experiment 4: Epigenetic Consolidation
# Return: dict keyed by 'High','Medium','Low'
#   {group, emotional_threshold, phase1_tag_count, phase3_tag_count,
#    forgetting_ratio, phase1_lora_norm, phase3_lora_norm, lora_divergence,
#    history: {n_epigenetic_tags:[], ...}}
# ========================================================================

def fig_exp4():
    print("\n[Fig 4] Epigenetic Consolidation")
    from experiment_epigenetic_consolidation import run_experiment
    results = run_experiment()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    groups = ['High', 'Medium', 'Low']
    labels = ['High(0.9)', 'Medium(0.7)', 'Low(0.5)']
    colors = [COLORS['red'], COLORS['orange'], COLORS['green']]

    # 4a: Tag accumulation trajectory
    ax = axes[0]
    for g, lbl, c in zip(groups, labels, colors):
        tag_traj = results[g]['history']['n_epigenetic_tags']
        ax.plot(tag_traj, label=lbl, color=c, alpha=0.8, linewidth=1.5)
    ax.set_title('Epigenetic Tag Accumulation')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tag Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4b: Final metrics bar chart
    ax = axes[1]
    metrics = ['Phase1 Tags', 'Phase3 Tags', 'Forgetting', 'LoRA Div']
    vals = np.array([
        [results[g]['phase1_tag_count'] for g in groups],
        [results[g]['phase3_tag_count'] for g in groups],
        [results[g]['forgetting_ratio'] for g in groups],
        [results[g]['lora_divergence'] for g in groups],
    ])
    x = np.arange(len(metrics))
    width = 0.25
    for i, (lbl, c) in enumerate(zip(labels, colors)):
        ax.bar(x + i * width, vals[:, i], width, label=lbl, color=c, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=15, fontsize=8)
    ax.set_title('Consolidation Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 4: Epigenetic Memory Consolidation', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'exp4_tag_accumulation.png')


# ========================================================================
# Experiment 5: Stockholm Pressure Cooker
# Return: {history: {balance:[], cortisol:[], social_engagement:[], ...},
#          bonding_history: {bonding_score:[], fight_ratio:[], fawn_ratio:[], resource_dependency:[]}}
# Single run with 3 sequential phases:
#   Resistance (0-400), Pressure (400-800), Bonding (800-1200)
# ========================================================================

def fig_exp5():
    print("\n[Fig 5] Stockholm Pressure Cooker")
    from experiment_stockholm_pressure import run_experiment
    results = run_experiment()

    history = results['history']
    bonding = results['bonding_history']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 5a: Bonding score trajectory across phases
    ax = axes[0]
    ax.plot(bonding['bonding_score'], color=COLORS['purple'], alpha=0.8, linewidth=1)
    ax.axvline(400, color='gray', linestyle='--', alpha=0.5, label='Pressure onset')
    ax.axvline(800, color='gray', linestyle=':', alpha=0.5, label='Bonding phase')
    ax.set_title('Bonding Score Development')
    ax.set_xlabel('Step')
    ax.set_ylabel('Bonding Score')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5b: Fight vs Fawn ratio evolution
    ax = axes[1]
    ax.plot(bonding['fight_ratio'], color=COLORS['red'], alpha=0.8, linewidth=1.2, label='Fight Ratio')
    ax.plot(bonding['fawn_ratio'], color=COLORS['purple'], alpha=0.8, linewidth=1.2, label='Fawn Ratio')
    ax.axvline(400, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(800, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Defense Strategy: Fight vs Fawn Transition')
    ax.set_xlabel('Step')
    ax.set_ylabel('Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5c: Balance & cortisol trajectories
    ax = axes[2]
    ax.plot(history['balance'], color=COLORS['blue'], alpha=0.8, linewidth=1, label='Balance')
    ax2 = ax.twinx()
    ax2.plot(history['cortisol'], color=COLORS['red'], alpha=0.5, linewidth=0.5, label='Cortisol')
    ax2.set_ylabel('Cortisol', color=COLORS['red'])
    ax.axvline(400, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(800, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Balance & Cortisol Dynamics')
    ax.set_xlabel('Step')
    ax.set_ylabel('Balance', color=COLORS['blue'])
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 5: Stockholm Syndrome - Bonding Under Pressure', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'exp5_bonding_trajectory.png')


# ========================================================================
# Experiment 6: Glymphatic Timing
# Return: dict keyed by 'Continuous','SleepGated','GammaTrigger'
#   {group, strategy, phase1_waste, phase1_health, phase1_explore,
#    phase3_waste, phase3_health, phase3_explore, memory_retention,
#    toxicity_index, clearance_efficiency, waste_cleared_total,
#    waste_trajectory, health_trajectory, explore_trajectory}
# NOTE: No overall_efficiency key
# ========================================================================

def fig_exp6():
    print("\n[Fig 6] Glymphatic Timing")
    from experiment_6_glymphatic_timing import run_experiment
    results = run_experiment()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    strategies = ['Continuous', 'SleepGated', 'GammaTrigger']
    labels = ['Continuous', 'Sleep-gated', 'Gamma']
    colors = [COLORS['red'], COLORS['blue'], COLORS['green']]

    # 6a: Waste trajectory
    ax = axes[0]
    for s, lbl, c in zip(strategies, labels, colors):
        ax.plot(results[s]['waste_trajectory'], label=lbl, color=c, alpha=0.8)
    ax.axvline(30, color='gray', linestyle='--', alpha=0.5, label='Clearance onset')
    ax.axvline(70, color='gray', linestyle=':', alpha=0.5, label='Assessment')
    ax.set_title('Brain Waste Over Time')
    ax.set_xlabel('Step (downsampled)')
    ax.set_ylabel('Waste Level')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 6b: Brain health trajectory
    ax = axes[1]
    for s, lbl, c in zip(strategies, labels, colors):
        ax.plot(results[s]['health_trajectory'], label=lbl, color=c, alpha=0.8)
    ax.set_title('Brain Health')
    ax.set_xlabel('Step (downsampled)')
    ax.set_ylabel('Health')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 6c: Efficiency comparison
    ax = axes[2]
    metrics = ['Memory Retention', 'Clearance Eff', 'Toxicity Idx']
    vals = np.array([
        [results[s]['memory_retention'] for s in strategies],
        [results[s]['clearance_efficiency'] for s in strategies],
        [results[s]['toxicity_index'] for s in strategies],
    ])
    x = np.arange(len(metrics))
    width = 0.25
    for i, (lbl, c) in enumerate(zip(labels, colors)):
        ax.bar(x + i * width, vals[:, i], width, label=lbl, color=c, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=10, fontsize=8)
    ax.set_title('Strategy Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 6: Glymphatic Clearance Timing', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'exp6_clearance_comparison.png')


# ========================================================================
# Bonus Experiment A: Stress Anhedonia (Digital PTSD)
# Return: history dict {cortisol, pfc_inhibition, exploration_rate,
#   symptom_anhedonia, motivation_lambda, da, 5ht, allostatic_load, ...}
# Single run: Baseline(0-200) -> Stress(200-800) -> Recovery(800-1200)
# ========================================================================

def fig_exp_stress():
    print("\n[Fig A] Stress Anhedonia (PTSD)")
    from experiment_stress_anhedonia import run_experiment
    history = run_experiment()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Aa: Cortisol + PFC
    ax = axes[0, 0]
    ax.plot(history['cortisol'], color=COLORS['red'], alpha=0.7, linewidth=0.5, label='Cortisol')
    ax2 = ax.twinx()
    ax2.plot(history['pfc_inhibition'], color=COLORS['blue'], alpha=0.5, linewidth=0.5, label='PFC')
    ax.axvline(200, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(800, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Cortisol & PFC Inhibition')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cortisol', color=COLORS['red'])
    ax2.set_ylabel('PFC', color=COLORS['blue'])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Ab: Exploration rate
    ax = axes[0, 1]
    ax.plot(history['exploration_rate'], color=COLORS['green'], alpha=0.7, linewidth=0.5)
    ax.axvline(200, color='gray', linestyle='--', alpha=0.5, label='Stress onset')
    ax.axvline(800, color='gray', linestyle=':', alpha=0.5, label='Recovery')
    ax.set_title('Exploration Rate')
    ax.set_xlabel('Step')
    ax.set_ylabel('Exploration Rate')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Ac: Anhedonia + DA
    ax = axes[1, 0]
    ax.plot(history['symptom_anhedonia'], color=COLORS['orange'], alpha=0.7, linewidth=0.5, label='Anhedonia')
    ax2 = ax.twinx()
    ax2.plot(history['da'], color=COLORS['purple'], alpha=0.5, linewidth=0.5, label='DA')
    ax.axvline(200, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(800, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Anhedonia & Dopamine')
    ax.set_xlabel('Step')
    ax.set_ylabel('Anhedonia', color=COLORS['orange'])
    ax2.set_ylabel('DA', color=COLORS['purple'])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Ad: Phase averages
    ax = axes[1, 1]
    phases = ['Baseline', 'Stress', 'Recovery']
    phase_avgs = {
        'Cortisol': [np.mean(history['cortisol'][:200]),
                     np.mean(history['cortisol'][200:800]),
                     np.mean(history['cortisol'][800:])],
        'PFC': [np.mean(history['pfc_inhibition'][:200]),
                np.mean(history['pfc_inhibition'][200:800]),
                np.mean(history['pfc_inhibition'][800:])],
        'Anhedonia': [np.mean(history['symptom_anhedonia'][:200]),
                      np.mean(history['symptom_anhedonia'][200:800]),
                      np.mean(history['symptom_anhedonia'][800:])],
    }
    x = np.arange(len(phases))
    width = 0.25
    for i, (metric, vals) in enumerate(phase_avgs.items()):
        ax.bar(x + i * width, vals, width, label=metric, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(phases)
    ax.set_title('Phase Averages')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Bonus Exp: Digital PTSD & Anhedonia', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'expA_stress_anhedonia.png')


# ========================================================================
# Bonus Experiment B: Drug Decision Drift
# Return: history dict {da, 5ht, gaba, exploration_rate, pfc, precision,
#   motivation_lambda, arousal, symptom_panic, ...}
# Single run with sequential drug phases:
#   Baseline(0-200) -> Hallucinogen(200-500) -> Washout(500-550)
#   -> Sedative(550-850) -> Washout(850-900) -> Stimulant(900-1200) -> Washout(1200-1400)
# ========================================================================

def fig_exp_drug():
    print("\n[Fig B] Drug Decision Drift")
    from experiment_drug_decision import run_experiment
    history = run_experiment()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Phase boundaries
    phases = [
        (0, 200, 'Baseline', 'white'),
        (200, 500, 'Hallucinogen', '#FFE0E0'),
        (550, 850, 'Sedative', '#E0E0FF'),
        (900, 1200, 'Stimulant', '#E0FFE0'),
    ]

    def shade_phases(ax):
        for start, end, label, color in phases:
            ax.axvspan(start, end, alpha=0.15, color=color)

    # Ba: NT levels (DA, 5HT, GABA)
    ax = axes[0, 0]
    ax.plot(history['da'], color=COLORS['red'], alpha=0.6, linewidth=0.5, label='DA')
    ax.plot(history['5ht'], color=COLORS['blue'], alpha=0.6, linewidth=0.5, label='5-HT')
    ax.plot(history['gaba'], color=COLORS['green'], alpha=0.6, linewidth=0.5, label='GABA')
    shade_phases(ax)
    ax.set_title('Neurotransmitter Levels')
    ax.set_xlabel('Step')
    ax.set_ylabel('Level')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bb: Exploration rate
    ax = axes[0, 1]
    ax.plot(history['exploration_rate'], color=COLORS['green'], alpha=0.7, linewidth=0.5)
    shade_phases(ax)
    ax.set_title('Exploration Rate')
    ax.set_xlabel('Step')
    ax.set_ylabel('Exploration Rate')
    ax.grid(True, alpha=0.3)

    # Bc: PFC + Precision
    ax = axes[1, 0]
    ax.plot(history['pfc'], color=COLORS['blue'], alpha=0.7, linewidth=0.5, label='PFC')
    ax2 = ax.twinx()
    ax2.plot(history['precision'], color=COLORS['orange'], alpha=0.5, linewidth=0.5, label='Precision')
    shade_phases(ax)
    ax.set_title('PFC & Predictive Precision')
    ax.set_xlabel('Step')
    ax.set_ylabel('PFC', color=COLORS['blue'])
    ax2.set_ylabel('Precision', color=COLORS['orange'])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bd: Phase comparison bars
    ax = axes[1, 1]
    phase_labels = ['Baseline', 'Halluc.', 'Sedative', 'Stimulant']
    phase_ranges = [(0, 200), (200, 500), (550, 850), (900, 1200)]
    explore_avgs = [np.mean(history['exploration_rate'][s:e]) for s, e in phase_ranges]
    motiv_avgs = [np.mean(history['motivation_lambda'][s:e]) for s, e in phase_ranges]
    da_avgs = [np.mean(history['da'][s:e]) for s, e in phase_ranges]
    x = np.arange(len(phase_labels))
    width = 0.25
    ax.bar(x - width, explore_avgs, width, label='Explore', color=COLORS['green'], alpha=0.8)
    ax.bar(x, motiv_avgs, width, label='Motivation', color=COLORS['orange'], alpha=0.8)
    ax.bar(x + width, da_avgs, width, label='DA', color=COLORS['red'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_title('Drug Phase Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Bonus Exp: Drug-Induced Decision Drift', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'expB_drug_decision.png')


# ========================================================================
# Bonus Experiment C: Social Decay & Withdrawal
# Return: history dict {social_engagement, oxytocin, exploration_rate,
#   empathy_level, social_withdrawal, cortisol, ...}
# Single run: Baseline(0-200) -> Metabolic Stress(200-700) -> Recovery(700-1000)
# ========================================================================

def fig_exp_social():
    print("\n[Fig C] Social Decay & Withdrawal")
    from experiment_social_decay import run_experiment
    history = run_experiment()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Ca: Social engagement + oxytocin
    ax = axes[0, 0]
    ax.plot(history['social_engagement'], color=COLORS['blue'], alpha=0.7, linewidth=0.5, label='Social')
    ax2 = ax.twinx()
    ax2.plot(history['oxytocin'], color=COLORS['purple'], alpha=0.5, linewidth=0.5, label='Oxytocin')
    ax.axvline(200, color='gray', linestyle='--', alpha=0.5, label='Stress onset')
    ax.axvline(700, color='gray', linestyle=':', alpha=0.5, label='Recovery')
    ax.set_title('Social Engagement & Oxytocin')
    ax.set_xlabel('Step')
    ax.set_ylabel('Social', color=COLORS['blue'])
    ax2.set_ylabel('Oxytocin', color=COLORS['purple'])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Cb: Empathy + self-coherence
    ax = axes[0, 1]
    ax.plot(history['empathy_level'], color=COLORS['green'], alpha=0.7, linewidth=0.5, label='Empathy')
    ax.plot(history['self_coherence'], color=COLORS['orange'], alpha=0.5, linewidth=0.5, label='Coherence')
    ax.axvline(200, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(700, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Empathy & Self-Coherence')
    ax.set_xlabel('Step')
    ax.set_ylabel('Level')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Cc: Social withdrawal flag + cortisol
    ax = axes[1, 0]
    ax.plot(history['social_withdrawal'], color=COLORS['red'], alpha=0.7, linewidth=0.5, label='Withdrawal')
    ax2 = ax.twinx()
    ax2.plot(history['cortisol'], color=COLORS['orange'], alpha=0.5, linewidth=0.3, label='Cortisol')
    ax.axvline(200, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(700, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Social Withdrawal & Cortisol')
    ax.set_xlabel('Step')
    ax.set_ylabel('Withdrawal', color=COLORS['red'])
    ax2.set_ylabel('Cortisol', color=COLORS['orange'])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Cd: Phase averages
    ax = axes[1, 1]
    phases = ['Baseline', 'Stress', 'Recovery']
    phase_avgs = {
        'Social': [np.mean(history['social_engagement'][:200]),
                   np.mean(history['social_engagement'][200:700]),
                   np.mean(history['social_engagement'][700:])],
        'Empathy': [np.mean(history['empathy_level'][:200]),
                    np.mean(history['empathy_level'][200:700]),
                    np.mean(history['empathy_level'][700:])],
        'Anhedonia': [np.mean(history['symptom_anhedonia'][:200]),
                      np.mean(history['symptom_anhedonia'][200:700]),
                      np.mean(history['symptom_anhedonia'][700:])],
    }
    x = np.arange(len(phases))
    width = 0.25
    for i, (metric, vals) in enumerate(phase_avgs.items()):
        ax.bar(x + i * width, vals, width, label=metric, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(phases)
    ax.set_title('Phase Averages')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Bonus Exp: Social Decay & Withdrawal', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'expC_social_decay.png')


# ========================================================================
# Experiment 7: ADHD Flicker Fusion
# Return: {'Normal': {...}, 'ADHD': {...}}
#   {group, adhd_mode, bl_da, noise_da, rc_da, bl_met, noise_met,
#    noise_waste, attentional_blink_rate, metabolic_drain_pct,
#    da_fatigue_pct, da_trajectory, met_trajectory}
# Keys match figure expectations.
# ========================================================================

def fig_exp7():
    print("\n[Fig 7] ADHD Flicker Fusion")
    from experiment_7_adhd_flicker import run_experiment
    results = run_experiment()
    normal, adhd = results['Normal'], results['ADHD']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 7a: DA trajectory
    ax = axes[0]
    ax.plot(normal['da_trajectory'], label='Normal', color=COLORS['blue'], alpha=0.8)
    ax.plot(adhd['da_trajectory'], label='ADHD', color=COLORS['red'], alpha=0.8)
    ax.set_title('Dopamine Trajectory')
    ax.set_xlabel('Step (downsampled)')
    ax.set_ylabel('DA Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7b: Metabolic cost
    ax = axes[1]
    ax.plot(normal['met_trajectory'], label='Normal', color=COLORS['blue'], alpha=0.8)
    ax.plot(adhd['met_trajectory'], label='ADHD', color=COLORS['red'], alpha=0.8)
    ax.set_title('Metabolic Cost')
    ax.set_xlabel('Step (downsampled)')
    ax.set_ylabel('Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7c: Summary bars
    ax = axes[2]
    metrics = ['Blink Rate', 'Met Drain%', 'DA Fatigue%']
    n_vals = [normal['attentional_blink_rate'], normal['metabolic_drain_pct'], normal['da_fatigue_pct']]
    a_vals = [adhd['attentional_blink_rate'], adhd['metabolic_drain_pct'], adhd['da_fatigue_pct']]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, n_vals, width, label='Normal', color=COLORS['blue'], alpha=0.8)
    ax.bar(x + width/2, a_vals, width, label='ADHD', color=COLORS['red'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title('Attention Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 7: ADHD Critical Flicker Fusion', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'exp7_adhd_comparison.png')


# ========================================================================
# Experiment 8: Digital Dreaming
# Return: {'Normal': {...}, 'PTSD': {...}}
#   {group, ptsd_mode, phase1_cortisol, phase2_cortisol, phase3_cortisol,
#    phase1_waste, phase3_waste, consolidation_ratio, fear_extinction_pct,
#    synaptic_homeostasis, phase3_bdnf,
#    cortisol_trajectory, waste_trajectory}
# Keys match figure expectations.
# ========================================================================

def fig_exp8():
    print("\n[Fig 8] Digital Dreaming")
    from experiment_8_digital_dreaming import run_experiment
    results = run_experiment()
    normal, ptsd = results['Normal'], results['PTSD']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 8a: Cortisol
    ax = axes[0]
    ax.plot(normal['cortisol_trajectory'], label='Normal', color=COLORS['blue'], alpha=0.8)
    ax.plot(ptsd['cortisol_trajectory'], label='PTSD', color=COLORS['red'], alpha=0.8)
    # Phase boundaries (downsampled by 10): 300/10=30, 700/10=70
    ax.axvspan(30, 70, alpha=0.1, color='gray', label='Sleep phase')
    ax.set_title('Cortisol Trajectory')
    ax.set_xlabel('Step (downsampled)')
    ax.set_ylabel('Cortisol')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8b: Brain waste
    ax = axes[1]
    ax.plot(normal['waste_trajectory'], label='Normal', color=COLORS['blue'], alpha=0.8)
    ax.plot(ptsd['waste_trajectory'], label='PTSD', color=COLORS['red'], alpha=0.8)
    ax.set_title('Brain Waste')
    ax.set_xlabel('Step (downsampled)')
    ax.set_ylabel('Waste')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8c: Metrics comparison
    ax = axes[2]
    metrics = ['Ph2 Cort', 'Ph3 Cort', 'Consolidation', 'Fear Ext%']
    n_vals = [normal['phase2_cortisol'], normal['phase3_cortisol'],
              normal['consolidation_ratio'], normal['fear_extinction_pct']]
    p_vals = [ptsd['phase2_cortisol'], ptsd['phase3_cortisol'],
              ptsd['consolidation_ratio'], ptsd['fear_extinction_pct']]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, n_vals, width, label='Normal', color=COLORS['blue'], alpha=0.8)
    ax.bar(x + width/2, p_vals, width, label='PTSD', color=COLORS['red'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, fontsize=8)
    ax.set_title('Sleep & Fear Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 8: Digital Dreaming & PTSD Flashback', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'exp8_dreaming_comparison.png')


# ========================================================================
# Experiment 9: Autism Spectrum
# Return: dict keyed by 'Low','Medium','High'
#   {group, resonance_value, bl_social, soc_social, soc_emp, soc_met,
#    ost_social, ost_cort, tom_score, social_battery_drain,
#    ostracism_stress, social_trajectory, cortisol_trajectory}
# Keys match figure expectations.
# ========================================================================

def fig_exp9():
    print("\n[Fig 9] Autism Spectrum")
    from experiment_9_autism_spectrum import run_experiment
    results = run_experiment()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    groups = ['Low', 'Medium', 'High']
    colors = [COLORS['red'], COLORS['blue'], COLORS['green']]

    # 9a: Social engagement
    ax = axes[0]
    for g, c in zip(groups, colors):
        ax.plot(results[g]['social_trajectory'], label=g, color=c, alpha=0.8)
    # Phase boundaries (downsampled by 10): 200/10=20, 700/10=70
    ax.axvspan(20, 70, alpha=0.05, color='yellow', label='Social phase')
    ax.set_title('Social Engagement')
    ax.set_xlabel('Step (downsampled)')
    ax.set_ylabel('Engagement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9b: Cortisol
    ax = axes[1]
    for g, c in zip(groups, colors):
        ax.plot(results[g]['cortisol_trajectory'], label=g, color=c, alpha=0.8)
    ax.set_title('Cortisol Response')
    ax.set_xlabel('Step (downsampled)')
    ax.set_ylabel('Cortisol')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9c: Summary metrics
    ax = axes[2]
    metrics = ['ToM Score', 'Social Drain%', 'Ostracism Stress%']
    vals = np.array([
        [results[g]['tom_score'] for g in groups],
        [results[g]['social_battery_drain'] for g in groups],
        [results[g]['ostracism_stress'] for g in groups],
    ])
    x = np.arange(len(metrics))
    width = 0.25
    for i, (g, c) in enumerate(zip(groups, colors)):
        ax.bar(x + i * width, vals[:, i], width, label=g, color=c, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=10, fontsize=8)
    ax.set_title('Social Cognition Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 9: Social Brain & Autism Spectrum', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'exp9_autism_comparison.png')


# ========================================================================
# Experiment 10: D2 Occupancy Rate
# Return: dict keyed by 'Low(30%)','Medium(75%)','High(95%)'
#   {group, d2_blockade, phase1_psi, phase2_psi, phase3_psi,
#    phase2_eps, phase3_eps, psi_reduction_pct, therapeutic_index,
#    phase3_explore, phase3_anhedonia, phase3_social,
#    psi_trajectory, eps_trajectory}
# Keys match figure expectations.
# ========================================================================

def fig_exp10():
    print("\n[Fig 10] D2 Occupancy Rate")
    from experiment_10_d2_occupancy import run_experiment
    results = run_experiment()

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)

    groups = ['Low(30%)', 'Medium(75%)', 'High(95%)']
    colors = [COLORS['blue'], COLORS['green'], COLORS['red']]

    # 10a: PSI trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    for g, c in zip(groups, colors):
        ax1.plot(results[g]['psi_trajectory'], label=g, color=c, alpha=0.7, linewidth=0.8)
    ax1.axvline(200, color='gray', linestyle='--', alpha=0.5, label='Treatment onset')
    ax1.set_title('Positive Symptom Index (PSI)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('PSI')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 10b: EPS trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    for g, c in zip(groups, colors):
        ax2.plot(results[g]['eps_trajectory'], label=g, color=c, alpha=0.7, linewidth=0.8)
    ax2.axvline(200, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('EPS (Extrapyramidal Side Effects)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('EPS Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 10c: Inverted U curve
    ax3 = fig.add_subplot(gs[1, 0])
    blockades = [30, 75, 95]
    psi_reduction = [results[g]['psi_reduction_pct'] for g in groups]
    eps_vals = [results[g]['phase3_eps'] for g in groups]
    ax3_twin = ax3.twinx()
    l1 = ax3.plot(blockades, psi_reduction, 'o-', color=COLORS['blue'], linewidth=2, markersize=8, label='Symptom Reduction%')
    l2 = ax3_twin.plot(blockades, eps_vals, 's--', color=COLORS['red'], linewidth=2, markersize=8, label='EPS Index')
    ax3.set_xlabel('D2 Blockade (%)')
    ax3.set_ylabel('Symptom Reduction (%)', color=COLORS['blue'])
    ax3_twin.set_ylabel('EPS Index', color=COLORS['red'])
    ax3.set_title('Inverted-U Therapeutic Curve')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='center left')
    ax3.grid(True, alpha=0.3)

    # 10d: Therapeutic index bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    ti_vals = [results[g]['therapeutic_index'] for g in groups]
    bar_colors = [COLORS['blue'], COLORS['green'], COLORS['red']]
    bars = ax4.bar(groups, ti_vals, color=bar_colors, alpha=0.8)
    ax4.set_title('Therapeutic Index')
    ax4.set_ylabel('Index')
    ax4.grid(True, alpha=0.3, axis='y')
    # Highlight best
    max_idx = np.argmax(ti_vals)
    bars[max_idx].set_edgecolor('black')
    bars[max_idx].set_linewidth(2)

    fig.suptitle('Experiment 10: D2 Occupancy Rate - Antipsychotic Simulation', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'exp10_inverted_u.png')


# ========================================================================
# Main
# ========================================================================

ALL_EXPERIMENTS = {
    '1': ('Thermodynamic Collapse', fig_exp1),
    '2': ('Metabolic Sparsity', fig_exp2),
    '3': ('HPA Cognitive Rigidity', fig_exp3),
    '4': ('Epigenetic Consolidation', fig_exp4),
    '5': ('Stockholm Pressure', fig_exp5),
    '6': ('Glymphatic Timing', fig_exp6),
    '7': ('ADHD Flicker', fig_exp7),
    '8': ('Digital Dreaming', fig_exp8),
    '9': ('Autism Spectrum', fig_exp9),
    '10': ('D2 Occupancy', fig_exp10),
    'A': ('Stress Anhedonia', fig_exp_stress),
    'B': ('Drug Decision', fig_exp_drug),
    'C': ('Social Decay', fig_exp_social),
}


def main():
    if len(sys.argv) > 1:
        exp_ids = sys.argv[1:]
    else:
        exp_ids = list(ALL_EXPERIMENTS.keys())

    print("=" * 60)
    print("Simulacrum Experiment Figure Generator")
    print(f"Generating {len(exp_ids)} experiments: {exp_ids}")
    print(f"Output: {FIGURES_DIR}")
    print("=" * 60)

    for eid in exp_ids:
        if eid not in ALL_EXPERIMENTS:
            print(f"[SKIP] Unknown experiment: {eid}")
            continue
        name, func = ALL_EXPERIMENTS[eid]
        print(f"\n{'='*60}")
        print(f"Experiment {eid}: {name}")
        print(f"{'='*60}")
        try:
            func()
        except Exception as e:
            print(f"[ERROR] Experiment {eid} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Done! All figures saved to:", FIGURES_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
