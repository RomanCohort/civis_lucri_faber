"""
Figure: Stress-Induced Anhedonia Recovery Trajectory
Target: Neurocomputing Journal Style
Data from NC_DRAFT.md Line 1112-1117
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch

# ==================== STYLE SETUP ====================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3
})

# ==================== DATA SECTOR ====================
# From NC_DRAFT.md Table: Stress Anhedonia Recovery
# Metrics at different time points

time_points = ['Baseline', 'Stress (t=T)', 'Recovery\n(t=T+7d)', 'Recovery\n(t=T+30d)']
time_numeric = [0, 1, 2, 3]

# Cortisol levels
cortisol_mean = np.array([0.45, 1.0, 0.48, 0.45])
cortisol_std = np.array([0.05, 0.0, 0.06, 0.05])

# PFC inhibition
pfc_mean = np.array([0.70, 0.42, 0.51, 0.65])
pfc_std = np.array([0.03, 0.04, 0.05, 0.04])

# Exploration rate
exploration_mean = np.array([0.099, 0.051, 0.068, 0.091])
exploration_std = np.array([0.01, 0.02, 0.02, 0.02])

# Mesolimbic DA
da_mean = np.array([0.80, 0.45, 0.55, 0.72])
da_std = np.array([0.05, 0.08, 0.07, 0.06])

# For smooth curves in recovery phase
recovery_time = np.linspace(1, 3, 50)

# ==================== FIGURE SETUP ====================
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

fig.suptitle('Stress-Induced Anhedonia: Recovery Trajectory\n(Experiment A)',
             fontsize=14, fontweight='bold', y=0.98)

# Color palette
COLORS = {
    'cortisol': '#d62728',    # Red
    'pfc': '#ff7f0e',         # Orange
    'exploration': '#1f77b4', # Blue
    'da': '#9467bd'           # Purple
}

# ==================== PANEL A: All Metrics Overview ====================
ax1 = fig.add_subplot(gs[0, :2])

# Plot all four metrics
ax1.errorbar(time_numeric, cortisol_mean, yerr=cortisol_std, fmt='o-',
             color=COLORS['cortisol'], capsize=5, markersize=8, lw=2,
             label='Cortisol')
ax1.errorbar(time_numeric, pfc_mean, yerr=pfc_std, fmt='s-',
             color=COLORS['pfc'], capsize=5, markersize=8, lw=2,
             label='PFC inhibition')
ax1.errorbar(time_numeric, exploration_mean * 10, yerr=exploration_std * 10, fmt='D-',
             color=COLORS['exploration'], capsize=5, markersize=8, lw=2,
             label='Exploration rate (×10)')
ax1.errorbar(time_numeric, da_mean, yerr=da_std, fmt='^-',
             color=COLORS['da'], capsize=5, markersize=8, lw=2,
             label='Mesolimbic DA')

# Phase shading
ax1.axvspan(0, 1, alpha=0.15, color='lightblue', label='Stress period')
ax1.axvspan(1, 3, alpha=0.15, color='lightgreen', label='Recovery period')

# Phase boundaries
ax1.axvline(x=1, color='gray', linestyle='--', lw=2)
ax1.axvline(x=2, color='gray', linestyle=':', lw=1.5)

ax1.set_xlabel('Time Point', fontsize=11)
ax1.set_ylabel('Normalized Value', fontsize=11)
ax1.set_title('A: Complete Recovery Trajectory', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', framealpha=0.9, ncol=2)
ax1.set_xticks(time_numeric)
ax1.set_xticklabels(time_points, fontsize=9)
ax1.set_xlim(-0.2, 3.2)
ax1.set_ylim(0, 1.2)
ax1.grid(True, linestyle='--', alpha=0.3)

# Annotations
ax1.annotate('Stress onset', xy=(1, 1.0), xytext=(0.5, 1.15),
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)
ax1.annotate('Partial recovery', xy=(2, 0.51), xytext=(1.5, 0.75),
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)
ax1.annotate('Near-baseline', xy=(3, 0.65), xytext=(2.5, 0.85),
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)

# ==================== PANEL B: Individual Recovery Curves ====================
axes_right = [fig.add_subplot(gs[0, 2])]

# Since we need 4 panels in the right, restructure
# Let's create individual panels for each metric

# PANEL B: Cortisol
ax_cort = fig.add_subplot(2, 3, 3)
ax_cort.plot(time_numeric, cortisol_mean, 'o-', color=COLORS['cortisol'],
             lw=2.5, markersize=10)
ax_cort.fill_between(time_numeric, cortisol_mean - cortisol_std,
                     cortisol_mean + cortisol_std, color=COLORS['cortisol'], alpha=0.2)
ax_cort.axvspan(0, 1, alpha=0.15, color='lightblue')
ax_cort.axvspan(1, 3, alpha=0.15, color='lightgreen')
ax_cort.axhline(y=0.45, color='gray', linestyle=':', lw=1.5, label='Baseline')
ax_cort.set_title('B: Cortisol Recovery', fontsize=11, fontweight='bold')
ax_cort.set_ylabel('Cortisol Level', fontsize=10)
ax_cort.set_xticks(time_numeric)
ax_cort.set_xticklabels(['B', 'S', 'R7', 'R30'], fontsize=8)
ax_cort.set_ylim(0, 1.1)
ax_cort.grid(True, linestyle='--', alpha=0.3)

# PANEL C: PFC
ax_pfc = fig.add_subplot(2, 3, 4)
ax_pfc.plot(time_numeric, pfc_mean, 's-', color=COLORS['pfc'],
            lw=2.5, markersize=10)
ax_pfc.fill_between(time_numeric, pfc_mean - pfc_std,
                    pfc_mean + pfc_std, color=COLORS['pfc'], alpha=0.2)
ax_pfc.axvspan(0, 1, alpha=0.15, color='lightblue')
ax_pfc.axvspan(1, 3, alpha=0.15, color='lightgreen')
ax_pfc.axhline(y=0.70, color='gray', linestyle=':', lw=1.5, label='Baseline')
ax_pfc.set_title('C: PFC Inhibition Recovery', fontsize=11, fontweight='bold')
ax_pfc.set_ylabel('PFC Inhibition Rate', fontsize=10)
ax_pfc.set_xticks(time_numeric)
ax_pfc.set_xticklabels(['B', 'S', 'R7', 'R30'], fontsize=8)
ax_pfc.set_ylim(0, 0.85)
ax_pfc.grid(True, linestyle='--', alpha=0.3)

# PANEL D: Exploration
ax_exp = fig.add_subplot(2, 3, 5)
ax_exp.plot(time_numeric, exploration_mean, 'D-', color=COLORS['exploration'],
            lw=2.5, markersize=10)
ax_exp.fill_between(time_numeric, exploration_mean - exploration_std,
                    exploration_mean + exploration_std, color=COLORS['exploration'], alpha=0.2)
ax_exp.axvspan(0, 1, alpha=0.15, color='lightblue')
ax_exp.axvspan(1, 3, alpha=0.15, color='lightgreen')
ax_exp.axhline(y=0.099, color='gray', linestyle=':', lw=1.5, label='Baseline')
ax_exp.set_title('D: Exploration Rate Recovery', fontsize=11, fontweight='bold')
ax_exp.set_ylabel('Exploration Rate', fontsize=10)
ax_exp.set_xticks(time_numeric)
ax_exp.set_xticklabels(['B', 'S', 'R7', 'R30'], fontsize=8)
ax_exp.set_ylim(0, 0.12)
ax_exp.grid(True, linestyle='--', alpha=0.3)

# PANEL E: Mesolimbic DA
ax_da = fig.add_subplot(2, 3, 6)
ax_da.plot(time_numeric, da_mean, '^-', color=COLORS['da'],
           lw=2.5, markersize=10)
ax_da.fill_between(time_numeric, da_mean - da_std,
                   da_mean + da_std, color=COLORS['da'], alpha=0.2)
ax_da.axvspan(0, 1, alpha=0.15, color='lightblue')
ax_da.axvspan(1, 3, alpha=0.15, color='lightgreen')
ax_da.axhline(y=0.80, color='gray', linestyle=':', lw=1.5, label='Baseline')
ax_da.set_title('E: Mesolimbic DA Recovery', fontsize=11, fontweight='bold')
ax_da.set_ylabel('Mesolimbic DA Level', fontsize=10)
ax_da.set_xticks(time_numeric)
ax_da.set_xticklabels(['Baseline', 'Stress', '7d', '30d'], fontsize=8)
ax_da.set_ylim(0, 0.9)
ax_da.grid(True, linestyle='--', alpha=0.3)

# Add percentage change annotations
for ax, base_val, metric in [(ax_pfc, 0.70, 'PFC'), (ax_da, 0.80, 'DA')]:
    stress_drop = base_val - 0.42 if metric == 'PFC' else base_val - 0.45
    recovery_7 = 0.51 - 0.42 if metric == 'PFC' else 0.55 - 0.45
    recovery_30 = base_val - 0.65 if metric == 'PFC' else base_val - 0.72
    ax.text(1, 0.42 - 0.05, f'-{stress_drop/base_val*100:.0f}%', fontsize=8, color='red', ha='center')
    ax.text(2, 0.51 + 0.05 if metric == 'PFC' else 0.55 + 0.05,
            f'+{recovery_7/stress_drop*100:.0f}%', fontsize=8, color='green', ha='center')

# ==================== STATISTICS SUMMARY ====================
stats_text = '''Statistics: PFC decline t=9.15, p<0.001, d=2.44
Recovery trajectories:
  7-day: PFC +9%, Exploration +17%
  30-day: PFC -7%, Exploration -8% (near-baseline)

Key finding: Stress-induced anhedonia persists
weeks-months but typically resolves with stressor removal'''

fig.text(0.02, 0.02, stats_text, fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
         verticalalignment='bottom')

# ==================== SAVE ====================
plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig('D:/civis_lucri_faber/figmirror-runs/fig_stress_recovery.png', bbox_inches='tight', dpi=300)
plt.savefig('D:/civis_lucri_faber/figmirror-runs/fig_stress_recovery.pdf', bbox_inches='tight', dpi=300)
print("Figure saved: Stress-Induced Anhedonia Recovery Trajectory")
plt.close()