"""
Figure: Seven Coupling Pathways Network Diagram
Target: Neurocomputing Journal Style
Data from NC_DRAFT.md Table 8, Line 655-664
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# ==================== STYLE SETUP ====================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
})

# ==================== PATHWAY DEFINITIONS ====================
# From NC_DRAFT.md Table 8: Seven Coupling Pathways

pathways = [
    {'id': 'P1', 'name': 'Cortisol → PFC', 'source': 'Cortisol', 'target': 'PFC',
     'mechanism': 'Stress inhibits\nexecutive function', 'ref': 'Arnsten 2009',
     'effect': 'PFC.inhibition *= (1 - cortisol × α)', 'color': '#d62728'},
    {'id': 'P2', 'name': 'DA → Exploration', 'source': 'DA', 'target': 'Exploration',
     'mechanism': 'Dopamine drives\nexploration-exploitation', 'ref': 'Schultz 2007',
     'effect': 'exploration_rate = f(DA, baseline)', 'color': '#2ca02c'},
    {'id': 'P3', 'name': 'Oxytocin → Empathy', 'source': 'Oxytocin', 'target': 'Empathy',
     'mechanism': 'Social bonding\nhormone', 'ref': 'Dunbar 2009',
     'effect': 'resonance *= (1 + oxytocin × γ)', 'color': '#1f77b4'},
    {'id': 'P4', 'name': 'ACh → WM Gate', 'source': 'ACh', 'target': 'WM',
     'mechanism': 'Attention modulates\nworking memory', 'ref': 'Hasselmo 1999',
     'effect': 'wm_update_rate = gate(ACh)', 'color': '#ff7f0e'},
    {'id': 'P5', 'name': 'GABA → Inhibition', 'source': 'GABA', 'target': 'Inhibition',
     'mechanism': 'E/I balance\ncontrol', 'ref': 'GABA theory',
     'effect': 'activity = clamp(x, GABA_threshold)', 'color': '#9467bd'},
    {'id': 'P6', 'name': '5-HT → Stability', 'source': '5-HT', 'target': 'Stability',
     'mechanism': 'Serotonin promotes\nbehavioral consistency', 'ref': 'Dayan 2009',
     'effect': 'decision_variance *= (1 - 5-HT × δ)', 'color': '#8c564b'},
    {'id': 'P7', 'name': 'NE → Arousal', 'source': 'NE', 'target': 'Arousal',
     'mechanism': 'Norepinephrine\nboosts salience', 'ref': 'Aston-Jones 2005',
     'effect': 'attention_gain = SNR_boost(NE)', 'color': '#e377c2'},
]

# ==================== FIGURE SETUP ====================
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# ==================== LEFT PANEL: Network Diagram ====================
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.axis('off')

# Neurotransmitter nodes (sources)
sources_y = 10
sources_x = [1, 2.5, 4, 5.5, 7, 8.5]
source_nodes = [
    ('Cortisol', 1, sources_y, '#d62728'),
    ('DA', 2.5, sources_y, '#2ca02c'),
    ('Oxytocin', 4, sources_y, '#1f77b4'),
    ('ACh', 5.5, sources_y, '#ff7f0e'),
    ('GABA', 7, sources_y, '#9467bd'),
    ('5-HT', 8.5, sources_y, '#8c564b'),
]

# Add NE node separately
ne_x, ne_y = 8.5, 8

for name, x, y, color in source_nodes:
    circle = Circle((x, y), 0.6, facecolor=color, edgecolor='black', lw=2, alpha=0.8)
    ax1.add_patch(circle)
    ax1.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Add NE node
circle_ne = Circle((ne_x, ne_y), 0.6, facecolor='#e377c2', edgecolor='black', lw=2, alpha=0.8)
ax1.add_patch(circle_ne)
ax1.text(ne_x, ne_y, 'NE', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Target nodes
targets_y = 3
target_nodes = [
    ('PFC', 1.5, targets_y, '#4472C4'),
    ('Exploration', 3, targets_y, '#70AD47'),
    ('Empathy', 4.5, targets_y, '#00B0F0'),
    ('WM Gate', 6, targets_y, '#FFC000'),
    ('Inhibition', 7.5, targets_y, '#7030A0'),
    ('Stability', 9, targets_y, '#C00000'),
]

# Add Arousal node
arousal_x, arousal_y = 8, 5

for name, x, y, color in target_nodes:
    rect = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8,
                          boxstyle="round,pad=0.05,rounding_size=0.15",
                          facecolor=color, edgecolor='black', lw=1.5, alpha=0.85)
    ax1.add_patch(rect)
    ax1.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Add Arousal node
rect_arousal = FancyBboxPatch((arousal_x-0.8, arousal_y-0.4), 1.6, 0.8,
                               boxstyle="round,pad=0.05,rounding_size=0.15",
                               facecolor='#e377c2', edgecolor='black', lw=1.5, alpha=0.85)
ax1.add_patch(rect_arousal)
ax1.text(arousal_x, arousal_y, 'Arousal', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# Draw pathway arrows
arrow_coords = [
    (1, sources_y, 1.5, targets_y + 0.4, '#d62728', 'P1'),      # Cortisol → PFC
    (2.5, sources_y, 3, targets_y + 0.4, '#2ca02c', 'P2'),      # DA → Exploration
    (4, sources_y, 4.5, targets_y + 0.4, '#1f77b4', 'P3'),      # Oxytocin → Empathy
    (5.5, sources_y, 6, targets_y + 0.4, '#ff7f0e', 'P4'),      # ACh → WM
    (7, sources_y, 7.5, targets_y + 0.4, '#9467bd', 'P5'),      # GABA → Inhibition
    (8.5, sources_y, 9, targets_y + 0.4, '#8c564b', 'P6'),      # 5-HT → Stability
]

for sx, sy, tx, ty, color, pid in arrow_coords:
    ax1.annotate('', xy=(tx, ty), xytext=(sx, sy),
                 arrowprops=dict(arrowstyle='->', color=color, lw=3,
                                connectionstyle='arc3,rad=0.1'))
    ax1.text((sx+tx)/2 + 0.3, (sy+ty)/2, pid, fontsize=10, fontweight='bold',
             color=color, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.9))

# P7 arrow (NE → Arousal)
ax1.annotate('', xy=(arousal_x, arousal_y + 0.4), xytext=(ne_x, ne_y),
             arrowprops=dict(arrowstyle='->', color='#e377c2', lw=3))
ax1.text((ne_x + arousal_x)/2 + 0.4, (ne_y + arousal_y)/2, 'P7', fontsize=10,
         fontweight='bold', color='#e377c2', ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='#e377c2', alpha=0.9))

# Title for network panel
ax1.text(5, 11.5, 'A: Coupling Pathways Network', ha='center', fontsize=12, fontweight='bold')

# Legend for node types
ax1.text(0.5, 0.5, 'Source nodes: Neurotransmitters', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
ax1.text(5, 0.5, 'Target nodes: Cognitive functions', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# ==================== RIGHT PANEL: Pathway Table ====================
ax2 = axes[1]
ax2.axis('off')

# Create pathway table
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)

ax2.text(5, 11.5, 'B: Pathway Mechanisms and Effects', ha='center', fontsize=12, fontweight='bold')

# Table header
headers = ['ID', 'Pathway', 'Mechanism', 'Reference', 'Computational Effect']
header_y = 10.5
for i, h in enumerate(headers):
    x = 0.5 + i * 2
    ax2.text(x, header_y, h, ha='left', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', alpha=0.9))

# Table rows
row_y_start = 9.5
for j, p in enumerate(pathways):
    row_y = row_y_start - j * 1.3
    row_data = [p['id'], p['name'], p['mechanism'], p['ref'], p['effect']]
    for i, data in enumerate(row_data):
        x = 0.5 + i * 2
        bg_color = p['color'] if i == 0 else 'white'
        ax2.text(x, row_y, data, ha='left', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor='gray', alpha=0.7))

# Parameter ranges section
param_y = 1.5
ax2.text(5, param_y + 0.5, 'Parameter Ranges (with Uncertainty)', ha='center', fontsize=10, fontweight='bold')
param_text = '''P1 (α): 0.4 [0.2, 0.6]  — Stress-PFC coupling
P2 (η): 0.2 [0.1, 0.4]  — DA-exploration sensitivity
P3 (γ): 0.5 [0.2, 0.8]  — Oxytocin-empathy modulation'''
ax2.text(0.5, param_y - 1, param_text, fontsize=8,
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

# ==================== COUPLING MATRIX VISUALIZATION ====================
# Add small coupling matrix heatmap
ax3 = fig.add_axes([0.55, 0.05, 0.35, 0.25])
ax3.set_title('Coupling Matrix A (lower triangular)', fontsize=9, fontweight='bold')

# Simplified coupling matrix
A = np.array([
    [-0.9, 0, 0, 0, 0, 0, 0],
    [-0.4, -0.9, 0, 0, 0, 0, 0],  # P1 cascade
    [0, 0, -0.9, 0, 0, 0, 0],
    [0, 0, 0, -0.9, 0, 0, 0],
    [0, 0, 0, 0, -0.9, 0, 0],
    [0, 0, 0, 0, 0, -0.9, 0],
    [0, 0, 0, 0, 0, 0, -0.9],
])

import seaborn as sns
sns.heatmap(A, ax=ax3, cmap='coolwarm', center=0,
            xticklabels=['C', 'DA', 'O', 'ACh', 'G', '5HT', 'NE'],
            yticklabels=['C', 'DA', 'O', 'ACh', 'G', '5HT', 'NE'],
            annot=True, fmt='.1f', cbar_kws={'shrink': 0.5})
ax3.set_xlabel('State variables', fontsize=8)
ax3.set_ylabel('State variables', fontsize=8)

# ==================== SAVE ====================
plt.tight_layout(rect=[0, 0.35, 1, 0.98])
plt.savefig('D:/civis_lucri_faber/figmirror-runs/fig_coupling_pathways.png', bbox_inches='tight', dpi=300)
plt.savefig('D:/civis_lucri_faber/figmirror-runs/fig_coupling_pathways.pdf', bbox_inches='tight', dpi=300)
print("Figure saved: Coupling Pathways Network Diagram")
plt.close()