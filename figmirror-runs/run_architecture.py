"""
Figure 1: Simulacrum Architecture Overview
Target: Neurocomputing Journal Style
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.lines as mlines
import numpy as np

# ==================== STYLE SETUP ====================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'text.usetex': False
})

# Color palette - Neurocomputing style
COLORS = {
    'sensory': '#4472C4',      # Blue
    'memory': '#70AD47',        # Green
    'decision': '#FFC000',      # Amber
    'emotional': '#C00000',     # Red
    'metabolic': '#7030A0',     # Purple
    'social': '#00B0F0',        # Cyan
    'modulation': '#ED7D31',    # Orange
    'eventbus': '#595959',      # Dark gray
    'coupling': '#A5A5A5'       # Light gray
}

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# ==================== MODULE DEFINITIONS ====================
# Format: (name, x, y, width, height, color, count)

modules = {
    'sensory': [
        ('Auditory\nCortex', 0.5, 8.5, 2.0, 1.2, COLORS['sensory']),
        ('Visual\nCortex', 2.7, 8.5, 2.0, 1.2, COLORS['sensory']),
        ('Thalamus', 4.9, 8.5, 2.0, 1.2, COLORS['sensory']),
    ],
    'memory': [
        ('Hippocampus', 7.3, 8.5, 2.0, 1.2, COLORS['memory']),
        ('Episodic\nMemory', 9.5, 8.5, 2.0, 1.2, COLORS['memory']),
    ],
    'decision': [
        ('Prefrontal\nCortex (PFC)', 0.5, 6.2, 2.4, 1.2, COLORS['decision']),
        ('Basal\nGanglia', 3.1, 6.2, 2.2, 1.2, COLORS['decision']),
        ('Action\nSelection', 5.5, 6.2, 2.2, 1.2, COLORS['decision']),
    ],
    'emotional': [
        ('Amygdala', 7.9, 6.2, 2.0, 1.2, COLORS['emotional']),
        ('HPA Axis', 10.1, 6.2, 1.6, 1.2, COLORS['emotional']),
    ],
    'metabolic': [
        ('Thermo-\ndynamics', 0.5, 3.9, 2.0, 1.2, COLORS['metabolic']),
        ('Sleep\nSystem', 2.7, 3.9, 2.0, 1.2, COLORS['metabolic']),
        ('Glial\nSystem', 4.9, 3.9, 2.0, 1.2, COLORS['metabolic']),
    ],
    'social': [
        ('Theory of\nMind', 7.3, 3.9, 2.0, 1.2, COLORS['social']),
        ('Empathy', 9.5, 3.9, 2.0, 1.2, COLORS['social']),
    ],
}

# Draw all modules
for category, mods in modules.items():
    for name, x, y, w, h, color in mods:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

# ==================== EVENTBUS (Central) ====================
eventbus_y = 5.3
eventbus_box = FancyBboxPatch((3, eventbus_y), 6, 0.7,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=COLORS['eventbus'], edgecolor='black',
                               linewidth=2, alpha=0.9)
ax.add_patch(eventbus_box)
ax.text(6, eventbus_y + 0.35, 'EventBus (Pub/Sub)  •  18 Event Types',
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# ==================== NEUROTRANSMITTER MODULE ====================
nt_y = 2.5
nt_box = FancyBboxPatch((0.5, nt_y), 11, 1.0,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=COLORS['modulation'], edgecolor='black',
                         linewidth=2, alpha=0.85)
ax.add_patch(nt_box)
ax.text(6, nt_y + 0.5, 'Neurotransmitter Module: DA | 5-HT | NE | ACh | GABA',
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# ==================== COUPLING PATHWAYS ====================
coupling_y = 1.3
coupling_box = FancyBboxPatch((0.5, coupling_y), 11, 0.9,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=COLORS['coupling'], edgecolor='black',
                               linewidth=1.5, alpha=0.7)
ax.add_patch(coupling_box)
ax.text(6, coupling_y + 0.45, 'Coupling Pathways (7): Cortisol→PFC | DA→Exploration | Oxytocin→Empathy | ACh→WM | GABA→Inhibition | 5-HT→Stability | NE→Arousal',
        ha='center', va='center', fontsize=8, fontweight='bold', color='#333333')

# ==================== CONNECTION ARROWS ====================
# Arrows from modules to EventBus
arrow_style = dict(arrowstyle='->', color='#666666', lw=1.5, mutation_scale=12)

# Sensory/Memory down to EventBus
for x in [1.5, 3.7, 5.9]:
    ax.annotate('', xy=(x, eventbus_y + 0.7), xytext=(x, 8.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['sensory'], lw=1.5))
for x in [8.3, 10.5]:
    ax.annotate('', xy=(x, eventbus_y + 0.7), xytext=(x, 8.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['memory'], lw=1.5))

# EventBus to Decision/Emotional
for x in [1.7, 4.2, 6.6]:
    ax.annotate('', xy=(x, 7.4), xytext=(x, eventbus_y),
                arrowprops=dict(arrowstyle='->', color=COLORS['decision'], lw=1.5))
for x in [8.9, 10.9]:
    ax.annotate('', xy=(x, 7.4), xytext=(x, eventbus_y),
                arrowprops=dict(arrowstyle='->', color=COLORS['emotional'], lw=1.5))

# EventBus to Metabolic/Social
for x in [1.5, 3.7, 5.9]:
    ax.annotate('', xy=(x, 5.1), xytext=(x, eventbus_y),
                arrowprops=dict(arrowstyle='->', color=COLORS['metabolic'], lw=1.5))
for x in [8.3, 10.5]:
    ax.annotate('', xy=(x, 5.1), xytext=(x, eventbus_y),
                arrowprops=dict(arrowstyle='->', color=COLORS['social'], lw=1.5))

# Neurotransmitter to modules (vertical dashed lines)
for x in [2, 4, 6, 8, 10]:
    ax.plot([x, x], [nt_y + 1.0, 3.9], 'k--', lw=0.8, alpha=0.4)

# ==================== LEGEND ====================
legend_y = 0.3
legend_items = [
    ('Sensory (3)', COLORS['sensory']),
    ('Memory (3)', COLORS['memory']),
    ('Decision (3)', COLORS['decision']),
    ('Emotional (2)', COLORS['emotional']),
    ('Metabolic (3)', COLORS['metabolic']),
    ('Social (2)', COLORS['social']),
]

for i, (label, color) in enumerate(legend_items):
    x = 0.5 + i * 1.9
    rect = Rectangle((x, legend_y), 0.3, 0.3, facecolor=color, edgecolor='black', lw=1)
    ax.add_patch(rect)
    ax.text(x + 0.4, legend_y + 0.15, label, fontsize=7, va='center')

# ==================== TITLE ====================
ax.text(6, 9.8, 'Simulacrum Architecture', ha='center', va='center',
        fontsize=14, fontweight='bold')
ax.text(6, 9.5, 'Neuro-Modulated Cognitive Architecture with Event-Driven Sparse Activation',
        ha='center', va='center', fontsize=10, style='italic', color='#555555')

# ==================== SAVE ====================
plt.tight_layout()
plt.savefig('D:/civis_lucri_faber/figmirror-runs/fig1_architecture.png', bbox_inches='tight', dpi=300)
plt.savefig('D:/civis_lucri_faber/figmirror-runs/fig1_architecture.pdf', bbox_inches='tight', dpi=300)
print("Figure 1 saved: Architecture Overview")
plt.close()
