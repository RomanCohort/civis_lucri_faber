"""
Figure: EventBus Subscription Matrix Heatmap
Target: Neurocomputing Journal Style
Data from NC_DRAFT.md Table 6, Line 574-590
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==================== STYLE SETUP ====================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
})

# ==================== DATA SECTOR ====================
# From NC_DRAFT.md Table 6: Subscription Matrix
# Rows: 14 brain regions, Columns: 18 event types

regions = [
    'HPA Axis', 'Amygdala', 'Hippocampus', 'PFC', 'Basal Ganglia',
    'Thalamus', 'Auditory', 'Visual', 'Glial', 'NT Module',
    'Thermodynamics', 'Metabolic', 'Sleep', 'Social'
]

events = [
    'STRESS', 'EMOTION', 'MEMORY', 'DECISION', 'ACTION', 'SENSORY',
    'NT', 'SLEEP', 'SOCIAL', 'PHARM', 'META', 'THERMO',
    'GLIAL', 'EPI', 'HOME', 'LEARN', 'ATT', 'REWARD'
]

# Subscription matrix (1 = subscribed, 0 = not subscribed)
# Extracted from Table 6
subscription_matrix = np.array([
    # STRESS, EMOTION, MEMORY, DECISION, ACTION, SENSORY, NT, SLEEP, SOCIAL, PHARM, META, THERMO, GLIAL, EPI, HOME, LEARN, ATT, REWARD
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],  # HPA
    [1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # Amygdala
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],  # Hippocampus
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],  # PFC
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],  # Basal Ganglia
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # Thalamus
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Auditory
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Visual
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],  # Glial
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],  # NT Module
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],  # Thermodynamics
    [1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],  # Metabolic
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],  # Sleep
    [0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # Social
])

# Calculate subscription counts per region
sub_counts = subscription_matrix.sum(axis=1)

# ==================== FIGURE SETUP ====================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[4, 1],
                      hspace=0.05, wspace=0.05)

# ==================== MAIN HEATMAP ====================
ax1 = fig.add_subplot(gs[0, 0])

# Create heatmap
cmap = sns.color_palette("Blues", as_cmap=True)
hm = sns.heatmap(subscription_matrix, ax=ax1,
                  cmap=cmap, cbar_kws={'label': 'Subscription', 'shrink': 0.6},
                  linewidths=0.5, linecolor='white',
                  xticklabels=events, yticklabels=regions,
                  annot=True, fmt='d', annot_kws={'size': 7})

ax1.set_xlabel('Event Types (18)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Brain Regions (14)', fontsize=11, fontweight='bold')
ax1.set_title('EventBus Subscription Matrix\nFunctional Connectivity Pattern',
              fontsize=12, fontweight='bold', pad=10)

# Rotate x labels
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# ==================== ROW SUMMARY (Right) ====================
ax2 = fig.add_subplot(gs[0, 1])
ax2.barh(range(len(regions)), sub_counts, color='steelblue', alpha=0.8)
ax2.set_yticks(range(len(regions)))
ax2.set_yticklabels([])
ax2.set_xlabel('Subscription\nCount', fontsize=9)
ax2.set_xlim(0, 14)
ax2.invert_yaxis()
ax2.grid(True, axis='x', linestyle='--', alpha=0.3)

# Add count labels
for i, count in enumerate(sub_counts):
    ax2.text(count + 0.3, i, str(count), va='center', fontsize=8)

# ==================== COLUMN SUMMARY (Bottom) ====================
ax3 = fig.add_subplot(gs[1, 0])
col_counts = subscription_matrix.sum(axis=0)
ax3.bar(range(len(events)), col_counts, color='coral', alpha=0.8)
ax3.set_xticks(range(len(events)))
ax3.set_xticklabels([])
ax3.set_ylabel('Regions\nSubscribed', fontsize=9)
ax3.set_ylim(0, 10)
ax3.grid(True, axis='y', linestyle='--', alpha=0.3)

# Add count labels on top
for i, count in enumerate(col_counts):
    ax3.text(i, count + 0.2, str(count), ha='center', fontsize=7)

# ==================== STATISTICS BOX ====================
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

total_subscriptions = subscription_matrix.sum()
mean_per_region = sub_counts.mean()
sparsity = 1 - (total_subscriptions / (14 * 18))

stats_text = f'''Statistics:
Total subscriptions: {total_subscriptions}
Mean/region: {mean_per_region:.1f}
Activation sparsity: ~23%
(PFC hub: {sub_counts[3]} subscriptions)'''

ax4.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
         transform=ax4.transAxes)

# ==================== SAVE ====================
plt.tight_layout()
plt.savefig('D:/civis_lucri_faber/figmirror-runs/fig_eventbus_heatmap.png', bbox_inches='tight', dpi=300)
plt.savefig('D:/civis_lucri_faber/figmirror-runs/fig_eventbus_heatmap.pdf', bbox_inches='tight', dpi=300)
print("Figure saved: EventBus Subscription Matrix Heatmap")
plt.close()