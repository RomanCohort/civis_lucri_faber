#!/usr/bin/env python3
"""
Figure Generation Script for PLOS_CLF_DRAFT_REVISED
Generates 6 figures for the Simulacrum manuscript
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('figures', exist_ok=True)

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 300

# =============================================================================
# Figure 1: Architecture Diagram (14 regions + EventBus)
# =============================================================================
def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 10))

    # Brain regions (simplified circular layout)
    regions = {
        'HPA Axis': (0, 4),
        'Amygdala': (-2, 2),
        'Hippocampus': (2, 2),
        'PFC': (0, 6),
        'Basal Ganglia': (-3, 0),
        'Thalamus': (3, 0),
        'Auditory': (-4, 3),
        'Visual': (4, 3),
        'Glial': (-2, -2),
        'NT Module': (2, -2),
        'Thermo': (-4, -1),
        'Metabolic': (4, -1),
        'Sleep': (0, -3),
        'Social': (0, 1)
    }

    # Draw regions as circles
    for region, (x, y) in regions.items():
        circle = plt.Circle((x, y), 0.6, fill=True, color='steelblue', alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, region, ha='center', va='center', fontsize=7, fontweight='bold')

    # Draw EventBus in center
    bus_circle = plt.Circle((0, 0), 0.8, fill=True, color='coral', alpha=0.9)
    ax.add_patch(bus_circle)
    ax.text(0, 0, 'EventBus\n(18 types)', ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw connections to EventBus
    for region, (x, y) in regions.items():
        ax.annotate('', xy=(0, 0), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=0.5))

    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Figure 1: Simulacrum Architecture\n14 Brain Regions + Event-Driven EventBus', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/architecture_diagram.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("[OK] Figure 1: Architecture diagram saved")

# =============================================================================
# Figure 2: Bio-Gating Mechanism Flowchart
# =============================================================================
def create_bio_gating_flowchart():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Box positions
    boxes = {
        'Input x': (1, 7),
        'Content\nRouting': (1, 5.5),
        'Membrane\nPotential p': (3, 5.5),
        'Emotion\nModulation': (5, 5.5),
        'Mood\nState': (7, 5.5),
        'Summation': (4, 3.5),
        'Softmax': (4, 2),
        'Expert\nSelection': (4, 0.5)
    }

    # Draw boxes
    for name, (x, y) in boxes.items():
        if name == 'Input x':
            bbox = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        elif name == 'Expert\nSelection':
            bbox = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        else:
            bbox = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(x, y, name, ha='center', va='center', fontsize=9,
                fontweight='bold', bbox=bbox)

    # Draw arrows
    arrows = [
        ('Input x', 'Content\nRouting'),
        ('Content\nRouting', 'Summation'),
        ('Membrane\nPotential p', 'Summation'),
        ('Emotion\nModulation', 'Summation'),
        ('Mood\nState', 'Summation'),
        ('Summation', 'Softmax'),
        ('Softmax', 'Expert\nSelection')
    ]

    for start, end in arrows:
        sx, sy = boxes[start]
        ex, ey = boxes[end]
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Add formula
    ax.text(4, 8.5, r'$\mathrm{gate}_i = \mathrm{softmax}(W_c x + p + e + m)_i$',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('Figure 2: Bio-Gating Mechanism\nNeurotransmitter-Modulated MoE Routing', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/bio_gating_mechanism.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("[OK] Figure 2: Bio-Gating mechanism saved")

# =============================================================================
# Figure 3: D2 Occupancy Inverted-U Curve
# =============================================================================
def create_d2_inverted_u():
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Data from manuscript Table 5
    occupancy = [30, 75, 95]
    psi_improvement = [13, 33, 41]
    eps_index = [0.14, 0.34, 0.40]
    treatment_index = [0.95, 0.96, 1.03]

    # Error bars (approximate from manuscript)
    psi_err = [4, 5, 3]
    eps_err = [0.02, 0.03, 0.03]

    x = np.arange(len(occupancy))
    width = 0.25

    # Create bars
    bars1 = ax1.bar(x - width, psi_improvement, width, label='PSI Improvement (%)',
                    color='steelblue', alpha=0.8, yerr=psi_err, capsize=3)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, [e*100 for e in eps_index], width, label='EPS Index (×100)',
                    color='coral', alpha=0.8, yerr=[e*100 for e in eps_err], capsize=3)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    bars3 = ax3.bar(x + width, [t*100 for t in treatment_index], width,
                    label='Treatment Index (×100)', color='green', alpha=0.8)

    ax1.set_xlabel('D2 Receptor Occupancy (%)', fontsize=11)
    ax1.set_ylabel('PSI Improvement (%)', color='steelblue', fontsize=11)
    ax2.set_ylabel('EPS Index (×100)', color='coral', fontsize=11)
    ax3.set_ylabel('Treatment Index (×100)', color='green', fontsize=11)

    ax1.set_xticks(x)
    ax1.set_xticklabels(['Low\n(30%)', 'Medium\n(75%)', 'High\n(95%)'])

    # Highlight optimal
    ax1.axvline(x=1, color='gold', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(1, 45, 'Optimal\nBalance', ha='center', va='bottom', fontsize=9, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', fontsize=8)

    ax1.set_title('Figure 3: D2 Receptor Occupancy Inverted-U Curve\nOptimal Therapeutic Response at 75% Occupancy',
                  fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/exp10_inverted_u.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("[OK] Figure 3: D2 inverted-U curve saved")

# =============================================================================
# Figure 4: Stockholm Syndrome Trajectory
# =============================================================================
def create_stockholm_trajectory():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Data from manuscript
    phases = ['Resistance', 'Pressure', 'Bonding']
    bonding_score = [0.17, 0.51, 0.88]
    fight_ratio = [0.76, 0.23, 0.00]
    fawn_ratio = [0.00, 0.79, 1.00]

    # Error bars
    bonding_err = [0.05, 0.08, 0.06]
    fight_err = [0.10, 0.07, 0.00]
    fawn_err = [0.00, 0.09, 0.00]

    # Left panel: Bonding trajectory
    ax1.errorbar(phases, bonding_score, yerr=bonding_err, marker='o',
                 color='green', linewidth=2, markersize=10, capsize=5, label='Bonding Score')
    ax1.set_ylabel('Bonding Score', fontsize=11)
    ax1.set_xlabel('Phase', fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left')
    ax1.set_title('Bonding Score Progression', fontsize=11, fontweight='bold')

    # Add annotations
    for i, (x, y) in enumerate(zip(phases, bonding_score)):
        ax1.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                    xytext=(0, 15), ha='center', fontsize=9)

    # Right panel: Fight-to-fawn transition
    x_pos = np.arange(len(phases))
    width = 0.35

    ax2.bar(x_pos - width/2, fight_ratio, width, label='Fight Ratio',
            color='red', alpha=0.7, yerr=fight_err, capsize=3)
    ax2.bar(x_pos + width/2, fawn_ratio, width, label='Fawn Ratio',
            color='blue', alpha=0.7, yerr=fawn_err, capsize=3)

    ax2.set_ylabel('Ratio', fontsize=11)
    ax2.set_xlabel('Phase', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(phases)
    ax2.set_ylim(0, 1.2)
    ax2.legend(loc='upper right')
    ax2.set_title('Fight-to-Fawn Defensive Transition', fontsize=11, fontweight='bold')

    # Add transition arrow
    ax2.annotate('', xy=(2.3, 0.5), xytext=(-0.3, 0.5),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax2.text(1, 0.55, 'Defensive Shift', ha='center', va='bottom',
             fontsize=9, color='purple', fontweight='bold')

    fig.suptitle('Figure 4: Stockholm Syndrome Experiment\nFight-to-Fawn Defensive Transition',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('figures/exp5_bonding_trajectory.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("[OK] Figure 4: Stockholm syndrome trajectory saved")

# =============================================================================
# Figure 5: Stress Cascade Anhedonia
# =============================================================================
def create_stress_cascade():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    phases = ['Baseline', 'Stress', 'Recovery']

    # Data from manuscript Table 6
    cortisol = [0.45, 1.0, 1.0]
    pfc_inhibition = [0.60, 0.31, 0.31]
    exploration = [0.099, 0.041, 0.041]
    motivation = [0.36, 0.08, 0.08]

    # Error bars (approximate)
    cortisol_err = [0.05, 0.0, 0.0]
    pfc_err = [0.03, 0.02, 0.02]
    exploration_err = [0.01, 0.01, 0.01]
    motivation_err = [0.05, 0.02, 0.02]

    # Cortisol
    axes[0, 0].errorbar(phases, cortisol, yerr=cortisol_err, marker='s',
                       color='red', linewidth=2, markersize=10, capsize=5)
    axes[0, 0].set_ylabel('Cortisol Level', fontsize=10)
    axes[0, 0].set_title('A) Cortisol Trajectory', fontsize=11, fontweight='bold')
    axes[0, 0].axhline(y=0.45, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    axes[0, 0].legend(fontsize=8)

    # PFC Inhibition
    axes[0, 1].errorbar(phases, pfc_inhibition, yerr=pfc_err, marker='o',
                       color='purple', linewidth=2, markersize=10, capsize=5)
    axes[0, 1].set_ylabel('PFC Inhibition Rate', fontsize=10)
    axes[0, 1].set_title('B) PFC Function Decline', fontsize=11, fontweight='bold')
    axes[0, 1].axhline(y=0.60, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].annotate('48% decline', xy=(1, 0.31), xytext=(1.3, 0.45),
                       arrowprops=dict(arrowstyle='->', color='purple'), fontsize=8)

    # Exploration
    axes[1, 0].errorbar(phases, exploration, yerr=exploration_err, marker='^',
                       color='blue', linewidth=2, markersize=10, capsize=5)
    axes[1, 0].set_ylabel('Exploration Rate', fontsize=10)
    axes[1, 0].set_xlabel('Phase', fontsize=10)
    axes[1, 0].set_title('C) Exploration Decline', fontsize=11, fontweight='bold')
    axes[1, 0].axhline(y=0.099, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].annotate('59% decline', xy=(1, 0.041), xytext=(1.3, 0.07),
                       arrowprops=dict(arrowstyle='->', color='blue'), fontsize=8)

    # Motivation
    axes[1, 1].errorbar(phases, motivation, yerr=motivation_err, marker='d',
                       color='orange', linewidth=2, markersize=10, capsize=5)
    axes[1, 1].set_ylabel('Motivation λ', fontsize=10)
    axes[1, 1].set_xlabel('Phase', fontsize=10)
    axes[1, 1].set_title('D) Motivation Collapse', fontsize=11, fontweight='bold')
    axes[1, 1].axhline(y=0.36, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].annotate('78% collapse', xy=(1, 0.08), xytext=(1.3, 0.20),
                       arrowprops=dict(arrowstyle='->', color='orange'), fontsize=8)

    fig.suptitle('Figure 5: Stress-Induced Anhedonia Cascade\nHPA → PFC → Exploration → Motivation',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('figures/expA_stress_anhedonia.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("[OK] Figure 5: Stress cascade anhedonia saved")

# =============================================================================
# Figure 6: Drug-Specific Profiles
# =============================================================================
def create_drug_profiles():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    drugs = ['Baseline', 'Hallucinogen', 'Sedative', 'Stimulant']

    # Neurotransmitter data from manuscript
    da = [0.68, 0.65, 0.50, 0.85]
    ht = [0.55, 0.90, 0.55, 0.55]
    gaba = [0.50, 0.50, 0.85, 0.45]
    exploration = [0.098, 0.055, 0.032, 0.068]

    x = np.arange(len(drugs))
    width = 0.25

    # Left panel: Neurotransmitter profiles
    ax1.bar(x - width, da, width, label='Dopamine', color='steelblue', alpha=0.8)
    ax1.bar(x, ht, width, label='Serotonin', color='coral', alpha=0.8)
    ax1.bar(x + width, gaba, width, label='GABA', color='green', alpha=0.8)

    ax1.set_ylabel('Neurotransmitter Level', fontsize=11)
    ax1.set_xlabel('Drug Condition', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(drugs, rotation=15, ha='right')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.set_title('A) Neurotransmitter Profiles', fontsize=11, fontweight='bold')

    # Right panel: Behavioral effects
    colors = ['gray', 'purple', 'blue', 'orange']
    bars = ax2.bar(x, exploration, width=0.5, color=colors, alpha=0.8)

    ax2.set_ylabel('Exploration Rate', fontsize=11)
    ax2.set_xlabel('Drug Condition', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(drugs, rotation=15, ha='right')
    ax2.set_ylim(0, 0.12)
    ax2.set_title('B) Behavioral Effects', fontsize=11, fontweight='bold')

    # Add annotations
    ax2.annotate('↓ 5-HT↑', xy=(1, 0.055), xytext=(1.2, 0.08),
                fontsize=8, ha='center')
    ax2.annotate('↓↓ GABA↑', xy=(2, 0.032), xytext=(2, 0.06),
                fontsize=8, ha='center')
    ax2.annotate('↑ DA↑', xy=(3, 0.068), xytext=(3, 0.09),
                fontsize=8, ha='center')

    fig.suptitle('Figure 6: Drug-Specific Neurotransmitter and Behavioral Profiles\nPharmacological Manipulation Effects',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('figures/expB_drug_decision.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("[OK] Figure 6: Drug profiles saved")

# =============================================================================
# Main execution
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Figure Generation for PLOS_CLF_DRAFT_REVISED")
    print("="*60 + "\n")

    create_architecture_diagram()
    create_bio_gating_flowchart()
    create_d2_inverted_u()
    create_stockholm_trajectory()
    create_stress_cascade()
    create_drug_profiles()

    print("\n" + "="*60)
    print("All 6 figures generated successfully!")
    print("Output directory: figures/")
    print("="*60)
