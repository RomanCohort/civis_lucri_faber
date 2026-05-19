"""
Neural Model Improvements Figure Generator
==========================================

Generates matplotlib figures for all 12 neural model improvements.
Reads experiment results and produces publication-quality plots.

Output: docs/figures/neural_improvements/
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from experiment_neural_improvements import run_all_experiments

# Output directory
FIG_DIR = os.path.join(os.path.dirname(__file__), 'docs', 'figures', 'neural_improvements')
os.makedirs(FIG_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
})


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════
# Figure 1: STDP Plasticity
# ══════════════════════════════════════════════════════

def fig1_stdp(results):
    print("\n[Figure 1] STDP Plasticity")
    tw = results['timing_window']
    dts = np.array(tw['dts'])
    ltp = np.array(tw['ltp_values'])
    ltd = np.array(tw['ltd_values'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1a: STDP timing window
    ax = axes[0]
    ax.plot(dts, ltp, 'b-', linewidth=2, label='LTP (Pre→Post)')
    ax.plot(-dts, ltd, 'r-', linewidth=2, label='LTD (Post→Pre)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Δt (ms)')
    ax.set_ylabel('Δw')
    ax.set_title('(a) STDP Timing Window\n(Bi & Poo, 1998)')
    ax.legend()
    ax.set_xlim(-100, 100)

    # 1b: LTP/LTD magnitude vs timing
    ax = axes[1]
    ltp_cases = results['ltp_cases']
    ltd_cases = results['ltd_cases']
    ltp_dts = [c['dt'] for c in ltp_cases]
    ltp_dws = [c['delta_w'] for c in ltp_cases]
    ltd_dts = [c['dt'] for c in ltd_cases]
    ltd_dws = [c['delta_w'] for c in ltd_cases]
    ax.bar(np.array(ltp_dts) - 2, ltp_dws, width=4, color='steelblue', label='LTP', alpha=0.8)
    ax.bar(np.array(ltd_dts) + 2, ltd_dws, width=4, color='salmon', label='LTD', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Δt (ms)')
    ax.set_ylabel('Δw')
    ax.set_title('(b) Weight Change by Timing')
    ax.legend()

    # 1c: Learning trajectory
    ax = axes[2]
    lr = results['learning_result']
    ax.bar(['Initial', 'Final'], [lr['initial_weight'], lr['final_weight']],
           color=['gray', 'steelblue'], alpha=0.8)
    ax.set_ylabel('Weight')
    ax.set_title(f'(c) STDP Learning\nLTP={lr["ltp_count"]}, LTD={lr["ltd_count"]}')

    fig.suptitle('Experiment 1: STDP Synaptic Plasticity', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig1_stdp_plasticity.png')


# ══════════════════════════════════════════════════════
# Figure 2: Dopamine Tonic-Phasic
# ══════════════════════════════════════════════════════

def fig2_dopamine(results):
    print("\n[Figure 2] Dopamine Tonic-Phasic")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 2a: RPE signaling
    ax = axes[0]
    rpe_tests = results['rpe_tests']
    descs = [r['desc'] for r in rpe_tests]
    rpes = [r['rpe'] for r in rpe_tests]
    phasics = [r['phasic'] for r in rpe_tests]
    tonics = [r['tonic'] for r in rpe_tests]
    x = np.arange(len(descs))
    w = 0.3
    ax.bar(x - w, rpes, w, label='RPE', color='gray', alpha=0.7)
    ax.bar(x, phasics, w, label='Phasic', color='salmon', alpha=0.8)
    ax.bar(x + w, tonics, w, label='Tonic', color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(descs, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Level')
    ax.set_title('(a) RPE Signaling\n(Schultz, 1997)')
    ax.legend()

    # 2b: Phasic decay
    ax = axes[1]
    phasic_decay = results['phasic_decay']
    ax.plot(phasic_decay, 'r-', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Phasic DA')
    ax.set_title('(b) Phasic DA Decay')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # 2c: Tonic adaptation
    ax = axes[2]
    tonic_adapt = results['tonic_adaptation']
    ax.plot(tonic_adapt, 'b-', linewidth=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Tonic DA')
    ax.set_title('(c) Tonic Adaptation\n(100 positive rewards)')

    fig.suptitle('Experiment 2: Dopamine Tonic-Phasic Separation', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig2_dopamine_tonic_phasic.png')


# ══════════════════════════════════════════════════════
# Figure 3: D2 Receptor Occupancy
# ══════════════════════════════════════════════════════

def fig3_d2_receptor(results):
    print("\n[Figure 3] D2 Receptor Occupancy")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 3a: Occupancy curve
    ax = axes[0]
    oc = results['occupancy_curve']
    ax.plot(oc['concentrations'], oc['occupancies'], 'b-', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% occupancy')
    ax.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='80% occupancy (antipsychotic)')
    ax.set_xlabel('[DA] (nM)')
    ax.set_ylabel('Occupancy')
    ax.set_title('(a) D2 Occupancy Curve\n(Langmuir equation)')
    ax.legend()

    # 3b: Desensitization (high DA)
    ax = axes[1]
    desens = results['desensitization']
    ax.plot(desens['densities'], 'r-', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Receptor density')
    ax.set_title('(b) Desensitization\n(Chronic high DA)')

    # 3c: Upregulation (low DA)
    ax = axes[2]
    upreg = results['upregulation']
    ax.plot(upreg['densities'], 'g-', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Receptor density')
    ax.set_title('(c) Upregulation\n(Chronic low DA)')

    fig.suptitle('Experiment 3: D2 Receptor Occupancy', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig3_d2_receptor_occupancy.png')


# ══════════════════════════════════════════════════════
# Figure 4: LIF Refractory + Adaptation
# ══════════════════════════════════════════════════════

def fig4_lif(results):
    print("\n[Figure 4] LIF Refractory + Adaptation")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 4a: Refractory period
    ax = axes[0]
    ref_data = results['refractory_test']
    steps = [r['step'] for r in ref_data]
    spikes = [r['spikes'] for r in ref_data]
    refractory = [r['in_refractory'] for r in ref_data]
    ax.bar(steps, spikes, color='steelblue', alpha=0.7, label='Spikes')
    ax.bar(steps, refractory, bottom=spikes, color='salmon', alpha=0.7, label='In refractory')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Count')
    ax.set_title('(a) Refractory Period Effect')
    ax.legend()

    # 4b: Adaptation trace
    ax = axes[1]
    adapt_data = results['adaptation_test']
    ax.plot(adapt_data['adaptation_trace'], 'r-', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Adaptation level')
    ax.set_title('(b) Spike-Frequency Adaptation')

    # 4c: Firing rate over time
    ax = axes[2]
    ax.plot(adapt_data['spike_counts'], 'b-', linewidth=1.5)
    # Moving average
    window = 10
    if len(adapt_data['spike_counts']) >= window:
        ma = np.convolve(adapt_data['spike_counts'], np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, window-1+len(ma)), ma, 'r-', linewidth=2, label='Moving avg')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Spike count')
    ax.set_title(f'(c) Firing Rate\n(Reduction: {results["rate_reduction"]:.1f}%)')
    ax.legend()

    fig.suptitle('Experiment 4: LIF Refractory Period + Adaptation', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig4_lif_refractory_adaptation.png')


# ══════════════════════════════════════════════════════
# Figure 5: Sharp-Wave Ripple
# ══════════════════════════════════════════════════════

def fig5_ripple(results):
    print("\n[Figure 5] Sharp-Wave Ripple")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 5a: Ripple triggering conditions
    ax = axes[0]
    events = results['ripple_events']
    labels = [f"{e['sleep']}\narousal={e['arousal']}" for e in events]
    triggers = [1 if e['trigger'] else 0 for e in events]
    colors = ['green' if t else 'red' for t in triggers]
    ax.bar(range(len(labels)), triggers, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Ripple triggered')
    ax.set_title('(a) Ripple Triggering\n(Buzsáki, 2015)')

    # 5b: Phase progression
    ax = axes[1]
    phases = results['phase_progression']
    if phases:
        ax.plot(phases, 'b-', linewidth=2)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Ripple phase')
        ax.set_title('(b) Ripple Phase Progression')
    else:
        ax.text(0.5, 0.5, 'No ripple data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(b) Ripple Phase Progression')

    # 5c: Replay summary
    ax = axes[2]
    ax.bar(['Replay events'], [results['replay_count']], color='steelblue', alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title(f'(c) Replay Output\n({results["replay_count"]} events)')

    fig.suptitle('Experiment 5: Sharp-Wave Ripple', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig5_sharp_wave_ripple.png')


# ══════════════════════════════════════════════════════
# Figure 6: HC-PFC Connection
# ══════════════════════════════════════════════════════

def fig6_hc_pfc(results):
    print("\n[Figure 6] HC-PFC Connection")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 6a: HC→PFC transfer
    ax = axes[0]
    hc_pfc = results['hc_to_pfc']
    types = [h['type'] for h in hc_pfc]
    norms = [h['pfc_norm'] for h in hc_pfc]
    ax.bar(types, norms, color='steelblue', alpha=0.8)
    ax.set_ylabel('PFC input norm')
    ax.set_title('(a) HC→PFC Transfer\n(Preston & Eichenbaum, 2013)')

    # 6b: PFC→HC modulation
    ax = axes[1]
    pfc_hc = results['pfc_to_hc']
    types2 = [p['type'] for p in pfc_hc]
    norms2 = [p['hc_norm'] for p in pfc_hc]
    ax.bar(types2, norms2, color='salmon', alpha=0.8)
    ax.set_ylabel('HC modulation norm')
    ax.set_title('(b) PFC→HC Modulation')

    # 6c: Connection stats
    ax = axes[2]
    stats = results['stats']
    ax.bar(['HC→PFC', 'PFC→HC'],
           [stats['hc_to_pfc_transfers'], stats['pfc_to_hc_transfers']],
           color=['steelblue', 'salmon'], alpha=0.8)
    ax.set_ylabel('Transfer count')
    ax.set_title(f'(c) Connection Stats\nStrength={stats["connection_strength"]:.3f}')

    fig.suptitle('Experiment 6: HC-PFC Bidirectional Connection', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig6_hc_pfc_connection.png')


# ══════════════════════════════════════════════════════
# Figure 7: vmPFC Fear Extinction
# ══════════════════════════════════════════════════════

def fig7_vmpfc(results):
    print("\n[Figure 7] vmPFC Fear Extinction")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 7a: Extinction learning curve
    ax = axes[0]
    ext_strengths = results['extinction_learning']
    ax.plot(ext_strengths, 'g-', linewidth=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Extinction strength')
    ax.set_title('(a) Extinction Learning\n(Milad & Quirk, 2012)')

    # 7b: Safety detection
    ax = axes[1]
    safety = results['safety_detection']
    ax.bar(['Safe context', 'Unsafe context'],
           [safety['safe'], safety['unsafe']],
           color=['green', 'red'], alpha=0.8)
    ax.set_ylabel('Safety level')
    ax.set_title('(b) Safety Detection')

    # 7c: Extinction signal
    ax = axes[2]
    ax.bar(['Extinction signal'], [results['extinction_signal']], color='teal', alpha=0.8)
    ax.set_ylabel('Signal strength')
    ax.set_title('(c) Extinction Output Signal')

    fig.suptitle('Experiment 7: vmPFC Fear Extinction', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig7_vmpfc_extinction.png')


# ══════════════════════════════════════════════════════
# Figure 8: Amygdala-Hippocampus
# ══════════════════════════════════════════════════════

def fig8_amygdala_hc(results):
    print("\n[Figure 8] Amygdala-Hippocampus")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 8a: Emotion→Memory modulation
    ax = axes[0]
    e2m = results['emotion_to_memory']
    descs = [e['desc'][:20] for e in e2m]
    norms = [e['modulation_norm'] for e in e2m]
    ax.barh(range(len(descs)), norms, color='salmon', alpha=0.8)
    ax.set_yticks(range(len(descs)))
    ax.set_yticklabels(descs, fontsize=7)
    ax.set_xlabel('Modulation norm')
    ax.set_title('(a) Emotion→Memory\n(Richter-Levin & Akirav, 2010)')

    # 8b: Memory→Emotion trigger
    ax = axes[1]
    m2e = results['memory_to_emotion']
    mem_ids = [m['memory_idx'] for m in m2e]
    e_norms = [m['emotion_norm'] for m in m2e]
    ax.bar(mem_ids, e_norms, color='steelblue', alpha=0.8)
    ax.set_xlabel('Memory index')
    ax.set_ylabel('Emotion signal norm')
    ax.set_title('(b) Memory→Emotion Trigger')

    # 8c: Connection stats
    ax = axes[2]
    stats = results['stats']
    ax.bar(['Emotion→Memory', 'Memory→Emotion'],
           [stats['emotion_to_memory_count'], stats['memory_to_emotion_count']],
           color=['salmon', 'steelblue'], alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title(f'(c) Connection Stats\nStrength={stats["modulation_strength"]:.3f}')

    fig.suptitle('Experiment 8: Amygdala-Hippocampus Emotional Modulation', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig8_amygdala_hippocampus.png')


# ══════════════════════════════════════════════════════
# Figure 9: Attractor Working Memory
# ══════════════════════════════════════════════════════

def fig9_attractor_wm(results):
    print("\n[Figure 9] Attractor Working Memory")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 9a: Drift over time
    ax = axes[0]
    persist = results['persistence']
    ax.plot(persist['drifts'], 'b-', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Average drift')
    ax.set_title('(a) Memory Drift\n(Compte et al., 2000)')

    # 9b: Active attractors
    ax = axes[1]
    ax.plot(persist['active_counts'], 'g-', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Active attractors')
    ax.set_title('(b) Active Attractors')

    # 9c: DA modulation effect
    ax = axes[2]
    da_mod = results['da_modulation']
    da_levels = list(da_mod.keys())
    avg_drifts = list(da_mod.values())
    ax.bar([f'DA={d}' for d in da_levels], avg_drifts,
           color=['salmon', 'steelblue', 'green'], alpha=0.8)
    ax.set_ylabel('Average drift')
    ax.set_title('(c) DA Modulation on Stability')

    fig.suptitle('Experiment 9: Attractor Working Memory', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig9_attractor_working_memory.png')


# ══════════════════════════════════════════════════════
# Figure 10: Thalamic ACh Gating
# ══════════════════════════════════════════════════════

def fig10_ach_gating(results):
    print("\n[Figure 10] Thalamic ACh Gating")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 10a: ACh level vs gating
    ax = axes[0]
    gl = results['gating_levels']
    ach_vals = [g['ach'] for g in gl]
    gating_vals = [g['avg_gating'] for g in gl]
    ax.plot(ach_vals, gating_vals, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('ACh level')
    ax.set_ylabel('Avg gating')
    ax.set_title('(a) ACh→Gating Curve\n(Hirsch et al., 2018)')

    # 10b: Phasic ACh response
    ax = axes[1]
    phasic = results['phasic_response']
    ax.plot(phasic, 'r-', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Total ACh')
    ax.set_title('(b) Phasic ACh Decay')

    # 10c: Sensory modulation
    ax = axes[2]
    sm = results['sensory_modulation']
    ax.bar(['High ACh', 'Low ACh'],
           [sm['high_ach'], sm['low_ach']],
           color=['green', 'gray'], alpha=0.8)
    ax.set_ylabel('Modulated input norm')
    ax.set_title('(c) Sensory Modulation')

    fig.suptitle('Experiment 10: Thalamic ACh Gating', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig10_thalamic_ach_gating.png')


# ══════════════════════════════════════════════════════
# Figure 11: Theta-Gamma Coupling
# ══════════════════════════════════════════════════════

def fig11_theta_gamma(results):
    print("\n[Figure 11] Theta-Gamma Coupling")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 11a: Phase-amplitude coupling
    ax = axes[0]
    pa = results['phase_amplitude']
    phases = np.array(pa['phases'])
    gamma_amps = np.array(pa['gamma_amps'])
    ax.scatter(phases, gamma_amps, c=phases, cmap='hsv', s=10, alpha=0.6)
    ax.set_xlabel('Theta phase (rad)')
    ax.set_ylabel('Gamma amplitude')
    ax.set_title('(a) Phase-Amplitude Coupling\n(Jensen & Tesche, 2002)')

    # 11b: Time series
    ax = axes[1]
    steps = np.arange(len(gamma_amps))
    ax.plot(steps, gamma_amps, 'b-', linewidth=1, alpha=0.7)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Gamma amplitude')
    ax.set_title('(b) Gamma Amplitude Time Series')

    # 11c: DA modulation
    ax = axes[2]
    da_mod = results['da_modulation']
    da_levels = list(da_mod.keys())
    couplings = list(da_mod.values())
    ax.bar([f'DA={d}' for d in da_levels], couplings,
           color=['salmon', 'steelblue', 'green'], alpha=0.8)
    ax.set_ylabel('Avg coupling strength')
    ax.set_title('(c) DA Modulation of Coupling')

    fig.suptitle('Experiment 11: Theta-Gamma Rhythm Coupling', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig11_theta_gamma_coupling.png')


# ══════════════════════════════════════════════════════
# Figure 12: NMDA Plasticity + Synaptic Scaling
# ══════════════════════════════════════════════════════

def fig12_nmda_scaling(results):
    print("\n[Figure 12] NMDA Plasticity + Synaptic Scaling")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 12a: LTP/LTD by Ca level
    ax = axes[0]
    ltp_ltd = results['ltp_ltd']
    cases = ['High Glu+Depol\n(LTP)', 'Medium Glu+Depol\n(LTD)']
    cas_data = [ltp_ltd['high_case'], ltp_ltd['medium_case']]
    ax.bar(cases, [c['ca'] for c in cas_data], color=['steelblue', 'salmon'], alpha=0.8)
    ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='LTP threshold')
    ax.axhline(0.2, color='orange', linestyle='--', alpha=0.5, label='LTD threshold')
    ax.set_ylabel('Ca²⁺ level')
    ax.set_title('(a) Ca²⁺ Threshold\n(Malenka & Bear, 2004)')
    ax.legend(fontsize=7)

    # 12b: Synaptic scaling
    ax = axes[1]
    scaling = results['scaling']
    ax.axhline(scaling['target'], color='green', linestyle='--', alpha=0.5, label='Target')
    ax.bar(['Initial', 'Final'],
           [scaling['initial_strength'], scaling['final_strength']],
           color=['gray', 'steelblue'], alpha=0.8)
    ax.set_ylabel('Avg synaptic strength')
    ax.set_title('(b) Synaptic Scaling\n(Turrigiano, 2008)')
    ax.legend()

    # 12c: Homeostatic regulation
    ax = axes[2]
    homeo = results['homeostatic']
    ax.bar(['Before scaling', 'After scaling'],
           [homeo['before'], homeo['after']],
           color=['salmon', 'green'], alpha=0.8)
    ax.set_ylabel('Total synaptic strength')
    reduction = (homeo['before'] - homeo['after']) / homeo['before'] * 100
    ax.set_title(f'(c) Homeostatic Regulation\n({reduction:.1f}% reduction)')

    fig.suptitle('Experiment 12: NMDA-dependent LTP/LTD + Synaptic Scaling', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig12_nmda_plasticity_scaling.png')


# ══════════════════════════════════════════════════════
# Summary Figure
# ══════════════════════════════════════════════════════

def fig_summary(all_results):
    print("\n[Summary Figure]")
    fig, ax = plt.subplots(figsize=(14, 6))

    experiments = [
        '1. STDP\nPlasticity',
        '2. DA\nTonic-Phasic',
        '3. D2\nReceptor',
        '4. LIF\nRefractory',
        '5. Sharp-Wave\nRipple',
        '6. HC-PFC\nConnection',
        '7. vmPFC\nExtinction',
        '8. Amygdala-HC\nModulation',
        '9. Attractor\nWM',
        '10. ACh\nGating',
        '11. Theta-Gamma\nCoupling',
        '12. NMDA\nLTP/LTD',
    ]

    # Status: 1=success, 0=error
    statuses = []
    for key in [
        'exp1_stdp_plasticity', 'exp2_dopamine_tonic_phasic',
        'exp3_d2_receptor_occupancy', 'exp4_lif_refractory_adaptation',
        'exp5_sharp_wave_ripple', 'exp6_hc_pfc_connection',
        'exp7_vmpfc_extinction', 'exp8_amygdala_hippocampus',
        'exp9_attractor_working_memory', 'exp10_thalamic_ach_gating',
        'exp11_theta_gamma_coupling', 'exp12_nmda_plasticity',
    ]:
        statuses.append(0 if 'error' in all_results.get(key, {}) else 1)

    colors = ['green' if s else 'red' for s in statuses]
    ax.bar(range(len(experiments)), statuses, color=colors, alpha=0.8)
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, fontsize=8)
    ax.set_ylabel('Status (1=Pass, 0=Fail)')
    ax.set_title('Neural Model Improvements: Experiment Results Summary', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.1, 1.3)

    # Add pass/fail labels
    for i, s in enumerate(statuses):
        ax.text(i, s + 0.05, 'PASS' if s else 'FAIL', ha='center', fontsize=8, fontweight='bold')

    save_fig(fig, 'fig_summary_all_experiments.png')


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def generate_all_figures():
    print("="*60)
    print("GENERATING NEURAL IMPROVEMENT FIGURES")
    print("="*60)

    # Run experiments
    all_results = run_all_experiments()

    # Generate figures
    fig_generators = [
        ('exp1_stdp_plasticity', fig1_stdp),
        ('exp2_dopamine_tonic_phasic', fig2_dopamine),
        ('exp3_d2_receptor_occupancy', fig3_d2_receptor),
        ('exp4_lif_refractory_adaptation', fig4_lif),
        ('exp5_sharp_wave_ripple', fig5_ripple),
        ('exp6_hc_pfc_connection', fig6_hc_pfc),
        ('exp7_vmpfc_extinction', fig7_vmpfc),
        ('exp8_amygdala_hippocampus', fig8_amygdala_hc),
        ('exp9_attractor_working_memory', fig9_attractor_wm),
        ('exp10_thalamic_ach_gating', fig10_ach_gating),
        ('exp11_theta_gamma_coupling', fig11_theta_gamma),
        ('exp12_nmda_plasticity', fig12_nmda_scaling),
    ]

    success_count = 0
    fail_count = 0

    for exp_key, fig_func in fig_generators:
        try:
            if exp_key in all_results and 'error' not in all_results[exp_key]:
                fig_func(all_results[exp_key])
                success_count += 1
            else:
                print(f"\nSkipping {exp_key}: experiment failed")
                fail_count += 1
        except Exception as e:
            print(f"\nError generating figure for {exp_key}: {e}")
            fail_count += 1

    # Summary figure
    try:
        fig_summary(all_results)
        success_count += 1
    except Exception as e:
        print(f"Error generating summary: {e}")

    print("\n" + "="*60)
    print(f"FIGURE GENERATION COMPLETE")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output: {FIG_DIR}")
    print("="*60)


if __name__ == "__main__":
    generate_all_figures()
