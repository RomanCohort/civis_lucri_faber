"""
Neural Model Improvements Experiments
=====================================

12 experiments validating the new neural model improvements:
1. STDP synaptic plasticity (neuroplasticity.py)
2. Dopamine Tonic-Phasic separation (neurotransmitter.py)
3. D2 receptor dynamics (neurotransmitter.py)
4. LIF refractory period + adaptation (snn_core.py)
5. Sharp-wave ripple mechanism (hippocampus.py)
6. HC-PFC bidirectional connection (hippocampus.py)
7. vmPFC fear extinction (limbic.py)
8. Amygdala-Hippocampus emotional modulation (limbic.py)
9. Attractor working memory (prefrontal_cortex.py)
10. Thalamic ACh gating (rhythm.py)
11. Theta-gamma rhythm coupling (rhythm.py)
12. NMDA-dependent LTP/LTD + Synaptic scaling (neuroplasticity.py)

Reference:
- Bi & Poo (1998) - STDP timing window
- Schultz (1997) - Dopamine RPE signaling
- Seeman et al. (2005) - D2 receptor occupancy
- Buzsáki (2015) - Sharp-wave ripples
- Milad & Quirk (2012) - vmPFC fear extinction
- Compte et al. (2000) - Attractor working memory
- Jensen & Tesche (2002) - Theta-gamma coupling
- Malenka & Bear (2004) - NMDA LTP/LTD
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, List, Any
import time

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Direct imports (bypass broken __init__.py)
from neuroplasticity import (
    STDPSynapse, STDPConfig, NMDADependentPlasticity,
    SynapticScaling, Synapse, NeuroplasticitySystem
)
from neurotransmitter import (
    DopamineSystem, DopamineReceptor, DATransporter
)
from snn_core import LeakyIntegrateAndFire, SNNConfig
from hippocampus import SharpWaveRipple, HCPFCConnection, Hippocampus
from limbic import (
    vmPFCExtinction, AmygdalaHippocampusConnection, EnhancedAmygdala,
    LimbicSystem
)
from prefrontal_cortex import AttractorWorkingMemory, PrefrontalCortex
from rhythm import ThetaGammaCoupling, ThalamicAChGating, RhythmSystem


# ══════════════════════════════════════════════════════
# Experiment 1: STDP Synaptic Plasticity
# ══════════════════════════════════════════════════════

def exp1_stdp_plasticity() -> Dict:
    """Test STDP learning window - Bi & Poo (1998)

    Expected behavior:
    - Pre before Post spike → LTP (weight increase)
    - Post before Pre spike → LTD (weight decrease)
    - Timing window ~20ms exponential decay
    """
    print("\n" + "="*60)
    print("Experiment 1: STDP Synaptic Plasticity")
    print("="*60)

    results = {
        'name': 'STDP Plasticity',
        'ltp_cases': [],
        'ltd_cases': [],
        'timing_window': [],
    }

    # Create STDP synapse
    config = STDPConfig(tau_plus=20e-3, tau_minus=20e-3, A_plus=0.01, A_minus=0.012)
    synapse = STDPSynapse(pre_neuron_id=0, post_neuron_id=1, initial_weight=0.5, config=config)

    # Test LTP: Pre before Post (positive dt)
    print("\n[1.1] Testing LTP (Pre before Post)")
    for dt in [5, 10, 20, 40, 80]:  # ms
        synapse.pre_spike_times = [0.0]
        synapse.post_spike_times = [float(dt)]
        delta_w = synapse.compute_weight_change(0.0, float(dt))
        results['ltp_cases'].append({'dt': dt, 'delta_w': delta_w})
        print(f"  dt = {dt}ms → Δw = {delta_w:.6f} (LTP)")

    # Test LTD: Post before Pre (negative dt)
    print("\n[1.2] Testing LTD (Post before Pre)")
    for dt in [5, 10, 20, 40, 80]:  # ms
        synapse.pre_spike_times = [float(dt)]
        synapse.post_spike_times = [0.0]
        delta_w = synapse.compute_weight_change(float(dt), 0.0)
        results['ltd_cases'].append({'dt': dt, 'delta_w': delta_w})
        print(f"  dt = {dt}ms → Δw = {delta_w:.6f} (LTD)")

    # Test timing window decay
    print("\n[1.3] Testing timing window decay")
    dts = np.linspace(1, 100, 50)
    ltp_values = []
    ltd_values = []
    for dt in dts:
        # LTP
        delta_ltp = config.A_plus * np.exp(-dt / (config.tau_plus * 1000))
        # LTD
        delta_ltd = -config.A_minus * np.exp(-dt / (config.tau_minus * 1000))
        ltp_values.append(delta_ltp)
        ltd_values.append(delta_ltd)

    results['timing_window'] = {
        'dts': dts.tolist(),
        'ltp_values': ltp_values,
        'ltd_values': ltd_values,
    }

    # Apply STDP learning
    print("\n[1.4] Testing STDP learning application")
    synapse2 = STDPSynapse(0, 1, 0.5, config)
    for step in range(100):
        t_pre = step * 10.0 + np.random.randn() * 5
        t_post = t_pre + 10.0  # Pre before Post → LTP
        synapse2.record_pre_spike(t_pre)
        synapse2.record_post_spike(t_post)
        synapse2.apply_stdp(t_pre + 50.0)

    print(f"  Initial weight: 0.5")
    print(f"  Final weight: {synapse2.weight:.4f}")
    print(f"  LTP events: {synapse2.ltp_count}, LTD events: {synapse2.ltd_count}")
    results['learning_result'] = {
        'initial_weight': 0.5,
        'final_weight': synapse2.weight,
        'ltp_count': synapse2.ltp_count,
        'ltd_count': synapse2.ltd_count,
    }

    print("\n✓ Experiment 1 completed")
    return results


# ══════════════════════════════════════════════════════
# Experiment 2: Dopamine Tonic-Phasic Separation
# ══════════════════════════════════════════════════════

def exp2_dopamine_tonic_phasic() -> Dict:
    """Test tonic/phasic separation and RPE signaling - Schultz (1997)

    Expected behavior:
    - Positive RPE → Phasic burst
    - Negative RPE → Phasic dip
    - Tonic slowly adapts to long-term reward rate
    """
    print("\n" + "="*60)
    print("Experiment 2: Dopamine Tonic-Phasic Separation")
    print("="*60)

    results = {
        'name': 'Dopamine Tonic-Phasic',
        'rpe_tests': [],
        'time_course': [],
    }

    da = DopamineSystem(baseline=0.5, tonic_baseline=0.3, phasic_decay_rate=0.3)

    # Test RPE signaling
    print("\n[2.1] Testing RPE signaling")
    test_cases = [
        {'reward': 1.0, 'expectation': 0.5, 'desc': 'Positive RPE'},
        {'reward': 0.0, 'expectation': 0.5, 'desc': 'Negative RPE'},
        {'reward': 0.5, 'expectation': 0.5, 'desc': 'No RPE'},
        {'reward': 1.0, 'expectation': 0.0, 'desc': 'Large positive RPE'},
        {'reward': 0.0, 'expectation': 1.0, 'desc': 'Large negative RPE'},
    ]

    for case in test_cases:
        da.dopamine_tonic = 0.3
        da.dopamine_phasic = 0.0
        total, tonic, phasic = da.compute_reward_signal(case['reward'], case['expectation'])
        rpe = case['reward'] - case['expectation']
        results['rpe_tests'].append({
            'desc': case['desc'],
            'rpe': rpe,
            'total': total,
            'tonic': tonic,
            'phasic': phasic,
        })
        print(f"  {case['desc']}: RPE={rpe:.2f} → Total={total:.3f}, Tonic={tonic:.3f}, Phasic={phasic:.3f}")

    # Test phasic decay
    print("\n[2.2] Testing phasic decay")
    da.dopamine_tonic = 0.3
    da.dopamine_phasic = 0.5  # Initial burst
    phasic_values = [da.dopamine_phasic]
    for _ in range(20):
        da.dopamine_phasic *= (1 - da.phasic_decay_rate)
        phasic_values.append(da.dopamine_phasic)
    print(f"  Initial phasic: 0.5")
    print(f"  After 20 steps: {da.dopamine_phasic:.4f}")
    results['phasic_decay'] = phasic_values

    # Test tonic adaptation
    print("\n[2.3] Testing tonic adaptation (long-term)")
    da2 = DopamineSystem(baseline=0.5, tonic_baseline=0.3)
    tonic_values = [da2.dopamine_tonic]
    for i in range(100):
        # Simulate consistent positive rewards
        reward = 0.8
        expectation = 0.5
        da2.compute_reward_signal(reward, expectation)
        tonic_values.append(da2.dopamine_tonic)
    print(f"  Initial tonic: 0.3")
    print(f"  After 100 positive rewards: {da2.dopamine_tonic:.4f}")
    results['tonic_adaptation'] = tonic_values

    print("\n✓ Experiment 2 completed")
    return results


# ══════════════════════════════════════════════════════
# Experiment 3: D2 Receptor Occupancy
# ══════════════════════════════════════════════════════

def exp3_d2_receptor_occupancy() -> Dict:
    """Test D2 receptor occupancy - Langmuir equation - Seeman et al. (2005)

    Expected behavior:
    - Occupancy = [DA] / ([DA] + Kd)
    - High affinity state saturates first
    - Low affinity state saturates later
    """
    print("\n" + "="*60)
    print("Experiment 3: D2 Receptor Occupancy")
    print("="*60)

    results = {
        'name': 'D2 Receptor Occupancy',
        'occupancy_curve': [],
        'desensitization': [],
    }

    receptor = DopamineReceptor(receptor_type="D2", density=1.0, high_affinity_fraction=0.2)

    # Test occupancy curve
    print("\n[3.1] Testing occupancy curve (Langmuir)")
    da_concentrations = np.linspace(0, 100, 50)  # nM
    occupancies = []
    for da_conc in da_concentrations:
        occ = receptor.compute_occupancy(da_conc)
        occupancies.append(occ)
    results['occupancy_curve'] = {
        'concentrations': da_concentrations.tolist(),
        'occupancies': occupancies,
    }

    # Key points
    print(f"  Kd_high = {receptor.Kd_high*1e9:.1f} nM")
    print(f"  Kd_low = {receptor.Kd_low*1e9:.1f} nM")
    print(f"  Occupancy at 1 nM: {receptor.compute_occupancy(1.0):.4f}")
    print(f"  Occupancy at 10 nM: {receptor.compute_occupancy(10.0):.4f}")
    print(f"  Occupancy at 100 nM: {receptor.compute_occupancy(100.0):.4f}")

    # Test desensitization/downregulation
    print("\n[3.2] Testing receptor desensitization")
    receptor2 = DopamineReceptor(receptor_type="D2", density=1.0)
    densities = [receptor2.density]
    for _ in range(100):
        receptor2.apply_desensitization(chronic_da_level=0.8)  # High chronic DA
        densities.append(receptor2.density)
    print(f"  Initial density: 1.0")
    print(f"  After 100 steps high DA: {receptor2.density:.4f}")
    print(f"  Desensitization: {receptor2.desensitization:.4f}")
    results['desensitization'] = {
        'densities': densities,
        'final_desensitization': receptor2.desensitization,
    }

    # Test upregulation
    print("\n[3.3] Testing receptor upregulation (low DA)")
    receptor3 = DopamineReceptor(receptor_type="D2", density=1.0)
    densities_up = [receptor3.density]
    for _ in range(100):
        receptor3.apply_desensitization(chronic_da_level=0.1)  # Low chronic DA
        densities_up.append(receptor3.density)
    print(f"  Initial density: 1.0")
    print(f"  After 100 steps low DA: {receptor3.density:.4f}")
    print(f"  Upregulation: {receptor3.upregulation:.4f}")
    results['upregulation'] = {
        'densities': densities_up,
        'final_upregulation': receptor3.upregulation,
    }

    print("\n✓ Experiment 3 completed")
    return results


# ══════════════════════════════════════════════════════
# Experiment 4: LIF Refractory Period + Adaptation
# ══════════════════════════════════════════════════════

def exp4_lif_refractory_adaptation() -> Dict:
    """Test refractory period and spike-frequency adaptation

    Expected behavior:
    - Refractory: No spikes during refractory period after spike
    - Adaptation: Threshold increases with each spike, reducing firing rate
    """
    print("\n" + "="*60)
    print("Experiment 4: LIF Refractory Period + Adaptation")
    print("="*60)

    results = {
        'name': 'LIF Refractory + Adaptation',
        'refractory_test': [],
        'adaptation_test': [],
    }

    # Test refractory period
    print("\n[4.1] Testing refractory period")
    config = SNNConfig(tau_mem=10e-3, v_thresh=-55e-3, v_rest=-70e-3, tau_ref=5e-3)
    lif = LeakyIntegrateAndFire(n_neurons=10, config=config)

    # Strong input to trigger spikes
    strong_input = torch.ones(1, 10) * 2.0
    refractory_states = []
    for step in range(20):
        result = lif(strong_input)
        refractory_states.append({
            'step': step,
            'spikes': result['spikes'].sum().item(),
            'in_refractory': result['in_refractory'].sum().item(),
        })
    results['refractory_test'] = refractory_states
    print(f"  Refractory period: {config.tau_ref*1000:.1f} ms")
    print(f"  Total spikes over 20 steps: {sum(s['spikes'] for s in refractory_states)}")

    # Test spike-frequency adaptation
    print("\n[4.2] Testing spike-frequency adaptation")
    config2 = SNNConfig(tau_mem=10e-3, v_thresh=-55e-3, v_rest=-70e-3,
                        tau_ref=1e-3, tau_adapt=100e-3, adapt_rate=0.1)
    lif2 = LeakyIntegrateAndFire(n_neurons=10, config=config2)

    adaptation_trace = []
    spike_counts = []
    for step in range(100):
        result = lif2(strong_input)
        adaptation_trace.append(result['adaptation'].mean().item())
        spike_counts.append(result['spikes'].sum().item())

    results['adaptation_test'] = {
        'adaptation_trace': adaptation_trace,
        'spike_counts': spike_counts,
    }
    print(f"  Initial adaptation: {adaptation_trace[0]:.4f}")
    print(f"  Final adaptation: {adaptation_trace[-1]:.4f}")
    print(f"  Initial spike rate: {spike_counts[0]:.1f}")
    print(f"  Final spike rate: {spike_counts[-1]:.1f}")

    # Test firing rate reduction due to adaptation
    print("\n[4.3] Testing firing rate reduction")
    early_rate = np.mean(spike_counts[:20])
    late_rate = np.mean(spike_counts[-20:])
    rate_reduction = (early_rate - late_rate) / early_rate * 100
    print(f"  Early firing rate: {early_rate:.2f}")
    print(f"  Late firing rate: {late_rate:.2f}")
    print(f"  Rate reduction: {rate_reduction:.1f}%")
    results['rate_reduction'] = rate_reduction

    print("\n✓ Experiment 4 completed")
    return results


# ══════════════════════════════════════════════════════
# Experiment 5: Sharp-Wave Ripple
# ══════════════════════════════════════════════════════

def exp5_sharp_wave_ripple() -> Dict:
    """Test sharp-wave ripple triggering and replay - Buzsáki (2015)

    Expected behavior:
    - Ripples triggered during sleep/low arousal
    - ~200Hz oscillation
    - Memory sequence replay
    """
    print("\n" + "="*60)
    print("Experiment 5: Sharp-Wave Ripple")
    print("="*60)

    results = {
        'name': 'Sharp-Wave Ripple',
        'ripple_events': [],
        'phase_progression': [],
    }

    ripple = SharpWaveRipple(n_neurons=128)

    # Test ripple triggering conditions
    print("\n[5.1] Testing ripple triggering conditions")
    conditions = [
        {'sleep': 'awake', 'arousal': 0.8, 'activity': 0.2},
        {'sleep': 'awake', 'arousal': 0.3, 'activity': 0.5},
        {'sleep': 'NREM', 'arousal': 0.2, 'activity': 0.3},
        {'sleep': 'REM', 'arousal': 0.5, 'activity': 0.4},
    ]

    for cond in conditions:
        trigger = ripple.check_ripple_trigger(
            sleep_stage=cond['sleep'],
            arousal_level=cond['arousal'],
            recent_activity=cond['activity']
        )
        results['ripple_events'].append({**cond, 'trigger': trigger})
        print(f"  {cond['sleep']}, arousal={cond['arousal']:.1f} → trigger={trigger}")

    # Test ripple progression
    print("\n[5.2] Testing ripple phase progression")
    # Create mock sequence
    sequence = [np.random.randn(128).astype(np.float32) for _ in range(10)]

    ripple.start_ripple(sequence)
    phases = [ripple.current_phase]
    for _ in range(20):
        result = ripple.step(sleep_stage='NREM', arousal_level=0.2, recent_sequence=sequence)
        phases.append(ripple.current_phase)
        if not ripple.is_active:
            break

    results['phase_progression'] = phases
    print(f"  Ripple frequency: ~200 Hz")
    print(f"  Phases recorded: {len(phases)}")
    print(f"  Total ripples: {len(ripple.ripple_events)}")

    # Test replay output
    print("\n[5.3] Testing replay output")
    ripple2 = SharpWaveRipple(n_neurons=128)
    replay_outputs = []
    for _ in range(50):
        result = ripple2.step(sleep_stage='NREM', arousal_level=0.2, recent_sequence=sequence)
        if result.get('replay_output') is not None:
            replay_outputs.append(result['replay_output'])

    print(f"  Replay outputs generated: {len(replay_outputs)}")
    results['replay_count'] = len(replay_outputs)

    print("\n✓ Experiment 5 completed")
    return results


# ══════════════════════════════════════════════════════
# Experiment 6: HC-PFC Bidirectional Connection
# ══════════════════════════════════════════════════════

def exp6_hc_pfc_connection() -> Dict:
    """Test bidirectional HC-PFC information transfer - Preston & Eichenbaum (2013)

    Expected behavior:
    - HC→PFC: Episodic memory transfer
    - PFC→HC: Goal/attention modulation
    """
    print("\n" + "="*60)
    print("Experiment 6: HC-PFC Bidirectional Connection")
    print("="*60)

    results = {
        'name': 'HC-PFC Connection',
        'hc_to_pfc': [],
        'pfc_to_hc': [],
    }

    connection = HCPFCConnection(hc_dim=128, pfc_dim=64, connection_strength=0.5)

    # Test HC→PFC transfer
    print("\n[6.1] Testing HC→PFC transfer")
    hc_encodings = [torch.randn(128) for _ in range(5)]
    transfer_types = ['episodic', 'prediction', 'spatial']

    for t_type in transfer_types:
        pfc_input = connection.transfer_to_pfc(hc_encodings[0], transfer_type=t_type)
        results['hc_to_pfc'].append({
            'type': t_type,
            'pfc_norm': pfc_input.norm().item(),
        })
        print(f"  {t_type}: PFC input norm = {pfc_input.norm().item():.4f}")

    # Test PFC→HC modulation
    print("\n[6.2] Testing PFC→HC modulation")
    pfc_signals = [torch.randn(64) for _ in range(5)]
    modulation_types = ['goal', 'attention', 'strategy']

    for m_type in modulation_types:
        hc_mod = connection.modulate_from_pfc(pfc_signals[0], modulation_type=m_type)
        results['pfc_to_hc'].append({
            'type': m_type,
            'hc_norm': hc_mod.norm().item(),
        })
        print(f"  {m_type}: HC modulation norm = {hc_mod.norm().item():.4f}")

    # Test connection statistics
    print("\n[6.3] Testing connection statistics")
    stats = connection.get_connection_stats()
    print(f"  Connection strength: {stats['connection_strength']:.4f}")
    print(f"  HC→PFC transfers: {stats['hc_to_pfc_transfers']}")
    print(f"  PFC→HC transfers: {stats['pfc_to_hc_transfers']}")
    results['stats'] = stats

    print("\n✓ Experiment 6 completed")
    return results


# ══════════════════════════════════════════════════════
# Experiment 7: vmPFC Fear Extinction
# ══════════════════════════════════════════════════════

def exp7_vmpfc_extinction() -> Dict:
    """Test fear extinction learning - Milad & Quirk (2012)

    Expected behavior:
    - vmPFC learns safety signals
    - Extinction signal inhibits amygdala fear response
    - Gradual reduction of fear with safe exposures
    """
    print("\n" + "="*60)
    print("Experiment 7: vmPFC Fear Extinction")
    print("="*60)

    results = {
        'name': 'vmPFC Fear Extinction',
        'extinction_learning': [],
        'safety_detection': [],
    }

    vmpfc = vmPFCExtinction(input_dim=64, extinction_rate=0.05)

    # Create fear cue
    fear_cue = np.random.randn(64).astype(np.float32)

    # Test extinction learning
    print("\n[7.1] Testing extinction learning")
    extinction_strengths = []
    for trial in range(20):
        result = vmpfc.learn_extinction(fear_cue, safety_context="safe_room", trial_count=1)
        extinction_strengths.append(result['extinction_strength'])
        if trial % 5 == 0:
            print(f"  Trial {trial}: extinction strength = {result['extinction_strength']:.4f}")

    results['extinction_learning'] = extinction_strengths

    # Test safety detection
    print("\n[7.2] Testing safety detection")
    safe_context = torch.randn(1, 64)
    unsafe_context = torch.randn(1, 64) * 2.0  # More extreme

    safe_level = vmpfc.detect_safety(safe_context)
    unsafe_level = vmpfc.detect_safety(unsafe_context)
    print(f"  Safe context level: {safe_level:.4f}")
    print(f"  Unsafe context level: {unsafe_level:.4f}")
    results['safety_detection'] = {
        'safe': safe_level,
        'unsafe': unsafe_level,
    }

    # Test extinction signal output
    print("\n[7.3] Testing extinction signal output")
    from limbic import FearCondition
    fear_memories = [FearCondition(cue=fear_cue, response='fear', strength=0.8)]

    extinction_signal = vmpfc.output_extinction_signal(safe_context, fear_memories)
    print(f"  Extinction signal: {extinction_signal:.4f}")
    results['extinction_signal'] = extinction_signal

    # Get statistics
    stats = vmpfc.get_extinction_stats()
    print(f"  Total extinction memories: {stats['n_extinction_memories']}")
    print(f"  Average extinction strength: {stats['avg_extinction_strength']:.4f}")
    results['stats'] = stats

    print("\n✓ Experiment 7 completed")
    return results


# ══════════════════════════════════════════════════════
# Experiment 8: Amygdala-Hippocampus Emotional Modulation
# ══════════════════════════════════════════════════════

def exp8_amygdala_hippocampus() -> Dict:
    """Test emotion-memory modulation - Richter-Levin & Akirav (2010)

    Expected behavior:
    - High arousal enhances memory encoding
    - Negative valence also strengthens memory (fear memory)
    - Memory retrieval triggers emotion
    """
    print("\n" + "="*60)
    print("Experiment 8: Amygdala-Hippocampus Emotional Modulation")
    print("="*60)

    results = {
        'name': 'Amygdala-HC Modulation',
        'emotion_to_memory': [],
        'memory_to_emotion': [],
    }

    connection = AmygdalaHippocampusConnection(amygdala_dim=64, hippocampus_dim=128)

    # Test emotion→memory modulation
    print("\n[8.1] Testing emotion→memory modulation")
    emotion_encodings = [torch.randn(64) for _ in range(5)]
    arousal_valence_pairs = [
        (0.9, -0.8, 'High arousal, negative (fear)'),
        (0.9, 0.8, 'High arousal, positive (joy)'),
        (0.3, 0.0, 'Low arousal, neutral'),
        (0.5, -0.5, 'Medium arousal, negative'),
    ]

    for arousal, valence, desc in arousal_valence_pairs:
        modulation = connection.modulate_memory_encoding(emotion_encodings[0], arousal, valence)
        results['emotion_to_memory'].append({
            'desc': desc,
            'modulation_norm': modulation.norm().item(),
        })
        print(f"  {desc}: modulation norm = {modulation.norm().item():.4f}")

    # Test memory→emotion trigger
    print("\n[8.2] Testing memory→emotion trigger")
    memory_encodings = [torch.randn(128) for _ in range(5)]

    for i, mem_enc in enumerate(memory_encodings):
        emotion_signal = connection.trigger_emotion_from_memory(mem_enc)
        results['memory_to_emotion'].append({
            'memory_idx': i,
            'emotion_norm': emotion_signal.norm().item(),
        })
        print(f"  Memory {i}: emotion signal norm = {emotion_signal.norm().item():.4f}")

    # Test connection statistics
    stats = connection.get_connection_stats()
    print(f"\n[8.3] Connection statistics")
    print(f"  Modulation strength: {stats['modulation_strength']:.4f}")
    print(f"  Emotion→Memory count: {stats['emotion_to_memory_count']}")
    print(f"  Memory→Emotion count: {stats['memory_to_emotion_count']}")
    results['stats'] = stats

    print("\n✓ Experiment 8 completed")
    return results


# ══════════════════════════════════════════════════════
# Experiment 9: Attractor Working Memory
# ══════════════════════════════════════════════════════

def exp9_attractor_working_memory() -> Dict:
    """Test persistent activity and DA modulation - Compte et al. (2000)

    Expected behavior:
    - Persistent activity maintains information without input
    - High DA → stable memory (low drift)
    - Low DA → memory drift/decay
    """
    print("\n" + "="*60)
    print("Experiment 9: Attractor Working Memory")
    print("="*60)

    results = {
        'name': 'Attractor Working Memory',
        'persistence': [],
        'da_modulation': [],
    }

    attractor_wm = AttractorWorkingMemory(n_units=64, n_attractors=7)

    # Test persistent activity
    print("\n[9.1] Testing persistent activity")
    # Write items to memory
    items = [torch.randn(64) for _ in range(3)]
    for i, item in enumerate(items):
        attractor_wm.write(item, importance=1.0)
        print(f"  Written item {i}: norm = {item.norm().item():.4f}")

    # Maintain without input
    print("\n  Maintaining without input...")
    drifts = []
    active_counts = []
    for step in range(50):
        maint_result = attractor_wm.maintain(dopamine_level=0.5)
        drifts.append(maint_result['avg_drift'])
        active_counts.append(maint_result['active_attractors'])

    results['persistence'] = {
        'drifts': drifts,
        'active_counts': active_counts,
    }
    print(f"  Initial drift: {drifts[0]:.4f}")
    print(f"  Final drift: {drifts[-1]:.4f}")
    print(f"  Active attractors: {active_counts[-1]}")

    # Test DA modulation
    print("\n[9.2] Testing DA modulation on stability")
    da_levels = [0.2, 0.5, 0.8]
    da_results = {}

    for da_level in da_levels:
        attractor_wm2 = AttractorWorkingMemory(n_units=64, n_attractors=7)
        attractor_wm2.write(items[0], importance=1.0)

        drifts_da = []
        for _ in range(30):
            maint = attractor_wm2.maintain(dopamine_level=da_level)
            drifts_da.append(maint['avg_drift'])

        avg_drift = np.mean(drifts_da)
        da_results[da_level] = avg_drift
        print(f"  DA={da_level}: avg drift = {avg_drift:.4f}")

    results['da_modulation'] = da_results

    # Test read
    print("\n[9.3] Testing memory read")
    query = items[0] + torch.randn(64) * 0.1  # Noisy query
    retrieved = attractor_wm.read(query, top_k=3)
    similarity = F.cosine_similarity(items[0].unsqueeze(0), retrieved.unsqueeze(0)).item()
    print(f"  Query-retrieved similarity: {similarity:.4f}")
    results['read_similarity'] = similarity

    print("\n✓ Experiment 9 completed")
    return results


# ══════════════════════════════════════════════════════
# Experiment 10: Thalamic ACh Gating
# ══════════════════════════════════════════════════════

def exp10_thalamic_ach_gating() -> Dict:
    """Test ACh-driven attention gating - Hirsch et al. (2018)

    Expected behavior:
    - High ACh → high gating (enhanced sensory throughput)
    - Low ACh → low gating (sensory filtering)
    - Phasic ACh bursts for salient events
    """
    print("\n" + "="*60)
    print("Experiment 10: Thalamic ACh Gating")
    print("="*60)

    results = {
        'name': 'Thalamic ACh Gating',
        'gating_levels': [],
        'phasic_response': [],
    }

    gating = ThalamicAChGating(n_senses=4, baseline_a_ch=0.5)

    # Test ACh level effects
    print("\n[10.1] Testing ACh level effects")
    ach_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    for ach in ach_levels:
        gating.a_ch_tonic = ach
        gating.a_ch_phasic = 0.0
        result = gating.update()
        results['gating_levels'].append({
            'ach': ach,
            'avg_gating': result['avg_gating'],
        })
        print(f"  ACh={ach:.1f}: avg gating = {result['avg_gating']:.4f}")

    # Test phasic ACh response
    print("\n[10.2] Testing phasic ACh response")
    gating2 = ThalamicAChGating(n_senses=4, baseline_a_ch=0.3)

    # Inject phasic burst
    gating2.inject_a_ch(0.8, phasic=True)
    phasic_trace = []
    for _ in range(20):
        result = gating2.update()
        phasic_trace.append(result['total_a_ch'])

    results['phasic_response'] = phasic_trace
    print(f"  Initial ACh: {phasic_trace[0]:.4f}")
    print(f"  Final ACh: {phasic_trace[-1]:.4f}")

    # Test sensory modulation
    print("\n[10.3] Testing sensory input modulation")
    gating3 = ThalamicAChGating(n_senses=4)
    sensory_input = torch.randn(64)

    # High ACh gating
    gating3.inject_a_ch(0.9, phasic=True)
    gating3.update()
    modulated_high = gating3.modulate_input(sensory_input, sense_idx=0)

    # Low ACh gating
    gating3.inject_a_ch(0.1, phasic=False)
    gating3.update()
    modulated_low = gating3.modulate_input(sensory_input, sense_idx=0)

    print(f"  High ACh modulation norm: {modulated_high.norm().item():.4f}")
    print(f"  Low ACh modulation norm: {modulated_low.norm().item():.4f}")
    results['sensory_modulation'] = {
        'high_ach': modulated_high.norm().item(),
        'low_ach': modulated_low.norm().item(),
    }

    print("\n✓ Experiment 10 completed")
    return results


# ══════════════════════════════════════════════════════
# Experiment 11: Theta-Gamma Coupling
# ══════════════════════════════════════════════════════

def exp11_theta_gamma_coupling() -> Dict:
    """Test phase-amplitude coupling - Jensen & Tesche (2002)

    Expected behavior:
    - Theta phase modulates gamma amplitude
    - High DA strengthens coupling
    - Coupling supports temporal coding
    """
    print("\n" + "="*60)
    print("Experiment 11: Theta-Gamma Coupling")
    print("="*60)

    results = {
        'name': 'Theta-Gamma Coupling',
        'phase_amplitude': [],
        'da_modulation': [],
    }

    tg = ThetaGammaCoupling(n_neurons=64)

    # Test phase-amplitude coupling
    print("\n[11.1] Testing phase-amplitude coupling")
    phases = []
    gamma_amps = []
    for step in range(100):
        result = tg.step(dt=0.01, dopamine_level=0.5)
        phases.append(result['theta_phase'])
        gamma_amps.append(result['gamma_amplitude'])

    results['phase_amplitude'] = {
        'phases': phases,
        'gamma_amps': gamma_amps,
    }
    print(f"  Theta frequency: {tg.theta_freq} Hz")
    print(f"  Gamma frequency: {tg.gamma_freq} Hz")
    print(f"  Gamma amplitude range: [{min(gamma_amps):.3f}, {max(gamma_amps):.3f}]")

    # Test DA modulation of coupling
    print("\n[11.2] Testing DA modulation")
    da_levels = [0.2, 0.5, 0.8]
    coupling_by_da = {}

    for da in da_levels:
        tg2 = ThetaGammaCoupling(n_neurons=64)
        coupling_strengths = []
        for _ in range(50):
            result = tg2.step(dt=0.01, dopamine_level=da)
            coupling_strengths.append(result['coupling_strength'])
        avg_coupling = np.mean(coupling_strengths)
        coupling_by_da[da] = avg_coupling
        print(f"  DA={da}: avg coupling = {avg_coupling:.4f}")

    results['da_modulation'] = coupling_by_da

    # Test coupling statistics
    print("\n[11.3] Testing coupling statistics")
    stats = tg.get_coupling_stats()
    print(f"  Theta-gamma coupling: {stats.get('theta_gamma_coupling', 0):.4f}")
    results['stats'] = stats

    print("\n✓ Experiment 11 completed")
    return results


# ══════════════════════════════════════════════════════
# Experiment 12: NMDA-dependent LTP/LTD + Synaptic Scaling
# ══════════════════════════════════════════════════════

def exp12_nmda_plasticity() -> Dict:
    """Test Ca-threshold LTP/LTD and homeostatic scaling - Malenka & Bear (2004)

    Expected behavior:
    - High Ca²⁺ → LTP
    - Low Ca²⁺ → LTD
    - Synaptic scaling maintains total strength
    """
    print("\n" + "="*60)
    print("Experiment 12: NMDA-dependent LTP/LTD + Synaptic Scaling")
    print("="*60)

    results = {
        'name': 'NMDA Plasticity + Scaling',
        'ltp_ltd': [],
        'scaling': [],
    }

    # Test NMDA-dependent plasticity
    print("\n[12.1] Testing NMDA-dependent LTP/LTD")
    nmda = NMDADependentPlasticity(ca_threshold_ltp=0.5, ca_threshold_ltd=0.2)

    # Simulate various Ca²⁺ levels
    ca_levels = np.linspace(0.0, 1.0, 20)
    plasticity_directions = []

    for glutamate in np.linspace(0.0, 1.0, 10):
        for depol in np.linspace(0.0, 1.0, 10):
            ca = nmda.compute_ca_influx(glutamate, depol)
            direction, magnitude = nmda.compute_plasticity_direction()
            if direction != 'none':
                plasticity_directions.append({
                    'ca': ca,
                    'direction': direction,
                    'magnitude': magnitude,
                })

    # Test specific cases
    nmda2 = NMDADependentPlasticity()

    # High glutamate + high depolarization → LTP
    ca_high = nmda2.compute_ca_influx(0.9, 0.9)
    dir_high, mag_high = nmda2.compute_plasticity_direction()
    print(f"  High Glu+Depol: Ca={ca_high:.3f} → {dir_high} (mag={mag_high:.4f})")

    # Medium → LTD
    nmda2.decay()
    nmda2.decay()
    ca_med = nmda2.compute_ca_influx(0.3, 0.3)
    dir_med, mag_med = nmda2.compute_plasticity_direction()
    print(f"  Medium Glu+Depol: Ca={ca_med:.3f} → {dir_med} (mag={mag_med:.4f})")

    results['ltp_ltd'] = {
        'high_case': {'ca': ca_high, 'direction': dir_high, 'magnitude': mag_high},
        'medium_case': {'ca': ca_med, 'direction': dir_med, 'magnitude': mag_med},
    }

    # Test synaptic scaling
    print("\n[12.2] Testing synaptic scaling")
    scaling = SynapticScaling(target_strength=0.5, scaling_rate=0.01)

    # Create synapses with varying strengths
    synapses = [Synapse(i, i+1, np.random.uniform(0.1, 1.5)) for i in range(20)]

    initial_strength = np.mean([s.weight for s in synapses])
    print(f"  Initial avg strength: {initial_strength:.4f}")

    scaling_factors = []
    for _ in range(50):
        factor = scaling.apply_scaling(synapses)
        scaling_factors.append(factor)

    final_strength = np.mean([s.weight for s in synapses])
    print(f"  Final avg strength: {final_strength:.4f}")
    print(f"  Target strength: {scaling.target_strength}")

    results['scaling'] = {
        'initial_strength': initial_strength,
        'final_strength': final_strength,
        'target': scaling.target_strength,
        'scaling_factors': scaling_factors,
    }

    # Test homeostatic regulation
    print("\n[12.3] Testing homeostatic regulation")
    # Create very strong synapses
    strong_synapses = [Synapse(i, i+1, 1.5) for i in range(20)]
    total_before = sum(s.weight for s in strong_synapses)

    scaling2 = SynapticScaling(target_strength=0.5)
    for _ in range(20):
        scaling2.apply_scaling(strong_synapses)

    total_after = sum(s.weight for s in strong_synapses)
    print(f"  Total strength before: {total_before:.2f}")
    print(f"  Total strength after: {total_after:.2f}")
    print(f"  Reduction: {(total_before - total_after) / total_before * 100:.1f}%")

    results['homeostatic'] = {
        'before': total_before,
        'after': total_after,
    }

    print("\n✓ Experiment 12 completed")
    return results


# ══════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════

def run_all_experiments() -> Dict[str, Dict]:
    """Run all 12 experiments and return results"""
    print("\n" + "="*60)
    print("RUNNING ALL NEURAL MODEL IMPROVEMENT EXPERIMENTS")
    print("="*60)

    all_results = {}

    experiments = [
        ("exp1_stdp_plasticity", exp1_stdp_plasticity),
        ("exp2_dopamine_tonic_phasic", exp2_dopamine_tonic_phasic),
        ("exp3_d2_receptor_occupancy", exp3_d2_receptor_occupancy),
        ("exp4_lif_refractory_adaptation", exp4_lif_refractory_adaptation),
        ("exp5_sharp_wave_ripple", exp5_sharp_wave_ripple),
        ("exp6_hc_pfc_connection", exp6_hc_pfc_connection),
        ("exp7_vmpfc_extinction", exp7_vmpfc_extinction),
        ("exp8_amygdala_hippocampus", exp8_amygdala_hippocampus),
        ("exp9_attractor_working_memory", exp9_attractor_working_memory),
        ("exp10_thalamic_ach_gating", exp10_thalamic_ach_gating),
        ("exp11_theta_gamma_coupling", exp11_theta_gamma_coupling),
        ("exp12_nmda_plasticity", exp12_nmda_plasticity),
    ]

    for name, exp_func in experiments:
        try:
            result = exp_func()
            all_results[name] = result
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            all_results[name] = {'error': str(e)}

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)

    return all_results


if __name__ == "__main__":
    results = run_all_experiments()

    # Save results
    import json
    output_path = os.path.join(os.path.dirname(__file__), 'experiment_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
