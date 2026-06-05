"""Deep Pharmacology Demo — Receptor PD, Addiction, Pathogen Inflammation, Symptom Tracking.

6 experimental arms:
1. Morphine chronic → withdrawal (tolerance + withdrawal demo)
2. Buspirone for GAD (receptor-level 5-HT1A partial agonism)
3. Insomnia + Suvorexant (orexin antagonist)
4. Lyme + Minocycline + Sertraline (pathogen inflammation + anti-inflammatory + antidepressant)
5. Buprenorphine MAT (opioid dependence substitution therapy)
6. Panic Disorder + Propranolol (symptom tracker demo)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import Dict

# Direct imports (avoid triggering core/__init__.py chain)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "drug_pipeline"))

from drug_registry import DRUG_REGISTRY, get_drug
from receptor_pd import (
    build_receptor_pd_targets, compute_receptor_deltas,
    aggregate_receptor_to_nt, get_receptor_time_profile,
    RECEPTOR_REGISTRY, DRUG_RECEPTOR_AFFINITY,
)
from addiction_dynamics import AddictionDynamicsEngine, AddictionProfile
from pathogen_neuroinflammation import (
    PathogenTriggeredInflammationEngine, PATHOGEN_REGISTRY,
)
from symptom_tracker import SymptomTracker, SymptomSnapshot
from sleep import OrexinSystem
from interoception import InteroceptivePredictionError


# ══════════════════════════════════════════════════════
# Helper: simulated _internal_state
# ══════════════════════════════════════════════════════

def make_baseline_state() -> Dict[str, float]:
    """Baseline internal state for a healthy individual."""
    return {
        "nt_serotonin": 0.5, "nt_dopamine": 0.5, "nt_norepinephrine": 0.5,
        "nt_gaba": 0.5, "nt_glutamate": 0.5, "nt_acetylcholine": 0.5,
        "limbic_arousal": 0.3, "prefrontal_inhibition": 0.6,
        "plasticity_bdnf": 0.5, "hormone_cortisol": 0.3,
        "hormone_oxytocin": 0.5, "brainstem_arousal_setpoint": 0.4,
        "predictive_coding_precision_mult": 0.4,
        "basal_ganglia_td_error_mult": 0.5,
        "scn_melatonin_amplitude_mult": 0.7,
        "orexin_level": 0.5, "interoceptive_pe": 0.1,
        "pathogen_load": 0.0, "bbb_disruption": 0.0,
        "craving_level": 0.0, "withdrawal_severity": 0.0,
        "tolerance_factor": 1.0,
        # Receptor keys
        "rct_5ht1a": 0.5, "rct_5ht2a": 0.5, "rct_sert": 0.5,
        "rct_d2": 0.5, "rct_dat": 0.5, "rct_gabaa": 0.5,
        "rct_nmda": 0.5, "rct_muopioid": 0.5,
        "rct_alpha1": 0.5, "rct_alpha2": 0.5, "rct_beta1": 0.5,
        "rct_orexin1": 0.5, "rct_orexin2": 0.5,
    }


# ══════════════════════════════════════════════════════
# Arm 1: Morphine Chronic → Withdrawal
# ══════════════════════════════════════════════════════

def demo_morphine_withdrawal():
    print("\n" + "=" * 70)
    print("ARM 1: Morphine Chronic -> Withdrawal (Tolerance + Withdrawal)")
    print("=" * 70)

    engine = AddictionDynamicsEngine()
    engine.register_drug("morphine", "opioid")

    state = make_baseline_state()

    # Phase 1: Chronic morphine (500 steps)
    print("\n[Phase 1] Chronic morphine 60mg x 4/day (500 steps)")
    for step in range(500):
        # Morphine PD effect: DA↑, GABA↓
        state["nt_dopamine"] = min(1.0, state["nt_dopamine"] + 0.02)
        state["nt_gaba"] = max(0.0, state["nt_gaba"] - 0.01)
        state["rct_muopioid"] = min(1.0, state["rct_muopioid"] + 0.03)

        tol_factors, wd_deltas, craving_levels = engine.step(
            drug_concentrations={"morphine": 0.7},
            drug_effects={"morphine": 0.6},
        )

        if step % 100 == 0:
            profile = engine.profiles["morphine"]
            tol = list(profile.tolerance.values())[0] if profile.tolerance else None
            tol_val = tol.downregulation_factor if tol else 1.0
            print(f"  Step {step:4d}: DA={state['nt_dopamine']:.3f} "
                  f"Tolerance(downreg)={tol_val:.3f} "
                  f"Craving={craving_levels.get('morphine', 0):.3f}")

    # Phase 2: Abrupt withdrawal (300 steps)
    print("\n[Phase 2] Abrupt withdrawal -- morphine stopped (300 steps)")
    for step in range(300):
        # Drug removed → NT levels drift back
        state["nt_dopamine"] = max(0.0, state["nt_dopamine"] - 0.005)
        state["nt_gaba"] = min(1.0, state["nt_gaba"] + 0.003)

        tol_factors, wd_deltas, craving_levels = engine.step(
            drug_concentrations={"morphine": 0.0},
            drug_effects={"morphine": 0.0},
        )

        if step % 50 == 0:
            profile = engine.profiles["morphine"]
            wd = profile.withdrawal
            cr = profile.craving
            print(f"  Step {step:4d}: DA={state['nt_dopamine']:.3f} "
                  f"Withdrawal={wd.current_severity:.3f} "
                  f"Craving={cr.current_level:.3f} "
                  f"Sensitization={cr.sensitization_factor:.3f}")


# ══════════════════════════════════════════════════════
# Arm 2: Buspirone for GAD (5-HT1A Partial Agonist)
# ══════════════════════════════════════════════════════

def demo_buspirone_gad():
    print("\n" + "=" * 70)
    print("ARM 2: Buspirone for GAD (Receptor-Level 5-HT1A Partial Agonism)")
    print("=" * 70)

    buspirone = get_drug("buspirone")
    print(f"\nDrug: {buspirone.name}")
    print(f"  Receptor targets: {list(buspirone.receptor_targets.keys())}")

    # Build receptor PD targets
    targets, rct_deltas = build_receptor_pd_targets("buspirone", drug_effect=0.5)
    print(f"\n  Receptor PD targets:")
    for t in targets:
        print(f"    {t.receptor_name}: {t.effect_type} (Ki={t.ki_nm}nM, "
              f"IA={t.intrinsic_activity}, dir={t.signal_direction:+.1f}, "
              f"wt={t.density_weight:.2f})")

    # Simulate time course: acute vs chronic
    state = make_baseline_state()
    total_steps = 28 * 24 * 6  # 28 days, 6 steps/hour
    print("\n  Time course (5-HT1A autoreceptor desensitization):")
    for day in [1, 3, 7, 14, 21, 28]:
        step_num = day * 24 * 6
        time_profile = get_receptor_time_profile("buspirone", step_num, total_steps)
        print(f"    Day {day:2d}: {time_profile}")


# ══════════════════════════════════════════════════════
# Arm 3: Insomnia + Suvorexant (Orexin Antagonist)
# ══════════════════════════════════════════════════════

def demo_insomnia_suvorexant():
    print("\n" + "=" * 70)
    print("ARM 3: Insomnia + Suvorexant (Orexin Antagonist / DORA)")
    print("=" * 70)

    orexin = OrexinSystem(baseline_orexin=0.5, circadian_coupling=0.3, gaba_inhibition=0.4)
    tracker = SymptomTracker()

    # Phase 1: Insomnia state (high orexin, low GABA)
    print("\n[Phase 1] Insomnia state: orexin overactive, GABA low")
    state = make_baseline_state()
    state["nt_gaba"] = 0.25
    state["limbic_arousal"] = 0.7
    state["orexin_level"] = 0.8
    state["nt_norepinephrine"] = 0.65
    state["scn_melatonin_amplitude_mult"] = 0.3

    for step in range(50):
        orexin_result = orexin.step(
            gaba_level=0.25,
            scn_wake_drive=0.7,  # evening but can't sleep
            stress_level=0.3,
        )
        state["orexin_level"] = orexin_result["orexin_level"]

        snap = tracker.step(state, step, time_h=22.0 + step * 0.1)

    print(f"  Orexin level: {orexin_result['orexin_level']:.3f}")
    print(f"  Insomnia detected: {snap.detected_symptoms.get('insomnia', False)}")
    print(f"  Insomnia severity: {snap.insomnia_severity:.3f}")

    # Phase 2: Suvorexant administered
    print("\n[Phase 2] Suvorexant 10mg — orexin receptor blockade")
    for step in range(50):
        orexin_result = orexin.step(
            gaba_level=0.30,  # slightly improved
            scn_wake_drive=0.3,  # nighttime
            stress_level=0.1,
            receptor_block=0.7,  # suvorexant occupancy
        )
        state["orexin_level"] = orexin_result["effective_orexin"]
        state["nt_gaba"] = min(0.5, state["nt_gaba"] + 0.003)
        state["limbic_arousal"] = max(0.3, state["limbic_arousal"] - 0.005)

        snap = tracker.step(state, 50 + step, time_h=23.0 + step * 0.1)

    print(f"  Effective orexin: {orexin_result['effective_orexin']:.3f}")
    print(f"  Insomnia detected: {snap.detected_symptoms.get('insomnia', False)}")
    print(f"  Insomnia severity: {snap.insomnia_severity:.3f}")


# ══════════════════════════════════════════════════════
# Arm 4: Lyme + Minocycline + Sertraline
# ══════════════════════════════════════════════════════

def demo_lyme_minocycline_sertraline():
    print("\n" + "=" * 70)
    print("ARM 4: Lyme Neuroborreliosis + Minocycline + Sertraline")
    print("=" * 70)

    pathogen_engine = PathogenTriggeredInflammationEngine()
    pathogen_engine.register_pathogen("lyme_neuroborreliosis")
    tracker = SymptomTracker()

    state = make_baseline_state()

    # Phase 1: Lyme infection (200 steps, no treatment)
    print("\n[Phase 1] Lyme infection -- no treatment (200 steps)")
    for step in range(200):
        damage_signal, cytokine_boost, state_deltas = pathogen_engine.step(
            treatment_efficacy={"lyme_neuroborreliosis": 0.0},
        )

        # Apply pathogen state deltas
        for k, v in state_deltas.items():
            if k in state:
                state[k] = float(np.clip(state[k] + v * 0.1, 0.0, 1.0))

        # Update pathogen-related state from engine
        for name, pstate in pathogen_engine.states.items():
            state["pathogen_load"] = pstate.load
            state["bbb_disruption"] = pstate.bbb_disruption

        if step % 50 == 0:
            snap = tracker.step(state, step, time_h=step * 0.1)
            pstate = pathogen_engine.states["lyme_neuroborreliosis"]
            print(f"  Step {step:4d}: Pathogen={pstate.load:.3f} "
                  f"BBB={pstate.bbb_disruption:.3f} "
                  f"5-HT={state['nt_serotonin']:.3f} "
                  f"DA={state['nt_dopamine']:.3f}")

    # Phase 2: Minocycline + Sertraline (300 steps)
    print("\n[Phase 2] Minocycline 100mg + Sertraline 50mg (300 steps)")
    for step in range(300):
        damage_signal, cytokine_boost, state_deltas = pathogen_engine.step(
            treatment_efficacy={"lyme_neuroborreliosis": 0.6},
        )

        # Sertraline PD: SERT inhibition -> 5-HT up
        state["nt_serotonin"] = min(1.0, state["nt_serotonin"] + 0.002)

        for k, v in state_deltas.items():
            if k in state:
                state[k] = float(np.clip(state[k] + v * 0.1, 0.0, 1.0))

        for name, pstate in pathogen_engine.states.items():
            state["pathogen_load"] = pstate.load
            state["bbb_disruption"] = pstate.bbb_disruption

        if step % 75 == 0:
            snap = tracker.step(state, 200 + step, time_h=(200 + step) * 0.1)
            pstate = pathogen_engine.states["lyme_neuroborreliosis"]
            print(f"  Step {200+step:4d}: Pathogen={pstate.load:.3f} "
                  f"5-HT={state['nt_serotonin']:.3f} "
                  f"DA={state['nt_dopamine']:.3f} "
                  f"Rumination={snap.rumination_level:.3f}")


# ══════════════════════════════════════════════════════
# Arm 5: Buprenorphine MAT (Opioid Dependence Substitution)
# ══════════════════════════════════════════════════════

def demo_buprenorphine_mat():
    print("\n" + "=" * 70)
    print("ARM 5: Buprenorphine MAT (Opioid Dependence Substitution)")
    print("=" * 70)

    engine = AddictionDynamicsEngine()
    engine.register_drug("morphine", "opioid")
    engine.register_drug("buprenorphine", "opioid")

    state = make_baseline_state()

    # Phase 1: Chronic morphine (200 steps)
    print("\n[Phase 1] Chronic morphine (200 steps)")
    for step in range(200):
        state["nt_dopamine"] = min(1.0, state["nt_dopamine"] + 0.01)
        state["rct_muopioid"] = min(1.0, state["rct_muopioid"] + 0.02)

        engine.step(
            drug_concentrations={"morphine": 0.7, "buprenorphine": 0.0},
            drug_effects={"morphine": 0.6, "buprenorphine": 0.0},
        )

        if step % 100 == 0:
            mp = engine.profiles["morphine"]
            tol = list(mp.tolerance.values())[0] if mp.tolerance else None
            tol_val = tol.downregulation_factor if tol else 1.0
            print(f"  Step {step}: Tolerance={tol_val:.3f}")

    # Phase 2: Switch to buprenorphine (300 steps)
    print("\n[Phase 2] Switch to buprenorphine 8mg (300 steps)")
    for step in range(300):
        # Buprenorphine: partial agonist → less DA surge, but prevents withdrawal
        state["nt_dopamine"] = float(np.clip(state["nt_dopamine"] - 0.002 + 0.001, 0.0, 1.0))

        engine.step(
            drug_concentrations={"morphine": 0.0, "buprenorphine": 0.4},
            drug_effects={"morphine": 0.0, "buprenorphine": 0.3},
        )

        if step % 75 == 0:
            mp = engine.profiles["morphine"]
            bp = engine.profiles["buprenorphine"]
            print(f"  Step {step}: Morphine withdrawal={mp.withdrawal.current_severity:.3f} "
                  f"Buprenorphine craving={bp.craving.current_level:.3f} "
                  f"DA={state['nt_dopamine']:.3f}")


# ══════════════════════════════════════════════════════
# Arm 6: Panic Disorder + Propranolol
# ══════════════════════════════════════════════════════

def demo_panic_propranolol():
    print("\n" + "=" * 70)
    print("ARM 6: Panic Disorder + Propranolol (Symptom Tracker Demo)")
    print("=" * 70)

    tracker = SymptomTracker()
    pe = InteroceptivePredictionError(salience_threshold=0.3)

    state = make_baseline_state()

    # Phase 1: Panic-prone state (100 steps)
    print("\n[Phase 1] Panic-prone state: NE↑, GABA↓, high arousal (100 steps)")
    panic_episodes = 0
    for step in range(100):
        # Panic-prone neurochemistry
        state["nt_norepinephrine"] = 0.7 + 0.1 * np.sin(step * 0.3)  # fluctuating
        state["nt_gaba"] = 0.3
        state["limbic_arousal"] = 0.6 + 0.15 * np.sin(step * 0.2)
        state["prefrontal_inhibition"] = 0.35

        # Interoceptive prediction error
        pe_result = pe.compute(
            actual_heart_rate=0.7 + 0.1 * np.sin(step * 0.3),
            actual_breathing_rate=0.65,
            actual_skin_conductance=0.7,
        )
        state["interoceptive_pe"] = pe_result["interoceptive_pe"]

        snap = tracker.step(state, step, time_h=step * 0.1)

        if snap.detected_symptoms.get("panic_attack", False):
            panic_episodes += 1

    print(f"  Panic episodes detected: {panic_episodes}")
    print(f"  Hypervigilance level: {snap.hypervigilance_level:.3f}")

    # Phase 2: Propranolol administered (100 steps)
    print("\n[Phase 2] Propranolol 40mg — beta-blockade (100 steps)")
    panic_episodes_post = 0
    for step in range(100):
        # Propranolol reduces NE-driven arousal
        state["nt_norepinephrine"] = max(0.3, state["nt_norepinephrine"] - 0.005)
        state["limbic_arousal"] = max(0.3, state["limbic_arousal"] - 0.003)
        state["prefrontal_inhibition"] = min(0.6, state["prefrontal_inhibition"] + 0.002)

        pe_result = pe.compute(
            actual_heart_rate=max(0.4, 0.7 - step * 0.003),
            actual_breathing_rate=max(0.4, 0.65 - step * 0.002),
            actual_skin_conductance=max(0.4, 0.7 - step * 0.003),
        )
        state["interoceptive_pe"] = pe_result["interoceptive_pe"]

        snap = tracker.step(state, 100 + step, time_h=10.0 + step * 0.1)

        if snap.detected_symptoms.get("panic_attack", False):
            panic_episodes_post += 1

    print(f"  Panic episodes post-propranolol: {panic_episodes_post}")
    print(f"  Hypervigilance level: {snap.hypervigilance_level:.3f}")
    print(f"  Reduction: {(1 - panic_episodes_post / max(1, panic_episodes)) * 100:.0f}%")


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("DEEP PHARMACOLOGY DEMO")
    print("Receptor PD | Addiction Dynamics | Pathogen Inflammation | Symptom Tracking")
    print("=" * 70)

    print("\n--- Registered Drugs ---")
    for name in sorted(DRUG_REGISTRY.keys()):
        drug = DRUG_REGISTRY[name]
        rct = list(drug.receptor_targets.keys()) if drug.receptor_targets else ["(none)"]
        print(f"  {name:15s}: {drug.drug_class:15s} risk={drug.addiction_risk:10s} "
              f"receptors={rct}")

    print(f"\n--- Receptor Registry: {len(RECEPTOR_REGISTRY)} subtypes ---")
    print(f"--- Drug-Receptor Affinity: {len(DRUG_RECEPTOR_AFFINITY)} drugs ---")

    demo_morphine_withdrawal()
    demo_buspirone_gad()
    demo_insomnia_suvorexant()
    demo_lyme_minocycline_sertraline()
    demo_buprenorphine_mat()
    demo_panic_propranolol()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
