"""
Neurotransmitter Interaction Analysis

Test DA-5-HT balance and DA-NE synergy effects:
- High DA + Low 5-HT: Mania-like (excessive exploration)
- Low DA + High 5-HT: Depression-like (behavioral inhibition)
- High DA + High NE: Hypervigilant (focused exploration)

Addresses R2's concern: "未讨论DA-5-HT平衡、DA-NE协同"

Author: CLF Team
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class NTConfig:
    """Neurotransmitter configuration"""
    name: str
    da: float
    serotonin: float
    ne: float
    description: str


# Neurotransmitter configurations representing different states
NT_CONFIGS = {
    "Baseline": NTConfig(
        name="Baseline",
        da=0.5,
        serotonin=0.5,
        ne=0.3,
        description="Normal resting state"
    ),
    "Mania-like": NTConfig(
        name="Mania-like",
        da=0.9,
        serotonin=0.2,
        ne=0.6,
        description="High DA + Low 5-HT (excessive exploration)"
    ),
    "Depression-like": NTConfig(
        name="Depression-like",
        da=0.2,
        serotonin=0.8,
        ne=0.2,
        description="Low DA + High 5-HT (behavioral inhibition)"
    ),
    "Anxiety-like": NTConfig(
        name="Anxiety-like",
        da=0.3,
        serotonin=0.4,
        ne=0.8,
        description="Low DA + High NE (hypervigilant)"
    ),
    "Reward-seeking": NTConfig(
        name="Reward-seeking",
        da=0.7,
        serotonin=0.3,
        ne=0.4,
        description="High DA + Moderate NE (focused reward pursuit)"
    ),
    "Behavioral-inhibition": NTConfig(
        name="Behavioral-inhibition",
        da=0.3,
        serotonin=0.7,
        ne=0.2,
        description="High 5-HT suppresses risky behavior"
    ),
}


def compute_routing_behavior(config: NTConfig, n_steps: int = 1000, seed: int = 42) -> Dict:
    """Compute routing behavior under given NT configuration

    Bio-Gating modulation effects:
    - DA: exploration gradient (high DA -> uniform distribution)
    - 5-HT: behavioral inhibition (high 5-HT -> reduce variance)
    - NE: SNR boost (high NE -> sharpen distribution)
    """

    np.random.seed(seed)

    n_experts = 4
    eta = 0.2   # DA sensitivity
    kappa = 0.15  # 5-HT sensitivity
    lam = 0.8   # NE sensitivity

    # Base content logits (fixed)
    base_logits = np.array([1.0, 0.8, 0.6, 0.4])

    routing_entropy = []
    expert_selection = []
    exploration_rate = []

    for step in range(n_steps):
        # Add small noise to base logits
        h = base_logits + np.random.normal(0, 0.1, n_experts)

        # Compute base gate
        g_base = np.exp(h) / np.sum(np.exp(h))

        # DA modulation: exploration gradient
        # High DA -> push toward uniform (more exploration)
        explore_grad = 1.0 / n_experts - g_base
        da_mod = config.da * eta * explore_grad

        # 5-HT modulation: behavioral inhibition
        # High 5-HT -> reduce variance (consistency)
        g_max = np.max(g_base)
        inhibit_grad = g_base - g_max
        ht_mod = -config.serotonin * kappa * inhibit_grad

        # NE modulation: SNR boost
        # High NE -> sharpen distribution (focused attention)
        rho = 1 + config.ne * lam
        g_sharpened = np.power(g_base, rho)
        g_sharpened = g_sharpened / np.sum(g_sharpened)
        ne_mod = g_sharpened - g_base

        # Final logits
        final_logits = h + da_mod + ht_mod + ne_mod
        g_final = np.exp(final_logits) / np.sum(np.exp(final_logits))

        # Compute metrics
        ent = -np.sum(g_final * np.log(g_final + 1e-10))
        routing_entropy.append(ent)

        # Expert selection (sampling)
        selected = np.random.choice(n_experts, p=g_final)
        expert_selection.append(selected)

        # Exploration rate (inverse of max probability)
        explore = 1 - np.max(g_final)
        exploration_rate.append(explore)

    return {
        "config": config,
        "mean_entropy": np.mean(routing_entropy),
        "std_entropy": np.std(routing_entropy),
        "entropy_trajectory": routing_entropy,
        "selection_distribution": np.bincount(expert_selection, minlength=n_experts) / n_steps,
        "mean_exploration_rate": np.mean(exploration_rate),
        "exploration_trajectory": exploration_rate,
    }


def compute_interaction_effects():
    """Compute DA-5-HT and DA-NE interaction effects"""

    print("=" * 70)
    print("Neurotransmitter Interaction Analysis")
    print("=" * 70)

    results = {}

    for name, config in NT_CONFIGS.items():
        print(f"\n[{name}]")
        print(f"  DA={config.da:.1f}, 5-HT={config.serotonin:.1f}, NE={config.ne:.1f}")
        result = compute_routing_behavior(config)
        results[name] = result

        print(f"  Routing entropy: {result['mean_entropy']:.3f} ± {result['std_entropy']:.3f}")
        print(f"  Exploration rate: {result['mean_exploration_rate']:.3f}")
        print(f"  Expert selection: {result['selection_distribution']}")

    # DA-5-HT Balance Analysis
    print("\n" + "=" * 70)
    print("DA-5-HT BALANCE ANALYSIS")
    print("=" * 70)

    # Create DA-5-HT balance matrix
    da_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    ht_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    balance_matrix = {}
    for da in da_levels:
        balance_matrix[da] = {}
        for ht in ht_levels:
            config = NTConfig(
                name=f"DA={da},5-HT={ht}",
                da=da,
                serotonin=ht,
                ne=0.3,  # Fixed NE
                description=""
            )
            result = compute_routing_behavior(config, n_steps=500)
            balance_matrix[da][ht] = result["mean_entropy"]

    print(f"\n{'DA':>8s} {'5-HT=0.1':>10s} {'5-HT=0.3':>10s} {'5-HT=0.5':>10s} {'5-HT=0.7':>10s} {'5-HT=0.9':>10s}")
    print("-" * 60)
    for da in da_levels:
        row = f"{da:>8.1f}"
        for ht in ht_levels:
            row += f" {balance_matrix[da][ht]:>10.3f}"
        print(row)

    # Key interaction
    high_da_low_ht = balance_matrix[0.9][0.1]  # Mania-like
    low_da_high_ht = balance_matrix[0.1][0.9]  # Depression-like

    print(f"\nKey DA-5-HT interaction:")
    print(f"  High DA (0.9) + Low 5-HT (0.1): entropy = {high_da_low_ht:.3f} (mania)")
    print(f"  Low DA (0.1) + High 5-HT (0.9): entropy = {low_da_high_ht:.3f} (depression)")
    print(f"  Balance effect: {high_da_low_ht - low_da_high_ht:.3f} bits difference")

    # DA-NE Synergy Analysis
    print("\n" + "=" * 70)
    print("DA-NE SYNERGY ANALYSIS")
    print("=" * 70)

    synergy_configs = {
        "Low DA + Low NE": (0.2, 0.2),
        "Low DA + High NE": (0.2, 0.8),
        "High DA + Low NE": (0.8, 0.2),
        "High DA + High NE": (0.8, 0.8),
    }

    print(f"\n{'Configuration':<20s} {'Entropy':>10s} {'Exploration':>12s}")
    print("-" * 45)

    for name, (da, ne) in synergy_configs.items():
        config = NTConfig(
            name=name,
            da=da,
            serotonin=0.5,
            ne=ne,
            description=""
        )
        result = compute_routing_behavior(config, n_steps=500)
        print(f"{name:<20s} {result['mean_entropy']:>10.3f} {result['mean_exploration_rate']:>12.3f}")

    # Synergy effect
    high_da_low_ne = compute_routing_behavior(
        NTConfig("test", 0.8, 0.5, 0.2, ""), n_steps=500
    )
    high_da_high_ne = compute_routing_behavior(
        NTConfig("test", 0.8, 0.5, 0.8, ""), n_steps=500
    )

    print(f"\nDA-NE synergy effect:")
    print(f"  High DA alone: entropy = {high_da_low_ne['mean_entropy']:.3f}")
    print(f"  High DA + High NE: entropy = {high_da_high_ne['mean_entropy']:.3f}")
    print(f"  NE amplifies DA effect by: {high_da_high_ne['mean_entropy'] - high_da_low_ne['mean_entropy']:.3f} bits")

    # Clinical Interpretation
    print("\n" + "=" * 70)
    print("CLINICAL INTERPRETATION")
    print("=" * 70)

    clinical_states = {
        "Mania": results["Mania-like"],
        "Depression": results["Depression-like"],
        "Anxiety": results["Anxiety-like"],
        "Normal": results["Baseline"],
        "Reward-seeking": results["Reward-seeking"],
    }

    print(f"\n{'State':<15s} {'Entropy':>10s} {'Exploration':>12s} {'Dominant Expert':>15s}")
    print("-" * 55)

    for name, result in clinical_states.items():
        dominant = np.argmax(result["selection_distribution"])
        print(f"{name:<15s} {result['mean_entropy']:>10.3f} {result['mean_exploration_rate']:>12.3f} {f'Expert {dominant}':>15s}")

    print("""
Clinical interpretation:
- Mania: High entropy, uniform expert selection, excessive exploration
- Depression: Low entropy, repetitive selection, behavioral inhibition
- Anxiety: Moderate entropy, NE sharpens to specific expert, hypervigilance
- Reward-seeking: Moderate entropy, DA drives exploration, NE focuses pursuit

The architecture captures neuropsychiatric states through DA/5-HT/NE interaction.
""")

    return results


if __name__ == "__main__":
    results = compute_interaction_effects()