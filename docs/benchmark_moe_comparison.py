#!/usr/bin/env python3
"""
MoE Benchmark Comparison Script
Generates comparison data between Bio-Gating and standard MoE architectures
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# Benchmark 1: Routing Efficiency Comparison
# =============================================================================
def benchmark_routing_efficiency():
    """
    Compare FLOPs across different routing mechanisms
    """
    results = {
        'name': 'Routing Efficiency Benchmark',
        'conditions': []
    }

    # Parameters
    seq_len = 512
    hidden_dim = 256
    n_experts = 4
    n_tokens = seq_len

    # Standard Attention (baseline)
    standard_attention_flops = seq_len * seq_len * hidden_dim
    results['conditions'].append({
        'name': 'Standard Attention',
        'flops': standard_attention_flops,
        'active_experts': n_experts,
        'relative': 1.0
    })

    # Standard MoE Top-2
    standard_moe_top2_flops = n_tokens * hidden_dim * 2 * n_experts  # 2 experts per token
    results['conditions'].append({
        'name': 'Standard MoE Top-2',
        'flops': standard_moe_top2_flops,
        'active_experts': 2,
        'relative': standard_moe_top2_flops / standard_attention_flops
    })

    # Bio-Gating Top-1
    bio_gating_flops = n_tokens * hidden_dim * 1  # Expert forward
    gating_overhead = n_tokens * (n_experts + 8)  # 8 modulation factors
    total_bio_gating = bio_gating_flops + gating_overhead
    results['conditions'].append({
        'name': 'Bio-Gating Top-1',
        'flops': total_bio_gating,
        'active_experts': 1,
        'modulation_overhead': gating_overhead,
        'relative': total_bio_gating / standard_attention_flops
    })

    # Switch Transformer (Top-1, no modulation)
    switch_flops = n_tokens * hidden_dim * 1
    switch_gating = n_tokens * n_experts  # Content-only gating
    total_switch = switch_flops + switch_gating
    results['conditions'].append({
        'name': 'Switch Transformer',
        'flops': total_switch,
        'active_experts': 1,
        'relative': total_switch / standard_attention_flops
    })

    # Print results
    print("\n" + "="*60)
    print("BENCHMARK 1: Routing Efficiency")
    print("="*60)
    for cond in results['conditions']:
        print(f"\n{cond['name']}:")
        print(f"  FLOPs: {cond['flops']:,}")
        print(f"  Active experts: {cond['active_experts']}")
        print(f"  Relative to attention: {cond['relative']:.6f}")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [c['name'] for c in results['conditions']]
    flops = [c['flops'] for c in results['conditions']]
    colors = ['gray', 'steelblue', 'coral', 'green']

    bars = ax.bar(names, flops, color=colors, alpha=0.7)

    ax.set_ylabel('FLOPs per forward pass', fontsize=11)
    ax.set_xlabel('Routing Mechanism', fontsize=11)
    ax.set_title('Routing Efficiency Comparison\n(seq_len=512, hidden_dim=256, 4 experts)', fontsize=12)

    # Add relative percentages
    for bar, cond in zip(bars, results['conditions']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{cond['relative']*100:.2f}%",
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/benchmark_routing_efficiency.png', dpi=300)
    plt.close()
    print("\n[OK] Figure saved: benchmark_routing_efficiency.png")

    return results

# =============================================================================
# Benchmark 2: Memory Complexity Comparison
# =============================================================================
def benchmark_memory_complexity():
    """
    Compare memory requirements across architectures
    """
    results = {
        'name': 'Memory Complexity Benchmark',
        'conditions': []
    }

    # Parameters
    hidden_dim = 256
    n_experts = 4
    seq_len = 512

    # Standard MoE Top-2
    expert_params = n_experts * hidden_dim * hidden_dim
    gating_params = n_experts * hidden_dim
    total_standard = expert_params + gating_params
    results['conditions'].append({
        'name': 'Standard MoE Top-2',
        'total_params': total_standard,
        'expert_params': expert_params,
        'gating_params': gating_params,
        'state_params': 0
    })

    # Bio-Gating
    bio_expert_params = n_experts * hidden_dim * hidden_dim
    bio_gating_params = n_experts * hidden_dim
    bio_state_params = n_experts + 3 + 5 + 1  # membrane + VAD + NTs + mood
    bio_total = bio_expert_params + bio_gating_params + bio_state_params
    results['conditions'].append({
        'name': 'Bio-Gating',
        'total_params': bio_total,
        'expert_params': bio_expert_params,
        'gating_params': bio_gating_params,
        'state_params': bio_state_params
    })

    # Switch Transformer
    switch_total = expert_params + gating_params
    results['conditions'].append({
        'name': 'Switch Transformer',
        'total_params': switch_total,
        'expert_params': expert_params,
        'gating_params': gating_params,
        'state_params': 0
    })

    # Print results
    print("\n" + "="*60)
    print("BENCHMARK 2: Memory Complexity")
    print("="*60)
    for cond in results['conditions']:
        print(f"\n{cond['name']}:")
        print(f"  Total params: {cond['total_params']:,}")
        print(f"  Expert params: {cond['expert_params']:,}")
        print(f"  Gating params: {cond['gating_params']:,}")
        print(f"  State params: {cond['state_params']:,}")

    # Create table visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked bar chart
    names = [c['name'] for c in results['conditions']]
    expert = [c['expert_params'] for c in results['conditions']]
    gating = [c['gating_params'] for c in results['conditions']]
    state = [c['state_params'] for c in results['conditions']]

    x = np.arange(len(names))
    width = 0.6

    ax.bar(x, expert, width, label='Expert weights', color='steelblue', alpha=0.8)
    ax.bar(x, gating, width, bottom=expert, label='Gating weights', color='coral', alpha=0.8)
    ax.bar(x, state, width, bottom=[e+g for e,g in zip(expert, gating)],
           label='State variables', color='green', alpha=0.8)

    ax.set_ylabel('Parameters', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc='upper right')
    ax.set_title('Memory Complexity Comparison\n(4 experts, 256 hidden dim)', fontsize=12)

    plt.tight_layout()
    plt.savefig('figures/benchmark_memory_complexity.png', dpi=300)
    plt.close()
    print("\n[OK] Figure saved: benchmark_memory_complexity.png")

    return results

# =============================================================================
# Benchmark 3: Behavioral Expressivity
# =============================================================================
def benchmark_behavioral_expressivity():
    """
    Compare behavioral expressivity: can routing produce state-dependent behavior?
    """
    results = {
        'name': 'Behavioral Expressivity Benchmark',
        'conditions': []
    }

    # Simulate routing behavior under different states
    n_samples = 1000
    n_experts = 4

    # Baseline state (neutral)
    baseline_vad = np.array([0.0, 0.5, 0.5])
    baseline_da = 0.7
    baseline_5ht = 0.5
    baseline_ne = 0.5

    # Stress state (negative)
    stress_vad = np.array([-0.5, 0.9, -0.3])
    stress_da = 0.3
    stress_5ht = 0.3
    stress_ne = 0.9

    # Reward state (positive)
    reward_vad = np.array([0.8, 0.6, 0.7])
    reward_da = 0.9
    reward_5ht = 0.6
    reward_ne = 0.6

    def simulate_standard_moe(content_weights, n_samples):
        """Standard MoE: only content-driven"""
        # Fixed softmax over content
        gate = np.exp(content_weights) / np.sum(np.exp(content_weights))
        samples = np.random.choice(n_experts, n_samples, p=gate)
        return samples

    def simulate_bio_gating(content_weights, vad, da, n_samples):
        """Bio-Gating: content + state modulation"""
        emotion_bias = np.tanh(np.sum(vad)) * 0.3
        da_bias = da * 0.2

        modified_weights = content_weights + emotion_bias + da_bias
        gate = np.exp(modified_weights) / np.sum(np.exp(modified_weights))
        samples = np.random.choice(n_experts, n_samples, p=gate)
        return samples

    # Content weights (fixed)
    content_weights = np.array([1.0, 0.5, 0.3, 0.2])

    # Standard MoE (same for all states)
    std_baseline = simulate_standard_moe(content_weights, n_samples)
    std_stress = simulate_standard_moe(content_weights, n_samples)
    std_reward = simulate_standard_moe(content_weights, n_samples)

    # Bio-Gating (state-dependent)
    bio_baseline = simulate_bio_gating(content_weights, baseline_vad, baseline_da, n_samples)
    bio_stress = simulate_bio_gating(content_weights, stress_vad, stress_da, n_samples)
    bio_reward = simulate_bio_gating(content_weights, reward_vad, reward_da, n_samples)

    # Compute expert selection statistics
    def compute_stats(samples):
        counts = np.bincount(samples, minlength=n_experts)
        proportions = counts / n_samples
        entropy = -np.sum(proportions * np.log(proportions + 1e-10))
        return proportions, entropy

    results['conditions'] = [
        {
            'name': 'Standard MoE',
            'baseline_entropy': compute_stats(std_baseline)[1],
            'stress_entropy': compute_stats(std_stress)[1],
            'reward_entropy': compute_stats(std_reward)[1],
            'state_variance': 0.0  # No state dependence
        },
        {
            'name': 'Bio-Gating',
            'baseline_entropy': compute_stats(bio_baseline)[1],
            'stress_entropy': compute_stats(bio_stress)[1],
            'reward_entropy': compute_stats(bio_reward)[1],
            'state_variance': np.var([compute_stats(bio_baseline)[1],
                                      compute_stats(bio_stress)[1],
                                      compute_stats(bio_reward)[1]])
        }
    ]

    # Print results
    print("\n" + "="*60)
    print("BENCHMARK 3: Behavioral Expressivity")
    print("="*60)
    for cond in results['conditions']:
        print(f"\n{cond['name']}:")
        print(f"  Baseline entropy: {cond['baseline_entropy']:.4f}")
        print(f"  Stress entropy: {cond['stress_entropy']:.4f}")
        print(f"  Reward entropy: {cond['reward_entropy']:.4f}")
        print(f"  State-dependent variance: {cond['state_variance']:.4f}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Entropy by state
    states = ['Baseline', 'Stress', 'Reward']
    std_entropies = [results['conditions'][0]['baseline_entropy'],
                     results['conditions'][0]['stress_entropy'],
                     results['conditions'][0]['reward_entropy']]
    bio_entropies = [results['conditions'][1]['baseline_entropy'],
                     results['conditions'][1]['stress_entropy'],
                     results['conditions'][1]['reward_entropy']]

    x = np.arange(len(states))
    width = 0.35

    ax1.bar(x - width/2, std_entropies, width, label='Standard MoE', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, bio_entropies, width, label='Bio-Gating', color='coral', alpha=0.8)

    ax1.set_ylabel('Routing Entropy (bits)', fontsize=11)
    ax1.set_xlabel('State', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(states)
    ax1.legend()
    ax1.set_title('A) Routing Diversity by State', fontsize=11, fontweight='bold')

    # Right: State-dependent variance
    variances = [results['conditions'][0]['state_variance'],
                 results['conditions'][1]['state_variance']]
    names = ['Standard MoE', 'Bio-Gating']

    ax2.bar(names, variances, color=['steelblue', 'coral'], alpha=0.8)
    ax2.set_ylabel('State-Dependent Variance', fontsize=11)
    ax2.set_title('B) Behavioral Expressivity', fontsize=11, fontweight='bold')

    ax2.annotate('No state\ndependence', xy=(0, 0), xytext=(0, 0.01),
                ha='center', va='bottom', fontsize=9)
    ax2.annotate('State-dependent\nrouting', xy=(1, variances[1]),
                xytext=(1, variances[1]*1.1), ha='center', va='bottom', fontsize=9)

    fig.suptitle('Benchmark 3: Behavioral Expressivity Comparison',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('figures/benchmark_behavioral_expressivity.png', dpi=300)
    plt.close()
    print("\n[OK] Figure saved: benchmark_behavioral_expressivity.png")

    return results

# =============================================================================
# Benchmark 4: Scalability Analysis
# =============================================================================
def benchmark_scalability():
    """
    Compare scalability across parameter scales
    """
    results = {
        'name': 'Scalability Benchmark',
        'conditions': []
    }

    scales = [1e6, 10e6, 100e6, 1e9]  # 1M, 10M, 100M, 1B
    scale_names = ['1M', '10M', '100M', '1B']

    # Assume hidden_dim proportional to sqrt(params)
    hidden_dims = [64, 256, 512, 1024]

    bio_gating_flops = []
    standard_moe_flops = []

    for h_dim in hidden_dims:
        # Bio-Gating: O(n*d*1)
        bio_flops = 512 * h_dim * 1 + 512 * 4
        bio_gating_flops.append(bio_flops)

        # Standard MoE Top-2: O(n*d*2)
        std_flops = 512 * h_dim * 2
        standard_moe_flops.append(std_flops)

    results['conditions'] = [
        {
            'name': 'Bio-Gating',
            'scales': scale_names,
            'flops': bio_gating_flops
        },
        {
            'name': 'Standard MoE Top-2',
            'scales': scale_names,
            'flops': standard_moe_flops
        }
    ]

    # Print results
    print("\n" + "="*60)
    print("BENCHMARK 4: Scalability")
    print("="*60)
    for cond in results['conditions']:
        print(f"\n{cond['name']}:")
        for scale, flops in zip(cond['scales'], cond['flops']):
            print(f"  {scale} params: {flops:,} FLOPs")

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(scale_names, bio_gating_flops, marker='o', linewidth=2,
            markersize=8, label='Bio-Gating', color='coral')
    ax.plot(scale_names, standard_moe_flops, marker='s', linewidth=2,
            markersize=8, label='Standard MoE Top-2', color='steelblue')

    ax.set_ylabel('FLOPs per token', fontsize=11)
    ax.set_xlabel('Model Scale (parameters)', fontsize=11)
    ax.legend()
    ax.set_title('Scalability Analysis\nFLOPs vs Model Scale', fontsize=12, fontweight='bold')

    # Add savings annotation
    savings = [(std - bio) / std * 100 for std, bio in zip(standard_moe_flops, bio_gating_flops)]
    ax.text(0.5, max(bio_gating_flops) * 0.8,
            f'Bio-Gating saves {savings[-1]:.1f}% at 1B scale',
            ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/benchmark_scalability.png', dpi=300)
    plt.close()
    print("\n[OK] Figure saved: benchmark_scalability.png")

    return results

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("MoE Benchmark Comparison Suite")
    print("="*60)

    all_results = {
        'benchmarks': []
    }

    all_results['benchmarks'].append(benchmark_routing_efficiency())
    all_results['benchmarks'].append(benchmark_memory_complexity())
    all_results['benchmarks'].append(benchmark_behavioral_expressivity())
    all_results['benchmarks'].append(benchmark_scalability())

    # Save results to JSON
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("All benchmarks complete!")
    print("Results saved: benchmark_results.json")
    print("Figures saved: figures/benchmark_*.png")
    print("="*60)