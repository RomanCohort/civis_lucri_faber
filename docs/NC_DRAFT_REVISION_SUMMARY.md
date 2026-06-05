# NC_DRAFT.tex Revision Summary

**Revision Date**: 2026-06-05
**Based on**: 5 Reviewer Comments (A: Cognitive Architecture, B: Computational Neuroscience, C: ML/MoE, D: Psychiatry, E: Mathematics)

---

## Major Revisions

### 1. Enhanced Architecture Comparison (Reviewer A)

**Location**: After Table 1 (Architecture Comparison)

**Added**:
- Mechanism-level analysis table with routing mechanism, dynamic reconfiguration, emergent behavior source
- Detailed comparison with SOAR's impasse-driven learning and ACT-R's utility learning
- Key distinction: SOAR/ACT-R state-dependence from task performance history; Simulacrum from neurochemical state history

**Rationale**: Reviewer A noted the original comparison was superficial and incorrectly labeled SOAR as "hard-coded"

---

### 2. Parameter Sensitivity Analysis: Behavioral Subtypes (Reviewer D)

**Location**: Section 6.3, after Stress Anhedonia Results

**Added**:
- Table 9: Behavioral Subtypes from Parameter Variation
- Demonstrates resilient (α=0.2) vs vulnerable (α=0.6) phenotypes
- Shows DA-sensitive (η=0.4) vs anhedonia-prone (η=0.1) behaviors
- Qualitative patterns preserved across parameter ranges

**Rationale**: Reviewer D requested demonstration of how parameter sensitivity produces clinical subtypes

---

### 3. Ablation Experiment (Reviewer C)

**Location**: Section 6.3, after Table 9

**Added**:
- Table 10: Ablation - Routing Entropy Under Stress
- Compares Bio-Gating + modulation vs fixed coefficients vs Standard Top-1 MoE
- Quantifies modulation contribution: ΔH = 1.6 bits entropy reduction

**Rationale**: Reviewer C requested ablation to separate Top-1 contribution from state modulation contribution

---

### 4. Complete Theorem 5.5 Proof (Reviewer E)

**Location**: Section 5.3 (Convergence Analysis)

**Added**:
- Full proof using continuity argument:
  1. Softmax is continuous
  2. Modulation functions are continuous
  3. Membrane potential update is continuous
  4. By sequential continuity: s_t → s* implies g(s_t) → g(s*)

**Rationale**: Reviewer E noted the original proof was incomplete ("by continuity" without elaboration)

---

### 5. Base Gate Approximation Error Analysis (Reviewer E)

**Location**: Section 5.3, after Theorem 5.5 proof

**Added**:
- Remark on approximation error bound: ||g - g^base|| ≤ (η + κ)/n_e × max(DA, 5-HT)
- Maximum error ≈ 8.75% for typical parameters
- Fixed-point iteration convergence condition: |λ_max((I+A))| × (η + κ) < 1

**Rationale**: Reviewer E requested discussion of approximation error and convergence conditions

---

### 6. Subscription Count Verification (Reviewer E)

**Location**: Lemma 3.5 (Sparsity Bound)

**Added**:
- Table: Subscription Counts by Region
- Explicit summation: 12+8+7+7+7+7+6+6+4+6+3+3+5+5 = 77
- Per-region breakdown for verification

**Rationale**: Reviewer E requested verification of the 77 total subscription count

---

### 7. Expanded Limitations Section (Reviewers B, C, D)

**Location**: Section 7.3 (Limitations)

**Original**: 3 brief points

**Revised**: 6 detailed points with sub-items:

1. **Scale and Trainability**: 12M parameters, scaling concerns
2. **Biological Grounding Simplifications**:
   - Scalar neurotransmitter values (ignore spatial gradients, receptor subtypes, temporal dynamics)
   - Linear coupling vs nonlinear threshold dynamics
   - Discrete brain regions (e.g., Basal Ganglia ignores striosome/matrix)
3. **Sparsity Mechanism Difference**: Structural (EventBus) vs dynamic (biological)
4. **Behavioral Demonstration Interpretation**: Qualitative consistency, not quantitative prediction
5. **Lack of Standard Benchmarks**: No n-back, Stroop, GLUE, WikiText-103 validation
6. **Individual Differences**: Population means reported, subtypes need explicit demonstration

**Added**:
- "What This Paper Does NOT Claim" list
- "What We DO Claim" list

**Rationale**: Reviewers B, C, D all requested more honest discussion of limitations

---

### 8. Discussion Section Enhancement

**Location**: Section 8 (Discussion and Limitations)

**Added**:
- Relation to Existing Work:
  - Conditional MoE: Bio-Gating modulates on internal state, not input features
  - Reinforcement Learning: Continuous softmax perturbation vs discrete action selection
  - Computational Psychiatry Models: Multi-pathway integration vs single-mechanism models
- Expanded Future Directions with specific tasks:
  - WikiText-103, GLUE benchmarks
  - Trier Social Stress Test fMRI comparison
  - 100M+ parameter scaling

**Rationale**: Strengthened positioning relative to existing literature

---

### 9. New References

**Added**:
- `conditional_moe`: Liu et al. (2022) - Deep mixture of experts via task-dependent gate
- `huys2016`: Huys et al. (2016) - Computational psychiatry as a bridge

**Rationale**: Support new Discussion content on relation to existing work

---

## Summary Statistics

| Metric | Original | Revised |
|--------|----------|---------|
| Total lines | ~900 | 1337 |
| Tables | 11 | 13 |
| Proofs | Brief | Complete |
| Limitations | 3 points | 6 points + lists |
| References | 21 | 23 |

---

## Response to Each Reviewer

### Reviewer A (Cognitive Architecture) → Major Revision → **Addressed**
- ✅ Deeper SOAR/ACT-R comparison
- ✅ Mechanism-level distinction
- ⚠️ Standard cognitive tasks (n-back, Stroop) - noted as limitation

### Reviewer B (Computational Neuroscience) → Major Revision → **Addressed**
- ✅ Detailed limitations on biological simplifications
- ✅ Sparsity mechanism difference explained
- ⚠️ Neural data fitting - noted as future direction

### Reviewer C (ML/MoE) → Minor Revision → **Addressed**
- ✅ Ablation experiment added
- ✅ Modulation contribution quantified
- ⚠️ Standard benchmarks - noted as limitation

### Reviewer D (Psychiatry) → Major Revision → **Addressed**
- ✅ Parameter sensitivity showing behavioral subtypes
- ✅ Resilient/vulnerable phenotypes demonstrated
- ✅ Clarified "consistency" vs "validation"

### Reviewer E (Mathematics) → Accept with Minor Revision → **Addressed**
- ✅ Complete Theorem 5.5 proof
- ✅ Approximation error analysis
- ✅ Subscription count verification table

---

## Remaining Limitations (Acknowledged in Paper)

1. No standard cognitive task validation (n-back, Stroop, Tower of Hanoi)
2. No NLP benchmark validation (GLUE, WikiText-103)
3. No neural data fitting (fMRI, electrophysiology)
4. Coefficients derived from literature, not patient-fitted
5. 12M parameter scale only

These are honestly disclosed in the revised Limitations section and Future Directions.
