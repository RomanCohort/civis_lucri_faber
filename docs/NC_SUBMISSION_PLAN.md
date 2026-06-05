# Neural Computation Submission Plan

## Target Journal Profile

**Journal**: Neural Computation (MIT Press)
**ISSN**: 0899-7667 (print), 1530-888X (online)
**Founded**: 1989 by Terrence Sejnowski
**Scope**: Computational neuroscience, neural networks, learning algorithms, brain modeling

### Why Neural Computation Fits

| Our Contribution | NC Interest Area |
|-----------------|------------------|
| 14-region event-driven architecture | Computational neuroscience models |
| Bio-Gating MoE routing | Learning algorithms & routing mechanisms |
| VAD + NT modulation | Neuromodulated computation |
| FLOP analysis | Computational efficiency papers |
| 7 coupling pathways | Systems neuroscience modeling |

### NC Publication Types Matching Our Work

1. **Full Research Articles** (primary target)
   - Novel computational frameworks
   - Mathematical formulations with validation
   - ~30-40 pages acceptable

2. **Letters/Short Communications** (alternative)
   - ~10 pages for focused contributions
   - Bio-Gating alone could fit here

---

## Repositioning Strategy

### Abstract Rewrite (NC Focus)

**Current (PLOS-style)**:
> "Computational psychiatry seeks to understand mental disorders through mathematical models..."

**Proposed (NC-style)**:
> "We present a neuromorphic cognitive architecture that implements neurochemically-modulated routing through a novel Bio-Gating mechanism. The architecture comprises 14 functionally-specialized modules connected via an event-driven EventBus, achieving sparse activation through selective event subscription. Bio-Gating extends standard Mixture-of-Experts routing by incorporating valence-arousal-dominance emotional states and dopamine/serotonin/norepinephrine signals into expert selection, reducing per-token computation from O(n²·d) to O(n·d) while preserving state-dependent behavioral flexibility..."

**Key shift**: Emphasize **computational contribution** first, clinical motivation second.

### Structural Changes

| Section | Current | Proposed for NC |
|---------|---------|-----------------|
| Introduction | Clinical gap → architecture | Computation gap → neuromodulation |
| Methods | Architecture + pathways | Architecture + Bio-Gating math |
| Results | Clinical behaviors | Computational properties + behaviors |
| Discussion | Clinical implications | Computational implications + applications |

### New Sections Required

1. **Bio-Gating Derivation**
   - Full mathematical derivation from standard MoE
   - Comparison with Top-K routing
   - Gradient analysis (if trainable)

2. **Computational Properties**
   - Sparse activation analysis
   - Memory complexity
   - Scalability discussion

3. **Benchmark Comparison**
   - vs Standard MoE (Switch Transformer)
   - vs Sparse MoE (Mistral)
   - vs Neuromorphic architectures (Nengo, LIDA)

---

## Additional Experiments for NC

### Required (NC standard)

| Experiment | Purpose | Estimated effort |
|------------|---------|------------------|
| Routing efficiency benchmark | Compare FLOPs vs MoE baselines | 1 day |
| Sparse activation rate | Measure actual activation percentage | 2 hours |
| Parameter sensitivity | Systematic coefficient sweep | 3 days |

### Optional (strengthens paper)

| Experiment | Purpose | Estimated effort |
|------------|---------|------------------|
| Trainable Bio-Gating | Show gradient-based learning | 5 days |
| Noise robustness | Test under perturbation | 2 days |
| Memory comparison | vs 7-slot WM baselines | 2 days |

---

## Manuscript Statistics

| Metric | Current | NC Target |
|--------|---------|-----------|
| Word count | ~5,200 | ~8,000-10,000 acceptable |
| Equations | 4 formulas | 15-20 formulas needed |
| Figures | 6 | 8-10 recommended |
| Tables | 8 | 6-8 acceptable |
| References | 30 | 40-50 recommended |

---

## Submission Checklist for NC

| Item | Status | Action |
|------|--------|--------|
| Mathematical derivation | INCOMPLETE | Add Bio-Gating gradient analysis |
| Benchmark experiments | MISSING | Add MoE comparison table |
| Neuromorphic comparison | PARTIAL | Expand LIDA/Nengo comparison |
| Code repository | READY | GitHub link already present |
| Competing interests | READY | "None declared" |
| Data availability | READY | GitHub link |
| Supplementary | READY | Table S1 + derivation |

---

## Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Abstract rewrite | 1 hour | NC-focused abstract |
| Math expansion | 1 day | Bio-Gating derivation section |
| Benchmark experiments | 3 days | MoE comparison data |
| Structure revision | 2 days | NC-formatted manuscript |
| Final polish | 1 day | Submission-ready draft |

**Total**: ~7 days to NC-ready manuscript

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Reviewer wants trainable version | Medium | Prepare gradient derivation + simple training experiment |
| Scale questioned | High | Acknowledge 12M limitation, propose scaling path |
| Clinical relevance questioned | Low | NC accepts computational contributions without clinical validation |

---

## Next Action

1. **Rewrite abstract** with NC focus (computational contribution first)
2. **Add Bio-Gating mathematical derivation** section
3. **Add MoE benchmark comparison** table
4. **Adjust introduction** to emphasize computation gap

Would you like to proceed with these changes?