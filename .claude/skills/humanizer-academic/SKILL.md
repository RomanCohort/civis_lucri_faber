---
name: humanizer-academic
description: Academic text humanization - reduce AI-generated patterns in scholarly writing while maintaining rigor
version: 1.0.0
author: Civis Lucri-Faber
invocation: /humanizer_academic
---

# Humanizer Academic Skill

Reduces "AI-generated feel" in academic writing while preserving scholarly rigor.

## When to Use

- Text suspected of being flagged as AI-generated
- Writing feels overly formulaic or mechanical
- Need to make academic prose more natural
- Pre-submission manuscript refinement

## Usage

```
/humanizer_academic [file_path] [mode]
```

**Modes**:
- `analyze` - Identify AI-pattern markers (default)
- `rewrite` - Apply humanization edits
- `report` - Generate detailed assessment report

## AI-Generated Pattern Markers

### 1. Structural Patterns
- Excessive use of "First, ... Second, ... Third, ..."
- Overuse of "However, ..." sentence starters
- Formulaic transition phrases: "Furthermore, ...", "In addition, ..."
- Repetitive paragraph structures

### 2. Lexical Patterns
- Overuse of passive constructions (>30%)
- Excessive nominalization ("the implementation of" vs "implementing")
- Hedging density > 5 per 1000 words ("may", "could", "possibly")
- Prefabricated phrases: "It is worth noting that..."
- Overuse of "significantly" without statistical context

### 3. Semantic Patterns
- Abstract-to-abstract transitions (no concrete examples)
- Missing author voice (no "we", "our approach")
- Perfectly balanced pros/cons lists
- Generic conclusions without specific claims

### 4. Statistical Patterns (AI detectors focus on these)
- Burstiness score: AI text has uniform sentence length distribution
- Perplexity score: AI text has predictable word choices
- Vocabulary entropy: AI text uses "safe" academic vocabulary

## Humanization Strategies

### Strategy 1: Sentence Variation
```
Before: The results demonstrate significant improvement. Furthermore, 
the analysis reveals important patterns.

After: Our results show meaningful gains—the improvement wasn't 
incremental but substantial. We also noticed some unexpected patterns.
```

### Strategy 2: Concrete Examples
```
Before: This approach provides substantial benefits for various applications.

After: This approach helped us cut inference time from 3.2s to 0.8s 
on the CASME II dataset. Similar gains appeared in our cross-dataset tests.
```

### Strategy 3: Author Voice Injection
```
Before: It is hypothesized that the mechanism operates through...

After: We believe the mechanism works through... (This is based on 
our experiments, not established theory.)
```

### Strategy 4: Hedging Calibration
```
Before: This may potentially suggest that the results could possibly
indicate a trend that might be significant.

After: The results suggest a clear trend (p < 0.05), though we caution
that our sample size limits generalization.
```

### Strategy 5: Breaking Formulaic Structures
```
Before: The architecture has three advantages: First, efficiency.
Second, scalability. Third, interpretability.

After: Efficiency is the architecture's main draw—it runs 3× faster 
than baselines. Scalability proved useful in our 1M-parameter tests, 
though interpretability remains a work in progress.
```

## Assessment Metrics

| Metric | AI-Suspect Range | Human-Like Range |
|--------|------------------|------------------|
| Passive voice % | >30% | 10-25% |
| Sentence length variance | <20% | 30-50% |
| Hedging density | >5/1000w | 2-4/1000w |
| First-person usage | 0% | 5-15% |
| Concrete examples | <1 per section | 2-5 per section |
| Formulaic transitions | >5 per page | 1-3 per page |

## Workflow

1. **Analyze**: Scan text for AI-pattern markers
2. **Score**: Calculate humanization score (0-100)
3. **Target**: Identify high-risk sections
4. **Rewrite**: Apply humanization strategies
5. **Verify**: Re-score and iterate

## Important Notes

- This skill does NOT make text less rigorous
- It does NOT add filler or padding
- It does NOT change factual content
- It makes writing MORE specific, not less
- Academic integrity is preserved

## Example Output

```markdown
## AI-Pattern Analysis Report

**File**: NC_DRAFT.md
**Score**: 72/100 (moderate AI-suspect)

### High-Risk Patterns Detected:
1. Overuse of "However," (8 instances in abstract)
2. No first-person voice in methodology section
3. Formulaic list structure in Section 2.4
4. Hedging density: 7.2 per 1000 words (above threshold)

### Recommended Edits:
- [L78] Replace "It is demonstrated" with "We show"
- [L163] Break formulaic list into narrative paragraph
- [L328] Add specific performance numbers
- [L408] Reduce hedging: "may potentially" → "suggests"
```

## Integration

Works with:
- `/sparc` methodology for revision workflow
- `/github-code-review` for manuscript review
- `/reasoningbank-intelligence` for pattern learning