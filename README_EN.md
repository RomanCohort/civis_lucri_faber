# Civis Lucri-Faber 🧠

> Bio-Inspired Cognitive AI Architecture | 生物启发式认知AI架构

[English](./README_EN.md) | [中文](./README.md)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

Civis Lucri-Faber (Latin "Craftsman Seeking Wealth") is a comprehensive bio-inspired AI agent system integrating **15 brain mechanisms**, **cognitive psychology**, and **adaptive pruning**.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Mechanisms](#mechanisms)
4. [Bio-Gating](#bio-gating)
5. [Cognitive Psychology](#cognitive-psychology)
6. [Pruning](#pruning)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Quick Start](#quick-start)

---

## Overview

Civis Lucri-Faber is a brain-inspired AI system that combines:

| Feature | Description |
|---------|-------------|
| **15 Brain Mechanisms** | Full brain region simulation |
| **Bio-Gating** | Emotion-dependent routing |
| **Cognitive Psychology** | Plutchik, dual-process, metacognition |
| **Adaptive Pruning** | ~65% computation savings |

### Comparison with Transformer

| Aspect | Transformer | Civis Lucri-Faber |
|--------|-------------|------------------|
| Routing | Full Attention | **Top-1 Selection** |
| Memory | O(n) Context | **7-Slot Limit** |
| Emotion | None | **VAD + Plutchik** |
| Metacognition | None | **Self-Monitoring** |
| Pruning | Manual | **Adaptive** |
| **Efficiency** | baseline | **~65% saved** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Civis Lucri-Faber                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Language  │  │  Auditory  │  │   Vision   │        │
│  │  Cortex   │  │  Cortex   │  │   Censor   │        │
│  │  (7.6M)  │  │  (1.0M)   │  │  (3.6M)    │        │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘        │
│        │             │              │                 │
│        └─────────────┼──────────────┘                 │
│                      ↓                                 │
│            ┌───────────────┐                         │
│            │ Multimodal    │                         │
│            │ Fusion        │                         │
│            └───────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Parameters

| Modality | Parameters | With Pruning |
|----------|------------|--------------|
| Language | 7.6M | ~1M |
| Auditory | 1.0M | ~250K |
| Vision | 3.6M | ~1.8M |
| **Total** | **~12M** | **~4M** |

---

## Mechanisms

### 15 Brain Mechanisms

| # | Mechanism | Component | Function |
|---|-----------|-----------|----------|
| 1 | Cochlea | AuditoryCortex | Frequency analysis |
| 2 | Inferior Colliculus | SubcorticalRelay | Gating |
| 3 | Primary Auditory (A1) | PrimaryAuditoryCortex | Sound processing |
| 4 | Ventral Stream | VentralStream | "What" pathway |
| 5 | Dorsal Stream | DorsalStream | "Where" pathway |
| 6 | Amygdala | Emotion heads | Emotion perception |
| 7 | Hippocampus | Memory systems | Context memory |
| 8 | Prefrontal Cortex | DualProcess | Planning |
| 9 | Thalamus | Censor fast path | Quick routing |
| 10 | Visual Cortex | SlowPathway | Detailed vision |
| 11-15 | ... | ... | ... |

---

## Bio-Gating

The core innovation: emotion-dependent expert routing.

```python
class BioGate(nn.Module):
    """
    Bio-Gating: Content + Membrane Potential + Emotion + Mood
    """
    def forward(self, input_emb):
        # 1. Content gating
        content_logits = self.content_gate(input_emb)
        
        # 2. Membrane potential (history)
        membrane_effect = self.membrane_potential.unsqueeze(0)
        
        # 3. Emotion (VAD)
        emotion_effect = torch.tanh(self.emotion_vector.sum()) * 0.2
        
        # 4. Mood (optimism/anxiety)
        mood_effect = self.mood.mood_affect_decision(content_logits)
        
        # Combined
        gate_logits = content_logits + membrane_effect + emotion_effect + mood_effect
        
        # Top-1 expert selection (75% compute saved)
        return gate_logits.softmax(dim=-1).argmax()
```

### Three-Layer Emotion-Cognition System

```
┌────────────────────────────────────────┐
│  Layer 3: Neuromodulator               │
│  Dopamine → Reward    Serotonin → Mood  │
├────────────────────────────────────────┤
│  Layer 2: MoodState (持久)             │
│  Optimism / Anxiety / Confidence       │
├────────────────────────────────────────┤
│  Layer 1: BioGate (即时)               │
│  Content + Membrane + Emotion        │
└────────────────────────────────────────┘
```

---

## Cognitive Psychology

### Integrated Theories

| Theory | Component | Implementation |
|--------|-----------|---------------|
| **Plutchik Emotion Wheel** | PlutchikEmotion | 8 basic emotions |
| **Dual Process** | DualProcessCognition | System 1/2 |
| **Embodied Cognition** | EmbodiedCognition | Body state affects perception |
| **Working Memory** | CognitiveLoadManager | 7±2 capacity |
| **Emotion Regulation** | EmotionRegulation | Reappraisal/Suppression |
| **Cognitive Bias** | CognitiveBias | Confirmation/Anchor |
| **Metacognition** | Metacognition | Self-monitoring |

### Example: Plutchik Emotions

```python
# 8 basic emotions
EMOTION_NAMES = ['joy', 'sadness', 'trust', 'disgust', 
               'fear', 'anger', 'surprise', 'anticipation']

# Emotion affects behavior
joy = sigmoid(plutchik.emotion_vector[0])  # joy → risk-seeking
fear = sigmoid(plutchik.emotion_vector[4])  # fear → risk-aversion
anger = sigmoid(plutchik.emotion_vector[5])  # anger → fast decisions
```

---

## Pruning

### Adaptive Pruning Mechanisms

| Type | Component | Saving |
|------|-----------|--------|
| **Expert Pruning** | DynamicExpertPruner | 75% |
| **Memory Pruning** | WorkingMemory (7 slots) | 57% |
| **Filter Pruning** | CochlearFilterPruner | 75% |
| **Synaptic Depression** | SynapticDepression | gradual |
| **Synaptic Enhancement** | OjaRule | Hebbian |

### Compute Efficiency

```python
# Before: 4 experts full computation
# After: Top-1 expert only
savings = (n_experts - top_k) / n_experts  # 75%
```

---

## Training

### Self-Supervised Learning

| Modality | Method |
|----------|--------|
| Language | Next Token Prediction |
| Auditory | SimCLR Contrastive |
| Vision | MAE Reconstruction |

### Training Scripts

```bash
# Language training
python train_language.py

# Auditory training  
python train_audio.py

# Vision training
python train_vision.py
```

---

## Evaluation

### Psychology Evaluation

```bash
python eval_psychology.py
```

Tests:
- Plutchik emotion recognition
- Dual-process task difficulty
- Metacognition self-monitoring
- Cognitive bias application
- Bio-Gating membrane potential

### Applications

```bash
# Emotional dialogue
python emotional_dialogue.py
```

---

## Quick Start

### Installation

```bash
pip install torch numpy streamlit
```

### Documentation

| File | Description |
|------|-------------|
| [README_EN.md](./README_EN.md) | Overview (English) |
| [README.md](./README.md) | 概述 (中文) |
| [docs/TECHNICAL_EN.md](./docs/TECHNICAL_EN.md) | Technical Doc (English) |
| [docs/TECHNICAL.md](./docs/TECHNICAL.md) | 技术文档 (中文) |

### Run Monitor

```bash
streamlit run monitor.py
```

### Load Models

```python
import torch
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# Language (with Bio-Gating)
lang = load('core/language_cortex.py', 'language_cortex')
model = lang.create_language_cortex(vocab_size=1000, use_parallel=False)

# Forward pass with emotion
tokens = torch.randint(0, 1000, (2, 16))
result = model(tokens, return_emotion=True)
print(f"Valence: {result['emotion_state']['valence']}")
print(f"Arousal: {result['emotion_state']['arousal']}")
```

---

## Paper Highlights

### Novel Contributions

1. **15 Brain Mechanisms**: First integrated brain-inspired architecture
2. **Bio-Gating**: Emotion-dependent routing (vs static MoE)
3. **Cognitive Psychology**: Full integration of Plutchik, dual-process, metacognition
4. **Adaptive Pruning**: ~65% computation savings

### Target Journals

| Priority | Journal |
|----------|---------|
| 1 | Neural Networks |
| 2 | Cognitive Computation |
| 3 | IEEE TNNLS |

---

## File Structure

```
civis_lucri_faber/
├── core/
│   ├── language_cortex.py    # Language + Bio-Gating
│   ├── auditory_cortex.py   # Auditory + psychology
│   └── multimodal_*.py     # Fusion
├── censor_bridge.py        # Vision + Censor
├── monitor.py              # Streamlit UI
├── train_*.py            # Training scripts
├── eval_psychology.py     # Evaluation
├── emotional_dialogue.py  # Application
└── README*.md            # Documentation
```

---

## License

MIT License

---

## Citation

```bibtex
@software{civis_lucri_faber,
  title={Civis Lucri-Faber: Bio-Inspired Cognitive AI},
  author={Civis Lab},
  year={2026},
  url={https://github.com/civis-lucri-faber}
}
```