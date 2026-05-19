# Simulacrum Technical Documentation

> Detailed Technical Specification v1.0

> **Complete Technical Documentation for Computer Science Faculty**

---

## 1. Project Overview and Research Background

### 1.1 Research Motivation

Simulacrum (Latin "Craftsman Seeking Wealth", Simulacrum) is a bio-inspired AI cognitive architecture with **8 implemented brain mechanisms**, **cognitive psychology**, and **adaptive pruning**. The core research question: **How to design more efficient, interpretable, and adaptive AI systems by inspired by the human brain?**

#### 1.1.1 Brain vs. Computer

| Aspect | Human Brain | Transformer |
|--------|-----------|------------|
| **Power** | ~20W | Hundreds to thousands W |
| **Computation** | Event-driven | Full computation |
| **Storage** | Distributed | Separate VRAM |
| **Learning** | Continuous | Batch training |
| **Latency** | ~100ms | Variable |
| **Tolerance** | High (plasticity) | Fragile |

The brain accomplishes vast cognitive tasks with ~20W power. This inspired our exploration of brain-inspired computing.

#### 1.1.2 Neuromorphic Computing History

| Year | Milestone | Reference |
|------|-----------|-----------|
| 1943 | McCulloch-Pitts neuron model | McCulloch & Pitts (1943) |
| 1949 | Hebb's learning rule | Hebb (1949) |
| 1958 | Rosenblatt perceptron | Rosenblatt (1958) |
| 1989 | Carver Mead's neuromorphic chip | Mead (1989) |
| 2008 | IBM TrueNorth (1M neurons) | Merolla et al. (2014) |
| 2014 | Intel Loihi chip | Davies et al. (2018) |
| 2021 | Spiking Neural Networks revival | Eshraghian et al. (2021) |

---

## 2. System Architecture

### 2.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Simulacrum                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Language  │  │  Auditory  │  │   Vision   │        │
│  │  Cortex   │  │  Cortex   │  │   Censor   │        │
│  │  (Broca)  │  │    (A1)   │  │(Thalamus)  │        │
│  │   7.6M   │  │   1.0M    │  │   3.6M    │        │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘        │
│        │             │              │                 │
│        └─────────────┼──────────────┘                 │
│                      ↓                            │
│            ┌───────────────┐                     │
│            │  Prefrontal  │                     │
│            │Prefrontal   │                     │
│            │ Decision/Learning│                  │
│            └───────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Fifteen Brain Mechanisms

**Note**: This list is a target/goal. Not all 15 are fully implemented.

| # | Brain Region | Location | Function | Implementation | Status |
|---|------------|----------|----------|------------|--------|
| 1 | **Cochlea** | Inner ear | Frequency analysis | Gabor filter | ✓ Implemented |
| 2 | **Inferior Colliculus** | Midbrain | Auditory relay | -- | Listed only |
| 3 | **MGN** | Thalamus | Auditory thalamus | -- | Listed only |
| 4 | **Primary Auditory (A1)** | Superior temporal | Sound processing | ✓ Implemented | ✓ Implemented |
| 5 | **Planum Temporale** | Superior temporal | Speech recognition | -- | Listed only |
| 6 | **SIP** | Parietal | Spatial localization | -- | Listed only |
| 7 | **Broca's Area** | Inferior frontal | Language production | ✓ Implemented | ✓ Implemented |
| 8 | **Wernicke's Area** | Superior temporal | Semantic understanding | ✓ Implemented | ✓ Implemented |
| 9 | **Arcuate Fasciculus** | White matter | Inter-regional connection | CrossModalBinder | ✓ Implemented |
| 10 | **Amygdala** | Medial temporal | Emotion perception | EmergentEmotion | ✓ Implemented |
| 11 | **Hippocampus** | Medial temporal | Working memory | Listed only |
| 12 | **Prefrontal Cortex** | Frontal lobe | Decision making | ✓ Implemented |
| 13 | **Pulvinar** | Thalamus | Fast routing | Listed only |
| 14 | **Visual Cortex V1-V4** | Occipital | Visual processing | ✓ Implemented | ✓ Implemented |
| 15 | **Midbrain nuclei** | Midbrain | Neuromodulation | -- | Listed only |

**Legend**:
- ✓ Implemented: Code exists in `core/`
- Listed only: Named in table but not implemented

---

## 3. Core Modules

### 3.1 Language Cortex — Broca's and Wernicke's Areas

#### 3.1.1 Neuroanatomical Background

**Dual-System Model for Language:**

```
┌──────────────────────────────────────────────────────┐
│          Language Processing: Two Pathways           │
├──────────────────────────────────────────────────────┤
│                                                  │
│  Understanding pathway:                          │
│  Auditory cortex → Wernicke → Arcuate → Broca → Motor│
│     (sound)    (semantic)     (transfer)  (grammar)   │
│                                                  │
│  Wernicke damage: "Can speak but cannot understand" │
│  Broca damage: "Can understand but cannot speak"  │
│                                                  │
│  Broca's aphasia (1861): Non-fluent, agrammatic    │
│  Wernicke's aphasia (1874): Fluent but meaningless │
│                                                  │
└──────────────────────────────────────────────────────┘
```

##### 3.1.1.1 Broca's Area

**Historical Background:** In 1861, French physician Pierre Paul Broca dissected the brain of a patient known as "Tan" (due to his only word) and found damage in the left inferior frontal gyrus. This was the first description of what is now known as Broca's aphasia.

**Location:** Left inferior frontal gyrus, BA 44/45

**Structure:**
- Pars opercularis (BA 44)
- Pars triangularis (BA 45)
- Pars orbitalis (BA 47)

**Function (Rickard et al., 2005):**
- Language production
- Syntactic processing
- Motor programming for speech

**Lesion Consequences:**
- Non-fluent speech
- Agrammatism
- Relatively intact comprehension

**Computational Significance:**
- Sequence-to-sequence transformation
- Grammar structure modeling
- Motor programming analogy

##### 3.1.1.2 Wernicke's Area

**Historical Background:** In 1874, German physician Carl Wernicke described another type of language deficit with features opposite to Broca's aphasia.

**Location:** Left posterior superior temporal gyrus, BA 22

**Structure:**
- Posterior superior temporal gyrus (STG)
- Planum temporale
- Angular gyrus region

**Function:**
- Semantic integration
- Word meaning decoding
- Phonological processing

**Lesion Consequences:**
- Fluent but meaningless speech
- Paragrammatic errors
- Impaired repetition

**Computational Significance:**
- Semantic representation learning
- Context understanding
- Attention mechanism

##### 3.1.1.3 Arcuate Fasciculus

**Anatomy:** Large white matter tract connecting Broca and Wernicke

**Function:**
- Speech repetition
- Inter-area information transfer

**Lesion:** Conduction aphasia

**Reference:** Catani et al. (2005). Segmental language mapped. NeuroImage, 26(2), 317-329.

##### 3.1.1.4 Working Memory

**Historical Background:** George Miller (1956) "The magical number seven, plus or minus two"

**Original Text:**
> "There appears to be a limit to the number of separate items that can be estimated or remembered without confusions, and this limit is roughly seven."

**Neuroimaging Evidence:**
- PFC maintained activation
- Parietal activation for storage
- PFC-parietal functional connectivity

**Capacity Limit:** Neural basis of 7±2:

| Explanation | Evidence |
|------------|----------|
| Limited attention | Adults focus on ~4 chunks |
| Neural sync limit | ~4 items can activate simultaneously |
| Thalamic filtering | 7 gating units |

**Reference:** Miller (1956). Psychological Review, 63(2), 81-97.

#### 3.1.2 Model Architecture

```python
class LanguageCortex(nn.Module):
    """
    Language Cortex + Bio-Gating
    
    Two modes:
    - use_parallel=True: Parallel GRU (batch fast processing)
    - use_parallel=False: Serial SSM + Bio-Gating (streaming + emotion)
    
    Input: [B, T] token sequence
    Output: {
        'features': [B, 256],
        'valence': [B],
        'arousal': [B],
        'semantic': dict,
        'surprise': float,
        'emotion_state': dict
    }
    """
    def __init__(self, vocab_size=10000, use_parallel=True):
        super().__init__()
        self.use_parallel = use_parallel
        embed_dim = 256
        
        # Word embedding (lexical storage)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position encoding (working memory index)
        self.pos_embedding = nn.Embedding(512, embed_dim)
        
        if use_parallel:
            # Parallel: BiGRU
            self.encoder = ParallelEncoder(embed_dim)
        else:
            # Serial: SSM + Bio-Gating
            self.ssm = SSMStateUpdate(embed_dim)
            self.memory = WorkingMemory(embed_dim)
            self.semantic = SemanticEncoder(embed_dim)
            
            # Psychology components
            self.plutchik = PlutchikEmotion()
            self.dual_process = DualProcessCognition(embed_dim)
            self.embodied = EmbodiedCognition()
            self.cognitive_load = CognitiveLoadManager()
            self.emotion_regulation = EmotionRegulation()
            self.cognitive_bias = CognitiveBias(embed_dim)
            self.metacognition = Metacognition(embed_dim)
            
            # Synaptic plasticity
            self.expert_pruner = DynamicExpertPruner(n_experts=4, top_k=1)
            self.synaptic_depression = SynapticDepression(decay_rate=0.01)
            self.oja = OjaRule(learning_rate=0.01)
        
        # Emotion head (amygdala simulation)
        self.emotion = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
```

#### 3.1.3 Mathematical Formulas

**GRU gating (Cho et al., 2014):**

Update gate:
$$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$

Reset gate:
$$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$

Candidate hidden:
$$\tilde{h}_t = \tanh(W x_t + r_t \odot (U h_{t-1}) + b)$$

Final state:
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Notation:**
- $\sigma$: sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$
- $\odot$: element-wise product (Hadamard)
- $W, U$: learnable weights

**Parameter Complexity:**

For input_dim=$d$, hidden_dim=$d_h$:
$$\text{params} = 3 \times (d \times d_h + d_h \times d_h) + 3 \times d_h$$

For input_dim=$d$, hidden_dim=$d_h$:
$$\text{params} = 3 \times (d \times d_h + d_h \times d_h) + 3 \times d_h$$

#### 3.1.5 Complete Implementation and Performance Analysis

##### 3.1.5.1 Complete LanguageCortex Code

```python
class LanguageCortex(nn.Module):
    """
    Complete Language Cortex Implementation
    
    Design Principles:
    - Two-layer bidirectional GRU for full context
    - Bio-Gating for dynamic expert routing
    - Working memory limit avoids O(n) complexity
    - Emotion head outputs VAD state
    
    Parameter Complexity:
    - Embedding: vocab_size × embed_dim
    - GRU: 3 × hidden_dim × (hidden_dim + embed_dim) × num_layers
    - Bio-Gating: embed_dim × n_experts
    - Total: O(vocab_size × d + d × h²)
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        n_experts: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        
        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 2. Parallel Encoder (bidirectional GRU)
        self.encoder = ParallelEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # 3. Bio-Gating
        self.bio_gate = BioGate(
            embed_dim=embed_dim,
            n_experts=n_experts,
        )
        
        # 4. Working Memory (7-slot)
        self.working_memory = WorkingMemory(
            hidden_dim=hidden_dim,
            n_slots=7,
        )
        
        # 5. Emotion Head (VAD)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        
        # 6. Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        
    def forward(
        self,
        input_ids: torch.Tensor,
        return_emotion: bool = False,
        return_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            {
                'logits': [batch, seq_len, vocab_size],
                'features': [batch, hidden_dim],
                'valence': [batch],
                'arousal': [batch],
                'dominance': [batch],
                'memory': [n_slots, hidden_dim],
            }
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Embedding
        x = self.embedding(input_ids)
        x = self.embed_dropout(x)
        
        # 2. Bio-Gating routing
        expert_idx, gate_weights = self.bio_gate(x.mean(dim=1))
        
        # 3. Bidirectional GRU
        encoded = self.encoder(x)
        features = encoded[:, -1, :]
        
        # 4. Working memory
        memory_output = self.working_memory(features)
        
        # 5. Emotion (VAD)
        emotion_raw = self.emotion_head(features[:, :self.hidden_dim])
        valence = torch.tanh(emotion_raw[:, 0])
        arousal = torch.sigmoid(emotion_raw[:, 1])
        dominance = torch.sigmoid(emotion_raw[:, 2])
        
        # 6. Output logits
        logits = self.output_proj(features[:, :self.hidden_dim])
        
        result = {
            'logits': logits,
            'features': features,
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'expert_idx': expert_idx,
            'gate_weights': gate_weights,
        }
        
        if return_emotion:
            result['emotion_state'] = {
                'valence': valence,
                'arousal': arousal,
                'dominance': dominance,
            }
            
        if return_memory:
            result['memory'] = memory_output
            
        return result
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        return_acc: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute language model loss"""
        outputs = self.forward(input_ids)
        logits = outputs['logits']
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, self.vocab_size),
            target_ids.view(-1),
        )
        
        result = {'loss': loss}
        
        if return_acc:
            preds = outputs['logits'].argmax(dim=-1)
            acc = (preds == target_ids[:, -1]).float().mean()
            result['acc'] = acc
            
        return result
```

##### 3.1.5.2 Mathematical Derivation: Attention Score

**Standard Self-Attention:**

For query $Q \in \mathbb{R}^{n \times d_k}$, key $K \in \mathbb{R}^{m \times d_k}$, value $V \in \mathbb{R}^{m \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Our Simplified Version:**

Using Top-1 gating instead of full attention:

$$\text{score}_i = \text{softmax}(W_c x + p + e + m)_i$$

This is a **hard attention** form, selecting only one expert:

$$\text{output} = \text{Expert}_{\arg\max_i \text{score}_i}(x)$$

**Complexity Comparison:**

| Method | Complexity | Experts |
|--------|-----------|---------|
| Standard Attention | $O(n^2 \cdot d)$ | All |
| MoE Top-2 | $O(n \cdot d \cdot 2)$ | 2 |
| **Bio-Gating** | $O(n \cdot d)$ | **1** |

##### 3.1.5.3 Performance Analysis: FLOPs and Latency

**FLOPs Calculation (single forward):**

```
Input: [batch, seq_len, embed_dim]

1. Embedding: 0 (lookup)
2. Bio-Gating:
   - Linear: batch × embed_dim × n_experts
   - Softmax/argmax: batch × n_experts
   Subtotal: O(batch × embed_dim × n_experts)

3. Two-layer Bidirectional GRU:
   - Per layer: 6 × batch × seq_len × hidden_dim × (hidden_dim + embed_dim)
   - Two layers bidirectional: 2 × 2 × 6 = 24
   Total: O(batch × seq_len × hidden_dim²)

4. Output projection:
   - Linear: batch × hidden_dim × vocab_size
   Total: O(batch × hidden_dim × vocab_size)
```

**Latency Estimate (PyTorch GPU):**

| Component | Latency(ms) | Percentage |
|----------|------------|-----------|
| Embedding | 0.1 | 2% |
| Bio-Gating | 0.2 | 4% |
| GRU | 3.5 | 70% |
| Output | 1.2 | 24% |
| **Total** | **5.0ms** | 100% |

**Memory Usage:**

| Component | Parameters | Percentage |
|-----------|-----------|-----------|
| Embedding | 2.56M | 34% |
| GRU | 4.19M | 55% |
| Bio-Gating | 0.26M | 3% |
| Output | 2.56M | 8% |
| **Total** | 7.6M | 100% |

##### 3.1.5.4 Gradient Computation and Backpropagation

**Gradient to Expert Weights:**

Bio-Gating uses hard selection (argmax), non-differentiable. Solution: Gumbel-Softmax approximation:

```python
def gumbel_softmax(logits, tau=1.0, hard=False):
    """Gumbel-Softmax approximation (Jang et al., 2017)"""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = F.softmax((logits + gumbel_noise) / tau, dim=-1)
    
    if hard:
        y_hard = torch.zeros_like(logits)
        y_hard.scatter_(1, y.argmax(dim=-1, keepdim=True), 1)
        y = (y_hard - y).detach() + y
        
    return y

# Gradient example
gate_logits = self.bio_gate.content_gate(x)
gate_weights = gumbel_softmax(gate_logits, tau=0.1)
output = torch.matmul(gate_weights, expert_outputs)  # Differentiable
```

---

#### 3.1.4 Usage Example

```python
import torch
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# Load model
lang = load('core/language_cortex.py', 'language_cortex')
model = lang.create_language_cortex(vocab_size=1000, use_parallel=False)
model.eval()

# Input
tokens = torch.randint(0, 1000, (2, 16))

# Forward
with torch.no_grad():
    result = model(tokens, return_emotion=True)

# Output
print(f"Features: {result['features'].shape}")    # [2, 256]
print(f"Valence: {result['valence']}")          # tensor([-0.0521,  0.1234])
print(f"Arousal: {result['arousal']}")          # tensor([0.5234,  0.4891])
print(f"Surprise: {result['surprise']}")         # 0.68
```

---

### 3.2 Bio-Gating — Amygdala and Neuromodulation

#### 3.2.1 Neurobiology

##### 3.2.1.1 Amygdala

**Anatomy:** Almond-shaped nuclei in medial temporal lobe, ~1.5cm.

```
┌─────────────────────────────────────────────────────┐
│        Amygdala Structure (LeDoux, 2000)          │
├─────────────────────────────────────────────────────┤
│                                                  │
│    Lateral Nucleus (LA) → Central Nucleus (CeA) →   │
│           ↓                    ↓                  │
│       Sensory input      Behavioral output        │
│                                                  │
│  Two pathways:                                   │
│  1. Thalamic→Cortical→Amygdala (slow, 500ms)    │
│  2. Censor pathway: Thalamus→Amygdala (fast)   │
│                                                  │
└─────────────────────────────────────────────────────┘
```

**Historical Background:** LeDoux (1996) "The Emotional Brain" established the dual pathway model for emotional processing.

**Function:**
- Emotional learning
- Fear conditioning
- Value assignment

**Reference:** LeDoux, J. E. (2000). Emotion circuits in the brain. Annual Review of Neuroscience, 23, 155-184.

##### 3.2.1.2 Membrane Potential

**Neurophysiology:** Resting potential ~-70mV.

| State | Potential | Time | Description |
|-------|----------|------|------------|
| Hyperpolarized | -90mV | 2ms | Inhibition |
| Resting | -70mV | Steady | Normal |
| Threshold | -55mV | Trigger | Action potential |
| Depolarized | +30mV | 1ms | Peak |

**H-H Equation (Hodgkin-Huxley, 1952):**
$$C_m \frac{dV}{dt} = I - g_{Na}m^3h(V-E_{Na}) - g_K n^4(V-E_K) - g_L(V-E_L)$$

Our simplified model uses: $p_{t+1} = p_t \times decay + update$

##### 3.2.1.3 Emotion Dimensions (VAD)

**Russell's Affective Circumplex (1980):**

```
              Arousal
                |
                |
     Calm ------+------ Excited
                |
                └---------- Valence
                |
         Negative --+-- Positive
                  |
                -1 ←———→ +1
```

##### 3.2.1.4 Neuromodulators

**Historical Background:** Schultz (1997) discovered dopamine neuron reward prediction error responses.

| Neuromodulator | Brain Region | Function | Behavioral Effect |
|----------------|-------------|----------|--------------------|
| **Dopamine** | VTA, SN | Reward learning | Motivation, reinforcement |
| **Serotonin** | Raphe nucleus | Mood regulation | Emotional stability |
| **Norepinephrine** | Locus coeruleus | Arousal/attention | Alertness |
| **Acetylcholine** | Basal forebrain | Memory attention | Learning |

**Schultz (2007) Reward Prediction Error:**
$$RPE = r_t - V(s_t)$$

**Reference:** Schultz, W. (2007). Multiple dopamine functions at different time courses. Annual Review of Neuroscience, 30, 259-288.

#### 3.2.2 BioGate Implementation

```python
class BioGate(nn.Module):
    """
    Bio-Gating: Content + Membrane + Emotion + Mood
    
    Core idea: Simulate "emotion affects decisions"
    - Content: input-driven selection
    - Membrane: history accumulation (LTP/LTD)
    - Emotion: VAD (immediate)
    - Mood: persistent background
    """
    def __init__(self, dim=256, n_experts=4):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        
        # 1. Content gate
        self.content_gate = nn.Linear(dim, n_experts)
        
        # 2. Membrane potential (memory)
        self.membrane_potential = nn.Parameter(torch.zeros(n_experts))
        self.membrane_decay = 0.9
        
        # 3. Emotion vector (VAD)
        self.emotion_vector = nn.Parameter(torch.zeros(3))
        
        # 4. Mood state
        self.mood = MoodState()
        
    def forward(self, input_emb):
        # Content
        content_logits = self.content_gate(input_emb)
        
        # Memory effect
        membrane_effect = self.membrane_potential.unsqueeze(0)
        
        # Emotion modulation
        emotion_effect = torch.tanh(self.emotion_vector.sum()) * 0.2
        
        # Mood background
        mood_effect = self.mood.mood_affect_decision(content_logits)
        
        # Combined decision
        gate_logits = content_logits + membrane_effect + emotion_effect + mood_effect
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Top-1 selection (75% save)
        expert_idx = gate_weights.argmax(dim=-1)
        
        # Update membrane (LTP/LTD simulation)
        with torch.no_grad():
            updates = torch.zeros_like(self.membrane_potential)
            updates[expert_idx] = 0.1
            self.membrane_potential.data = (
                self.membrane_potential.data * self.membrane_decay + updates
            )
        
        return expert_idx, gate_weights
    
    @property
    def emotion_state(self):
        return {
            'valence': torch.tanh(self.emotion_vector[0]),
            'arousal': torch.sigmoid(self.emotion_vector[1]),
            'dominance': torch.sigmoid(self.emotion_vector[2]),
        }
```

#### 3.2.3 Mathematical Formulas

**Gate formula:**
$$\text{gate}_i = \text{softmax}(W_c x + p + e + m)_i$$

- $W_c x$: Content gate (input)
- $p = \text{membrane\_potential}$: Memory
- $e = \tanh(\sum(VAD)) \times 0.2$: Emotion
- $m = \text{mood} \times \text{decision\_bias}$: Mood

**Membrane update:**
$$p_{t+1} = p_t \times \text{decay} + \mathbb{1}[selected]$$

**Emotion affects behavior:**
$$\text{effect} = (+joy-0.5) \times 0.3 - (fear-0.5) \times 0.3 + anger \times 0.2$$

**Significance:** This simulates:
- People become irritable when tired/hungry (membrane accumulation)
- Positive emotion increases risk preference (emotion modulation)
- Optimists more willing to try new things (mood background)

---

### 3.3 Auditory Cortex

#### 3.3.1 Neurobiology

```
┌─────────────────────────────────────────────────────┐
│       Auditory Pathway (Pickles, 2015)               │
├─────────────────────────────────────────────────────┤
│                                                  │
│  Sound → Cochlea → Auditory nerve → Cochlear nucleus│
│                           ↓                       │
│                    Inferior Colliculus           │
│                           ↓                       │
│                       MGN                        │
│                           ↓                       │
│                 Primary Auditory Cortex (A1)       │
│                           ↓                       │
│              ┌──────────┴──────────┐              │
│              ↓                       ↓              │
│        Ventral Stream         Dorsal Stream        │
│         (What)               (Where)            │
│              ↓                       ↓              │
│         Temporal            Parietal            │
│                                                  │
└─────────────────────────────────────────────────────┘
```

##### 3.3.1.1 Cochlea

**Historical:** Georg von Békésy received the 1961 Nobel Prize for work on traveling waves in the basilar membrane.

**Function:** Mechanical-neural transduction, frequency decomposition

**Structure:** Basilar membrane
- Base: narrow/stiff → high frequency (20kHz)
- Apex: wide/soft → low frequency (20Hz)

**Reference:** von Békésy, G. (1961). Concerning the pleasures and pains of stimulating the cortex. Human Frontiers.

##### 3.3.1.2 Inferior Colliculus

**Location:** Midbrain

**Function:** Binaural integration, sound localization

##### 3.3.1.3 Dual Stream Theory (Hickok & Poeppel, 2007)

**Ventral Stream:**
- Function: Speech recognition, semantic understanding

**Dorsal Stream:**
- Function: Spatial localization, sound-motion integration

**Reference:** Hickok, G., & Poeppel, D. (2007). The cortical organization of speech processing. Nature Reviews Neuroscience, 8(5), 393-402.

---

### 3.4 Vision Censor — Thalamic Fast Pathway

#### 3.4.1 Neurobiology

```
┌─────────────────────────────────────────────┐
│       Visual Pathways (LeDoux, 2000)       │
├─────────────────────────────────────────────┤
│                                          │
│  Retina → LGN → Visual Cortex → Recognition │
│        (slow, ~500ms)                     │
│                                          │
│  Retina → Thalamus (Censor) → Amygdala   │
│        (fast, ~100ms)                     │
│                                          │
│  Fast pathway: "Act first, think later"   │
│                                          │
└─────────────────────────────────────────────┘
```

**Reference:** LeDoux, J. E. (1996). The Emotional Brain. Simon & Schuster.

---

### 3.5 Cognitive Psychology

#### 3.5.0 Complete Implementations

##### 3.5.0.1 WorkingMemory Implementation

```python
class WorkingMemory(nn.Module):
    """Working Memory: 7±2 slot limit (Miller, 1956)"""
    
    def __init__(self, hidden_dim: int = 512, n_slots: int = 7, attention_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_slots = n_slots
        
        self.slots = nn.Parameter(torch.zeros(n_slots, hidden_dim), requires_grad=False)
        
        self.query_proj = nn.Linear(hidden_dim, attention_dim)
        self.key_proj = nn.Linear(hidden_dim, attention_dim)
        self.value_proj = nn.Linear(hidden_dim, attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=4, batch_first=True)
        
        self.forget_gate = nn.Linear(hidden_dim, n_slots)
        self.write_ptr = 0
        
    def forward(self, new_state: torch.Tensor, return_attention: bool = False):
        batch_size = new_state.shape[0]
        
        with torch.no_grad():
            self.slots.data[self.write_ptr] = new_state[0].detach()
            self.write_ptr = (self.write_ptr + 1) % self.n_slots
        
        query = self.query_proj(new_state).unsqueeze(1)
        keys = self.key_proj(self.slots.unsqueeze(0).expand(batch_size, -1, -1))
        values = self.value_proj(self.slots.unsqueeze(0).expand(batch_size, -1, -1))
        
        attn_output, attn_weights = self.attention(query, keys, values)
        
        forget_weights = torch.sigmoid(self.forget_gate(new_state)).unsqueeze(1)
        attn_weights = F.softmax(attn_weights * forget_weights, dim=-1)
        
        aggregated = torch.matmul(attn_weights, values).squeeze(1)
        
        return {'aggregated': aggregated, 'attention_weights': attn_weights[0]} if return_attention else aggregated
```

##### 3.5.0.2 PlutchikEmotion Implementation

```python
class PlutchikEmotion(nn.Module):
    """Plutchik Emotion Wheel (1980) - 8 Basic Emotions"""
    
    EMOTION_NAMES = ['joy', 'sadness', 'trust', 'disgust', 'fear', 'anger', 'surprise', 'anticipation']
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.emotion_intensities = nn.Parameter(torch.zeros(8), requires_grad=True)
        
    def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        emotions = torch.sigmoid(self.emotion_intensities)
        emotions = emotions.unsqueeze(0).expand(hidden_state.shape[0], -1)
        
        valence = (+ emotions[:, 0] - emotions[:, 1] + emotions[:, 2] - emotions[:, 3] - emotions[:, 5])
        arousal = (emotions[:, 4] + emotions[:, 5] + emotions[:, 6] + emotions[:, 7]) / 2
        dominance = (emotions[:, 0] + emotions[:, 2] + emotions[:, 7] - emotions[:, 4])
        
        return {
            'valence': torch.tanh(valence),
            'arousal': torch.clamp(arousal, 0, 1),
            'dominance': torch.clamp(dominance, 0, 1),
            'emotions': emotions[0],
            'primary': self.EMOTION_NAMES[emotions.argmax().item()],
        }
```

##### 3.5.0.3 DualProcessCognition Implementation

```python
class DualProcessCognition(nn.Module):
    """Dual Process Cognition (Kahneman, 2011)"""
    
    def __init__(self, hidden_dim: int = 256, system1_hidden: int = 128, system2_hidden: int = 256):
        super().__init__()
        
        self.system1 = nn.Sequential(nn.Linear(hidden_dim, system1_hidden), nn.ReLU(), nn.Linear(system1_hidden, hidden_dim))
        self.system2 = nn.Sequential(nn.Linear(hidden_dim, system2_hidden), nn.ReLU(), nn.LayerNorm(system2_hidden), nn.Linear(system2_hidden, system2_hidden), nn.ReLU(), nn.Linear(system2_hidden, hidden_dim))
        self.switch = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        
    def forward(self, x: torch.Tensor, forced_system: str = None):
        switch_prob = self.switch(x)
        
        if forced_system == 'system1':
            return {'output': self.system1(x), 'system_used': 'system1', 'switch_prob': switch_prob}
        elif forced_system == 'system2':
            return {'output': self.system2(x), 'system_used': 'system2', 'switch_prob': switch_prob}
        else:
            used = 'system2' if torch.rand(1) < switch_prob else 'system1'
            output = self.system2(x) if used == 'system2' else self.system1(x)
            return {'output': output, 'system_used': used, 'switch_prob': switch_prob}
```

#### 3.5.1 Plutchik Emotion Wheel (1980)

| Emotion | Function | Intensity Variation |
|--------|----------|---------------------|
| Joy | Pleasure→Joy→Ecstasy | ↑ |
| Sadness | Sadness→Grief→Depression | ↑ |
| Trust | Doubt→Trust→Faith | ↑ |
| Disgust | Disgust→Revulsion→Vomiting | ↑ |
| Fear | Worry→Fear→Terror | ↑ |
| Anger | Annoyance→Anger→Rage | ↑ |
| Surprise | Surprise→Amazement→Astonishment | ↑ |
| Anticipation | Interest→Anticipation→Alert | ↑ |

**Reference:** Plutchik, R. (1980). Emotion: Psychoevolutionary Synthesis. Harper & Row.

#### 3.5.2 Dual Process Theory (Kahneman, 2011)

| Aspect | System 1 | System 2 |
|--------|----------|----------|
| Speed | Fast (~100ms) | Slow (~500ms) |
| Consciousness | Unconscious | Conscious |
| Computation | Parallel | Serial |
| Effort | Automatic | Controlled |

**Reference:** Kahneman, D. (2011). Thinking, Fast and Slow. Farrar.

#### 3.5.3 Metacognition

**Definition:** Metacognition is "cognition about cognition" (Flavell, 1979)

**Reference:** Flavell, J. H. (1979). Metacognition and cognitive monitoring. American Psychologist, 34(10), 906-911.

---

### 3.6 Synaptic Plasticity

#### 3.6.1 Hebbian Learning

**Original (Hebb, 1949):**
> "Neurons that fire together, wire together"

**Hebb's rule:**
$$\Delta w_{ij} = \eta \cdot a_i \cdot a_j$$

**Oja's rule (1982):**
$$\Delta w_{ij} = \eta \cdot a_j(a_i - w_{ij} \cdot a_j)$$

#### 3.6.2 LTP and LTD

| Process | Duration | Molecular Mechanism | Simulation |
|---------|----------|-------------------|------------|
| **LTP** | Long-term | Ca²⁺ → NMDA | Oja's rule |
| **LTD** | Long-term | Endocytosis | Depression |

**Reference:** Hebb, D. O. (1949). The Organization of Behavior. Wiley.

---

### 3.7 Improved Modules (Censor-aligned)

本节描述与Censor项目对齐的改进模块。

#### 3.7.1 AdaptiveVisualAttention — 两阶段视觉注意力

**对应Censor模块**: AdaptiveOpticalFlow

**功能**: 快速初筛 + 精细注意力

```python
class AdaptiveVisualAttention(nn.Module):
    """
    两阶段视觉注意力：
    - Stage 1: 快速筛选 (saliency screening)
    - Stage 2: 精细注意力 (fine attention) 仅当motion detected
    """
    def __init__(self, embed_dim=768, num_heads=8, threshold=0.1):
        self.saliency_scorer = nn.Sequential(...)
        self.fine_attn = nn.MultiheadAttention(...)
    
    def forward(self, x):
        saliency_scores = self.saliency_scorer(x)
        if saliency_scores.mean() > self.threshold:
            # 精细注意力
            attn_out = self.fine_attn(x, x, x)
            stage = 'fine'
        else:
            # 快速路径
            attn_out = x
            stage = 'fast'
        return {'output': attn_out, 'stage': stage}
```

#### 3.7.2 SaliencyDetectorE2E — 全端到端显著性检测

**对应Censor模块**: SaliencyDetectorE2E

**改进**:
1. 所有参数可学习 (sigma_ratio, center_bias, fusion_weights)
2. 分辨率自适应sigma: sigma = sigma_ratio × min(H, W)

```python
class SaliencyDetectorE2E(nn.Module):
    def __init__(self, embed_dim=768, sigma_ratio=0.15):
        self.sigma_ratio = nn.Parameter(torch.tensor(sigma_ratio))
        self.center_bias = nn.Parameter(torch.tensor(0.5))
        self.fusion_weights = nn.Parameter(torch.ones(levels) / levels)
```

#### 3.7.3 StandardMoE — 标准MoE对比

**对应Censor模块**: StandardMoE (对比BioMoE)

| 特性 | BioMoE | StandardMoE |
|------|--------|------------|
| Routing | gate(input, membrane) | gate(input) |
| Memory | Membrane potential | None |
| Emotional | Mood bias | None |
| Persistence | Stateful | Stateless |

```python
class StandardMoE(nn.Module):
    """标准MoE - 无生物学先验，更客观"""
    def __init__(self, input_dim, output_dim, num_experts=3):
        self.experts = nn.ModuleList([...])  # 3个MLP
        self.gate = nn.Linear(input_dim, num_experts)
```

#### 3.7.4 AmygdalaWithPrior — 面部区域先验

**对应Censor模块**: AmygdalaWithPrior

**改进**: 添加面部区域先验（眼/鼻/嘴）

```python
class AmygdalaWithPrior(nn.Module):
    def __init__(self, input_dim=64):
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 14*14)
        self.prior_strength = nn.Parameter(torch.tensor(0.3))
        self.register_buffer('face_prior', self._create_face_prior())
    
    def _create_face_prior(self):
        # 眼区域 (top) | 鼻区域 (center) | 嘴区域 (bottom)
        prior[h//6:h//3, w//4:3*w//4] = 1.0  # 眼
        prior[h//3:h//2, w//3:2*w//3] = 1.0  # 鼻
        prior[2*h//3:5*h//6, w//4:3*w//4] = 1.0  # 嘴
```

#### 3.7.5 PersonalizedAdaptation — 个性化适配

**对应Censor模块**: PersonalizedRadarEnhanced

**改进**: Warmup LR + Test-time adaptation

```python
class PersonalizedAdaptation(nn.Module):
    def _get_lr(self, step):
        warmup_steps = 2
        if step < warmup_steps:
            return 1e-5 + (self.base_lr - 1e-5) * step / warmup_steps
        # Cosine decay
        progress = (step - warmup_steps) / (self.adapt_steps - warmup_steps)
        return self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

---

### 3.8 Emergent Emotion -- 从动力学涌现情绪

**核心思想**：情绪不再硬编码Plutchik轮，而是从底层机制涌现。

#### 3.8.1 为什么需要涌现？

| 硬编码 | 涌现 |
|--------|------|
| 预设8种情绪 | 从交互中涌现 |
| 固定VAD映射 | 从学习中生成 |
| 多巴胺=奖励 | 从预测误差生成 |
| 固定阈值 | 动态适应 |

#### 3.8.2 底层机制

```python
class EmergentEmotion(nn.Module):
    """从价值学习 + 紧迫度 + 社会推理的交互中涌现情绪"""
    
    def __init__(self):
        self.value_learner = ValueLearner()      # TD误差驱动
        self.urgency_detector = UrgencyDetector()  # 时间尺度
        self.social_inference = SocialInference()  # ToM
        self.emergence_net = nn.Sequential(...)  # 涌现动力学
    
    def forward(self, state, reward, other_obs):
        # 1. 价值学习 → TD误差（多巴胺信号）
        value = self.value_learner(state, reward)  # δ = r - V
        
        # 2. 紧迫度
        urgency = self.urgency_detector(state)  # fast vs slow
        
        # 3. 社会推理
        social = self.social_inference(other_obs, state)  # ToM
        
        # 4. 涌现：从交互动力学中生成情绪
        combined = [value, urgency, social]
        emotion = self.emergence_net(combined)
        return emotion
```

#### 3.8.3 验证方法

**时间动态匹配**：
- 恐惧反应：0-200ms生理唤醒 → 对应fast_urgency
- 愉悦期望：渐进式TD衰减 → 对应value learning

**预测验证**：
- 输入模糊刺激 → 产生特定混淆（gorilla vs elephant）
- 个体差异可预测（内向/外向）

**与其他模块的耦合**：
- 情绪输出 → 影响决策（价值学习器的输入）
- 决策结果 → 反馈给情绪（reward信号）

---

### 3.9 Cross-Modal Binding -- 跨模态绑定

**问题**：视觉"看到蛇" + 听觉"听到嘶嘶声" → 如何统一为"危险"体验？

#### 3.9.1 当前问题

```
视觉Censor ──┐
听觉Censor ──┼──→ 前额叶（并行输入，缺乏动态交互）
语言Censor ──┘
```

#### 3.9.2 Binding Mechanism

```python
class CrossModalBinder(nn.Module):
    """
    1. Temporal Binding Buffer: 存储时序事件
    2. Cross-Modal Attention: 跨模态调制
    3. Scene Detector: 场景检测
    """
    
    def forward(self, modalities, timestamp):
        # 编码各模态
        encoded = {mod: self.encoder[mod](input) for mod, input in modalities.items()}
        
        # 时序绑定：找到时间窗口内的事件
        if 'vision' in encoded and 'audio' in encoded:
            bound = self.cross_attention(
                query_mod='vision',
                query_repr=encoded['vision'],
                key_mod='audio',
                key_repr=encoded['audio'],
            )
        
        # 统一表征 → 场景检测
        scene = self.scene_detector(unified_repr)
        return scene
```

#### 3.9.3 解决的具体问题

| 问题 | 解决方案 |
|------|-----------|
| Temporal Binding | 时序缓冲区，找到±500ms的事件 |
| Cross-modal Attention | Query-Key-Value跨模态调制 |
| Scene Detection | 统一表征→场景分类 |

---

## 4. Training

### 4.1 Language: Next Token Prediction

```python
def train_language():
    phrases = [
        "the cat sat on the mat",
        "a dog runs in the park",
    ]
    
    lang = load('core/language_cortex.py', 'language_cortex')
    model = lang.create_language_cortex(vocab_size=100, use_parallel=True)
    model.lm_head = nn.Linear(256, 100)
    
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ce = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    for epoch in range(5):
        for tokens in data:
            result = model(tokens)
            target = torch.roll(tokens, -1, dims=1)[:, -1]
            pred = model.lm_head(result['features'])
            loss = ce(pred, target.long())
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
```

### 4.2 Auditory: SimCLR Contrastive Learning

```python
def contrastive_loss(z_i, z_j, temperature=0.1):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    sim = torch.cat([z_i, z_j], dim=0)
    sim = sim @ sim.T / temperature
    
    labels = torch.arange(z_i.size(0))
    labels = torch.cat([labels, labels], dim=0)
    
    return F.cross_entropy(sim, labels)
```

### 4.3 Vision: MAE Reconstruction

```python
def mask_random_patches(x, mask_ratio=0.75):
    B, C, T, H, W = x.shape
    mask = torch.rand(B, T, H, W).float() > mask_ratio
    masked = x.clone()
    masked = masked * mask.unsqueeze(1).float()
    return masked, mask
```

---

## 5. Parameter Configuration

### 5.1 Model Parameters

| Parameter | Language | Auditory | Vision |
|-----------|----------|----------|--------|
| vocab_size | 10000 | - | - |
| embed_dim | 256 | - | - |
| n_filters | - | 128 | - |
| hidden_dim | 256 | 256 | 64 |
| sample_rate | - | 16000 | - |
| n_experts | 4 | 3 | - |
| top_k | 1 | 1 | - |
| memory_slots | 7 | - | - |

### 5.2 Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|------------|
| lr | 0.001 | 0.0001-0.01 | Learning rate |
| batch_size | 8 | 1-64 | Batch size |
| epochs | 5 | 1-100 | Training epochs |
| warmup_steps | 500 | 0-1000 | Warmup |
| grad_clip | 1.0 | 0.1-10 | Gradient clipping |

### 5.3 Bio-Gating Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|------------|
| membrane_decay | 0.9 | 0.8-0.99 | Memory decay |
| emotion_scale | 0.2 | 0.1-0.5 | Emotion effect |
| mood_effect | 0.3 | 0.1-0.5 | Mood effect |

---

## 6. FAQ and Troubleshooting

### 6.1 Clarifications: Biological Metaphor vs Engineering

**Q: You named modules with biological names (Broca's Area, Amygdala, Synaptic Plasticity), but under the hood they're all matrix multiplications. What's the practical guidance for AI engineers?**

**A:分层原则**

| Layer | 生物命名 | 实际实现 | 调参价值 |
|-------|----------|----------|----------|
| **Architecture** | 稀疏门控 | Top-K routing | 有用→"可用稀疏门控" |
| **Implementation** | 杏仁核 | Dense + Sigmoid | 无区别→标准DL |
| **Tuning** | LTP/LTD | weight update | 无区别→直接调参 |

**生物隐喻只在架构设计阶段有用**：
- "可以用稀疏门控替代全连接" → 功耗↓
- TD学习 → 效率↑

**调参时不需要神经科学**：
- Loss不收敛？直接调lr、改数据
- 不需要查LTP论文

### 6.2 Scale Limitations

**Q: Human brain has 86B neurons, Simulacrum has only ~12M parameters. Will bio-mechanisms work at scale?**

**A: 大部分机制会失效或需要重构**

| 机制 | 小规模 | 大规模(70B) |
|------|--------|-------------|
| 7槽位WM | 有效 | 失效→分层记忆 |
| Bio-Gating | 效率优势 | 失效→标准MoE |
| 固定瓶颈 | 有效 | 失效→Dynamic Comp |

**保留的有效机制**：TD学习、模块化、事件驱动（仍有效）

### 6.3 Comparison with LLM

**Q: Compared to mainstream LLMs, besides low-power narrative, does Simulacrum have performance comparability?**

**A: 诚实的差距**

| 能力 | 7B LLM | Simulacrum当前 |
|------|--------|--------|
| 复杂推理 | 自回归生成 | 原型阶段 |
| 知识问答 | 预训练海量 | 无大规模训练 |
| 上下文 | 32K+ tokens | 早期框架 |

**Simulacrum的价值定位**：
- 低功耗实时场景（需验证）
- 可解释性（模块化）
- 持续学习（元学习）
- 特定场景补充，非LLM替代品

### 6.4 Cross-Modal Binding

**Q: Vision, Auditory, Language inputs are parallel. How to solve binding problem like "see snake + hear hiss → danger"?**

**A: Cross-Modal Binding机制** (见 §3.9)

```python
# 时序绑定：找到±500ms内的事件
# 跨模态注意力：Query-Key-Value
# 场景检测：统一表征→场景分类
```

---

### 6.5 Model Loading Fails

**Solution:**
```bash
python -c "import os; print(os.getcwd())"
ls -la core/
```

### 6.2 Out of Memory

**Solution:**
```python
batch_size = 2
model = create_language_cortex(vocab_size=5000, use_parallel=True)
```

### 6.3 Training Not Converging

**Solution:**
```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

---

## 7. Complete API Example

```python
import torch
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# Language
lang = load('core/language_cortex.py', 'language_cortex')
model = lang.create_language_cortex(vocab_size=1000, use_parallel=False)
tokens = torch.randint(0, 1000, (2, 16))
result = model(tokens, return_emotion=True)

# Auditory
audit = load('core/auditory_cortex.py', 'auditory_cortex')
model = audit.create_auditory_cortex(n_filters=128)
audio = torch.randn(1, 8000)
result = model(audio)

# Vision
censor = load('censor_bridge.py', 'censor_bridge')
model = censor.create_censor_vision('dual')
flow = torch.randn(1, 16, 32, 32)
rgb = torch.randn(1, 3, 32, 32)
result = model(flow, rgb)
```

---

## 8. References

### Neuroscience Classics

1. Broca, P. (1861). Remarques sur le siége de la faculté du langage articulé. *Bulletin de la Société Anatomique*, 6, 330-357.

2. Wernicke, C. (1874). *Der Aphasche Symptom Complex*. Breslau: Cohn & Weigert.

3. Hebb, D. O. (1949). *The Organization of Behavior*. New York: Wiley.

4. Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current. *Journal of Physiology*, 117(4), 500-544.

5. Miller, G. A. (1956). The magical number seven, plus or minus two. *Psychological Review*, 63(2), 81-97.

6. von Békésy, G. (1961). Concerning the pleasures and pains of stimulating the cortex. *Human Frontiers*, 116-132.

7. LeDoux, J. E. (2000). Emotion circuits in the brain. *Annual Review of Neuroscience*, 23, 155-184.

8. Kahneman, D. (2011). *Thinking, Fast and Slow*. New York: Farrar.

9. Schultz, W. (2007). Multiple dopamine functions at different time courses. *Annual Review of Neuroscience*, 30, 259-288.

10. Hickok, G., & Poeppel, D. (2007). The cortical organization of speech processing. *Nature Reviews Neuroscience*, 8(5), 393-402.

11. Plutchik, R. (1980). *Emotion: Psychoevolutionary Synthesis*. New York: Harper & Row.

12. Flavell, J. H. (1979). Metacognition and cognitive monitoring. *American Psychologist*, 34(10), 906-911.

13. Sherman, S. M. (2007). The thalamus. *Scholarpedia*, 2(9), 1587.

14. Eichenbaum, H. (2001). The hippocampus and declarative memory. *Nature Reviews Neuroscience*, 2, 51-60.

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-05-11 | Initial release |

---

## 10. Citation

```bibtex
@software{simulacrum,
  title={Simulacrum Technical Documentation},
  author={Simulacrum Lab},
  year={2026},
  version={1.0}
}
```