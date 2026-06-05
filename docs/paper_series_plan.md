# CLF系列论文投稿计划

## Paper 1: ICONS (神经拟态听觉)
**定位:** Neuromorphic Computing / Spiking Neural Networks

**标题:** Bio-inspired Spiking Auditory Cortex: From Cochlea Mechanics to Cognitive Integration

**篇幅:** 4-6页 (ICONS short paper)

**核心内容:**
- Cochlea: traveling wave, IHC nonlinear, OHC amplification
- SNN: LIF, phase-locking, lateral inhibition
- STDP: online learning
- Binaural: ITD/ILD
- 实验: spike sparsity (<5%), phase-locking accuracy, emotion classification

**避开:**
- VTuber
- 其他脑区（amygdala、basal ganglia等）
- 语言生成

**提交时间:** ICONS通常7-8月，deadline约4-5月

---

## Paper 2: AAAI/CogSci (认知架构)
**定位:** Cognitive Architecture / Interactive Agents

**标题:** Civis Lucri-Faber: A Neuromorphic Cognitive Architecture for Interactive Conversational Agents

**篇幅:** 7页 (AAAI full paper) 或 6页 (CogSci)

**核心内容:**
- 14脑区架构图
- 事件驱动EventBus
- 多模态整合
- 认知闭环: perception → emotion → decision → action → memory → learning
- 认知扩展: phonological loop, auditory-motor coupling, meta-learning
- 实验: 情绪识别(66%) + 语言对话 + 任务决策

**定位包装:**
- 不说VTuber → "interactive conversational agent"
- 强调: 神经拟态基础 + 认知架构设计 + agent行为验证

**引用Paper 1:** "详见ICONS论文的spiking auditory细节"

**提交时间:**
- AAAI: deadline通常8-9月
- CogSci: deadline通常1-2月

---

## 时间线 (假设现在是2026年5月)

| 时间 | 任务 |
|------|------|
| 2026-05 | 完成Paper 1大纲 + 系统架构图 |
| 2026-06 | Paper 1正文 + 实验（spike可视化） |
| 2026-07 | Paper 1投稿ICONS |
| 2026-08 | 开始Paper 2，等ICONS结果 |
| 2026-09 | Paper 2投稿AAAI |
| 2026-10 | Paper 1结果（希望accept） |

---

## 互相引用策略

**Paper 1引用Paper 2:**
> "This auditory system is part of a larger cognitive architecture (Civis Lucri-Faber) described in our companion paper, enabling phonological working memory and auditory-motor coupling."

**Paper 2引用Paper 1:**
> "The spiking auditory cortex implements cochlea mechanics and spike-timing-dependent plasticity (see our ICONS paper for technical details), providing event-driven sensory input to the cognitive architecture."

---

## 需要准备的图表

**Paper 1 (ICONS):**
1. Cochlea architecture图（basilar membrane → IHC → OHC → SNN）
2. Spike waveform示例（音频输入 → spike输出）
3. STDP learning曲线（weight change vs timing）
4. Binaural ITD/ILD示意图

**Paper 2 (AAAI):**
1. 14脑区架构图（带EventBus连接）
2. 认知闭环流程图（perception → emotion → action）
3. 多模态整合示意图
4. Agent行为截图（对话交互）

---

## GitHub准备

**Paper 1 repo:** `neuromorphic-auditory-cortex`
- 只包含: spiking_auditory_cortex.py, snn_core.py, 实验代码
- README: 神经拟态角度

**Paper 2 repo:** `civis-lucri-faber` (原名)
- 完整系统
- README: 认知架构角度
- 引用Paper 1 repo