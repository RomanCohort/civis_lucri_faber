# Civis Lucri-Faber 🧠

> 生物启发式认知AI架构 | Bio-Inspired Cognitive AI System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

Civis Lucri-Faber（拉丁语"追求财富的工匠"）是一个整合了**15种脑区机制**、**认知心理学**和**自适应剪枝**的生物启发式AI系统。

---

## 目录

1. [概述](#概述)
2. [架构](#架构)
3. [脑区机制](#脑区机制)
4. [生物门控Bio-Gating](#生物门控bio-gating)
5. [认知心理学](#认知心理学)
6. [剪枝机制](#剪枝机制)
7. [训练](#训练)
8. [评估](#评估)
9. [快速开始](#快速开始)

---

## 概述

Civis Lucri-Faber是一个生物启发式AI系统，整合了：

| 特性 | 描述 |
|-------|------|
| **15种脑区机制** | 全脑区域模拟 |
| **生物门控Bio-Gating** | 情绪依赖的路由 |
| **认知心理学** | Plutchik、双过程、元认知 |
| **自适应剪枝** | ~65%算力节省 |

### 与Transformer对比

| 特性 | Transformer | Civis Lucri-Faber |
|------|-------------|------------------|
| 路由 | 全量Attention | **Top-1选择** |
| 记忆 | O(n)上下文 | **7槽限制** |
| 情绪 | 无 | **VAD+Plutchik** |
| 元认知 | 无 | **自我监控** |
| 剪枝 | 需手动 | **自适应** |
| **效率** | 基线 | **~65%节省** |

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Civis Lucri-Faber                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  语言皮层  │  │  听觉皮层  │  │  视觉     │ │
│  │  Cortex  │  │  Cortex  │  │  Censor   │ │
│  │  (7.6M) │  │  (1.0M)  │  │  (3.6M)  │ │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘ │
│        │             │               │       │
│        └─────────────┼──────────────┘       │
│                      ↓                    │
│            ┌───────────────┐              │
│            │   多模态     │              │
│            │   融合      │              │
│            └───────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### 参数量

| 模态 | 原始参数 | 剪枝后 |
|------|---------|---------|
| 语言 | 7.6M | ~1M |
| 听觉 | 1.0M | ~250K |
| 视觉 | 3.6M | ~1.8M |
| **总计** | **~12M** | **~4M** |

---

## 脑区机制

### 15种脑区机制

| # | 机制 | 组件 | 功能 |
|----|------|------|------|
| 1 | 耳蜗 | AuditoryCortex | 时频分析 |
| 2 | 下丘 | SubcorticalRelay | 门控中继 |
| 3 | 初级听觉皮层A1 | PrimaryAuditoryCortex | 声音处理 |
| 4 | 腹侧流 | VentralStream | "是什么"通路 |
| 5 | 背侧流 | DorsalStream | "��哪里"通路 |
| 6 | 杏仁核 | Emotion heads | 情绪感知 |
| 7 | 海马体 | Memory systems | 情境记忆 |
| 8 | 前额叶 | DualProcess | 规划 |
| 9 | 丘脑 | Censor fast path | 快速路由 |
| 10 | 视觉皮层 | SlowPathway | 精细视觉 |
| 11-15 | ... | ... | ... |

---

## 生物门控Bio-Gating

核心创新：情绪依赖的专家路由

```python
class BioGate(nn.Module):
    """
    生物门控: 内容 + 膜电位 + 情绪 + 心境
    """
    def forward(self, input_emb):
        # 1. 内容门控
        content_logits = self.content_gate(input_emb)
        
        # 2. 膜电位 (历史累积)
        membrane_effect = self.membrane_potential.unsqueeze(0)
        
        # 3. 情绪VAD
        emotion_effect = torch.tanh(self.emotion_vector.sum()) * 0.2
        
        # 4. 心境状态
        mood_effect = self.mood.mood_affect_decision(content_logits)
        
        # 综合
        gate_logits = content_logits + membrane_effect + emotion_effect + mood_effect
        
        # Top-1专家选择 (节省75%计算)
        return gate_logits.softmax(dim=-1).argmax()
```

### 三层情绪-认知系统

```
┌────────────────────────────────────────┐
│  第3层: 神经调节器                     │
│  多巴胺 → 奖励    血清素 → 情绪稳定    │
├────────────────────────────────────────┤
│  第2层: 心境状态 (持久)                │
│  乐观 / 焦虑 / 自信                  │
├────────────────────────────────────────┤
│  第1层: 生物门控 (即时)              │
│  内容 + 膜电位 + 情绪                │
└────────────────────────────────────────┘
```

---

## 认知心理学

### 整合的心理学理论

| 理论 | 组件 | 实现 |
|------|------|------|
| **Plutchik情绪轮** | PlutchikEmotion | 8种基本情绪 |
| **双过程理论** | DualProcessCognition | System 1/2 |
| **具身认知** | EmbodiedCognition | 身体状态影响感知 |
| **工作记忆** | CognitiveLoadManager | 7±2容量限制 |
| **情绪调节** | EmotionRegulation | 重评/压抑 |
| **认知偏差** | CognitiveBias | 确认/锚定 |
| **元认知** | Metacognition | 自我监控 |

### Plutchik情绪示例

```python
# 8种基本情绪
EMOTION_NAMES = ['joy', 'sadness', 'trust', 'disgust', 
               'fear', 'anger', 'surprise', 'anticipation']

# 情绪影响行为
joy = sigmoid(plutchik.emotion_vector[0])    # 喜悦 → 风险寻求
fear = sigmoid(plutchik.emotion_vector[4])    # 恐惧 → 风险规避
anger = sigmoid(plutchik.emotion_vector[5]) # 愤怒 → 快速决策
```

---

## 剪枝机制

### 自适应剪枝

| 类型 | 组件 | 节省 |
|------|------|------|
| **专家剪枝** | DynamicExpertPruner | 75% |
| **记忆剪枝** | WorkingMemory (7槽) | 57% |
| **滤波器剪枝** | CochlearFilterPruner | 75% |
| **突触Depression** | SynapticDepression | 渐进弱化 |
| **Hebbian增强** | OjaRule | 共同激活增强 |

### 计算效率

```python
# 之前: 4个专家全量计算
# 之后: 只选Top-1专家
savings = (n_experts - top_k) / n_experts  # 75%
```

---

## 训练

### 自监督学习

| 模态 | 方法 |
|------|------|
| 语言 | 下一个词预测 |
| 听觉 | SimCLR对比学习 |
| 视觉 | MAE重建 |

### 训练脚本

```bash
# 语言训练
python train_language.py

# 听觉训练  
python train_audio.py

# 视觉训练
python train_vision.py
```

---

## 评估

### 心理学期评

```bash
python eval_psychology.py
```

测试项目：
- Plutchik情绪识别
- 双过程任务难度
- 元认知自我监控
- 认知偏差应用
- Bio-Gating膜电位

### 应用

```bash
# 情绪对话
python emotional_dialogue.py
```

---

## 快速开始

### 安装

```bash
pip install torch numpy streamlit
```

### 文档

| 文件 | 描述 |
|------|------|
| [README_EN.md](./README_EN.md) | Overview (English) |
| [README.md](./README.md) | 概述 (中文) |
| [docs/TECHNICAL_EN.md](./docs/TECHNICAL_EN.md) | Technical Doc (English) |
| [docs/TECHNICAL.md](./docs/TECHNICAL.md) | 技术文档 (中文) |

### 运行监控界面

```bash
streamlit run monitor.py
```

### 加载模型

```python
import torch
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# 语言模型 (带Bio-Gating)
lang = load('core/language_cortex.py', 'language_cortex')
model = lang.create_language_cortex(vocab_size=1000, use_parallel=False)

# 前向传播
tokens = torch.randint(0, 1000, (2, 16))
result = model(tokens, return_emotion=True)
print(f"效价: {result['emotion_state']['valence']}")
print(f"唤醒度: {result['emotion_state']['arousal']}")
```

---

## 论文亮点

### 创新贡献

1. **15种脑区机制**: 首个整合的生物启发架构
2. **Bio-Gating**: 情绪依赖的路由（vs静态MoE）
3. **认知心理学**: 完整整合Plutchik、双过程、元认知
4. **自适应剪枝**: ~65%算力节省

### 推荐期刊

| 优先级 | 期刊 |
|--------|------|
| 1 | Neural Networks |
| 2 | Cognitive Computation |
| 3 | IEEE TNNLS |

---

## 文件结构

```
civis_lucri_faber/
├── core/
│   ├── language_cortex.py    # 语言 + Bio-Gating
│   ├── auditory_cortex.py   # 听觉 + 心理学
│   └── multimodal_*.py     # 融合
├── censor_bridge.py        # 视觉 + Censor
├── monitor.py              # Streamlit界面
├── train_*.py            # 训练脚本
├── eval_psychology.py     # 评估
├── emotional_dialogue.py # 应用
└── README*.md           # 文档
```

---

## 许可证

MIT License

---

## 引用

```bibtex
@software{civis_lucri_faber,
  title={Civis Lucri-Faber: Bio-Inspired Cognitive AI},
  author={Civis Lab},
  year={2026},
  url={https://github.com/civis-lucri-faber}
}
```