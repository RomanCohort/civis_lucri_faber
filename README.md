# Civis Lucri-Faber 🧠

> 生物启发式AI智能体系统 | Bio-Inspired Autonomous AI Agent

[English](./README_EN.md) | [中文](./docs/TOTAL_README.md)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT/)

Civis Lucri-Faber（拉丁语"追求财富的工匠"）是一个整合了**10+种生物机制**的AI智能体系统。

## 核心特性

| 机制 | 功能 | 生物对应 |
|------|------|---------|
| Curiosity | 自主探索 | 多巴胺奖励 |
| Information Gain | 信息增益 | 好奇本能 |
| Meta Learning | 快速适应 | 元学习 |
| Self Alignment | 自对齐 | 自我反思 |
| Thermodynamics | 经济生存 | 能量代谢 |
| Personality | 心理人格 | 存在三象性 |
| Neuromodulation | 神经调质 | 多巴胺/血清素 |
| Epigenetic | 表观遗传 | DNA甲基化 |
| Metabolic | 资源预算 | 代谢限制 |

## 快速开始

```python
from civis_lucri_faber.core.agent import CivisLucriFaber

agent = CivisLucriFaber()
states = agent.run_episodes(n_episodes=10)

# 获取统计
stats = agent.get_full_statistics()
```

## 架构一览

```
CivisLucriFaber
├── CuriosityEngine (探索)
├── InformationGainCalculator (信息增益)
├── MetaLearner (元学习)
├── SelfAlignmentModule (自对齐)
├── ThermodynamicsSystem (生存)
└── PersonalityModule (人格)
    ├── TripartiteCompetitiveEngine
    ├── StreamingIdentityCore
    ├── RelationalEmbedding
    ├── AttentionGating
    ├── MotivationSurvivalSystem
    ├── NeuromodulationSystem
    └── EpigeneticLearner
```

## 安装

```bash
pip install -r requirements.txt
python main.py
```

## 文档

详细技术文档：[docs/TOTAL_README.md](./docs/TOTAL_README.md)

## 参考

- 好奇机制: [DeepLyapunov](https://arxiv.org/abs/2401.02124)
- 信息增益: [ICML 2017](https://proceedings.mlr.press/v80/guhebert17a.html)
- Meta Learning: [MAML](https://arxiv.org/abs/1805.11000)

---
