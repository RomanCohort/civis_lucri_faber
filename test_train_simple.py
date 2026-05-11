"""语言训练 - 简化测试"""
import os
import torch
import torch.nn as nn
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

base = os.path.dirname(__file__)
lang = load(os.path.join(base, 'core', 'language_cortex.py'), 'language_cortex')

# 简单测试
print("=== 测试 ===")
model = lang.create_language_cortex(vocab_size=100, use_parallel=True)
print(f"参数: {sum(p.numel() for p in model.parameters())}")

# 简单前向
tokens = torch.randint(0, 100, (4, 8))
result = model(tokens)
print(f"features: {result['features'].shape}")

# 添加LM头测试
model.lm_head = nn.Linear(64, 100)
pred = model.lm_head(result['features'])
print(f"LM pred: {pred.shape}")

# 损失
target = torch.randint(0, 100, (4,))
loss = nn.functional.cross_entropy(pred, target)
print(f"loss: {loss.item():.4f}")

# 反向
loss.backward()
print("反向 OK!")

print("\n训练模块可用了!")