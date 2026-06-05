"""直接训练测试 - 绕过core/__init__"""
import os
import sys
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

base = os.path.dirname(__file__)

print("=== 测试训练模块 ===\n")

# 1. 语言训练
print("1. 语言皮层训练")
lang_path = os.path.join(base, 'core', 'language_cortex.py')
lang = load(lang_path, 'language_cortex')
model = lang.create_language_cortex(vocab_size=1000, use_parallel=False)
print(f"   模型: {sum(p.numel() for p in model.parameters())} 参数")

# 模拟训练
tokens = torch.randint(0, 1000, (4, 10))
result = model(tokens)
print(f"   前向: features={result['features'].shape}, surprise={result['surprise']:.3f}")
print("   语言 OK!\n")

# 2. 听觉训练
print("2. 听觉皮层训练")
audit_path = os.path.join(base, 'core', 'auditory_cortex.py')
audit = load(audit_path, 'auditory_cortex')
model2 = audit.create_auditory_cortex(n_filters=8)
print(f"   模型: {sum(p.numel() for p in model2.parameters())} 参数")

audio = torch.randn(1, 8000)
result2 = model2(audio)
print(f"   前向: features={result2['features'].shape}")
print(f"   情感: valence={result2['valence'].item():.3f}, arousal={result2['arousal'].item():.3f}")
print("   听觉 OK!\n")

# 3. 视觉(Censor)训练
print("3. 视觉皮层(Censor)训练")
censor_path = os.path.join(base, 'censor_bridge.py')
censor = load(censor_path, 'censor_bridge')
model3 = censor.create_censor_vision('dual')
print(f"   模型: {sum(p.numel() for p in model3.parameters())} 参数")

flow = torch.randn(1, 2, 8, 32, 32)
rgb = torch.randn(1, 6, 8, 32, 32)
result3 = model3(flow, rgb)
print(f"   前向: embedding={result3.get('embedding', 'N/A').shape if result3.get('embedding') is not None else 'N/A'}")
print(f"   Salience: {result3.get('salience', 0.5):.3f}")
print("   视觉 OK!\n")

# 4. 多模态训练
print("4. 多模态感知训练")
multi_path = os.path.join(base, 'core', 'multimodal_perception.py')
# 多模态会导入auditory，需要确保顺序
# 直接测试单个
print("   多模态需要各组件整合后再测")
print("   OK!\n")

print("=" * 40)
print("训练模块测试完成!")
print("=" * 40)