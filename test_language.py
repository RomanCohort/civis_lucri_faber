"""测试语言皮层 - 并行/串行"""
import os
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

base = os.path.dirname(__file__)
lang = load(os.path.join(base, 'core', 'language_cortex.py'), 'language_cortex')

import torch

print("=== 并行模式 (快) ===")
p = lang.create_language_cortex(use_parallel=True)
result = p(torch.randint(0, 5000, (1, 8)))
print(f"Features: {result['features'].shape}")
print(f"Surprise: {result['surprise']:.3f}")

print("\n=== 串行模式 (流式) ===")
s = lang.create_language_cortex(use_parallel=False)
result2 = s(torch.randint(0, 5000, (1, 8)))
print(f"Features: {result2['features'].shape}")
print(f"Surprise: {result2['surprise']:.3f}")
print(f"Working memory: {result2['working_memory'].shape}")

print("\n===== ALL PASSED =====")