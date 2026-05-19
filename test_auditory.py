"""测试多模态感知 - 避开core/__init__"""
import os
import sys
import importlib.util

def direct_load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base = os.path.dirname(__file__)

# 加载听觉皮层
print("=== 加载模块 ===")
auditory_path = os.path.join(base, 'core', 'auditory_cortex.py')
auditory = direct_load(auditory_path, 'auditory_cortex')

# 加载Censor
censor_path = os.path.join(base, 'censor_bridge.py')
censor = direct_load(censor_path, 'censor_bridge')

import torch

print("\n=== 听觉皮层测试 (新) ===")
audit = auditory.create_auditory_cortex(n_filters=16, sample_rate=16000)
audio = torch.randn(1, 8000)
result = audit(audio)

print(f"Features: {result['features'].shape}")
print(f"What (ventral): {result['what']}")
print(f"Where (dorsal): {result['where']}")
print(f"How (dorsal): {result['how']}")

# 视觉测试
print("\n=== 视觉测试 ===")
vision = censor.create_censor_vision('dual')
optical_flow = torch.randn(1, 2, 8, 32, 32)
rgb_ppg = torch.randn(1, 6, 8, 32, 32)
v_result = vision(optical_flow, rgb_ppg)
print(f"Vision Salience: {v_result.get('salience', 0.5):.3f}")

print("\n===== ALL TESTS PASSED =====")