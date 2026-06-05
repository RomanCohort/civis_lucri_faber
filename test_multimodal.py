"""测试多模态感知系统 - 直接导入"""
import sys
import os

# 避免导入core/__init__
# 直接读取模块
import importlib.util

def load_module_directly(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 加载多模态感知
multimodal_path = os.path.join(os.path.dirname(__file__), 'core', 'multimodal_perception.py')
multimodal = load_module_directly(multimodal_path, 'multimodal_perception')

# 加载Censor桥接
censor_path = os.path.join(os.path.dirname(__file__), 'censor_bridge.py')
censor = load_module_directly(censor_path, 'censor_bridge')

import torch

# 重定向
create_multimodal_perception = multimodal.create_multimodal_perception
AuditoryCortex = multimodal.AuditoryCortex
LanguageCortex = multimodal.LanguageCortex


def test_auditory():
    """测试听觉"""
    print("=== 听觉测试 ===")
    auditory = AuditoryCortex()

    # 模拟音频: 1秒, 16kHz
    audio = torch.randn(1, 16000)

    result = auditory.process_audio(audio)
    print(f"Features: {result['features'].shape}")
    print(f"Valence: {result['valence'].item():.3f}")
    print(f"Arousal: {result['arousal'].item():.3f}")
    print(f"Pleasantness: {result['pleasantness'].item():.3f}")
    print("听觉 OK!\n")

def test_language():
    """测试语言"""
    print("=== 语言测试 ===")
    language = LanguageCortex(vocab_size=10000)

    # 模拟文本tokens
    text_tokens = torch.randint(0, 10000, (1, 10))

    result = language.process_text(text_tokens)
    print(f"Features: {result['features'].shape}")
    print(f"Valence: {result['valence'].item():.3f}")
    print(f"Arousal: {result['arousal'].item():.3f}")
    print("语言 OK!\n")

def test_multimodal():
    """测试完整多模态感知"""
    print("=== 多模态感知测试 ===")
    perception = create_multimodal_perception()

    # 测试只用视觉
    print("1. 仅视觉:")
    optical_flow = torch.randn(1, 2, 8, 32, 32)
    rgb_ppg = torch.randn(1, 6, 8, 32, 32)
    result = perception(optical_flow=optical_flow, rgb_ppg=rgb_ppg)
    print(f"   Salience: {result['salience']:.3f}")
    print(f"   Emotion: {result['emotion']['emotion'] if result['emotion'] else None}")

    # 测试只用听觉
    print("2. 仅听觉:")
    audio = torch.randn(1, 16000)
    result = perception(audio=audio)
    print(f"   Salience: {result['salience']:.3f}")
    print(f"   Emotion: {result['emotion']['emotion'] if result['emotion'] else None}")

    # 测试只用语言
    print("3. 仅语言:")
    text_tokens = torch.randint(0, 10000, (1, 10))
    result = perception(text_tokens=text_tokens)
    print(f"   Salience: {result['salience']:.3f}")
    print(f"   Emotion: {result['emotion']['emotion'] if result['emotion'] else None}")

    # 测试多模态融合
    print("4. 视觉+听觉+语言:")
    result = perception(
        optical_flow=optical_flow,
        rgb_ppg=rgb_ppg,
        audio=audio,
        text_tokens=text_tokens,
    )
    print(f"   Salience: {result['salience']:.3f}")
    if result['emotion']:
        e = result['emotion']
        print(f"   Emotion: {e['emotion']}")
        print(f"   Valence: {e['valence'].item():.3f}")
        print(f"   Arousal: {e['arousal'].item():.3f}")
        print(f"   Intensity: {e['intensity'].item():.3f}")

    print("\n多模态感知 OK!")

if __name__ == "__main__":
    test_auditory()
    test_language()
    test_multimodal()
    print("\n===== ALL TESTS PASSED =====")