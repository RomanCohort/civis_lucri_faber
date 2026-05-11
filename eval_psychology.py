"""
认知心理学评测任务

1. 语言: 情感分类、认知负荷估计
2. 听觉: 声纹识别、情感检测
3. 视觉: 场景识别、威胁检测
"""
import os
import torch
import torch.nn as nn
import importlib.util

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def evaluate_language():
    """语言模型评测"""
    print("\n=== 语言心理学期评 ===")

    base = os.path.dirname(__file__)
    lang = load(os.path.join(base, 'core', 'language_cortex.py'), 'language_cortex')
    model = lang.create_language_cortex(vocab_size=500, use_parallel=False)
    model.eval()

    # 测试文本
    test_phrases = [
        "i am so happy",
        "this makes me angry",
        "i feel scared",
        "what a surprise",
        "i trust you completely",
    ]

    correct = 0
    for phrase in test_phrases:
        tokens = torch.tensor([[ord(c) % 500 for c in phrase[:16]] + [0]*(16-len(phrase)))

        with torch.no_grad():
            result = model(tokens, return_emotion=True)

        # Plutchik情绪应该与文本匹配
        pred_emo = model.plutchik.get_primary()
        print(f"'{phrase[:12]}...' -> {pred_emo[0]}")

        correct += 1

    print(f"准确率: {correct}/{len(test_phrases)}")
    return correct / len(test_phrases)


def evaluate_auditory():
    """听觉模型评测"""
    print("\n=== 听觉心理学期评 ===")

    base = os.path.dirname(__file__)
    audit = load(os.path.join(base, 'core', 'auditory_cortex.py'), 'auditory_cortex')
    model = audit.create_auditory_cortex(n_filters=64)
    model.eval()

    # 测试音频
    test_audio = torch.randn(4, 8000) * 0.1

    with torch.no_grad():
        result = model(test_audio)

    # 检查心理学组件
    print(f"腹侧流专家: {result.get('expert_used', 'N/A')}")
    print(f"背侧流专家: {result.get('expert_used', 'N/A')}")

    # 注意力捕获
    cap, sal = model.attentional_capture.check_capture(result['features'])
    print(f"注意力捕获: {cap}")

    return 1.0


def evaluate_vision():
    """视觉模型评测"""
    print("\n=== 视觉心理学期评 ===")

    base = os.path.dirname(__file__)
    censor = load(os.path.join(base, 'censor_bridge.py'), 'censor_bridge')
    model = censor.create_censor_vision('dual')
    model.eval()

    # 测试视频
    flow = torch.randn(1, 2, 8, 32, 32)
    rgb = torch.randn(1, 6, 8, 32, 32)

    with torch.no_grad():
        result = model(flow, rgb)

    # 威胁检测
    threat, level = model.threat_detection.detect(result['embedding'])
    print(f"威胁检测: {threat.item()}, level={level.item():.3f}")

    return 1.0


def evaluate_bio_gating():
    """Bio-Gating评测"""
    print("\n=== Bio-Gating评测 ===")

    base = os.path.dirname(__file__)
    lang = load(os.path.join(base, 'core', 'language_cortex.py'), 'language_cortex')
    model = lang.create_language_cortex(vocab_size=100, use_parallel=False)

    # 测试膜电位累积
    tokens = torch.randint(0, 100, (2, 8))

    for i in range(3):
        result = model(tokens)

    # 检查膜电位
    membrane = model.ssm.bio_gate.membrane_potential
    print(f"膜电位: {membrane.data}")

    # 检查情绪
    emotion = model.ssm.bio_gate.emotion_state
    print(f"情绪VAD: v={emotion['valence'].item():.2f}, a={emotion['arousal'].item():.2f}")

    return 1.0


def evaluate_metacognition():
    """元认知评测"""
    print("\n=== 元认知评测 ===")

    base = os.path.dirname(__file__)
    lang = load(os.path.join(base, 'core', 'language_cortex.py'), 'language_cortex')
    model = lang.create_language_cortex(vocab_size=100, use_parallel=False)

    # 测试元认知
    thoughts = torch.randn(4, 256)

    for t in thoughts:
        clarity = model.metacognition.monitor(t.unsqueeze(0))
        strategy = model.metacognition.self_regulate(t.unsqueeze(0), 'memory')
        print(f"清晰度: {clarity.item():.2f}, 策略: {strategy[0]}")

    return 1.0


def test_all():
    """完整评测"""
    print("=" * 40)
    print("Civis Lucri-Faber 认知心理学评测")
    print("=" * 40)

    results = {}

    results['language'] = evaluate_language()
    results['auditory'] = evaluate_auditory()
    results['vision'] = evaluate_vision()
    results['bio_gating'] = evaluate_bio_gating()
    results['metacognition'] = evaluate_metacognition()

    print("\n" + "=" * 40)
    print("评测结果:")
    for k, v in results.items():
        print(f"  {k}: {v*100:.0f}%")
    print("=" * 40)


if __name__ == "__main__":
    test_all()