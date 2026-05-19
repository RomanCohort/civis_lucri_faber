"""
角回测试 — Angular Gyrus Tests

验证跨模态翻译器的核心功能：
1. 单模态 → 预测缺失模态
2. 双模态 → 翻译 + 融合
3. 三模态 → 完整翻译矩阵
4. 时序绑定
5. 场景检测
"""

import torch
import sys
import os
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "angular_gyrus",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "angular_gyrus.py")
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

AngularGyrus = _mod.AngularGyrus


def test_single_modality():
    """单模态输入 → 预测缺失模态"""
    print("\n" + "=" * 60)
    print("测试1: 单模态 → 预测缺失模态")
    print("=" * 60)

    ag = AngularGyrus()

    # 只有视觉输入
    result = ag({'vision': torch.randn(1, 768)})
    print(f"  存在模态: {result['present_modalities']}")
    print(f"  缺失模态预测: {list(result['predictions'].keys())}")
    print(f"  统一表征形状: {result['unified_repr'].shape}")

    assert result['n_present'] == 1
    assert result['n_predicted'] == 2  # audio + language
    assert 'audio' in result['predictions']
    assert 'language' in result['predictions']
    assert result['predictions']['audio'].shape == (1, 256)
    assert result['predictions']['language'].shape == (1, 128)
    print("  [PASS]")


def test_dual_modality():
    """双模态 → 翻译 + 融合"""
    print("\n" + "=" * 60)
    print("测试2: 双模态 → 翻译 + 融合")
    print("=" * 60)

    ag = AngularGyrus()

    result = ag({
        'vision': torch.randn(1, 768),
        'audio': torch.randn(1, 256),
    })

    print(f"  存在模态: {result['present_modalities']}")
    print(f"  翻译矩阵: {list(result['translations'].keys())}")

    # 应有 vision 和 audio 之间的双向翻译
    assert 'vision' in result['translations']
    assert 'audio' in result['translations']
    assert 'audio' in result['translations']['vision']
    assert 'vision' in result['translations']['audio']

    # 应预测 language
    assert 'language' in result['predictions']
    print(f"  预测 language 形状: {result['predictions']['language'].shape}")
    print(f"  统一表征形状: {result['unified_repr'].shape}")
    print("  [PASS]")


def test_triple_modality():
    """三模态 → 完整翻译矩阵"""
    print("\n" + "=" * 60)
    print("测试3: 三模态 → 完整翻译矩阵")
    print("=" * 60)

    ag = AngularGyrus()

    result = ag({
        'vision': torch.randn(1, 768),
        'audio': torch.randn(1, 256),
        'language': torch.randn(1, 128),
    })

    print(f"  存在模态: {result['present_modalities']}")
    print(f"  翻译矩阵维度: ", end="")
    for src in result['translations']:
        for tgt in result['translations'][src]:
            print(f"{src}→{tgt} ", end="")
    print()

    # 完整 3×3 翻译矩阵
    assert len(result['translations']) == 3
    for src in ['vision', 'audio', 'language']:
        assert src in result['translations']
        assert len(result['translations'][src]) == 3

    # 无缺失预测
    assert result['n_predicted'] == 0
    print(f"  无缺失预测: {result['n_predicted']}")
    print(f"  场景: {result['scene']['scene_name']}, 置信度: {result['scene']['confidence'].item():.4f}")
    print("  [PASS]")


def test_temporal_binding():
    """时序绑定"""
    print("\n" + "=" * 60)
    print("测试4: 时序绑定")
    print("=" * 60)

    ag = AngularGyrus()

    # T=0: 视觉事件
    ag({'vision': torch.randn(1, 768)}, timestamp=0.0)
    # T=0.1: 听觉事件（在时间窗口内）
    ag({'audio': torch.randn(1, 256)}, timestamp=0.1)
    # T=5.0: 语言事件（超出时间窗口）
    ag({'language': torch.randn(1, 128)}, timestamp=5.0)

    # 检查时序匹配
    matches = ag.temporal_buffer.find_temporal_matches('vision', 'audio', timestamp=0.0)
    print(f"  T=0 视觉→听觉匹配: {len(matches)} (时间窗口=0.5s)")
    assert len(matches) > 0, "时间窗口内应有匹配"

    matches_far = ag.temporal_buffer.find_temporal_matches('vision', 'language', timestamp=0.0)
    print(f"  T=0 视觉→语言匹配: {len(matches_far)} (超出窗口)")
    assert len(matches_far) == 0, "超出时间窗口应无匹配"

    # 缓冲区大小
    sizes = {mod: len(ag.temporal_buffer.buffers[mod]) for mod in ['vision', 'audio', 'language']}
    print(f"  缓冲区大小: {sizes}")
    print("  [PASS]")


def test_scene_detection():
    """场景检测"""
    print("\n" + "=" * 60)
    print("测试5: 场景检测")
    print("=" * 60)

    ag = AngularGyrus()

    result = ag({
        'vision': torch.randn(1, 768),
        'audio': torch.randn(1, 256),
        'language': torch.randn(1, 128),
    })

    scene = result['scene']
    print(f"  场景: {scene['scene_name']}")
    print(f"  置信度: {scene['confidence'].item():.4f}")
    print(f"  场景概率分布: ", end="")
    for i, name in enumerate(ag.scene_detector.SCENES):
        p = scene['scene_probs'][0, i].item()
        if p > 0.05:
            print(f"{name}={p:.3f} ", end="")
    print()

    assert scene['scene_name'] in ag.scene_detector.SCENES
    assert scene['confidence'].item() > 0
    print("  [PASS]")


def test_topdown_modulation():
    """自上而下调制"""
    print("\n" + "=" * 60)
    print("测试6: 自上而下调制 (PFC → Angular Gyrus)")
    print("=" * 60)

    ag = AngularGyrus()

    # 无 PFC 信号
    result_no_pfc = ag({'vision': torch.randn(1, 768)})
    unified_no_pfc = result_no_pfc['unified_repr']

    # 有 PFC 信号
    pfc_signal = torch.randn(1, 256)
    result_with_pfc = ag({'vision': torch.randn(1, 768)}, top_down_signal=pfc_signal)
    unified_with_pfc = result_with_pfc['unified_repr']

    # PFC 信号应该改变统一表征
    diff = (unified_no_pfc - unified_with_pfc).abs().mean().item()
    print(f"  无 PFC 统一表征范数: {unified_no_pfc.norm():.4f}")
    print(f"  有 PFC 统一表征范数: {unified_with_pfc.norm():.4f}")
    print(f"  差异: {diff:.4f}")
    print("  [PASS]")


def test_batch_processing():
    """批处理"""
    print("\n" + "=" * 60)
    print("测试7: 批处理")
    print("=" * 60)

    ag = AngularGyrus()

    result = ag({
        'vision': torch.randn(4, 768),
        'audio': torch.randn(4, 256),
        'language': torch.randn(4, 128),
    })

    print(f"  batch=4, unified_repr: {result['unified_repr'].shape}")
    assert result['unified_repr'].shape == (4, 256)
    print(f"  场景: {result['scene']['scene_name']}")
    print("  [PASS]")


# ============ 运行 ============

if __name__ == "__main__":
    print("=" * 60)
    print("角回测试 — Angular Gyrus Tests")
    print("=" * 60)

    tests = [
        test_single_modality,
        test_dual_modality,
        test_triple_modality,
        test_temporal_binding,
        test_scene_detection,
        test_topdown_modulation,
        test_batch_processing,
    ]

    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed} passed, {failed} failed")
    print("=" * 60)
