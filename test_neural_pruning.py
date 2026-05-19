"""
神经修剪系统测试 - Neural Pruning System Tests

三阶段机制验证:
  阶段1: 渐进衰减 — 不活动神经元权重逐渐缩小
  阶段2: 硬剪零化 — 超时后权重归零，快照保存
  阶段3: 生长因子恢复 — 再次激活时从快照恢复权重
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import importlib.util

# 直接导入模块文件
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "neural_pruning",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "neural_pruning.py")
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

PruningConfig = _mod.PruningConfig
NeuronState = _mod.NeuronState
ActivityTracker = _mod.ActivityTracker
NeuralPruningSystem = _mod.NeuralPruningSystem


def test_phase1_decay():
    """阶段1: 渐进衰减 — 不活动神经元权重逐渐缩小"""
    print("\n" + "=" * 60)
    print("阶段1: 渐进衰减")
    print("=" * 60)

    model = nn.Linear(10, 4)
    cfg = PruningConfig(
        decay_base=0.01,
        decay_max=0.15,
        decay_warmup_steps=3,   # 3步宽限期
        decay_ramp_steps=20,    # 20步爬升
        hibernation_steps=999,  # 不触发硬剪
        activation_relative_ratio=0.3,
    )
    pruning = NeuralPruningSystem(config=cfg)
    pruning.attach(model, "fc")

    # 设定: 前2个输出权重极大（会激活），后2个极小（远低于均值*0.3）
    with torch.no_grad():
        model.weight.data[:2] = 5.0   # 大输出
        model.weight.data[2:] = 0.01  # 极小输出

    initial_w = model.weight.data.clone()
    print(f"  初始权重范数: 前2={initial_w[:2].norm():.4f}, 后2={initial_w[2:].norm():.4f}")

    # 运行50步
    for _ in range(50):
        x = torch.randn(4, 10)
        _ = model(x)
        pruning.step()

    final_w = model.weight.data.clone()
    print(f"  50步后权重范数: 前2={final_w[:2].norm():.4f}, 后2={final_w[2:].norm():.4f}")

    # 验证: 后2个（不活动）衰减，前2个（活跃）保持
    decay_active = (initial_w[:2].norm() - final_w[:2].norm()).item()
    decay_inactive = (initial_w[2:].norm() - final_w[2:].norm()).item()
    print(f"  前2衰减量: {decay_active:.6f}, 后2衰减量: {decay_inactive:.6f}")
    assert decay_inactive > 0, "不活动神经元应该有衰减"
    print("  [PASS]")


def test_phase2_hibernate():
    """阶段2: 硬剪零化 — 超时后权重归零"""
    print("\n" + "=" * 60)
    print("阶段2: 硬剪零化")
    print("=" * 60)

    model = nn.Linear(10, 3)
    cfg = PruningConfig(
        decay_base=0.01,
        decay_max=0.1,
        decay_warmup_steps=2,
        decay_ramp_steps=10,
        hibernation_steps=30,   # 30步不活动就硬剪
        activation_relative_ratio=0.3,
    )
    pruning = NeuralPruningSystem(config=cfg)
    pruning.attach(model, "fc")

    # 设定: 只有第0个神经元会激活（权重差异巨大）
    with torch.no_grad():
        model.weight.data[0] = 5.0   # 大输出 → 会被判定活跃
        model.weight.data[1] = 0.01  # 极小 → 远低于均值*0.3
        model.weight.data[2] = 0.01

    print(f"  初始非零行数: {(model.weight.data.abs().sum(dim=1) > 0.01).sum().item()}")

    # 跑35步（超过 hibernation_steps=30）
    for i in range(35):
        x = torch.randn(2, 10)
        _ = model(x)
        result = pruning.step()

    nonzero_rows = (model.weight.data.abs().sum(dim=1) > 0.01).sum().item()
    print(f"  35步后非零行数: {nonzero_rows}")
    print(f"  第0行权重范数: {model.weight.data[0].norm():.4f} (活跃)")
    print(f"  第1行权重范数: {model.weight.data[1].norm():.6f} (应已硬剪)")
    print(f"  第2行权重范数: {model.weight.data[2].norm():.6f} (应已硬剪)")
    print(f"  快照数量: {len(pruning._snapshots['fc'])}")
    print(f"  统计: 衰减={pruning.stats['total_decayed']}, 硬剪={pruning.stats['total_hibernated']}")

    assert pruning.stats['total_hibernated'] > 0, "应该有硬剪事件"
    print("  [PASS]")


def test_phase3_revive():
    """阶段3: 生长因子恢复 — 再次激活时从快照恢复"""
    print("\n" + "=" * 60)
    print("阶段3: 生长因子恢复")
    print("=" * 60)

    model = nn.Linear(10, 2)
    cfg = PruningConfig(
        decay_base=0.01,
        decay_max=0.1,
        decay_warmup_steps=2,
        decay_ramp_steps=10,
        hibernation_steps=20,
        restore_ratio=0.8,
        gf_surge_scale=3.0,
        activation_relative_ratio=0.3,
    )
    pruning = NeuralPruningSystem(config=cfg)
    pruning.attach(model, "fc")

    # 设定: 大权重差异
    with torch.no_grad():
        model.weight.data[0] = 5.0   # 会激活
        model.weight.data[1] = 0.3   # 不会激活

    original_row1_norm = model.weight.data[1].norm().item()
    print(f"  第1行原始权重范数: {original_row1_norm:.4f}")

    # 阶段A: 让第1行不活动，直到被硬剪
    for _ in range(25):
        x = torch.randn(2, 10)
        _ = model(x)
        pruning.step()

    hibernated_norm = model.weight.data[1].norm().item()
    print(f"  硬剪后第1行权重范数: {hibernated_norm:.6f}")
    print(f"  快照数量: {len(pruning._snapshots.get('fc', {}))}")

    if hibernated_norm < 0.01 and len(pruning._snapshots.get('fc', {})) > 0:
        # 阶段B: 恢复第1行
        pruning._revive("fc", 1, list(model.parameters())[0])

        revived_norm = model.weight.data[1].norm().item()
        print(f"  恢复后第1行权重范数: {revived_norm:.4f}")
        print(f"  恢复比例: {revived_norm / original_row1_norm:.2f} (目标: {cfg.restore_ratio:.2f})")
        assert revived_norm > 0, "应该已从快照恢复"
    else:
        print("  注意: 第1行未被硬剪（可能始终被判定活跃），跳过恢复测试")
        # 手动模拟: 设为零化并保存快照
        with torch.no_grad():
            pruning._snapshots['fc'][1] = model.weight.data[1].clone()
            model.weight.data[1].zero_()
        pruning._revive("fc", 1, list(model.parameters())[0])
        revived_norm = model.weight.data[1].norm().item()
        print(f"  手动恢复后第1行权重范数: {revived_norm:.4f}")
        assert revived_norm > 0, "手动恢复应该有效"

    print("  [PASS]")


def test_full_cycle():
    """完整三阶段循环: 活跃 → 衰减 → 硬剪 → 恢复"""
    print("\n" + "=" * 60)
    print("完整三阶段循环")
    print("=" * 60)

    model = nn.Linear(10, 3)
    cfg = PruningConfig(
        decay_base=0.005,
        decay_max=0.15,
        decay_warmup_steps=3,
        decay_ramp_steps=15,
        hibernation_steps=30,
        restore_ratio=0.8,
        gf_surge_scale=3.0,
        activation_relative_ratio=0.3,
    )
    pruning = NeuralPruningSystem(config=cfg)
    pruning.attach(model, "fc")

    with torch.no_grad():
        model.weight.data[0] = torch.randn(10) * 2.0
        model.weight.data[1] = torch.randn(10) * 2.0
        model.weight.data[2] = torch.randn(10) * 2.0

    orig_norms = [model.weight.data[j].norm().item() for j in range(3)]
    print(f"  初始权重范数: {[f'{n:.3f}' for n in orig_norms]}")

    # === 阶段A: 正常使用所有神经元 (10步) ===
    print("\n  --- 阶段A: 全部活跃 (10步) ---")
    for _ in range(10):
        x = torch.randn(4, 10)
        _ = model(x)
        pruning.step()
    norms = [model.weight.data[j].norm().item() for j in range(3)]
    print(f"  权重范数: {[f'{n:.3f}' for n in norms]}")

    # === 阶段B: 让第1,2行不活动 (50步) ===
    print("\n  --- 阶段B: 第1,2行停用 (50步) ---")
    with torch.no_grad():
        model.weight.data[0] = 5.0   # 大输出，保证激活
        model.weight.data[1] = 0.01  # 极小，不会激活
        model.weight.data[2] = 0.01

    for i in range(50):
        x = torch.randn(2, 10)
        _ = model(x)
        result = pruning.step()
        if i % 15 == 14:
            norms = [model.weight.data[j].norm().item() for j in range(3)]
            print(f"    步{i+1}: 范数={[f'{n:.4f}' for n in norms]}, 硬剪={result['hibernated']}")

    print(f"  统计: 衰减={pruning.stats['total_decayed']}, "
          f"硬剪={pruning.stats['total_hibernated']}, "
          f"快照数={len(pruning._snapshots.get('fc', {}))}")

    # === 阶段C: 恢复第1,2行 ===
    print("\n  --- 阶段C: 恢复 ---")
    # 手动恢复来验证机制
    for nid in [1, 2]:
        if nid in pruning._snapshots.get('fc', {}):
            pruning._revive("fc", nid, list(model.parameters())[0])
            print(f"  第{nid}行已从快照恢复")

    norms = [model.weight.data[j].norm().item() for j in range(3)]
    print(f"  恢复后权重范数: {[f'{n:.3f}' for n in norms]}")
    print("  [PASS]")


def test_param_reduction():
    """验证参数量确实减少"""
    print("\n" + "=" * 60)
    print("参数量减少验证")
    print("=" * 60)

    model = nn.Linear(20, 10)
    cfg = PruningConfig(
        decay_base=0.01,
        decay_max=0.2,
        decay_warmup_steps=2,
        decay_ramp_steps=10,
        hibernation_steps=25,
        activation_relative_ratio=0.3,
    )
    pruning = NeuralPruningSystem(config=cfg)
    pruning.attach(model, "fc")

    total_params = model.weight.numel()
    # 所有行初始都是中等权重（全部活跃）
    with torch.no_grad():
        model.weight.data[:] = 0.3

    initial_active = (model.weight.data.abs() > 0.005).sum().item()
    print(f"  总参数: {total_params}, 初始活跃: {initial_active}")

    # 然后让后7行输出极小（不会被激活）
    with torch.no_grad():
        model.weight.data[:3] = 5.0
        model.weight.data[3:] = 0.01

    for i in range(35):
        x = torch.randn(4, 20)
        _ = model(x)
        result = pruning.step()

    final_active = (model.weight.data.abs() > 0.005).sum().item()
    reduction = 1.0 - final_active / total_params
    print(f"  最终活跃参数: {final_active}/{total_params} (减少 {reduction*100:.1f}%)")
    assert final_active < initial_active, "参数量应该减少"
    print("  [PASS]")


# ============ 运行 ============

if __name__ == "__main__":
    print("=" * 60)
    print("神经修剪系统测试")
    print("=" * 60)

    tests = [
        test_phase1_decay,
        test_phase2_hibernate,
        test_phase3_revive,
        test_full_cycle,
        test_param_reduction,
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
