"""小脑-脊髓系统测试"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from civis_lucri_faber.core.cerebello_spinal import (
    CerebelloSpinalCoordination,
    SpinalCord,
    Cerebellum,
    CentralPatternGenerator,
    ReflexPathway,
    create_cerebello_spinal,
)


def test_cpg():
    """测试中央模式生成器"""
    print("=== Test Central Pattern Generator ===")

    cpg = CentralPatternGenerator(n_joints=4, osc_frequency=1.0)

    # 设置步态 (对角线行走)
    cpg.set_phase_offset(0, 0)      # 左前
    cpg.set_phase_offset(1, np.pi)   # 右后
    cpg.set_phase_offset(2, np.pi)   # 左后
    cpg.set_phase_offset(3, 0)       # 右前

    cpg.activate()

    print("Gait pattern (first 3 steps):")
    for i in range(3):
        pos, vel = cpg.forward()
        print(f"  Step {i+1}: {pos.detach().numpy().round(2)}")

    cpg.deactivate()
    print("[PASS] CPG test\n")


def test_reflex():
    """测试反射通路"""
    print("=== Test Reflex Pathway ===")

    reflex = ReflexPathway(n_joints=4)

    # 测试肌梭反射
    joint_angles = np.array([0.5, 0.3, 0.4, 0.2])
    joint_velocities = np.array([0.1, -0.1, 0.05, -0.05])

    stretch_reflex = reflex.compute_stretch_reflex(
        np.array([joint_angles[0]]),
        np.array([joint_velocities[0]]),
    )
    print(f"Stretch reflex: {float(stretch_reflex):.3f}")

    # 测试屈肌反射
    pain_signal = 0.9
    withdrawal = reflex.compute_withdrawal_reflex(pain_signal)
    print(f"Withdrawal reflex (pain={pain_signal}): {withdrawal}")

    # 综合反射
    sensory_input = np.array([pain_signal])
    combined = reflex.compute_reflex(
        joint_angles,
        joint_velocities,
        sensory_input,
    )
    print(f"Combined reflex: {combined.round(3)}")
    print("[PASS] Reflex test\n")


def test_spinal():
    """测试脊髓系统"""
    print("=== Test Spinal Cord ===")

    spinal = SpinalCord(n_joints=4)

    # 激活CPG行走
    spinal.cpg.set_phase_offset(0, 0)
    spinal.cpg.set_phase_offset(1, np.pi/2)
    spinal.cpg.set_phase_offset(2, np.pi)
    spinal.cpg.set_phase_offset(3, 3*np.pi/2)

    print("Walking (CPG active):")
    for i in range(3):
        cmd = spinal.forward(cpg_command=True)
        print(f"  Step {i+1}: joints={cmd.joint_angles.round(3)}")

    # 测试反射
    spinal.joint_positions = np.array([0.5, 0.3, 0.4, 0.2])
    spinal.joint_velocities = np.array([0.2, -0.1, 0.1, -0.1])

    cmd = spinal.forward(cpg_command=False, reflex_input=np.array([0.8]))
    print(f"With reflex: {cmd.muscle_activations.round(3)}")
    print("[PASS] Spinal test\n")


def test_cerebellum():
    """测试小脑"""
    print("=== Test Cerebellum ===")

    cerebellum = Cerebellum(sensory_dim=64, n_motor_joints=4)

    # 正常模式
    sensory = torch.randn(1, 64)
    result = cerebellum(sensory)

    print(f"Motor command: {result['motor_command'].shape}")
    print(f"Predicted error: {result['predicted_error']:.3f}")

    # 时序模式
    seq = torch.randn(2, 10, 64)  # batch=2, seq=10, dim=64
    result = cerebellum(sensory=torch.randn(2, 64), mode="sequence")
    print(f"Sequence output: {result['motor_command'].shape}")

    # 监督学习
    coord = CerebelloSpinalCoordination(sensory_dim=64, n_joints=4)
    desired = torch.randn(1, 4)
    error = coord.learn_motor_program(sensory, desired)
    print(f"Learning error: {error:.3f}")

    stats = cerebellum.get_statistics()
    print(f"Stats: {stats}")
    print("[PASS] Cerebellum test\n")


def test_coordination():
    """测试小脑-脊髓协同"""
    print("=== Test Cerebello-Spinal Coordination ===")

    coord = create_cerebello_spinal(sensory_dim=64, n_joints=4)

    sensory = torch.randn(1, 64)

    # 只计算不执行
    result = coord(sensory, execute=False)
    print(f"Planning mode: pred_error={result['predicted_error']:.3f}")

    # 执行
    result = coord(sensory, execute=True)
    print(f"Execute mode:")
    print(f"  motor_output: {result['motor_command'].shape}")
    print(f"  predicted_error: {result['predicted_error']:.3f}")
    print(f"  actual_error: {result['actual_error']:.3f}")

    # 误差适应
    coord.adapt_to_error(sensory, 0.3)
    print("Adapted to error")

    # CPG行走模式
    coord.enable_cpg_walk()
    spinal_result = coord.spinal.forward(cpg_command=True)
    print(f"CPG walk: {spinal_result.joint_angles.round(2)}")

    summary = coord.get_summary()
    print(f"Summary: {summary}")
    print("[PASS] Coordination test\n")


def test_motor_learning():
    """测试运动学习"""
    print("=== Test Motor Learning ===")

    coord = create_cerebello_spinal(sensory_dim=64, n_joints=4)

    # 监督学习一系列运动
    errors = []
    for i in range(5):
        sensory = torch.randn(1, 64)
        desired = torch.randn(1, 4)

        error = coord.learn_motor_program(sensory, desired)
        errors.append(error)
        print(f"  Episode {i+1}: error={error:.3f}")

    avg_error = np.mean(errors)
    print(f"Average error: {avg_error:.3f}")
    print("[PASS] Motor Learning test\n")


def test_reflex_learning():
    """测试反射学习"""
    print("=== Test Reflex Learning ===")

    coord = create_cerebello_spinal(sensory_dim=64, n_joints=4)

    # 模拟意外干扰
    disturbances = [0.2, 0.5, 0.8, 0.9, 0.7]
    for d in disturbances:
        result = coord.learn_reflex(
            sensory=torch.randn(1, 64),
            unexpected_disturbance=d,
        )
        print(f"  Disturbance={d:.1f}: reflex={result['reflex_response'].round(2)}")

    print("[PASS] Reflex Learning test\n")


if __name__ == "__main__":
    test_cpg()
    test_reflex()
    test_spinal()
    test_cerebellum()
    test_coordination()
    test_motor_learning()
    test_reflex_learning()
    print("=" * 50)
    print("All cerebello-spinal tests passed!")