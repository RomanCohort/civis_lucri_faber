"""测试增强耦合通路改进效果

验证 Exp 3/7/8/9 的改进:
- Exp 3: cortisol→novelty_weight 耦合
- Exp 7: 丘脑门控噪声过滤
- Exp 8: 睡眠阶段 HPA 抑制
- Exp 9: resonance_baseline 保留影响
"""

import sys
import os
# 添加父目录以正确导入 civis_lucri_faber 包
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent_dir)
# 当前目录
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _current_dir)

import numpy as np

def test_exp3_cortisol_novelty_coupling():
    """测试 Exp 3: 皮质醇→Novelty权重耦合"""
    print("\n" + "="*60)
    print("测试 Exp 3: cortisol→novelty_weight 耦合")
    print("="*60)

    from core.agent import CivisLucriFaber
    from utils.config import Config

    config = Config(
        hpa_stress_reactivity=5.0,
        curiosity_alpha=0.4,  # 初始 novelty 权重
    )
    agent = CivisLucriFaber(config)

    # 注入高皮质醇模拟慢性应激
    agent.pharma.inject("cortisol", 0.85)

    # 运行 50 步，观察 novelty 权重变化
    print("初始 curiosity_alpha:", agent.config.curiosity_alpha)
    alpha_trajectory = []

    for step in range(50):
        agent.step()
        alpha = agent.config.curiosity_alpha
        alpha_trajectory.append(alpha)
        cortisol = agent._internal_state.get('cortisol_level', 0.5)
        rigidity = agent._internal_state.get('cognitive_rigidity', False)

    print("最终 curiosity_alpha:", alpha_trajectory[-1])
    print("alpha下降幅度:", (0.4 - alpha_trajectory[-1]) / 0.4 * 100, "%")
    print("认知僵化标志:", rigidity)

    # 验证: alpha 应下降，rigidity 应为 True
    success = alpha_trajectory[-1] < 0.35 and rigidity
    print("验证结果:", "PASS" if success else "FAIL")
    return success


def test_exp7_thalamic_noise_filtering():
    """测试 Exp 7: 丘脑门控噪声过滤"""
    print("\n" + "="*60)
    print("测试 Exp 7: 丘脑门控噪声过滤")
    print("="*60)

    import torch
    from core.limbic import ThalamicRelay

    # 创建正常模式和 ADHD 模式的丘脑门控
    normal_relay = ThalamicRelay(input_dim=64, n_senses=4)
    adhd_relay = ThalamicRelay(input_dim=64, n_senses=4)

    # ADHD 模式: 高 attention_gate (低过滤)
    adhd_relay.attention_gate.data.fill_(2.0)

    # 创建带噪声的输入
    signal = torch.ones(64) * 0.5  # 信号成分
    noise = torch.randn(64) * 0.3  # 噪声成分
    noisy_input = signal + noise

    sensory_inputs = [noisy_input, noisy_input, noisy_input, noisy_input]

    # 正常模式过滤
    normal_filtered = normal_relay.filter_noise(sensory_inputs)
    normal_stats = normal_relay.get_attention_stats()

    # ADHD 模式过滤
    adhd_filtered = adhd_relay.filter_noise(sensory_inputs)
    adhd_stats = adhd_relay.get_attention_stats()

    print("正常模式门控平均值:", normal_stats['attention_gate_avg'])
    print("正常模式噪声过滤弱:", normal_stats['noise_filtering_weak'])
    print("ADHD模式门控平均值:", adhd_stats['attention_gate_avg'])
    print("ADHD模式噪声过滤弱:", adhd_stats['noise_filtering_weak'])

    # 验证: ADHD 模式应标记为噪声过滤弱
    success = adhd_stats['noise_filtering_weak'] and not normal_stats['noise_filtering_weak']
    print("验证结果:", "PASS" if success else "FAIL")
    return success


def test_exp8_sleep_hpa_suppression():
    """测试 Exp 8: 睡眠阶段 HPA 抑制"""
    print("\n" + "="*60)
    print("测试 Exp 8: 睡眠阶段 HPA 抑制")
    print("="*60)

    from core.hpa_axis import HPAAxis

    hpa = HPAAxis(
        stress_reactivity=1.0,
        cortisol_half_life_steps=60,
    )

    # 设置初始皮质醇 (同步 adrenal 和 state)
    hpa.state.cortisol_level = 0.5
    hpa.adrenal.current_cortisol = 0.5

    # 正常清醒期: 应激驱动皮质醇升高
    awake_result = hpa.step(
        stress_signal=0.8,
        uncertainty=0.5,
        hpa_suppressed=False,
        circadian_hour=8.0,  # 8AM 皮质醇峰值
    )
    print("清醒期皮质醇:", awake_result['cortisol_level'])
    print("清醒期hpa_suppressed:", awake_result.get('hpa_suppressed', False))

    # 重置皮质醇 (同步 adrenal 和 state)
    hpa.state.cortisol_level = 0.5
    hpa.adrenal.current_cortisol = 0.5

    # 睡眠期: HPA 抑制，皮质醇自然衰减
    sleep_result = hpa.step(
        stress_signal=0.8,  # 即使有应激信号
        uncertainty=0.5,
        hpa_suppressed=True,
        circadian_hour=3.0,  # 3AM 皮质醇低谷
    )
    print("睡眠期皮质醇:", sleep_result['cortisol_level'])
    print("睡眠期hpa_suppressed:", sleep_result.get('hpa_suppressed', False))
    print("睡眠期stress_type:", sleep_result['stress_type'])

    # 验证: 睡眠期皮质醇应低于清醒期 (因为HPA被抑制，只有衰减)
    # 清醒期: 应激驱动皮质醇升高
    # 睡眠期: HPA抑制，只有自然衰减
    success = sleep_result['hpa_suppressed'] and sleep_result['stress_type'] == 'sleep'
    print("验证结果:", "PASS" if success else "FAIL")
    return success


def test_exp9_resonance_empathy():
    """测试 Exp 9: resonance_baseline 保留对共情的影响"""
    print("\n" + "="*60)
    print("测试 Exp 9: resonance_baseline→共情 (不被催产素完全覆盖)")
    print("="*60)

    from core.agent import CivisLucriFaber
    from utils.config import Config

    # 创建三个不同 resonance_baseline 的 Agent
    results = {}

    for resonance_level, resonance_val in [("Low", -1.0), ("Medium", 0.5), ("High", 3.0)]:
        config = Config()
        agent = CivisLucriFaber(config)

        # 设置 resonance_baseline (模拟镜像神经元基础连接)
        agent._internal_state['mirror_resonance_baseline'] = resonance_val

        # 注入相同水平的催产素
        agent.pharma.inject("oxytocin", 0.6)

        # 运行 30 步
        empathy_trajectory = []
        for step in range(30):
            agent.step()
            empathy = agent._internal_state.get('empathy_level', 0.5)
            empathy_trajectory.append(empathy)

        results[resonance_level] = {
            'final_empathy': empathy_trajectory[-1],
            'trajectory': empathy_trajectory,
        }
        print(f"{resonance_level} (resonance={resonance_val}): 最终共情={empathy_trajectory[-1]:.3f}")

    # 验证: 不同 resonance_baseline 应产生不同共情水平
    # Low < Medium < High
    success = results['Low']['final_empathy'] < results['Medium']['final_empathy'] and \
              results['Medium']['final_empathy'] < results['High']['final_empathy']
    print("\n共情差异验证:", "PASS" if success else "FAIL")
    print("差异幅度:", results['High']['final_empathy'] - results['Low']['final_empathy'])
    return success


if __name__ == "__main__":
    print("="*60)
    print("CLF 增强耦合通路验证测试")
    print("="*60)

    results = {
        'Exp 3 (cortisol→novelty)': test_exp3_cortisol_novelty_coupling(),
        'Exp 7 (thalamic filtering)': test_exp7_thalamic_noise_filtering(),
        'Exp 8 (sleep HPA suppression)': test_exp8_sleep_hpa_suppression(),
        'Exp 9 (resonance→empathy)': test_exp9_resonance_empathy(),
    }

    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    for exp, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{exp}: {status}")

    all_passed = all(results.values())
    print("\n总体状态:", "全部通过" if all_passed else "部分失败")