"""测试情绪系统增强功能"""
import sys
import os

# 添加项目根目录
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from civis_lucri_faber.utils.config import load_config
from civis_lucri_faber.core.agent import CivisLucriFaber


def test_emotion_integration():
    """测试情绪系统集成"""
    print("=" * 60)
    print("Test: Emotion System Integration")
    print("=" * 60)

    config = load_config(seed=42)
    agent = CivisLucriFaber(config=config)

    # Test 1: 默认调用（无情绪输入）
    print("\n[1] step() - without emotion input")
    state = agent.step()
    print(f"    Status: {state.status}, Step: {state.step}")

    # Test 2: 带情绪输入
    print("\n[2] step(user_sentiment=0.5) - positive user sentiment")
    state2 = agent.step(user_sentiment=0.5)
    print(f"    Status: {state2.status}, Step: {state2.step}")

    # Test 3: 带负面情绪
    print("\n[3] step(user_sentiment=-0.5) - negative user sentiment")
    state3 = agent.step(user_sentiment=-0.5)
    print(f"    Status: {state3.status}, Step: {state3.step}")

    # Test 4: 检查内部状态
    print("\n[4] Check internal state")
    print(f"    Keys: {list(agent._internal_state.keys())}")
    print(f"    mood_valence: {agent._internal_state.get('mood_valence', 'N/A')}")
    print(f"    mood_arousal: {agent._internal_state.get('mood_arousal', 'N/A')}")
    print(f"    current_emotion: {agent._internal_state.get('current_emotion', 'N/A')}")

    # Test 5: 运行多步并检查统计
    print("\n[5] Run 10 more steps with alternating emotions")
    for i in range(10):
        sentiment = 0.3 if i % 2 == 0 else -0.3
        agent.step(user_sentiment=sentiment)

    stats = agent.get_full_statistics()
    emotion_stats = stats.get('advanced_emotion', {})
    print(f"    Advanced emotion stats: {emotion_stats}")

    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_emotion_integration()