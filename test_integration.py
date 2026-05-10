"""
实际集成测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from civis_lucri_faber.core.agent import CivisLucriFaber

# 创建Agent
agent = CivisLucriFaber()

# 运行5步
states = agent.run_episodes(5, verbose=True)

# 获取统计
stats = agent.get_full_statistics()

print("\n" + "=" * 60)
print("FULL STATISTICS")
print("=" * 60)

# Personality stats
p = stats.get('personality', {})
print("\n[Personality Module]:")
for k, v in p.items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for k2, v2 in v.items():
            print(f"    - {k2}: {v2}")
    else:
        print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("TEST PASSED!")
print("=" * 60)