"""pytest configuration - add paths before test collection"""
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
core_dir = os.path.join(root_dir, 'core')
personality_dir = os.path.join(root_dir, 'core', 'personality')
utils_dir = os.path.join(root_dir, 'utils')
parent_dir = os.path.dirname(root_dir)

for p in [core_dir, personality_dir, utils_dir, root_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)
