"""Curiosity Engine Tests

Run: python -m pytest tests/test_curiosity.py -v
"""
import os
import sys

import numpy as np
import pytest

# Add core to path directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

import torch

from curiosity import (
    CuriosityEngine,
    ExplorationGoal,
    GoalEncoder,
    LearnedNoveltyEngine,
    SimpleNoveltyCalculator,
)


class TestSimpleNoveltyCalculator:
    """Test simple novelty calculator"""

    def test_compute_novelty_empty_history(self):
        """Novelty should be 1.0 for empty history"""
        calc = SimpleNoveltyCalculator()
        goal = ExplorationGoal(id="test", description="new goal")
        novelty = calc.compute(goal, [])
        assert novelty == 1.0

    def test_compute_novelty_no_overlap(self):
        """Novelty should be 1.0 for no word overlap"""
        calc = SimpleNoveltyCalculator()
        goal = ExplorationGoal(id="test", description="completely new topic")
        history = [ExplorationGoal(id="h1", description="old stuff")]
        novelty = calc.compute(goal, history)
        assert novelty == 1.0

    def test_compute_novelty_full_overlap(self):
        """Novelty should be 0.0 for full overlap"""
        calc = SimpleNoveltyCalculator()
        goal = ExplorationGoal(id="test", description="same goal")
        history = [ExplorationGoal(id="h1", description="same goal")]
        novelty = calc.compute(goal, history)
        assert novelty == 0.0


class TestCuriosityEngine:
    """Test main curiosity engine"""

    def test_init(self):
        """Test initialization"""
        engine = CuriosityEngine(alpha=0.4, beta=0.3, gamma=0.3)
        assert engine.alpha == 0.4
        assert engine.beta == 0.3
        assert engine.gamma == 0.3

    def test_generate_candidate_goals(self):
        """Test goal generation"""
        engine = CuriosityEngine()
        candidates = engine.generate_candidate_goals(n=5)
        assert len(candidates) == 5
        assert all(isinstance(g, ExplorationGoal) for g in candidates)

    def test_select_goal(self):
        """Test goal selection"""
        engine = CuriosityEngine(exploration_rate=0.0)  # Greedy
        candidates = engine.generate_candidate_goals(n=3)
        selected = engine.select_goal(candidates)
        assert isinstance(selected, ExplorationGoal)
        assert selected.id in [g.id for g in candidates]

    def test_update_reward(self):
        """Test reward update"""
        engine = CuriosityEngine()
        candidates = engine.generate_candidate_goals(n=1)
        goal = candidates[0]
        engine.goal_history.append(goal)
        engine.update_reward(goal.id, 0.8)
        assert goal.completed is True

    def test_reset(self):
        """Test reset"""
        engine = CuriosityEngine()
        engine.goal_history.append(ExplorationGoal(id="test", description="test"))
        engine.reset()
        assert len(engine.goal_history) == 0


class TestGoalEncoder:
    """Test goal encoder"""

    def test_forward_cpu(self):
        """Test forward pass on CPU"""
        encoder = GoalEncoder(vocab_size=1000, embedding_dim=64, hidden_dim=128)
        tokens = torch.randint(0, 1000, (1, 10))
        mu, logvar = encoder(tokens)
        assert mu.shape == (1, 64)
        assert logvar.shape == (1, 64)


class TestLearnedNoveltyEngine:
    """Test learned novelty engine"""

    def test_compute_novelty_fallback(self):
        """Test fallback when no history"""
        engine = LearnedNoveltyEngine()
        tokens = torch.randint(0, 1000, (10,))
        novelty = engine.compute_novelty(tokens, use_learned=False)
        assert novelty >= 0.0

    def test_add_history(self):
        """Test adding history"""
        engine = LearnedNoveltyEngine()
        embedding = np.random.randn(128)
        engine.add_history(embedding)
        assert len(engine.goal_history) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
