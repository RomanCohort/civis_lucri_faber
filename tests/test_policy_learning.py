"""Policy Learning Tests"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

import numpy as np

from policy_learning import EpsilonGreedyBaseline, SimpleQLearning, UCBAction


class TestSimpleQLearning:
    """Test simple Q-learning"""

    def test_init(self):
        """Test initialization"""
        ql = SimpleQLearning(state_dim=4, n_actions=2, learning_rate=0.1)
        assert ql is not None

    def test_select_action(self):
        """Test action selection"""
        ql = SimpleQLearning(state_dim=4, n_actions=2, epsilon=0.0)
        state = np.random.randn(4)
        action = ql.select_action(state, training=False)
        assert action in [0, 1]

    def test_update(self):
        """Test update"""
        ql = SimpleQLearning(state_dim=4, n_actions=2)
        state = np.random.randn(4)
        action = 0
        reward = 1.0
        next_state = np.random.randn(4)
        done = False
        ql.update(state, action, reward, next_state, done)

    def test_get_q(self):
        """Test Q value - returns Q array for all actions"""
        ql = SimpleQLearning(state_dim=4, n_actions=2, epsilon=0.0)
        state = np.random.randn(4)
        q = ql.get_q(state)
        assert isinstance(q, np.ndarray)
        assert len(q) == 2


class TestEpsilonGreedyBaseline:
    """Test epsilon greedy"""

    def test_init(self):
        """Test initialization"""
        eg = EpsilonGreedyBaseline(n_actions=4, epsilon=0.1)
        assert eg.epsilon == 0.1

    def test_select_action(self):
        """Test action selection"""
        eg = EpsilonGreedyBaseline(n_actions=4, epsilon=0.0)
        state = np.random.randn(4)
        action = eg.select_action(state, training=False)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < eg.n_actions


class TestUCBAction:
    """Test UCB action"""

    def test_init(self):
        """Test initialization"""
        ucb = UCBAction(n_actions=4, c=1.0)
        assert ucb.c == 1.0

    def test_select_action(self):
        """Test action selection"""
        ucb = UCBAction(n_actions=4, c=1.0)
        state = np.random.randn(4)
        action = ucb.select_action(state)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < ucb.n_actions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
