"""Information Gain Tests"""
import numpy as np
import pytest
import torch

from information_gain import TrueInformationGainCalculator, VariationalWorldModel


class TestVariationalWorldModel:
    """Test variational world model"""

    def test_init(self):
        """Test initialization"""
        model = VariationalWorldModel(state_dim=64, n_actions=4, hidden_dim=32)
        assert model is not None

    def test_forward_shape(self):
        """Test forward pass shape"""
        model = VariationalWorldModel(state_dim=64, n_actions=4, hidden_dim=16)
        state = torch.randn(1, 64)
        action = torch.randn(1, 4)
        results = model(state, action)
        # Returns tuple: (next_state_mean, next_state_std, kl, log_prob)
        assert isinstance(results, tuple)
        assert results[0].shape[-1] == 64  # next_state_mean


class TestInformationGainCalculator:
    """Test information gain calculator"""

    def test_init(self):
        """Test initialization"""
        calc = TrueInformationGainCalculator(
            state_dim=64,
            action_dim=4,
            latent_dim=16,
            lr=0.001
        )
        assert calc is not None

    def test_compute_reward(self):
        """Test reward computation"""
        calc = TrueInformationGainCalculator(state_dim=64, action_dim=4)
        state = np.random.randn(64)
        action = np.random.randn(4)
        reward = np.random.randn(1).item()
        next_state = np.random.randn(64)

        reward_obj = calc.compute_reward(state, action, reward, next_state, use_intrinsic=True)
        assert hasattr(reward_obj, 'total')
        assert hasattr(reward_obj, 'intrinsic')

    def test_train_step(self):
        """Test training step"""
        calc = TrueInformationGainCalculator(state_dim=64, action_dim=4)
        result = calc.train_step()
        assert isinstance(result, dict)
        assert 'loss' in result

    def test_statistics(self):
        """Test statistics"""
        calc = TrueInformationGainCalculator(state_dim=64, action_dim=4)
        stats = calc.get_statistics()
        assert 'info_gain_avg' in stats or 'buffer_size' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
