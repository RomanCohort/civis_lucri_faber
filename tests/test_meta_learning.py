"""Meta Learning Tests"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

import torch
import torch.nn as nn
from meta_learning import (
    FirstOrderMAML, UncertaintyAwareActiveLearner, CognitiveDissonanceDetector
)


class TestFirstOrderMAML:
    """Test first-order MAML"""

    def test_init(self):
        """Test initialization"""
        maml = FirstOrderMAML(
            input_dim=4,
            output_dim=2,
            hidden_dim=32,
            inner_lr=0.01
        )
        assert maml is not None

    def test_forward(self):
        """Test forward pass"""
        maml = FirstOrderMAML(input_dim=4, output_dim=2, hidden_dim=16)
        state = torch.randn(2, 4)
        output = maml(state)
        assert output.shape[0] == 2


class TestUncertaintyAwareActiveLearner:
    """Test active learner"""

    @staticmethod
    def _make_model():
        """Create a simple model that can be cloned by UncertaintyAwareActiveLearner"""
        class SimpleModel(nn.Module):
            def __init__(self, in_features=4, out_features=1, **kwargs):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.linear = nn.Linear(in_features, out_features)

            def forward(self, x):
                return self.linear(x)

        return SimpleModel(in_features=4, out_features=1)

    def test_init(self):
        """Test initialization"""
        model = self._make_model()
        learner = UncertaintyAwareActiveLearner(model=model, num_ensemble=3)
        assert learner is not None

    def test_estimate_uncertainty(self):
        """Test uncertainty estimation"""
        model = self._make_model()
        learner = UncertaintyAwareActiveLearner(model=model, num_ensemble=3)
        state = torch.randn(1, 4)
        mean, variance = learner.estimate_epistemic_uncertainty(state)
        assert isinstance(mean, float)
        assert isinstance(variance, float)

    def test_compute_acquisition(self):
        """Test acquisition function"""
        model = self._make_model()
        learner = UncertaintyAwareActiveLearner(model=model, num_ensemble=3)
        state = torch.randn(1, 4)
        score = learner.compute_acquisition(state, info_gain=0.5)
        assert isinstance(score, float)


class TestCognitiveDissonanceDetector:
    """Test cognitive dissonance detector"""

    def test_init(self):
        """Test initialization"""
        detector = CognitiveDissonanceDetector()
        assert detector is not None

    def test_detect_contradiction(self):
        """Test contradiction detection"""
        detector = CognitiveDissonanceDetector()
        result = detector.detect_contradiction("This is a test memory")
        assert result is None or hasattr(result, 'inconsistency_score')

    def test_empty_input(self):
        """Test empty input"""
        detector = CognitiveDissonanceDetector()
        result = detector.detect_contradiction("")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])