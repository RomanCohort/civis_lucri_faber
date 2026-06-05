"""Tests for core.limbic — 5 tests covering instantiation and forward."""
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_limbic(input_dim=16, hippocampus_dim=32):
    """Create a LimbicSystem with minimal dimensions."""
    from core.limbic import LimbicSystem
    return LimbicSystem(input_dim=input_dim, hippocampus_dim=hippocampus_dim)


# ---------------------------------------------------------------------------
# Instantiation (2 tests)
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_limbic_creates_without_error(self):
        ls = _make_limbic()
        assert ls is not None

    def test_limbic_has_required_attributes(self):
        ls = _make_limbic()
        assert hasattr(ls, "output_keys")
        assert hasattr(ls, "step")
        assert hasattr(ls, "forward")
        assert hasattr(ls, "amygdala")
        assert hasattr(ls, "thalamus")


# ---------------------------------------------------------------------------
# Forward (3 tests)
# ---------------------------------------------------------------------------

class TestForward:
    def test_forward_returns_dict(self):
        ls = _make_limbic()
        state = torch.randn(1, 16)
        result = ls.forward(state)
        assert isinstance(result, dict)

    def test_forward_contains_emotion_keys(self):
        ls = _make_limbic()
        state = torch.randn(1, 16)
        result = ls.forward(state)
        # Should have emotion-related keys
        has_emotion = any("emotion" in k or "arousal" in k or "valence" in k
                          for k in result.keys())
        assert has_emotion

    def test_step_delegates_to_forward(self):
        ls = _make_limbic()
        state = torch.randn(1, 16)
        result = ls.step(state)
        assert isinstance(result, dict)
