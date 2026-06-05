"""Tests for core.prefrontal_cortex — 5 tests covering instantiation and forward."""
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pfc(input_dim=16, hidden_dim=32, num_actions=4):
    """Create a PrefrontalCortex with minimal dimensions."""
    from core.prefrontal_cortex import PrefrontalCortex
    return PrefrontalCortex(input_dim=input_dim, hidden_dim=hidden_dim,
                             num_actions=num_actions)


# ---------------------------------------------------------------------------
# Instantiation (2 tests)
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_pfc_creates_without_error(self):
        pfc = _make_pfc()
        assert pfc is not None

    def test_pfc_has_required_attributes(self):
        pfc = _make_pfc()
        assert hasattr(pfc, "output_keys")
        assert hasattr(pfc, "step")
        assert hasattr(pfc, "forward")


# ---------------------------------------------------------------------------
# Forward (3 tests)
# ---------------------------------------------------------------------------

class TestForward:
    def test_forward_returns_dict(self):
        pfc = _make_pfc()
        state = torch.randn(1, 16)
        result = pfc.forward(state)
        assert isinstance(result, dict)

    def test_forward_contains_action(self):
        pfc = _make_pfc()
        state = torch.randn(1, 16)
        result = pfc.forward(state)
        assert "action" in result
        assert "action_logits" in result

    def test_action_in_valid_range(self):
        pfc = _make_pfc(num_actions=4)
        state = torch.randn(1, 16)
        result = pfc.forward(state)
        action = result["action"].item()
        assert 0 <= action < 4
