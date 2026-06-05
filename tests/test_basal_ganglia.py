"""Tests for core.basal_ganglia — 9 tests covering instantiation,
eligibility trace, trace-driven update, and action selection."""
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bg(state_dim=16, n_actions=4):
    """Create a BasalGangliaSystem with minimal dimensions."""
    from core.basal_ganglia import BasalGangliaSystem
    return BasalGangliaSystem(state_dim=state_dim, n_actions=n_actions)


def _make_inner_bg(state_dim=16, n_actions=4):
    """Create the inner BasalGanglia (not the System wrapper)."""
    from core.basal_ganglia import BasalGanglia
    return BasalGanglia(state_dim=state_dim, n_actions=n_actions)


# ---------------------------------------------------------------------------
# Instantiation (2 tests)
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_bg_system_creates_without_error(self):
        bg = _make_bg()
        assert bg is not None

    def test_bg_system_has_required_attributes(self):
        bg = _make_bg()
        assert hasattr(bg, "bg")
        assert hasattr(bg.bg, "output_keys")
        assert hasattr(bg.bg, "step")


# ---------------------------------------------------------------------------
# Eligibility Trace (3 tests)
# ---------------------------------------------------------------------------

class TestEligibilityTrace:
    def test_trace_replacing_not_accumulating(self):
        """After update(), selected action trace should be 1.0 (replacing trace)."""
        bg = _make_inner_bg()
        bg.training = False  # deterministic for testing
        state = torch.randn(1, 16)
        next_state = torch.randn(1, 16)
        # Call update with action=0
        bg.update(state, action=0, reward=1.0, next_state=next_state)
        assert bg.eligibility_trace.get(0) == 1.0

    def test_trace_decay_for_non_selected(self):
        """Non-selected traces should decay by gamma * trace_decay."""
        bg = _make_inner_bg()
        bg.training = False
        state = torch.randn(1, 16)
        next_state = torch.randn(1, 16)
        # Pre-set traces
        bg.eligibility_trace[1] = 0.5
        bg.eligibility_trace[2] = 0.8
        bg.update(state, action=0, reward=1.0, next_state=next_state)
        # Non-selected traces should have decayed
        expected_decay = bg.gamma * bg.trace_decay  # 0.99 * 0.9 = 0.891
        assert bg.eligibility_trace[1] < 0.5
        assert bg.eligibility_trace[2] < 0.8

    def test_trace_starts_at_replacing_value(self):
        """When a new action is selected, its trace should be set to 1.0."""
        bg = _make_inner_bg()
        bg.training = False
        bg.eligibility_trace.clear()
        state = torch.randn(1, 16)
        next_state = torch.randn(1, 16)
        bg.update(state, action=3, reward=0.5, next_state=next_state)
        assert bg.eligibility_trace.get(3) == 1.0


# ---------------------------------------------------------------------------
# Trace-driven Update (2 tests)
# ---------------------------------------------------------------------------

class TestTraceDrivenUpdate:
    def test_td_error_is_computed(self):
        """Update should return td_error in the result dict."""
        bg = _make_inner_bg()
        state = torch.randn(1, 16)
        next_state = torch.randn(1, 16)
        result = bg.update(state, action=0, reward=1.0, next_state=next_state)
        assert "td_error" in result

    def test_update_does_not_crash_with_consecutive_calls(self):
        """Multiple update calls should not crash."""
        bg = _make_inner_bg()
        for _ in range(10):
            state = torch.randn(1, 16)
            next_state = torch.randn(1, 16)
            bg.update(state, action=0, reward=1.0, next_state=next_state)
        assert len(bg.eligibility_trace) > 0


# ---------------------------------------------------------------------------
# Action Selection (2 tests)
# ---------------------------------------------------------------------------

class TestActionSelection:
    def test_action_in_valid_range(self):
        """Selected action should be within [0, n_actions)."""
        bg = _make_bg(n_actions=4, state_dim=16)
        state = torch.randn(1, 16)
        result = bg.forward(state, epsilon=0.1)
        assert 0 <= result["action"] < 4

    def test_forward_returns_expected_keys(self):
        """BasalGangliaSystem.forward() should return standard keys."""
        bg = _make_bg()
        state = torch.randn(1, 16)
        result = bg.forward(state)
        assert "action" in result
        assert "q_values" in result
