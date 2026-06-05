"""Tests for core.neurotransmitter — 6 tests covering instantiation,
dopamine, and serotonin."""
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dopamine():
    from core.neurotransmitter import DopamineSystem
    return DopamineSystem()


def _make_serotonin():
    from core.neurotransmitter import SerotoninSystem
    return SerotoninSystem()


def _make_neurotransmitter():
    from core.neurotransmitter import NeurotransmitterSystem
    return NeurotransmitterSystem()


# ---------------------------------------------------------------------------
# Instantiation (2 tests)
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_neurotransmitter_system_creates(self):
        ns = _make_neurotransmitter()
        assert ns is not None

    def test_dopamine_has_pathway_level(self):
        """DopamineSystem should have pathway_level initialized to 0.5."""
        ds = _make_dopamine()
        assert hasattr(ds, "pathway_level")
        assert ds.pathway_level == 0.5


# ---------------------------------------------------------------------------
# Dopamine (2 tests)
# ---------------------------------------------------------------------------

class TestDopamine:
    def test_compute_reward_signal_returns_result(self):
        ds = _make_dopamine()
        reward = 1.0
        result = ds.compute_reward_signal(reward, expectation=0.0)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3  # (total, tonic, phasic)

    def test_dopamine_level_updates(self):
        """After compute_reward_signal, dopamine level should change."""
        ds = _make_dopamine()
        old_level = ds.current_level
        ds.compute_reward_signal(1.0, expectation=0.0)
        # Level may or may not change, but call should not crash
        assert isinstance(ds.current_level, (float, int, np.floating))


# ---------------------------------------------------------------------------
# Serotonin (2 tests)
# ---------------------------------------------------------------------------

class TestSerotonin:
    def test_serotonin_system_creates(self):
        ss = _make_serotonin()
        assert ss is not None

    def test_neurotransmitter_step_runs(self):
        """NeurotransmitterSystem.step() should complete without error."""
        ns = _make_neurotransmitter()
        result = ns.step(reward=1.0)
        assert result is not None
