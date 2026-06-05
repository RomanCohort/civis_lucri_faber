"""Tests for core.state_proxy — 8 tests covering write protection and views."""
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proxy(strict: bool = False):
    from core.state_proxy import StateProxy
    state = {"dopamine": 0.5, "arousal": 0.3}
    proxy = StateProxy(state, name="test", strict=strict)
    proxy.register_writer("bg", ["bg_action", "bg_dopamine"])
    return proxy, state


# ---------------------------------------------------------------------------
# Write Protection (4 tests)
# ---------------------------------------------------------------------------

class TestWriteProtection:
    def test_registered_write_succeeds(self):
        proxy, state = _make_proxy()
        proxy.write("bg", "bg_action", 2)
        assert state["bg_action"] == 2

    def test_unregistered_write_non_strict(self):
        proxy, state = _make_proxy(strict=False)
        proxy.write("bg", "unknown_key", 42)
        assert state["unknown_key"] == 42

    def test_unregistered_write_strict_raises(self):
        proxy, state = _make_proxy(strict=True)
        with pytest.raises(KeyError):
            proxy.write("bg", "unknown_key", 42)

    def test_write_overwrites_existing(self):
        proxy, state = _make_proxy()
        proxy.write("bg", "bg_dopamine", 0.8)
        assert state["bg_dopamine"] == 0.8
        proxy.write("bg", "bg_dopamine", 0.9)
        assert state["bg_dopamine"] == 0.9


# ---------------------------------------------------------------------------
# Views (4 tests)
# ---------------------------------------------------------------------------

class TestViews:
    def test_read_existing_key(self):
        proxy, _ = _make_proxy()
        assert proxy.read("dopamine") == 0.5

    def test_read_missing_key_returns_default(self):
        proxy, _ = _make_proxy()
        assert proxy.read("nonexistent", default=-1) == -1

    def test_view_all_keys(self):
        proxy, _ = _make_proxy()
        snapshot = proxy.view()
        assert "dopamine" in snapshot
        assert "arousal" in snapshot

    def test_view_selected_keys(self):
        proxy, _ = _make_proxy()
        snapshot = proxy.view(keys=["dopamine"])
        assert "dopamine" in snapshot
        assert "arousal" not in snapshot
