"""Tests for core.simulation_clock — 9 tests covering tick, circadian, time dict."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clock(step_duration: float = 0.1, start_hour: float = 8.0):
    from core.simulation_clock import SimulationClock
    return SimulationClock(step_duration=step_duration, start_hour=start_hour)


# ---------------------------------------------------------------------------
# Tick (3 tests)
# ---------------------------------------------------------------------------

class TestTick:
    def test_tick_increments_step(self):
        clock = _make_clock()
        result = clock.tick()
        assert result["step"] == 1

    def test_tick_advances_hour(self):
        clock = _make_clock(step_duration=0.5)
        clock.tick()
        assert abs(clock.state.hour - 8.5) < 1e-6

    def test_multiple_ticks(self):
        clock = _make_clock(step_duration=1.0)
        for _ in range(5):
            clock.tick()
        assert clock.state.step == 5
        assert abs(clock.state.hour - 13.0) < 1e-6


# ---------------------------------------------------------------------------
# Circadian (3 tests)
# ---------------------------------------------------------------------------

class TestCircadian:
    def test_circadian_phase_is_float(self):
        clock = _make_clock(start_hour=12.0)
        phase = clock.circadian_phase()
        assert isinstance(phase, float)

    def test_circadian_noon_is_positive(self):
        """At noon (12h), circadian phase should be positive (daytime)."""
        clock = _make_clock(start_hour=12.0)
        assert clock.circadian_phase() > 0

    def test_midnight_is_negative(self):
        """At midnight (0h), circadian phase should be near 0 or negative."""
        clock = _make_clock(start_hour=0.0)
        assert clock.circadian_phase() < 0.01  # sin(0) = 0


# ---------------------------------------------------------------------------
# Time Dict (3 tests)
# ---------------------------------------------------------------------------

class TestTimeDict:
    def test_time_dict_has_required_keys(self):
        clock = _make_clock()
        d = clock.time_dict()
        assert "step" in d
        assert "hour" in d
        assert "day" in d

    def test_time_dict_step_zero_initially(self):
        clock = _make_clock()
        d = clock.time_dict()
        assert d["step"] == 0

    def test_day_wraps_at_24h(self):
        """When hour exceeds 24, day should increment."""
        clock = _make_clock(step_duration=5.0, start_hour=20.0)
        clock.tick()  # hour -> 25 -> 1, day -> 1
        assert clock.state.day == 1
        assert abs(clock.state.hour - 1.0) < 1e-6
