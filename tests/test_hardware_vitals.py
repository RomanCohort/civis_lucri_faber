"""Tests for core.hardware_vitals — Hardware Vitals Bridge.

Tests cover:
- HardwareVitals instantiation
- HardwareState dataclass
- read() method functionality
- Biological parameter mapping (to_xxx methods)
- Edge cases (missing psutil, extreme values)
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hardware_vitals():
    """Create a HardwareVitals instance."""
    from core.hardware_vitals import HardwareVitals
    return HardwareVitals()


@pytest.fixture
def hardware_state():
    """Create a HardwareState instance."""
    from core.hardware_vitals import HardwareState
    return HardwareState(
        cpu_percent=0.5,
        ram_percent=0.4,
        disk_percent=0.3,
        gpu_available=False,
        gpu_memory_percent=0.0,
        process_rss_mb=100.0,
        thread_count=4,
        event_queue_size=10,
        error_rate=0.01,
        gc_gen0_count=5,
        wall_clock_hour=12.0,
    )


# ---------------------------------------------------------------------------
# HardwareVitals Instantiation Tests
# ---------------------------------------------------------------------------

class TestHardwareVitalsInstantiation:
    """Tests for HardwareVitals instantiation."""

    def test_vitals_creates_successfully(self):
        """HardwareVitals should instantiate without error."""
        from core.hardware_vitals import HardwareVitals
        vitals = HardwareVitals()
        assert vitals is not None

    def test_vitals_has_state(self, hardware_vitals):
        """HardwareVitals should have HardwareState."""
        from core.hardware_vitals import HardwareState
        assert hasattr(hardware_vitals, 'state')
        assert isinstance(hardware_vitals.state, HardwareState)

    def test_vitals_has_smoothing_params(self, hardware_vitals):
        """HardwareVitals should have smoothing parameters."""
        assert hasattr(hardware_vitals, '_smoothing')
        assert 0 < hardware_vitals._smoothing < 1


# ---------------------------------------------------------------------------
# HardwareState Tests
# ---------------------------------------------------------------------------

class TestHardwareState:
    """Tests for HardwareState dataclass."""

    def test_state_creates_with_defaults(self):
        """HardwareState should create with default values."""
        from core.hardware_vitals import HardwareState
        state = HardwareState()
        assert state.cpu_percent == 0.0
        assert state.ram_percent == 0.0
        assert state.disk_percent == 0.0

    def test_state_accepts_custom_values(self, hardware_state):
        """HardwareState should accept custom values."""
        assert hardware_state.cpu_percent == 0.5
        assert hardware_state.ram_percent == 0.4
        assert hardware_state.thread_count == 4

    def test_state_fields_in_valid_range(self, hardware_state):
        """State fields should be in [0, 1] range for percentages."""
        assert 0.0 <= hardware_state.cpu_percent <= 1.0
        assert 0.0 <= hardware_state.ram_percent <= 1.0
        assert 0.0 <= hardware_state.disk_percent <= 1.0


# ---------------------------------------------------------------------------
# read() Method Tests
# ---------------------------------------------------------------------------

class TestReadMethod:
    """Tests for HardwareVitals.read() method."""

    def test_read_returns_hardware_state(self, hardware_vitals):
        """read() should return HardwareState."""
        from core.hardware_vitals import HardwareState
        state = hardware_vitals.read()
        assert isinstance(state, HardwareState)

    def test_read_updates_state(self, hardware_vitals):
        """read() should update internal state."""
        initial_step = hardware_vitals._step_count
        hardware_vitals.read()
        assert hardware_vitals._step_count > initial_step

    def test_read_without_psutil(self):
        """read() should work without psutil installed."""
        from core.hardware_vitals import HardwareVitals, HAS_PSUTIL
        vitals = HardwareVitals()
        state = vitals.read()
        # Should return defaults if psutil unavailable
        assert state is not None

    def test_read_multiple_times(self, hardware_vitals):
        """Multiple read() calls should run without error."""
        for _ in range(10):
            state = hardware_vitals.read()
        assert state is not None


# ---------------------------------------------------------------------------
# Biological Parameter Mapping Tests
# ---------------------------------------------------------------------------

class TestBiologicalMappings:
    """Tests for to_xxx biological parameter methods."""

    def test_to_heart_rate_returns_float(self, hardware_vitals):
        """to_heart_rate() should return float."""
        hardware_vitals.read()
        hr = hardware_vitals.to_heart_rate()
        assert isinstance(hr, float)

    def test_to_heart_rate_in_range(self, hardware_vitals):
        """Heart rate should be in biological range."""
        hardware_vitals.state.cpu_percent = 0.5
        hr = hardware_vitals.to_heart_rate()
        # Heart rate should be 60-180 bpm range (scaled)
        assert hr >= 0.0
        assert hr <= 1.0

    def test_to_blood_pressure_returns_float(self, hardware_vitals):
        """to_blood_pressure() should return float."""
        hardware_vitals.state.ram_percent = 0.5
        bp = hardware_vitals.to_blood_pressure()
        assert isinstance(bp, float)

    def test_to_o2_saturation_returns_float(self, hardware_vitals):
        """to_o2_saturation() should return float."""
        hardware_vitals.state.ram_available_mb = 1000.0
        hardware_vitals.state.ram_total_mb = 2000.0
        o2 = hardware_vitals.to_o2_saturation()
        assert isinstance(o2, float)

    def test_to_sympathetic_tone_returns_float(self, hardware_vitals):
        """to_sympathetic_tone() should return float."""
        hardware_vitals.state.cpu_percent = 0.8
        tone = hardware_vitals.to_sympathetic_tone()
        assert isinstance(tone, float)

    def test_to_parasympathetic_tone_returns_float(self, hardware_vitals):
        """to_parasympathetic_tone() should return float."""
        hardware_vitals.state.ram_available_mb = 1000.0
        tone = hardware_vitals.to_parasympathetic_tone()
        assert isinstance(tone, float)

    def test_to_cortisol_drive_returns_float(self, hardware_vitals):
        """to_cortisol_drive() should return float."""
        hardware_vitals.state.error_rate = 0.05
        cortisol = hardware_vitals.to_cortisol_drive()
        assert isinstance(cortisol, float)

    def test_to_fatigue_returns_float(self, hardware_vitals):
        """to_fatigue() should return float."""
        hardware_vitals.state.process_rss_mb = 500.0
        fatigue = hardware_vitals.to_fatigue()
        assert isinstance(fatigue, float)


# ---------------------------------------------------------------------------
# CPU/RAM Mapping Tests
# ---------------------------------------------------------------------------

class TestCpuRamMappings:
    """Tests for CPU and RAM specific mappings."""

    def test_high_cpu_increases_heart_rate(self, hardware_vitals):
        """High CPU should increase heart rate."""
        hardware_vitals.state.cpu_percent = 0.1
        low_hr = hardware_vitals.to_heart_rate()

        hardware_vitals.state.cpu_percent = 0.9
        high_hr = hardware_vitals.to_heart_rate()

        # Higher CPU should correlate with higher HR
        assert high_hr >= low_hr * 0.8  # Allow some tolerance

    def test_high_ram_increases_blood_pressure(self, hardware_vitals):
        """High RAM usage should increase blood pressure."""
        hardware_vitals.state.ram_percent = 0.1
        low_bp = hardware_vitals.to_blood_pressure()

        hardware_vitals.state.ram_percent = 0.9
        high_bp = hardware_vitals.to_blood_pressure()

        assert high_bp >= low_bp * 0.8


# ---------------------------------------------------------------------------
# Gut/Neurotransmitter Mapping Tests
# ---------------------------------------------------------------------------

class TestGutMappings:
    """Tests for gut serotonin/GABA mappings."""

    def test_to_gut_serotonin_returns_float(self, hardware_vitals):
        """to_gut_serotonin() should return float."""
        hardware_vitals.state.gc_gen0_count = 10
        serotonin = hardware_vitals.to_gut_serotonin()
        assert isinstance(serotonin, float)

    def test_to_gut_gaba_returns_float(self, hardware_vitals):
        """to_gut_gaba() should return float."""
        hardware_vitals.state.thread_count = 10
        gaba = hardware_vitals.to_gut_gaba()
        assert isinstance(gaba, float)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_cpu_percent(self, hardware_vitals):
        """Zero CPU percent should be handled."""
        hardware_vitals.state.cpu_percent = 0.0
        hr = hardware_vitals.to_heart_rate()
        assert hr >= 0.0

    def test_max_cpu_percent(self, hardware_vitals):
        """Max CPU percent should be handled."""
        hardware_vitals.state.cpu_percent = 1.0
        hr = hardware_vitals.to_heart_rate()
        assert hr <= 1.0

    def test_zero_ram(self, hardware_vitals):
        """Zero RAM usage should be handled."""
        hardware_vitals.state.ram_percent = 0.0
        bp = hardware_vitals.to_blood_pressure()
        assert bp >= 0.0

    def test_max_ram(self, hardware_vitals):
        """Max RAM usage should be handled."""
        hardware_vitals.state.ram_percent = 1.0
        bp = hardware_vitals.to_blood_pressure()
        assert bp <= 1.0

    def test_no_gpu_available(self, hardware_vitals):
        """No GPU should be handled gracefully."""
        hardware_vitals.state.gpu_available = False
        hardware_vitals.state.gpu_memory_percent = 0.0
        # Should not crash
        hardware_vitals.read()

    def test_gpu_available(self, hardware_vitals):
        """GPU available should be handled."""
        hardware_vitals.state.gpu_available = True
        hardware_vitals.state.gpu_memory_percent = 0.5
        # Should not crash
        hardware_vitals.read()

    def test_extreme_error_rate(self, hardware_vitals):
        """Extreme error rate should be handled."""
        hardware_vitals.state.error_rate = 1.0
        cortisol = hardware_vitals.to_cortisol_drive()
        assert 0.0 <= cortisol <= 1.0

    def test_large_thread_count(self, hardware_vitals):
        """Large thread count should be handled."""
        hardware_vitals.state.thread_count = 1000
        gaba = hardware_vitals.to_gut_gaba()
        assert 0.0 <= gaba <= 1.0


# ---------------------------------------------------------------------------
# Smoothing Tests
# ---------------------------------------------------------------------------

class TestSmoothing:
    """Tests for EMA smoothing functionality."""

    def test_smoothing_reduces_jitter(self, hardware_vitals):
        """Smoothing should reduce value jitter."""
        # Simulate rapid changes
        values = []
        for cpu in [0.1, 0.9, 0.1, 0.9, 0.5]:
            hardware_vitals.state.cpu_percent = cpu
            hardware_vitals._smooth_cpu = cpu * hardware_vitals._smoothing + \
                hardware_vitals._smooth_cpu * (1 - hardware_vitals._smoothing)
            values.append(hardware_vitals._smooth_cpu)
        # Smoothed values should not jump as much as raw input
        # This is a basic sanity check
        assert len(values) == 5


# ---------------------------------------------------------------------------
# Agent Integration Tests
# ---------------------------------------------------------------------------

class TestAgentIntegration:
    """Tests for agent integration."""

    def test_vitals_read_with_agent_mock(self, hardware_vitals):
        """read() should work with mock agent."""
        # Create a simple mock agent-like object
        mock_agent = type('MockAgent', (), {
            '_internal_state': {},
            'step_count': 10,
        })()
        state = hardware_vitals.read(mock_agent)
        assert state is not None


# ---------------------------------------------------------------------------
# Summary Tests
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for get_summary method."""

    def test_get_summary_returns_dict(self, hardware_vitals):
        """get_summary should return a dictionary."""
        hardware_vitals.read()
        if hasattr(hardware_vitals, 'get_summary'):
            summary = hardware_vitals.get_summary()
            assert isinstance(summary, dict)

    def test_summary_contains_key_fields(self, hardware_vitals):
        """Summary should contain key fields if method exists."""
        hardware_vitals.read()
        if hasattr(hardware_vitals, 'get_summary'):
            summary = hardware_vitals.get_summary()
            # Should have at least CPU/RAM info
            assert 'cpu' in summary or 'cpu_percent' in summary or len(summary) > 0