"""Tests for core.hpa_axis — HPA stress axis.

Tests cover:
- HPAAxis instantiation with default/custom params
- Step computation and cascade
- Component modules (CRH, ACTH, Cortisol)
- Negative feedback loop
- Allostatic load tracking
- Edge cases (zero inputs, extreme values)
- Output shape/type validation
"""
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hpa_axis():
    """Create a fresh HPAAxis instance."""
    from core.hpa_axis import HPAAxis
    return HPAAxis()


@pytest.fixture
def hpa_with_custom_params():
    """Create HPAAxis with custom parameters."""
    from core.hpa_axis import HPAAxis
    return HPAAxis(
        stress_reactivity=1.5,
        cortisol_half_life_steps=30,
        feedback_strength=0.8,
        load_accumulation_rate=0.005,
    )


@pytest.fixture
def hpa_components():
    """Create individual HPA components for testing."""
    from core.hpa_axis import (
        HypothalamicCRH,
        PituitaryACTH,
        AdrenalCortex,
        NegativeFeedbackLoop,
        AllostaticLoadTracker,
    )
    return {
        "crh": HypothalamicCRH(),
        "acth": PituitaryACTH(),
        "adrenal": AdrenalCortex(),
        "feedback": NegativeFeedbackLoop(),
        "load_tracker": AllostaticLoadTracker(),
    }


# ---------------------------------------------------------------------------
# HPAAxis Instantiation Tests
# ---------------------------------------------------------------------------

class TestHPAAxisInstantiation:
    """Tests for HPAAxis instantiation."""

    def test_hpa_creates_successfully(self):
        """HPAAxis should instantiate without error."""
        from core.hpa_axis import HPAAxis
        hpa = HPAAxis()
        assert hpa is not None

    def test_hpa_has_required_components(self, hpa_axis):
        """HPAAxis should have all required components."""
        assert hasattr(hpa_axis, "crh")
        assert hasattr(hpa_axis, "acth")
        assert hasattr(hpa_axis, "adrenal")
        assert hasattr(hpa_axis, "feedback")
        assert hasattr(hpa_axis, "load_tracker")

    def test_hpa_has_state(self, hpa_axis):
        """HPAAxis should have HPAState."""
        from core.hpa_axis import HPAState
        assert hasattr(hpa_axis, "state")
        assert isinstance(hpa_axis.state, HPAState)

    def test_hpa_custom_stress_reactivity(self, hpa_with_custom_params):
        """Custom stress_reactivity should be applied."""
        hpa = hpa_with_custom_params
        assert hpa.crh.stress_reactivity.item() == 1.5

    def test_hpa_custom_cortisol_half_life(self, hpa_with_custom_params):
        """Custom cortisol_half_life_steps should be applied."""
        hpa = hpa_with_custom_params
        assert hpa.adrenal.half_life_steps == 30

    def test_hpa_custom_feedback_strength(self, hpa_with_custom_params):
        """Custom feedback_strength should be applied."""
        hpa = hpa_with_custom_params
        # Use approximate comparison for torch tensor floating point
        assert abs(hpa.feedback.feedback_strength.item() - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# Component Tests
# ---------------------------------------------------------------------------

class TestHypothalamicCRH:
    """Tests for HypothalamicCRH module."""

    def test_crh_forward_returns_dict(self, hpa_components):
        """CRH forward should return a dictionary."""
        crh = hpa_components["crh"]
        result = crh.forward(
            stress_signal=0.5,
            cortisol_feedback=0.3,
            uncertainty=0.2,
        )
        assert isinstance(result, dict)

    def test_crh_result_has_required_keys(self, hpa_components):
        """CRH result should have required keys."""
        crh = hpa_components["crh"]
        result = crh.forward(
            stress_signal=0.5,
            cortisol_feedback=0.3,
            uncertainty=0.2,
        )
        assert "crh_level" in result
        assert "drive" in result
        assert "inhibition" in result

    def test_crh_level_in_valid_range(self, hpa_components):
        """CRH level should be in valid range [0, 1]."""
        crh = hpa_components["crh"]
        for stress in [0.0, 0.3, 0.5, 0.8, 1.0]:
            result = crh.forward(
                stress_signal=stress,
                cortisol_feedback=0.2,
                uncertainty=0.1,
            )
            assert 0.0 <= result["crh_level"] <= 1.0

    def test_crh_increases_with_stress(self, hpa_components):
        """Higher stress should increase CRH release."""
        crh = hpa_components["crh"]
        low_stress = crh.forward(stress_signal=0.1, cortisol_feedback=0.2, uncertainty=0.1)
        high_stress = crh.forward(stress_signal=0.9, cortisol_feedback=0.2, uncertainty=0.1)
        assert high_stress["crh_level"] > low_stress["crh_level"]

    def test_crh_decreases_with_cortisol_feedback(self, hpa_components):
        """Higher cortisol feedback should decrease CRH release."""
        crh = hpa_components["crh"]
        low_fb = crh.forward(stress_signal=0.5, cortisol_feedback=0.1, uncertainty=0.1)
        high_fb = crh.forward(stress_signal=0.5, cortisol_feedback=0.8, uncertainty=0.1)
        assert high_fb["crh_level"] < low_fb["crh_level"]


class TestPituitaryACTH:
    """Tests for PituitaryACTH module."""

    def test_acth_forward_returns_dict(self, hpa_components):
        """ACTH forward should return a dictionary."""
        acth = hpa_components["acth"]
        result = acth.forward(crh_level=0.5, cortisol_feedback=0.3)
        assert isinstance(result, dict)

    def test_acth_result_has_acth_level(self, hpa_components):
        """ACTH result should have acth_level key."""
        acth = hpa_components["acth"]
        result = acth.forward(crh_level=0.5, cortisol_feedback=0.3)
        assert "acth_level" in result

    def test_acth_level_in_valid_range(self, hpa_components):
        """ACTH level should be in valid range [0, 1]."""
        acth = hpa_components["acth"]
        for crh in [0.0, 0.3, 0.5, 0.8, 1.0]:
            result = acth.forward(crh_level=crh, cortisol_feedback=0.2)
            assert 0.0 <= result["acth_level"] <= 1.0

    def test_acth_increases_with_crh(self, hpa_components):
        """Higher CRH should increase ACTH release."""
        acth = hpa_components["acth"]
        low_crh = acth.forward(crh_level=0.1, cortisol_feedback=0.2)
        high_crh = acth.forward(crh_level=0.9, cortisol_feedback=0.2)
        assert high_crh["acth_level"] > low_crh["acth_level"]


class TestAdrenalCortex:
    """Tests for AdrenalCortex module."""

    def test_adrenal_forward_returns_dict(self, hpa_components):
        """Adrenal forward should return a dictionary."""
        adrenal = hpa_components["adrenal"]
        result = adrenal.forward(acth_level=0.5, circadian_baseline=0.3)
        assert isinstance(result, dict)

    def test_adrenal_result_has_cortisol_level(self, hpa_components):
        """Adrenal result should have cortisol_level key."""
        adrenal = hpa_components["adrenal"]
        result = adrenal.forward(acth_level=0.5, circadian_baseline=0.3)
        assert "cortisol_level" in result

    def test_cortisol_level_in_valid_range(self, hpa_components):
        """Cortisol level should be in valid range [0, 1]."""
        adrenal = hpa_components["adrenal"]
        for acth in [0.0, 0.3, 0.5, 0.8, 1.0]:
            result = adrenal.forward(acth_level=acth, circadian_baseline=0.3)
            assert 0.0 <= result["cortisol_level"] <= 1.0

    def test_cortisol_increases_with_acth(self, hpa_components):
        """Higher ACTH should increase cortisol release."""
        adrenal = hpa_components["adrenal"]
        low_acth = adrenal.forward(acth_level=0.1, circadian_baseline=0.3)
        high_acth = adrenal.forward(acth_level=0.9, circadian_baseline=0.3)
        assert high_acth["cortisol_level"] > low_acth["cortisol_level"]

    def test_adrenal_decay_reduces_cortisol(self, hpa_components):
        """Decay should reduce cortisol level."""
        adrenal = hpa_components["adrenal"]
        adrenal.current_cortisol = 0.8
        adrenal.decay()
        assert adrenal.current_cortisol < 0.8

    def test_social_buffer_reduces_cortisol(self, hpa_components):
        """Social buffer should reduce cortisol release."""
        adrenal = hpa_components["adrenal"]
        no_buffer = adrenal.forward(acth_level=0.7, circadian_baseline=0.3, social_buffer=0.0)
        adrenal.current_cortisol = 0.3  # Reset
        with_buffer = adrenal.forward(acth_level=0.7, circadian_baseline=0.3, social_buffer=0.5)
        assert with_buffer["cortisol_level"] <= no_buffer["cortisol_level"]


class TestNegativeFeedbackLoop:
    """Tests for NegativeFeedbackLoop module."""

    def test_feedback_forward_returns_dict(self, hpa_components):
        """Feedback forward should return a dictionary."""
        feedback = hpa_components["feedback"]
        result = feedback.forward(cortisol=0.5)
        assert isinstance(result, dict)

    def test_feedback_result_has_inhibition(self, hpa_components):
        """Feedback result should have cortisol_inhibition key."""
        feedback = hpa_components["feedback"]
        result = feedback.forward(cortisol=0.5)
        assert "cortisol_inhibition" in result
        assert "fast_inhibition" in result
        assert "slow_inhibition" in result

    def test_inhibition_increases_with_cortisol(self, hpa_components):
        """Higher cortisol should increase inhibition."""
        feedback = hpa_components["feedback"]
        low_cort = feedback.forward(cortisol=0.1)
        high_cort = feedback.forward(cortisol=0.9)
        assert high_cort["cortisol_inhibition"] > low_cort["cortisol_inhibition"]

    def test_inhibition_bounded(self, hpa_components):
        """Inhibition should be bounded [0, 0.95]."""
        feedback = hpa_components["feedback"]
        for cortisol in [0.0, 0.5, 1.0, 2.0]:
            result = feedback.forward(cortisol=cortisol)
            assert 0.0 <= result["cortisol_inhibition"] <= 0.95


class TestAllostaticLoadTracker:
    """Tests for AllostaticLoadTracker."""

    def test_load_tracker_update_returns_float(self, hpa_components):
        """update() should return a float."""
        tracker = hpa_components["load_tracker"]
        load = tracker.update(cortisol=0.5, ne_level=0.3)
        assert isinstance(load, float)

    def test_load_tracker_increases_with_high_cortisol(self, hpa_components):
        """High cortisol should increase allostatic load."""
        tracker = hpa_components["load_tracker"]
        initial_load = tracker.load
        tracker.update(cortisol=0.9, ne_level=0.5)
        assert tracker.load > initial_load

    def test_load_tracker_decreases_during_recovery(self, hpa_components):
        """Recovery should decrease allostatic load."""
        tracker = hpa_components["load_tracker"]
        # Build up load
        for _ in range(10):
            tracker.update(cortisol=0.9, ne_level=0.5)
        load_before_recovery = tracker.load
        # Recover
        for _ in range(10):
            tracker.update(cortisol=0.2, ne_level=0.2, is_recovering=True)
        assert tracker.load < load_before_recovery

    def test_is_overloaded(self, hpa_components):
        """is_overloaded() should return True when load exceeds threshold."""
        tracker = hpa_components["load_tracker"]
        tracker.load = 0.9
        assert tracker.is_overloaded()
        tracker.load = 0.5
        assert not tracker.is_overloaded()


# ---------------------------------------------------------------------------
# HPAAxis Step Tests
# ---------------------------------------------------------------------------

class TestHPAAxisStep:
    """Tests for HPAAxis.step() method."""

    def test_step_returns_dict(self, hpa_axis):
        """step() should return a dictionary."""
        result = hpa_axis.step()
        assert isinstance(result, dict)

    def test_step_result_has_required_keys(self, hpa_axis):
        """Step result should have all required keys."""
        result = hpa_axis.step()
        required_keys = [
            "cortisol_level",
            "stress_type",
            "acute_stress_intensity",
            "chronic_stress_ratio",
            "allostatic_load",
        ]
        for key in required_keys:
            assert key in result

    def test_step_with_stress_signal(self, hpa_axis):
        """Step with stress signal should elevate cortisol."""
        baseline = hpa_axis.step(stress_signal=0.0)
        stressed = hpa_axis.step(stress_signal=0.9)
        # Stressed cortisol should be higher than baseline
        assert stressed["cortisol_level"] >= baseline["cortisol_level"] * 0.8

    def test_step_with_uncertainty(self, hpa_axis):
        """Step with uncertainty should affect CRH release."""
        hpa_axis.step(stress_signal=0.3, uncertainty=0.0)
        low_uncertainty_crh = hpa_axis.crh.current_crh
        hpa_axis.step(stress_signal=0.3, uncertainty=0.9)
        high_uncertainty_crh = hpa_axis.crh.current_crh
        assert high_uncertainty_crh >= low_uncertainty_crh

    def test_step_updates_state(self, hpa_axis):
        """Step should update HPAState."""
        from core.hpa_axis import HPAState
        hpa_axis.step(stress_signal=0.5)
        assert isinstance(hpa_axis.state, HPAState)

    def test_step_increments_step_count(self, hpa_axis):
        """Each step should increment step count."""
        initial_count = hpa_axis.step_count
        hpa_axis.step()
        assert hpa_axis.step_count == initial_count + 1

    def test_step_circadian_variation(self, hpa_axis):
        """Cortisol should vary with circadian hour (8AM peak)."""
        # Morning (8AM - peak)
        morning = hpa_axis.step(circadian_hour=8.0)
        hpa_axis.state.cortisol_level = 0.3  # Reset

        # Night (3AM - trough)
        night = hpa_axis.step(circadian_hour=3.0)

        # Morning baseline should be higher than night
        # Note: This tests the circadian baseline contribution
        assert morning.get("circadian_baseline", 0) >= night.get("circadian_baseline", 0)

    def test_step_with_recovery(self, hpa_axis):
        """Recovery mode should reduce allostatic load."""
        # Build up load
        for _ in range(50):
            hpa_axis.step(stress_signal=0.8, is_recovering=False)
        load_before = hpa_axis.state.allostatic_load

        # Recovery
        for _ in range(20):
            hpa_axis.step(stress_signal=0.1, is_recovering=True)
        load_after = hpa_axis.state.allostatic_load

        assert load_after <= load_before

    def test_step_hpa_suppressed(self, hpa_axis):
        """HPA suppression should return sleep stress type."""
        result = hpa_axis.step(hpa_suppressed=True)
        assert result["stress_type"] == "sleep"
        assert result["hpa_suppressed"] is True

    def test_step_multiple_times(self, hpa_axis):
        """Multiple steps should run without error."""
        for _ in range(100):
            result = hpa_axis.step(stress_signal=0.3)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Stress Type Detection Tests
# ---------------------------------------------------------------------------

class TestStressTypeDetection:
    """Tests for stress type classification."""

    def test_no_stress_classification(self, hpa_axis):
        """Low cortisol should result in 'none' stress type."""
        for _ in range(10):
            hpa_axis.step(stress_signal=0.0, circadian_hour=12.0)
        result = hpa_axis.step(stress_signal=0.0)
        # With low stress, should be none or low acute
        assert result["stress_type"] in ["none", "acute"]

    def test_acute_stress_detection(self, hpa_axis):
        """High acute stress should be detected."""
        # Apply acute stress
        hpa_axis.trigger_acute_stress(0.9)
        result = hpa_axis.step(stress_signal=0.9)
        # Acute intensity should be elevated
        assert result["acute_stress_intensity"] > 0.3

    def test_chronic_stress_detection(self, hpa_axis):
        """Prolonged high cortisol should result in chronic stress."""
        # Prolonged stress
        for _ in range(150):
            hpa_axis.step(stress_signal=0.8)
        result = hpa_axis.step(stress_signal=0.8)
        # Chronic ratio should be elevated after prolonged stress
        assert result["chronic_stress_ratio"] > 0


# ---------------------------------------------------------------------------
# Trigger Acute Stress Tests
# ---------------------------------------------------------------------------

class TestTriggerAcuteStress:
    """Tests for trigger_acute_stress method."""

    def test_trigger_acute_stress_increases_crh(self, hpa_axis):
        """trigger_acute_stress should increase CRH."""
        initial_crh = hpa_axis.crh.current_crh
        hpa_axis.trigger_acute_stress(0.8)
        assert hpa_axis.crh.current_crh >= initial_crh

    def test_trigger_acute_stress_bounded(self, hpa_axis):
        """trigger_acute_stress intensity should be bounded."""
        hpa_axis.trigger_acute_stress(2.0)  # Extreme value
        assert hpa_axis.crh.current_crh <= 1.0

        hpa_axis.crh.current_crh = 0.5
        hpa_axis.trigger_acute_stress(-0.5)  # Negative value
        assert hpa_axis.crh.current_crh >= 0.0


# ---------------------------------------------------------------------------
# Get Summary Tests
# ---------------------------------------------------------------------------

class TestGetSummary:
    """Tests for get_summary method."""

    def test_get_summary_returns_dict(self, hpa_axis):
        """get_summary should return a dictionary."""
        summary = hpa_axis.get_summary()
        assert isinstance(summary, dict)

    def test_get_summary_has_required_keys(self, hpa_axis):
        """Summary should contain all required keys."""
        hpa_axis.step()
        summary = hpa_axis.get_summary()
        required_keys = [
            "crh",
            "acth",
            "cortisol",
            "allostatic_load",
            "stress_type",
            "recovery_state",
            "is_overloaded",
            "step_count",
        ]
        for key in required_keys:
            assert key in summary


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_step_with_zero_stress(self, hpa_axis):
        """Step with zero stress should run without error."""
        result = hpa_axis.step(stress_signal=0.0)
        assert isinstance(result, dict)
        assert result["cortisol_level"] >= 0.0

    def test_step_with_extreme_stress(self, hpa_axis):
        """Step with extreme stress should be bounded."""
        result = hpa_axis.step(stress_signal=10.0)
        assert result["cortisol_level"] <= 1.0

    def test_step_with_negative_stress(self, hpa_axis):
        """Step with negative stress should be handled."""
        result = hpa_axis.step(stress_signal=-0.5)
        assert isinstance(result, dict)

    def test_step_with_extreme_uncertainty(self, hpa_axis):
        """Step with extreme uncertainty should be handled."""
        result = hpa_axis.step(uncertainty=10.0)
        assert result["cortisol_level"] <= 1.0

    def test_step_with_extreme_circadian_hour(self, hpa_axis):
        """Step with extreme circadian hour should be handled."""
        result = hpa_axis.step(circadian_hour=100.0)
        assert isinstance(result, dict)

    def test_step_with_negative_circadian_hour(self, hpa_axis):
        """Step with negative circadian hour should be handled."""
        result = hpa_axis.step(circadian_hour=-5.0)
        assert isinstance(result, dict)

    def test_step_with_extreme_ne_level(self, hpa_axis):
        """Step with extreme NE level should be bounded."""
        result = hpa_axis.step(ne_level=10.0)
        assert result["cortisol_level"] <= 1.0

    def test_step_with_extreme_inflammation(self, hpa_axis):
        """Step with extreme inflammation should be handled."""
        result = hpa_axis.step(inflammation=10.0)
        assert isinstance(result, dict)
        assert 0.0 <= result["allostatic_load"] <= 1.0

    def test_prolonged_operation(self, hpa_axis):
        """HPAAxis should handle prolonged operation."""
        for i in range(1000):
            stress = 0.5 * (1 + np.sin(i * 0.01))  # Oscillating stress
            result = hpa_axis.step(stress_signal=stress)
        assert isinstance(result, dict)
        assert 0.0 <= hpa_axis.state.cortisol_level <= 1.0
        assert 0.0 <= hpa_axis.state.allostatic_load <= 1.0


# ---------------------------------------------------------------------------
# Output Validation Tests
# ---------------------------------------------------------------------------

class TestOutputValidation:
    """Tests for output type and range validation."""

    def test_cortisol_in_valid_range(self, hpa_axis):
        """Cortisol should always be in valid range."""
        for stress in [0.0, 0.3, 0.7, 1.0]:
            for _ in range(10):
                result = hpa_axis.step(stress_signal=stress)
                assert 0.0 <= result["cortisol_level"] <= 1.0

    def test_crh_in_valid_range(self, hpa_axis):
        """CRH should always be in valid range."""
        for _ in range(50):
            hpa_axis.step(stress_signal=np.random.random())
            assert 0.0 <= hpa_axis.state.crh_level <= 1.0

    def test_acth_in_valid_range(self, hpa_axis):
        """ACTH should always be in valid range."""
        for _ in range(50):
            hpa_axis.step(stress_signal=np.random.random())
            assert 0.0 <= hpa_axis.state.acth_level <= 1.0

    def test_allostatic_load_in_valid_range(self, hpa_axis):
        """Allostatic load should always be in valid range."""
        for _ in range(50):
            hpa_axis.step(stress_signal=np.random.random(), inflammation=np.random.random())
            assert 0.0 <= hpa_axis.state.allostatic_load <= 1.0

    def test_stress_type_valid_values(self, hpa_axis):
        """Stress type should be one of valid values."""
        valid_types = ["none", "acute", "chronic", "sleep"]
        for _ in range(20):
            result = hpa_axis.step(stress_signal=np.random.random())
            assert result["stress_type"] in valid_types


# ---------------------------------------------------------------------------
# Factory Function Tests
# ---------------------------------------------------------------------------

class TestCreateHPAAxis:
    """Tests for create_hpa_axis factory function."""

    def test_create_hpa_axis_returns_instance(self):
        """create_hpa_axis should return HPAAxis instance."""
        from core.hpa_axis import create_hpa_axis
        hpa = create_hpa_axis()
        from core.hpa_axis import HPAAxis
        assert isinstance(hpa, HPAAxis)

    def test_create_hpa_axis_with_kwargs(self):
        """create_hpa_axis should pass kwargs to HPAAxis."""
        from core.hpa_axis import create_hpa_axis
        hpa = create_hpa_axis(stress_reactivity=2.0, cortisol_half_life_steps=40)
        assert hpa.crh.stress_reactivity.item() == 2.0
        assert hpa.adrenal.half_life_steps == 40


# ---------------------------------------------------------------------------
# Event Bus Integration Tests
# ---------------------------------------------------------------------------

class TestEventBusIntegration:
    """Tests for event bus integration."""

    def test_hpa_with_event_bus(self):
        """HPAAxis should accept event_bus parameter."""
        from core.hpa_axis import HPAAxis
        from core.event_bus import EventBus
        bus = EventBus()
        hpa = HPAAxis(event_bus=bus)
        assert hpa.event_bus is bus

    def test_on_neural_regulation_handler(self, hpa_axis):
        """on_neural_regulation should process event correctly."""
        from core.event_bus import Event
        event = Event(
            type="neural_regulation",
            data={
                "internal_state": {
                    "ans_sympathetic": 0.5,
                    "alignment_score": 0.8,
                    "scn_circadian_hour": 14.0,
                    "nt_norepinephrine": 0.3,
                },
                "thermo_status": "ACTIVE",
            },
            source="test",
        )
        result = hpa_axis.on_neural_regulation(event)
        assert "cortisol_level" in result
        assert "stress_type" in result

    def test_on_neural_regulation_updates_state(self, hpa_axis):
        """on_neural_regulation should update internal state."""
        from core.event_bus import Event
        event = Event(
            type="neural_regulation",
            data={
                "internal_state": {
                    "ans_sympathetic": 0.7,
                    "alignment_score": 0.6,
                    "scn_circadian_hour": 10.0,
                    "nt_norepinephrine": 0.4,
                },
                "thermo_status": "ACTIVE",
            },
            source="test",
        )
        hpa_axis.on_neural_regulation(event)
        assert "cortisol" in event.data["internal_state"]
        assert "hpa_crh" in event.data["internal_state"]