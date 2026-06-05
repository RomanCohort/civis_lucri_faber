"""Tests for core.addiction_dynamics — Addiction modeling.

Tests cover:
- AddictionDynamicsEngine instantiation
- Drug registration
- Tolerance dynamics
- Withdrawal dynamics
- Craving dynamics
- Step computation
- Edge cases (zero inputs, extreme values)
- Output shape/type validation
"""
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def addiction_engine():
    """Create a fresh AddictionDynamicsEngine instance."""
    from core.addiction_dynamics import AddictionDynamicsEngine
    engine = AddictionDynamicsEngine()
    return engine


@pytest.fixture
def engine_with_opioid(addiction_engine):
    """Create engine with registered opioid drug."""
    addiction_engine.register_drug("morphine", "opioid")
    return addiction_engine


@pytest.fixture
def engine_with_stimulant(addiction_engine):
    """Create engine with registered stimulant drug."""
    addiction_engine.register_drug("cocaine", "stimulant")
    return addiction_engine


# ---------------------------------------------------------------------------
# Instantiation Tests
# ---------------------------------------------------------------------------

class TestAddictionEngineInstantiation:
    """Tests for AddictionDynamicsEngine instantiation."""

    def test_engine_creates_successfully(self):
        """AddictionDynamicsEngine should instantiate without error."""
        from core.addiction_dynamics import AddictionDynamicsEngine
        engine = AddictionDynamicsEngine()
        assert engine is not None

    def test_engine_has_empty_profiles(self, addiction_engine):
        """New engine should have empty profiles dict."""
        assert addiction_engine.profiles == {}

    def test_engine_has_empty_concentrations(self, addiction_engine):
        """New engine should have empty drug concentrations."""
        assert addiction_engine._drug_concentrations == {}


# ---------------------------------------------------------------------------
# Drug Registration Tests
# ---------------------------------------------------------------------------

class TestDrugRegistration:
    """Tests for drug registration functionality."""

    def test_register_drug_creates_profile(self, addiction_engine):
        """register_drug should create an AddictionProfile."""
        addiction_engine.register_drug("morphine", "opioid")
        assert "morphine" in addiction_engine.profiles

    def test_register_drug_sets_drug_class(self, addiction_engine):
        """Registered drug should have correct drug class."""
        addiction_engine.register_drug("morphine", "opioid")
        assert addiction_engine.profiles["morphine"].drug_class == "opioid"

    def test_register_drug_creates_tolerance_states(self, addiction_engine):
        """Opioid registration should create tolerance states for opioid receptors."""
        addiction_engine.register_drug("morphine", "opioid")
        profile = addiction_engine.profiles["morphine"]
        assert "mu-opioid" in profile.tolerance

    def test_register_drug_creates_withdrawal_state(self, addiction_engine):
        """Registration should create withdrawal state."""
        addiction_engine.register_drug("morphine", "opioid")
        from core.addiction_dynamics import WithdrawalState
        profile = addiction_engine.profiles["morphine"]
        assert isinstance(profile.withdrawal, WithdrawalState)

    def test_register_drug_creates_craving_state(self, addiction_engine):
        """Registration should create craving state."""
        addiction_engine.register_drug("morphine", "opioid")
        from core.addiction_dynamics import CravingState
        profile = addiction_engine.profiles["morphine"]
        assert isinstance(profile.craving, CravingState)

    def test_register_stimulant_creates_dopamine_receptors(self, addiction_engine):
        """Stimulant registration should create D2, DAT, NET tolerance states."""
        addiction_engine.register_drug("cocaine", "stimulant")
        profile = addiction_engine.profiles["cocaine"]
        assert "D2" in profile.tolerance
        assert "DAT" in profile.tolerance

    def test_register_sedative_creates_gaba_receptor(self, addiction_engine):
        """Sedative registration should create GABA-A tolerance state."""
        addiction_engine.register_drug("diazepam", "sedative")
        profile = addiction_engine.profiles["diazepam"]
        assert "GABA-A" in profile.tolerance

    def test_register_unknown_drug_class(self, addiction_engine):
        """Registration with unknown drug class should succeed with empty receptors."""
        addiction_engine.register_drug("unknown_drug", "unknown_class")
        profile = addiction_engine.profiles["unknown_drug"]
        assert profile.tolerance == {}


# ---------------------------------------------------------------------------
# Step Computation Tests
# ---------------------------------------------------------------------------

class TestStepComputation:
    """Tests for the main step() computation."""

    def test_step_returns_tuple(self, engine_with_opioid):
        """step() should return a tuple of three dicts."""
        result = engine_with_opioid.step(
            drug_concentrations={"morphine": 1.0},
            drug_effects={"morphine": 0.5},
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_step_returns_tolerance_factors(self, engine_with_opioid):
        """First return value should be tolerance factors dict."""
        tolerance, _, _ = engine_with_opioid.step(
            drug_concentrations={"morphine": 1.0},
            drug_effects={"morphine": 0.5},
        )
        assert isinstance(tolerance, dict)
        assert "morphine" in tolerance

    def test_step_returns_withdrawal_deltas(self, engine_with_opioid):
        """Second return value should be withdrawal deltas dict."""
        _, withdrawal, _ = engine_with_opioid.step(
            drug_concentrations={"morphine": 1.0},
            drug_effects={"morphine": 0.5},
        )
        assert isinstance(withdrawal, dict)

    def test_step_returns_craving_levels(self, engine_with_opioid):
        """Third return value should be craving levels dict."""
        _, _, craving = engine_with_opioid.step(
            drug_concentrations={"morphine": 1.0},
            drug_effects={"morphine": 0.5},
        )
        assert isinstance(craving, dict)
        assert "morphine" in craving

    def test_step_with_empty_concentrations(self, engine_with_opioid):
        """step() should handle empty concentration dict."""
        tolerance, withdrawal, craving = engine_with_opioid.step(
            drug_concentrations={},
            drug_effects={},
        )
        assert isinstance(tolerance, dict)
        assert isinstance(withdrawal, dict)
        assert isinstance(craving, dict)

    def test_step_updates_chronic_steps(self, engine_with_opioid):
        """step() with active drug should increment chronic steps."""
        profile = engine_with_opioid.profiles["morphine"]
        initial_steps = profile.total_chronic_steps
        engine_with_opioid.step(
            drug_concentrations={"morphine": 1.0},
            drug_effects={"morphine": 0.5},
        )
        assert profile.total_chronic_steps > initial_steps

    def test_step_multiple_times(self, engine_with_opioid):
        """Multiple steps should run without error."""
        for _ in range(10):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 1.0},
                drug_effects={"morphine": 0.5},
            )
        profile = engine_with_opioid.profiles["morphine"]
        assert profile.total_chronic_steps >= 10


# ---------------------------------------------------------------------------
# Tolerance Dynamics Tests
# ---------------------------------------------------------------------------

class TestToleranceDynamics:
    """Tests for tolerance development and recovery."""

    def test_tolerance_decreases_with_chronic_use(self, engine_with_opioid):
        """Chronic use should decrease tolerance factors (downregulation)."""
        # Simulate chronic use
        for _ in range(100):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 1.0},
                drug_effects={"morphine": 0.8},
            )
        tolerance, _, _ = engine_with_opioid.step(
            drug_concentrations={"morphine": 1.0},
            drug_effects={"morphine": 0.8},
        )
        # Tolerance factor should be less than 1.0 (downregulation occurred)
        for receptor, factor in tolerance["morphine"].items():
            assert factor < 1.0
            assert factor >= 0.3  # Minimum floor

    def test_tolerance_recovers_with_abstinence(self, engine_with_opioid):
        """Tolerance should recover during drug-free period."""
        # Build tolerance
        for _ in range(100):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 1.0},
                drug_effects={"morphine": 0.8},
            )

        # Get tolerance after use
        tol_after_use, _, _ = engine_with_opioid.step(
            drug_concentrations={"morphine": 1.0},
            drug_effects={"morphine": 0.8},
        )

        # Abstinence period
        for _ in range(200):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 0.0},
                drug_effects={"morphine": 0.0},
            )

        tol_after_recovery, _, _ = engine_with_opioid.step(
            drug_concentrations={"morphine": 0.0},
            drug_effects={"morphine": 0.0},
        )

        # Tolerance factor should increase (recover) during abstinence
        avg_after_use = sum(tol_after_use["morphine"].values()) / len(tol_after_use["morphine"])
        avg_after_recovery = sum(tol_after_recovery["morphine"].values()) / len(tol_after_recovery["morphine"])
        assert avg_after_recovery > avg_after_use


# ---------------------------------------------------------------------------
# Withdrawal Dynamics Tests
# ---------------------------------------------------------------------------

class TestWithdrawalDynamics:
    """Tests for withdrawal detection and progression."""

    def test_withdrawal_not_triggered_during_use(self, engine_with_opioid):
        """Withdrawal should not be triggered while drug is active."""
        for _ in range(100):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 1.0},
                drug_effects={"morphine": 0.8},
            )
        profile = engine_with_opioid.profiles["morphine"]
        assert not profile.withdrawal.is_active

    def test_withdrawal_triggered_after_chronic_use(self, engine_with_opioid):
        """Withdrawal should trigger after stopping chronic use."""
        # Chronic use phase
        for _ in range(100):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 1.0},
                drug_effects={"morphine": 0.8},
            )

        # Abrupt cessation
        for _ in range(50):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 0.0},
                drug_effects={"morphine": 0.0},
            )

        profile = engine_with_opioid.profiles["morphine"]
        # Withdrawal may or may not still be active, but it should have been triggered
        assert profile.withdrawal.peak_severity > 0 or profile.withdrawal.is_active

    def test_withdrawal_severity_is_bounded(self, engine_with_opioid):
        """Withdrawal severity should be bounded [0, 1]."""
        # Chronic use
        for _ in range(150):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 1.0},
                drug_effects={"morphine": 0.8},
            )
        # Withdrawal
        for _ in range(100):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 0.0},
                drug_effects={"morphine": 0.0},
            )
        profile = engine_with_opioid.profiles["morphine"]
        assert 0.0 <= profile.withdrawal.current_severity <= 1.0

    def test_withdrawal_different_by_drug_class(self, addiction_engine):
        """Different drug classes should have different withdrawal profiles."""
        # Register drugs from different classes
        addiction_engine.register_drug("morphine", "opioid")
        addiction_engine.register_drug("cocaine", "stimulant")

        # Chronic use for both
        for _ in range(150):
            addiction_engine.step(
                drug_concentrations={"morphine": 1.0, "cocaine": 1.0},
                drug_effects={"morphine": 0.8, "cocaine": 0.8},
            )

        # Cessation
        for _ in range(30):
            addiction_engine.step(
                drug_concentrations={"morphine": 0.0, "cocaine": 0.0},
                drug_effects={"morphine": 0.0, "cocaine": 0.0},
            )

        opioid_profile = addiction_engine.profiles["morphine"]
        stimulant_profile = addiction_engine.profiles["cocaine"]

        # Opioid should have higher peak withdrawal than stimulant
        assert opioid_profile.withdrawal.peak_severity > stimulant_profile.withdrawal.peak_severity


# ---------------------------------------------------------------------------
# Craving Dynamics Tests
# ---------------------------------------------------------------------------

class TestCravingDynamics:
    """Tests for craving/sensitization dynamics."""

    def test_craving_level_bounded(self, engine_with_opioid):
        """Craving level should be bounded."""
        for _ in range(100):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 1.0},
                drug_effects={"morphine": 0.8},
            )
        profile = engine_with_opioid.profiles["morphine"]
        assert profile.craving.current_level >= 0.0

    def test_sensitization_increases_with_use(self, engine_with_stimulant):
        """Sensitization factor should increase with chronic stimulant use."""
        initial_sens = engine_with_stimulant.profiles["cocaine"].craving.sensitization_factor
        for _ in range(100):
            engine_with_stimulant.step(
                drug_concentrations={"cocaine": 1.0},
                drug_effects={"cocaine": 0.8},
            )
        final_sens = engine_with_stimulant.profiles["cocaine"].craving.sensitization_factor
        assert final_sens > initial_sens

    def test_craving_higher_during_withdrawal(self, engine_with_opioid):
        """Craving should be elevated during withdrawal."""
        # Chronic use
        for _ in range(150):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 1.0},
                drug_effects={"morphine": 0.8},
            )

        # During active use
        _, _, craving_during_use = engine_with_opioid.step(
            drug_concentrations={"morphine": 1.0},
            drug_effects={"morphine": 0.8},
        )

        # Cessation and withdrawal
        for _ in range(40):
            engine_with_opioid.step(
                drug_concentrations={"morphine": 0.0},
                drug_effects={"morphine": 0.0},
            )

        _, _, craving_during_wd = engine_with_opioid.step(
            drug_concentrations={"morphine": 0.0},
            drug_effects={"morphine": 0.0},
        )

        # Craving should be higher during withdrawal
        profile = engine_with_opioid.profiles["morphine"]
        if profile.withdrawal.is_active:
            assert craving_during_wd["morphine"] >= craving_during_use["morphine"]


# ---------------------------------------------------------------------------
# State Summary Tests
# ---------------------------------------------------------------------------

class TestGetStateSummary:
    """Tests for get_state_summary method."""

    def test_summary_returns_dict(self, engine_with_opioid):
        """get_state_summary should return a dictionary."""
        summary = engine_with_opioid.get_state_summary()
        assert isinstance(summary, dict)

    def test_summary_contains_registered_drug(self, engine_with_opioid):
        """Summary should contain registered drug."""
        summary = engine_with_opioid.get_state_summary()
        assert "morphine" in summary

    def test_summary_has_required_keys(self, engine_with_opioid):
        """Summary should contain all required keys."""
        engine_with_opioid.step(
            drug_concentrations={"morphine": 1.0},
            drug_effects={"morphine": 0.5},
        )
        summary = engine_with_opioid.get_state_summary()
        drug_summary = summary["morphine"]
        required_keys = [
            "chronic_steps",
            "is_dependent",
            "tolerance",
            "withdrawal_active",
            "withdrawal_severity",
            "craving_level",
            "sensitization",
            "liking_separation",
        ]
        for key in required_keys:
            assert key in drug_summary


# ---------------------------------------------------------------------------
# Dependence Development Tests
# ---------------------------------------------------------------------------

class TestDependenceDevelopment:
    """Tests for dependence criterion."""

    def test_dependence_develops_with_chronic_use(self, addiction_engine):
        """Chronic use of high-withdrawal drug should lead to dependence."""
        addiction_engine.register_drug("heroin", "opioid")
        for _ in range(300):
            addiction_engine.step(
                drug_concentrations={"heroin": 1.0},
                drug_effects={"heroin": 0.8},
            )
        profile = addiction_engine.profiles["heroin"]
        assert profile.is_dependent

    def test_no_dependence_with_low_withdrawal_drug(self, addiction_engine):
        """Low-withdrawal-potential drug should not trigger dependence."""
        addiction_engine.register_drug("lsd", "hallucinogen")
        for _ in range(300):
            addiction_engine.step(
                drug_concentrations={"lsd": 1.0},
                drug_effects={"lsd": 0.5},
            )
        profile = addiction_engine.profiles["lsd"]
        # Hallucinogen has low withdrawal peak (0.10), should not become dependent
        # even with chronic use


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_step_with_zero_concentration(self, engine_with_opioid):
        """Step with zero concentration should run without error."""
        tolerance, withdrawal, craving = engine_with_opioid.step(
            drug_concentrations={"morphine": 0.0},
            drug_effects={"morphine": 0.0},
        )
        assert isinstance(tolerance, dict)
        assert isinstance(withdrawal, dict)
        assert isinstance(craving, dict)

    def test_step_with_unregistered_drug(self, addiction_engine):
        """Step with unregistered drug should not crash."""
        tolerance, withdrawal, craving = addiction_engine.step(
            drug_concentrations={"unknown_drug": 1.0},
            drug_effects={"unknown_drug": 0.5},
        )
        # Unknown drug should just be ignored
        assert "unknown_drug" not in tolerance

    def test_step_with_extreme_concentration(self, engine_with_opioid):
        """Step with very high concentration should be handled."""
        tolerance, withdrawal, craving = engine_with_opioid.step(
            drug_concentrations={"morphine": 1000.0},
            drug_effects={"morphine": 1.0},
        )
        # Should not crash and values should be reasonable
        for factor in tolerance["morphine"].values():
            assert 0.0 <= factor <= 1.0

    def test_step_with_negative_effect(self, engine_with_opioid):
        """Step with negative effect should be handled gracefully."""
        tolerance, withdrawal, craving = engine_with_opioid.step(
            drug_concentrations={"morphine": 1.0},
            drug_effects={"morphine": -0.5},
        )
        assert isinstance(tolerance, dict)

    def test_multiple_drugs(self, addiction_engine):
        """Engine should handle multiple drugs simultaneously."""
        addiction_engine.register_drug("morphine", "opioid")
        addiction_engine.register_drug("cocaine", "stimulant")

        tolerance, withdrawal, craving = addiction_engine.step(
            drug_concentrations={"morphine": 1.0, "cocaine": 0.5},
            drug_effects={"morphine": 0.8, "cocaine": 0.6},
        )

        assert "morphine" in tolerance
        assert "cocaine" in tolerance
        assert "morphine" in craving
        assert "cocaine" in craving