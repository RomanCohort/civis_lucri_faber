"""Tests for core.drug_pipeline.pkpd — PK/PD drug simulation.

Tests cover:
- PKPDParams instantiation with default/custom params
- infer_pkpd_params parameter inference
- simulate_pkpd simulation execution
- summarize_pkpd_curve summary statistics
- Edge cases (zero inputs, extreme values)
- Output shape/type validation
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_pkpd_params():
    """Create PKPDParams with default values."""
    from core.drug_pipeline.pkpd import PKPDParams
    return PKPDParams(
        ka=0.5,
        k12=0.1,
        k21=0.08,
        ke=0.05,
        v1_l=3.0,
        emax=1.0,
        ec50_mg_per_l=0.5,
        hill=1.2,
    )


@pytest.fixture
def inferred_params():
    """Create PKPDParams via inference function."""
    from core.drug_pipeline.pkpd import infer_pkpd_params
    return infer_pkpd_params(
        binding=0.5,
        immune=0.5,
        inflammation=0.3,
        dose_mg=10.0,
        freq_per_day=2.0,
    )


# ---------------------------------------------------------------------------
# PKPDParams Instantiation Tests
# ---------------------------------------------------------------------------

class TestPKPDParamsInstantiation:
    """Tests for PKPDParams dataclass instantiation."""

    def test_pkpd_params_creates_with_all_fields(self):
        """PKPDParams should create successfully with all required fields."""
        from core.drug_pipeline.pkpd import PKPDParams
        params = PKPDParams(
            ka=0.5, k12=0.1, k21=0.08, ke=0.05,
            v1_l=3.0, emax=1.0, ec50_mg_per_l=0.5,
        )
        assert params.ka == 0.5
        assert params.k12 == 0.1
        assert params.k21 == 0.08
        assert params.ke == 0.05
        assert params.v1_l == 3.0
        assert params.emax == 1.0
        assert params.ec50_mg_per_l == 0.5

    def test_pkpd_params_hill_default(self):
        """PKPDParams should have hill coefficient default to 1.2."""
        from core.drug_pipeline.pkpd import PKPDParams
        params = PKPDParams(
            ka=0.5, k12=0.1, k21=0.08, ke=0.05,
            v1_l=3.0, emax=1.0, ec50_mg_per_l=0.5,
        )
        assert params.hill == 1.2

    def test_pkpd_params_custom_hill(self):
        """PKPDParams should accept custom hill coefficient."""
        from core.drug_pipeline.pkpd import PKPDParams
        params = PKPDParams(
            ka=0.5, k12=0.1, k21=0.08, ke=0.05,
            v1_l=3.0, emax=1.0, ec50_mg_per_l=0.5,
            hill=2.0,
        )
        assert params.hill == 2.0


# ---------------------------------------------------------------------------
# infer_pkpd_params Tests
# ---------------------------------------------------------------------------

class TestInferPKPDParams:
    """Tests for parameter inference from drug properties."""

    def test_infer_pkpd_params_returns_pkpd_params(self, inferred_params):
        """infer_pkpd_params should return a PKPDParams instance."""
        from core.drug_pipeline.pkpd import PKPDParams
        assert isinstance(inferred_params, PKPDParams)

    def test_infer_pkpd_params_all_fields_populated(self, inferred_params):
        """All PK parameters should be populated after inference."""
        assert inferred_params.ka > 0
        assert inferred_params.k12 > 0
        assert inferred_params.k21 > 0
        assert inferred_params.ke > 0
        assert inferred_params.v1_l > 0
        assert inferred_params.emax > 0
        assert inferred_params.ec50_mg_per_l > 0
        assert inferred_params.hill > 0

    def test_infer_pkpd_params_binding_effect(self):
        """Higher binding should increase absorption rate (ka)."""
        from core.drug_pipeline.pkpd import infer_pkpd_params
        low_binding = infer_pkpd_params(binding=0.1, immune=0.5, inflammation=0.3, dose_mg=10.0, freq_per_day=2.0)
        high_binding = infer_pkpd_params(binding=0.9, immune=0.5, inflammation=0.3, dose_mg=10.0, freq_per_day=2.0)
        assert high_binding.ka > low_binding.ka

    def test_infer_pkpd_params_immune_effect(self):
        """Higher immune activation should increase distribution rate (k12)."""
        from core.drug_pipeline.pkpd import infer_pkpd_params
        low_immune = infer_pkpd_params(binding=0.5, immune=0.1, inflammation=0.3, dose_mg=10.0, freq_per_day=2.0)
        high_immune = infer_pkpd_params(binding=0.5, immune=0.9, inflammation=0.3, dose_mg=10.0, freq_per_day=2.0)
        assert high_immune.k12 > low_immune.k12

    def test_infer_pkpd_params_inflammation_effect(self):
        """Higher inflammation should increase elimination rate (ke)."""
        from core.drug_pipeline.pkpd import infer_pkpd_params
        low_inf = infer_pkpd_params(binding=0.5, immune=0.5, inflammation=0.1, dose_mg=10.0, freq_per_day=2.0)
        high_inf = infer_pkpd_params(binding=0.5, immune=0.5, inflammation=0.9, dose_mg=10.0, freq_per_day=2.0)
        assert high_inf.ke > low_inf.ke


# ---------------------------------------------------------------------------
# simulate_pkpd Tests
# ---------------------------------------------------------------------------

class TestSimulatePKPD:
    """Tests for PK/PD simulation execution."""

    def test_simulate_pkpd_returns_dataframe(self, default_pkpd_params):
        """simulate_pkpd should return a pandas DataFrame."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        assert isinstance(result, pd.DataFrame)

    def test_simulate_pkpd_has_required_columns(self, default_pkpd_params):
        """Result DataFrame should have all required columns."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        required_columns = [
            "time_h",
            "pkpd_depot_mg",
            "pkpd_central_mg",
            "pkpd_peripheral_mg",
            "pkpd_conc_mg_per_l",
            "pkpd_effect",
        ]
        for col in required_columns:
            assert col in result.columns

    def test_simulate_pkpd_non_empty_result(self, default_pkpd_params):
        """Simulation with positive dose should return non-empty result."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params, horizon=24)
        assert len(result) > 0

    def test_simulate_pkpd_time_progression(self, default_pkpd_params):
        """Time values should progress from 0 to horizon."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params, horizon=24)
        assert result["time_h"].iloc[0] >= 0
        assert result["time_h"].iloc[-1] <= 24

    def test_simulate_pkpd_concentration_positive(self, default_pkpd_params):
        """Concentration values should be non-negative."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        assert (result["pkpd_conc_mg_per_l"] >= 0).all()

    def test_simulate_pkpd_effect_in_valid_range(self, default_pkpd_params):
        """Effect values should be in valid range [0, emax]."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        assert (result["pkpd_effect"] >= 0).all()
        assert (result["pkpd_effect"] <= default_pkpd_params.emax * 1.01).all()  # Allow 1% tolerance

    def test_simulate_pkpd_effect_increases_with_dose(self, default_pkpd_params):
        """Higher dose should produce higher peak effect."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        low_dose = simulate_pkpd(dose_mg=5.0, freq_per_day=2.0, params=default_pkpd_params)
        high_dose = simulate_pkpd(dose_mg=20.0, freq_per_day=2.0, params=default_pkpd_params)
        assert high_dose["pkpd_effect"].max() > low_dose["pkpd_effect"].max()


# ---------------------------------------------------------------------------
# summarize_pkpd_curve Tests
# ---------------------------------------------------------------------------

class TestSummarizePKPDCurve:
    """Tests for PK/PD curve summary statistics."""

    def test_summarize_returns_dict(self, default_pkpd_params):
        """summarize_pkpd_curve should return a dictionary."""
        from core.drug_pipeline.pkpd import simulate_pkpd, summarize_pkpd_curve
        curve = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        summary = summarize_pkpd_curve(curve, default_pkpd_params)
        assert isinstance(summary, dict)

    def test_summarize_has_required_keys(self, default_pkpd_params):
        """Summary should contain all required statistical keys."""
        from core.drug_pipeline.pkpd import simulate_pkpd, summarize_pkpd_curve
        curve = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        summary = summarize_pkpd_curve(curve, default_pkpd_params)
        required_keys = [
            "pkpd_half_life_h",
            "pkpd_vd_ss_l",
            "pkpd_clearance_lph",
            "pkpd_cmax_mg_per_l",
            "pkpd_tmax_h",
            "pkpd_auc_conc",
            "pkpd_auc_effect",
            "pkpd_effect_peak",
            "pkpd_pk_effect_corr",
        ]
        for key in required_keys:
            assert key in summary

    def test_summarize_values_are_numeric(self, default_pkpd_params):
        """All summary values should be numeric (float)."""
        from core.drug_pipeline.pkpd import simulate_pkpd, summarize_pkpd_curve
        curve = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        summary = summarize_pkpd_curve(curve, default_pkpd_params)
        for value in summary.values():
            assert isinstance(value, (float, int, np.floating))

    def test_summarize_cmax_positive_with_dose(self, default_pkpd_params):
        """Cmax should be positive when dose is given."""
        from core.drug_pipeline.pkpd import simulate_pkpd, summarize_pkpd_curve
        curve = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        summary = summarize_pkpd_curve(curve, default_pkpd_params)
        assert summary["pkpd_cmax_mg_per_l"] > 0

    def test_summarize_auc_positive_with_dose(self, default_pkpd_params):
        """AUC should be positive when dose is given."""
        from core.drug_pipeline.pkpd import simulate_pkpd, summarize_pkpd_curve
        curve = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        summary = summarize_pkpd_curve(curve, default_pkpd_params)
        assert summary["pkpd_auc_conc"] > 0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_simulate_pkpd_zero_dose(self, default_pkpd_params):
        """Zero dose should produce zero or minimal concentrations."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=0.0, freq_per_day=2.0, params=default_pkpd_params)
        assert (result["pkpd_depot_mg"] == 0).all() or (result["pkpd_conc_mg_per_l"] < 1e-6).all()

    def test_simulate_pkpd_zero_dose_empty_or_minimal_effect(self, default_pkpd_params):
        """Zero dose should produce minimal or zero effect."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=0.0, freq_per_day=2.0, params=default_pkpd_params)
        assert result["pkpd_effect"].max() < 0.01

    def test_infer_pkpd_params_zero_values(self):
        """infer_pkpd_params should handle zero inputs gracefully."""
        from core.drug_pipeline.pkpd import infer_pkpd_params
        params = infer_pkpd_params(
            binding=0.0,
            immune=0.0,
            inflammation=0.0,
            dose_mg=0.0,
            freq_per_day=1.0,
        )
        assert params.ka > 0
        assert params.emax >= 0.2  # Should have minimum emax

    def test_infer_pkpd_params_extreme_values(self):
        """infer_pkpd_params should handle extreme (1.0) inputs."""
        from core.drug_pipeline.pkpd import infer_pkpd_params
        params = infer_pkpd_params(
            binding=1.0,
            immune=1.0,
            inflammation=1.0,
            dose_mg=100.0,
            freq_per_day=4.0,
        )
        assert params.ka <= 1.5  # Should be clipped to max
        assert params.emax <= 2.2

    def test_simulate_pkpd_very_small_horizon(self, default_pkpd_params):
        """Very small horizon should still produce valid output."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(
            dose_mg=10.0,
            freq_per_day=2.0,
            params=default_pkpd_params,
            horizon=2,
        )
        assert isinstance(result, pd.DataFrame)

    def test_summarize_empty_dataframe(self, default_pkpd_params):
        """summarize_pkpd_curve should handle empty DataFrame."""
        from core.drug_pipeline.pkpd import summarize_pkpd_curve
        empty_df = pd.DataFrame()
        summary = summarize_pkpd_curve(empty_df, default_pkpd_params)
        assert summary["pkpd_cmax_mg_per_l"] == 0.0
        assert summary["pkpd_auc_conc"] == 0.0

    def test_simulate_pkpd_high_frequency_dosing(self, default_pkpd_params):
        """High frequency dosing should accumulate more drug."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        low_freq = simulate_pkpd(dose_mg=10.0, freq_per_day=1.0, params=default_pkpd_params, horizon=48)
        high_freq = simulate_pkpd(dose_mg=10.0, freq_per_day=4.0, params=default_pkpd_params, horizon=48)
        # Higher frequency should generally produce higher AUC due to accumulation
        assert high_freq["pkpd_conc_mg_per_l"].mean() >= low_freq["pkpd_conc_mg_per_l"].mean() * 0.8


# ---------------------------------------------------------------------------
# Type/Shape Validation
# ---------------------------------------------------------------------------

class TestOutputValidation:
    """Tests for output type and shape validation."""

    def test_simulate_pkpd_time_column_numeric(self, default_pkpd_params):
        """Time column should be numeric."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        assert np.issubdtype(result["time_h"].dtype, np.number)

    def test_simulate_pkpd_all_columns_numeric(self, default_pkpd_params):
        """All data columns should be numeric."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        for col in result.columns:
            assert np.issubdtype(result[col].dtype, np.number)

    def test_simulate_pkpd_no_nan_values(self, default_pkpd_params):
        """Simulation should not produce NaN values."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        assert not result.isna().any().any()

    def test_simulate_pkpd_no_inf_values(self, default_pkpd_params):
        """Simulation should not produce infinite values."""
        from core.drug_pipeline.pkpd import simulate_pkpd
        result = simulate_pkpd(dose_mg=10.0, freq_per_day=2.0, params=default_pkpd_params)
        assert not np.isinf(result.values).any()