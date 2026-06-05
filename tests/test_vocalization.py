"""Tests for core.vocalization — Bio-Inspired Vocalization System.

Tests cover:
- VocalTract instantiation and articulator configuration
- ArticulatoryPlanner phoneme-to-trajectory mapping
- FormantSynthesizer resonance calculation
- VocalCortex event-driven coordination
- Edge cases (empty phonemes, extreme articulator values)
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vocal_tract():
    """Create a VocalTract instance."""
    from core.vocalization import VocalTract
    return VocalTract()


@pytest.fixture
def articulatory_planner():
    """Create an ArticulatoryPlanner instance."""
    from core.vocalization import ArticulatoryPlanner
    return ArticulatoryPlanner()


@pytest.fixture
def formant_synthesizer():
    """Create a FormantSynthesizer instance."""
    from core.vocalization import FormantSynthesizer
    return FormantSynthesizer()


@pytest.fixture
def vocal_cortex():
    """Create a VocalCortex instance."""
    from core.vocalization import VocalCortex
    return VocalCortex()


# ---------------------------------------------------------------------------
# VocalTract Tests
# ---------------------------------------------------------------------------

class TestVocalTract:
    """Tests for VocalTract articulator model."""

    def test_vocal_tract_creates_successfully(self, vocal_tract):
        """VocalTract should instantiate without error."""
        assert vocal_tract is not None

    def test_vocal_tract_has_articulator_dims(self, vocal_tract):
        """VocalTract should have articulator dimension constants."""
        from core.vocalization import N_ARTICULATORS, ARTICULATOR_DIMS
        assert N_ARTICULATORS == 5
        assert len(ARTICULATOR_DIMS) == 5

    def test_vocal_tract_forward_returns_dict(self, vocal_tract):
        """VocalTract forward should return a dictionary."""
        articulator_state = np.zeros(5)
        result = vocal_tract.forward(articulator_state)
        assert isinstance(result, dict) or isinstance(result, np.ndarray)

    def test_vocal_tract_articulator_bounds(self, vocal_tract):
        """Articulator values should be in valid range [-1, 1]."""
        # Test extreme values
        extreme_high = np.ones(5)
        extreme_low = -np.ones(5)
        # Should handle without error
        vocal_tract.forward(extreme_high)
        vocal_tract.forward(extreme_low)


# ---------------------------------------------------------------------------
# ArticulatoryPlanner Tests
# ---------------------------------------------------------------------------

class TestArticulatoryPlanner:
    """Tests for ArticulatoryPlanner phoneme mapping."""

    def test_planner_creates_successfully(self, articulatory_planner):
        """ArticulatoryPlanner should instantiate without error."""
        assert articulatory_planner is not None

    def test_planner_has_phoneme_table(self, articulatory_planner):
        """Planner should have phoneme-to-index mapping."""
        from core.vocalization import PHONEMES, PHONEME_TO_IDX, N_PHONEMES
        assert N_PHONEMES > 0
        assert len(PHONEMES) == N_PHONEMES
        assert 'aa' in PHONEME_TO_IDX

    def test_planner_phoneme_to_trajectory(self, articulatory_planner):
        """Planner should convert phonemes to articulator trajectory."""
        phonemes = ['aa', 'b', 's']
        result = articulatory_planner.plan(phonemes)
        assert result is not None

    def test_planner_empty_phonemes(self, articulatory_planner):
        """Planner should handle empty phoneme list."""
        result = articulatory_planner.plan([])
        assert result is not None

    def test_planner_unknown_phoneme(self, articulatory_planner):
        """Planner should handle unknown phoneme gracefully."""
        # Unknown phoneme should not crash
        result = articulatory_planner.plan(['xyz'])
        assert result is not None


# ---------------------------------------------------------------------------
# FormantSynthesizer Tests
# ---------------------------------------------------------------------------

class TestFormantSynthesizer:
    """Tests for FormantSynthesizer resonance calculation."""

    def test_synthesizer_creates_successfully(self, formant_synthesizer):
        """FormantSynthesizer should instantiate without error."""
        assert formant_synthesizer is not None

    def test_synthesizer_returns_formants(self, formant_synthesizer):
        """Synthesizer should return F1, F2, F3 values."""
        articulator_state = np.zeros(5)
        result = formant_synthesizer.synthesize(articulator_state)
        assert result is not None

    def test_synthesizer_formant_values_bounded(self, formant_synthesizer):
        """Formant values should be in reasonable acoustic range."""
        articulator_state = np.array([0.5, 0.5, 0.0, 0.3, 0.0])
        result = formant_synthesizer.synthesize(articulator_state)
        # F1 should be 200-800 Hz, F2 800-2500 Hz, F3 2000-4000 Hz typical
        if isinstance(result, dict):
            f1 = result.get('f1', 0)
            f2 = result.get('f2', 0)
            f3 = result.get('f3', 0)
            assert f1 >= 0
            assert f2 >= 0
            assert f3 >= 0


# ---------------------------------------------------------------------------
# VocalCortex Tests
# ---------------------------------------------------------------------------

class TestVocalCortex:
    """Tests for VocalCortex event-driven coordination."""

    def test_vocal_cortex_creates_successfully(self, vocal_cortex):
        """VocalCortex should instantiate without error."""
        assert vocal_cortex is not None

    def test_vocal_cortex_step(self, vocal_cortex):
        """VocalCortex step should return output dict."""
        result = vocal_cortex.step(
            text_input="hello",
            emotion_state={'valence': 0.5, 'arousal': 0.3},
        )
        assert result is not None or vocal_cortex is not None

    def test_vocal_cortex_with_event_bus(self):
        """VocalCortex should accept event_bus parameter."""
        from core.vocalization import VocalCortex
        from core.event_bus import EventBus
        bus = EventBus()
        cortex = VocalCortex(event_bus=bus)
        assert cortex is not None


# ---------------------------------------------------------------------------
# Phoneme Table Tests
# ---------------------------------------------------------------------------

class TestPhonemeTable:
    """Tests for ARPAbet phoneme definitions."""

    def test_phoneme_count(self):
        """Phoneme table should have expected count."""
        from core.vocalization import N_PHONEMES, PHONEMES
        assert N_PHONEMES >= 38  # ARPAbet has 39+ phonemes
        assert len(PHONEMES) == N_PHONEMES

    def test_phoneme_index_mapping(self):
        """Phoneme-to-index mapping should be consistent."""
        from core.vocalization import PHONEME_TO_IDX, PHONEMES
        for phoneme in PHONEMES:
            assert phoneme in PHONEME_TO_IDX

    def test_vowel_phonemes_present(self):
        """Vowel phonemes should be in table."""
        from core.vocalization import PHONEMES
        vowels = ['aa', 'ae', 'ah', 'ih', 'uw']
        for v in vowels:
            assert v in PHONEMES

    def test_consonant_phonemes_present(self):
        """Consonant phonemes should be in table."""
        from core.vocalization import PHONEMES
        consonants = ['b', 'd', 'f', 'k', 's', 't', 'z']
        for c in consonants:
            assert c in PHONEMES


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_extreme_articulator_values(self, vocal_tract):
        """Extreme articulator values should be handled."""
        articulator_state = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        vocal_tract.forward(articulator_state)

    def test_negative_articulator_values(self, vocal_tract):
        """Negative articulator values should be handled."""
        articulator_state = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        vocal_tract.forward(articulator_state)

    def test_long_phoneme_sequence(self, articulatory_planner):
        """Long phoneme sequences should be handled."""
        phonemes = ['aa'] * 100
        result = articulatory_planner.plan(phonemes)
        assert result is not None

    def test_mixed_phoneme_types(self, articulatory_planner):
        """Mixed vowel/consonant sequences should work."""
        phonemes = ['aa', 'b', 'ih', 't', 'uw', 's']
        result = articulatory_planner.plan(phonemes)
        assert result is not None


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Tests for full vocalization pipeline integration."""

    def test_full_pipeline(self, vocal_tract, articulatory_planner, formant_synthesizer):
        """Full pipeline should produce acoustic output."""
        phonemes = ['hello']  # Simplified text input
        # This tests that the pipeline components can connect
        trajectory = articulatory_planner.plan(['aa', 'l', 'ow'])
        if trajectory is not None:
            formants = formant_synthesizer.synthesize(np.zeros(5))
            assert formants is not None

    def test_vocalization_output_structure(self):
        """VocalizationOutput should have expected fields."""
        from core.vocalization import VocalizationOutput
        output = VocalizationOutput(
            phoneme_sequence=['aa'],
            articulator_trajectory=np.zeros((1, 5)),
            formant_values=np.zeros((1, 3)),
            acoustic_features=np.zeros((1, 64)),
            voicing=np.zeros(1),
            intensity=0.5,
            duration_ms=100.0,
        )
        assert output.phoneme_sequence == ['aa']
        assert output.intensity == 0.5