"""
Unit tests for the acoustic inference engine.

Tests cover:
- Gender estimation from synthetic pitched signals
- Age bracket estimation heuristics
- Edge cases: silence, very short clips, clipped audio
"""

import numpy as np
import pytest

from app.services.inference import (
    AcousticInferenceProvider,
    _estimate_gender,
    _estimate_age_bracket,
    _detect_language,
)

SR = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sine(freq_hz: float, duration_sec: float = 3.0, vol: float = 0.4, sr: int = SR) -> np.ndarray:
    """Generate a pure sine wave — controllable pitch for gender testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) * vol).astype(np.float32)


def make_noise(duration_sec: float = 3.0, vol: float = 0.3, sr: int = SR) -> np.ndarray:
    """White noise — no clear pitch, simulates noisy logistics environment."""
    rng = np.random.default_rng(42)
    return (rng.uniform(-vol, vol, int(sr * duration_sec))).astype(np.float32)


def make_silence(duration_sec: float = 3.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration_sec), dtype=np.float32)


# ---------------------------------------------------------------------------
# Gender tests
# ---------------------------------------------------------------------------

class TestGenderEstimation:

    def test_low_pitch_classified_as_male(self):
        """Pure 100 Hz tone (well in male range) → 'male'."""
        samples = make_sine(freq_hz=100.0)
        pred, conf = _estimate_gender(samples, SR)
        assert pred == "male", f"Expected male for 100Hz, got {pred}"
        assert conf > 0.5

    def test_high_pitch_classified_as_female(self):
        """Pure 220 Hz tone (well in female range) → 'female'."""
        samples = make_sine(freq_hz=220.0)
        pred, conf = _estimate_gender(samples, SR)
        assert pred == "female", f"Expected female for 220Hz, got {pred}"
        assert conf > 0.5

    def test_silence_returns_unknown(self):
        """Silence has no voiced frames → 'unknown'."""
        samples = make_silence()
        pred, conf = _estimate_gender(samples, SR)
        assert pred == "unknown"
        assert conf == 0.0

    def test_very_short_clip_returns_unknown(self):
        """Sub-0.5s clip → insufficient data → 'unknown'."""
        samples = make_sine(freq_hz=120.0, duration_sec=0.3)
        pred, conf = _estimate_gender(samples, SR)
        assert pred == "unknown"

    def test_confidence_is_bounded(self):
        """Confidence must always be in [0, 1]."""
        for freq in [80, 130, 165, 200, 260]:
            samples = make_sine(freq_hz=float(freq))
            _, conf = _estimate_gender(samples, SR)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds for {freq}Hz"

    def test_noisy_audio_returns_valid_output(self):
        """White noise should not crash and should return a valid prediction."""
        samples = make_noise()
        pred, conf = _estimate_gender(samples, SR)
        assert pred in ("male", "female", "unknown")
        assert 0.0 <= conf <= 1.0

    def test_boundary_pitch_low_confidence(self):
        """A 165 Hz tone right on the boundary → low confidence."""
        samples = make_sine(freq_hz=165.0)
        pred, conf = _estimate_gender(samples, SR)
        assert pred in ("male", "female")
        # Confidence should be modest in the ambiguous zone
        assert conf < 0.70, f"Expected low confidence at boundary, got {conf}"


# ---------------------------------------------------------------------------
# Age bracket tests
# ---------------------------------------------------------------------------

class TestAgeBracketEstimation:

    def test_returns_valid_bracket(self):
        """Any clip should return a valid bracket label or 'unknown'."""
        samples = make_sine(freq_hz=180.0)
        pred, conf = _estimate_age_bracket(samples, SR)
        assert pred in ("18-30", "31-45", "46-60", "60+", "unknown")
        assert 0.0 <= conf <= 1.0

    def test_short_clip_returns_unknown(self):
        """Sub-0.5s → unknown."""
        samples = make_sine(freq_hz=150.0, duration_sec=0.3)
        pred, conf = _estimate_age_bracket(samples, SR)
        assert pred == "unknown"

    def test_silence_returns_unknown(self):
        samples = make_silence()
        pred, conf = _estimate_age_bracket(samples, SR)
        assert pred == "unknown"

    def test_confidence_increases_with_duration(self):
        """Longer clips → higher confidence (more reliable features)."""
        short_s = make_sine(freq_hz=200.0, duration_sec=1.5)
        long_s  = make_sine(freq_hz=200.0, duration_sec=5.0)
        _, conf_short = _estimate_age_bracket(short_s, SR)
        _, conf_long  = _estimate_age_bracket(long_s, SR)
        assert conf_long >= conf_short, "Longer clip should have >= confidence"

    def test_noise_does_not_crash(self):
        """Noisy audio should return a valid response."""
        samples = make_noise()
        pred, conf = _estimate_age_bracket(samples, SR)
        assert pred in ("18-30", "31-45", "46-60", "60+", "unknown")


# ---------------------------------------------------------------------------
# Provider integration tests
# ---------------------------------------------------------------------------

class TestAcousticInferenceProvider:

    @pytest.fixture(scope="class")
    def provider(self):
        return AcousticInferenceProvider()

    def test_full_pipeline_male(self, provider):
        """100 Hz signal → full pipeline → male prediction."""
        samples = make_sine(freq_hz=100.0, duration_sec=4.0)
        g_pred, g_conf, a_pred, a_conf, lang = provider.infer_attributes(samples, SR)
        assert g_pred == "male"
        assert g_conf > 0.5
        assert a_pred in ("18-30", "31-45", "46-60", "60+", "unknown")
        assert lang in ("en", "es", "de", "unknown")

    def test_full_pipeline_female(self, provider):
        """220 Hz signal → full pipeline → female prediction."""
        samples = make_sine(freq_hz=220.0, duration_sec=4.0)
        g_pred, g_conf, a_pred, a_conf, lang = provider.infer_attributes(samples, SR)
        assert g_pred == "female"
        assert g_conf > 0.5

    def test_resampling_handled(self, provider):
        """Provider should handle 44100 Hz input gracefully (resamples to 16k)."""
        samples = make_sine(freq_hz=100.0, duration_sec=3.0, sr=44100)
        g_pred, g_conf, a_pred, a_conf, lang = provider.infer_attributes(samples, 44100)
        assert g_pred in ("male", "female", "unknown")

    def test_returns_five_tuple(self, provider):
        """infer_attributes must return exactly 5 values."""
        samples = make_sine(freq_hz=150.0)
        result = provider.infer_attributes(samples, SR)
        assert len(result) == 5

    def test_all_confidence_bounded(self, provider):
        """All confidence scores must be in [0, 1]."""
        samples = make_noise()
        _, g_conf, _, a_conf, _ = provider.infer_attributes(samples, SR)
        assert 0.0 <= g_conf <= 1.0
        assert 0.0 <= a_conf <= 1.0
