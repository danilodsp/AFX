"""
Regression tests for pitch and HNR framing behavior.

These tests verify correct behavior after fixing the double-framing bug
where @framewise_extractor and internal framing were both active.
"""
import numpy as np
import pytest
from AFX.extractors.harmonic_features import extract_pitch, extract_hnr

SR = 22050
FRAME_SIZE = 2048
HOP_LENGTH = 512


def _make_sine(freq: float = 440.0, duration: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def _make_noisy_sine(freq: float = 440.0, duration: float = 1.0,
                     noise_level: float = 0.01, sr: int = SR, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t) + noise_level * rng.randn(int(sr * duration))


# ---------------------------------------------------------------------------
# extract_pitch
# ---------------------------------------------------------------------------

class TestExtractPitch:
    """Tests for extract_pitch framing correctness."""

    def test_returns_correct_key(self):
        result = extract_pitch(_make_sine(), SR)
        assert 'pitch' in result

    def test_output_shape_default_params(self):
        signal = _make_sine()
        result = extract_pitch(signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH)
        expected_n_frames = 1 + (len(signal) - FRAME_SIZE) // HOP_LENGTH
        assert result['pitch'].shape == (expected_n_frames,)

    def test_non_default_frame_size(self):
        """The old bug: with non-default frame_size, decorator framed to 1024
        then yin() saw len < frame_length(2048) and returned [0.0] for every frame."""
        signal = _make_noisy_sine(freq=440.0, duration=1.0)
        fs, hl = 1024, 256
        result = extract_pitch(signal, SR, frame_size=fs, hop_length=hl)
        expected_n_frames = 1 + (len(signal) - fs) // hl
        assert result['pitch'].shape == (expected_n_frames,)
        # At least some frames should detect the 440 Hz pitch
        valid = result['pitch'][result['pitch'] > 0]
        assert len(valid) > 0, "No pitch detected with non-default frame_size"

    def test_different_frame_sizes_detect_same_pitch(self):
        """Different frame sizes should detect roughly the same fundamental."""
        signal = _make_noisy_sine(freq=440.0, duration=1.0)
        results = {}
        for fs, hl in [(2048, 512), (1024, 256), (4096, 1024)]:
            r = extract_pitch(signal, SR, frame_size=fs, hop_length=hl)
            valid = r['pitch'][r['pitch'] > 0]
            if len(valid) > 0:
                results[fs] = np.median(valid)

        assert len(results) >= 2, "Not enough frame sizes produced valid pitch"
        medians = list(results.values())
        # All medians should be within 10% of 440 Hz
        for m in medians:
            assert abs(m - 440.0) < 44.0, f"Median pitch {m} too far from 440 Hz"

    def test_metadata(self):
        signal = _make_sine()
        result = extract_pitch(
            signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH,
            return_metadata=True,
        )
        assert 'metadata' in result
        assert 'times' in result['metadata']
        n_frames = len(result['pitch'])
        expected_times = np.arange(n_frames) * HOP_LENGTH / SR
        np.testing.assert_allclose(result['metadata']['times'], expected_times)

    def test_sine_detects_correct_frequency(self):
        signal = _make_noisy_sine(freq=440.0, duration=1.0)
        result = extract_pitch(signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH)
        valid = result['pitch'][result['pitch'] > 0]
        assert len(valid) > 0
        assert abs(np.median(valid) - 440.0) < 22.0  # within 5%


# ---------------------------------------------------------------------------
# extract_hnr
# ---------------------------------------------------------------------------

class TestExtractHNR:
    """Tests for extract_hnr framing correctness."""

    def test_returns_correct_key(self):
        result = extract_hnr(_make_sine(), SR)
        assert 'hnr' in result

    def test_whole_signal_returns_scalar(self):
        """Without frame_size/hop_length, should return a single value."""
        result = extract_hnr(_make_sine(), SR)
        assert result['hnr'].shape == (1,)

    def test_framewise_output_shape(self):
        signal = _make_sine()
        result = extract_hnr(signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH)
        expected_n_frames = 1 + (len(signal) - FRAME_SIZE) // HOP_LENGTH
        assert result['hnr'].shape == (expected_n_frames,)

    def test_non_default_frame_size(self):
        """The old bug: with non-default frame_size, internal loop saw
        n_frames = max(1, 1 + (1024 - 2048) // 512) = 1, producing a
        single value per decorator frame instead of correct behavior."""
        signal = _make_sine()
        fs, hl = 1024, 256
        result = extract_hnr(signal, SR, frame_size=fs, hop_length=hl)
        expected_n_frames = 1 + (len(signal) - fs) // hl
        assert result['hnr'].shape == (expected_n_frames,)
        assert np.all(np.isfinite(result['hnr']))

    def test_pure_tone_higher_hnr_than_noise(self):
        """Pure sine should have higher HNR than noisy signal."""
        pure = _make_sine(freq=440.0)
        noisy = _make_noisy_sine(freq=440.0, noise_level=1.0)
        hnr_pure = extract_hnr(pure, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH)['hnr']
        hnr_noisy = extract_hnr(noisy, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH)['hnr']
        assert np.median(hnr_pure) > np.median(hnr_noisy)

    def test_hnr_finite_values(self):
        """HNR should produce finite values for normal signals."""
        for signal in [_make_sine(), _make_noisy_sine()]:
            result = extract_hnr(signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH)
            assert np.all(np.isfinite(result['hnr']))

    def test_metadata(self):
        signal = _make_sine()
        result = extract_hnr(
            signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH,
            return_metadata=True,
        )
        assert 'metadata' in result
        assert 'times' in result['metadata']
        n_frames = len(result['hnr'])
        expected_times = np.arange(n_frames) * HOP_LENGTH / SR
        np.testing.assert_allclose(result['metadata']['times'], expected_times)

    def test_silent_signal(self):
        """Silent signal should return NaN (zero autocorrelation)."""
        signal = np.zeros(SR)
        result = extract_hnr(signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH)
        assert np.all(np.isnan(result['hnr']))
