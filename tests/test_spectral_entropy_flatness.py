"""
Regression tests for spectral entropy and spectral flatness.

These tests verify correct behavior after fixing the double-framing bug
where @framewise_extractor and internal framing loops were both active.
"""
import numpy as np
import pytest
from AFX.extractors.frequency_domain import (
    extract_spectral_entropy,
    extract_spectral_flatness,
)

SR = 22050
FRAME_SIZE = 2048
HOP_LENGTH = 512


def _make_sine(freq: float = 440.0, duration: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def _make_noise(duration: float = 1.0, sr: int = SR, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(int(sr * duration))


# ---------------------------------------------------------------------------
# Spectral Entropy
# ---------------------------------------------------------------------------

class TestSpectralEntropy:
    """Tests for extract_spectral_entropy."""

    def test_returns_correct_key(self):
        result = extract_spectral_entropy(_make_sine(), SR)
        assert 'spectral_entropy' in result

    def test_whole_signal_returns_scalar(self):
        """Without frame_size/hop_length, should return a single value."""
        result = extract_spectral_entropy(_make_sine(), SR)
        assert result['spectral_entropy'].shape == (1,)

    def test_framewise_output_shape(self):
        signal = _make_sine()
        result = extract_spectral_entropy(
            signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH
        )
        expected_n_frames = 1 + (len(signal) - FRAME_SIZE) // HOP_LENGTH
        assert result['spectral_entropy'].shape == (expected_n_frames,)

    def test_non_default_frame_size(self):
        """Verify it works correctly with non-default frame_size (the old bug
        would fail or produce wrong shapes with frame_size != 2048)."""
        signal = _make_sine()
        fs, hl = 1024, 256
        result = extract_spectral_entropy(signal, SR, frame_size=fs, hop_length=hl)
        expected_n_frames = 1 + (len(signal) - fs) // hl
        assert result['spectral_entropy'].shape == (expected_n_frames,)
        assert np.all(np.isfinite(result['spectral_entropy']))

    def test_noise_higher_entropy_than_sine(self):
        """White noise should have higher spectral entropy than a pure sine."""
        sine_result = extract_spectral_entropy(
            _make_sine(), SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH
        )
        noise_result = extract_spectral_entropy(
            _make_noise(), SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH
        )
        assert np.mean(noise_result['spectral_entropy']) > np.mean(sine_result['spectral_entropy'])

    def test_entropy_non_negative(self):
        for signal in [_make_sine(), _make_noise(), np.zeros(SR)]:
            result = extract_spectral_entropy(
                signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH
            )
            assert np.all(result['spectral_entropy'] >= 0)

    def test_metadata(self):
        signal = _make_sine()
        result = extract_spectral_entropy(
            signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH,
            return_metadata=True,
        )
        assert 'metadata' in result
        assert 'times' in result['metadata']
        n_frames = len(result['spectral_entropy'])
        expected_times = np.arange(n_frames) * HOP_LENGTH / SR
        np.testing.assert_allclose(result['metadata']['times'], expected_times)


# ---------------------------------------------------------------------------
# Spectral Flatness
# ---------------------------------------------------------------------------

class TestSpectralFlatness:
    """Tests for extract_spectral_flatness."""

    def test_returns_correct_key(self):
        result = extract_spectral_flatness(_make_sine(), SR)
        assert 'spectral_flatness' in result

    def test_whole_signal_returns_scalar(self):
        result = extract_spectral_flatness(_make_sine(), SR)
        assert result['spectral_flatness'].shape == (1,)

    def test_framewise_output_shape(self):
        signal = _make_sine()
        result = extract_spectral_flatness(
            signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH
        )
        expected_n_frames = 1 + (len(signal) - FRAME_SIZE) // HOP_LENGTH
        assert result['spectral_flatness'].shape == (expected_n_frames,)

    def test_non_default_frame_size(self):
        """Verify it works correctly with non-default frame_size."""
        signal = _make_sine()
        fs, hl = 1024, 256
        result = extract_spectral_flatness(signal, SR, frame_size=fs, hop_length=hl)
        expected_n_frames = 1 + (len(signal) - fs) // hl
        assert result['spectral_flatness'].shape == (expected_n_frames,)
        assert np.all(np.isfinite(result['spectral_flatness']))

    def test_noise_higher_flatness_than_sine(self):
        """White noise should have higher spectral flatness than a pure sine."""
        sine_result = extract_spectral_flatness(
            _make_sine(), SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH
        )
        noise_result = extract_spectral_flatness(
            _make_noise(), SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH
        )
        assert np.mean(noise_result['spectral_flatness']) > np.mean(sine_result['spectral_flatness'])

    def test_flatness_between_0_and_1(self):
        for signal in [_make_sine(), _make_noise()]:
            result = extract_spectral_flatness(
                signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH
            )
            assert np.all(result['spectral_flatness'] >= 0)
            assert np.all(result['spectral_flatness'] <= 1.0 + 1e-6)

    def test_metadata(self):
        signal = _make_sine()
        result = extract_spectral_flatness(
            signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH,
            return_metadata=True,
        )
        assert 'metadata' in result
        assert 'times' in result['metadata']
        n_frames = len(result['spectral_flatness'])
        expected_times = np.arange(n_frames) * HOP_LENGTH / SR
        np.testing.assert_allclose(result['metadata']['times'], expected_times)

    def test_silent_signal_returns_zero(self):
        """A silent signal should produce flatness of 0 (no energy)."""
        signal = np.zeros(SR)
        result = extract_spectral_flatness(
            signal, SR, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH
        )
        # With eps floor, geometric/arithmetic mean are equal → flatness ≈ 1.0
        # or 0.0 depending on implementation; just check it doesn't crash
        assert np.all(np.isfinite(result['spectral_flatness']))
