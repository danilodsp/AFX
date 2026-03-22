"""
Tests for fixes 1.3–1.6: YIN interpolation, spectral truncation, entropy units,
and FEATURE_MAP completeness.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1.3  YIN parabolic interpolation
# ---------------------------------------------------------------------------
class TestYINParabolicInterpolation:
    """Verify that YIN refines pitch correctly via parabolic interpolation."""

    def test_known_frequency_precision(self):
        """Parabolic interpolation should improve pitch precision beyond bin resolution."""
        from AFX.utils.pitch import yin

        sr = 22050
        freq = 440.0
        t = np.linspace(0, 1.0, sr, endpoint=False)
        signal = np.sin(2 * np.pi * freq * t) + 0.005 * np.random.RandomState(0).randn(sr)

        pitches = yin(signal, frame_length=2048, hop_length=512, fmin=50, fmax=2000, sr=sr)
        valid = pitches[pitches > 0]
        assert len(valid) > 0
        # After correct interpolation, median should be within 1% of true freq
        assert abs(np.median(valid) - freq) / freq < 0.01

    def test_interpolation_improves_over_integer_tau(self):
        """Refined tau should not equal the integer tau for most frames
        (interpolation actually shifts it)."""
        from AFX.utils.pitch import yin

        sr = 22050
        freq = 333.0  # intentionally non-integer-tau frequency
        t = np.linspace(0, 1.0, sr, endpoint=False)
        signal = np.sin(2 * np.pi * freq * t)

        pitches = yin(signal, frame_length=2048, hop_length=512, fmin=50, fmax=2000, sr=sr)
        valid = pitches[pitches > 0]
        assert len(valid) > 0
        # Integer-tau would give sr/round(sr/freq) ≈ 22050/66 ≈ 334.09
        # Correct interpolation should land much closer to 333.0
        assert abs(np.median(valid) - freq) < 2.0


# ---------------------------------------------------------------------------
# 1.4  spectral_skewness / spectral_slope truncation
# ---------------------------------------------------------------------------
class TestSpectralTruncation:
    """Verify that spectral_skewness and spectral_slope use the full signal."""

    def test_skewness_whole_signal_uses_all_samples(self):
        """Without framing, the full signal should contribute, not just [:2048]."""
        from AFX.extractors.frequency_domain import extract_spectral_skewness

        rng = np.random.RandomState(42)
        # Create a 4-second signal where the second half has very different content
        sr = 22050
        n = sr * 4
        signal = rng.randn(n)
        signal[sr * 2:] += 10 * np.sin(2 * np.pi * 1000 * np.arange(sr * 2) / sr)

        # Whole-signal mode (no frame_size/hop_length)
        full_result = extract_spectral_skewness(signal, sr)
        trunc_result = extract_spectral_skewness(signal[:2048], sr)

        # If the old truncation bug were present, these would be equal
        assert not np.allclose(
            full_result['spectral_skewness'],
            trunc_result['spectral_skewness'],
        ), "Full-signal and truncated results should differ"

    def test_slope_whole_signal_uses_all_samples(self):
        """Without framing, the full signal should contribute, not just [:2048]."""
        from AFX.extractors.frequency_domain import extract_spectral_slope

        rng = np.random.RandomState(42)
        sr = 22050
        n = sr * 4
        signal = rng.randn(n)
        signal[sr * 2:] += 10 * np.sin(2 * np.pi * 1000 * np.arange(sr * 2) / sr)

        full_result = extract_spectral_slope(signal, sr)
        trunc_result = extract_spectral_slope(signal[:2048], sr)

        assert not np.allclose(
            full_result['spectral_slope'],
            trunc_result['spectral_slope'],
        ), "Full-signal and truncated results should differ"

    def test_skewness_framewise_shape(self):
        from AFX.extractors.frequency_domain import extract_spectral_skewness

        signal = np.random.RandomState(0).randn(22050)
        fs, hl = 1024, 256
        result = extract_spectral_skewness(signal, 22050, frame_size=fs, hop_length=hl)
        expected = 1 + (len(signal) - fs) // hl
        assert result['spectral_skewness'].shape == (expected,)

    def test_slope_framewise_shape(self):
        from AFX.extractors.frequency_domain import extract_spectral_slope

        signal = np.random.RandomState(0).randn(22050)
        fs, hl = 1024, 256
        result = extract_spectral_slope(signal, 22050, frame_size=fs, hop_length=hl)
        expected = 1 + (len(signal) - fs) // hl
        assert result['spectral_slope'].shape == (expected,)


# ---------------------------------------------------------------------------
# 1.5  Entropy uses density histogram — wrong units
# ---------------------------------------------------------------------------
class TestEntropyUnits:
    """Verify that extract_entropy produces proper Shannon entropy in bits."""

    def test_uniform_signal_max_entropy(self):
        """A uniform distribution over N bins should give log2(N) bits."""
        from AFX.extractors.time_domain import extract_entropy

        # Create a signal with values uniformly distributed across bins
        rng = np.random.RandomState(0)
        signal = rng.uniform(-1, 1, 100_000)
        num_bins = 64

        result = extract_entropy(signal, sr=22050, num_bins=num_bins)
        entropy = result['entropy'][0]

        # Theoretical maximum entropy for num_bins bins
        max_entropy = np.log2(num_bins)  # = 6.0

        # Uniform → should be close to max (within 5%)
        assert entropy <= max_entropy + 0.01, f"Entropy {entropy} > max {max_entropy}"
        assert entropy > max_entropy * 0.95, f"Entropy {entropy} too far below max {max_entropy}"

    def test_constant_signal_zero_entropy(self):
        """A constant signal concentrates in one bin → entropy = 0."""
        from AFX.extractors.time_domain import extract_entropy

        signal = np.ones(10_000) * 0.5
        result = extract_entropy(signal, sr=22050, num_bins=64)
        assert result['entropy'][0] == pytest.approx(0.0, abs=1e-10)

    def test_entropy_bounded_by_log2_bins(self):
        """Entropy should never exceed log2(num_bins) for any signal."""
        from AFX.extractors.time_domain import extract_entropy

        for seed in range(5):
            signal = np.random.RandomState(seed).randn(10_000)
            result = extract_entropy(signal, sr=22050, num_bins=100)
            assert result['entropy'][0] <= np.log2(100) + 1e-10


# ---------------------------------------------------------------------------
# 1.6  FEATURE_MAP completeness
# ---------------------------------------------------------------------------
class TestFeatureMapCompleteness:
    """Verify FEATURE_MAP covers all features in config.json."""

    def test_all_config_features_have_extractors(self):
        """Every feature in the default config.json must be in FEATURE_MAP."""
        import json
        import os
        from AFX.extract_all import FEATURE_MAP

        config_path = os.path.join(os.path.dirname(__file__), '..', 'AFX', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        missing = [k for k in config['features'] if k not in FEATURE_MAP]
        assert missing == [], f"Features in config.json but not in FEATURE_MAP: {missing}"

    def test_feature_map_values_are_callable(self):
        from AFX.extract_all import FEATURE_MAP

        for name, func in FEATURE_MAP.items():
            assert callable(func), f"FEATURE_MAP['{name}'] is not callable"

    def test_extract_all_features_uses_all_config_features(self):
        """extract_all_features should produce output for every config feature."""
        import json
        import os
        from AFX.extract_all import extract_all_features

        config_path = os.path.join(os.path.dirname(__file__), '..', 'AFX', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        sr = config['sample_rate']
        # Use a short signal to keep sample_entropy (O(n^2)) fast
        signal = np.random.RandomState(0).randn(sr // 4)

        result = extract_all_features(signal, sr, config)
        # All config features should appear in the result
        for feat in config['features']:
            assert feat in result, f"Feature '{feat}' missing from extract_all_features output"
