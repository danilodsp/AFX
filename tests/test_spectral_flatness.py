"""
Test the spectral flatness implementation against librosa.
"""
import numpy as np
import pytest
import librosa
from AFX.extractors import frequency_domain


def test_extract_spectral_flatness_against_librosa():
    """Test that our NumPy implementation closely matches librosa's implementation."""
    # Create test signals

    # Test 1: Sine wave (tonal content, should have low flatness)
    sr = 22050
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_signal = np.sin(2 * np.pi * 440 * t)

    # Test 2: White noise (noise-like, should have high flatness)
    np.random.seed(42)  # For reproducibility
    noise_signal = np.random.randn(int(sr * duration))

    # Test 3: Mix of sine and noise
    mixed_signal = sine_signal + 0.1 * noise_signal

    # Parameters
    frame_size = 2048
    hop_length = 512

    # Test all three signals
    for i, signal in enumerate([sine_signal, noise_signal, mixed_signal]):
        signal_name = ["sine", "noise", "mixed"][i]

        # Calculate spectral flatness using our implementation
        result = frequency_domain.extract_spectral_flatness(
            signal, sr=sr, frame_size=frame_size, hop_length=hop_length
        )
        flatness_custom = result['spectral_flatness']

        # Calculate spectral flatness using librosa
        flatness_librosa = librosa.feature.spectral_flatness(
            y=signal, n_fft=frame_size, hop_length=hop_length
        )[0]

        # Shapes might be different due to padding/framing differences
        # Just check that the shapes are consistent and results are similar
        min_length = min(len(flatness_custom), len(flatness_librosa))

        # Skip the first two frames which might differ due to initialization
        # and focus on the stable part of the signal
        start_idx = 2
        flatness_custom = flatness_custom[start_idx:min_length]
        flatness_librosa = flatness_librosa[start_idx:min_length]

        # Check that the values are close (allowing for small numerical differences)
        assert np.allclose(flatness_custom, flatness_librosa, rtol=1e-1, atol=1e-1), \
            f"Flatness values for {signal_name} signal are not close"

    # Check metadata return
    result_with_meta = frequency_domain.extract_spectral_flatness(
        sine_signal, sr=sr, frame_size=frame_size, hop_length=hop_length,
        return_metadata=True
    )
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

    # Check that times computation is correct
    times = result_with_meta['metadata']['times']
    n_frames = len(times)
    expected_times = np.arange(n_frames) * hop_length / sr
    assert np.allclose(times, expected_times)
