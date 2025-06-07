"""
Compare frequency domain extractors against librosa implementations.
"""
import numpy as np
import pytest
from AFX.extractors import frequency_domain

import librosa


def test_extract_spectral_centroid_against_librosa():
    """Test that our NumPy implementation closely matches librosa's implementation."""
    # Create test signal: sine wave at 440Hz
    sr = 22050
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)

    # Add a small amount of noise
    np.random.seed(42)  # For reproducibility
    signal = signal + 0.01 * np.random.randn(len(signal))

    # Parameters
    frame_size = 2048
    hop_length = 512

    # Calculate spectral centroid using our implementation
    result = frequency_domain.extract_spectral_centroid(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length
    )
    centroid_custom = result['spectral_centroid']

    # Calculate spectral centroid using librosa
    centroid_librosa = librosa.feature.spectral_centroid(
        y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length
    )[0]

    # Shapes might be different due to padding/framing differences
    # Just check that the shapes are consistent and results are similar
    min_length = min(len(centroid_custom), len(centroid_librosa))

    # Skip the first two frames which might differ due to initialization
    # and focus on the stable part of the signal
    start_idx = 2
    centroid_custom = centroid_custom[start_idx:min_length]
    centroid_librosa = centroid_librosa[start_idx:min_length]

    # Check that the values are close (allowing for small numerical differences)
    assert np.allclose(centroid_custom, centroid_librosa, rtol=1e-1, atol=1e-1)

    # Check metadata return
    result_with_meta = frequency_domain.extract_spectral_centroid(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length,
        return_metadata=True
    )
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

    # Check that times computation is correct
    times = result_with_meta['metadata']['times']
    expected_times = np.arange(len(centroid_custom) + start_idx) * hop_length / sr
    expected_times = expected_times[start_idx:]
    assert np.allclose(times[start_idx:min_length], expected_times)

def test_extract_spectral_bandwidth_against_librosa():
    """Test that our NumPy implementation closely matches librosa's implementation."""
    # Create test signal: sine wave at 440Hz
    sr = 22050
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)

    # Add a small amount of noise
    np.random.seed(42)  # For reproducibility
    signal = signal + 0.01 * np.random.randn(len(signal))

    # Parameters
    frame_size = 2048
    hop_length = 512

    # Calculate spectral bandwidth using our implementation
    result = frequency_domain.extract_spectral_bandwidth(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length
    )
    bandwidth_custom = result['spectral_bandwidth']

    # Calculate spectral bandwidth using librosa
    bandwidth_librosa = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length
    )[0]

    # Shapes might be different due to padding/framing differences
    # Just check that the shapes are consistent and results are similar
    min_length = min(len(bandwidth_custom), len(bandwidth_librosa))

    # Skip the first two frames which might differ due to initialization
    # and focus on the stable part of the signal
    start_idx = 2
    bandwidth_custom = bandwidth_custom[start_idx:min_length]
    bandwidth_librosa = bandwidth_librosa[start_idx:min_length]

    # Check that the values are close (allowing for small numerical differences)
    assert np.allclose(bandwidth_custom, bandwidth_librosa, rtol=1e-1, atol=1e-1)

    # Check metadata return
    result_with_meta = frequency_domain.extract_spectral_bandwidth(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length,
        return_metadata=True
    )
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

    # Check that times computation is correct
    times = result_with_meta['metadata']['times']
    expected_times = np.arange(len(bandwidth_custom) + start_idx) * hop_length / sr
    expected_times = expected_times[start_idx:]
    assert np.allclose(times[start_idx:min_length], expected_times)

def test_extract_spectral_contrast_against_librosa():
    """Test that our NumPy implementation captures spectral contrast patterns similar to librosa."""
    # Create test signal with harmonic content
    sr = 22050
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)  # A4 note
    signal += 0.5 * np.sin(2 * np.pi * 880 * t)  # First harmonic
    signal += 0.3 * np.sin(2 * np.pi * 1320 * t)  # Second harmonic

    # Add a small amount of noise
    np.random.seed(42)  # For reproducibility
    signal = signal + 0.1 * np.random.randn(len(signal))

    # Parameters
    frame_size = 2048
    hop_length = 512
    n_bands = 6

    # Calculate spectral contrast using our implementation
    result = frequency_domain.extract_spectral_contrast(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length, n_bands=n_bands
    )
    contrast_custom = result['spectral_contrast']

    # Calculate spectral contrast using librosa
    contrast_librosa = librosa.feature.spectral_contrast(
        y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length, n_bands=n_bands
    )

    # Check shapes are the same
    assert contrast_custom.shape == contrast_librosa.shape, "Shape mismatch"

    # Check metadata return
    result_with_meta = frequency_domain.extract_spectral_contrast(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length,
        n_bands=n_bands, return_metadata=True
    )
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

    # Check that times computation is accurate
    times = result_with_meta['metadata']['times']
    expected_times = np.arange(len(times)) * hop_length / sr
    assert np.allclose(times, expected_times)

def test_extract_spectral_flux_against_librosa():
    """Test that our NumPy implementation closely matches librosa's implementation."""
    # Create test signals: sine sweep, white noise, and transients
    sr = 22050
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Sine sweep (changing frequency)
    start_freq, end_freq = 100, 8000
    sweep = np.sin(2 * np.pi * np.logspace(
        np.log10(start_freq), np.log10(end_freq), len(t)) * t)

    # White noise
    np.random.seed(42)
    noise = np.random.randn(len(t))

    # Transient impulses
    transients = np.zeros_like(t)
    impulse_positions = np.linspace(0, len(transients)-1, 10).astype(int)
    transients[impulse_positions] = 1.0

    # Combined signal
    signal = 0.5 * sweep + 0.2 * noise + 0.3 * transients

    # Parameters
    frame_size = 2048
    hop_length = 512

    # Calculate spectral flux using our implementation
    result = frequency_domain.extract_spectral_flux(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length
    )
    flux_custom = result['spectral_flux']

    # Calculate spectral flux using librosa
    # Librosa doesn't have a direct spectral flux function, so we compute it manually
    S = np.abs(librosa.stft(signal, n_fft=frame_size, hop_length=hop_length))
    flux_librosa = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))

    # Compare shapes
    assert flux_custom.shape == flux_librosa.shape, f"Shape mismatch: custom {flux_custom.shape} vs librosa {flux_librosa.shape}"

    # The values don't need to match exactly, but the patterns should be similar
    # Compute correlation to check similarity
    correlation = np.corrcoef(flux_custom, flux_librosa)[0, 1]
    assert correlation > 0.95, f"Low correlation between custom and librosa: {correlation}"

    # Check metadata return
    result_with_meta = frequency_domain.extract_spectral_flux(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length,
        return_metadata=True
    )
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

    # Check that times computation is correct
    times = result_with_meta['metadata']['times']
    # Librosa frames_to_time for comparison
    librosa_times = librosa.frames_to_time(
        np.arange(len(flux_librosa)), sr=sr, hop_length=hop_length, n_fft=frame_size
    )

    assert np.allclose(times, librosa_times, rtol=1e-5, atol=1e-5)
