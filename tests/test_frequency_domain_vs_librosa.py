"""
Compare frequency domain extractors against librosa implementations.
"""
import librosa
import numpy as np

from afx_audio.extractors import frequency_domain


def test_extract_spectral_centroid_against_librosa():
    """Test that our NumPy implementation closely matches librosa's implementation."""
    sr = 22050
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)

    np.random.seed(42)
    signal = signal + 0.01 * np.random.randn(len(signal))

    frame_size = 2048
    hop_length = 512

    result = frequency_domain.extract_spectral_centroid(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length
    )
    centroid_custom = result['spectral_centroid']

    centroid_librosa = librosa.feature.spectral_centroid(
        y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length
    )[0]

    assert centroid_custom.shape == centroid_librosa.shape
    np.testing.assert_allclose(centroid_custom, centroid_librosa, rtol=1e-6)

    result_with_meta = frequency_domain.extract_spectral_centroid(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length,
        return_metadata=True
    )
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

    # Centered by default, as in librosa: frame i sits at i * hop
    times = result_with_meta['metadata']['times']
    assert np.allclose(times, librosa.frames_to_time(np.arange(len(times)), sr=sr, hop_length=hop_length))

def test_extract_spectral_bandwidth_against_librosa():
    """Test that our NumPy implementation closely matches librosa's implementation."""
    sr = 22050
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)

    np.random.seed(42)
    signal = signal + 0.01 * np.random.randn(len(signal))

    frame_size = 2048
    hop_length = 512

    result = frequency_domain.extract_spectral_bandwidth(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length
    )
    bandwidth_custom = result['spectral_bandwidth']

    bandwidth_librosa = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length
    )[0]

    assert bandwidth_custom.shape == bandwidth_librosa.shape
    np.testing.assert_allclose(bandwidth_custom, bandwidth_librosa, rtol=1e-6)

    result_with_meta = frequency_domain.extract_spectral_bandwidth(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length,
        return_metadata=True
    )
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

    # Centered by default, as in librosa: frame i sits at i * hop
    times = result_with_meta['metadata']['times']
    assert np.allclose(times, librosa.frames_to_time(np.arange(len(times)), sr=sr, hop_length=hop_length))

def test_extract_spectral_contrast_against_librosa():
    """Test that our NumPy implementation captures spectral contrast patterns similar to librosa."""
    sr = 22050
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)
    signal += 0.5 * np.sin(2 * np.pi * 880 * t)
    signal += 0.3 * np.sin(2 * np.pi * 1320 * t)

    np.random.seed(42)
    signal = signal + 0.1 * np.random.randn(len(signal))

    frame_size = 2048
    hop_length = 512
    n_bands = 6

    result = frequency_domain.extract_spectral_contrast(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length, n_bands=n_bands
    )
    contrast_custom = result['spectral_contrast']

    contrast_librosa = librosa.feature.spectral_contrast(
        y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length, n_bands=n_bands
    )

    assert contrast_custom.shape == contrast_librosa.shape, "Shape mismatch"

    result_with_meta = frequency_domain.extract_spectral_contrast(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length,
        n_bands=n_bands, return_metadata=True
    )
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

    times = result_with_meta['metadata']['times']
    expected_times = np.arange(len(times)) * hop_length / sr
    assert np.allclose(times, expected_times)

def test_extract_spectral_flux_against_librosa():
    """Test that our NumPy implementation closely matches librosa's implementation."""
    sr = 22050
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    start_freq, end_freq = 100, 8000
    sweep = np.sin(2 * np.pi * np.logspace(
        np.log10(start_freq), np.log10(end_freq), len(t)) * t)

    np.random.seed(42)
    noise = np.random.randn(len(t))

    transients = np.zeros_like(t)
    impulse_positions = np.linspace(0, len(transients)-1, 10).astype(int)
    transients[impulse_positions] = 1.0

    signal = 0.5 * sweep + 0.2 * noise + 0.3 * transients

    frame_size = 2048
    hop_length = 512

    result = frequency_domain.extract_spectral_flux(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length
    )
    flux_custom = result['spectral_flux']

    # librosa has no spectral flux; this is the same L2 frame difference on librosa.stft
    S = np.abs(librosa.stft(signal, n_fft=frame_size, hop_length=hop_length))
    flux_librosa = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))

    assert flux_custom.shape == flux_librosa.shape, (
        f"Shape mismatch: custom {flux_custom.shape} vs librosa {flux_librosa.shape}"
    )

    # spectral_flux pads centered frames by reflection, librosa.stft by zeros,
    # so the edge frames differ and only the pattern is compared
    correlation = np.corrcoef(flux_custom, flux_librosa)[0, 1]
    assert correlation > 0.95, f"Low correlation between custom and librosa: {correlation}"

    result_with_meta = frequency_domain.extract_spectral_flux(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length,
        return_metadata=True
    )
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

    # flux[i] compares centered frames i and i+1 and is stamped with the
    # time of frame i+1
    times = result_with_meta['metadata']['times']
    librosa_times = librosa.frames_to_time(
        np.arange(1, len(flux_librosa) + 1), sr=sr, hop_length=hop_length
    )

    assert np.allclose(times, librosa_times, rtol=1e-5, atol=1e-5)
