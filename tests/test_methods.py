"""
Unit tests for modular signal processing methods.
"""
import numpy as np
import pytest
from AFX.methods.stft import stft, magnitude_spectrogram, power_spectrogram
from AFX.methods.mel import mel_filterbank, apply_mel_filterbank, log_mel_spectrogram
from AFX.methods.dct import extract_mfcc_coefficients, dct_ii


def test_stft_basic():
    """Test basic STFT functionality."""
    signal = np.random.randn(2048)
    sr = 22050

    result = stft(signal, frame_size=1024, hop_length=512)

    # Check shape
    expected_n_freq = 1024 // 2 + 1  # 513
    expected_n_frames = 1 + (len(signal) + 1024 - 1024) // 512  # With center padding

    assert result.shape[0] == expected_n_freq
    assert result.shape[1] > 0
    assert result.dtype == np.complex128


def test_stft_magnitude_power():
    """Test magnitude and power spectrogram computation."""
    signal = np.random.randn(1024)
    stft_result = stft(signal, frame_size=512, hop_length=256)

    mag_spec = magnitude_spectrogram(stft_result)
    power_spec = power_spectrogram(stft_result)

    # Check shapes
    assert mag_spec.shape == stft_result.shape
    assert power_spec.shape == stft_result.shape

    # Check relationships
    np.testing.assert_array_almost_equal(mag_spec, np.abs(stft_result))
    np.testing.assert_array_almost_equal(power_spec, np.abs(stft_result) ** 2)

    # Check that values are non-negative
    assert np.all(mag_spec >= 0)
    assert np.all(power_spec >= 0)


def test_mel_filterbank():
    """Test mel filterbank construction."""
    n_mels = 40
    n_fft = 1024
    sr = 22050

    filterbank = mel_filterbank(n_mels, n_fft, sr)

    # Check shape
    expected_n_freq = n_fft // 2 + 1
    assert filterbank.shape == (n_mels, expected_n_freq)

    # Check that filters are non-negative
    assert np.all(filterbank >= 0)

    # Check that each filter sums to a reasonable value (with normalization)
    filter_sums = np.sum(filterbank, axis=1)
    assert np.all(filter_sums > 0)  # All filters should have positive area


def test_mel_filterbank_frequency_range():
    """Test mel filterbank respects frequency bounds."""
    n_mels = 20
    n_fft = 512
    sr = 16000
    fmin = 300.0
    fmax = 4000.0

    filterbank = mel_filterbank(n_mels, n_fft, sr, fmin=fmin, fmax=fmax)

    # Create frequency array
    freqs = np.fft.fftfreq(n_fft, 1.0/sr)[:n_fft//2 + 1]

    # Check that filters are zero outside the specified range
    low_freq_bins = freqs < fmin
    high_freq_bins = freqs > fmax

    if np.any(low_freq_bins):
        assert np.allclose(filterbank[:, low_freq_bins], 0, atol=1e-10)
    if np.any(high_freq_bins):
        assert np.allclose(filterbank[:, high_freq_bins], 0, atol=1e-10)


def test_apply_mel_filterbank():
    """Test application of mel filterbank to spectrogram."""
    n_freq = 513
    n_frames = 10
    n_mels = 40

    # Create a fake spectrogram
    spectrogram = np.random.rand(n_freq, n_frames)

    # Create a fake filterbank
    filterbank = np.random.rand(n_mels, n_freq)

    result = apply_mel_filterbank(spectrogram, filterbank)

    # Check shape
    assert result.shape == (n_mels, n_frames)

    # Check that it's equivalent to matrix multiplication
    expected = np.dot(filterbank, spectrogram)
    np.testing.assert_array_almost_equal(result, expected)


def test_log_mel_spectrogram():
    """Test log mel spectrogram conversion."""
    mel_spec = np.random.rand(40, 20) * 100  # Positive values

    log_mel = log_mel_spectrogram(mel_spec)

    # Check shape
    assert log_mel.shape == mel_spec.shape

    # Check that it's in dB scale
    expected = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    np.testing.assert_array_almost_equal(log_mel, expected)


def test_log_mel_spectrogram_dynamic_range():
    """Test log mel spectrogram dynamic range limiting."""
    mel_spec = np.array([[1000.0, 1.0], [0.1, 0.001]])  # Wide dynamic range

    log_mel = log_mel_spectrogram(mel_spec, top_db=60.0)

    # Check that dynamic range is limited
    max_val = np.max(log_mel)
    min_val = np.min(log_mel)
    dynamic_range = max_val - min_val

    assert dynamic_range <= 60.0


def test_dct_ii():
    """Test DCT-II computation."""
    x = np.random.randn(64, 10)

    result = dct_ii(x, norm='ortho', axis=0)

    # Check shape
    assert result.shape == x.shape

    # Test with scipy directly for comparison
    from scipy.fftpack import dct
    expected = dct(x, type=2, norm='ortho', axis=0)
    np.testing.assert_array_almost_equal(result, expected)


def test_extract_mfcc_coefficients():
    """Test MFCC coefficient extraction from log mel spectrogram."""
    n_mels = 128
    n_frames = 20
    n_mfcc = 13

    log_mel_spec = np.random.randn(n_mels, n_frames)

    mfcc = extract_mfcc_coefficients(log_mel_spec, n_mfcc)

    # Check shape
    assert mfcc.shape == (n_mfcc, n_frames)

    # Test that it's equivalent to manual DCT
    from scipy.fftpack import dct
    expected = dct(log_mel_spec, type=2, norm='ortho', axis=0)[:n_mfcc, :]
    np.testing.assert_array_almost_equal(mfcc, expected)


def test_mfcc_pipeline_integration():
    """Test the complete MFCC pipeline using modular components."""
    # Create a test signal
    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))

    # MFCC parameters
    frame_size = 2048
    hop_length = 512
    n_mels = 128
    n_mfcc = 13

    # Step 1: STFT
    stft_result = stft(signal, frame_size=frame_size, hop_length=hop_length)

    # Step 2: Power spectrogram
    power_spec = power_spectrogram(stft_result)

    # Step 3: Mel filterbank
    filterbank = mel_filterbank(n_mels, frame_size, sr)
    mel_spec = apply_mel_filterbank(power_spec, filterbank)

    # Step 4: Log scale
    log_mel = log_mel_spectrogram(mel_spec)

    # Step 5: MFCC
    mfcc = extract_mfcc_coefficients(log_mel, n_mfcc)

    # Check final shape
    assert mfcc.shape[0] == n_mfcc
    assert mfcc.shape[1] > 0  # Should have some frames

    # Check that values are reasonable
    assert np.isfinite(mfcc).all()
    assert not np.isnan(mfcc).any()
