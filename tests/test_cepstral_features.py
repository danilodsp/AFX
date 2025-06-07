"""
Unit tests for cepstral_features extractors.
"""
import numpy as np
import pytest
from AFX.extractors import cepstral_features

import librosa


def test_extract_mfcc():
    signal = np.random.randn(22050)
    result = cepstral_features.extract_mfcc(signal, sr=22050)
    assert 'mfcc' in result
    assert isinstance(result['mfcc'], np.ndarray)

def test_extract_mfcc_shape_and_metadata():
    """Test MFCC output shape and metadata."""
    np.random.seed(42)
    signal = np.random.randn(22050)

    # Test without metadata
    result = cepstral_features.extract_mfcc(signal, sr=22050, n_mfcc=13)
    assert result['mfcc'].shape[0] == 13  # n_mfcc coefficients
    assert result['mfcc'].shape[1] > 0    # Some number of frames

    # Test with metadata
    result_meta = cepstral_features.extract_mfcc(signal, sr=22050, n_mfcc=13, return_metadata=True)
    assert 'metadata' in result_meta
    assert 'times' in result_meta['metadata']
    assert result_meta['metadata']['times'].shape[0] == result_meta['mfcc'].shape[1]

def test_extract_mfcc_librosa_compatibility():
    """Test that our MFCC implementation produces reasonable results compared to librosa."""
    # Set seed for reproducible tests
    np.random.seed(42)

    # Test signal (white noise works better for comparison than tonal signals)
    signal = np.random.randn(22050) * 0.1
    sr = 22050
    n_mfcc = 13

    # Our implementation
    result_ours = cepstral_features.extract_mfcc(
        signal,
        sr=sr,
        n_mfcc=n_mfcc,
        center=True,
        return_metadata=True,
    )
    mfcc_ours = result_ours['mfcc']
    times_ours = result_ours['metadata']['times']

    # Librosa implementation
    mfcc_librosa = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    times_librosa = librosa.frames_to_time(
        np.arange(mfcc_librosa.shape[1]), sr=sr, hop_length=512, n_fft=2048
    )

    # Check shapes match
    assert mfcc_ours.shape == mfcc_librosa.shape
    assert times_ours.shape == times_librosa.shape

    # Check timing alignment
    time_diff = np.abs(times_ours - times_librosa)
    assert np.max(time_diff) < 1e-6, "Time alignment should be very precise"

    # Check feature similarity (allowing for implementation differences)
    cosine_similarities = []
    for i in range(n_mfcc):
        cos_sim = np.dot(mfcc_ours[i], mfcc_librosa[i]) / (
            np.linalg.norm(mfcc_ours[i]) * np.linalg.norm(mfcc_librosa[i]) + 1e-10
        )
        cosine_similarities.append(cos_sim)

    avg_cosine_sim = np.mean(cosine_similarities)
    min_cosine_sim = np.min(cosine_similarities)

    # Mean Absolute Error
    mae = np.mean(np.abs(mfcc_ours - mfcc_librosa))

    # Reasonable similarity thresholds (implementation differences are expected)
    assert avg_cosine_sim > 0.70, f"Average cosine similarity too low: {avg_cosine_sim}"
    assert min_cosine_sim > 0.3, f"Minimum cosine similarity too low: {min_cosine_sim}"
    assert mae < 30.0, f"Mean absolute error too high: {mae}"

def test_extract_mfcc_delta():
    signal = np.random.randn(22050)
    result = cepstral_features.extract_mfcc_delta(signal, sr=22050)
    assert 'mfcc_delta' in result
    assert isinstance(result['mfcc_delta'], np.ndarray)

def test_extract_mfcc_delta_librosa_compatibility():
    """Test that our MFCC delta implementation produces reasonable results compared to librosa."""
    # Set seed for reproducible tests
    np.random.seed(42)

    # Test signal
    signal = np.random.randn(22050) * 0.1
    sr = 22050
    n_mfcc = 13

    # Our implementation (order=1)
    result_ours = cepstral_features.extract_mfcc_delta(
        signal,
        sr=sr,
        n_mfcc=n_mfcc,
        order=1,
        center=True,
        return_metadata=True,
    )
    delta_ours = result_ours['mfcc_delta']
    times_ours = result_ours['metadata']['times']

    # Librosa implementation for comparison
    mfcc_librosa = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    delta_librosa = librosa.feature.delta(mfcc_librosa, width=9, order=1)
    times_librosa = librosa.frames_to_time(
        np.arange(delta_librosa.shape[1]), sr=sr, hop_length=512, n_fft=2048
    )

    # Check shapes match
    assert delta_ours.shape == delta_librosa.shape
    assert times_ours.shape == times_librosa.shape

    # Check timing alignment
    time_diff = np.abs(times_ours - times_librosa)
    assert np.max(time_diff) < 1e-6, "Time alignment should be very precise"

    # Check feature similarity (allowing for implementation differences)
    cosine_similarities = []
    for i in range(n_mfcc):
        cos_sim = np.dot(delta_ours[i], delta_librosa[i]) / (
            np.linalg.norm(delta_ours[i]) * np.linalg.norm(delta_librosa[i]) + 1e-10
        )
        cosine_similarities.append(cos_sim)

    avg_cosine_sim = np.mean(cosine_similarities)
    min_cosine_sim = np.min(cosine_similarities)

    # Mean Absolute Error (scaled by signal range for robustness)
    signal_range = np.max(delta_librosa) - np.min(delta_librosa)
    mae = np.mean(np.abs(delta_ours - delta_librosa)) / (signal_range + 1e-10)

    # Reasonable similarity thresholds (more lenient than MFCC since delta computation can vary)
    assert avg_cosine_sim > 0.3, f"Average cosine similarity too low: {avg_cosine_sim}"
    assert min_cosine_sim > -0.5, f"Minimum cosine similarity too low: {min_cosine_sim}"
    assert mae < 2.0, f"Normalized mean absolute error too high: {mae}"

def test_extract_mfcc_delta_orders():
    """Test that both order=1 and order=2 work correctly."""
    np.random.seed(42)
    signal = np.random.randn(22050)
    sr = 22050

    # Test order=1
    result1 = cepstral_features.extract_mfcc_delta(signal, sr=sr, order=1)
    assert result1['mfcc_delta'].shape[0] == 13  # Default n_mfcc
    assert result1['mfcc_delta'].shape[1] > 0   # Some frames

    # Test order=2
    result2 = cepstral_features.extract_mfcc_delta(signal, sr=sr, order=2)
    assert result2['mfcc_delta'].shape[0] == 13  # Default n_mfcc
    assert result2['mfcc_delta'].shape[1] > 0   # Some frames

    # Shapes should match
    assert result1['mfcc_delta'].shape == result2['mfcc_delta'].shape

def test_extract_gfcc_basic():
    """Test GFCC extraction: shape, dtype, and no NaNs/Infs."""
    np.random.seed(42)
    signal = np.random.randn(22050)
    sr = 22050
    n_gfcc = 13
    result = cepstral_features.extract_gfcc(signal, sr=sr, n_gfcc=n_gfcc)
    gfcc = result['gfcc']
    assert isinstance(gfcc, np.ndarray)
    assert gfcc.shape[0] == n_gfcc
    assert gfcc.shape[1] > 0
    assert np.issubdtype(gfcc.dtype, np.floating)
    assert not np.isnan(gfcc).any(), "GFCC contains NaNs"
    assert not np.isinf(gfcc).any(), "GFCC contains Infs"

def test_extract_gfcc_metadata():
    """Test GFCC extraction with metadata (frame times)."""
    np.random.seed(42)
    signal = np.random.randn(22050)
    sr = 22050
    n_gfcc = 13
    result = cepstral_features.extract_gfcc(signal, sr=sr, n_gfcc=n_gfcc, return_metadata=True)
    gfcc = result['gfcc']
    times = result['metadata']['times']
    assert times.shape[0] == gfcc.shape[1]
    assert np.all(times[1:] > times[:-1]), "Frame times should be increasing"

def test_extract_gfcc_parameter_variation():
    """Test GFCC with different n_gfcc and n_gammatone values."""
    np.random.seed(0)
    signal = np.random.randn(16000)
    sr = 16000
    for n_gfcc, n_gammatone in [(5, 8), (20, 32), (40, 64)]:
        result = cepstral_features.extract_gfcc(signal, sr=sr, n_gfcc=n_gfcc, n_gammatone=n_gammatone)
        gfcc = result['gfcc']
        assert gfcc.shape[0] == n_gfcc
        assert gfcc.shape[1] > 0

def test_extract_gfcc_short_signal():
    """Test GFCC extraction on a very short signal (shorter than frame size)."""
    np.random.seed(1)
    signal = np.random.randn(100)
    sr = 8000
    result = cepstral_features.extract_gfcc(signal, sr=sr, n_gfcc=8)
    gfcc = result['gfcc']
    assert gfcc.shape[0] == 8
    assert gfcc.shape[1] > 0

def test_extract_gfcc_silence():
    """Test GFCC extraction on a silent signal (all zeros)."""
    signal = np.zeros(2048)
    sr = 16000
    result = cepstral_features.extract_gfcc(signal, sr=sr, n_gfcc=10)
    gfcc = result['gfcc']
    assert gfcc.shape[0] == 10
    assert gfcc.shape[1] > 0
    # All values should be finite (no NaN/Inf)
    assert np.all(np.isfinite(gfcc))

def test_extract_chroma_cqt():
    signal = np.random.randn(22050)
    result = cepstral_features.extract_chroma_cqt(signal, sr=22050)
    assert 'chroma_cqt' in result
    assert isinstance(result['chroma_cqt'], np.ndarray)

def test_extract_cqt():
    """Test basic functionality of extract_cqt."""
    signal = np.random.randn(22050)
    result = cepstral_features.extract_cqt(signal, sr=22050)
    assert 'cqt' in result
    assert isinstance(result['cqt'], np.ndarray)

    # Test with metadata
    result_with_meta = cepstral_features.extract_cqt(signal, sr=22050, return_metadata=True)
    assert 'cqt' in result_with_meta
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

def test_extract_chroma_stft():
    """Test basic functionality of extract_chroma_stft."""
    signal = np.random.randn(22050)
    result = cepstral_features.extract_chroma_stft(signal, sr=22050)
    assert 'chroma_stft' in result
    assert isinstance(result['chroma_stft'], np.ndarray)
    assert result['chroma_stft'].shape[0] == 12  # 12 chroma bins

def test_extract_chroma_stft_metadata():
    """Test that chroma_stft metadata is computed correctly."""
    signal = np.random.randn(22050) 
    sr = 22050
    frame_size = 2048
    hop_length = 512

    result = cepstral_features.extract_chroma_stft(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length, return_metadata=True
    )

    assert 'chroma_stft' in result
    assert 'metadata' in result
    assert 'times' in result['metadata']

    chroma = result['chroma_stft']
    times = result['metadata']['times']

    # Check shapes match
    assert chroma.shape[0] == 12
    assert times.shape[0] == chroma.shape[1]

    # Check time computation (manual calculation)
    n_frames = chroma.shape[1]
    frame_offset = frame_size // 2
    expected_times = (np.arange(n_frames) * hop_length + frame_offset) / sr

    np.testing.assert_array_almost_equal(times, expected_times)

def test_extract_chroma_stft_librosa_compatibility():
    """Test that our chroma STFT implementation produces reasonable results compared to librosa."""
    # Set seed for reproducible tests
    np.random.seed(42)

    # Create a tonal test signal (A4 note at 440 Hz)
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Pure A4 tone should activate the A chroma bin (index 9)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Our implementation
    result_ours = cepstral_features.extract_chroma_stft(
        signal, sr=sr, frame_size=2048, hop_length=512, return_metadata=True
    )
    chroma_ours = result_ours['chroma_stft']
    times_ours = result_ours['metadata']['times']

    # Librosa implementation for comparison
    chroma_librosa = librosa.feature.chroma_stft(
        y=signal, sr=sr, n_fft=2048, hop_length=512
    )
    times_librosa = librosa.frames_to_time(
        np.arange(chroma_librosa.shape[1]), sr=sr, hop_length=512, n_fft=2048
    )

    # Check shapes match
    assert chroma_ours.shape == chroma_librosa.shape
    assert times_ours.shape == times_librosa.shape

    # Check timing alignment
    time_diff = np.abs(times_ours - times_librosa)
    assert np.max(time_diff) < 1e-6, "Time alignment should be very precise"

    # For a pure A4 tone, both implementations should have strong energy in the A chroma bin (index 9)
    # We'll check that the A bin has the highest average energy
    avg_energy_ours = np.mean(chroma_ours, axis=1)
    avg_energy_librosa = np.mean(chroma_librosa, axis=1)

    # Both should peak at index 9 (A note)
    assert np.argmax(avg_energy_ours) == 9, "Our implementation should peak at A (index 9)"
    assert np.argmax(avg_energy_librosa) == 9, "Librosa should peak at A (index 9)"

    # The correlation between the two implementations should be reasonable
    # (they might differ due to implementation details, but should show similar patterns)
    correlation_by_bin = []
    for bin_idx in range(12):
        corr = np.corrcoef(chroma_ours[bin_idx], chroma_librosa[bin_idx])[0, 1]
        if not np.isnan(corr):  # Ignore bins with no energy
            correlation_by_bin.append(corr)

    if correlation_by_bin:  # Only check if we have valid correlations
        avg_correlation = np.mean(correlation_by_bin)
        assert avg_correlation > 0.5, f"Average correlation should be reasonable, got {avg_correlation}"

def test_extract_mfcc_delta_delta_basic():
    """Test basic functionality of extract_mfcc_delta_delta."""
    signal = np.random.randn(22050)
    result = cepstral_features.extract_mfcc_delta_delta(signal, sr=22050)
    assert 'mfcc_delta_delta' in result
    assert isinstance(result['mfcc_delta_delta'], np.ndarray)

def test_extract_mfcc_delta_delta_librosa_compatibility():
    """Test that our MFCC delta-delta implementation produces reasonable results compared to librosa."""
    # Set seed for reproducible tests
    np.random.seed(42)

    # Test signal
    signal = np.random.randn(22050) * 0.1
    sr = 22050
    n_mfcc = 13

    # Our implementation
    result_ours = cepstral_features.extract_mfcc_delta_delta(
        signal,
        sr=sr,
        n_mfcc=n_mfcc,
        center=True,
        return_metadata=True,
    )
    delta_delta_ours = result_ours['mfcc_delta_delta']
    times_ours = result_ours['metadata']['times']

    # Librosa implementation for comparison
    mfcc_librosa = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    n_frames = mfcc_librosa.shape[1]
    width = min(9, n_frames)
    if width < 3:
        delta_delta_librosa = np.zeros_like(mfcc_librosa)
    else:
        if width % 2 == 0:
            width -= 1
        delta_delta_librosa = librosa.feature.delta(mfcc_librosa, width=width, order=2)
    times_librosa = librosa.frames_to_time(
        np.arange(delta_delta_librosa.shape[1]), sr=sr, hop_length=512, n_fft=2048
    )

    # Check shapes match
    assert delta_delta_ours.shape == delta_delta_librosa.shape
    assert times_ours.shape == times_librosa.shape

    # Check timing alignment
    time_diff = np.abs(times_ours - times_librosa)
    assert np.max(time_diff) < 1e-6, "Time alignment should be very precise"

    # Check feature similarity (allowing for implementation differences)
    cosine_similarities = []
    for i in range(n_mfcc):
        cos_sim = np.dot(delta_delta_ours[i], delta_delta_librosa[i]) / (
            np.linalg.norm(delta_delta_ours[i]) * np.linalg.norm(delta_delta_librosa[i]) + 1e-10
        )
        cosine_similarities.append(cos_sim)

    avg_cosine_sim = np.mean(cosine_similarities)
    min_cosine_sim = np.min(cosine_similarities)

    # Mean Absolute Error (scaled by signal range for robustness)
    signal_range = np.max(delta_delta_librosa) - np.min(delta_delta_librosa)
    mae = np.mean(np.abs(delta_delta_ours - delta_delta_librosa)) / (signal_range + 1e-10)

    # Reasonable similarity thresholds (more lenient for delta-delta since it's second derivative)
    assert avg_cosine_sim > 0.3, f"Average cosine similarity too low: {avg_cosine_sim}"
    assert min_cosine_sim > -0.5, f"Minimum cosine similarity too low: {min_cosine_sim}"
    assert mae < 2.0, f"Normalized mean absolute error too high: {mae}"

def test_extract_melspectrogram():
    signal = np.random.randn(22050)
    result = cepstral_features.extract_melspectrogram(signal, sr=22050)
    assert 'melspectrogram' in result
    assert isinstance(result['melspectrogram'], np.ndarray)

def test_extract_melspectrogram_librosa_compatibility():
    """Test that our melspectrogram implementation produces results compatible with librosa."""
    # Set seed for reproducible tests
    np.random.seed(42)

    # Create test signal (sine wave + noise for better coverage)
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))

    # Test parameters
    n_mels = 128
    frame_size = 2048
    hop_length = 512

    # Our implementation
    result_ours = cepstral_features.extract_melspectrogram(
        signal, sr=sr, n_mels=n_mels, frame_size=frame_size, 
        hop_length=hop_length, return_metadata=True
    )
    mel_ours = result_ours['melspectrogram']
    times_ours = result_ours['metadata']['times']

    # Librosa implementation for comparison
    mel_librosa = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=n_mels, n_fft=frame_size, hop_length=hop_length
    )
    times_librosa = librosa.frames_to_time(
        np.arange(mel_librosa.shape[1]), sr=sr, hop_length=hop_length, n_fft=frame_size
    )

    # Check shapes
    assert mel_ours.shape == mel_librosa.shape, f"Shape mismatch: {mel_ours.shape} vs {mel_librosa.shape}"
    assert times_ours.shape == times_librosa.shape, f"Times shape mismatch: {times_ours.shape} vs {times_librosa.shape}"

    # Check that all values are non-negative
    assert np.all(mel_ours >= 0), "Found negative values in mel spectrogram"

    # Check timing alignment - should be very precise
    time_diff = np.abs(times_ours - times_librosa)
    assert np.max(time_diff) < 1e-6, f"Time alignment error: max diff = {np.max(time_diff)}"

    # Check correlation with librosa - should be very high
    correlation = np.corrcoef(mel_ours.flatten(), mel_librosa.flatten())[0, 1]
    assert correlation > 0.95, f"Correlation too low: {correlation:.6f}"

    # Check relative error
    mean_abs_diff = np.mean(np.abs(mel_ours - mel_librosa))
    rel_error = mean_abs_diff / np.mean(mel_librosa)
    assert rel_error < 0.1, f"Relative error too high: {rel_error:.4f}"

    # Test with custom frequency range
    result_custom = cepstral_features.extract_melspectrogram(
        signal, sr=sr, n_mels=64, frame_size=frame_size, hop_length=hop_length,
        fmin=80.0, fmax=8000.0
    )
    mel_custom = result_custom['melspectrogram']

    mel_librosa_custom = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=64, n_fft=frame_size, hop_length=hop_length,
        fmin=80.0, fmax=8000.0
    )

    assert mel_custom.shape == mel_librosa_custom.shape, "Custom range shape mismatch"
    correlation_custom = np.corrcoef(mel_custom.flatten(), mel_librosa_custom.flatten())[0,1]
    assert correlation_custom > 0.95, f"Custom range correlation too low: {correlation_custom:.6f}"


def test_edge_cases_short_signals():
    """Test cepstral features with very short signals."""
    sr = 22050

    # Very short signal (less than one frame)
    short_signal = np.random.randn(512)  # Less than default frame_size=2048

    # Should still work but return limited frames
    mfcc_result = cepstral_features.extract_mfcc(short_signal, sr=sr)
    assert 'mfcc' in mfcc_result
    assert mfcc_result['mfcc'].shape[0] == 13  # n_mfcc

    # Delta should handle short sequences gracefully
    delta_result = cepstral_features.extract_mfcc_delta(short_signal, sr=sr)
    assert 'mfcc_delta' in delta_result
    assert delta_result['mfcc_delta'].shape[0] == 13


def test_edge_cases_constant_signal():
    """Test cepstral features with constant (DC) signals."""
    sr = 22050
    duration = 1.0

    # Constant signal
    constant_signal = np.ones(int(sr * duration)) * 0.5

    # MFCC should handle constant signal (mostly energy in first coefficient)
    mfcc_result = cepstral_features.extract_mfcc(constant_signal, sr=sr)
    assert 'mfcc' in mfcc_result
    assert np.isfinite(mfcc_result['mfcc']).all()

    # Mel spectrogram should also work
    mel_result = cepstral_features.extract_melspectrogram(constant_signal, sr=sr)
    assert 'melspectrogram' in mel_result
    assert np.all(mel_result['melspectrogram'] >= 0)  # Should be non-negative


def test_edge_cases_silence():
    """Test cepstral features with silence (zero signal)."""
    sr = 22050
    duration = 1.0

    # Silent signal
    silence = np.zeros(int(sr * duration))

    # MFCC should handle silence without crashing
    mfcc_result = cepstral_features.extract_mfcc(silence, sr=sr)
    assert 'mfcc' in mfcc_result
    assert np.isfinite(mfcc_result['mfcc']).all()

    # Chroma should also handle silence
    chroma_result = cepstral_features.extract_chroma_stft(silence, sr=sr)
    assert 'chroma_stft' in chroma_result
    assert chroma_result['chroma_stft'].shape[0] == 12


def test_edge_cases_delta_padding():
    """Test delta computation with different frame counts and padding scenarios."""
    sr = 22050

    # Test with very few frames (less than minimum delta width)
    very_short = np.random.randn(1024)  # Should produce ~1 frame

    delta_result = cepstral_features.extract_mfcc_delta(very_short, sr=sr)
    assert 'mfcc_delta' in delta_result
    # With insufficient frames, delta should return zeros but with correct shape
    assert delta_result['mfcc_delta'].shape[0] == 13

    # Test delta-delta with minimal frames
    delta_delta_result = cepstral_features.extract_mfcc_delta_delta(very_short, sr=sr)
    assert 'mfcc_delta_delta' in delta_delta_result
    assert delta_delta_result['mfcc_delta_delta'].shape[0] == 13


def test_different_parameter_combinations():
    """Test extractors with various parameter combinations."""
    sr = 22050
    signal = np.random.randn(22050)  # 1 second

    # Test MFCC with different parameters (avoid triggering framewise processing)
    mfcc_large = cepstral_features.extract_mfcc(
        signal, sr=sr, n_mfcc=20, n_mels=256
    )
    assert mfcc_large['mfcc'].shape[0] == 20

    # Test melspectrogram with different mel count  
    mel_small = cepstral_features.extract_melspectrogram(
        signal, sr=sr, n_mels=64
    )
    assert mel_small['melspectrogram'].shape[0] == 64

    # Test CQT with different bins
    cqt_result = cepstral_features.extract_cqt(
        signal, sr=sr, n_bins=48, bins_per_octave=12
    )
    assert cqt_result['cqt'].shape[0] == 48
