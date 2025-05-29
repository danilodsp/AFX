from AFX.extractors.time_domain import (
    extract_kurtosis,
    extract_short_time_energy,
    extract_energy_ratio,
    extract_sample_entropy,
    extract_mobility
)
"""
Unit tests for time_domain feature extractors.
"""
import numpy as np
from AFX.extractors import time_domain

def test_extract_zero_crossing_rate():
    signal = np.array([0, 1, -1, 1, -1, 0], dtype=np.float32)
    result = time_domain.extract_zero_crossing_rate(signal, sr=22050)
    assert 'zcr' in result
    assert isinstance(result['zcr'], np.ndarray)

def test_extract_variance():
    signal = np.ones(100)
    result = time_domain.extract_variance(signal, sr=22050)
    assert 'variance' in result
    assert np.allclose(result['variance'], 0.0)

def test_extract_entropy():
    signal = np.random.randn(1000)
    result = time_domain.extract_entropy(signal, sr=22050)
    assert 'entropy' in result
    assert result['entropy'].shape == (1,)

def test_extract_crest_factor():
    signal = np.array([0, 1, -1, 1, -1, 0], dtype=np.float32)
    result = time_domain.extract_crest_factor(signal, sr=22050)
    assert 'crest_factor' in result
    assert result['crest_factor'].shape == (1,)

def test_extract_kurtosis():
    signal = np.random.randn(1000)
    result = extract_kurtosis(signal, sr=22050)
    assert 'kurtosis' in result
    assert result['kurtosis'].shape == (1,)

def test_extract_short_time_energy():
    signal = np.ones(100)
    result = extract_short_time_energy(signal, sr=22050)
    assert 'short_time_energy' in result
    assert result['short_time_energy'].shape == (1,)

def test_extract_energy_ratio():
    signal = np.concatenate([np.ones(50), np.zeros(50)])
    result = extract_energy_ratio(signal, sr=22050)
    assert 'energy_ratio' in result
    assert result['energy_ratio'].shape == (1,)

def test_extract_sample_entropy():
    signal = np.random.randn(100)
    result = extract_sample_entropy(signal, sr=22050)
    assert 'sample_entropy' in result
    assert result['sample_entropy'].shape == (1,)

def test_extract_mobility():
    signal = np.random.randn(100)
    result = extract_mobility(signal, sr=22050)
    assert 'mobility' in result
    assert result['mobility'].shape == (1,)

def test_extract_zero_crossing_rate_compare_with_librosa():
    """Test that our implementation closely matches librosa's implementation."""
    import librosa

    # Create a test signal with known zero-crossings
    np.random.seed(42)  # for reproducibility
    signal = np.random.randn(10000)
    frame_size = 2048
    hop_length = 512
    sr = 22050

    # We'll test using the same frames
    frames = librosa.util.frame(signal, frame_length=frame_size, hop_length=hop_length)
    n_frames = frames.shape[1]

    # Calculate ZCR using our implementation
    result = time_domain.extract_zero_crossing_rate(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length
    )
    zcr_custom = result['zcr']

    # Calculate ZCR using librosa manually on the same frames
    zcr_librosa = np.zeros(n_frames)
    for i in range(n_frames):
        frame = frames[:, i]
        changes = np.abs(np.diff(np.signbit(frame).astype(int)))
        zcr_librosa[i] = np.sum(changes) / (frame_size - 1)

    # Check that values match (with tolerance)
    assert np.allclose(zcr_librosa, zcr_custom, rtol=1e-5, atol=1e-5)

    # Check metadata return
    result_with_meta = time_domain.extract_zero_crossing_rate(
        signal, sr=sr, frame_size=frame_size, hop_length=hop_length,
        return_metadata=True
    )
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

    # Check that times computation is correct
    expected_times = np.arange(len(zcr_custom)) * hop_length / sr
    assert np.allclose(result_with_meta['metadata']['times'], expected_times)
