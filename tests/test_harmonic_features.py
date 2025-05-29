"""
Unit tests for harmonic_features extractors.
"""
import numpy as np
import pytest
from AFX.extractors import harmonic_features
from AFX.utils.pitch import yin

def test_extract_pitch():
    signal = np.random.randn(22050)
    result = harmonic_features.extract_pitch(signal, sr=22050)
    assert 'pitch' in result
    assert isinstance(result['pitch'], np.ndarray)

def test_extract_thd():
    signal = np.random.randn(22050)
    result = harmonic_features.extract_thd(signal, sr=22050)
    assert 'thd' in result
    assert result['thd'].shape == (1,)

def test_extract_hnr():
    signal = np.random.randn(22050)
    result = harmonic_features.extract_hnr(signal, sr=22050)
    assert 'hnr' in result
    assert isinstance(result['hnr'], np.ndarray)

def test_pitch_detection_accuracy():
    """Test the accuracy of the custom YIN implementation against a known signal."""
    # Create a sine wave with known frequency (440 Hz)
    sr = 22050
    duration = 1.0  # seconds
    frequency = 440.0  # Hz
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)

    # Add a small amount of noise to make it slightly realistic
    signal = signal + 0.01 * np.random.randn(len(signal))

    # Test our implementation
    frame_size = 2048
    hop_length = 512
    pitches = yin(signal, frame_length=frame_size, hop_length=hop_length, 
                  fmin=50.0, fmax=2000.0, sr=sr)

    # Get non-zero pitch values
    valid_pitches = pitches[pitches > 0]

    # Check if we have at least some valid pitch estimations
    assert len(valid_pitches) > 0

    # Check if the mean of valid pitches is close to the true frequency
    # Allow a 5% margin of error
    assert abs(np.mean(valid_pitches) - frequency) < 0.05 * frequency

def test_hnr_consistency():
    """Test that the custom HNR implementation produces expected results."""
    # Create a signal with known harmonic content: a sine wave with some noise
    sr = 22050
    duration = 1.0  # seconds
    frequency = 440.0  # Hz
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Pure sine wave (high HNR)
    pure_tone = np.sin(2 * np.pi * frequency * t)

    # Sine wave with noise (lower HNR)
    noisy_tone = pure_tone + 0.5 * np.random.randn(len(pure_tone))

    # Very noisy signal (very low HNR)
    very_noisy = pure_tone + 2.0 * np.random.randn(len(pure_tone))

    # Extract HNR for all signals
    frame_size = 2048
    hop_length = 512

    pure_hnr = harmonic_features.extract_hnr(
        pure_tone, sr=sr, frame_size=frame_size, hop_length=hop_length
    )['hnr']

    noisy_hnr = harmonic_features.extract_hnr(
        noisy_tone, sr=sr, frame_size=frame_size, hop_length=hop_length
    )['hnr']

    very_noisy_hnr = harmonic_features.extract_hnr(
        very_noisy, sr=sr, frame_size=frame_size, hop_length=hop_length
    )['hnr']

    # Check expected behavior: Pure tone should have higher HNR than noisy tone
    assert np.median(pure_hnr) > np.median(noisy_hnr)

    # Noisy tone should have higher HNR than very noisy tone
    assert np.median(noisy_hnr) > np.median(very_noisy_hnr)

    # Test a direct call to the function to validate the return metadata format
    # Create a shorter signal for direct function testing
    short_tone = pure_tone[:frame_size]

    # Directly call the function to bypass the decorator
    direct_result = harmonic_features.extract_hnr.__wrapped__(
        short_tone, sr=sr, frame_size=frame_size, 
        hop_length=hop_length, return_metadata=True
    )

    # Verify metadata structure and content for direct function call
    assert 'metadata' in direct_result
    assert 'times' in direct_result['metadata']

    # Check that times are calculated correctly
    times = direct_result['metadata']['times']
    expected_time = np.array([0.0])  # Only one frame for short_tone
    assert np.allclose(times, expected_time)
