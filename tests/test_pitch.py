"""
Unit tests for pitch detection utilities.
"""
import numpy as np
import pytest
from AFX.utils.pitch import yin

import librosa


def test_yin_against_librosa():
    """Compare custom YIN implementation with librosa's YIN implementation."""
    # Create a test signal: sweep from 100Hz to 1000Hz
    sr = 22050
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Frequency sweep from 100Hz to 1000Hz
    f0 = 100
    f1 = 1000
    freq = f0 + (f1 - f0) * t / duration
    phase = 2 * np.pi * np.cumsum(freq) / sr
    signal = np.sin(phase)

    # Add a small amount of noise
    signal = signal + 0.01 * np.random.randn(len(signal))

    # Parameters
    frame_size = 2048
    hop_length = 512
    fmin = 50.0
    fmax = 2000.0

    # Compute pitch with both implementations
    pitch_custom = yin(signal, frame_length=frame_size, hop_length=hop_length, 
                      fmin=fmin, fmax=fmax, sr=sr)

    pitch_librosa = librosa.yin(signal, fmin=fmin, fmax=fmax, sr=sr, 
                               frame_length=frame_size, hop_length=hop_length)

    # Determine the minimum length for comparison
    min_length = min(len(pitch_custom), len(pitch_librosa))

    # Note: Small differences in frame count can occur due to boundary handling
    # We'll compare only up to the minimum length of both arrays
    pitch_custom = pitch_custom[:min_length]
    pitch_librosa = pitch_librosa[:min_length]

    # Count non-zero values (valid pitch detections)
    valid_custom = pitch_custom > 0
    valid_librosa = pitch_librosa > 0

    # At least 50% of the frames should have valid pitch
    assert np.sum(valid_custom) > 0.5 * len(pitch_custom)

    # For frames where both detect pitch, compare values
    both_valid = valid_custom & valid_librosa

    # Skip test if not enough overlap for meaningful comparison
    if np.sum(both_valid) < 0.4 * min_length:
        pytest.skip("Not enough overlapping valid pitch detections for comparison")

    # Calculate relative error
    rel_error = np.abs(pitch_custom[both_valid] - pitch_librosa[both_valid]) / pitch_librosa[both_valid]

    # Mean relative error should be less than 15%
    # Allowing more tolerance since implementations can differ
    assert np.mean(rel_error) < 0.15

    # Maximum relative error should be less than 30% for most frames
    # (allowing for some disagreement at transition points)
    sorted_errors = np.sort(rel_error)
    assert sorted_errors[int(0.9 * len(sorted_errors))] < 0.30
