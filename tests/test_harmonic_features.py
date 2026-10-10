"""
Unit tests for harmonic_features extractors.
"""
import numpy as np

from afx_audio.extractors import harmonic_features
from afx_audio.utils.pitch import yin


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

    # Add a small amount of noise to make it slightly realistic (seeded)
    np.random.seed(42)
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

    # Sine wave with noise (lower HNR); seeded for determinism
    np.random.seed(42)
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

    # extract_hnr frames internally (no decorator), so call it directly
    direct_result = harmonic_features.extract_hnr(
        short_tone, sr=sr, frame_size=frame_size,
        hop_length=hop_length, center=False, return_metadata=True
    )

    # Verify metadata structure and content for direct function call
    assert 'metadata' in direct_result
    assert 'times' in direct_result['metadata']

    # One uncentered frame, stamped at the middle of the signal
    times = direct_result['metadata']['times']
    expected_time = np.array([frame_size // 2 / sr])
    assert np.allclose(times, expected_time)


def test_hnr_is_not_capped_by_frame_length():
    """The biased autocorrelation estimate caps r_max at (N - lag)/N, which
    limited HNR of a pure tone to ~13 dB at 2048/22050 and ~2 dB for the
    speech preset framing (400 samples, 100 Hz). The unbiased estimate
    must report a pure tone well above any realistic noisy-speech value."""
    sr = 16000
    t = np.arange(2 * sr) / sr
    pure = np.sin(2 * np.pi * 100 * t)
    hnr = harmonic_features.extract_hnr(
        pure, sr=sr, frame_size=400, hop_length=160
    )['hnr']
    assert np.median(hnr) > 40.0


def test_hnr_tracks_snr():
    """HNR of tone + white noise should sit within a few dB of the SNR."""
    sr = 16000
    rng = np.random.RandomState(1)
    t = np.arange(2 * sr) / sr
    tone = np.sin(2 * np.pi * 100 * t)
    for snr_db in (20.0, 10.0):
        noise = rng.randn(len(t)) * np.sqrt(0.5 / 10 ** (snr_db / 10))
        hnr = harmonic_features.extract_hnr(
            tone + noise, sr=sr, frame_size=400, hop_length=160
        )['hnr']
        assert abs(np.median(hnr) - snr_db) < 3.0, (snr_db, np.median(hnr))
