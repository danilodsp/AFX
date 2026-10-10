"""
Unit tests for pitch detection utilities.
"""
import librosa
import numpy as np

from afx_audio.utils.pitch import yin


def test_yin_against_librosa():
    """Compare custom YIN implementation with librosa's YIN implementation."""
    sr = 22050
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    f0 = 100
    f1 = 1000
    freq = f0 + (f1 - f0) * t / duration
    phase = 2 * np.pi * np.cumsum(freq) / sr
    signal = np.sin(phase)

    # Add a small amount of noise (seeded for determinism)
    np.random.seed(42)
    signal = signal + 0.01 * np.random.randn(len(signal))

    frame_size = 2048
    hop_length = 512
    fmin = 50.0
    fmax = 2000.0

    pitch_custom = yin(signal, frame_length=frame_size, hop_length=hop_length,
                       fmin=fmin, fmax=fmax, sr=sr)
    # utils.pitch.yin frames without padding
    pitch_librosa = librosa.yin(signal, fmin=fmin, fmax=fmax, sr=sr,
                                frame_length=frame_size, hop_length=hop_length, center=False)
    assert pitch_custom.shape == pitch_librosa.shape
    assert np.all(pitch_custom > 0)

    # librosa 0.10 is up to 8% off on this sweep, librosa 1.0 within 0.3%
    old = tuple(int(p) for p in librosa.__version__.split('.')[:2]) < (0, 11)
    np.testing.assert_allclose(pitch_custom, pitch_librosa, rtol=0.1 if old else 0.01)
