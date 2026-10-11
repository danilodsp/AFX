"""
Test the spectral flatness implementation against librosa.
"""
import librosa
import numpy as np

from afx_audio.extractors import frequency_domain


def test_extract_spectral_flatness_against_librosa():
    """Test that our NumPy implementation closely matches librosa's implementation."""
    sr = 22050
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_signal = np.sin(2 * np.pi * 440 * t)  # tonal, flatness near 0
    np.random.seed(42)
    noise_signal = np.random.randn(int(sr * duration))  # flatness near 1
    mixed_signal = sine_signal + 0.1 * noise_signal

    frame_size = 2048
    hop_length = 512

    for signal_name, signal in [('sine', sine_signal), ('noise', noise_signal), ('mixed', mixed_signal)]:
        result = frequency_domain.extract_spectral_flatness(
            signal, sr=sr, frame_size=frame_size, hop_length=hop_length
        )
        flatness_custom = result['spectral_flatness']
        flatness_librosa = librosa.feature.spectral_flatness(
            y=signal, n_fft=frame_size, hop_length=hop_length
        )[0]
        assert flatness_custom.shape == flatness_librosa.shape
        np.testing.assert_allclose(flatness_custom, flatness_librosa, rtol=1e-6, atol=1e-12,
                                   err_msg=f'{signal_name} signal')

    result_with_meta = frequency_domain.extract_spectral_flatness(
        sine_signal, sr=sr, frame_size=frame_size, hop_length=hop_length,
        return_metadata=True
    )
    assert 'metadata' in result_with_meta
    assert 'times' in result_with_meta['metadata']

    times = result_with_meta['metadata']['times']
    n_frames = len(times)
    expected_times = np.arange(n_frames) * hop_length / sr
    assert np.allclose(times, expected_times)
