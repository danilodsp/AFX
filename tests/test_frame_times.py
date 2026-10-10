"""
Frame-time metadata must follow one convention for every extractor: the
time of each frame's center in the original signal, which is what
librosa.frames_to_time returns (n_fft=None for centered frames, n_fft set
for non-centered ones). Time-domain, spectral and cepstral features
computed with the same framing therefore share one time axis.
"""
import librosa
import numpy as np
import pytest

from afx_audio.extractors import (
    cepstral_features,
    frequency_domain,
    harmonic_features,
    time_domain,
)
from afx_audio.utils.frame_utils import frame_times

SR = 22050
FRAME = 2048
HOP = 512


@pytest.fixture
def signal():
    t = np.arange(SR) / SR
    return np.sin(2 * np.pi * 220 * t) + 0.01 * np.random.RandomState(0).randn(SR)


def test_frame_times_matches_librosa_both_conventions():
    n = 10
    assert np.allclose(
        frame_times(n, SR, HOP, FRAME, center=True),
        librosa.frames_to_time(np.arange(n), sr=SR, hop_length=HOP),
    )
    assert np.allclose(
        frame_times(n, SR, HOP, FRAME, center=False),
        librosa.frames_to_time(np.arange(n), sr=SR, hop_length=HOP, n_fft=FRAME),
    )


def test_center_true_and_false_share_one_axis(signal):
    """Frame k of a non-centered STFT lines up with frame k+2 of the
    centered one when frame_size // 2 == 2 * hop, so their times must too."""
    a = frequency_domain.extract_spectral_centroid(
        signal, SR, frame_size=FRAME, hop_length=HOP, center=False,
        return_metadata=True,
    )['metadata']['times']
    b = frequency_domain.extract_spectral_centroid(
        signal, SR, frame_size=FRAME, hop_length=HOP, center=True,
        return_metadata=True,
    )['metadata']['times']
    assert np.allclose(a, b[2:2 + len(a)])


def test_all_extractors_agree_on_times(signal):
    """Extractors with the same framing report identical times (centered by default)."""
    params = dict(frame_size=FRAME, hop_length=HOP, return_metadata=True)
    times = {
        'zcr': time_domain.extract_zero_crossing_rate(signal, SR, **params),
        'rms_energy': time_domain.extract_rms_energy(signal, SR, **params),
        'spectral_centroid': frequency_domain.extract_spectral_centroid(
            signal, SR, **params),
        'mfcc': cepstral_features.extract_mfcc(signal, SR, **params),
        'pitch': harmonic_features.extract_pitch(signal, SR, **params),
        'hnr': harmonic_features.extract_hnr(signal, SR, **params),
        'melspectrogram': cepstral_features.extract_melspectrogram(
            signal, SR, **params),
        'chroma_stft': cepstral_features.extract_chroma_stft(
            signal, SR, **params),
    }
    ref = times['zcr']['metadata']['times']
    expected = librosa.frames_to_time(np.arange(len(ref)), sr=SR, hop_length=HOP)
    assert np.allclose(ref, expected)
    for name, out in times.items():
        assert np.allclose(out['metadata']['times'], ref), name


def test_flux_is_stamped_with_the_later_frame(signal):
    out = frequency_domain.extract_spectral_flux(
        signal, SR, frame_size=FRAME, hop_length=HOP, center=True,
        return_metadata=True,
    )
    n_flux = len(out['spectral_flux'])
    expected = librosa.frames_to_time(
        np.arange(1, n_flux + 1), sr=SR, hop_length=HOP
    )
    assert np.allclose(out['metadata']['times'], expected)


def test_whole_signal_mode_reports_signal_midpoint(signal):
    out = frequency_domain.extract_spectral_centroid(
        signal, SR, frame_size=None, hop_length=None, return_metadata=True
    )
    assert np.allclose(out['metadata']['times'], [len(signal) / (2 * SR)])


def test_short_signal_framewise_is_not_zero_padded():
    """An uncentered clip shorter than one frame is analyzed as-is (one frame)."""
    rng = np.random.RandomState(0)
    clip = rng.randn(100)
    framed = time_domain.extract_variance(
        clip, SR, frame_size=2048, hop_length=512, center=False, return_metadata=True
    )
    whole = time_domain.extract_variance(clip, SR)
    assert framed['variance'].shape == (1,)
    assert np.allclose(framed['variance'], whole['variance'])
    assert np.allclose(framed['metadata']['times'], [50 / SR])
