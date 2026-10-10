"""
Known answers and input errors for the signal-processing building blocks (delta, mel scale,
autocorrelation) and the mel filterbank cache.
"""
import numpy as np
import pytest

import afx_audio as afx
from afx_audio.extractors.cepstral_features import extract_mfcc
from afx_audio.methods.autocorrelation import autocorrelation
from afx_audio.methods.delta import compute_delta, delta_kernel
from afx_audio.methods.mel import hz_to_mel, mel_frequencies, mel_to_hz

SR = 16000


def test_delta_of_a_ramp_is_its_slope_up_to_the_edges():
    t = np.arange(30, dtype=np.float64)
    feats = np.stack([2.0 * t, -0.5 * t + 3])
    np.testing.assert_allclose(compute_delta(feats, width=9), [[2.0] * 30, [-0.5] * 30], atol=1e-12)


def test_delta_delta_of_a_parabola_is_twice_the_curvature():
    t = np.arange(30, dtype=np.float64)
    np.testing.assert_allclose(compute_delta((3.0 * t**2)[None, :], width=5, order=2), 6.0, atol=1e-9)


def test_delta_kernel_interior_matches_compute_delta():
    feats = np.random.default_rng(0).standard_normal((1, 40))
    interior = np.correlate(feats[0], delta_kernel(9), mode='valid')
    np.testing.assert_allclose(compute_delta(feats, width=9)[0, 4:-4], interior, atol=1e-12)


@pytest.mark.parametrize(('width', 'message'), [(2, 'at least 3'), (4, 'odd')])
def test_delta_kernel_rejects_bad_width(width, message):
    with pytest.raises(ValueError, match=message):
        delta_kernel(width)


@pytest.mark.parametrize(('features', 'kwargs', 'message'), [
    (np.zeros(20), {}, r'2D \(n_features, n_frames\)'),
    (np.zeros((2, 20)), {'width': 4}, 'width must be odd and at least 3, got 4'),
    (np.zeros((2, 20)), {'width': 1}, 'width must be odd and at least 3, got 1'),
    (np.zeros((2, 2)), {}, 'need at least 3 frames, got 2'),
])
def test_delta_input_errors(features, kwargs, message):
    with pytest.raises(ValueError, match=message):
        compute_delta(features, **kwargs)


def test_delta_on_even_frame_count_uses_the_largest_odd_width():
    feats = np.random.default_rng(1).standard_normal((2, 6))
    np.testing.assert_allclose(compute_delta(feats, width=9), compute_delta(feats, width=5))


@pytest.mark.parametrize('htk', [False, True])
def test_mel_scale_round_trip(htk):
    hz = np.array([0.0, 100.0, 999.0, 1000.0, 1001.0, 4000.0, 8000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(hz, htk=htk), htk=htk), hz, rtol=1e-12, atol=1e-9)


def test_htk_mel_formula():
    # 2595 log10(1 + f / 700): 700 Hz is 2595 log10(2) mel
    assert hz_to_mel(700.0, htk=True) == pytest.approx(2595.0 * np.log10(2.0))
    assert mel_to_hz(2595.0 * np.log10(2.0), htk=True) == pytest.approx(700.0)


def test_mel_scales_match_librosa():
    librosa = pytest.importorskip('librosa')
    hz = np.linspace(0, 11025, 50)
    for htk in (False, True):
        np.testing.assert_allclose(hz_to_mel(hz, htk=htk), librosa.hz_to_mel(hz, htk=htk), rtol=1e-12)


def test_mel_frequencies_default_to_nyquist():
    freqs = mel_frequencies(10, sr=SR)
    assert freqs[0] == 0.0 and freqs[-1] == pytest.approx(SR / 2)
    np.testing.assert_allclose(freqs, mel_frequencies(10, fmax=SR / 2, sr=SR))


def test_autocorrelation_known_values():
    np.testing.assert_allclose(autocorrelation(np.array([1.0, 2.0, 3.0]), 5), [14.0, 8.0, 3.0])


def test_autocorrelation_rows_match_numpy():
    frames = np.random.default_rng(2).standard_normal((3, 64))
    expected = [np.correlate(f, f, 'full')[63:63 + 10] for f in frames]
    np.testing.assert_allclose(autocorrelation(frames, 10), expected, atol=1e-10)


def test_autocorrelation_of_empty_frames_keeps_the_leading_shape():
    assert autocorrelation(np.zeros((3, 0)), 4).shape == (3, 0)


def test_cache_stats_count_one_filterbank_per_parameter_set():
    signal = np.random.default_rng(3).standard_normal(SR // 4)
    afx.clear_feature_cache()
    assert afx.get_cache_stats() == {'mel_filterbanks': 0, 'total_items': 0}
    extract_mfcc(signal, SR, n_mels=40)
    extract_mfcc(signal, SR, n_mels=40, n_mfcc=13)
    assert afx.get_cache_stats() == {'mel_filterbanks': 1, 'total_items': 1}
    extract_mfcc(signal, SR, n_mels=64)
    assert afx.get_cache_stats()['mel_filterbanks'] == 2
    afx.clear_feature_cache()
    assert afx.get_cache_stats()['total_items'] == 0
