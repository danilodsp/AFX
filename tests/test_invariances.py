"""Properties every implementation of these definitions must have, independent of any reference library."""
import numpy as np
import pytest

import AFX

SR = 16000
F = AFX.FEATURE_MAP
RNG = np.random.default_rng(11)
T = np.arange(SR) / SR
SIGNAL = 0.5 * np.sin(2 * np.pi * 220 * T) + 0.2 * np.sin(2 * np.pi * 1330 * T) + 0.05 * RNG.standard_normal(SR)

FRAMED = {'frame_size': 1024, 'hop_length': 256}
GAIN_INVARIANT = [
    'zcr', 'entropy', 'crest_factor', 'kurtosis', 'sample_entropy', 'mobility', 'complexity', 'energy_ratio',
    'autocorr_snr', 'autocorr_zero_crossings', 'autocorr_peak_width', 'autocorr_decay_rate',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness', 'spectral_entropy',
    'spectral_skewness', 'spectral_contrast', 'chroma_stft', 'chroma_cqt', 'pitch', 'hnr', 'thd',
]


@pytest.mark.parametrize('name', GAIN_INVARIANT)
def test_gain_invariance(name):
    params = dict(FRAMED) if name != 'sample_entropy' else {'frame_size': 512, 'hop_length': 512}
    a = F[name](SIGNAL, SR, **params)[name]
    b = F[name](0.1 * SIGNAL, SR, **params)[name]
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6 * max(1.0, np.nanmax(np.abs(a))), err_msg=name)


@pytest.mark.parametrize('name, power', [('rms_energy', 1), ('variance', 2), ('short_time_energy', 2),
                                         ('melspectrogram', 2), ('cqt', 1), ('autocorr_variance', 4)])
def test_power_law_scaling(name, power):
    params = FRAMED if name not in ('melspectrogram', 'cqt') else {}
    a = F[name](SIGNAL, SR, **params)[name]
    b = F[name](3.0 * SIGNAL, SR, **params)[name]
    np.testing.assert_allclose(b, 3.0 ** power * a, rtol=1e-6)


def test_mfcc_gain_moves_only_c0():
    a = F['mfcc'](SIGNAL, SR, n_mfcc=13, n_mels=40)['mfcc']
    b = F['mfcc'](0.1 * SIGNAL, SR, n_mfcc=13, n_mels=40)['mfcc']
    # -20 dB on every mel band shifts the orthonormal DCT's first coefficient by -20 * sqrt(n_mels)
    np.testing.assert_allclose(b[0] - a[0], -20.0 * np.sqrt(40), atol=1e-6)
    np.testing.assert_allclose(b[1:], a[1:], atol=1e-6)


@pytest.mark.parametrize('f0', [110.0, 220.0, 440.0, 880.0])
def test_pure_tone(f0):
    x = np.sin(2 * np.pi * f0 * T)
    pitch = F['pitch'](x, SR, frame_size=1024, hop_length=256)['pitch']
    np.testing.assert_allclose(pitch, f0, rtol=2e-3)
    centroid = F['spectral_centroid'](x, SR, frame_size=4096, hop_length=1024)['spectral_centroid']
    np.testing.assert_allclose(centroid, f0, rtol=0.02)
    assert np.all(F['spectral_flatness'](x, SR, frame_size=2048, hop_length=512)['spectral_flatness'] < 1e-3)
    assert F['hnr'](x, SR, frame_size=1024, hop_length=256)['hnr'].min() > 30
    # A2, A3, A4, A5 are all pitch class A (9)
    chroma = F['chroma_stft'](x, SR, frame_size=4096, hop_length=1024)['chroma_stft']
    assert np.all(np.argmax(chroma, axis=0) == 9)


def test_white_noise_flatness_and_zcr():
    x = np.random.default_rng(12).standard_normal(SR * 4)
    flatness = F['spectral_flatness'](x, SR, frame_size=2048, hop_length=512)['spectral_flatness']
    # For a white Gaussian periodogram the ratio tends to exp(-euler_gamma) ~ 0.56
    assert abs(np.median(flatness) - np.exp(-np.euler_gamma)) < 0.03
    zcr = F['zcr'](x, SR, frame_size=2048, hop_length=512)['zcr']
    assert abs(np.mean(zcr) - 0.5) < 0.01


@pytest.mark.parametrize('name', ['rms_energy', 'variance', 'kurtosis', 'entropy', 'crest_factor', 'zcr'])
def test_time_reversal_invariance_of_clip_statistics(name):
    params = {'frame_size': None, 'hop_length': None} if name == 'zcr' else {}
    a = F[name](SIGNAL, SR, **params)[name]
    b = F[name](SIGNAL[::-1].copy(), SR, **params)[name]
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_spectral_features_shift_with_resampled_content():
    # The same tone at twice the sample rate keeps its centroid and rolloff in Hz
    x1 = np.sin(2 * np.pi * 500 * np.arange(SR) / SR)
    x2 = np.sin(2 * np.pi * 500 * np.arange(2 * SR) / (2 * SR))
    for name in ['spectral_centroid', 'spectral_rolloff']:
        a = np.median(F[name](x1, SR, frame_size=2048, hop_length=512)[name])
        b = np.median(F[name](x2, 2 * SR, frame_size=4096, hop_length=1024)[name])
        assert abs(a - b) < 2 * SR / 2048, name
