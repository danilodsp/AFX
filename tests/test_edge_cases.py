"""Every feature on degenerate input, and rejection of invalid input."""
import warnings

import numpy as np
import pytest

import afx_audio

SR = 16000
FEATURES = sorted(afx_audio.FEATURE_MAP)
RNG = np.random.default_rng(0)
SIGNALS = {
    'one_sample': np.ones(1),
    'very_short': RNG.standard_normal(37),
    'silence': np.zeros(SR // 2),
    'dc': np.full(SR // 2, 0.25),
    'float32_noise': RNG.standard_normal(SR // 2).astype(np.float32),
}


@pytest.mark.parametrize('kind', sorted(SIGNALS))
@pytest.mark.parametrize('name', FEATURES)
def test_degenerate_input_returns_values_without_warnings(name, kind):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = afx_audio.FEATURE_MAP[name](SIGNALS[kind], SR, return_metadata=True)
    values = out[name]
    assert values.size > 0
    assert values.shape[-1] == len(out['metadata']['times'])
    assert not np.isinf(values).any()


def test_list_input_is_accepted():
    x = RNG.standard_normal(2000)
    a = afx_audio.FEATURE_MAP['mfcc'](list(x), SR)['mfcc']
    b = afx_audio.FEATURE_MAP['mfcc'](x, SR)['mfcc']
    np.testing.assert_array_equal(a, b)


def test_aggregated_silence_is_finite_where_defined():
    config = afx_audio.load_preset('speech')
    feats = afx_audio.extract_all_features(np.zeros(SR), SR, config)
    for key, value in feats.items():
        # HNR and formants are undefined on every frame of silence; everything else is a number
        if key.startswith(('hnr', 'formants')):
            continue
        assert np.isfinite(value).all(), key


@pytest.mark.parametrize('name', ['mfcc', 'zcr', 'pitch', 'hnr', 'spectral_centroid', 'variance'])
@pytest.mark.parametrize('bad, error, match', [
    (np.array([0.0, np.nan, 1.0]), ValueError, 'NaN'),
    (np.array([0.0, np.inf]), ValueError, 'NaN or infinite'),
    (np.zeros(0), ValueError, 'empty'),
    (np.zeros((2, 100)), ValueError, '1-D'),
    (np.zeros(100, dtype=np.int16), TypeError, 'floating point'),
    (np.zeros(100, dtype=complex), TypeError, 'real floating-point'),
])
def test_invalid_signal_rejected(name, bad, error, match):
    with pytest.raises(error, match=match):
        afx_audio.FEATURE_MAP[name](bad, SR)


@pytest.mark.parametrize('sr', [0, -16000, None, True])
def test_invalid_sample_rate_rejected(sr):
    config = {'features': {'zcr': {}}}
    with pytest.raises(ValueError, match='sample rate'):
        afx_audio.extract_all_features(np.zeros(1000), sr, config)


def test_nan_frames_are_skipped_by_aggregation():
    sr = 16000
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 200 * t)
    tone[: sr // 2] = 0.0  # first half silent: HNR is NaN there
    config = {'features': {'hnr': {'frame_size': 400, 'hop_length': 160}}, 'aggregation': 'summary'}
    feats = afx_audio.extract_all_features(tone, sr, config)
    assert np.isfinite(feats['hnr_mean']).all() and feats['hnr_mean'][0] > 20


@pytest.mark.parametrize('sr', [float('nan'), float('inf')])
def test_non_finite_sample_rate_rejected(sr):
    with pytest.raises(ValueError, match='sample rate'):
        afx_audio.extract_all_features(np.zeros(1000), sr, {'features': {'zcr': {}}})
