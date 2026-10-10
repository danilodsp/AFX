"""Feature parameter checking and the uniform framing contract."""
import inspect

import numpy as np
import pytest

import afx_audio
from afx_audio.extractors import harmonic_features, time_domain
from afx_audio.utils.config_loader import check_feature_params, feature_parameters, validate_config

SR = 16000


def _config(features):
    return {'sample_rate': SR, 'features': features}


def test_unknown_parameter_rejected_at_validation():
    with pytest.raises(ValueError, match="Unknown parameter 'frame_sise' for feature 'zcr'. Did you mean 'frame_size'"):
        validate_config(_config({'zcr': {'frame_sise': 400}}))


@pytest.mark.parametrize('config, match', [
    ({'features': ['mfcc'], 'agregation': 'summary'}, "Unknown config key 'agregation'. Did you mean 'aggregation'"),
    ({'features': ['mfccs']}, "Unknown feature 'mfccs'. Did you mean 'mfcc'"),
    ({'features': 'mfcc'}, "'features' block"),
    ({'features': ['mfcc'], 'aggregation': 'summery'}, "Did you mean 'summary'"),
    ({'features': ['mfcc'], 'aggregation': 'sliding_window'}, 'window_size and hop_length'),
    ({'features': ['mfcc'], 'aggregation_params': {'window': 3}}, "Unknown aggregation_params key 'window'"),
])
def test_config_mistakes_are_explained(config, match):
    with pytest.raises(ValueError, match=match):
        validate_config(config, require_sample_rate=False)


def test_signal_type_mistakes_are_explained():
    with pytest.raises(TypeError, match='load_audio'):
        afx_audio.extract_all_features('clip.wav', SR, {'features': ['zcr']})
    with pytest.raises(TypeError, match='extract_features_from_folder'):
        afx_audio.extract_all_features(['a.wav', 'b.wav'], SR, {'features': ['zcr']})
    with pytest.raises(ValueError, match='got a scalar'):
        afx_audio.extract_all_features(0.5, SR, {'features': ['zcr']})


def test_sample_rate_mismatch_warns():
    with pytest.warns(UserWarning, match="differs from the config's sample_rate"):
        afx_audio.extract_all_features(np.zeros(4000), 44100, {'sample_rate': SR, 'features': ['zcr']})


def test_unknown_parameter_rejected_at_extraction():
    signal = np.random.default_rng(0).standard_normal(SR)
    with pytest.raises(ValueError, match='n_mfc'):
        afx_audio.extract_all_features(signal, SR, _config({'mfcc': {'n_mfc': 20}}))


@pytest.mark.parametrize('value', [0, -1, 2.5, True, '400'])
def test_bad_frame_size_rejected(value):
    with pytest.raises(ValueError, match='positive integer'):
        check_feature_params('mfcc', {'frame_size': value})


def test_bad_sample_rate_rejected():
    with pytest.raises(ValueError, match='positive'):
        validate_config({'sample_rate': 0, 'features': {}})


def test_every_feature_reports_its_parameters():
    for name, fn in afx_audio.FEATURE_MAP.items():
        params = feature_parameters(name)
        assert 'signal' not in params and 'return_metadata' not in params
        # The listed parameters are exactly what the extractor accepts
        sig = inspect.signature(fn)
        for key in params:
            assert key in sig.parameters, (name, key)


def test_framewise_wrapper_exposes_framing_parameters():
    params = feature_parameters('sample_entropy')
    assert {'frame_size', 'hop_length', 'm', 'r', 'max_samples'} <= set(params)


@pytest.mark.parametrize('name', ['variance', 'rms_energy', 'kurtosis', 'thd', 'autocorr_snr'])
def test_whole_clip_and_framed_shapes(name):
    signal = np.random.default_rng(1).standard_normal(SR)
    fn = afx_audio.FEATURE_MAP[name]
    whole = fn(signal, SR, return_metadata=True)
    assert whole[name].shape == (1,)
    assert whole['metadata']['times'] == pytest.approx([0.5])
    framed = fn(signal, SR, frame_size=400, hop_length=160, return_metadata=True)
    # Centered by default: SR // 160 hops plus the frame at sample 0
    n_frames = 1 + SR // 160
    assert framed[name].shape == (n_frames,)
    assert framed['metadata']['times'][0] == 0.0


def test_framed_matches_per_frame_whole_clip():
    signal = np.random.default_rng(2).standard_normal(4000)
    for name in ['variance', 'entropy', 'crest_factor', 'kurtosis', 'mobility', 'autocorr_decay_rate', 'thd']:
        fn = afx_audio.FEATURE_MAP[name]
        framed = fn(signal, SR, frame_size=512, hop_length=256, center=False)[name]
        expected = [fn(signal[i:i + 512], SR)[name][0] for i in range(0, 4000 - 512 + 1, 256)]
        np.testing.assert_allclose(framed, expected, rtol=1e-10, err_msg=name)


def test_entropy_matches_numpy_histogram():
    rng = np.random.default_rng(3)
    for signal in [rng.standard_normal(1000), rng.integers(-3, 3, 1000).astype(float), np.full(50, 0.25)]:
        hist, _ = np.histogram(signal, bins=37)
        p = hist[hist > 0] / hist.sum()
        expected = -np.sum(p * np.log2(p))
        got = time_domain.extract_entropy(signal, num_bins=37)['entropy'][0]
        assert got == pytest.approx(expected, abs=1e-12)


def test_kurtosis_matches_scipy():
    from scipy.stats import kurtosis

    signal = np.random.default_rng(4).standard_normal(3000) ** 3
    for fisher in (True, False):
        got = time_domain.extract_kurtosis(signal, fisher=fisher)['kurtosis'][0]
        assert got == pytest.approx(kurtosis(signal, fisher=fisher), rel=1e-12)
    assert np.isnan(time_domain.extract_kurtosis(np.ones(100))['kurtosis'][0])


def test_thd_of_clipped_sine():
    sr = 8000
    t = np.arange(sr) / sr
    clean = np.sin(2 * np.pi * 100 * t)
    clipped = np.clip(clean, -0.5, 0.5)
    assert harmonic_features.extract_thd(clean, sr)['thd'][0] < 1e-6
    assert harmonic_features.extract_thd(clipped, sr)['thd'][0] > 0.1


def _sample_entropy_reference(x, m, r):
    # Direct transcription of Richman and Moorman (2000): N - m templates for both lengths
    n = len(x)
    tol = r * np.std(x)

    def count(length):
        t = np.array([x[i:i + length] for i in range(n - m)])
        d = np.max(np.abs(t[:, None, :] - t[None, :, :]), axis=2)
        return np.sum(d <= tol) - len(t)

    return -np.log(count(m + 1) / count(m))


@pytest.mark.parametrize('m, r', [(2, 0.2), (3, 0.15), (1, 0.3)])
def test_sample_entropy_matches_definition(m, r):
    x = np.random.default_rng(5).standard_normal(300)
    got = time_domain.extract_sample_entropy(x, m=m, r=r)['sample_entropy'][0]
    assert got == pytest.approx(_sample_entropy_reference(x, m, r), rel=1e-12)


def test_framewise_arguments_are_keyword_only():
    x = np.random.default_rng(0).standard_normal(500)
    with pytest.raises(TypeError):
        time_domain.extract_sample_entropy(x, SR, 3)


@pytest.mark.parametrize('name', ['rms_energy', 'kurtosis', 'thd', 'sample_entropy', 'autocorr_snr'])
def test_centered_time_domain_frames_align_with_centered_stft(name):
    x = np.random.default_rng(6).standard_normal(4000)
    out = afx_audio.FEATURE_MAP[name](x, SR, frame_size=512, hop_length=128, center=True, return_metadata=True)
    ref = afx_audio.FEATURE_MAP['spectral_centroid'](
        x, SR, frame_size=512, hop_length=128, center=True, return_metadata=True
    )
    assert out[name].shape == ref['spectral_centroid'].shape
    np.testing.assert_allclose(out['metadata']['times'], ref['metadata']['times'])


def test_features_list_shorthand():
    x = np.random.default_rng(7).standard_normal(SR)
    short = afx_audio.extract_all_features(x, SR, {'features': ['zcr', 'mfcc']})
    full = afx_audio.extract_all_features(x, SR, {'features': {'zcr': {}, 'mfcc': {}}})
    assert list(short) == ['zcr', 'mfcc']
    for key in full:
        np.testing.assert_array_equal(short[key], full[key])
    config = {'sample_rate': SR, 'features': ['zcr']}
    assert validate_config(config)['features'] == {'zcr': {}}
    assert config['features'] == ['zcr']  # caller's dict untouched
    with pytest.raises(ValueError):
        validate_config({'sample_rate': SR, 'features': ['zcr', 3]})
