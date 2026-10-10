import numpy as np
import pytest

import AFX
from AFX.utils.config_loader import validate_config
from AFX.utils.framewise import framewise_extractor


@pytest.fixture
def peak_feature():
    @framewise_extractor
    def peak_amplitude(signal, sr=None, *, scale=1.0):
        return {'peak_amplitude': np.array([scale * np.max(np.abs(signal))])}

    AFX.register_feature('peak_amplitude', peak_amplitude, units='amplitude')
    yield peak_amplitude
    AFX.unregister_feature('peak_amplitude')


def test_registered_feature_works_everywhere(peak_feature):
    x = np.sin(2 * np.pi * 5 * np.arange(1600) / 1600)
    config = {'sample_rate': 1600, 'features': {'peak_amplitude': {'frame_size': 400, 'hop_length': 400, 'scale': 2.0}},
              'aggregation': 'summary'}
    validate_config(config)
    out = AFX.extract_all_features(x, 1600, config)
    assert out['peak_amplitude_mean'] == pytest.approx([2.0])
    info = AFX.feature_info('peak_amplitude')
    assert info['group'] == 'custom' and info['units'] == 'amplitude'
    assert {'frame_size', 'hop_length', 'scale'} <= set(info['parameters'])
    assert 'peak_amplitude' in AFX.list_features('custom')
    with pytest.raises(ValueError, match='Unknown parameter'):
        validate_config({'sample_rate': 1600, 'features': {'peak_amplitude': {'scal': 2.0}}})


def test_plain_function_without_return_metadata():
    def dc_offset(signal, sr):
        return {'dc_offset': np.array([np.mean(signal)])}

    AFX.register_feature('dc_offset', dc_offset)
    try:
        out = AFX.extract_all_features(np.full(100, 0.25), 8000, {'features': ['dc_offset']}, return_metadata=True)
        assert out['features']['dc_offset'] == pytest.approx([0.25])
    finally:
        AFX.unregister_feature('dc_offset')
    assert 'dc_offset' not in AFX.FEATURE_MAP


def test_cannot_shadow_builtin_features():
    with pytest.raises(ValueError, match='already exists'):
        AFX.register_feature('mfcc', lambda signal, sr: {})
    with pytest.raises(KeyError):
        AFX.unregister_feature('mfcc')


def mean_abs(signal, sr=None):
    """Module level, so worker processes can unpickle it."""
    return {'mean_abs': np.array([np.mean(np.abs(signal))])}


def test_custom_features_reach_worker_processes(tmp_path):
    import soundfile as sf

    from AFX.sklearn import AFXFeatureExtractor

    signals = [np.full(4000, v) for v in (0.1, 0.2, 0.3)]
    for i, s in enumerate(signals):
        sf.write(tmp_path / f'{i}.wav', s.astype(np.float32), 8000, subtype='FLOAT')
    AFX.register_feature('mean_abs', mean_abs)
    try:
        out = AFXFeatureExtractor(features=['mean_abs'], sample_rate=8000, n_jobs=2).fit_transform(signals)
        np.testing.assert_allclose(out[:, 0], [0.1, 0.2, 0.3])
        res = AFX.extract_features_from_folder(str(tmp_path), {'sample_rate': 8000, 'features': ['mean_abs']},
                                               str(tmp_path / 'out.json'), n_jobs=2, verbose=False)
        assert res['1.wav']['mean_abs'] == pytest.approx([0.2])
    finally:
        AFX.unregister_feature('mean_abs')


def test_kwargs_function_accepts_any_parameter():
    def scaled(signal, sr, **params):
        return {'scaled': np.array([params.get('k', 1) * np.max(signal)])}

    AFX.register_feature('scaled', scaled)
    try:
        out = AFX.extract_all_features(np.ones(10), 8000, {'features': {'scaled': {'k': 3}}})
        assert out['scaled'] == pytest.approx([3.0])
    finally:
        AFX.unregister_feature('scaled')


def test_unregister_restores_an_overwritten_builtin():
    original = AFX.FEATURE_MAP['zcr']
    AFX.register_feature('zcr', lambda signal, sr: {'zcr': np.array([42.0])}, overwrite=True)
    assert AFX.feature_info('zcr')['group'] == 'custom'
    AFX.unregister_feature('zcr')
    assert AFX.FEATURE_MAP['zcr'] is original and AFX.feature_info('zcr')['group'] == 'time'
