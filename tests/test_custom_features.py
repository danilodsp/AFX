import numpy as np
import pytest

import afx_audio
from afx_audio.utils.config_loader import validate_config
from afx_audio.utils.framewise import framewise_extractor


@pytest.fixture
def peak_feature():
    @framewise_extractor
    def peak_amplitude(signal, sr=None, *, scale=1.0):
        return {'peak_amplitude': np.array([scale * np.max(np.abs(signal))])}

    afx_audio.register_feature('peak_amplitude', peak_amplitude, units='amplitude')
    yield peak_amplitude
    afx_audio.unregister_feature('peak_amplitude')


def test_registered_feature_works_everywhere(peak_feature):
    x = np.sin(2 * np.pi * 5 * np.arange(1600) / 1600)
    config = {'sample_rate': 1600, 'features': {'peak_amplitude': {'frame_size': 400, 'hop_length': 400, 'scale': 2.0}},
              'aggregation': 'summary'}
    validate_config(config)
    out = afx_audio.extract_all_features(x, 1600, config)
    assert out['peak_amplitude_mean'] == pytest.approx([2.0])
    info = afx_audio.feature_info('peak_amplitude')
    assert info['group'] == 'custom' and info['units'] == 'amplitude'
    assert {'frame_size', 'hop_length', 'scale'} <= set(info['parameters'])
    assert 'peak_amplitude' in afx_audio.list_features('custom')
    with pytest.raises(ValueError, match='Unknown parameter'):
        validate_config({'sample_rate': 1600, 'features': {'peak_amplitude': {'scal': 2.0}}})


def test_plain_function_without_return_metadata():
    def dc_offset(signal, sr):
        return {'dc_offset': np.array([np.mean(signal)])}

    afx_audio.register_feature('dc_offset', dc_offset)
    try:
        out = afx_audio.extract_all_features(
            np.full(100, 0.25), 8000, {'features': ['dc_offset']}, return_metadata=True
        )
        assert out['features']['dc_offset'] == pytest.approx([0.25])
    finally:
        afx_audio.unregister_feature('dc_offset')
    assert 'dc_offset' not in afx_audio.FEATURE_MAP


def test_cannot_shadow_builtin_features():
    with pytest.raises(ValueError, match='already exists'):
        afx_audio.register_feature('mfcc', lambda signal, sr: {})
    with pytest.raises(KeyError):
        afx_audio.unregister_feature('mfcc')


def mean_abs(signal, sr=None):
    """Module level, so worker processes can unpickle it."""
    return {'mean_abs': np.array([np.mean(np.abs(signal))])}


@pytest.mark.slow
def test_custom_features_reach_worker_processes(tmp_path):
    import soundfile as sf

    from afx_audio.sklearn import AFXFeatureExtractor

    signals = [np.full(4000, v) for v in (0.1, 0.2, 0.3)]
    for i, s in enumerate(signals):
        sf.write(tmp_path / f'{i}.wav', s.astype(np.float32), 8000, subtype='FLOAT')
    afx_audio.register_feature('mean_abs', mean_abs)
    try:
        out = AFXFeatureExtractor(features=['mean_abs'], sample_rate=8000, n_jobs=2).fit_transform(signals)
        np.testing.assert_allclose(out[:, 0], [0.1, 0.2, 0.3])
        res = afx_audio.extract_features_from_folder(str(tmp_path), {'sample_rate': 8000, 'features': ['mean_abs']},
                                               str(tmp_path / 'out.json'), n_jobs=2, verbose=False)
        assert res['1.wav']['mean_abs'] == pytest.approx([0.2])
    finally:
        afx_audio.unregister_feature('mean_abs')


def _write_clips(folder, n=2):
    import soundfile as sf

    for i in range(n):
        sf.write(folder / f'{i}.wav', np.full(4000, 0.1 * (i + 1), dtype=np.float32), 8000, subtype='FLOAT')


def test_interactive_feature_rejected_before_workers_start(tmp_path, monkeypatch):
    import types

    def notebook_feature(signal, sr):
        return {'notebook_feature': np.array([1.0])}

    _write_clips(tmp_path)
    monkeypatch.setattr(notebook_feature, '__module__', '__main__')
    monkeypatch.setitem(__import__('sys').modules, '__main__', types.ModuleType('__main__'))
    afx_audio.register_feature('notebook_feature', notebook_feature)
    config = {'sample_rate': 8000, 'features': ['notebook_feature']}
    try:
        with pytest.raises(ValueError, match='defined interactively.*n_jobs=1'):
            afx_audio.extract_features_from_folder(str(tmp_path), config, str(tmp_path / 'out.json'),
                                                   n_jobs=2, verbose=False)
    finally:
        afx_audio.unregister_feature('notebook_feature')


def test_unused_unpicklable_registration_does_not_reach_workers(tmp_path):
    _write_clips(tmp_path)
    afx_audio.register_feature('unused', lambda signal, sr: {'unused': np.array([0.0])})
    try:
        res = afx_audio.extract_features_from_folder(str(tmp_path), {'sample_rate': 8000, 'features': ['zcr']},
                                                     str(tmp_path / 'out.json'), n_jobs=2, verbose=False)
        assert sorted(res) == ['0.wav', '1.wav']
    finally:
        afx_audio.unregister_feature('unused')


def test_restore_in_a_worker_repeats_registrations_once():
    from afx_audio.registry import custom_features, restore_custom_features

    afx_audio.register_feature('mean_abs', mean_abs, units='amplitude')
    shipped = custom_features()
    afx_audio.unregister_feature('mean_abs')
    try:
        restore_custom_features(shipped)
        assert afx_audio.feature_info('mean_abs')['units'] == 'amplitude'
        restore_custom_features(shipped)
        assert custom_features() == shipped
    finally:
        afx_audio.unregister_feature('mean_abs')


def test_kwargs_function_accepts_any_parameter():
    def scaled(signal, sr, **params):
        return {'scaled': np.array([params.get('k', 1) * np.max(signal)])}

    afx_audio.register_feature('scaled', scaled)
    try:
        out = afx_audio.extract_all_features(np.ones(10), 8000, {'features': {'scaled': {'k': 3}}})
        assert out['scaled'] == pytest.approx([3.0])
    finally:
        afx_audio.unregister_feature('scaled')


def test_unregister_restores_an_overwritten_builtin():
    original = afx_audio.FEATURE_MAP['zcr']
    afx_audio.register_feature('zcr', lambda signal, sr: {'zcr': np.array([42.0])}, overwrite=True)
    assert afx_audio.feature_info('zcr')['group'] == 'custom'
    afx_audio.unregister_feature('zcr')
    assert afx_audio.FEATURE_MAP['zcr'] is original and afx_audio.feature_info('zcr')['group'] == 'time'
