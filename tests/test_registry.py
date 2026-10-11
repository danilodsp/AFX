import numpy as np
import pytest

import afx_audio
from afx_audio.registry import GROUPS


def test_every_feature_is_described():
    assert sorted(afx_audio.list_features()) == sorted(afx_audio.FEATURE_MAP)
    for name in afx_audio.list_features():
        info = afx_audio.feature_info(name)
        assert info['group'] in GROUPS
        assert info['summary']
        assert info['units']


def test_feature_info_torch_entry():
    assert afx_audio.feature_info('mfcc')['torch'] == {
        'function': 'afx_audio.torch.functional.mfcc', 'rtol': 1e-9, 'atol': 1e-9, 'dtype': 'float64'}
    assert afx_audio.feature_info('zcr')['torch']['function'] == 'afx_audio.torch.functional.zero_crossing_rate'
    assert afx_audio.feature_info('hnr')['torch'] is None


def test_overridden_feature_has_no_torch_entry():
    def mfcc(signal, sr):
        return {'mfcc': np.zeros(1)}

    afx_audio.register_feature('mfcc', mfcc, overwrite=True)
    try:
        assert afx_audio.feature_info('mfcc')['torch'] is None
    finally:
        afx_audio.unregister_feature('mfcc')
    assert afx_audio.feature_info('mfcc')['torch'] is not None


def test_groups_partition_features():
    grouped = [name for g in GROUPS for name in afx_audio.list_features(g)]
    assert sorted(grouped) == sorted(afx_audio.FEATURE_MAP)
    with pytest.raises(ValueError):
        afx_audio.list_features('nope')
    with pytest.raises(KeyError):
        afx_audio.feature_info('nope')


def test_rows_match_output_shape():
    sr = 16000
    signal = np.random.default_rng(0).standard_normal(sr)
    for name in afx_audio.list_features():
        info = afx_audio.feature_info(name)
        out = afx_audio.FEATURE_MAP[name](signal, sr)[name]
        if info['rows'] == '1':
            assert out.ndim == 1, name
        else:
            params = info['parameters']
            expected = eval(info['rows'], {}, dict(params))  # e.g. 'n_bands + 1'
            assert out.shape[0] == expected, name


def test_flatten_features_names_and_order():
    feats = {'a': np.array([1.0]), 'b': np.array([[1.0, 2.0], [3.0, 4.0]]), 'c': np.float64(5.0)}
    names, values = afx_audio.flatten_features(feats)
    assert names == ['a', 'b_0', 'b_1', 'b_2', 'b_3', 'c']
    np.testing.assert_array_equal(values, [1, 1, 2, 3, 4, 5])


def test_flatten_features_accepts_a_stacked_vector():
    names, values = afx_audio.flatten_features(np.array([1.0, 2.0]))
    assert names == ['stack_0', 'stack_1'] and values.tolist() == [1.0, 2.0]
