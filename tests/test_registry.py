import numpy as np
import pytest

import AFX
from AFX.registry import GROUPS


def test_every_feature_is_described():
    assert sorted(AFX.list_features()) == sorted(AFX.FEATURE_MAP)
    for name in AFX.list_features():
        info = AFX.feature_info(name)
        assert info['group'] in GROUPS
        assert info['summary']
        assert info['units']


def test_groups_partition_features():
    grouped = [name for g in GROUPS for name in AFX.list_features(g)]
    assert sorted(grouped) == sorted(AFX.FEATURE_MAP)
    with pytest.raises(ValueError):
        AFX.list_features('nope')
    with pytest.raises(KeyError):
        AFX.feature_info('nope')


def test_rows_match_output_shape():
    sr = 16000
    signal = np.random.default_rng(0).standard_normal(sr)
    for name in AFX.list_features():
        info = AFX.feature_info(name)
        out = AFX.FEATURE_MAP[name](signal, sr)[name]
        if info['rows'] == '1':
            assert out.ndim == 1, name
        else:
            params = info['parameters']
            expected = eval(info['rows'], {}, dict(params))  # e.g. 'n_bands + 1'
            assert out.shape[0] == expected, name


def test_flatten_features_names_and_order():
    feats = {'a': np.array([1.0]), 'b': np.array([[1.0, 2.0], [3.0, 4.0]]), 'c': np.float64(5.0)}
    names, values = AFX.flatten_features(feats)
    assert names == ['a', 'b_0', 'b_1', 'b_2', 'b_3', 'c']
    np.testing.assert_array_equal(values, [1, 1, 2, 3, 4, 5])


def test_flatten_features_accepts_a_stacked_vector():
    names, values = AFX.flatten_features(np.array([1.0, 2.0]))
    assert names == ['stack_0', 'stack_1'] and values.tolist() == [1.0, 2.0]
