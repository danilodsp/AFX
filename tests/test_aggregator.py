"""
Unit tests for the new aggregate_features API (flatten and normalization).
"""
import numpy as np
from AFX.utils.aggregator import aggregate_features

def test_aggregate_features_flatten():
    features = {'a': np.ones((2, 5)), 'b': np.arange(10).reshape(2, 5)}
    agg = aggregate_features(features, method='mean', flatten=True)
    assert all(v.ndim == 1 for v in agg.values())
    assert agg['a'].shape == (2,)
    assert agg['b'].shape == (2,)

def test_aggregate_features_zscore():
    features = {'a': np.array([1.0, 2.0, 3.0]), 'b': np.array([10.0, 20.0, 30.0])}
    agg = aggregate_features(features, method='mean', flatten=True, normalize='zscore')
    for v in agg.values():
        assert np.allclose(np.mean(v), 0, atol=1e-7)
        assert np.allclose(np.std(v), 0, atol=1e-7)  # mean of mean is always 0, std is 0

def test_aggregate_features_minmax():
    features = {'a': np.array([1.0, 2.0, 3.0]), 'b': np.array([10.0, 20.0, 30.0])}
    agg = aggregate_features(features, method='mean', flatten=True, normalize='minmax')
    for v in agg.values():
        assert np.all((v >= 0) & (v <= 1))

def test_aggregate_features_stack():
    features = {'a': np.ones((2, 5)), 'b': np.arange(10).reshape(2, 5)}
    stacked = aggregate_features(features, method='stack')
    # Should be a 1D vector of all features concatenated
    assert isinstance(stacked, np.ndarray)
    assert stacked.ndim == 1
    # Length should match sum of flattened features
    expected_len = sum(np.prod(v.shape) for v in features.values())
    assert stacked.shape[0] == expected_len
