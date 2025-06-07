"""
Unit tests for aggregator and shape_utils utilities.
"""
import numpy as np
from AFX.utils import aggregator, shape_utils

def test_mean_pooling():
    features = {'a': np.ones((3, 10)), 'b': np.arange(10)}
    pooled = aggregator.mean_pooling(features)
    assert 'a' in pooled and 'b' in pooled
    assert pooled['a'].shape == (3, 1)
    assert pooled['b'].shape == (1,)

def test_statistical_summary():
    features = {'a': np.ones((2, 5)), 'b': np.arange(5)}
    summary = aggregator.statistical_summary(features)
    assert 'a_mean' in summary and 'a_std' in summary
    assert summary['a_mean'].shape == (2, 1)
    assert summary['a_std'].shape == (2, 1)

def test_normalize_feature_shape():
    arr = np.ones((2, 5))
    flat = shape_utils.normalize_feature_shape(arr, mode='flatten')
    assert flat.shape == (10,)
    kept = shape_utils.normalize_feature_shape(arr, mode='keepdims')
    assert kept.shape == (2, 5)
