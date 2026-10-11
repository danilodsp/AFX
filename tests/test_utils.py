"""
Unit tests for the aggregator pooling helpers.
"""
import numpy as np

from afx_audio.utils import aggregator


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
