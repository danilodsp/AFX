"""
Per-feature normalization of constant features: centred, never divided by zero.
"""
import numpy as np

from afx_audio.utils.normalization import minmax_normalize, zscore_normalize


def test_constant_feature_is_centred_by_zscore():
    with np.errstate(all='raise'):
        out = zscore_normalize({'flat': np.full(5, 3.0)})
    np.testing.assert_array_equal(out['flat'], np.zeros(5))


def test_constant_feature_is_shifted_to_zero_by_minmax():
    with np.errstate(all='raise'):
        out = minmax_normalize({'flat': np.full(5, 3.0)})
    np.testing.assert_array_equal(out['flat'], np.zeros(5))


def test_zero_dataset_std_only_centres():
    out = zscore_normalize({'a': np.array([1.0, 2.0])}, mean={'a': 1.0}, std={'a': 0.0})
    np.testing.assert_array_equal(out['a'], [0.0, 1.0])


def test_equal_dataset_min_and_max_only_shift():
    out = minmax_normalize({'a': np.array([1.0, 2.0])}, min_={'a': 1.0}, max_={'a': 1.0})
    np.testing.assert_array_equal(out['a'], [0.0, 1.0])
