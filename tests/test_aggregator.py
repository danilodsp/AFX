"""
Unit tests for the new aggregate_features API (flatten and normalization).
"""
import numpy as np
import pytest

from afx_audio.utils.aggregator import aggregate_features


def test_aggregate_features_flatten():
    features = {'a': np.ones((2, 5)), 'b': np.arange(10).reshape(2, 5)}
    agg = aggregate_features(features, method='mean', flatten=True)
    assert all(v.ndim == 1 for v in agg.values())
    assert agg['a'].shape == (2,)
    assert agg['b'].shape == (2,)

def test_aggregate_features_zscore():
    # Create 2D features (channels x frames) to test mean pooling properly
    features = {'a': np.array([[1.0, 2.0, 3.0]]), 'b': np.array([[10.0, 20.0, 30.0]])}
    agg = aggregate_features(features, method='mean', flatten=True, normalize='zscore')
    print('agg:', agg)
    for v in agg.values():
        print('mean:', np.mean(v), 'std:', np.std(v), 'value:', v)
    # After mean pooling, each feature becomes a single value
    # Normalization is skipped for scalars, so the value should be the mean
    for k, v in agg.items():
        expected = np.mean(features[k])
        assert np.allclose(v, expected, atol=1e-7)

def test_aggregate_features_minmax():
    features = {'a': np.array([1.0, 2.0, 3.0]), 'b': np.array([10.0, 20.0, 30.0])}
    agg = aggregate_features(features, method='mean', flatten=True, normalize='minmax')
    # After mean pooling and flattening, each feature becomes a single value
    # Normalization is skipped for scalars, so the value should be the mean
    for k, v in agg.items():
        expected = np.mean(features[k])
        assert np.allclose(v, expected, atol=1e-7)

def test_aggregate_features_stack():
    features = {'a': np.ones((2, 5)), 'b': np.arange(10).reshape(2, 5)}
    stacked = aggregate_features(features, method='stack')
    # Should be a 1D vector of all features concatenated
    assert isinstance(stacked, np.ndarray)
    assert stacked.ndim == 1
    # Length should match sum of flattened features
    expected_len = sum(np.prod(v.shape) for v in features.values())
    assert stacked.shape[0] == expected_len

def test_aggregate_features_advanced_summary():
    """Test advanced statistical summary aggregation."""
    features = {'test': np.random.rand(3, 20)}
    agg = aggregate_features(features, method='advanced_summary', flatten=True)
    expected_keys = ['test_mean', 'test_std', 'test_skew', 'test_kurt']
    assert all(key in agg for key in expected_keys)
    assert all(v.shape == (3,) for v in agg.values())

def test_aggregate_features_skew():
    """Test skewness aggregation."""
    features = {'test': np.random.rand(3, 20)}
    agg = aggregate_features(features, method='skew', flatten=True)
    assert 'test' in agg
    assert agg['test'].shape == (3,)

def test_aggregate_features_kurtosis():
    """Test kurtosis aggregation."""
    features = {'test': np.random.rand(3, 20)}
    agg = aggregate_features(features, method='kurtosis', flatten=True)
    assert 'test' in agg
    assert agg['test'].shape == (3,)

def test_sliding_window_aggregation():
    """Test sliding window aggregation."""
    features = {'test': np.random.rand(3, 50)}  # 3 features x 50 time frames

    # Test with valid window size
    agg = aggregate_features(features, method='sliding_window',
                           window_size=10, hop_length=5, flatten=True)
    assert 'test' in agg

    # Test with window larger than signal (should fall back to global)
    agg_large = aggregate_features(features, method='sliding_window',
                                 window_size=100, hop_length=50, flatten=True)
    assert 'test' in agg_large
    assert agg_large['test'].shape == (3,)  # Should be global aggregation

def test_sliding_window_parameters():
    """Test that sliding window requires proper parameters."""
    features = {'test': np.random.rand(3, 20)}

    try:
        # Should raise error without window_size and hop_length
        aggregate_features(features, method='sliding_window')
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "window_size and hop_length must be specified" in str(e)

def test_mean_pooling_1d_array():
    """1-D arrays aggregate to one value."""
    features = {'x': np.arange(10)}
    agg = aggregate_features(features, method='mean', flatten=True)
    assert agg['x'].shape == (1,)
    assert np.allclose(agg['x'], np.mean(np.arange(10)))

def test_sliding_window_1d_array():
    """Sliding window should work on 1D arrays."""
    arr = np.arange(10)
    features = {'x': arr}
    agg = aggregate_features(features, method='sliding_window',
                             window_size=4, hop_length=2, flatten=True)
    expected_windows = (len(arr) - 4) // 2 + 1
    expected = [np.mean(arr[i*2:i*2+4]) for i in range(expected_windows)]
    assert agg['x'].shape == (expected_windows,)
    assert np.allclose(agg['x'], expected)

def test_scalar_normalization_skipped():
    """Test that normalization is skipped for scalar features (single value)."""
    from afx_audio.utils.normalization import minmax_normalize, zscore_normalize
    # Scalar features
    features = {'a': np.array([42.0]), 'b': np.array([0.5])}
    z_norm = zscore_normalize(features)
    minmax_norm = minmax_normalize(features)
    # Should be unchanged
    assert np.allclose(z_norm['a'], features['a'])
    assert np.allclose(z_norm['b'], features['b'])
    assert np.allclose(minmax_norm['a'], features['a'])
    assert np.allclose(minmax_norm['b'], features['b'])

def test_zscore_normalize_vector():
    """Test z-score normalization for a vector (not scalar)."""
    from afx_audio.utils.normalization import zscore_normalize
    arr = np.array([1.0, 2.0, 3.0])
    features = {'a': arr}
    normed = zscore_normalize(features)
    expected = (arr - np.mean(arr)) / np.std(arr)
    assert np.allclose(normed['a'], expected)

def test_minmax_normalize_vector():
    """Test min-max normalization for a vector (not scalar)."""
    from afx_audio.utils.normalization import minmax_normalize
    arr = np.array([1.0, 2.0, 3.0])
    features = {'a': arr}
    normed = minmax_normalize(features)
    expected = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    assert np.allclose(normed['a'], expected)


def test_sliding_window_reachable_from_config():
    """aggregation_params must be forwarded to aggregate_features; before,
    aggregation='sliding_window' in a config always raised."""
    from afx_audio.extract_all import extract_all_features
    sr = 22050
    signal = np.random.RandomState(0).randn(sr)
    config = {
        'sample_rate': sr,
        'features': {'zcr': {'frame_size': 2048, 'hop_length': 512},
                     'mfcc': {'n_mfcc': 5, 'frame_size': 2048, 'hop_length': 512}},
        'aggregation': 'sliding_window',
        'aggregation_params': {'window_size': 10, 'hop_length': 5,
                               'sliding_agg': 'max'},
        'flatten': False,
    }
    out = extract_all_features(signal, sr, config)
    n_frames = 1 + (sr - 2048) // 512
    n_windows = (n_frames - 10) // 5 + 1
    assert out['zcr'].shape == (n_windows,)
    assert out['mfcc'].shape == (5, n_windows)


def test_extract_all_features_rejects_stereo():
    from afx_audio.extract_all import extract_all_features
    stereo = np.zeros((1000, 2))
    with pytest.raises(ValueError, match='1-D'):
        extract_all_features(stereo, 22050, {'features': {'zcr': {}}})


def test_extract_segments():
    import afx_audio

    sr = 8000
    x = np.random.default_rng(0).standard_normal(sr * 3)
    config = {'features': {'rms_energy': {}, 'mfcc': {'n_mfcc': 5, 'frame_size': 512, 'hop_length': 256}},
              'aggregation': 'summary'}
    names, X, starts = afx_audio.extract_segments(x, sr, config, segment_length=1.0, hop=0.5)
    assert X.shape == (5, len(names)) and names[0] == 'rms_energy_mean'
    np.testing.assert_allclose(starts, [0, 0.5, 1.0, 1.5, 2.0])
    second = afx_audio.flatten_features(afx_audio.extract_all_features(x[sr // 2: sr // 2 + sr], sr, config))[1]
    np.testing.assert_allclose(X[1], second)
    _, X1, _ = afx_audio.extract_segments(x[:1000], sr, config, segment_length=1.0)
    assert X1.shape[0] == 1
    with pytest.raises(ValueError):
        afx_audio.extract_segments(x, sr, {**config, 'aggregation': 'stack'}, segment_length=1.0)
