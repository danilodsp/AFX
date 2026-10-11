"""
Unit tests for afx_audio.utils.selectors.
"""
import numpy as np
import pytest

from afx_audio.utils.selectors import (
    correlation_selector,
    mutual_info_selector,
    pca_reducer,
)


def test_mutual_info_selector_deterministic_with_random_state():
    rng = np.random.RandomState(42)
    X = rng.randn(80, 6)
    y = (X[:, 0] + 0.1 * rng.randn(80) > 0).astype(int)
    sel1 = mutual_info_selector(X, y, k=3, random_state=0)
    sel2 = mutual_info_selector(X, y, k=3, random_state=0)
    assert sel1 == sel2
    assert len(sel1) == 3
    # Feature 0 drives the label, so it should be selected
    assert 0 in sel1


def test_pca_reducer_shape():
    rng = np.random.RandomState(0)
    features = {
        'a': rng.randn(50),          # 1D: one value per frame
        'b': rng.randn(3, 50),       # 2D: (n_dims, n_frames)
    }
    X_reduced, pca = pca_reducer(features, n_components=2)
    assert X_reduced.shape == (50, 2)
    assert pca.n_components_ == 2


def test_pca_reducer_frame_mismatch_raises():
    rng = np.random.RandomState(0)
    features = {
        'a': rng.randn(50),
        'b': rng.randn(3, 40),
    }
    with pytest.raises(ValueError, match='a|b'):
        pca_reducer(features, n_components=2)


def test_pca_reducer_too_many_components_raises():
    rng = np.random.RandomState(0)
    features = {'a': rng.randn(10), 'b': rng.randn(10)}
    with pytest.raises(ValueError, match='n_components'):
        pca_reducer(features, n_components=5)


def test_correlation_selector_drops_duplicate_keeps_uncorrelated():
    rng = np.random.RandomState(1)
    a = rng.randn(100)
    b = rng.randn(100)
    X = np.column_stack([a, a, b])  # col 1 duplicates col 0
    selected = correlation_selector(X, threshold=0.95)
    assert selected == [0, 2]


def test_correlation_selector_constant_column_no_crash():
    rng = np.random.RandomState(2)
    a = rng.randn(100)
    const = np.ones(100)
    X = np.column_stack([a, const, rng.randn(100)])
    selected = correlation_selector(X, threshold=0.95)
    # Constant column has NaN correlation -> treated as 0, so it is kept
    assert selected == [0, 1, 2]
