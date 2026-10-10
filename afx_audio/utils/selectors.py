"""
Feature selection and dimensionality reduction utilities for afx_audio.
"""

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_classif
except ImportError as e:
    raise ImportError('afx_audio.utils.selectors requires scikit-learn: pip install "afx-audio[sklearn]"') from e


def pca_reducer(
    features: dict[str, np.ndarray],
    n_components: int = 2,
    feature_keys: list[str] | None = None
    ) -> tuple[np.ndarray, PCA]:
    """
    Reduce dimensionality of selected features using PCA.

    Args:
        features: Dict of feature arrays. 1D arrays are treated as one value
            per frame; 2D arrays are (n_dims, n_frames).
        n_components: Number of PCA components. Must not exceed
            min(n_samples, n_features).
        feature_keys: List of feature keys to include (default: all)
    Returns:
        Tuple of (reduced_features, fitted PCA object)
    Raises:
        ValueError: If features have mismatched frame counts, or if
            n_components exceeds min(n_samples, n_features).
    """
    if feature_keys is None:
        feature_keys = list(features.keys())
    if not feature_keys:
        raise ValueError('No feature keys provided for PCA reduction.')

    frame_counts = {}
    for k in feature_keys:
        arr = features[k]
        n_frames = arr.shape[0] if arr.ndim == 1 else arr.shape[-1]
        frame_counts[k] = n_frames
    unique_counts = set(frame_counts.values())
    if len(unique_counts) > 1:
        detail = ', '.join(f'{k}: {v}' for k, v in sorted(frame_counts.items()))
        raise ValueError(
            f'All features must have the same number of frames for PCA; '
            f'got mismatched frame counts ({detail})'
        )

    X = np.concatenate(
        [features[k].reshape(-1, 1) if features[k].ndim == 1 else features[k].T
         for k in feature_keys],
        axis=1,
    )

    max_components = min(X.shape[0], X.shape[1])
    if not (1 <= n_components <= max_components):
        raise ValueError(
            f'n_components={n_components} is invalid: must be between 1 and '
            f'min(n_samples, n_features) = {max_components} '
            f'(X has shape {X.shape})'
        )

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


def mutual_info_selector(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    random_state: int | None = None
    ) -> list[int]:
    """
    Select top-k features based on mutual information with target.

    Args:
        X: Feature matrix (samples x features)
        y: Target labels
        k: Number of features to select
        random_state: Seed forwarded to mutual_info_classif for
            reproducible results.
    Returns:
        List of selected feature indices (highest MI first)
    """
    mi = mutual_info_classif(X, y, random_state=random_state)
    selected = np.argsort(mi)[-k:][::-1]
    return selected.tolist()


def correlation_selector(
    X: np.ndarray,
    threshold: float = 0.95
    ) -> list[int]:
    """
    Select features by removing those highly correlated with earlier ones.

    Args:
        X: Feature matrix (samples x features)
        threshold: Absolute correlation threshold for removal
    Returns:
        List of selected (kept) feature indices

    Notes:
        NaN correlations (e.g. from constant columns, which have zero
        variance) are treated as correlation 0 via np.nan_to_num, so
        constant columns are never flagged as correlated and are kept.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.atleast_2d(corr)
    # Constant columns yield NaN correlations; treat those as 0 (uncorrelated)
    corr = np.nan_to_num(corr, nan=0.0)
    # Upper-triangle mask (excluding diagonal): column j is removed if any
    # earlier column i < j correlates with it above the threshold.
    upper = np.triu(np.abs(corr) > threshold, k=1)
    to_remove = upper.any(axis=0)
    selected = np.flatnonzero(~to_remove)
    return selected.tolist()
