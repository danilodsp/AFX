"""
Feature normalization utilities (z-score, min-max, etc.).
"""

import numpy as np


def zscore_normalize(
    features: dict[str, np.ndarray],
    mean: dict[str, float] | None = None,
    std: dict[str, float] | None = None
    ) -> dict[str, np.ndarray]:
    """
    Apply z-score normalization to each feature (per feature key).
    If mean/std are not provided, compute from the data.
    Args:
        features: Dict of feature arrays
        mean: Optional dict of means per feature
        std: Optional dict of stds per feature
    Returns:
        Dict of normalized features
    """
    normed = {}
    for k, v in features.items():
        has_stats = mean is not None and std is not None and k in mean and k in std
        if v.size == 1 and not has_stats:
            # Skip normalization for scalars when stats must be computed
            # from the data itself (a scalar's own std is 0). With
            # precomputed dataset-level stats, scalars are normalized too.
            normed[k] = v
            continue
        m = mean[k] if mean and k in mean else np.mean(v)
        s = std[k] if std and k in std else np.std(v)
        if s == 0:
            normed[k] = v - m  # avoid division by zero
        else:
            normed[k] = (v - m) / s
    return normed

def minmax_normalize(
    features: dict[str, np.ndarray],
    min_: dict[str, float] | None = None,
    max_: dict[str, float] | None = None
    ) -> dict[str, np.ndarray]:
    """
    Apply min-max normalization to each feature (per feature key).
    If min/max are not provided, compute from the data.
    Args:
        features: Dict of feature arrays
        min_: Optional dict of min per feature
        max_: Optional dict of max per feature
    Returns:
        Dict of normalized features
    """
    normed = {}
    for k, v in features.items():
        has_stats = min_ is not None and max_ is not None and k in min_ and k in max_
        if v.size == 1 and not has_stats:
            # Skip normalization for scalars when stats must be computed
            # from the data itself. With precomputed dataset-level stats,
            # scalars are normalized too.
            normed[k] = v
            continue
        minv = min_[k] if min_ and k in min_ else np.min(v)
        maxv = max_[k] if max_ and k in max_ else np.max(v)
        if maxv == minv:
            normed[k] = v - minv  # avoid division by zero
        else:
            normed[k] = (v - minv) / (maxv - minv)
    return normed
