"""
Feature normalization utilities (z-score, min-max, etc.).
"""
from typing import Dict, Optional
import numpy as np

def zscore_normalize(
    features: Dict[str, np.ndarray],
    mean: Optional[Dict[str, float]] = None,
    std: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
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
        if v.size == 1:
            # Skip normalization for scalars
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
    features: Dict[str, np.ndarray],
    min_: Optional[Dict[str, float]] = None,
    max_: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
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
        if v.size == 1:
            # Skip normalization for scalars
            normed[k] = v
            continue
        minv = min_[k] if min_ and k in min_ else np.min(v)
        maxv = max_[k] if max_ and k in max_ else np.max(v)
        if maxv == minv:
            normed[k] = v - minv  # avoid division by zero
        else:
            normed[k] = (v - minv) / (maxv - minv)
    return normed
