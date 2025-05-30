"""
Feature aggregation utilities (mean, std, pooling strategies).
"""
from typing import Dict, Callable
import numpy as np

def mean_pooling(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Aggregate features by computing the mean along the last axis.
    """
    return {k: np.mean(v, axis=-1, keepdims=True) if v.ndim > 1 else v for k, v in features.items()}

def std_pooling(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Aggregate features by computing the standard deviation along the last axis.
    """
    return {k: np.std(v, axis=-1, keepdims=True) if v.ndim > 1 else v for k, v in features.items()}

def statistical_summary(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Return both mean and std for each feature.
    """
    summary = {}
    for k, v in features.items():
        if v.ndim > 1:
            summary[f'{k}_mean'] = np.mean(v, axis=-1, keepdims=True)
            summary[f'{k}_std'] = np.std(v, axis=-1, keepdims=True)
        else:
            summary[f'{k}_mean'] = v
            summary[f'{k}_std'] = np.zeros_like(v)
    return summary

def aggregate_features(
    features: Dict[str, np.ndarray],
    method: str = 'mean',
    flatten: bool = True,
    normalize: str = None  # 'zscore', 'minmax', or None
) -> Dict[str, np.ndarray]:
    """
    Aggregate features using the specified method ('mean', 'std', 'summary', 'stack'),
    then optionally flatten and normalize each feature.
    Args:
        features: Dict of feature arrays
        method: Aggregation method ('mean', 'std', 'summary', 'stack')
        flatten: Whether to flatten each feature to 1D (ignored if method='stack')
        normalize: Normalization method ('zscore', 'minmax', or None)
    Returns:
        Dict of aggregated (and optionally flattened/normalized) features, or a stacked numpy array if method='stack'.
    """
    if method == 'mean':
        agg = mean_pooling(features)
    elif method == 'std':
        agg = std_pooling(features)
    elif method == 'summary':
        agg = statistical_summary(features)
    elif method == 'stack':
        from AFX.utils import stack_feature_vectors
        # For single sample, wrap in list
        return stack_feature_vectors([features], flatten=True)[0]
    else:
        raise ValueError(f'Unknown aggregation method: {method}')

    # Flatten each feature if requested
    if flatten:
        agg = {k: v.flatten() for k, v in agg.items()}

    # Normalize if requested
    if normalize is not None:
        if normalize == 'zscore':
            from .normalization import zscore_normalize
            agg = zscore_normalize(agg)
        elif normalize == 'minmax':
            from .normalization import minmax_normalize
            agg = minmax_normalize(agg)
        else:
            raise ValueError(f'Unknown normalization method: {normalize}')
    return agg
