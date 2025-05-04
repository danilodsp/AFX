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

def aggregate_features(features: Dict[str, np.ndarray], method: str = 'mean') -> Dict[str, np.ndarray]:
    """
    Aggregate features using the specified method ('mean', 'std', 'summary').
    """
    if method == 'mean':
        return mean_pooling(features)
    elif method == 'std':
        return std_pooling(features)
    elif method == 'summary':
        return statistical_summary(features)
    else:
        raise ValueError(f'Unknown aggregation method: {method}')
