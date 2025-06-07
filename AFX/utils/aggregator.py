"""
Feature aggregation utilities (mean, std, pooling strategies).
"""
from typing import Dict, Callable, Union
import numpy as np
from scipy.stats import skew, kurtosis

from AFX.utils import stack_feature_vectors, zscore_normalize, minmax_normalize


def mean_pooling(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Aggregate features by computing the mean along the last axis."""
    result = {}
    for k, v in features.items():
        if v.ndim == 0:
            result[k] = v
        else:
            result[k] = np.mean(v, axis=-1, keepdims=True)
    return result

def std_pooling(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Aggregate features by computing the standard deviation along the last axis."""
    result = {}
    for k, v in features.items():
        if v.ndim == 0:
            result[k] = v
        else:
            result[k] = np.std(v, axis=-1, keepdims=True)
    return result

def statistical_summary(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return both mean and standard deviation for each feature."""
    summary = {}
    for k, v in features.items():
        if v.ndim == 0:
            summary[f'{k}_mean'] = v
            summary[f'{k}_std'] = np.zeros_like(v)
        else:
            summary[f'{k}_mean'] = np.mean(v, axis=-1, keepdims=True)
            summary[f'{k}_std'] = np.std(v, axis=-1, keepdims=True)
    return summary

def skew_pooling(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Aggregate features by computing skewness along the last axis."""
    result = {}
    for k, v in features.items():
        if v.ndim == 0:
            result[k] = np.zeros_like(v)
        else:
            result[k] = skew(v, axis=-1, keepdims=True)
    return result

def kurtosis_pooling(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Aggregate features by computing kurtosis along the last axis."""
    result = {}
    for k, v in features.items():
        if v.ndim == 0:
            result[k] = np.zeros_like(v)
        else:
            result[k] = kurtosis(v, axis=-1, keepdims=True)
    return result

def advanced_statistical_summary(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return mean, std, skewness and kurtosis for each feature."""
    summary = {}
    for k, v in features.items():
        if v.ndim == 0:
            summary[f'{k}_mean'] = v
            summary[f'{k}_std'] = np.zeros_like(v)
            summary[f'{k}_skew'] = np.zeros_like(v)
            summary[f'{k}_kurt'] = np.zeros_like(v)
        else:
            summary[f'{k}_mean'] = np.mean(v, axis=-1, keepdims=True)
            summary[f'{k}_std'] = np.std(v, axis=-1, keepdims=True)
            summary[f'{k}_skew'] = skew(v, axis=-1, keepdims=True)
            summary[f'{k}_kurt'] = kurtosis(v, axis=-1, keepdims=True)
    return summary

def sliding_window_aggregation(
    features: Dict[str, np.ndarray], 
    window_size: int, 
    hop_length: int, 
    agg_method: str = 'mean'
) -> Dict[str, np.ndarray]:
    """
    Apply sliding window aggregation to features.
    Args:
        features: Dict of feature arrays (shape: [..., time_frames])
        window_size: Size of the sliding window in frames
        hop_length: Hop length between windows in frames
        agg_method: Aggregation method within each window ('mean', 'std', 'max', 'min')
    Returns:
        Dict of aggregated features with reduced temporal dimension
    """
    result = {}
    for k, v in features.items():
        if v.ndim == 0:
            result[k] = v
            continue

        # Apply sliding window to the last axis (time dimension)
        time_frames = v.shape[-1]
        if time_frames < window_size:
            # If signal is shorter than window, use global aggregation
            if agg_method == 'mean':
                result[k] = np.mean(v, axis=-1, keepdims=True)
            elif agg_method == 'std':
                result[k] = np.std(v, axis=-1, keepdims=True)
            elif agg_method == 'max':
                result[k] = np.max(v, axis=-1, keepdims=True)
            elif agg_method == 'min':
                result[k] = np.min(v, axis=-1, keepdims=True)
            else:
                raise ValueError(f"Unknown aggregation method: {agg_method}")
        else:
            # Calculate number of windows
            n_windows = (time_frames - window_size) // hop_length + 1
            window_results = []

            for i in range(n_windows):
                start_idx = i * hop_length
                end_idx = start_idx + window_size
                window_data = v[..., start_idx:end_idx]

                if agg_method == 'mean':
                    window_agg = np.mean(window_data, axis=-1)
                elif agg_method == 'std':
                    window_agg = np.std(window_data, axis=-1)
                elif agg_method == 'max':
                    window_agg = np.max(window_data, axis=-1)
                elif agg_method == 'min':
                    window_agg = np.min(window_data, axis=-1)
                else:
                    raise ValueError(f"Unknown aggregation method: {agg_method}")

                window_results.append(window_agg)

            # Stack window results along the time axis
            result[k] = np.stack(window_results, axis=-1)

    return result

def aggregate_features(
    features: Dict[str, np.ndarray],
    method: str = 'mean',
    flatten: bool = True,
    normalize: str = None,
    window_size: int = None,
    hop_length: int = None
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Aggregate features using the specified method, then optionally flatten and normalize each feature.
    Args:
        features: Dict of feature arrays
        method: Aggregation method ('mean', 'std', 'summary', 'advanced_summary', 'skew', 'kurtosis', 'sliding_window', 'stack')
        flatten: Whether to flatten each feature to 1D (ignored if method='stack')
        normalize: Normalization method ('zscore', 'minmax', or None)
        window_size: Window size for sliding window aggregation (required if method='sliding_window')
        hop_length: Hop length for sliding window aggregation (required if method='sliding_window')
    Returns:
        Dict of aggregated (and optionally flattened/normalized) features, or a stacked numpy array if method='stack'.
    """
    # Pooling features
    if method == 'mean':
        agg = mean_pooling(features)
    elif method == 'std':
        agg = std_pooling(features)
    elif method == 'summary':
        agg = statistical_summary(features)
    elif method == 'advanced_summary':
        agg = advanced_statistical_summary(features)
    elif method == 'skew':
        agg = skew_pooling(features)
    elif method == 'kurtosis':
        agg = kurtosis_pooling(features)
    elif method == 'sliding_window':
        if window_size is None or hop_length is None:
            raise ValueError("window_size and hop_length must be specified for sliding_window method")
        agg = sliding_window_aggregation(features, window_size, hop_length, 'mean')
    elif method == 'stack':
        return stack_feature_vectors([features], flatten=True)[0]
    else:
        raise ValueError(f'Unknown aggregation method: {method}')

    # Flatten each feature if requested
    if flatten:
        agg = {k: v.flatten() for k, v in agg.items()}

    # Normalize if requested
    if normalize is not None:
        if normalize == 'zscore':
            agg = zscore_normalize(agg)
        elif normalize == 'minmax':
            agg = minmax_normalize(agg)
        else:
            raise ValueError(f'Unknown normalization method: {normalize}')
    return agg
