"""
Feature aggregation utilities (mean, std, pooling strategies).
"""

import warnings
from collections.abc import Callable

import numpy as np

from AFX.utils import minmax_normalize, stack_feature_vectors, zscore_normalize


def _moments(v: np.ndarray, which: tuple[str, ...]) -> dict[str, np.ndarray]:
    """
    NaN-ignoring statistics over the last axis, kept as a length-1 axis.

    Frames where a feature is undefined (NaN, e.g. HNR of a silent frame)
    are skipped; a statistic is NaN only when every frame is NaN. Skewness
    and kurtosis use biased moments with Fisher's definition and are NaN at
    zero variance, as in scipy.stats.skew / kurtosis.
    """
    out = {}
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(v, axis=-1, keepdims=True)
        if 'mean' in which:
            out['mean'] = mean
        if 'std' in which:
            out['std'] = np.nanstd(v, axis=-1, keepdims=True)
        if 'skew' in which or 'kurt' in which:
            dev = v - mean
            m2 = np.nanmean(dev ** 2, axis=-1, keepdims=True)
            zero = m2 <= (np.finfo(np.float64).eps * mean) ** 2
            if 'skew' in which:
                m3 = np.nanmean(dev ** 3, axis=-1, keepdims=True)
                out['skew'] = np.where(zero, np.nan, m3 / m2 ** 1.5)
            if 'kurt' in which:
                m4 = np.nanmean(dev ** 4, axis=-1, keepdims=True)
                out['kurt'] = np.where(zero, np.nan, m4 / m2 ** 2 - 3.0)
    return out


def _pool(features, which, suffix):
    result = {}
    for k, v in features.items():
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 0:
            stats = {'mean': v, 'std': np.zeros_like(v), 'skew': np.zeros_like(v), 'kurt': np.zeros_like(v)}
        else:
            stats = _moments(v, which)
        for name in which:
            result[f'{k}_{name}' if suffix else k] = stats[name]
    return result


def mean_pooling(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mean over the last (time) axis, ignoring NaN frames."""
    return _pool(features, ('mean',), suffix=False)


def std_pooling(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Standard deviation over the last axis, ignoring NaN frames."""
    return _pool(features, ('std',), suffix=False)


def statistical_summary(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """<name>_mean and <name>_std over the last axis, ignoring NaN frames."""
    return _pool(features, ('mean', 'std'), suffix=True)


def skew_pooling(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Skewness over the last axis, ignoring NaN frames."""
    return _pool(features, ('skew',), suffix=False)


def kurtosis_pooling(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Excess kurtosis over the last axis, ignoring NaN frames."""
    return _pool(features, ('kurt',), suffix=False)


def advanced_statistical_summary(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """<name>_mean, _std, _skew and _kurt over the last axis, ignoring NaN frames."""
    return _pool(features, ('mean', 'std', 'skew', 'kurt'), suffix=True)


def sliding_window_aggregation(
    features: dict[str, np.ndarray],
    window_size: int,
    hop_length: int,
    agg_method: str = 'mean'
) -> dict[str, np.ndarray]:
    """
    Apply sliding window aggregation to features.
    Args:
        features: Dict of feature arrays (shape: [..., time_frames])
        window_size: Size of the sliding window in frames
        hop_length: Hop length between windows in frames
        agg_method: Aggregation method within each window ('mean', 'std', 'max', 'min'),
            NaN frames ignored
    Returns:
        Dict of arrays with the last axis replaced by one value per window
    """
    reducers: dict[str, Callable[..., np.ndarray]] = {
        'mean': np.nanmean, 'std': np.nanstd, 'max': np.nanmax, 'min': np.nanmin,
    }
    if agg_method not in reducers:
        raise ValueError(f'Unknown aggregation method: {agg_method}')
    reduce = reducers[agg_method]
    result = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for k, v in features.items():
            v = np.asarray(v, dtype=np.float64)
            if v.ndim == 0:
                result[k] = v
            elif v.shape[-1] < window_size:
                # Shorter than one window: one global value
                result[k] = reduce(v, axis=-1, keepdims=True)
            else:
                windows = np.lib.stride_tricks.sliding_window_view(v, window_size, axis=-1)[..., ::hop_length, :]
                result[k] = reduce(windows, axis=-1)
    return result


def aggregate_features(
    features: dict[str, np.ndarray],
    method: str = 'mean',
    flatten: bool = True,
    normalize: str | None = None,
    window_size: int | None = None,
    hop_length: int | None = None,
    norm_stats: dict[str, dict[str, float]] | None = None,
    sliding_agg: str = 'mean'
) -> dict[str, np.ndarray] | np.ndarray:
    """
    Aggregate features using the specified method, then optionally flatten and normalize each feature.
    Args:
        features: Dict of feature arrays
        method: Aggregation method ('mean', 'std', 'summary', 'advanced_summary',
            'skew', 'kurtosis', 'sliding_window', 'stack')
        flatten: Whether to flatten each feature to 1D (ignored if method='stack')
        normalize: Normalization method ('zscore', 'minmax', or None)
        window_size: Window size for sliding window aggregation (required if method='sliding_window')
        hop_length: Hop length for sliding window aggregation (required if method='sliding_window')
        norm_stats: Optional precomputed dataset-level statistics per feature
            key. For 'zscore': {'mean': {feat: m, ...}, 'std': {feat: s, ...}};
            for 'minmax': {'min': {...}, 'max': {...}}.
        sliding_agg: Inner aggregation for method='sliding_window'
            ('mean', 'std', 'max', 'min').
    Returns:
        Dict of aggregated (and optionally flattened/normalized) features, or a stacked numpy array if method='stack'.

    Note on normalization:
        Without ``norm_stats``, normalization is computed *within this call*
        (i.e., per file, across the values of each feature). That rescales
        every file by its own statistics and is NOT comparable across files.
        For ML pipelines, compute mean/std (or min/max) per feature across
        the whole dataset and pass them via ``norm_stats``.
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
        agg = sliding_window_aggregation(features, window_size, hop_length, sliding_agg)
    elif method == 'stack':
        return stack_feature_vectors([features], flatten=True)[0]
    else:
        raise ValueError(f'Unknown aggregation method: {method}')

    # Flatten each feature if requested
    if flatten:
        agg = {k: v.flatten() for k, v in agg.items()}

    # Normalize if requested
    if normalize is not None:
        norm_stats = norm_stats or {}
        if normalize == 'zscore':
            agg = zscore_normalize(
                agg,
                mean=norm_stats.get('mean'),
                std=norm_stats.get('std'),
            )
        elif normalize == 'minmax':
            agg = minmax_normalize(
                agg,
                min_=norm_stats.get('min'),
                max_=norm_stats.get('max'),
            )
        else:
            raise ValueError(f'Unknown normalization method: {normalize}')
    return agg
