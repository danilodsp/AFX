"""
Caching for expensive computations in feature extraction.
"""
import numpy as np
from typing import Dict, Tuple, Optional, Any

# Global cache for expensive computations
_feature_cache: Dict[str, Any] = {}


def _cache_key_mel_filterbank(n_mels: int, frame_size: int, sr: int, fmin: float, fmax: float, norm: str) -> str:
    """Generate cache key for mel filterbank (floats normalized so 50 == 50.0)."""
    return f"mel_filterbank_{n_mels}_{frame_size}_{sr}_{float(fmin)}_{float(fmax)}_{norm}"


def _cache_key_window(window_name: str, frame_size: int) -> str:
    """Generate cache key for window function."""
    return f"window_{window_name}_{frame_size}"


def get_cached_mel_filterbank(n_mels: int, frame_size: int, sr: int, fmin: float, fmax: float, norm: str = 'slaney') -> Optional[np.ndarray]:
    """Get cached mel filterbank if available."""
    key = _cache_key_mel_filterbank(n_mels, frame_size, sr, fmin, fmax, norm)
    return _feature_cache.get(key)


def cache_mel_filterbank(filterbank: np.ndarray, n_mels: int, frame_size: int, sr: int, fmin: float, fmax: float, norm: str = 'slaney') -> None:
    """Cache mel filterbank for reuse. The stored array is made read-only so
    callers can't corrupt the cache by mutating it in place."""
    key = _cache_key_mel_filterbank(n_mels, frame_size, sr, fmin, fmax, norm)
    filterbank.flags.writeable = False
    _feature_cache[key] = filterbank


def get_cached_window(window_name: str, frame_size: int) -> Optional[np.ndarray]:
    """Get cached window function if available."""
    key = _cache_key_window(window_name, frame_size)
    return _feature_cache.get(key)


def cache_window(window: np.ndarray, window_name: str, frame_size: int) -> None:
    """Cache window function for reuse (stored read-only)."""
    key = _cache_key_window(window_name, frame_size)
    window.flags.writeable = False
    _feature_cache[key] = window


def clear_feature_cache() -> None:
    """Clear all cached data."""
    global _feature_cache
    _feature_cache.clear()


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about cached items."""
    stats = {
        'mel_filterbanks': 0,
        'windows': 0,
        'total_items': len(_feature_cache)
    }

    for key in _feature_cache:
        if key.startswith('mel_filterbank_'):
            stats['mel_filterbanks'] += 1
        elif key.startswith('window_'):
            stats['windows'] += 1

    return stats
