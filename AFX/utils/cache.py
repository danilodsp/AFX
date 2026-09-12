"""
Process-wide cache for mel filterbanks.

A filterbank depends only on (n_mels, n_fft, sr, fmin, fmax, norm), so it
is built once per parameter set and shared by every extractor that needs
it. Cached arrays are read-only; copy before mutating.
"""
from typing import Dict, Optional

import numpy as np

_feature_cache: Dict[str, np.ndarray] = {}


def _cache_key_mel_filterbank(n_mels: int, frame_size: int, sr: int, fmin: float, fmax: float, norm: str) -> str:
    """Cache key for a mel filterbank (floats normalized so 50 == 50.0)."""
    return f"mel_filterbank_{n_mels}_{frame_size}_{sr}_{float(fmin)}_{float(fmax)}_{norm}"


def get_cached_mel_filterbank(n_mels: int, frame_size: int, sr: int, fmin: float, fmax: float, norm: str = 'slaney') -> Optional[np.ndarray]:
    """Return the cached mel filterbank for these parameters, or None."""
    key = _cache_key_mel_filterbank(n_mels, frame_size, sr, fmin, fmax, norm)
    return _feature_cache.get(key)


def cache_mel_filterbank(filterbank: np.ndarray, n_mels: int, frame_size: int, sr: int, fmin: float, fmax: float, norm: str = 'slaney') -> None:
    """Store a mel filterbank (made read-only) for reuse."""
    key = _cache_key_mel_filterbank(n_mels, frame_size, sr, fmin, fmax, norm)
    filterbank.flags.writeable = False
    _feature_cache[key] = filterbank


def clear_feature_cache() -> None:
    """Drop every cached filterbank."""
    _feature_cache.clear()


def get_cache_stats() -> Dict[str, int]:
    """Number of cached items, in total and per kind."""
    n_mel = sum(1 for key in _feature_cache if key.startswith('mel_filterbank_'))
    return {'mel_filterbanks': n_mel, 'total_items': len(_feature_cache)}
