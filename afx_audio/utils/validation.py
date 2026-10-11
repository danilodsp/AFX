"""
Input checks shared by the extractors.
"""
import math
import numbers

import numpy as np


def validate_signal(signal) -> np.ndarray:
    """
    Return ``signal`` as a 1-D floating-point array, or raise.

    Integer arrays are rejected rather than silently scaled: PCM samples
    must be converted by the caller (e.g. ``x / 32768.0`` for int16).
    Float16 is promoted to float32; other float dtypes are kept.
    """
    arr = np.asarray(signal)
    if arr.ndim == 0:
        raise ValueError('signal must be a 1-D array of samples, got a scalar')
    if arr.ndim != 1:
        raise ValueError(
            f'signal must be 1-D (mono), got shape {arr.shape}. '
            'Downmix multichannel audio first, e.g. signal.mean(axis=1).'
        )
    if arr.size == 0:
        raise ValueError('signal is empty')
    if not np.issubdtype(arr.dtype, np.floating):
        if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
            raise TypeError(
                f'signal must be floating point, got {arr.dtype}. Scale integer PCM first, '
                'e.g. x.astype(np.float32) / 32768 for int16.'
            )
        raise TypeError(f'signal must be a real floating-point array, got {arr.dtype}')
    if arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        raise ValueError('signal contains NaN or infinite values')
    return arr


def float64_signal(signal) -> np.ndarray:
    """validate_signal, then float64: extractors compute and return features in double precision."""
    return validate_signal(signal).astype(np.float64, copy=False)


def validate_sr(sr) -> None:
    """Raise unless ``sr`` is a positive finite number."""
    if isinstance(sr, bool) or not isinstance(sr, numbers.Real) or not 0 < float(sr) < math.inf:
        raise ValueError(f'sample rate must be a positive finite number, got {sr!r}')
